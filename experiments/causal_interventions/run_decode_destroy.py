"""Destruction test: after STEERING the belief, is the original NTP/log-NTP
information *gone* from the residual stream — not just off the old readout?

DISTINCT FROM run_decode_shift.py (the frozen-probe causality/mediation result).
Here we train a FRESH probe on the POST-intervention activations and ask whether
it can recover the ORIGINAL ntp/log/belief. If it can't, the info is destroyed.

Why steering, not patching (Andrew's point): patching overwrites the whole
residual with dec(belief) (a ~3-dim image) so everything non-belief is gone by
construction — confounded. Steering adds emb(new) - emb(enc(h)): it swaps only the
belief component and leaves the rest of the residual intact, so a low retrain-R²
is attributable to the belief swap.

Why FULL 20k context (Andrew's point): the claim is about the model's genuine
activations, which attend over the entire context; a truncated window would change
the activation distribution and could remove originally-present info regardless of
the intervention, making "info gone" an artifact of truncation. So the steered
forward is a chunked, KV-cached pass over the FULL sequence (no window).

PROTOCOL (per family, per seed):
  0. Split the last k tokens (the last 5k of the 20k) into 1k-train / 4k-test. The
     encoder/decoder AND the fresh retrain probe both train on `tr`; `te` is never
     used to fit anything -> pristine test set.
  1. Full-20k clean chunked pass -> clean acts at the last-k positions (all layers).
  2. Fit steering maps enc_L: act->belief, emb_L: belief->act per layer on the
     last-5k TRAIN positions (convention). (No clean readout probe is fit — you
     already have those R²s.)
  3. Per intervention layer L: full-20k chunked STEERED pass — add the steering
     delta at the last k positions at layer L (KV propagates it forward exactly as
     in real inference), capture residuals at read-out layers >= L at those k
     positions.
  4. Retrain probe: fit fresh probes act->{orig,new} x {belief,ntp,log_ntp} on the
     intervened TRAIN acts; report R² on the held-out TEST acts.

Outputs: decode_destroy_{model}.csv  (metric="retrain_R2").
"""
import argparse, copy, gc, os, sys
import numpy as np, pandas as pd, torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs.hmm_configs import HMMS, REPRESENTATIVES
from src.hmm import (stationary_distribution, sample_hmm_sequence,
                     full_bayesian_beliefs, emission_matrix, next_token_probs)
from src.metrics.probes import predict_probe
from src.model_utils import (load_model, tokens_to_prompt, match_positions,
                             get_tok_ids, extract_activations_chunked)
from run_intervene import _kl, _fit

SRC = ["belief", "ntp", "log_ntp"]          # decode targets (and the steered source is belief)
CONDS = ["past_inconsistent"]               # steer to a real donor belief (random control dropped)
EPS = 1e-12


def _r2(pred, target):
    pred = np.asarray(pred, np.float64); target = np.asarray(target, np.float64)
    if len(pred) < 2:
        return float("nan")
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean(0)) ** 2).sum()
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _vals(src, belief_seq, M):
    b = np.asarray(belief_seq, np.float64)
    if src == "belief":
        return b
    ntp = b @ M
    return ntp if src == "ntp" else np.log(ntp + EPS)


def _kerM_dir(M, tol=1e-8):
    """Left-null-space direction delta of M (delta @ M = 0), or None if M is full row-rank.
    Since emission rows sum to 1 (M @ 1 = 1), delta @ M = 0 => delta @ 1 = 0, so moving a belief
    along delta keeps it on the simplex AND leaves the next-token probs (ntp = b @ M) unchanged."""
    U, S, _ = np.linalg.svd(np.asarray(M, np.float64))
    rank = int((S > S.max() * tol).sum()) if S.size and S.max() > 0 else 0
    if rank >= M.shape[0]:
        return None                                      # rank-full (e.g. Mess3): no kernel
    return U[:, rank]                                     # (n_states,) singular vec, ~0 singular value


def _kerM_donor(b, M, frac=0.9):
    """NEW belief with the SAME ntp as `b` but a DIFFERENT belief: move each row maximally along
    ker(M) within the simplex. 'fix prediction, change belief'. Returns None if M is full-rank."""
    d = _kerM_dir(M)
    if d is None:
        return None
    b = np.asarray(b, np.float64); out = b.copy()
    for i in range(len(b)):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = -b[i] / d                                # b_i + t*d >= 0  =>  bounds on t
        t_hi = np.min(np.where(d < 0, r, np.inf))        # largest +t keeping b' >= 0
        t_lo = np.max(np.where(d > 0, r, -np.inf))       # most negative t
        t = t_hi if abs(t_hi) >= abs(t_lo) else t_lo     # take the larger-magnitude belief move
        if np.isfinite(t):
            out[i] = b[i] + frac * t * d
    return out


def _chunked_steered_readout(wrapper, input_ids, layer, steer_pos, steer_val,
                             readout_layers, chunk_size, device, dtype):
    """Chunked KV-cached forward over the FULL sequence with steering at `layer`.

    steer_pos: sorted abs positions to steer (the last k). steer_val: (len, d) the
    delta ADDED at `layer` (row i -> steer_pos[i]). Captures residuals at every
    read-out layer >= `layer` at steer_pos. Returns {lr: (len, d) cpu fp32}."""
    seq_len = input_ids.shape[1]
    rl = [lr for lr in readout_layers if lr >= layer]
    cap = {lr: [] for lr in rl}
    pos2row = {int(p): i for i, p in enumerate(steer_pos)}
    past_kv = None
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = input_ids[:, start:end].to(device)
        in_chunk = [p for p in steer_pos if start <= p < end]
        hooks = []
        if in_chunk:
            local = torch.tensor([p - start for p in in_chunk], device=device, dtype=torch.long)
            rows = [pos2row[int(p)] for p in in_chunk]
            delta = steer_val[rows].to(device=device, dtype=dtype)        # (m, d)

            def steer_hook(module, inp, out, _local=local, _delta=delta):
                h = out[0] if isinstance(out, tuple) else out
                h[0, _local, :] = h[0, _local, :] + _delta
                return None
            hooks.append(wrapper.get_layer(layer).register_forward_hook(steer_hook))

            def mk(li, _local=local):
                def fn(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    cap[li].append(h[0, _local, :].detach().float().cpu())
                return fn
            for lr in rl:                       # steer hook registered first => capture sees steered
                hooks.append(wrapper.get_layer(lr).register_forward_hook(mk(lr)))
        with torch.no_grad():
            out = wrapper.forward(chunk, past_key_values=past_kv, use_cache=True)
        for h in hooks:
            h.remove()
        past_kv = out.past_key_values
        del out
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del past_kv
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {lr: torch.cat(v, 0) for lr, v in cap.items() if v}


# ── prefix-KV caching (EXACT-equivalent speedup) ──────────────────────────────
# The intervention only touches the last k positions, so the prefix [0, suffix_start)
# KV is identical for the clean pass and every steered pass. Compute it once per
# sequence, reprocess only the ~k-token suffix per layer. Bit-identical to the full
# pass (validate with --no_cache diff); ~3.7x fewer token-forwards.
def _clone_cache(kv):
    """Deep-copy a KV cache so a steered suffix pass can't mutate the shared prefix.
    copy.deepcopy is version-agnostic (clones the underlying tensors) — robust to the
    transformers cache API changing (DynamicCache internals differ across versions)."""
    if kv is None:
        return None
    if isinstance(kv, tuple):
        return tuple((k.clone(), v.clone()) for (k, v) in kv)
    return copy.deepcopy(kv)


def _run_chunks(wrapper, ids, past, chunk_size, device, hook_fn=None):
    """Run `ids` left-to-right in chunks from cache `past`; hook_fn(start,end,chunk_len)
    may register hooks per chunk and returns a list of handles to remove. Returns the
    final cache."""
    for start in range(0, ids.shape[1], chunk_size):
        end = min(start + chunk_size, ids.shape[1])
        handles = hook_fn(start, end) if hook_fn else []
        with torch.no_grad():
            out = wrapper.forward(ids[:, start:end].to(device), past_key_values=past, use_cache=True)
        for h in handles:
            h.remove()
        past = out.past_key_values
        del out
    return past


def _prefix_and_clean_suffix(wrapper, input_ids, suffix_start, local_pos, capture_layers,
                             chunk_size, device):
    """Clean pass: cache the prefix [0, suffix_start) KV (all layers), then process the
    suffix capturing clean acts at suffix-local positions `local_pos`. Returns
    (prefix_kv, suffix_ids, {layer: (len(local_pos), d) clean acts})."""
    prefix_kv = _run_chunks(wrapper, input_ids[:, :suffix_start], None, chunk_size, device)
    prefix_kv = _clone_cache(prefix_kv)                   # preserve prefix-only for reuse
    suffix_ids = input_ids[:, suffix_start:]
    cap = {l: [] for l in capture_layers}

    def hooks(start, end):
        loc = [p for p in local_pos if start <= p < end]
        if not loc:
            return []
        idx = torch.tensor([p - start for p in loc], device=device, dtype=torch.long)
        hs = []
        for l in capture_layers:
            def mk(li, _idx=idx):
                def fn(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    cap[li].append(h[0, _idx, :].detach().float().cpu())
                return fn
            hs.append(wrapper.get_layer(l).register_forward_hook(mk(l)))
        return hs

    _run_chunks(wrapper, suffix_ids, _clone_cache(prefix_kv), chunk_size, device, hooks)
    acts = {l: torch.cat(v, 0) for l, v in cap.items() if v}
    return prefix_kv, suffix_ids, acts


def _steered_suffix(wrapper, suffix_ids, prefix_kv, layer, local_pos, steer_val,
                    readout_layers, chunk_size, device, dtype):
    """Steered suffix pass from a fresh copy of the cached prefix. Steers at `layer`
    and captures read-out layers >= `layer` at `local_pos`. Returns {lr: (len, d)}."""
    rl = [lr for lr in readout_layers if lr >= layer]
    cap = {lr: [] for lr in rl}
    pos2row = {int(p): i for i, p in enumerate(local_pos)}

    def hooks(start, end):
        loc = [p for p in local_pos if start <= p < end]
        if not loc:
            return []
        idx = torch.tensor([p - start for p in loc], device=device, dtype=torch.long)
        delta = steer_val[[pos2row[int(p)] for p in loc]].to(device=device, dtype=dtype)
        hs = []

        def steer(module, inp, out, _idx=idx, _delta=delta):
            h = out[0] if isinstance(out, tuple) else out
            h[0, _idx, :] = h[0, _idx, :] + _delta
            return None
        hs.append(wrapper.get_layer(layer).register_forward_hook(steer))
        for lr in rl:                                     # steer hook first => capture sees steered
            def mk(li, _idx=idx):
                def fn(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    cap[li].append(h[0, _idx, :].detach().float().cpu())
                return fn
            hs.append(wrapper.get_layer(lr).register_forward_hook(mk(lr)))
        return hs

    _run_chunks(wrapper, suffix_ids, _clone_cache(prefix_kv), chunk_size, device, hooks)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {lr: torch.cat(v, 0) for lr, v in cap.items() if v}


def run_sequence(wrapper, tokenizer, cfg, T_matrices, beliefs_all, tokens_all, seed,
                 layers, k, tok_ids, M, args, device, dtype, pbar):
    n_states = cfg["n_states"]
    beliefs = beliefs_all[seed]
    ntp_all = next_token_probs(beliefs, T_matrices)
    prompt = tokens_to_prompt(tokens_all[seed], cfg["token_names"])
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
    pos_indices, _ = match_positions(input_ids, tok_ids)
    n_matched = min(len(tokens_all[seed]), len(pos_indices))
    if n_matched <= k + 50:
        return []
    if args.donor == "ntp_matched" and _kerM_dir(M) is None:   # full-rank M: no ntp-preserving move
        pbar.update(len(layers)); return []

    intv_idx = np.arange(n_matched - k, n_matched)             # the last k HMM-token indices
    # BLOCK split with a gap (NOT random): nearby positions have autocorrelated
    # beliefs, so a random 1k/4k split leaks (test positions sit next to train ones
    # and inflate R²). Train = first n_train, then a >=tau gap, then test = the rest,
    # so train/test are decorrelated and R²(orig) is an honest "is the info gone".
    rng = np.random.default_rng(seed)
    tr = np.arange(args.n_train)
    te = np.arange(args.n_train + args.split_gap, k)

    steer_abs = pos_indices[intv_idx]                          # absolute model positions of intervened
    suffix_start = int(steer_abs[0])                          # prefix [0, suffix_start) is never steered
    local_pos = (steer_abs - suffix_start).astype(int)       # positions within the suffix

    # ── clean pass: cache the prefix KV (reused by every steered pass) + clean acts at
    #    the last-k positions. --no_cache uses the plain full pass (for diff validation). ──
    pbar.set_postfix_str(f"{cfg['_name']} s{seed} | clean pass", refresh=True)
    prefix_kv = suffix_ids = None
    if args.no_cache:
        acts, _ = extract_activations_chunked(wrapper, input_ids, layers, steer_abs,
                                              args.chunk_size, device)
    else:
        prefix_kv, suffix_ids, acts = _prefix_and_clean_suffix(
            wrapper, input_ids, suffix_start, local_pos, layers, args.chunk_size, device)
    # encoder act->SOURCE and decoder SOURCE->act, fit on the last-5k TRAIN positions.
    # SOURCE is the subspace we STEER (--source belief|ntp|log_ntp); decode targets are
    # always all three. --source ntp/log_ntp gives the reciprocal test (steer ntp -> decode belief).
    src_tr = torch.tensor(_vals(args.source, beliefs[intv_idx[tr]], M),
                          device=device, dtype=torch.float32)
    enc, emb, clean_intv = {}, {}, {}
    for l in layers:
        X = acts.get(l)
        if X is None or X.numel() == 0:
            continue
        X = X.to(device).float()                                # cached path captures to CPU; move to GPU
        enc[l] = _fit(X[tr], src_tr, device)                    # act -> source (last-5k train)
        emb[l] = _fit(src_tr, X[tr], device)                    # source -> act
        clean_intv[l] = X                                       # (k, d) clean acts at all last-k pos
    del acts; gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    layers = [l for l in layers if l in enc]

    # ── targets at the intervened positions ──
    orig = {"belief": beliefs[intv_idx], "ntp": ntp_all[intv_idx]}
    orig["log_ntp"] = np.log(orig["ntp"] + EPS)
    if args.donor == "ntp_matched":                        # ker(M): same ntp, different belief
        new_bel = {"ntp_matched": _kerM_donor(beliefs[intv_idx], M)}
        conds = ["ntp_matched"]
    else:
        donor = (seed + 1) % args.n_seeds
        donor = donor if donor != seed else seed
        new_bel = {"past_inconsistent": beliefs_all[donor][intv_idx]}
        conds = CONDS
    rows = []
    base_info = dict(hmm=cfg["_name"], param=cfg["_label"], seed=seed)

    if args.full_story:
        # DEFINITIVE run for one donor. BEFORE (clean acts) + AFTER (steered), scoring BOTH a
        # retrain probe (fresh on the acts measured) and a FROZEN probe (fit on clean->orig, applied
        # unchanged). Targets: full belief, HIDDEN axis (ker delta-coordinate = the part of belief
        # NTP can't see), ntp, log_ntp; x {orig,new}; all (intervene L, readout r). ntp_matched gives
        # the NTP-fixed hidden-axis swap (full belief stays flat by construction); past_inconsistent
        # gives belief+NTP following. Extra "probe" column distinguishes clean/retrain/frozen.
        cond = conds[0]; nb = new_bel[cond]
        dhat = _kerM_dir(M)                                    # unit left-null vec, or None (rank-full)
        def _mk(b):
            b = np.asarray(b, np.float64); ntp = b @ M
            d = {"belief": b, "ntp": ntp, "log_ntp": np.log(ntp + EPS)}
            if dhat is not None:
                d["hidden"] = (b @ dhat).reshape(-1, 1)        # scalar coord NTP is blind to
            return d
        targ = {"orig": _mk(beliefs[intv_idx]), "new": _mk(nb)}
        TGT = list(targ["orig"].keys())
        def _aug(X): return torch.cat([X, torch.ones(X.shape[0], 1, device=device)], 1)
        def _pinv(Xa):
            try: return torch.linalg.pinv(Xa)
            except (RuntimeError, NotImplementedError): return torch.linalg.pinv(Xa.cpu()).to(device)
        # frozen rulers: clean train -> ORIG target, one per (readout layer, target)
        Pf = {(r, tk): _fit(clean_intv[r][tr],
                            torch.tensor(targ["orig"][tk][tr], device=device, dtype=torch.float32), device)
              for r in layers for tk in TGT}
        # BEFORE: fresh probe on CLEAN acts -> {orig,new} (== frozen-on-clean for ref=orig)
        for r in layers:
            P = _pinv(_aug(clean_intv[r][tr])); Xte = _aug(clean_intv[r][te])
            for tk in TGT:
                for ref in ("orig", "new"):
                    Ytr = torch.tensor(targ[ref][tk][tr], device=device, dtype=torch.float32)
                    pred = (Xte @ (P @ Ytr)).cpu().numpy()
                    rows.append({**base_info, "intervene_layer": -1, "readout_layer": int(r),
                                 "condition": cond, "intervention": "none", "phase": "before",
                                 "probe": "clean", "source_kind": args.source, "target_kind": tk,
                                 "ref": ref, "metric": "clean_R2", "value": _r2(pred, targ[ref][tk][te]),
                                 "n_train": len(tr), "n_test": len(te)})
        # AFTER: per intervene layer L, one steered pass; score retrain AND frozen
        for l in layers:
            new_src = torch.tensor(_vals(args.source, nb, M), device=device, dtype=torch.float32)
            emb_new = predict_probe(new_src, emb[l])
            src_val = predict_probe(predict_probe(clean_intv[l], enc[l]), emb[l])
            delta = (emb_new - src_val).detach()
            if args.no_cache:
                cap = _chunked_steered_readout(wrapper, input_ids, l, steer_abs, delta, layers, args.chunk_size, device, dtype)
            else:
                cap = _steered_suffix(wrapper, suffix_ids, prefix_kv, l, local_pos, delta, layers, args.chunk_size, device, dtype)
            for r, A in cap.items():
                Atr = A[tr].to(device).float(); Ate = A[te].to(device).float()
                P = _pinv(_aug(Atr)); Ate_a = _aug(Ate)
                for tk in TGT:
                    for ref in ("orig", "new"):
                        Ytr = torch.tensor(targ[ref][tk][tr], device=device, dtype=torch.float32)
                        pr = (Ate_a @ (P @ Ytr)).cpu().numpy()                       # retrain
                        rows.append({**base_info, "intervene_layer": l, "readout_layer": int(r),
                                     "condition": cond, "intervention": "steer", "phase": "after",
                                     "probe": "retrain", "source_kind": args.source, "target_kind": tk,
                                     "ref": ref, "metric": "retrain_R2", "value": _r2(pr, targ[ref][tk][te]),
                                     "n_train": len(tr), "n_test": len(te)})
                        pf = predict_probe(Ate, Pf[(r, tk)]).cpu().numpy()            # frozen
                        rows.append({**base_info, "intervene_layer": l, "readout_layer": int(r),
                                     "condition": cond, "intervention": "steer", "phase": "after",
                                     "probe": "frozen", "source_kind": args.source, "target_kind": tk,
                                     "ref": ref, "metric": "frozen_R2", "value": _r2(pf, targ[ref][tk][te]),
                                     "n_train": len(tr), "n_test": len(te)})
            del cap
            if device.type == "cuda": torch.cuda.empty_cache()
            pbar.update(1)
        del prefix_kv, suffix_ids
        if device.type == "cuda": torch.cuda.empty_cache()
        return rows

    if args.clean_baseline:
        # CLEAN 'before' baseline: fresh probe on CLEAN train acts -> {orig,new}x{belief,ntp,log_ntp},
        # scored on held-out CLEAN test acts. Mirrors the retrain branch exactly but reads clean_intv
        # (not steered acts), so before/new uses the SAME _kerM_donor b', SAME positions, SAME tr/te
        # split as the kerm 'after' run -> directly comparable. L-independent: one pass over readout
        # layers, no steered forwards. metric="clean_R2", intervention="none", phase="before".
        cond = conds[0]
        nb = new_bel[cond]
        new = {"belief": nb, "ntp": nb @ M}; new["log_ntp"] = np.log(new["ntp"] + EPS)
        targ = {"orig": orig, "new": new}
        for r in layers:
            Xtr = clean_intv[r][tr]; Xte = clean_intv[r][te]
            Xtr_aug = torch.cat([Xtr, torch.ones(Xtr.shape[0], 1, device=device)], 1)
            Xte_aug = torch.cat([Xte, torch.ones(Xte.shape[0], 1, device=device)], 1)
            try:
                P = torch.linalg.pinv(Xtr_aug)                     # one pinv, reused across all 6 targets
            except (RuntimeError, NotImplementedError):
                P = torch.linalg.pinv(Xtr_aug.cpu()).to(device)
            for tk in SRC:
                for ref in ("orig", "new"):
                    tgt = targ[ref][tk]
                    Ytr = torch.tensor(tgt[tr], device=device, dtype=torch.float32)
                    pred = (Xte_aug @ (P @ Ytr)).cpu().numpy()
                    rows.append({**base_info, "intervene_layer": -1, "readout_layer": int(r),
                                 "condition": cond, "intervention": "none", "phase": "before",
                                 "source_kind": args.source, "target_kind": tk, "ref": ref,
                                 "metric": "clean_R2", "value": _r2(pred, tgt[te]),
                                 "n_train": len(tr), "n_test": len(te)})
        pbar.update(len(layers))
        del prefix_kv, suffix_ids
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return rows

    if args.frozen:
        # FROZEN before/after swap: fit read-probes act->{belief,ntp,log_ntp} on CLEAN train, then
        # read those SAME fixed probes on CLEAN acts (before) and STEERED acts (after). One fixed
        # probe => orig belief swaps high->low, new belief low->high, while ntp readout stays put.
        cond = conds[0]
        nb = new_bel[cond]
        new = {"belief": nb, "ntp": nb @ M}; new["log_ntp"] = np.log(new["ntp"] + EPS)
        targ = {"orig": orig, "new": new}
        Pf = {(l, tk): _fit(clean_intv[l][tr],
                            torch.tensor(orig[tk][tr], device=device, dtype=torch.float32), device)
              for l in layers for tk in SRC}
        # BEFORE: fixed probe on CLEAN acts (intervention=none; L-independent)
        for r in layers:
            for tk in SRC:
                pc = predict_probe(clean_intv[r][te], Pf[(r, tk)]).cpu().numpy()
                for ref in ("orig", "new"):
                    rows.append({**base_info, "intervene_layer": -1, "readout_layer": int(r),
                                 "condition": cond, "intervention": "none", "phase": "before",
                                 "source_kind": args.source, "target_kind": tk, "ref": ref,
                                 "metric": "frozen_R2", "value": _r2(pc, targ[ref][tk][te]),
                                 "n_train": len(tr), "n_test": len(te)})
        # AFTER: same fixed probe on STEERED acts, per intervention layer L
        for l in layers:
            new_src = torch.tensor(_vals(args.source, nb, M), device=device, dtype=torch.float32)
            emb_new = predict_probe(new_src, emb[l])
            src_val = predict_probe(predict_probe(clean_intv[l], enc[l]), emb[l])
            delta = (emb_new - src_val).detach()
            if args.no_cache:
                cap = _chunked_steered_readout(wrapper, input_ids, l, steer_abs, delta,
                                               layers, args.chunk_size, device, dtype)
            else:
                cap = _steered_suffix(wrapper, suffix_ids, prefix_kv, l, local_pos, delta,
                                      layers, args.chunk_size, device, dtype)
            for r, A in cap.items():
                Ate = A[te].to(device).float()
                for tk in SRC:
                    pa = predict_probe(Ate, Pf[(r, tk)]).cpu().numpy()
                    for ref in ("orig", "new"):
                        rows.append({**base_info, "intervene_layer": l, "readout_layer": int(r),
                                     "condition": cond, "intervention": "steer", "phase": "after",
                                     "source_kind": args.source, "target_kind": tk, "ref": ref,
                                     "metric": "frozen_R2", "value": _r2(pa, targ[ref][tk][te]),
                                     "n_train": len(tr), "n_test": len(te)})
            del cap
            if device.type == "cuda":
                torch.cuda.empty_cache()
            pbar.update(1)
        del prefix_kv, suffix_ids
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return rows

    for l in layers:
        for cond in conds:
            nb = new_bel[cond]                                  # (k, n_states) donor belief
            # steering delta at layer l: emb(new_src) - emb(enc(clean))   (steer, not patch)
            # new_src is the donor's SOURCE value (belief/ntp/log_ntp) we steer toward.
            new_src = torch.tensor(_vals(args.source, nb, M), device=device, dtype=torch.float32)
            emb_new = predict_probe(new_src, emb[l])
            src_val = predict_probe(predict_probe(clean_intv[l], enc[l]), emb[l])
            delta = (emb_new - src_val).detach()                # (k, d)

            if args.no_cache:
                cap = _chunked_steered_readout(wrapper, input_ids, l, steer_abs, delta,
                                               layers, args.chunk_size, device, dtype)
            else:
                cap = _steered_suffix(wrapper, suffix_ids, prefix_kv, l, local_pos, delta,
                                      layers, args.chunk_size, device, dtype)
            new = {"belief": nb, "ntp": nb @ M}
            new["log_ntp"] = np.log(new["ntp"] + EPS)
            for lr, A in cap.items():                           # A: (k, d) cpu, read-out >= l
                Atr = A[tr].to(device).float(); Ate = A[te].to(device).float()
                # pinv(Atr) is the cost; it's identical across all 6 targets -> compute ONCE
                # and reuse (W = pinv(X)@Y exactly == fit_probe per target). bias = ones col.
                Atr_aug = torch.cat([Atr, torch.ones(Atr.shape[0], 1, device=device)], 1)
                Ate_aug = torch.cat([Ate, torch.ones(Ate.shape[0], 1, device=device)], 1)
                try:
                    P = torch.linalg.pinv(Atr_aug)
                except (RuntimeError, NotImplementedError):
                    P = torch.linalg.pinv(Atr_aug.cpu()).to(device)
                for tk in SRC:
                    for ref, tgt in (("orig", orig[tk]), ("new", new[tk])):
                        Ytr = torch.tensor(tgt[tr], device=device, dtype=torch.float32)
                        pred = (Ate_aug @ (P @ Ytr)).cpu().numpy()   # reuses the single pinv
                        rows.append({**base_info, "intervene_layer": l, "readout_layer": int(lr),
                                     "condition": cond, "intervention": "steer",
                                     "source_kind": args.source, "target_kind": tk, "ref": ref,
                                     "metric": "retrain_R2", "value": _r2(pred, tgt[te]),
                                     "n_train": len(tr), "n_test": len(te)})
            del cap
            if device.type == "cuda":
                torch.cuda.empty_cache()
        pbar.update(1)
    del prefix_kv, suffix_ids
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def main():
    P = argparse.ArgumentParser(description="Retrain/destruction test: is orig ntp info gone after steering")
    P.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    P.add_argument("--output_dir", default="results")
    P.add_argument("--families", nargs="+", default=["Strata", "Wing"])
    P.add_argument("--all_params", action="store_true")
    P.add_argument("--seq_len", type=int, default=20000)
    P.add_argument("--k", type=int, default=5000, help="final tokens intervened + decoded (the last 5k)")
    P.add_argument("--n_train", type=int, default=1000,
                   help="last-5k positions used to fit enc/dec AND train the fresh probe (rest = test)")
    P.add_argument("--split_gap", type=int, default=100,
                   help="decorrelating gap (>=tau) between the train block and the test block")
    P.add_argument("--source", choices=["belief", "ntp", "log_ntp"], default="belief",
                   help="subspace to STEER; decode targets are always belief/ntp/log_ntp. "
                        "ntp/log_ntp = the reciprocal test (steer ntp -> decode belief)")
    P.add_argument("--donor", choices=["past_inconsistent", "ntp_matched"], default="past_inconsistent",
                   help="how to pick the NEW belief. ntp_matched moves along ker(M): SAME ntp, "
                        "different belief ('fix prediction, change belief'). Needs rank-deficient M "
                        "(Arch/Wing/Strata/Spiral); rank-full families (Mess3) are skipped.")
    P.add_argument("--frozen", action="store_true",
                   help="FROZEN-probe before/after swap (not retrain): fit read-probes act->{belief,ntp,"
                        "log_ntp} on CLEAN train, then read those SAME probes on clean acts (before) and "
                        "steered acts (after). Yields orig belief high->low, new belief low->high, ntp flat.")
    P.add_argument("--full_story", action="store_true",
                   help="DEFINITIVE run for one donor: BEFORE (clean) + AFTER (retrain AND frozen), "
                        "targets = full belief, HIDDEN axis (ker delta-coordinate NTP can't see), ntp, "
                        "log_ntp, x {orig,new}, all layers. ntp_matched -> hidden-axis swap (NTP fixed); "
                        "past_inconsistent -> belief+NTP follow. Saves everything so no re-runs.")
    P.add_argument("--clean_baseline", action="store_true",
                   help="CLEAN 'before' baseline (no steering): fit a FRESH probe on CLEAN train acts -> "
                        "{orig,new}x{belief,ntp,log_ntp}, score on held-out CLEAN test. Same _kerM_donor b', "
                        "same positions/split as the kerm 'after' run, so before/new is directly comparable. "
                        "No steered passes -> ~L times cheaper.")
    P.add_argument("--n_seeds", type=int, default=10)
    P.add_argument("--layers", type=int, nargs="+", default=None, help="intervention layers (default: all)")
    P.add_argument("--chunk_size", type=int, default=2048)
    P.add_argument("--no_cache", action="store_true",
                   help="disable prefix-KV caching (plain full passes; for diff validation)")
    P.add_argument("--device", default="cuda")
    P.add_argument("--smoke", action="store_true")
    args = P.parse_args()

    if args.smoke:
        args.seq_len = 4000; args.k = 1000; args.n_train = 200; args.n_seeds = 2
        if not args.all_params:
            args.families = args.families[:1]

    os.makedirs(args.output_dir, exist_ok=True)
    wrapper, tokenizer = load_model(args.model, args.device)
    device = next(wrapper.model.parameters()).device
    if device.type == "cpu":
        wrapper.model.float()
    dtype = next(wrapper.model.parameters()).dtype
    layers = args.layers or list(range(wrapper.n_layers))
    if args.smoke:
        layers = [wrapper.n_layers // 3, 2 * wrapper.n_layers // 3]
    ms = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "")

    combos = sum((len(HMMS[f]["params"]) if args.all_params else 1)
                 for f in args.families if f in HMMS)
    pbar = tqdm(total=combos * args.n_seeds * len(layers), desc="decode_destroy", smoothing=0.05)
    tag = "" if args.source == "belief" else f"{args.source}_"   # belief-source keeps old filename
    if args.donor == "ntp_matched":                              # ker(M) "fix ntp, change belief" run
        tag = "kerm_"
    if args.frozen:                                              # frozen-probe before/after swap
        tag = "swap_"
    if args.clean_baseline:                                      # clean 'before' baseline (no steering)
        tag = "cleanbase_"
    if args.full_story:                                           # definitive per-donor run (donor in name)
        tag = f"story_{args.donor}_"
    out_csv = os.path.join(args.output_dir, f"decode_destroy_{tag}{ms}.csv")
    all_rows, done = [], set()
    if os.path.exists(out_csv):                              # RESUME: keep finished (hmm,param,seed)
        prev = pd.read_csv(out_csv)
        all_rows = prev.to_dict("records")
        done = set(zip(prev["hmm"], prev["param"], prev["seed"]))
        tqdm.write(f"resume: {len(done)} (hmm,param,seed) already done -> skipping")
    for hmm_name in args.families:
        cfg = HMMS.get(hmm_name)
        if not cfg:
            continue
        tok_ids = get_tok_ids(tokenizer, cfg["token_names"])
        params = cfg["params"] if args.all_params else [REPRESENTATIVES[hmm_name]]
        for param in params:
            label = cfg["label_fn"](param)
            cfg = {**cfg, "_name": hmm_name, "_label": label}
            T_matrices = cfg["fn"](*param); T_stack = np.stack(T_matrices)
            pi = stationary_distribution(T_matrices); M = emission_matrix(T_matrices)
            sv = np.linalg.svd(M, compute_uv=False)
            mrank = int((sv > sv.max() * 1e-8).sum())
            mcond = float(sv.max() / sv[sv > 0].min()) if (sv > 0).any() else float("inf")
            beliefs_all, tokens_all = {}, {}
            for s in range(args.n_seeds):
                tk = sample_hmm_sequence(T_matrices, pi, args.seq_len, seed=s).astype(np.int64)
                tokens_all[s] = tk; beliefs_all[s] = full_bayesian_beliefs(tk, T_stack, pi)
            tqdm.write(f"===== {hmm_name} {label} | M rank {mrank}/{M.shape[0]} cond {mcond:.2f} =====")
            for seed in range(args.n_seeds):
                if (hmm_name, label, seed) in done:          # already computed -> skip (resume)
                    pbar.update(len(layers)); continue
                rows = run_sequence(wrapper, tokenizer, cfg, T_matrices, beliefs_all, tokens_all,
                                    seed, layers, args.k, tok_ids, M, args, device, dtype, pbar)
                for r in rows:
                    r["M_rank"] = mrank; r["M_cond"] = mcond
                all_rows += rows
                pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    pbar.close()
    print("Done.")


if __name__ == "__main__":
    main()
