"""Direction of causality: beliefs -> NTP & log-NTP (Dani's experiment).

THE CLAIM  (Xavier's refinement, SPAR Slack)
─────────────────────────────────────────────────────────────────────────────
NTP & log-NTP are *computed from* the belief state; the belief is not computed
from them. So causality runs  beliefs -> {NTP, log-NTP}.  Operationally:

  (forward)  changing the BELIEF at layer L changes the corresponding NTP &
             log-NTP in subsequent layers.
  (reverse)  changing the NTP or log-NTP at layer L does NOT change the
             corresponding belief.

Decodability (high R² for act->NTP) is then a side effect of the belief being
present, not evidence of a standalone NTP code.

THE MEASUREMENT
─────────────────────────────────────────────────────────────────────────────
For one 20k sequence we intervene at layer L on the FINAL k tokens (k=10 by
default) and decode FROZEN clean probes at those same k positions, at every
read-out layer. R² is pooled over the k positions (per sequence / per seed).

Treat NTP & log-NTP exactly like beliefs: fit an encoder act->v and an embedding
v->act for each v in {belief, ntp, log_ntp}; patch/steer v at L toward the value
implied by a DIFFERENT belief b' and re-read:

  steer BELIEF   -> decode {belief, ntp, log_ntp}
  steer NTP      -> decode {belief, ntp, log_ntp}
  steer LOG_NTP  -> decode {belief, ntp, log_ntp}

ref="orig": R² vs the original value (expected to DROP when the source drives the
target); ref="new": R² vs the value implied by b' (expected to RISE). Load-bearing:

  source=belief, target=ntp/log_ntp : R²_orig DROPS, R²_new RISES  (forward)
  source=ntp/log_ntp, target=belief : R²_orig STAYS (belief preserved)  (reverse)

READOUT LAYERS: decode at EVERY layer (default). The intervention overwrites
resid_post[L], so read-outs < L are untouched (== clean baseline) and the first
genuine downstream observation is resid_post[L+1]; R² breaks away at L. The
crossover at L is the propagation signature. model_kl rows give the model's own
output KL to orig/new (run_intervene.py metric) to confirm the intervention took.

THE SUBSPACE CAVEAT  (Xavier & Dani, same thread)
─────────────────────────────────────────────────────────────────────────────
ntp = belief @ M is linear in the belief; if M is invertible (e.g. 3 tokens /
3 states) belief and ntp carry the SAME information and the embedding column
spaces belief->act, ntp->act, log_ntp->act may coincide. Then "steer NTP" *is*
"steer belief" and the reverse test is vacuous — the worry cuts both ways. We emit
per-layer principal-angle similarity between the three subspaces
(metric="subspace_sim") and the rank/conditioning of M (M_rank/M_cond). Read the
reverse-direction result ONLY where belief|ntp overlap is well below 1.

CHOOSING b'  (avoid trivial targets)
─────────────────────────────────────────────────────────────────────────────
condition=past_inconsistent borrows a REAL donor belief (same HMM) at the matched
positions, so b'@M is on-manifold and non-degenerate (no near-[1,0,0]); random is
an off-manifold control; round_trip sets v <- enc(h_clean) ≈ original (positive
control). Each row logs mean NTP entropy and mean KL(orig‖new).

Outputs: decode_shift_{model}.csv  (long format).  Reuses src.model_utils /
src.hmm / src.metrics.probes and run_intervene.py's window conventions.
"""
import argparse, gc, os, sys
import numpy as np, pandas as pd, torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))            # for `import run_intervene`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs.hmm_configs import HMMS, REPRESENTATIVES
from src.hmm import (stationary_distribution, sample_hmm_sequence,
                     full_bayesian_beliefs, emission_matrix, next_token_probs)
from src.metrics.probes import predict_probe
from src.model_utils import (load_model, tokens_to_prompt, match_positions,
                             get_tok_ids, extract_activations_chunked)
from run_intervene import _kl, _split, _fit, _llm_ntp

SRC = ["belief", "ntp", "log_ntp"]          # things we steer (and decode)
CONDS = ["past_inconsistent", "random"]     # b' = real donor belief / off-manifold control
EPS = 1e-12


# ── pure numeric helpers ──────────────────────────────────────────────────────
def _entropy(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, None)
    return float(-(p * np.log(p)).sum(-1))


def _r2(pred, target):
    """Pooled R² over a stack of rows (probes.compute_r2 semantics)."""
    pred = np.asarray(pred, dtype=np.float64); target = np.asarray(target, dtype=np.float64)
    if len(pred) < 2:
        return float("nan")
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean(0)) ** 2).sum()
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _vals(src, belief_seq, M):
    """Map a belief sequence (k, n_states) to the value of `src` (k, in_dim)."""
    belief_seq = np.asarray(belief_seq, dtype=np.float64)
    if src == "belief":
        return belief_seq
    ntp = belief_seq @ M
    return ntp if src == "ntp" else np.log(ntp + EPS)


def _orthobasis(W):
    A = W[:-1].detach().cpu().numpy().astype(np.float64).T          # (d, in)
    if A.size == 0:
        return A.reshape(A.shape[0] if A.ndim else 0, 0)
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    tol = (s.max() * 1e-6) if s.size else 0.0
    return U[:, s > tol]


def _subspace_sim(Wa, Wb):
    """(mean cos²(principal angles)=overlap, mean cos) between two embedding images.
    1.0 ⇒ subspaces coincide ⇒ steering one quantity steers the other."""
    Qa, Qb = _orthobasis(Wa), _orthobasis(Wb)
    if Qa.shape[1] == 0 or Qb.shape[1] == 0:
        return float("nan"), float("nan")
    sv = np.linalg.svd(Qa.T @ Qb, compute_uv=False); sv = np.clip(sv, 0.0, 1.0)
    r = min(Qa.shape[1], Qb.shape[1])
    return float((sv ** 2).sum() / r), float(sv.mean())


# ── forward passes with read-out capture (last max_k positions) ───────────────
def _clean_readout(wrapper, window_ids, capture_layers, max_k, device):
    """One clean forward. Returns {layer: tail (max_k, d)} (last max_k resid rows)."""
    tail = {}
    def mk(li):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            tail[li] = h[0, -max_k:, :].detach().float()
        return fn
    handles = [wrapper.get_layer(l).register_forward_hook(mk(l)) for l in capture_layers]
    with torch.no_grad():
        wrapper.forward(window_ids.to(device), use_cache=False)
    for hd in handles:
        hd.remove()
    return tail


def _intervened_readout(wrapper, window_ids, layer, plan, readout_layers, device, dtype, max_k):
    """Apply every plan variant at `layer` in ONE batched forward; capture the last
    max_k resid rows at each read-out layer >= `layer`. The intervention hook precedes
    the capture hook on a shared module, so a read-out at `layer` reflects the
    intervened value. Returns ({lr: (B, max_k, d)}, last_hidden (B, max_k, d))."""
    B = len(plan)
    batch_ids = window_ids.to(device).expand(B, -1).contiguous()
    offs = [p["offsets"].to(device) for p in plan]
    vals = [p["value"].to(device=device, dtype=dtype) for p in plan]
    modes = [p["mode"] for p in plan]
    captured = {}

    def intervene(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        for r in range(B):
            if modes[r] == "patch":
                h[r, offs[r], :] = vals[r]
            else:
                h[r, offs[r], :] = h[r, offs[r], :] + vals[r]
        return None

    def mk_capture(li):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[li] = h[:, -max_k:, :].detach().float()
        return fn

    handles = [wrapper.get_layer(layer).register_forward_hook(intervene)]
    for lr in readout_layers:
        if lr >= layer:
            handles.append(wrapper.get_layer(lr).register_forward_hook(mk_capture(lr)))
    with torch.no_grad():
        out = wrapper.forward(batch_ids, use_cache=False)
    for hd in handles:
        hd.remove()
    return captured, out.last_hidden_state[:, -max_k:, :].detach()


# ── per-sequence driver ───────────────────────────────────────────────────────
def run_sequence(wrapper, tokenizer, cfg, T_matrices, pi, beliefs_all, tokens_all,
                 seed, layers, readout_layers, k_values, tok_ids, lm_head_w, cap, M,
                 args, device, dtype, pbar):
    n_states = cfg["n_states"]
    beliefs = beliefs_all[seed]; tokens = tokens_all[seed]
    ntp_all = next_token_probs(beliefs, T_matrices)
    prompt = tokens_to_prompt(tokens, cfg["token_names"])
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
    pos_indices, _ = match_positions(input_ids, tok_ids)
    n_matched = min(len(tokens), len(pos_indices))
    max_k = max(k_values)
    margin = max(args.fit_margin, max_k)          # never let the decode window leak into the fit
    W = args.context_window if args.context_window > 0 else None
    if n_matched <= args.probe_start + margin + 5:
        return []
    probe_layers = sorted(set(layers) | set(readout_layers))

    # ── fit positions: exclude the final `margin` tokens so the decode window
    #    (the last max_k positions) never leaks into the frozen probe / enc / dec ──
    fit_all = np.arange(args.probe_start, n_matched - margin)
    fit_idx = fit_all[np.sort(_split(len(fit_all), args.train_frac, seed)[0])]
    fit_pos = pos_indices[fit_idx]
    y = {"belief": beliefs[fit_idx], "ntp": ntp_all[fit_idx]}
    y["log_ntp"] = np.log(y["ntp"] + EPS)
    pbar.set_postfix_str(f"{cfg['_name']} s{seed} | fit probes", refresh=True)
    acts, _ = extract_activations_chunked(wrapper, input_ids, probe_layers, fit_pos,
                                          args.chunk_size, device)
    ytr = {s: torch.tensor(y[s], device=device, dtype=torch.float32) for s in SRC}
    ENC, EMB, PROBE = {}, {}, {}
    for l in probe_layers:
        X = acts[l]
        if X.numel() == 0:
            continue
        Xtr = X.float()
        if l in layers:
            for s in SRC:
                ENC[(s, l)] = _fit(Xtr, ytr[s], device)
                EMB[(s, l)] = _fit(ytr[s], Xtr, device)
        if l in readout_layers:
            for s in SRC:
                PROBE[(s, l)] = _fit(Xtr, ytr[s], device)
    del acts; gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    layers = [l for l in layers if (SRC[0], l) in ENC]
    readout_layers = [l for l in readout_layers if (SRC[0], l) in PROBE]
    if not layers or not readout_layers:
        return []

    base_info = dict(hmm=cfg["_name"], param=cfg["_label"], seed=seed)
    sim_rows = []
    for l in layers:
        for a, b in [("belief", "ntp"), ("belief", "log_ntp"), ("ntp", "log_ntp")]:
            ov, mc = _subspace_sim(EMB[(a, l)], EMB[(b, l)])
            for stat, val in (("cos2_overlap", ov), ("mean_cos", mc)):
                sim_rows.append({**base_info, "intervene_layer": l, "readout_layer": -1,
                                 "intervention": "none", "source_kind": f"{a}|{b}",
                                 "condition": "subspace", "k": 0, "draw": 0,
                                 "target_kind": stat, "ref": "na", "metric": "subspace_sim",
                                 "value": val, "mean_target_entropy": float("nan"),
                                 "mean_orig_new_kl": float("nan"), "n_pos": 0})

    # ── the single decode window: final token of the sequence; the last max_k HMM
    #    positions are intervened + decoded (R² pooled over them) ──
    N = n_matched - 1
    mp = int(pos_indices[N]); w0 = (mp - W + 1) if W else 0
    window_ids = input_ids[:, w0:mp + 1]
    dec_idx = np.arange(n_matched - max_k, n_matched)               # max_k HMM-token indices
    dec_off = torch.tensor(pos_indices[dec_idx] - w0, dtype=torch.long)   # window offsets
    orig = {"belief": beliefs[dec_idx], "ntp": ntp_all[dec_idx]}
    orig["log_ntp"] = np.log(orig["ntp"] + EPS)                     # each (max_k, dim)

    tail = _clean_readout(wrapper, window_ids, probe_layers, max_k, device)

    n_seeds = args.n_seeds
    donors = [(seed + j) % n_seeds for j in range(1, args.n_draws + 1)]
    donors = [d for d in donors if d != seed] or [seed]
    rng = np.random.default_rng(seed + 999)

    A, base, kl = {}, {}, {}
    def slot(store, key):
        return store.setdefault(key, {s: {"pred": [], "orig": [], "new": []} for s in SRC}
                                | {"tgt_H": [], "orig_new_kl": []})

    # clean baseline decodability at the max_k positions, per read-out layer
    for lr in readout_layers:
        h = tail[lr]                                                # (max_k, d)
        s = slot(base, (lr,))
        for tk in SRC:
            pred = predict_probe(h, PROBE[(tk, lr)]).cpu().numpy()
            for j in range(max_k):
                s[tk]["pred"].append(pred[j]); s[tk]["orig"].append(orig[tk][j])

    for l in layers:
        ct = tail[l]                                               # (max_k, d) clean tail at L
        plan, meta = [], []

        def add(src, mode, cond, k, belief_seq):
            enc_s, emb_s = ENC[(src, l)], EMB[(src, l)]
            vseq = torch.tensor(_vals(src, belief_seq, M), device=device, dtype=torch.float32)
            emb_t = predict_probe(vseq, emb_s)                     # (k, d)
            if mode == "steer":
                src_val = predict_probe(predict_probe(ct[-k:], enc_s), emb_s)
                value = emb_t - src_val
            else:
                value = emb_t
            plan.append(dict(offsets=dec_off[-k:], mode=mode, value=value))
            meta.append((src, mode, cond, k, np.asarray(belief_seq, dtype=np.float64)))

        for k in k_values:
            rt = predict_probe(ct[-k:], ENC[("belief", l)]).cpu().numpy()   # model's own belief
            for src in SRC:
                add(src, "patch", "round_trip", k, rt)             # positive control
                for d in range(args.n_draws):
                    ds = donors[d % len(donors)]
                    pi_seq = beliefs_all[ds][dec_idx[max_k - k:]]  # donor belief at same positions
                    rnd_seq = rng.dirichlet(np.ones(n_states), size=k)
                    for cond, bseq in (("past_inconsistent", pi_seq), ("random", rnd_seq)):
                        add(src, "patch", cond, k, bseq)
                        add(src, "steer", cond, k, bseq)

        captured, lh = _intervened_readout(wrapper, window_ids, l, plan,
                                           readout_layers, device, dtype, max_k)
        Bn = lh.shape[0]
        p_llm = _llm_ntp(lh.reshape(Bn * max_k, -1), lm_head_w, tok_ids, cap)
        p_llm = p_llm.reshape(Bn, max_k, -1)
        dec_preds = {}                                             # (lr, tk) -> (B, max_k, dim)
        for lr in readout_layers:
            if lr < l:
                continue
            capt = captured[lr]                                    # (B, max_k, d)
            flat = capt.reshape(Bn * max_k, -1)
            for tk in SRC:
                dec_preds[(lr, tk)] = predict_probe(flat, PROBE[(tk, lr)]).reshape(Bn, max_k, -1).cpu().numpy()

        for r, (src, mode, cond, k, bseq) in enumerate(meta):
            new = {"belief": bseq, "ntp": bseq @ M}
            new["log_ntp"] = np.log(new["ntp"] + EPS)
            for lr in readout_layers:
                if lr < l:
                    continue
                s = slot(A, (l, lr, mode, src, cond, k))
                for j in range(k):
                    pj = max_k - k + j                             # index into the max_k axis
                    for tk in SRC:
                        s[tk]["pred"].append(dec_preds[(lr, tk)][r, pj])
                        s[tk]["orig"].append(orig[tk][pj]); s[tk]["new"].append(new[tk][j])
                    s["tgt_H"].append(_entropy(new["ntp"][j]))
                    s["orig_new_kl"].append(_kl(orig["ntp"][pj], new["ntp"][j]))
            kk = kl.setdefault((l, mode, src, cond, k), {"orig": [], "new": []})
            for j in range(k):
                pj = max_k - k + j
                kk["orig"].append(_kl(orig["ntp"][pj], p_llm[r, pj]))
                kk["new"].append(_kl(new["ntp"][j], p_llm[r, pj]))
        pbar.update(1)
    del tail
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── reduce to rows ──
    rows = list(sim_rows)
    for (lr,), s in base.items():
        n = len(s["belief"]["pred"])
        for tk in SRC:
            rows.append({**base_info, "intervene_layer": -1, "readout_layer": lr,
                         "intervention": "none", "source_kind": "none", "condition": "clean",
                         "k": 0, "draw": 0, "target_kind": tk, "ref": "orig", "metric": "R2",
                         "value": _r2(s[tk]["pred"], s[tk]["orig"]),
                         "mean_target_entropy": float("nan"), "mean_orig_new_kl": float("nan"),
                         "n_pos": n})

    for (l, lr, mode, src, cond, k), s in A.items():
        n = len(s["belief"]["pred"])
        if n == 0:
            continue
        mH = float(np.mean(s["tgt_H"])); mKL = float(np.mean(s["orig_new_kl"]))
        for tk in SRC:
            for ref in ("orig", "new"):
                rows.append({**base_info, "intervene_layer": l, "readout_layer": lr,
                             "intervention": mode, "source_kind": src, "condition": cond,
                             "k": k, "draw": 0, "target_kind": tk, "ref": ref, "metric": "R2",
                             "value": _r2(s[tk]["pred"], s[tk][ref]),
                             "mean_target_entropy": mH, "mean_orig_new_kl": mKL, "n_pos": n})

    for (l, mode, src, cond, k), kk in kl.items():
        for ref in ("orig", "new"):
            rows.append({**base_info, "intervene_layer": l, "readout_layer": -1,
                         "intervention": mode, "source_kind": src, "condition": cond,
                         "k": k, "draw": 0, "target_kind": "ntp", "ref": ref,
                         "metric": "model_kl", "value": float(np.mean(kk[ref])),
                         "mean_target_entropy": float("nan"), "mean_orig_new_kl": float("nan"),
                         "n_pos": len(kk[ref])})
    return rows


def main():
    P = argparse.ArgumentParser(description="Belief<->NTP causality + subspace similarity")
    P.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    P.add_argument("--output_dir", default="results")
    P.add_argument("--families", nargs="+", default=["Strata", "Wing"])
    P.add_argument("--all_params", action="store_true")
    P.add_argument("--seq_len", type=int, default=20000)
    P.add_argument("--probe_start", type=int, default=15000)
    P.add_argument("--n_seeds", type=int, default=10)
    P.add_argument("--train_frac", type=float, default=0.2,
                   help="fraction of fit-window positions used to fit the probes")
    P.add_argument("--fit_margin", type=int, default=200,
                   help="trailing tokens reserved for the decode window (excluded from the fit)")
    P.add_argument("--context_window", type=int, default=512)
    P.add_argument("--k_values", type=int, nargs="+", default=[10],
                   help="number of FINAL tokens intervened on at L and decoded for R²")
    P.add_argument("--n_draws", type=int, default=1)
    P.add_argument("--layers", type=int, nargs="+", default=None, help="intervention layers")
    P.add_argument("--readout_layers", type=int, nargs="+", default=None,
                   help="decode layers (default: ALL layers)")
    P.add_argument("--chunk_size", type=int, default=4096)
    P.add_argument("--device", default="cuda")
    P.add_argument("--smoke", action="store_true")
    args = P.parse_args()

    if args.smoke:
        args.seq_len = min(args.seq_len, 800); args.probe_start = 200; args.fit_margin = 50
        args.n_seeds = 2; args.n_draws = 1; args.k_values = [5]; args.context_window = 128
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
    # Decode at EVERY layer so the per-L propagation curve is dense (read-outs < L
    # equal clean; the curve breaks away at L). Capturing extra layers is ~free.
    readout_layers = args.readout_layers or list(range(wrapper.n_layers))
    ms = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "")

    lm_head_w = wrapper.model.lm_head.weight
    cap = None
    if wrapper.family == "gemma":
        tc = getattr(wrapper.model.config, "text_config", wrapper.model.config)
        cap = getattr(tc, "final_logit_softcapping", None)

    combos = sum((len(HMMS[f]["params"]) if args.all_params else 1)
                 for f in args.families if f in HMMS)
    pbar = tqdm(total=combos * args.n_seeds * len(layers), desc="decode_shift", smoothing=0.05)

    all_rows = []
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
            pi = stationary_distribution(T_matrices)
            M = emission_matrix(T_matrices)
            sv = np.linalg.svd(M, compute_uv=False)
            mrank = int((sv > sv.max() * 1e-8).sum())
            mcond = float(sv.max() / sv[sv > 0].min()) if (sv > 0).any() else float("inf")
            beliefs_all, tokens_all = {}, {}
            for s in range(args.n_seeds):
                tk = sample_hmm_sequence(T_matrices, pi, args.seq_len, seed=s).astype(np.int64)
                tokens_all[s] = tk
                beliefs_all[s] = full_bayesian_beliefs(tk, T_stack, pi)
            tqdm.write(f"===== {hmm_name} {label} | M rank {mrank}/{M.shape[0]}"
                       f" cond {mcond:.2f} =====")
            for seed in range(args.n_seeds):
                rows = run_sequence(wrapper, tokenizer, cfg, T_matrices, pi, beliefs_all,
                                    tokens_all, seed, layers, readout_layers, args.k_values,
                                    tok_ids, lm_head_w, cap, M, args, device, dtype, pbar)
                for r in rows:
                    r["M_rank"] = mrank; r["M_cond"] = mcond
                all_rows += rows
                pd.DataFrame(all_rows).to_csv(
                    os.path.join(args.output_dir, f"decode_shift_{ms}.csv"), index=False)
    pbar.close()
    print("Done.")


if __name__ == "__main__":
    main()
