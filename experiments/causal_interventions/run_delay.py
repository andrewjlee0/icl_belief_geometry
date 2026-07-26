"""Delay-process interventions with MULTI-STEP output readout.

The Delay process makes the kernel coordinate behaviorally loaded at a delay:
during the quiet phase the regime (A1 vs A2) has zero effect on the current NTP
(delta = e_A1 - e_A2 is exactly ker M) but fully determines the NEXT REVEAL token.
Theory ceiling (eps=0.10, x=0.002): P(correct reveal | true belief) = 0.92,
flipped belief = 0.26, log-NTP-implied (regime-hedged) state = 0.50 exactly.

Arms (all patch the last k positions at layer L, then read the model's output at
positions i..i+m as it consumes the REAL continuation):
  none         clean forward (learnability baseline: does the model track the regime?)
  self_belief  patch dec_belief(true beliefs)     -> should preserve reveal prediction
  ker_delta    patch dec_belief(kerdonor beliefs) -> should ANTI-predict the reveal
  self_logntp  patch dec_logntp(true log-NTPs)    -> regime info absent: chance

Output: delay_{model}.csv, one row per (param, seed, pos, layer, cond, step) with the
renormalized 3-token output distribution and the actual next token.
"""
import argparse, gc, os, sys
import numpy as np, pandas as pd, torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs.hmm_configs import HMMS, REPRESENTATIVES
from src.hmm import (stationary_distribution, sample_hmm_sequence,
                     full_bayesian_beliefs, emission_matrix)
from src.metrics.probes import fit_probe, predict_probe
from src.model_utils import (load_model, tokens_to_prompt, match_positions,
                             get_tok_ids, extract_activations_chunked)

EPS_LOG = 1e-12


def _kerM_dir(M, tol=1e-8):
    U, S, _ = np.linalg.svd(np.asarray(M, np.float64))
    rank = int((S > S.max() * tol).sum()) if S.size and S.max() > 0 else 0
    if rank >= M.shape[0]:
        return None
    return U[:, rank]


def _kerM_donor_rows(B, M, frac=0.9):
    d = _kerM_dir(M)
    out = np.asarray(B, np.float64).copy()
    for i in range(len(out)):
        b = out[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = -b / d
        t_hi = np.min(np.where(d < 0, r, np.inf))
        t_lo = np.max(np.where(d > 0, r, -np.inf))
        t = t_hi if abs(t_hi) >= abs(t_lo) else t_lo
        if np.isfinite(t):
            out[i] = b + frac * t * d
    return out


def _fit(X, Y, device):
    return fit_probe(torch.as_tensor(np.asarray(X), device=device, dtype=torch.float32),
                     torch.as_tensor(np.asarray(Y), device=device, dtype=torch.float32),
                     use_bias=True)


def _split(n, train_frac, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(n * train_frac)
    return idx[:cut], idx[cut:]


def _probs_slice(hidden_slice, lm_head_w, tok_ids, cap):
    """hidden_slice (B, m+1, d) -> (B, m+1, n_tokens) renormalized over HMM tokens."""
    h = hidden_slice.float().to(lm_head_w.device)
    with torch.no_grad():
        logits = h @ lm_head_w[tok_ids].float().T
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return torch.softmax(logits, dim=-1).cpu().numpy()


def _forward_slice(wrapper, batch_ids, layer, plan, s0, s1, device, dtype):
    """Batched forward with per-row patch at `layer`; returns hidden[:, s0:s1, :]."""
    if plan is not None:
        offs = [p["offsets"].to(device) for p in plan]
        vals = [p["value"].to(device=device, dtype=dtype) for p in plan]

        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            for r in range(len(plan)):
                h[r, offs[r], :] = vals[r]
            return None
        handle = wrapper.get_layer(layer).register_forward_hook(hook)
    with torch.no_grad():
        out = wrapper.forward(batch_ids, use_cache=False)
    if plan is not None:
        handle.remove()
    return out.last_hidden_state[:, s0:s1, :].detach()


def main():
    P = argparse.ArgumentParser(description="Delay-process multi-step interventions")
    P.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    P.add_argument("--output_dir", default="results")
    P.add_argument("--families", nargs="+", default=["Delay"])
    P.add_argument("--all_params", action="store_true")
    P.add_argument("--seq_len", type=int, default=20000)
    P.add_argument("--probe_start", type=int, default=15000)
    P.add_argument("--n_seeds", type=int, default=10)
    P.add_argument("--train_frac", type=float, default=0.2)
    P.add_argument("--n_eval", type=int, default=6)
    P.add_argument("--min_gap", type=int, default=20,
                   help="eval positions must sit >= this many tokens after the last reveal, "
                        "where the model's context-derived regime knowledge has decayed to "
                        "~chance (measured), so the patch is its only regime source")
    P.add_argument("--context_window", type=int, default=512)
    P.add_argument("--k", type=int, default=10)
    P.add_argument("--measure_horizon", type=int, default=40)
    P.add_argument("--chunk_size", type=int, default=4096)
    P.add_argument("--device", default="cuda")
    P.add_argument("--smoke", action="store_true")
    P.add_argument("--param", default=None, help="override representative param, e.g. '0.2,0.002'")
    args = P.parse_args()

    if args.smoke:
        args.seq_len = 2000; args.probe_start = 800; args.n_seeds = 1
        args.n_eval = 2; args.measure_horizon = 20

    os.makedirs(args.output_dir, exist_ok=True)
    wrapper, tokenizer = load_model(args.model, args.device)
    device = next(wrapper.model.parameters()).device
    if device.type == "cpu":
        wrapper.model.float()
    dtype = next(wrapper.model.parameters()).dtype
    layers = list(range(wrapper.n_layers))
    if args.smoke:
        layers = [wrapper.n_layers // 3, 2 * wrapper.n_layers // 3]
    ms = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "")
    lm_head_w = wrapper.model.lm_head.weight
    cap = None
    if wrapper.family == "gemma":
        tc = getattr(wrapper.model.config, "text_config", wrapper.model.config)
        cap = getattr(tc, "final_logit_softcapping", None)

    W, K, m = args.context_window, args.k, args.measure_horizon
    all_rows = []
    for hmm_name in args.families:
        cfg = HMMS[hmm_name]
        tok_ids = get_tok_ids(tokenizer, cfg["token_names"])
        params = ([tuple(float(v) for v in args.param.split(","))] if args.param
                  else (cfg["params"] if args.all_params else [REPRESENTATIVES[hmm_name]]))
        for param in params:
            label = cfg["label_fn"](param)
            T = cfg["fn"](*param); T_stack = np.stack(T)
            pi = stationary_distribution(T); M = emission_matrix(T)
            if _kerM_dir(M) is None:
                print(f"{hmm_name} {label}: full-rank M, skipping"); continue
            pbar = tqdm(total=args.n_seeds * args.n_eval * len(layers),
                        desc=f"{hmm_name} {label}", smoothing=0.05)
            for seed in range(args.n_seeds):
                tokens = sample_hmm_sequence(T, pi, args.seq_len, seed=seed).astype(np.int64)
                beliefs = full_bayesian_beliefs(tokens, T_stack, pi)
                prompt = tokens_to_prompt(tokens, cfg["token_names"])
                input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
                pos_indices, _ = match_positions(input_ids, tok_ids)
                n_matched = min(len(tokens), len(pos_indices))
                if n_matched <= args.probe_start + m + 50:
                    pbar.update(args.n_eval * len(layers)); continue

                # enc/dec fit (belief and log-NTP coordinates) on the late window
                late_idx = np.arange(args.probe_start, n_matched - m - 2)
                late_pos = pos_indices[late_idx]
                acts, _ = extract_activations_chunked(wrapper, input_ids, layers,
                                                      late_pos, args.chunk_size, device)
                idx_tr, idx_te = _split(len(late_idx), args.train_frac, seed)
                y_bel = beliefs[late_idx]
                y_lnt = np.log(y_bel @ M + EPS_LOG)
                dec_b, dec_l = {}, {}
                for l in layers:
                    X = acts[l]
                    if X.numel() == 0: continue
                    Xtr = X[idx_tr].float()
                    dec_b[l] = _fit(y_bel[idx_tr], Xtr.cpu().numpy(), device)
                    dec_l[l] = _fit(y_lnt[idx_tr], Xtr.cpu().numpy(), device)
                del acts; gc.collect(); torch.cuda.empty_cache()
                lyrs = [l for l in layers if l in dec_b]

                # eval positions: quiet token, held-out, room for horizon
                elig = late_idx[idx_te]
                elig = elig[((tokens[elig] == 0) if args.min_gap >= 0 else np.ones(len(elig), bool)) & (elig >= K) &
                            (elig + m + 2 < n_matched) &
                            (pos_indices[elig] >= (W - 1 if W else 0))]
                # long-quiet selection: last reveal must be >= min_gap tokens back
                rev = np.where(np.isin(tokens[:n_matched], (1, 2)))[0]
                if len(rev) and args.min_gap > 0:
                    j = np.searchsorted(rev, elig) - 1
                    gap = np.where(j >= 0, elig - rev[np.clip(j, 0, None)], elig + 1)
                    elig = elig[gap >= args.min_gap]
                if args.min_gap < 0:
                    # LEVERAGE selection (generic rank-deficient families): pick eval
                    # positions where the ker-flip has the largest future NTP consequence.
                    def _roll_ntp_gap(t):
                        b = beliefs[t]; bp = _kerM_donor_rows(b[None], M)[0]
                        g = 0.0
                        for z in tokens[t + 1:t + 17]:
                            b = b @ T_stack[z]; b = b / b.sum() if b.sum() > 0 else b
                            bp = bp @ T_stack[z]; bp = bp / bp.sum() if bp.sum() > 0 else bp
                            p = np.clip(b @ M, 1e-12, None); q = np.clip(bp @ M, 1e-12, None)
                            g = max(g, float((q * np.log(q / p)).sum()))
                        return g
                    cand = elig[np.random.default_rng(seed + 13).permutation(len(elig))][:600]
                    lev = np.array([_roll_ntp_gap(int(t)) for t in cand])
                    elig = cand[np.argsort(lev)[::-1][:max(args.n_eval * 2, 12)]]
                if len(elig) == 0:
                    pbar.update(args.n_eval * len(layers)); continue
                rng = np.random.default_rng(seed + 7)
                eval_idx = np.sort(rng.choice(elig, size=min(args.n_eval, len(elig)),
                                              replace=False))

                for i in eval_idx:
                    mp = int(pos_indices[i])
                    w0 = (mp - W + 1) if W else 0
                    end_pos = int(pos_indices[i + m])            # model pos of HMM idx i+m
                    window_ids = window = input_ids[:, w0:end_pos + 1]
                    # slice indices of the readout positions (HMM idx i..i+m) within window
                    sl = [int(pos_indices[i + j]) - w0 for j in range(m + 1)]
                    mp_idx = sl[0]
                    next_toks = [int(tokens[i + j + 1]) for j in range(m + 1)]

                    # clean baseline
                    hs = _forward_slice(wrapper, window.to(device), None, None,
                                        min(sl), max(sl) + 1, device, dtype)
                    probs = _probs_slice(hs, lm_head_w, tok_ids, cap)[0]
                    for j in range(m + 1):
                        p = probs[sl[j] - min(sl)]
                        all_rows.append(dict(hmm=hmm_name, param=label, seed=seed,
                                             pos=int(i), layer=-1, condition="none",
                                             step=j, p0=p[0], p1=p[1],
                                             p2=(p[2] if len(p) > 2 else float("nan")),
                                             next_tok=next_toks[j]))

                    # injected sequences at the last K patched positions (HMM idx i-K+1..i)
                    B_true = beliefs[i - K + 1:i + 1]
                    B_flip = _kerM_donor_rows(B_true, M)
                    L_true = np.log(B_true @ M + EPS_LOG)
                    offsets = torch.tensor([int(pos_indices[i - K + 1 + t]) - w0
                                            for t in range(K)], dtype=torch.long)
                    for l in lyrs:
                        plan, conds = [], []
                        for cond, seq, dec in [("self_belief", B_true, dec_b),
                                               ("ker_delta",  B_flip, dec_b),
                                               ("self_logntp", L_true, dec_l)]:
                            bt = torch.tensor(np.asarray(seq), device=device,
                                              dtype=torch.float32)
                            val = predict_probe(bt, dec[l], use_bias=True)
                            plan.append(dict(offsets=offsets, value=val))
                            conds.append(cond)
                        batch_ids = window.to(device).expand(len(plan), -1).contiguous()
                        hs = _forward_slice(wrapper, batch_ids, l, plan,
                                            min(sl), max(sl) + 1, device, dtype)
                        probs = _probs_slice(hs, lm_head_w, tok_ids, cap)
                        for r, cond in enumerate(conds):
                            for j in range(m + 1):
                                p = probs[r, sl[j] - min(sl)]
                                all_rows.append(dict(hmm=hmm_name, param=label,
                                                     seed=seed, pos=int(i), layer=l,
                                                     condition=cond, step=j,
                                                     p0=p[0], p1=p[1],
                                                     p2=(p[2] if len(p) > 2 else float("nan")),
                                                     next_tok=next_toks[j]))
                        pbar.update(1)
                pd.DataFrame(all_rows).to_csv(
                    os.path.join(args.output_dir, f"delay_{ms}.csv"), index=False)
            pbar.close()
    print("Done.")


if __name__ == "__main__":
    main()
