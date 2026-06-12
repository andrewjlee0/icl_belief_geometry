"""Belief-bottleneck test, per layer: act -> probe -> etahat -> logNTP.

For each (family, param, seed, layer):
  - fit linear probe act->beliefs on train split        (stage 1)
  - map etahat -> logNTP, linear and quadratic, on train (stage 2)
  - score the end-to-end pipeline on the held-out test split
  - also score act -> logNTP directly (same split)
  - also compute the analytic ceiling eta(true) -> logNTP (lin + quad)

All regressions are plain OLS with bias via numpy SVD lstsq (same solver
class as run_r2 / run_obsprob). All R2s variance-weighted (pooled SS).

Outputs: bottleneck_{model}.csv with targets:
  act→log_ntp, etahat→log_ntp, etahat_quad→log_ntp,
  eta→log_ntp, eta_quad→log_ntp   (eta targets constant over layers)

Defaults: main-text params only. Use --all_params for the full sweep.
"""
import argparse, gc, sys, os
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs.hmm_configs import HMMS
from src.hmm import stationary_distribution, sample_hmm_sequence, full_bayesian_beliefs, next_token_probs
from src.model_utils import load_model, tokens_to_prompt, match_positions, get_tok_ids, extract_activations_chunked

MAIN_TEXT = {
    "Mess3":  "a=0.01, x=0.02",
    "Arch":   "a=0.99",
    "Wing":   "a=0.98, x=0.4",
    "Strata": "a=0.97, t0=0.38, t1=0.54",
}

def _quad(b):
    m = b.shape[1]
    cross = np.stack([b[:, a] * b[:, c] for a in range(m) for c in range(a, m)], 1)
    return np.hstack([b, cross])

def _ols_pred(A_tr, B_tr, A_te):
    """Plain OLS with bias via numpy SVD lstsq, float64."""
    A_tr = np.hstack([A_tr, np.ones((len(A_tr), 1))]).astype(np.float64)
    A_te = np.hstack([A_te, np.ones((len(A_te), 1))]).astype(np.float64)
    W, *_ = np.linalg.lstsq(A_tr, B_tr.astype(np.float64), rcond=None)
    return A_tr @ W, A_te @ W

def _r2vw(y, p):
    y = y.astype(np.float64)
    ssr = ((y - p) ** 2).sum()
    sst = ((y - y.mean(0)) ** 2).sum()
    return float(1.0 - ssr / sst)

def main():
    P = argparse.ArgumentParser()
    P.add_argument("--model", default="Qwen/Qwen3.5-9B")
    P.add_argument("--seq_len", type=int, default=20000)
    P.add_argument("--probe_start", type=int, default=15000)
    P.add_argument("--n_seeds", type=int, default=10)
    P.add_argument("--train_frac", type=float, default=0.2)
    P.add_argument("--chunk_size", type=int, default=4096)
    P.add_argument("--output_dir", default="results")
    P.add_argument("--families", nargs="+", default=None)
    P.add_argument("--all_params", action="store_true",
                   help="run every parametrization (default: main-text only)")
    P.add_argument("--device", default="cuda")
    args = P.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    wrapper, tokenizer = load_model(args.model, device)
    layers = list(range(wrapper.n_layers))
    ms = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "")
    all_rows = []
    out_path = os.path.join(args.output_dir, f"bottleneck_{ms}.csv")

    for hmm_name in (args.families or list(HMMS.keys())):
        if hmm_name == "Spiral":
            continue
        cfg = HMMS.get(hmm_name)
        if not cfg:
            continue
        params = cfg["params"]
        if not args.all_params:
            if hmm_name not in MAIN_TEXT:
                continue                      # drop Spiral and any other extras
            params = [p for p in params if cfg["label_fn"](p) == MAIN_TEXT[hmm_name]]
        if not params:
            continue
        print(f"\n===== {hmm_name} ({len(params)} param(s)) =====")
        tok_ids = get_tok_ids(tokenizer, cfg["token_names"])
        pbar = tqdm(total=len(params) * args.n_seeds, desc=hmm_name)

        for param in params:
            label = cfg["label_fn"](param)
            T = cfg["fn"](*param); T_stack = np.stack(T); pi = stationary_distribution(T)

            for seed in range(args.n_seeds):
                tokens = sample_hmm_sequence(T, pi, args.seq_len, seed=seed)
                beliefs = full_bayesian_beliefs(tokens.astype(np.int64), T_stack, pi)
                ntp = next_token_probs(beliefs, T)
                prompt = tokens_to_prompt(tokens, cfg["token_names"])
                input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
                pos_indices, _ = match_positions(input_ids, tok_ids)
                n_matched = min(len(tokens), len(pos_indices))
                y_beliefs = beliefs[args.probe_start:n_matched]
                y_log_ntp = np.log(ntp[args.probe_start:n_matched] + 1e-12)
                late_pos = pos_indices[args.probe_start:n_matched]
                n_late = len(y_beliefs)
                if n_late == 0:
                    pbar.update(1); continue

                idx_tr, idx_te = train_test_split(np.arange(n_late),
                                                  train_size=args.train_frac, random_state=seed)
                yb_tr, yb_te = y_beliefs[idx_tr], y_beliefs[idx_te]
                yl_tr, yl_te = y_log_ntp[idx_tr], y_log_ntp[idx_te]

                # analytic ceilings: eta(true) -> logNTP (constant over layers)
                _, p_te = _ols_pred(yb_tr, yl_tr, yb_te)
                r2_eta_lin = _r2vw(yl_te, p_te)
                _, p_te = _ols_pred(_quad(yb_tr), yl_tr, _quad(yb_te))
                r2_eta_quad = _r2vw(yl_te, p_te)

                acts, _ = extract_activations_chunked(wrapper, input_ids, layers,
                                                      late_pos, args.chunk_size, device)
                for l in layers:
                    X = acts[l]
                    if X.numel() == 0:
                        continue
                    X_np = X.float().cpu().numpy()
                    X_tr, X_te = X_np[idx_tr], X_np[idx_te]
                    # direct: act -> logNTP
                    _, p_te = _ols_pred(X_tr, yl_tr, X_te)
                    r2_act = _r2vw(yl_te, p_te)
                    # stage 1: probe act -> beliefs; etahat on both splits
                    eh_tr, eh_te = _ols_pred(X_tr, yb_tr, X_te)
                    # stage 2: etahat -> logNTP (linear, quadratic)
                    _, p_te = _ols_pred(eh_tr, yl_tr, eh_te)
                    r2_eh = _r2vw(yl_te, p_te)
                    _, p_te = _ols_pred(_quad(eh_tr), yl_tr, _quad(eh_te))
                    r2_ehq = _r2vw(yl_te, p_te)
                    for t, r2 in [("act→log_ntp", r2_act),
                                  ("etahat→log_ntp", r2_eh),
                                  ("etahat_quad→log_ntp", r2_ehq),
                                  ("eta→log_ntp", r2_eta_lin),
                                  ("eta_quad→log_ntp", r2_eta_quad)]:
                        all_rows.append({"hmm": hmm_name, "param": label, "layer": l,
                                         "seed": seed, "target": t, "R2": r2})
                del acts, X_np, X_tr, X_te
                gc.collect(); torch.cuda.empty_cache()
                pbar.update(1)
                pd.DataFrame(all_rows).to_csv(out_path, index=False)
        pbar.close()
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"Done. {len(all_rows)} rows -> {out_path}")

if __name__ == "__main__":
    main()