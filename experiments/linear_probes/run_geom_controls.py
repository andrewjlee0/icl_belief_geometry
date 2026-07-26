"""Geometry visualizations for the probe CONTROLS + an early-context probe.

For each (hmm, param), at that param's best (peak seed-mean real R^2) layer from
r2_{model}.csv, on the SAME seed=0 sequence used by geom_{model}.npz:

  real (reference): EXACTLY regen_geom.py's protocol -- probe fit in float64 on ALL
    late-window positions (probe_start:), predictions saved in-sample. Reproduces the
    existing geometry figure.

  shuffle / random controls (late window): EXACTLY run_r2.py's protocol -- float32
    pseudoinverse fit (fp32's implicit singular-value truncation is the published
    regularization; fp64 here explodes out-of-sample), targets built with the run_r2
    rngs at seed=0 (shuffle: permutation rng(0+77777); random: Dirichlet rng(0+88888)),
    split train_test_split(train_size=train_frac, random_state=0), predictions saved on
    the TEST split.

  early window (first early_len positions): real targets, float32 fit, train on a
    random n_early_train positions, predictions saved on the held-out rest
    (train_test_split, random_state=0).

Outputs geomctl_{model}.npz with, per "{hmm}__{label}" key:
  best_layer, late_true, late_pred_real, late_r2_real (in-sample fp64, reference),
  late_true_te, late_pred_shuffle, late_pred_random, late_r2_shuffle, late_r2_random,
  early_true, early_pred, early_r2
"""
import argparse, gc, sys, os
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs.hmm_configs import HMMS
from src.hmm import stationary_distribution, sample_hmm_sequence, full_bayesian_beliefs
from src.metrics.probes import fit_probe, predict_probe, compute_r2
from src.model_utils import load_model, tokens_to_prompt, match_positions, get_tok_ids, extract_activations_chunked


def _fit_pred_f32(X_tr, X_te, y_tr, y_te, device):
    """run_r2's fit path: float32 pinv (implicitly regularized), out-of-sample."""
    W = fit_probe(X_tr.float(), torch.tensor(y_tr, device=device, dtype=torch.float32), use_bias=True)
    pred = predict_probe(X_te.float(), W, use_bias=True)
    r2 = compute_r2(torch.tensor(y_te, device=device, dtype=torch.float32), pred)
    return pred.cpu().numpy(), float(r2)


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--model", default="Qwen/Qwen3.5-9B")
    P.add_argument("--seq_len", type=int, default=20000)
    P.add_argument("--probe_start", type=int, default=15000)
    P.add_argument("--early_len", type=int, default=5000)
    P.add_argument("--n_early_train", type=int, default=1000)
    P.add_argument("--train_frac", type=float, default=0.2)
    P.add_argument("--chunk_size", type=int, default=4096)
    P.add_argument("--results_dir", default="results")   # r2_{model}.csv in; npz out
    P.add_argument("--families", nargs="+", default=None)
    P.add_argument("--device", default="cuda")
    args = P.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    ms = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "")

    r2_df = pd.read_csv(os.path.join(args.results_dir, f"r2_{ms}.csv"))
    wrapper, tokenizer = load_model(args.model, device)

    geom = {}
    families = args.families or list(HMMS.keys())
    for hmm_name in families:
        cfg = HMMS.get(hmm_name)
        if not cfg or hmm_name not in r2_df["hmm"].values: continue
        tok_ids = get_tok_ids(tokenizer, cfg["token_names"])
        for param in cfg["params"]:
            label = cfg["label_fn"](param)
            sub = r2_df[(r2_df["hmm"] == hmm_name) & (r2_df["param"] == label) & (r2_df["target"] == "real")]
            if len(sub) == 0: continue
            best_layer = int(sub.groupby("layer")["R2"].mean().idxmax())

            T = cfg["fn"](*param); T_stack = np.stack(T); pi = stationary_distribution(T)
            tokens = sample_hmm_sequence(T, pi, args.seq_len, seed=0)
            beliefs = full_bayesian_beliefs(tokens.astype(np.int64), T_stack, pi)
            prompt = tokens_to_prompt(tokens, cfg["token_names"])
            input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
            pos_indices, _ = match_positions(input_ids, tok_ids)
            n_matched = min(len(tokens), len(pos_indices))

            early_pos = pos_indices[:args.early_len]
            y_early = beliefs[:args.early_len]
            n_early = len(y_early)
            late_pos = pos_indices[args.probe_start:n_matched]
            y_late = beliefs[args.probe_start:n_matched]
            n_late = len(y_late)
            if n_late == 0 or n_early <= args.n_early_train: continue

            all_pos = np.concatenate([early_pos, late_pos])
            acts, _ = extract_activations_chunked(wrapper, input_ids, [best_layer], all_pos,
                                                  args.chunk_size, device)
            X = acts[best_layer]
            X_early, X_late = X[:n_early], X[n_early:]

            out = {"best_layer": best_layer}

            # -- real reference: regen_geom protocol (fp64, in-sample on ALL late positions)
            W = fit_probe(X_late.double(), torch.tensor(y_late, device=device, dtype=torch.float64),
                          use_bias=True)
            pred = predict_probe(X_late.double(), W, use_bias=True)
            out["late_true"] = y_late
            out["late_pred_real"] = pred.cpu().numpy()
            out["late_r2_real"] = float(compute_r2(
                torch.tensor(y_late, device=device, dtype=torch.float64), pred))

            # -- controls: run_r2 protocol (fp32, out-of-sample), rngs at seed=0
            rng_sh = np.random.default_rng(0 + 77777)
            y_shuffle = y_late[rng_sh.permutation(n_late)]
            rng_rd = np.random.default_rng(0 + 88888)
            y_random = rng_rd.dirichlet(np.ones(cfg["n_states"]), size=n_late)
            itr, ite = train_test_split(np.arange(n_late), train_size=args.train_frac, random_state=0)
            out["late_true_te"] = y_late[ite]
            for tname, y in [("shuffle", y_shuffle), ("random", y_random)]:
                pred, r2 = _fit_pred_f32(X_late[itr], X_late[ite], y[itr], y[ite], device)
                out[f"late_pred_{tname}"] = pred
                out[f"late_r2_{tname}"] = r2

            # -- real, late window, EARLY protocol (fp32, 1k train, held-out): the
            #    protocol-matched visual control for the early-context geometry
            pred, r2 = _fit_pred_f32(X_late[itr], X_late[ite], y_late[itr], y_late[ite], device)
            out["late_pred_real_oos"] = pred
            out["late_r2_real_oos"] = r2

            # -- early window: real targets, fp32, train n_early_train / test the rest
            jtr, jte = train_test_split(np.arange(n_early), train_size=args.n_early_train, random_state=0)
            pred, r2 = _fit_pred_f32(X_early[jtr], X_early[jte], y_early[jtr], y_early[jte], device)
            out["early_true"] = y_early[jte]
            out["early_pred"] = pred
            out["early_r2"] = r2

            geom[f"{hmm_name}__{label}"] = out
            del acts, X, X_early, X_late; gc.collect(); torch.cuda.empty_cache()
            print(f"  {hmm_name} [{label}]: L{best_layer}  real(in-sample,fp64) {out['late_r2_real']:.3f}  "
                  f"shuf {out['late_r2_shuffle']:.3f} rand {out['late_r2_random']:.3f}  "
                  f"early {out['early_r2']:.3f}", flush=True)

    np.savez(os.path.join(args.results_dir, f"geomctl_{ms}.npz"),
             **{f"{k}_{f}": np.asarray(v)
                for k, d in geom.items() for f, v in d.items()})
    print("Done.")


if __name__ == "__main__":
    main()
