"""Regenerate geom_{model}.npz with ALL params' geometries, reusing existing r2_{model}.csv.
Only does the (cheap) single-layer forward pass + probe fit per param — no R^2 recompute.
"""
import argparse, gc, sys, os
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs.hmm_configs import HMMS
from src.hmm import stationary_distribution, sample_hmm_sequence, full_bayesian_beliefs
from src.metrics.probes import fit_probe, predict_probe, compute_r2
from src.model_utils import load_model, tokens_to_prompt, match_positions, get_tok_ids, extract_activations_chunked

def main():
    P = argparse.ArgumentParser()
    P.add_argument("--model", default="Qwen/Qwen3.5-9B")
    P.add_argument("--seq_len", type=int, default=20000)
    P.add_argument("--probe_start", type=int, default=15000)
    P.add_argument("--chunk_size", type=int, default=4096)
    P.add_argument("--results_dir", default="results")     # where r2_*.csv lives + npz goes
    P.add_argument("--families", nargs="+", default=None)
    P.add_argument("--device", default="cuda")
    args = P.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    ms = args.model.split("/")[-1].lower().replace("-","_").replace(".","")

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
            sub = r2_df[(r2_df["hmm"]==hmm_name)&(r2_df["param"]==label)&(r2_df["target"]=="real")]
            if len(sub)==0: continue
            lm = sub.groupby("layer")["R2"].mean()
            best_layer, best_r2 = int(lm.idxmax()), float(lm.max())
            T = cfg["fn"](*param); T_stack = np.stack(T); pi = stationary_distribution(T)
            tokens = sample_hmm_sequence(T, pi, args.seq_len, seed=0)
            beliefs = full_bayesian_beliefs(tokens.astype(np.int64), T_stack, pi)
            prompt = tokens_to_prompt(tokens, cfg["token_names"])
            input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
            pos_indices, _ = match_positions(input_ids, tok_ids)
            n_matched = min(len(tokens), len(pos_indices))
            late_pos = pos_indices[args.probe_start:n_matched]
            y_true = beliefs[args.probe_start:n_matched]
            acts, _ = extract_activations_chunked(wrapper, input_ids, [best_layer], late_pos, args.chunk_size, device)
            X = acts[best_layer]
            W = fit_probe(X.double(), torch.tensor(y_true, device=device, dtype=torch.float64), use_bias=True)
            y_pred = predict_probe(X.double(), W, use_bias=True).cpu().numpy()
            geom[f"{hmm_name}__{label}"] = {"true": y_true, "pred": y_pred,
                                            "best_layer": best_layer, "r2": best_r2, "param": label}
            del acts; gc.collect(); torch.cuda.empty_cache()
            print(f"  {hmm_name} [{label}]: layer {best_layer}, R²={best_r2:.4f}")

    np.savez(os.path.join(args.results_dir, f"geom_{ms}.npz"),
             **{f"{k}_{v}": (np.array(geom[k][v]) if not isinstance(geom[k][v], np.ndarray) else geom[k][v])
                for k in geom for v in ["true","pred","best_layer","r2","param"]})
    print("Done.")

if __name__ == "__main__": main()