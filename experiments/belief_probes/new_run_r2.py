"""R² probing with shuffled/random controls + geometry extraction at best layer.

Outputs: r2_{model}.csv, geom_{model}.npz
"""
import argparse, gc, sys, os
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs.hmm_configs import HMMS, REPRESENTATIVES
from src.hmm import stationary_distribution, sample_hmm_sequence, full_bayesian_beliefs
from src.metrics.probes import fit_and_evaluate_multi, fit_probe, predict_probe, compute_r2
from src.model_utils import load_model, tokens_to_prompt, match_positions, get_tok_ids, extract_activations_chunked

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
    P.add_argument("--device", default="cuda")
    args = P.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    wrapper, tokenizer = load_model(args.model, device)
    layers = list(range(wrapper.n_layers))
    ms = args.model.split("/")[-1].lower().replace("-","_").replace(".","")

    all_rows = []
    families = args.families or list(HMMS.keys())
    for hmm_name in families:
        cfg = HMMS.get(hmm_name); 
        if not cfg: continue
        print(f"\n===== {hmm_name} =====")
        tok_ids = get_tok_ids(tokenizer, cfg["token_names"])
        pbar = tqdm(total=len(cfg["params"])*args.n_seeds, desc=hmm_name)
        for param in cfg["params"]:
            label = cfg["label_fn"](param)
            T = cfg["fn"](*param); T_stack = np.stack(T); pi = stationary_distribution(T)
            for seed in range(args.n_seeds):
                tokens = sample_hmm_sequence(T, pi, args.seq_len, seed=seed)
                beliefs = full_bayesian_beliefs(tokens.astype(np.int64), T_stack, pi)
                prompt = tokens_to_prompt(tokens, cfg["token_names"])
                input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
                pos_indices, _ = match_positions(input_ids, tok_ids)
                n_matched = min(len(tokens), len(pos_indices))
                y_real = beliefs[args.probe_start:n_matched]; n_late = len(y_real)
                late_pos = pos_indices[args.probe_start:n_matched]
                if n_late == 0: pbar.update(1); continue
                rng_sh = np.random.default_rng(seed+77777)
                y_shuffle = y_real[rng_sh.permutation(n_late)]
                rng_rd = np.random.default_rng(seed+88888)
                y_random = rng_rd.dirichlet(np.ones(cfg["n_states"]), size=n_late)
                acts, _ = extract_activations_chunked(wrapper, input_ids, layers, late_pos, args.chunk_size, device)
                idx_tr, idx_te = train_test_split(np.arange(n_late), train_size=args.train_frac, random_state=seed)
                tgts_np = {"real": y_real, "shuffle": y_shuffle, "random": y_random}
                for l in layers:
                    X = acts[l]; 
                    if X.numel() == 0: continue
                    tgts = {t: (torch.tensor(y[idx_tr],device=device,dtype=torch.float32),
                               torch.tensor(y[idx_te],device=device,dtype=torch.float32)) for t,y in tgts_np.items()}
                    r2s = fit_and_evaluate_multi(X[idx_tr], X[idx_te], tgts, use_bias=True)
                    for t, r2 in r2s.items():
                        all_rows.append({"hmm":hmm_name,"param":label,"layer":l,"seed":seed,"target":t,"R2":r2})
                del acts; gc.collect(); torch.cuda.empty_cache(); pbar.update(1)
        pbar.close()
        pd.DataFrame(all_rows).to_csv(os.path.join(args.output_dir, f"r2_{ms}.csv"), index=False)

    # ── Geometry extraction at best layer (representatives only) ──
    # Uses float64 for precise geometry visualization
    r2_df = pd.DataFrame(all_rows)
    geom = {}
    for hmm_name, rep_param in REPRESENTATIVES.items():
        cfg = HMMS.get(hmm_name); 
        if not cfg or hmm_name not in r2_df["hmm"].values: continue
        label = cfg["label_fn"](rep_param)
        sub = r2_df[(r2_df["hmm"]==hmm_name)&(r2_df["param"]==label)&(r2_df["target"]=="real")]
        if len(sub)==0: continue
        best_layer = int(sub.groupby("layer")["R2"].mean().idxmax())
        T = cfg["fn"](*rep_param); T_stack = np.stack(T); pi = stationary_distribution(T)
        tok_ids = get_tok_ids(tokenizer, cfg["token_names"])
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
        geom[hmm_name] = {"true": y_true, "pred": y_pred, "best_layer": best_layer, "param": label}
        del acts; gc.collect(); torch.cuda.empty_cache()
        print(f"  {hmm_name} geom: layer {best_layer}, R²={compute_r2(torch.tensor(y_true), torch.tensor(y_pred)):.4f}")
    np.savez(os.path.join(args.output_dir, f"geom_{ms}.npz"),
             **{f"{k}_{v}": (np.array(geom[k][v]) if not isinstance(geom[k][v], np.ndarray) else geom[k][v])
                for k in geom for v in ["true", "pred", "best_layer", "param"]})
    print("Done.")

if __name__ == "__main__": main()