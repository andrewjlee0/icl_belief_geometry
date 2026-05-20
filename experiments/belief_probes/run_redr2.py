"""Reduced-order R²: probe activations for k-suffix beliefs from 3 source distributions.

Sources: real HMM, order-1 HMM, order-0 (random tokens).
For each source, generates a sequence, extracts activations, then probes for
k-suffix beliefs computed under the REAL HMM.

Trick: concatenates all k-suffix beliefs horizontally, fits ONE probe, slices R² per k.

Outputs: redr2_{model}.csv
"""
import argparse, gc, os, sys
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs.hmm_configs import HMMS
from src.hmm import (stationary_distribution, sample_hmm_sequence, full_bayesian_beliefs,
                     precompute_belief_tables, compute_k_beliefs)
from src.metrics.probes import fit_and_evaluate_multi_stacked
from src.model_utils import load_model, tokens_to_prompt, match_positions, get_tok_ids, extract_activations_chunked

def main():
    P = argparse.ArgumentParser()
    P.add_argument("--model", default="Qwen/Qwen3.5-9B")
    P.add_argument("--seq_len", type=int, default=20000)
    P.add_argument("--probe_start", type=int, default=15000)
    P.add_argument("--n_seeds", type=int, default=10)
    P.add_argument("--train_frac", type=float, default=0.2)
    P.add_argument("--chunk_size", type=int, default=4096)
    P.add_argument("--k_max", type=int, default=20)
    P.add_argument("--output_dir", default="results")
    P.add_argument("--families", nargs="+", default=None)
    P.add_argument("--device", default="cuda")
    args = P.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    wrapper, tokenizer = load_model(args.model, device)
    layers = list(range(wrapper.n_layers))
    ms = args.model.split("/")[-1].lower().replace("-","_").replace(".","")
    K_VALUES = list(range(1, args.k_max + 1))

    o0_cache = {}  # shared across families with same n_tok
    all_rows = []

    for hmm_name in (args.families or list(HMMS.keys())):
        cfg = HMMS.get(hmm_name);
        if not cfg: continue
        print(f"\n===== {hmm_name} =====")
        tok_ids = get_tok_ids(tokenizer, cfg["token_names"])
        n_tok = cfg["n_tokens"]; n_states = cfg["n_states"]
        max_k_lookup = 20 if n_tok == 2 else 12

        # ── Order-0 activations: shared per token count ──
        if n_tok not in o0_cache:
            o0_cache[n_tok] = {}
            for seed in range(args.n_seeds):
                rng = np.random.default_rng(seed + 9999)
                tokens = rng.integers(0, n_tok, size=args.seq_len)
                prompt = tokens_to_prompt(tokens, cfg["token_names"])
                input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
                pos_indices, tok_at_pos = match_positions(input_ids, tok_ids)
                n_matched = min(len(tokens), len(pos_indices))
                late_pos = pos_indices[args.probe_start:n_matched]
                acts, _ = extract_activations_chunked(wrapper, input_ids, layers, late_pos, args.chunk_size, device)
                o0_cache[n_tok][seed] = {
                    "tok_at_pos": tok_at_pos[:n_matched].astype(np.int64),
                    "acts": {l: acts[l].cpu().numpy() for l in layers},
                    "n": len(late_pos), "probe_start": args.probe_start,
                }
                del acts; gc.collect(); torch.cuda.empty_cache()
            print(f"  Order-0 ({n_tok}-tok) cached ({args.n_seeds} seeds)")

        pbar = tqdm(total=len(cfg["params"])*2*args.n_seeds, desc=hmm_name)
        for param in cfg["params"]:
            label = cfg["label_fn"](param)
            T = cfg["fn"](*param); T_stack = np.stack(T); pi = stationary_distribution(T)
            T_o1 = cfg["order_one_fn"](*param); pi_o1 = stationary_distribution(T_o1)
            small_ks = [k for k in K_VALUES if k <= max_k_lookup]
            tables = precompute_belief_tables(small_ks, T, pi) if small_ks else {}

            param_data = {}
            for seed in range(args.n_seeds):
                param_data[("order-0", seed)] = o0_cache[n_tok][seed]

            for dist_name, (T_d, pi_d) in [("real", (T, pi)), ("order-1", (T_o1, pi_o1))]:
                for seed in range(args.n_seeds):
                    tokens = sample_hmm_sequence(T_d, pi_d, args.seq_len, seed=seed)
                    prompt = tokens_to_prompt(tokens, cfg["token_names"])
                    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
                    pos_indices, tok_at_pos = match_positions(input_ids, tok_ids)
                    n_matched = min(len(tokens), len(pos_indices))
                    late_pos = pos_indices[args.probe_start:n_matched]
                    acts, _ = extract_activations_chunked(wrapper, input_ids, layers, late_pos, args.chunk_size, device)
                    param_data[(dist_name, seed)] = {
                        "tok_at_pos": tok_at_pos[:n_matched].astype(np.int64),
                        "acts": {l: acts[l].cpu().numpy() for l in layers},
                        "n": len(late_pos), "probe_start": args.probe_start,
                    }
                    del acts; gc.collect(); torch.cuda.empty_cache(); pbar.update(1)

            # ── R² per dist/seed/layer/k ──
            for dist_name in ["real", "order-1", "order-0"]:
                for seed in range(args.n_seeds):
                    d = param_data[(dist_name, seed)]
                    tok = d["tok_at_pos"]
                    beliefs_by_k = compute_k_beliefs(tok, d["probe_start"], d["n"], K_VALUES, tables, T_stack, pi, n_tok, max_k_lookup)
                    idx_tr, idx_te = train_test_split(np.arange(d["n"]), train_size=args.train_frac, random_state=seed)
                    y_all_tr = np.hstack([beliefs_by_k[k][idx_tr] for k in K_VALUES])
                    y_all_te = np.hstack([beliefs_by_k[k][idx_te] for k in K_VALUES])
                    Y_tr = torch.tensor(y_all_tr, device=device, dtype=torch.float32)
                    Y_te = torch.tensor(y_all_te, device=device, dtype=torch.float32)
                    for l in layers:
                        X_tr = torch.tensor(d["acts"][l][idx_tr], device=device, dtype=torch.float32)
                        X_te = torch.tensor(d["acts"][l][idx_te], device=device, dtype=torch.float32)
                        r2s = fit_and_evaluate_multi_stacked(X_tr, X_te, Y_tr, Y_te, n_states, K_VALUES, use_bias=True)
                        for k, r2 in r2s.items():
                            all_rows.append({"hmm":hmm_name,"param":label,"dist":dist_name,"layer":l,"k":k,"seed":seed,"R2":r2})
                        del X_tr, X_te; torch.cuda.empty_cache()
                    del Y_tr, Y_te
            del param_data, tables; gc.collect(); torch.cuda.empty_cache()
        pbar.close()
        pd.DataFrame(all_rows).to_csv(os.path.join(args.output_dir, f"redr2_{ms}.csv"), index=False)
    print(f"Done. {len(all_rows)} rows.")

if __name__ == "__main__": main()
