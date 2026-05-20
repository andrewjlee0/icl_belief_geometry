"""Cross-parameterization R²: entry (i,j) = probe from param i's activations
to param j's beliefs computed on param i's token sequence.

Paper: "entry (i, j) is the R² of a probe trained from the activations produced
on parametrization i's token sequence to the belief states computed with
parametrization j on the same token sequence"

Two outputs:
  - cross_{model}.csv: empirical cross-R² (activations → beliefs, WITH BIAS)
  - gt_cross_{model}.csv: ground-truth cross-R² (beliefs_i → beliefs_j, WITH BIAS)
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse, gc, sys
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs.hmm_configs import HMMS
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
    ms = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "")
    cross_rows = []

    for hmm_name in (args.families or list(HMMS.keys())):
        cfg = HMMS.get(hmm_name)
        if not cfg: continue
        print(f"\n===== {hmm_name} =====")
        tok_ids = get_tok_ids(tokenizer, cfg["token_names"])
        param_labels = [cfg["label_fn"](p) for p in cfg["params"]]
        n_states = cfg["n_states"]

        # Precompute all T_stacks and pis
        all_T = {}
        all_pi = {}
        for param in cfg["params"]:
            label = cfg["label_fn"](param)
            T = cfg["fn"](*param)
            all_T[label] = np.stack(T)
            all_pi[label] = stationary_distribution(T)

        pbar = tqdm(total=len(cfg["params"]) * args.n_seeds, desc=hmm_name)

        for source_param in cfg["params"]:
            source_label = cfg["label_fn"](source_param)

            for seed in range(args.n_seeds):
                # ── Generate source sequence and get activations ──
                tokens = sample_hmm_sequence(
                    cfg["fn"](*source_param), all_pi[source_label],
                    args.seq_len, seed=seed
                )
                tok_i64 = tokens.astype(np.int64)
                prompt = tokens_to_prompt(tokens, cfg["token_names"])
                input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
                pos_indices, tok_at_pos = match_positions(input_ids, tok_ids)
                n_matched = min(len(tokens), len(pos_indices))
                late_pos = pos_indices[args.probe_start:n_matched]
                n_late = len(late_pos)
                if n_late == 0:
                    pbar.update(1); continue

                acts, _ = extract_activations_chunked(
                    wrapper, input_ids, layers, late_pos, args.chunk_size, device
                )

                # ── Compute beliefs under ALL params on THIS source sequence ──
                beliefs = {}
                for target_label in param_labels:
                    b = full_bayesian_beliefs(tok_i64, all_T[target_label], all_pi[target_label])
                    beliefs[target_label] = b[args.probe_start:n_matched]

                # ── Train/test split ──
                idx_tr, idx_te = train_test_split(
                    np.arange(n_late), train_size=args.train_frac, random_state=seed
                )

                # ── Probe: for each layer, for each target param ──
                for l in layers:
                    X = acts[l]
                    if X.numel() == 0: continue
                    X_tr, X_te = X[idx_tr], X[idx_te]

                    # Build targets dict: all params' beliefs on this source sequence
                    targets = {}
                    for target_label in param_labels:
                        y = beliefs[target_label]
                        Y_tr = torch.tensor(y[idx_tr], device=device, dtype=torch.float32)
                        Y_te = torch.tensor(y[idx_te], device=device, dtype=torch.float32)
                        targets[target_label] = (Y_tr, Y_te)

                    r2s = fit_and_evaluate_multi(X_tr, X_te, targets, use_bias=True)

                    for target_label, r2 in r2s.items():
                        cross_rows.append({
                            "hmm": hmm_name,
                            "source": source_label,
                            "target": target_label,
                            "layer": l,
                            "seed": seed,
                            "R2": r2,
                            "self": source_label == target_label,
                        })

                del acts, beliefs
                gc.collect(); torch.cuda.empty_cache()
                pbar.update(1)

        pbar.close()
        pd.DataFrame(cross_rows).to_csv(
            os.path.join(args.output_dir, f"cross_{ms}.csv"), index=False
        )

    # ═══════════════════════════════════════════════════════════
    # Ground-truth cross-R²: beliefs_i → beliefs_j on same sequence
    # (no model involved, matches paper's ground-truth matrix)
    # ═══════════════════════════════════════════════════════════
    print("\n===== Ground-truth cross-R² =====")
    gt_rows = []
    for hmm_name in (args.families or list(HMMS.keys())):
        cfg = HMMS.get(hmm_name)
        if not cfg: continue
        param_labels = [cfg["label_fn"](p) for p in cfg["params"]]

        all_T = {}; all_pi = {}
        for param in cfg["params"]:
            label = cfg["label_fn"](param)
            T = cfg["fn"](*param)
            all_T[label] = np.stack(T)
            all_pi[label] = stationary_distribution(T)

        pbar = tqdm(total=len(cfg["params"]) * args.n_seeds, desc=f"GT {hmm_name}")

        for source_param in cfg["params"]:
            source_label = cfg["label_fn"](source_param)

            for seed in range(args.n_seeds):
                # Source sequence
                tokens = sample_hmm_sequence(
                    cfg["fn"](*source_param), all_pi[source_label],
                    args.seq_len, seed=seed
                )
                tok_i64 = tokens.astype(np.int64)

                # Source beliefs (used as X in the regression)
                b_source = full_bayesian_beliefs(
                    tok_i64, all_T[source_label], all_pi[source_label]
                )[args.probe_start:]

                n = len(b_source)
                idx_tr, idx_te = train_test_split(
                    np.arange(n), train_size=args.train_frac, random_state=seed
                )

                X_tr = torch.tensor(b_source[idx_tr], device=device, dtype=torch.float32)
                X_te = torch.tensor(b_source[idx_te], device=device, dtype=torch.float32)

                # Compute beliefs under all params, build targets
                targets = {}
                for target_label in param_labels:
                    b_target = full_bayesian_beliefs(
                        tok_i64, all_T[target_label], all_pi[target_label]
                    )[args.probe_start:]
                    Y_tr = torch.tensor(b_target[idx_tr], device=device, dtype=torch.float32)
                    Y_te = torch.tensor(b_target[idx_te], device=device, dtype=torch.float32)
                    targets[target_label] = (Y_tr, Y_te)

                r2s = fit_and_evaluate_multi(X_tr, X_te, targets, use_bias=True)

                for target_label, r2 in r2s.items():
                    gt_rows.append({
                        "hmm": hmm_name,
                        "source": source_label,
                        "target": target_label,
                        "seed": seed,
                        "R2": r2,
                        "self": source_label == target_label,
                    })

                pbar.update(1)
        pbar.close()

    gt_df = pd.DataFrame(gt_rows)
    gt_df.to_csv(os.path.join(args.output_dir, f"gt_cross_{ms}.csv"), index=False)
    print(f"Done. Cross: {len(cross_rows)} rows, GT: {len(gt_rows)} rows.")

if __name__ == "__main__":
    main()
