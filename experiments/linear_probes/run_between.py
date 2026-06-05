"""Cross-family R²: probe from family A's activations to family B's beliefs,
computed on family A's token sequence.

Mirrors run_within.py but source and target params come from DIFFERENT families.
Restricted to families with matching n_states: Wing ↔ Strata (both 3 states).

Entry (A_i, B_j) = R² of a probe trained from the activations produced on
Wing param i's token sequence to the belief states computed with Strata param j
on the same token sequence (and vice versa).

Two outputs:
  - between_{model}.csv: empirical cross-family R² (activations → beliefs, WITH BIAS)
  - gt_between_{model}.csv: ground-truth cross-family R² (beliefs_i → beliefs_j, WITH BIAS)
"""
import argparse, gc, sys, os
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs.hmm_configs import HMMS
from src.hmm import stationary_distribution, sample_hmm_sequence, full_bayesian_beliefs
from src.metrics.probes import fit_and_evaluate_multi
from src.model_utils import (load_model, tokens_to_prompt, match_positions,
                             get_tok_ids, extract_activations_chunked)

# Families to cross — must share n_states
CROSS_FAMILIES = ["Wing", "Strata"]


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--model", default="Qwen/Qwen3.5-9B")
    P.add_argument("--seq_len", type=int, default=20000)
    P.add_argument("--probe_start", type=int, default=15000)
    P.add_argument("--n_seeds", type=int, default=10)
    P.add_argument("--train_frac", type=float, default=0.2)
    P.add_argument("--chunk_size", type=int, default=4096)
    P.add_argument("--output_dir", default="results")
    P.add_argument("--device", default="cuda")
    args = P.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    wrapper, tokenizer = load_model(args.model, device)
    layers = list(range(wrapper.n_layers))
    ms = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "")

    # Verify matching n_states
    n_states_set = set(HMMS[f]["n_states"] for f in CROSS_FAMILIES)
    assert len(n_states_set) == 1, f"Families have different n_states: {n_states_set}"
    print(f"Cross-family: {CROSS_FAMILIES}, n_states={n_states_set.pop()}")

    # Precompute HMM params for each family
    family_cfgs = {}
    family_T = {}    # {fam: {label: T_stack}}
    family_pi = {}   # {fam: {label: pi}}
    family_labels = {}
    for fam in CROSS_FAMILIES:
        cfg = HMMS[fam]
        family_cfgs[fam] = cfg
        family_T[fam] = {}
        family_pi[fam] = {}
        family_labels[fam] = []
        for param in cfg["params"]:
            label = cfg["label_fn"](param)
            T = cfg["fn"](*param)
            family_T[fam][label] = np.stack(T)
            family_pi[fam][label] = stationary_distribution(T)
            family_labels[fam].append(label)

    cross_rows = []

    # For each source family, loop over source params
    for source_fam in CROSS_FAMILIES:
        target_fam = [f for f in CROSS_FAMILIES if f != source_fam][0]
        src_cfg = family_cfgs[source_fam]
        tok_ids = get_tok_ids(tokenizer, src_cfg["token_names"])
        src_labels = family_labels[source_fam]
        tgt_labels = family_labels[target_fam]

        pbar = tqdm(total=len(src_cfg["params"]) * args.n_seeds,
                    desc=f"{source_fam} → {target_fam}")

        for source_param in src_cfg["params"]:
            source_label = src_cfg["label_fn"](source_param)
            for seed in range(args.n_seeds):
                # Generate source sequence
                tokens = sample_hmm_sequence(
                    src_cfg["fn"](*source_param),
                    family_pi[source_fam][source_label],
                    args.seq_len, seed=seed
                )
                tok_i64 = tokens.astype(np.int64)
                prompt = tokens_to_prompt(tokens, src_cfg["token_names"])
                input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
                pos_indices, _ = match_positions(input_ids, tok_ids)
                n_matched = min(len(tokens), len(pos_indices))
                late_pos = pos_indices[args.probe_start:n_matched]
                n_late = len(late_pos)
                if n_late == 0:
                    pbar.update(1); continue

                acts, _ = extract_activations_chunked(
                    wrapper, input_ids, layers, late_pos, args.chunk_size, device
                )

                # Compute beliefs under ALL target family params on this source sequence
                beliefs = {}
                for tgt_label in tgt_labels:
                    b = full_bayesian_beliefs(tok_i64, family_T[target_fam][tgt_label],
                                             family_pi[target_fam][tgt_label])
                    beliefs[tgt_label] = b[args.probe_start:n_matched]

                # Train/test split
                idx_tr, idx_te = train_test_split(
                    np.arange(n_late), train_size=args.train_frac, random_state=seed
                )

                # Probe each layer
                for l in layers:
                    X = acts[l]
                    if X.numel() == 0: continue
                    X_tr, X_te = X[idx_tr], X[idx_te]

                    targets = {}
                    for tgt_label in tgt_labels:
                        y = beliefs[tgt_label]
                        Y_tr = torch.tensor(y[idx_tr], device=device, dtype=torch.float32)
                        Y_te = torch.tensor(y[idx_te], device=device, dtype=torch.float32)
                        targets[tgt_label] = (Y_tr, Y_te)

                    r2s = fit_and_evaluate_multi(X_tr, X_te, targets, use_bias=True)

                    for tgt_label, r2 in r2s.items():
                        cross_rows.append({
                            "source_family": source_fam,
                            "target_family": target_fam,
                            "source_param": source_label,
                            "target_param": tgt_label,
                            "layer": l,
                            "seed": seed,
                            "R2": r2,
                        })

                del acts, beliefs
                gc.collect(); torch.cuda.empty_cache()
                pbar.update(1)

        pbar.close()

    # Save empirical
    df = pd.DataFrame(cross_rows)
    df.to_csv(os.path.join(args.output_dir, f"between_{ms}.csv"), index=False)

    # ═══════════════════════════════════════════════════════════
    # Ground-truth cross-family R²: beliefs_source → beliefs_target
    # (no model involved — how well do source beliefs linearly predict
    #  target beliefs from a different family on the same sequence?)
    # ═══════════════════════════════════════════════════════════
    print("\n===== Ground-truth between-family R² =====")
    gt_rows = []

    for source_fam in CROSS_FAMILIES:
        target_fam = [f for f in CROSS_FAMILIES if f != source_fam][0]
        src_cfg = family_cfgs[source_fam]
        src_labels = family_labels[source_fam]
        tgt_labels = family_labels[target_fam]

        pbar = tqdm(total=len(src_cfg["params"]) * args.n_seeds,
                    desc=f"GT {source_fam} → {target_fam}")

        for source_param in src_cfg["params"]:
            source_label = src_cfg["label_fn"](source_param)
            for seed in range(args.n_seeds):
                tokens = sample_hmm_sequence(
                    src_cfg["fn"](*source_param),
                    family_pi[source_fam][source_label],
                    args.seq_len, seed=seed
                )
                tok_i64 = tokens.astype(np.int64)

                # Source beliefs (used as X)
                b_source = full_bayesian_beliefs(
                    tok_i64, family_T[source_fam][source_label],
                    family_pi[source_fam][source_label]
                )[args.probe_start:]

                n = len(b_source)
                idx_tr, idx_te = train_test_split(
                    np.arange(n), train_size=args.train_frac, random_state=seed
                )
                X_tr = torch.tensor(b_source[idx_tr], device=device, dtype=torch.float32)
                X_te = torch.tensor(b_source[idx_te], device=device, dtype=torch.float32)

                # Target beliefs from other family
                targets = {}
                for tgt_label in tgt_labels:
                    b_target = full_bayesian_beliefs(
                        tok_i64, family_T[target_fam][tgt_label],
                        family_pi[target_fam][tgt_label]
                    )[args.probe_start:]
                    Y_tr = torch.tensor(b_target[idx_tr], device=device, dtype=torch.float32)
                    Y_te = torch.tensor(b_target[idx_te], device=device, dtype=torch.float32)
                    targets[tgt_label] = (Y_tr, Y_te)

                r2s = fit_and_evaluate_multi(X_tr, X_te, targets, use_bias=True)

                for tgt_label, r2 in r2s.items():
                    gt_rows.append({
                        "source_family": source_fam,
                        "target_family": target_fam,
                        "source_param": source_label,
                        "target_param": tgt_label,
                        "seed": seed,
                        "R2": r2,
                    })

                pbar.update(1)
        pbar.close()

    gt_df = pd.DataFrame(gt_rows)
    gt_df.to_csv(os.path.join(args.output_dir, f"gt_between_{ms}.csv"), index=False)
    print(f"\nDone. Cross: {len(cross_rows)} rows, GT: {len(gt_rows)} rows.")

    # ── Summary ──
    for (sf, tf), grp in df.groupby(["source_family", "target_family"]):
        peak = grp.groupby("layer")["R2"].mean().max()
        peak_l = grp.groupby("layer")["R2"].mean().idxmax()
        print(f"  {sf} → {tf}: peak R² = {peak:.4f} at layer {peak_l}")


if __name__ == "__main__":
    main()