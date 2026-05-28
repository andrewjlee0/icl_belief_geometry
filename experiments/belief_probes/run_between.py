"""Cross-family R²: probe from family A's activations to family B's beliefs,
computed on family A's token sequence.

Same logic as run_within but across families.
Restricted to families with matching n_states: Wing ↔ Strata (both 3 states).

Entry (A, B) = R² of probing family A's activations for family B's beliefs
              on family A's token sequence.

Outputs: between_{model}.csv
"""
import argparse, gc, sys, os
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs.hmm_configs import HMMS, REPRESENTATIVES
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

    # Precompute HMM params for each family — ALL params, not just representatives
    family_hmms = {}  # {fam: [{T_stack, pi, T_matrices, param, label, token_names}, ...]}
    for fam in CROSS_FAMILIES:
        cfg = HMMS[fam]
        family_hmms[fam] = []
        for param in cfg["params"]:
            T = cfg["fn"](*param)
            family_hmms[fam].append({
                "T_stack": np.stack(T),
                "pi": stationary_distribution(T),
                "T_matrices": T,
                "param": param,
                "label": cfg["label_fn"](param),
                "token_names": cfg["token_names"],
            })

    all_rows = []
    total = sum(len(family_hmms[f]) for f in CROSS_FAMILIES) * args.n_seeds
    pbar = tqdm(total=total, desc="Between")

    for source_fam in CROSS_FAMILIES:
        for src in family_hmms[source_fam]:
            tok_ids = get_tok_ids(tokenizer, src["token_names"])

            for seed in range(args.n_seeds):
                tokens = sample_hmm_sequence(
                    src["T_matrices"], src["pi"], args.seq_len, seed=seed
                )
                tok_i64 = tokens.astype(np.int64)
                prompt = tokens_to_prompt(tokens, src["token_names"])
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

                # Compute beliefs under ALL target params across ALL families
                all_beliefs = {}
                for target_fam in CROSS_FAMILIES:
                    for tgt in family_hmms[target_fam]:
                        key = (target_fam, tgt["label"])
                        b = full_bayesian_beliefs(tok_i64, tgt["T_stack"], tgt["pi"])
                        all_beliefs[key] = b[args.probe_start:n_matched]

                idx_tr, idx_te = train_test_split(
                    np.arange(n_late), train_size=args.train_frac, random_state=seed
                )

                for l in layers:
                    X = acts[l]
                    if X.numel() == 0: continue
                    X_tr, X_te = X[idx_tr], X[idx_te]

                    targets = {}
                    for (target_fam, target_label), y in all_beliefs.items():
                        Y_tr = torch.tensor(y[idx_tr], device=device, dtype=torch.float32)
                        Y_te = torch.tensor(y[idx_te], device=device, dtype=torch.float32)
                        targets[(target_fam, target_label)] = (Y_tr, Y_te)

                    r2s = fit_and_evaluate_multi(X_tr, X_te, targets, use_bias=True)

                    for (target_fam, target_label), r2 in r2s.items():
                        all_rows.append({
                            "source_family": source_fam,
                            "target_family": target_fam,
                            "source_param": src["label"],
                            "target_param": target_label,
                            "layer": l,
                            "seed": seed,
                            "R2": r2,
                            "self": source_fam == target_fam,
                        })

                del acts, all_beliefs
                gc.collect(); torch.cuda.empty_cache()
                pbar.update(1)

    pbar.close()
    df = pd.DataFrame(all_rows)
    out_path = os.path.join(args.output_dir, f"between_{ms}.csv")
    df.to_csv(out_path, index=False)
    print(f"\nDone. {len(df)} rows saved to {out_path}")

    # ── Summary ──
    for (sf, tf), grp in df.groupby(["source_family", "target_family"]):
        peak = grp.groupby("layer")["R2"].mean().max()
        peak_l = grp.groupby("layer")["R2"].mean().idxmax()
        print(f"  {sf} → {tf}: peak R² = {peak:.4f} at layer {peak_l}")


if __name__ == "__main__":
    main()
