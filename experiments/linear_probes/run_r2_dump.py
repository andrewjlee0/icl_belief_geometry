"""Dump best-layer activations for the late window, per (hmm, param, seed).

One forward pass per (param, seed); saves one npz per pass with fp16 activations at
that param's best published layer (from r2_{model}.csv, target='real', peak seed-mean),
plus the ground-truth beliefs. Enables CPU-only probe-protocol analyses afterwards
(cross-sequence evaluation, contiguous-split sweeps, per-position error analysis).

Output: {output_dir}/{ms}/acts_{hmm}__{label}__s{seed}.npz
  acts (n_late, d) float16, beliefs (n_late, n_states) float64, best_layer, positions
Resumes by skipping existing npz files.
"""
import argparse, gc, os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs.hmm_configs import HMMS
from src.hmm import stationary_distribution, sample_hmm_sequence, full_bayesian_beliefs
from src.model_utils import load_model, tokens_to_prompt, match_positions, get_tok_ids, extract_activations_chunked


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--model", default="Qwen/Qwen3.5-9B")
    P.add_argument("--seq_len", type=int, default=20000)
    P.add_argument("--probe_start", type=int, default=15000)
    P.add_argument("--n_seeds", type=int, default=10)
    P.add_argument("--chunk_size", type=int, default=4096)
    P.add_argument("--r2_csv", required=True, help="published r2_{model}.csv for best layers")
    P.add_argument("--output_dir", required=True)
    P.add_argument("--families", nargs="+", default=["Mess3", "Arch", "Wing", "Strata"])
    P.add_argument("--device", default="cuda")
    args = P.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    ms = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "")
    out_dir = os.path.join(args.output_dir, ms)
    os.makedirs(out_dir, exist_ok=True)

    r2 = pd.read_csv(args.r2_csv)
    r2 = r2[r2["target"] == "real"]
    wrapper, tokenizer = load_model(args.model, device)

    for hmm_name in args.families:
        cfg = HMMS.get(hmm_name)
        if not cfg: continue
        tok_ids = get_tok_ids(tokenizer, cfg["token_names"])
        for param in cfg["params"]:
            label = cfg["label_fn"](param)
            sub = r2[(r2["hmm"] == hmm_name) & (r2["param"] == label)]
            if len(sub) == 0: continue
            best_layer = int(sub.groupby("layer")["R2"].mean().idxmax())
            T = cfg["fn"](*param); T_stack = np.stack(T); pi = stationary_distribution(T)
            for seed in range(args.n_seeds):
                fn = os.path.join(out_dir, f"acts_{hmm_name}__{label}__s{seed}.npz")
                if os.path.exists(fn): continue
                tokens = sample_hmm_sequence(T, pi, args.seq_len, seed=seed)
                beliefs = full_bayesian_beliefs(tokens.astype(np.int64), T_stack, pi)
                prompt = tokens_to_prompt(tokens, cfg["token_names"])
                input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
                pos_indices, _ = match_positions(input_ids, tok_ids)
                n_matched = min(len(tokens), len(pos_indices))
                late_pos = pos_indices[args.probe_start:n_matched]
                if len(late_pos) == 0: continue
                acts, _ = extract_activations_chunked(wrapper, input_ids, [best_layer],
                                                      late_pos, args.chunk_size, device)
                X = acts[best_layer].half().cpu().numpy()
                np.savez(fn, acts=X, beliefs=beliefs[args.probe_start:n_matched],
                         best_layer=best_layer,
                         positions=np.arange(args.probe_start, n_matched))
                del acts, X; gc.collect(); torch.cuda.empty_cache()
                print(f"{hmm_name} [{label}] s{seed}: L{best_layer} saved", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
