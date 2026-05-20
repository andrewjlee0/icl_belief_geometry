"""NTP vs beliefs: 5 curves per HMM family.

Layer-varying curves (probed from activations):
  - act → beliefs (same as run_r2, but included for the plot)
  - act → ntp
  - act → log_ntp

Constant baselines (model-free, pooled across all seeds):
  - ntp → beliefs  (WITH BIAS)
  - log_ntp → beliefs  (WITH BIAS)

Outputs: obsprob_{model}.csv
"""
import argparse, gc, os, sys
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs.hmm_configs import HMMS
from src.hmm import stationary_distribution, sample_hmm_sequence, full_bayesian_beliefs, next_token_probs
from src.metrics.probes import fit_and_evaluate_multi
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

    for hmm_name in (args.families or list(HMMS.keys())):
        cfg = HMMS.get(hmm_name);
        if not cfg: continue
        print(f"\n===== {hmm_name} =====")
        tok_ids = get_tok_ids(tokenizer, cfg["token_names"])
        pbar = tqdm(total=len(cfg["params"])*args.n_seeds, desc=hmm_name)

        for param in cfg["params"]:
            label = cfg["label_fn"](param)
            T = cfg["fn"](*param); T_stack = np.stack(T); pi = stationary_distribution(T)

            # ── Model-free baselines: ntp→beliefs, log_ntp→beliefs (pooled, WITH BIAS) ──
            all_b, all_ntp = [], []
            for seed in range(args.n_seeds):
                tokens = sample_hmm_sequence(T, pi, args.seq_len, seed=seed)
                beliefs = full_bayesian_beliefs(tokens.astype(np.int64), T_stack, pi)
                ntp = next_token_probs(beliefs, T)
                all_b.append(beliefs[args.probe_start:]); all_ntp.append(ntp[args.probe_start:])
            b_pool = np.concatenate(all_b); ntp_pool = np.concatenate(all_ntp)
            log_ntp_pool = np.log(ntp_pool + 1e-12)

            # ntp → beliefs (WITH BIAS: append ones column)
            X_ntp = np.hstack([ntp_pool, np.ones((len(ntp_pool), 1))])
            W_ntp = np.linalg.pinv(X_ntp) @ b_pool
            pred_ntp = X_ntp @ W_ntp
            ss_r = ((b_pool - pred_ntp)**2).sum(); ss_t = ((b_pool - b_pool.mean(0))**2).sum()
            r2_ntp_to_b = 1.0 - ss_r / ss_t

            # log_ntp → beliefs (WITH BIAS)
            X_log = np.hstack([log_ntp_pool, np.ones((len(log_ntp_pool), 1))])
            W_log = np.linalg.pinv(X_log) @ b_pool
            pred_log = X_log @ W_log
            ss_r = ((b_pool - pred_log)**2).sum(); ss_t = ((b_pool - b_pool.mean(0))**2).sum()
            r2_logntp_to_b = 1.0 - ss_r / ss_t

            del all_b, all_ntp, b_pool, ntp_pool, log_ntp_pool

            # ── Per-seed: act→beliefs, act→ntp, act→log_ntp ──
            for seed in range(args.n_seeds):
                tokens = sample_hmm_sequence(T, pi, args.seq_len, seed=seed)
                beliefs = full_bayesian_beliefs(tokens.astype(np.int64), T_stack, pi)
                ntp = next_token_probs(beliefs, T)
                prompt = tokens_to_prompt(tokens, cfg["token_names"])
                input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
                pos_indices, _ = match_positions(input_ids, tok_ids)
                n_matched = min(len(tokens), len(pos_indices))
                y_beliefs = beliefs[args.probe_start:n_matched]; n_late = len(y_beliefs)
                y_ntp = ntp[args.probe_start:n_matched]
                y_log_ntp = np.log(y_ntp + 1e-12)
                late_pos = pos_indices[args.probe_start:n_matched]
                if n_late == 0: pbar.update(1); continue
                acts, _ = extract_activations_chunked(wrapper, input_ids, layers, late_pos, args.chunk_size, device)
                idx_tr, idx_te = train_test_split(np.arange(n_late), train_size=args.train_frac, random_state=seed)
                tgts_np = {"act→beliefs": y_beliefs, "act→ntp": y_ntp, "act→log_ntp": y_log_ntp}
                for l in layers:
                    X = acts[l];
                    if X.numel() == 0: continue
                    tgts = {t: (torch.tensor(y[idx_tr],device=device,dtype=torch.float32),
                               torch.tensor(y[idx_te],device=device,dtype=torch.float32)) for t,y in tgts_np.items()}
                    r2s = fit_and_evaluate_multi(X[idx_tr], X[idx_te], tgts, use_bias=True)
                    for t, r2 in r2s.items():
                        all_rows.append({"hmm":hmm_name,"param":label,"layer":l,"seed":seed,"target":t,"R2":r2})
                    # Constant baselines (same value every layer/seed)
                    all_rows.append({"hmm":hmm_name,"param":label,"layer":l,"seed":seed,"target":"ntp→beliefs","R2":r2_ntp_to_b})
                    all_rows.append({"hmm":hmm_name,"param":label,"layer":l,"seed":seed,"target":"log_ntp→beliefs","R2":r2_logntp_to_b})
                del acts; gc.collect(); torch.cuda.empty_cache(); pbar.update(1)
        pbar.close()
        pd.DataFrame(all_rows).to_csv(os.path.join(args.output_dir, f"obsprob_{ms}.csv"), index=False)
    print(f"Done. {len(all_rows)} rows.")

if __name__ == "__main__": main()
