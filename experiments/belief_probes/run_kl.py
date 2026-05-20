"""KL divergence: KL(HMM || LLM) vs order-1 and order-0 baselines.

Uses fullvocab KL (lm_head over entire vocabulary for proper normalization).
Outputs: kl_{model}.csv (100-token rolling average per position).
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse, gc, sys
import numpy as np, pandas as pd, torch
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs.hmm_configs import HMMS
from src.hmm import stationary_distribution, sample_hmm_sequence, full_bayesian_beliefs, next_token_probs
from src.metrics.kl import kl_divergence
from src.model_utils import load_model, tokens_to_prompt, match_positions, get_tok_ids, extract_activations_chunked, compute_fullvocab_kl

def main():
    P = argparse.ArgumentParser()
    P.add_argument("--model", default="Qwen/Qwen3.5-9B")
    P.add_argument("--seq_len", type=int, default=20000)
    P.add_argument("--probe_start", type=int, default=15000)
    P.add_argument("--n_seeds", type=int, default=10)
    P.add_argument("--chunk_size", type=int, default=4096)
    P.add_argument("--output_dir", default="results")
    P.add_argument("--families", nargs="+", default=None)
    P.add_argument("--device", default="cuda")
    args = P.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    wrapper, tokenizer = load_model(args.model, device)
    ms = args.model.split("/")[-1].lower().replace("-","_").replace(".","")
    kl_chunks = []

    for hmm_name in (args.families or list(HMMS.keys())):
        cfg = HMMS.get(hmm_name); 
        if not cfg: continue
        print(f"\n===== {hmm_name} =====")
        tok_ids = get_tok_ids(tokenizer, cfg["token_names"])
        n_tok = cfg["n_tokens"]
        pbar = tqdm(total=len(cfg["params"])*args.n_seeds, desc=f"{hmm_name} KL")
        for param in cfg["params"]:
            label = cfg["label_fn"](param)
            T = cfg["fn"](*param); T_stack = np.stack(T); pi = stationary_distribution(T)
            # Order-1
            T_o1 = cfg["order_one_fn"](*param); pi_o1 = stationary_distribution(T_o1)
            T_o1_stack = np.stack(T_o1)
            ntp_o1_lut = np.zeros((n_tok, n_tok))
            for zp in range(n_tok):
                b = pi_o1 @ T_o1_stack[zp]; b /= b.sum()
                for zn in range(n_tok): ntp_o1_lut[zp, zn] = (b @ T_o1_stack[zn]).sum()
            # Order-0
            if cfg["order_zero_fn"] is not None:
                T_o0 = cfg["order_zero_fn"](*param); pi_o0 = stationary_distribution(T_o0)
                ntp_o0_row = next_token_probs(pi_o0.reshape(1,-1), T_o0)[0]
            else:
                ntp_o0_row = np.full(n_tok, 1.0/n_tok)
            for seed in range(args.n_seeds):
                tokens = sample_hmm_sequence(T, pi, args.seq_len, seed=seed)
                prompt = tokens_to_prompt(tokens, cfg["token_names"])
                input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
                pos_indices, _ = match_positions(input_ids, tok_ids)
                n_matched = min(len(tokens), len(pos_indices))
                # Forward pass collecting hidden states only (no layer hooks needed)
                _, hidden_cat = extract_activations_chunked(
                    wrapper, input_ids, [], np.array([], dtype=int),
                    args.chunk_size, device, collect_hidden=True)
                beliefs = full_bayesian_beliefs(tokens.astype(np.int64), T_stack, pi)
                ntp_true = next_token_probs(beliefs, T)
                ntp_o1_arr = ntp_o1_lut[tokens]
                ntp_o0_arr = np.broadcast_to(ntp_o0_row, ntp_true.shape).copy()
                kl_llm = compute_fullvocab_kl(wrapper.model, hidden_cat, pos_indices[:n_matched], n_matched, ntp_true, tok_ids, device)
                kl_o1 = kl_divergence(ntp_true[:n_matched], ntp_o1_arr[:n_matched])
                kl_o0 = kl_divergence(ntp_true[:n_matched], ntp_o0_arr[:n_matched])
                del hidden_cat
                for vals, source in [(kl_llm,"LLM"),(kl_o1,"Order-1"),(kl_o0,"Order-0")]:
                    if len(vals) <= 100: continue
                    cs = np.cumsum(vals); rm = (cs[100:] - cs[:-100]) / 100
                    kl_chunks.append(pd.DataFrame({"position":np.arange(100,100+len(rm)),"KL":rm,"source":source,"hmm":hmm_name,"param":label,"seed":seed}))
                gc.collect(); torch.cuda.empty_cache(); pbar.update(1)
        pbar.close()
    kl_df = pd.concat(kl_chunks, ignore_index=True) if kl_chunks else pd.DataFrame()
    kl_df.to_csv(os.path.join(args.output_dir, f"kl_{ms}.csv"), index=False)
    print(f"Done. {len(kl_df)} rows.")

if __name__ == "__main__": main()
