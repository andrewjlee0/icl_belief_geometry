"""Decompose the LLM's converged next-token KL into off-alphabet leak + within-alphabet.

The published KL (run_kl.py) uses the full-vocabulary softmax, so the LLM pays
-log(mass on {letter tokens}) that no k-HMM baseline pays. Exactly:
  KL(p_true || p_llm) = -log(mass_t) + KL(p_true || p_llm / mass_t)
This script reports, per (param, seed), means over the late window (t >= probe_start):
  kl_total (reproduces published), kl_within (renormalized), leak (= mean -log mass),
  eps (= mean off-alphabet mass).

Representative params only. Output: kldecomp_{model}.csv
"""
import argparse, gc, os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs.hmm_configs import HMMS
from src.hmm import stationary_distribution, sample_hmm_sequence, full_bayesian_beliefs, next_token_probs
from src.model_utils import load_model, tokens_to_prompt, match_positions, get_tok_ids, extract_activations_chunked

REPS = [("Mess3", "a=0.01, x=0.02"), ("Arch", "a=0.99"),
        ("Wing", "a=0.98, x=0.4"), ("Strata", "a=0.97, t0=0.38, t1=0.54")]
EPS = 1e-12


def fullvocab_probs(model, hidden_cat, pos_indices, n, tok_ids, device, family):
    """Model's full-vocab-softmax probabilities of the alphabet tokens. (n, |alphabet|)."""
    W = model.lm_head.weight
    cap = None
    if family == "gemma":
        cap = getattr(getattr(model.config, 'text_config', model.config), 'final_logit_softcapping', None)
    out = np.zeros((n, len(tok_ids)))
    for b in range(0, n, 512):
        e = min(b + 512, n)
        h = hidden_cat[pos_indices[b:e]].to(device).float()
        hmm_logits = h @ W[tok_ids].float().T
        if cap is not None: hmm_logits = torch.tanh(hmm_logits / cap) * cap
        lse = torch.full((len(h),), float('-inf'), device=device)
        for i in range(0, W.shape[0], 2000):
            partial = h @ W[i:i + 2000].float().T
            if cap is not None: partial = torch.tanh(partial / cap) * cap
            lse = torch.logaddexp(lse, torch.logsumexp(partial, dim=-1))
            del partial
        out[b:e] = (hmm_logits - lse.unsqueeze(-1)).exp().detach().cpu().numpy()
        del h, hmm_logits, lse
        torch.cuda.empty_cache()
    return out


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--model", default="Qwen/Qwen3.5-9B")
    P.add_argument("--seq_len", type=int, default=20000)
    P.add_argument("--probe_start", type=int, default=15000)
    P.add_argument("--n_seeds", type=int, default=10)
    P.add_argument("--chunk_size", type=int, default=4096)
    P.add_argument("--output_dir", default="results")
    P.add_argument("--device", default="cuda")
    args = P.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    ms = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "")
    wrapper, tokenizer = load_model(args.model, device)

    rows = []
    for hmm_name, plabel in REPS:
        cfg = HMMS[hmm_name]
        param = next(p for p in cfg["params"] if cfg["label_fn"](p) == plabel)
        tok_ids = get_tok_ids(tokenizer, cfg["token_names"])
        T = cfg["fn"](*param); T_stack = np.stack(T); pi = stationary_distribution(T)
        for seed in range(args.n_seeds):
            tokens = sample_hmm_sequence(T, pi, args.seq_len, seed=seed)
            prompt = tokens_to_prompt(tokens, cfg["token_names"])
            input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
            pos_indices, _ = match_positions(input_ids, tok_ids)
            n_matched = min(len(tokens), len(pos_indices))
            _, hidden_cat = extract_activations_chunked(
                wrapper, input_ids, [], np.array([], dtype=int),
                args.chunk_size, device, collect_hidden=True)
            beliefs = full_bayesian_beliefs(tokens.astype(np.int64), T_stack, pi)
            ntp_true = next_token_probs(beliefs, T)[:n_matched]
            p_llm = fullvocab_probs(wrapper.model, hidden_cat, pos_indices[:n_matched],
                                    n_matched, tok_ids, device, wrapper.family)
            del hidden_cat
            w = slice(args.probe_start, n_matched)
            pt, pl = ntp_true[w], p_llm[w]
            mass = pl.sum(axis=1)
            kl_tot = (pt * (np.log(pt + EPS) - np.log(pl + EPS))).sum(axis=1)
            pl_ren = pl / mass[:, None]
            kl_win = (pt * (np.log(pt + EPS) - np.log(pl_ren + EPS))).sum(axis=1)
            rows.append(dict(hmm=hmm_name, param=plabel, seed=seed,
                             kl_total=float(kl_tot.mean()), kl_within=float(kl_win.mean()),
                             leak=float((-np.log(mass)).mean()), eps=float((1 - mass).mean())))
            print(f"{hmm_name} [{plabel}] s{seed}: total {kl_tot.mean():.4f} "
                  f"within {kl_win.mean():.4f} leak {(-np.log(mass)).mean():.4f}", flush=True)
            gc.collect(); torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, f"kldecomp_{ms}.csv"), index=False)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
