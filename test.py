# Quick KL sanity check for Gemma — one param, one seed, one HMM
import torch, numpy as np, sys, os
sys.path.insert(0, '../..')
from configs.hmm_configs import HMMS
from src.hmm import stationary_distribution, sample_hmm_sequence, full_bayesian_beliefs, next_token_probs
from src.metrics.kl import kl_divergence
from src.model_utils import load_model, tokens_to_prompt, match_positions, get_tok_ids, extract_activations_chunked, compute_fullvocab_kl

MODEL = "google/gemma-4-E2B"  # or gemma-4-E4B
device = "cuda"
wrapper, tokenizer = load_model(MODEL, device)

hmm_name = "Arch"
cfg = HMMS[hmm_name]
param = cfg["params"][0]
label = cfg["label_fn"](param)
T = cfg["fn"](*param); T_stack = np.stack(T); pi = stationary_distribution(T)
tok_ids = get_tok_ids(tokenizer, cfg["token_names"])

tokens = sample_hmm_sequence(T, pi, 5000, seed=0)
beliefs = full_bayesian_beliefs(tokens.astype(np.int64), T_stack, pi)
ntp_true = next_token_probs(beliefs, T)

prompt = tokens_to_prompt(tokens, cfg["token_names"])
input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
pos_indices, _ = match_positions(input_ids, tok_ids)
n_matched = min(len(tokens), len(pos_indices))

_, hidden_cat = extract_activations_chunked(
    wrapper, input_ids, [], np.array([], dtype=int),
    4096, device, collect_hidden=True)

kl_llm = compute_fullvocab_kl(wrapper.model, hidden_cat, pos_indices[:n_matched],
                               n_matched, ntp_true, tok_ids, device, family=wrapper.family)

# Order-1
T_o1 = cfg["order_one_fn"](*param); pi_o1 = stationary_distribution(T_o1)
T_o1_stack = np.stack(T_o1)
n_tok = cfg["n_tokens"]
ntp_o1_lut = np.zeros((n_tok, n_tok))
for zp in range(n_tok):
    b = pi_o1 @ T_o1_stack[zp]; b /= b.sum()
    for zn in range(n_tok): ntp_o1_lut[zp, zn] = (b @ T_o1_stack[zn]).sum()
kl_o1 = kl_divergence(ntp_true[:n_matched], ntp_o1_lut[tokens[:n_matched]])

# Order-0
T_o0 = cfg["order_zero_fn"](*param); pi_o0 = stationary_distribution(T_o0)
ntp_o0_row = next_token_probs(pi_o0.reshape(1,-1), T_o0)[0]
kl_o0 = kl_divergence(ntp_true[:n_matched], np.broadcast_to(ntp_o0_row, ntp_true[:n_matched].shape))

# Last 1000 tokens (converged)
print(f"Model: {MODEL}")
print(f"HMM: {hmm_name} ({label})")
print(f"KL(HMM || LLM)     last 1k mean: {kl_llm[-1000:].mean():.6f}")
print(f"KL(HMM || Order-1)  last 1k mean: {kl_o1[-1000:].mean():.6f}")
print(f"KL(HMM || Order-0)  last 1k mean: {kl_o0[-1000:].mean():.6f}")
print(f"LLM < Order-1: {kl_llm[-1000:].mean() < kl_o1[-1000:].mean()}")
print(f"LLM < Order-0: {kl_llm[-1000:].mean() < kl_o0[-1000:].mean()}")