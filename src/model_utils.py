"""Model loading, tokenization, activation extraction, fullvocab KL.

Handles Qwen, Llama, and Gemma model families.
"""
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_CONFIGS = {
    "Qwen/Qwen3.5-9B":        {"family": "qwen",  "sdpa": True},
    "Qwen/Qwen3.5-4B":        {"family": "qwen",  "sdpa": True},
    "meta-llama/Llama-3.1-8B": {"family": "llama", "sdpa": True},
    "meta-llama/Llama-3.2-3B": {"family": "llama", "sdpa": True},
    "google/gemma-4-E4B":      {"family": "gemma", "sdpa": False},
    "google/gemma-4-E2B":      {"family": "gemma", "sdpa": False},
}

def _detect_family(name):
    for key in ["gemma", "llama", "qwen"]:
        if key in name.lower(): return key
    return "generic"

class ModelWrapper:
    def __init__(self, model, family):
        self.model = model; self.family = family
        if family == "gemma":
            self._layers = model.model.language_model.layers
            self.hidden_size = getattr(model.config, 'text_config', model.config).hidden_size
        else:
            self._layers = model.model.layers
            self.hidden_size = model.config.hidden_size
        self.n_layers = len(self._layers)
    def get_layer(self, i): return self._layers[i]
    def forward(self, input_ids, past_key_values=None, use_cache=True):
        return self.model.model(input_ids, past_key_values=past_key_values, use_cache=use_cache)

def load_model(model_name, device="cuda"):
    cfg = MODEL_CONFIGS.get(model_name, {})
    family = cfg.get("family", _detect_family(model_name))
    token = os.environ.get("HF_TOKEN")
    kw = dict(torch_dtype=torch.float16, device_map="auto")
    if token: kw["token"] = token
    if cfg.get("sdpa", family != "gemma"): kw["attn_implementation"] = "sdpa"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kw); model.eval()
    wrapper = ModelWrapper(model, family)
    print(f"Loaded {model_name} ({family}): {wrapper.n_layers} layers, d={wrapper.hidden_size}")
    return wrapper, tokenizer

def tokens_to_prompt(tokens, token_names, sep=" "):
    return sep + sep.join(token_names[t] for t in tokens)

def match_positions(input_ids, tok_ids):
    ids = input_ids[0].cpu().numpy()
    tok_id_map = {tid: zi for zi, tid in enumerate(tok_ids)}
    pos, tok = [], []
    for i, tid in enumerate(ids):
        if tid in tok_id_map: pos.append(i); tok.append(tok_id_map[tid])
    return np.array(pos), np.array(tok)

def get_tok_ids(tokenizer, token_names):
    return [tokenizer.encode(f" {n}", add_special_tokens=False)[-1] for n in token_names]

def extract_activations_chunked(wrapper, input_ids, layers, positions, chunk_size=4096, device="cuda", collect_hidden=False):
    """Extract residual stream activations. If collect_hidden=True, also return last hidden state."""
    seq_len = input_ids.shape[1]
    pos_set = set(positions.tolist()) if len(positions) > 0 else set()
    past_kv = None; acts = {l: [] for l in layers}; all_hidden = []

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = input_ids[:, start:end].to(device)
        chunk_positions = set(range(start, end))
        need_hooks = bool(chunk_positions & pos_set) if pos_set else False
        hooks = []; chunk_acts = {}

        if need_hooks:
            for l in layers:
                def make_hook(li):
                    def fn(module, inp, out):
                        h = out[0] if isinstance(out, tuple) else out
                        chunk_acts[li] = h[0]
                    return fn
                hooks.append(wrapper.get_layer(l).register_forward_hook(make_hook(l)))

        with torch.no_grad():
            out = wrapper.forward(chunk, past_key_values=past_kv, use_cache=True)

        for h in hooks: h.remove()

        if need_hooks:
            chunk_range = np.arange(start, end)
            needed = np.isin(chunk_range, positions)
            if needed.any():
                idx = torch.tensor(np.where(needed)[0], device=device)
                for l in layers: acts[l].append(chunk_acts[l][idx])

        if collect_hidden:
            all_hidden.append(out.last_hidden_state[0].half().cpu())

        past_kv = out.past_key_values
        del out, chunk_acts; torch.cuda.empty_cache()

    del past_kv; torch.cuda.empty_cache()

    for l in layers:
        acts[l] = torch.cat(acts[l], dim=0).float() if acts[l] else torch.empty(0)
    hidden_cat = torch.cat(all_hidden, dim=0) if collect_hidden else None
    return acts, hidden_cat

# def compute_fullvocab_kl(model, hidden_cat, pos_indices, n_matched, ntp_true, tok_ids, device="cuda"):
#     """Compute KL(HMM || LLM) using lm_head over full vocabulary for proper normalization."""
#     from .metrics.kl import kl_divergence
#     W = model.lm_head.weight
#     n = min(n_matched, len(ntp_true))
#     ntp_llm = np.zeros((n, len(tok_ids)))
    
#     batch_size = 512
#     for b_start in range(0, n, batch_size):
#         b_end = min(b_start + batch_size, n)
#         h = hidden_cat[pos_indices[b_start:b_end]].to(device).float()
        
#         hmm_logits = h @ W[tok_ids].float().T
        
#         lse = torch.full((len(h),), float('-inf'), device=device)
#         for i in range(0, W.shape[0], 2000):
#             partial = h @ W[i:i+2000].float().T
#             lse = torch.logaddexp(lse, torch.logsumexp(partial, dim=-1))
#             del partial
        
#         log_probs = hmm_logits - lse.unsqueeze(-1)
#         ntp_llm[b_start:b_end] = log_probs.exp().detach().cpu().numpy()
#         del h, hmm_logits, lse, log_probs
#         torch.cuda.empty_cache()
    
#     return kl_divergence(ntp_true[:n], ntp_llm)
def compute_fullvocab_kl(model, hidden_cat, pos_indices, n_matched, ntp_true, tok_ids, device="cuda", family="qwen"):
    """Compute KL(HMM || LLM) using lm_head over full vocabulary for proper normalization."""
    from .metrics.kl import kl_divergence
    W = model.lm_head.weight
    n = min(n_matched, len(ntp_true))
    ntp_llm = np.zeros((n, len(tok_ids)))
    
    # Gemma uses logit softcapping
    cap = None
    if family == "gemma":
        cap = getattr(getattr(model.config, 'text_config', model.config), 'final_logit_softcapping', None)
    
    batch_size = 512
    for b_start in range(0, n, batch_size):
        b_end = min(b_start + batch_size, n)
        h = hidden_cat[pos_indices[b_start:b_end]].to(device).float()
        
        hmm_logits = h @ W[tok_ids].float().T
        if cap is not None: hmm_logits = torch.tanh(hmm_logits / cap) * cap
        
        lse = torch.full((len(h),), float('-inf'), device=device)
        for i in range(0, W.shape[0], 2000):
            partial = h @ W[i:i+2000].float().T
            if cap is not None: partial = torch.tanh(partial / cap) * cap
            lse = torch.logaddexp(lse, torch.logsumexp(partial, dim=-1))
            del partial
        
        log_probs = hmm_logits - lse.unsqueeze(-1)
        ntp_llm[b_start:b_end] = log_probs.exp().detach().cpu().numpy()
        del h, hmm_logits, lse, log_probs
        torch.cuda.empty_cache()
    
    return kl_divergence(ntp_true[:n], ntp_llm)