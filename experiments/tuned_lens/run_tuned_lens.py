"""Per-layer tuned lens vs. logit lens over HMM data, with belief-state R2.

Self-contained for this repo's HF-transformers conventions (uses src.model_utils,
src.hmm, src.metrics.probes) — no dependency on the partner's tuned_lens_per_layer
module. Produces, per (hmm, param, layer), the KL of each lens's next-token
distribution to (a) the HMM oracle and (b) the model's own final output, plus the
belief-probe R2 — everything needed for the "KL mirrors the inverted R2 curve" figure.

Lenses (all read out over the HMM/concept tokens only):
  logit          : final_norm + unembed applied to the layer-l residual (no training;
                   the standard logit-lens baseline that suffers from basis drift).
  tuned_concept  : per-layer affine translator trained to match the MODEL's own final
                   concept distribution  -> "where does the model commit its prediction".
  tuned_hmm      : per-layer affine translator trained to match the HMM ORACLE
                   distribution           -> "where is the optimal prediction affinely
                   readable". The headline curve: its KL-to-HMM by layer should trace
                   the INVERSE of the belief-probe R2 curve.

Readout: lens_logits = unembed_concept( final_norm( translator(h_l) ) ), so for the
last layer with translator=identity it reproduces the model's own output. Alignment
matches run_r2 / the intervention code: activation at HMM-token index i -> beliefs[i],
oracle NTP = next_token_probs(beliefs[i]).

Output: tunedlens_{model}.csv  (rows: hmm, param, layer, lens, kl_hmm, kl_final, r2_belief)

Usage:
    python experiments/tuned_lens/run_tuned_lens.py --model Qwen/Qwen3.5-9B \
        --families Mess3 Arch Wing Strata --all_params --output_dir results
    python experiments/tuned_lens/run_tuned_lens.py --model meta-llama/Llama-3.2-3B --smoke
"""
import argparse, gc, os, sys
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
torch.backends.cuda.matmul.allow_tf32 = True   # ~2x on Ampere; negligible effect here
torch.backends.cudnn.allow_tf32 = True
from configs.hmm_configs import HMMS, REPRESENTATIVES
from src.hmm import (stationary_distribution, sample_hmm_sequence,
                     full_bayesian_beliefs, emission_matrix, next_token_probs)
from src.metrics.probes import fit_probe, predict_probe
from src.model_utils import (load_model, tokens_to_prompt, match_positions,
                             get_tok_ids, extract_activations_chunked)


# ── helpers ───────────────────────────────────────────────────────────────────
def _final_norm(wrapper):
    """The model's final norm module (applied after the last decoder block, before
    lm_head). Family-aware; raise a clear error if the attribute path differs."""
    m = wrapper.model
    cands = ([m.model.language_model] if wrapper.family == "gemma" else [m.model])
    for base in cands + [m.model]:
        for attr in ("norm", "final_layernorm", "ln_f"):
            if hasattr(base, attr):
                return getattr(base, attr)
    raise AttributeError("Could not locate the final norm on this model; adjust _final_norm().")


def _split(n, train_frac, seed):
    perm = np.random.default_rng(seed).permutation(n)
    n_tr = int(round(n * train_frac))
    return perm[:n_tr], perm[n_tr:]


def _r2(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean(0, keepdims=True)) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-10))


def _kl(p, q):
    """KL(p || q) per row, over the concept-token simplex. p,q: (...,C)."""
    p = np.clip(p, 1e-12, None); q = np.clip(q, 1e-12, None)
    return (p * np.log(p / q)).sum(-1)


def _readout(h, norm, Wc, cap, norm_dtype):
    """Concept logits = unembed_concept( final_norm(h) ). h:(B,d) fp32 -> (B,C) fp32.
    norm runs in the model's dtype; Wc is the concept slice of lm_head.weight (C,d)."""
    normed = norm(h.to(norm_dtype)).float()
    logits = normed @ Wc.float().T
    if cap is not None:
        logits = torch.tanh(logits / cap) * cap
    return logits


def _train_translator(X_tr, target_tr, norm, Wc, cap, norm_dtype, d, device,
                      epochs, lr, batch):
    """Affine translator (init identity) trained to minimise KL(target || lens)."""
    T = torch.nn.Linear(d, d, bias=True).to(device)
    with torch.no_grad():
        T.weight.copy_(torch.eye(d, device=device)); T.bias.zero_()
    opt = torch.optim.Adam(T.parameters(), lr=lr)
    n = X_tr.shape[0]
    logt = torch.log(target_tr.clamp_min(1e-12))
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            logits = _readout(T(X_tr[idx]), norm, Wc, cap, norm_dtype)
            logq = F.log_softmax(logits, dim=-1)
            loss = (target_tr[idx] * (logt[idx] - logq)).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return T


# ── per-(hmm, param) pipeline ──────────────────────────────────────────────────
def run_param(wrapper, tokenizer, cfg, param, layers, tok_ids, Wc, norm, cap,
              norm_dtype, args, device):
    n_states = cfg["n_states"]
    T_matrices = cfg["fn"](*param); T_stack = np.stack(T_matrices)
    pi = stationary_distribution(T_matrices); M = emission_matrix(T_matrices)
    label = cfg["label_fn"](param); ps = args.probe_start
    controls = args.controls

    # cross-HMM donor (same family, next parametrization) for the 'cross' control
    dstack = dpi = dM = None
    if "cross" in controls:
        plist = cfg["params"]; di = (plist.index(param) + 1) % len(plist)
        dT = cfg["fn"](*plist[di]); dstack = np.stack(dT)
        dpi = stationary_distribution(dT); dM = emission_matrix(dT)

    acts_pool = {l: [] for l in layers}
    oracle_pool, belief_pool, o1_pool, cross_pool = [], [], [], []
    for seed in range(args.n_seeds):
        tokens = sample_hmm_sequence(T_matrices, pi, args.seq_len, seed=seed).astype(np.int64)
        beliefs = full_bayesian_beliefs(tokens, T_stack, pi)
        prompt = tokens_to_prompt(tokens, cfg["token_names"])
        input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
        pos, _ = match_positions(input_ids, tok_ids)
        n = min(len(tokens), len(pos)); pos = pos[:n]
        if n <= ps:
            continue
        acts, _ = extract_activations_chunked(wrapper, input_ids, layers, pos[ps:n],
                                              args.chunk_size, device)
        for l in layers:
            acts_pool[l].append(acts[l].detach().cpu())
        oracle_pool.append((beliefs[ps:n] @ M).astype(np.float32))
        belief_pool.append(beliefs[ps:n].astype(np.float32))
        if "order1" in controls:                                 # order-1 (1-HMM) NTP
            tk = tokens[ps:n]
            ob = np.einsum("s,nsd->nd", pi, T_stack[tk]); ob /= ob.sum(1, keepdims=True)
            o1_pool.append((ob @ M).astype(np.float32))
        if "cross" in controls:                                  # donor HMM's NTP on these tokens
            cb = full_bayesian_beliefs(tokens, dstack, dpi)[ps:n]
            cross_pool.append((cb @ dM).astype(np.float32))
        del acts; torch.cuda.empty_cache()

    for l in layers:
        acts_pool[l] = torch.cat(acts_pool[l], dim=0)
    oracle = np.concatenate(oracle_pool, 0); belief = np.concatenate(belief_pool, 0)
    N, C = oracle.shape
    tr, te = _split(N, args.train_frac, 42)

    # model's own final concept distribution = readout of the LAST layer's activation
    last = layers[-1]
    with torch.no_grad():
        h = acts_pool[last].to(device)
        final = F.softmax(_readout(h, norm, Wc, cap, norm_dtype), dim=-1).cpu().numpy()
        del h; torch.cuda.empty_cache()

    # target distribution for each trained lens / control (mirrors run_r2's controls):
    #   shuffle = HMM NTP with position<->target correspondence permuted (rng 77777)
    #   random  = i.i.d. symmetric-Dirichlet NTP over the C tokens          (rng 88888)
    #   order1  = order-1 (1-HMM) NTP; cross = a donor HMM's NTP on these tokens
    targets = {"tuned_concept": final, "tuned_hmm": oracle}
    if "shuffle" in controls:
        targets["shuffle"] = oracle[np.random.default_rng(77777).permutation(N)]
    if "random" in controls:
        targets["random"] = np.random.default_rng(88888).dirichlet(np.ones(C), size=N).astype(np.float32)
    if "order1" in controls:
        targets["order1"] = np.concatenate(o1_pool, 0)
    if "cross" in controls:
        targets["cross"] = np.concatenate(cross_pool, 0)

    d = wrapper.hidden_size
    rows = []
    for l in layers:
        Xtr = acts_pool[l][tr].to(device); Xte = acts_pool[l][te].to(device)

        # belief-state R2 probe (once per layer)
        try:
            Wp = fit_probe(Xtr, torch.from_numpy(belief[tr]).to(device), use_bias=True)
        except (RuntimeError, NotImplementedError):
            Wp = fit_probe(Xtr.cpu(), torch.from_numpy(belief[tr]), use_bias=True).to(device)
        r2 = _r2(belief[te], predict_probe(Xte, Wp, use_bias=True).cpu().numpy())

        variants = {}
        with torch.no_grad():   # logit lens — untrained; self-target = model NTP
            variants["logit"] = (F.softmax(_readout(Xte, norm, Wc, cap, norm_dtype), -1).cpu().numpy(), final)
        for name, tgt in targets.items():
            tgt_tr = torch.from_numpy(np.ascontiguousarray(tgt[tr])).to(device)
            Tr = _train_translator(Xtr, tgt_tr, norm, Wc, cap, norm_dtype, d, device,
                                   args.tl_epochs, args.tl_lr, args.tl_batch)
            with torch.no_grad():
                probs = F.softmax(_readout(Tr(Xte), norm, Wc, cap, norm_dtype), -1).cpu().numpy()
            variants[name] = (probs, tgt)
            del Tr; torch.cuda.empty_cache()

        for name, (probs, tgt) in variants.items():
            rows.append(dict(
                hmm=cfg["_name"], param=label, layer=l, lens=name,
                kl_self=float(_kl(tgt[te], probs).mean()),   # KL to this lens's own target
                kl_hmm=float(_kl(oracle[te], probs).mean()), # KL to the real HMM oracle
                kl_final=float(_kl(final[te], probs).mean()),# KL to the model's own NTP
                r2_belief=r2,
            ))
        del Xtr, Xte; torch.cuda.empty_cache()
    del acts_pool; gc.collect(); torch.cuda.empty_cache()
    return rows

def main():
    P = argparse.ArgumentParser(description="Per-layer tuned lens over HMM data")
    P.add_argument("--model", default="Qwen/Qwen3.5-9B")
    P.add_argument("--output_dir", default="results")
    P.add_argument("--families", nargs="+", default=["Mess3", "Arch", "Wing", "Strata"])
    P.add_argument("--all_params", action="store_true")
    P.add_argument("--param_index", type=int, default=-1, help="select one param per family")
    P.add_argument("--seq_len", type=int, default=20000)
    P.add_argument("--n_seeds", type=int, default=10)
    P.add_argument("--probe_start", type=int, default=15000,
                   help="use the last (seq_len - probe_start) positions; matches run_r2")
    P.add_argument("--train_frac", type=float, default=0.2,
                   help="20/80 split shared by translator training and the belief probe")
    P.add_argument("--controls", nargs="*", default=["shuffle", "random"],
                   choices=["shuffle", "random", "order1", "cross"],
                   help="control targets to also fit a lens to (mirror the probe controls)")
    P.add_argument("--layers", type=int, nargs="+", default=None)
    P.add_argument("--chunk_size", type=int, default=2048)
    P.add_argument("--tl_epochs", type=int, default=20)
    P.add_argument("--tl_lr", type=float, default=1e-5)
    P.add_argument("--tl_batch", type=int, default=512)
    P.add_argument("--device", default="cuda")
    P.add_argument("--smoke", action="store_true", help="tiny local plumbing test")
    args = P.parse_args()

    if args.smoke:
        args.seq_len = 400; args.n_seeds = 4; args.probe_start = 100
        args.tl_epochs = 3; args.families = args.families[:1]

    os.makedirs(args.output_dir, exist_ok=True)
    wrapper, tokenizer = load_model(args.model, args.device)
    device = next(wrapper.model.parameters()).device
    if device.type == "cpu":
        wrapper.model.float()
    for p in wrapper.model.parameters():
        p.requires_grad_(False)               # only translators are trained
    norm = _final_norm(wrapper)
    norm_dtype = next(norm.parameters()).dtype if any(True for _ in norm.parameters()) else \
                 next(wrapper.model.parameters()).dtype
    lm_W = wrapper.model.lm_head.weight
    cap = None
    if wrapper.family == "gemma":
        tc = getattr(wrapper.model.config, "text_config", wrapper.model.config)
        cap = getattr(tc, "final_logit_softcapping", None)
    layers = args.layers or list(range(wrapper.n_layers))
    if args.smoke:
        layers = [wrapper.n_layers // 3, 2 * wrapper.n_layers // 3]
    ms = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "")

    def params_for(fam):
        c = HMMS[fam]
        if args.param_index >= 0:
            return [c["params"][args.param_index]] if args.param_index < len(c["params"]) else []
        return c["params"] if args.all_params else [REPRESENTATIVES[fam]]

    total = sum(len(params_for(f)) for f in args.families if f in HMMS)
    pbar = tqdm(total=total, desc="tuned-lens (hmm,param)")
    all_rows = []
    for fam in args.families:
        cfg = HMMS.get(fam)
        if not cfg:
            continue
        tok_ids = get_tok_ids(tokenizer, cfg["token_names"])
        Wc = lm_W[tok_ids].detach()                          # (C, d) concept unembed
        for param in params_for(fam):
            cfg = {**cfg, "_name": fam}
            pbar.set_postfix_str(f"{fam} {cfg['label_fn'](param)}")
            all_rows += run_param(wrapper, tokenizer, cfg, param, layers, tok_ids,
                                  Wc, norm, cap, norm_dtype, args, device)
            pd.DataFrame(all_rows).to_csv(
                os.path.join(args.output_dir, f"tunedlens_{ms}.csv"), index=False)
            pbar.update(1)
    pbar.close()
    print("Done.")


if __name__ == "__main__":
    main()
