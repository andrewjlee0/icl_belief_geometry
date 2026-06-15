"""Causal belief-subspace ablation under the tuned lens.

Question: does the tuned lens's prediction DEPEND on the belief subspace
specifically, or would removing any few directions hurt equally?

For each (family, param, layer, seed):
  1. fit belief probe P_seed (act -> beliefs) on the seed's TRAIN positions
  2. build the orthogonal projector onto rowspace(P_seed):
        Prow = P^T (P P^T + eps I)^{-1} P
  3. ablate the belief subspace from TEST activations (train-mean centered):
        h_abl = h - (h - mu) @ Prow^T
  4. train a tuned-lens translator on TRAIN activations (the standard lens),
     then evaluate its KL-to-HMM-oracle on TEST activations under:
        - intact            (no ablation)
        - belief-ablated    (belief subspace removed)
        - random-ablated    (a matched random m-dim subspace removed; k draws)
        - shuffled-ablated  (probe rows shuffled across features -> same row stats,
                              wrong directions)
  The probe, projectors, and random controls are all re-fit/re-drawn PER SEED,
  because each sequence yields a different belief subspace.

Independent-evidence signature:
  belief-ablated KL >> intact KL, while random-ablated KL ~ intact KL.

Output: lens_ablation_{model}.csv
  rows: hmm, param, layer, seed, condition, kl_hmm

Defaults: main-text params, belief-probe-best layers (pass --layers to override),
frozen-lens variant (train once on intact, evaluate on ablated). Use
--retrain_on_ablated to instead train the lens on ablated activations (stronger:
"is the prediction recoverable AT ALL without the belief subspace").
"""
import argparse, gc, os, sys
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from configs.hmm_configs import HMMS, REPRESENTATIVES
from src.hmm import (stationary_distribution, sample_hmm_sequence,
                     full_bayesian_beliefs, emission_matrix)
from src.model_utils import (load_model, tokens_to_prompt, match_positions,
                             get_tok_ids, extract_activations_chunked)

MAIN_TEXT = {"Mess3": "a=0.01, x=0.02", "Arch": "a=0.99",
             "Wing": "a=0.98, x=0.4", "Strata": "a=0.97, t0=0.38, t1=0.54"}


def _final_norm(wrapper):
    m = wrapper.model
    cands = ([m.model.language_model] if wrapper.family == "gemma" else [m.model])
    for base in cands + [m.model]:
        for attr in ("norm", "final_layernorm", "ln_f"):
            if hasattr(base, attr):
                return getattr(base, attr)
    raise AttributeError("Could not locate the final norm; adjust _final_norm().")


def _readout(h, norm, Wc, cap, norm_dtype):
    normed = norm(h.to(norm_dtype)).float()
    logits = normed @ Wc.float().T
    if cap is not None:
        logits = torch.tanh(logits / cap) * cap
    return logits


def _train_translator(X_tr, target_tr, norm, Wc, cap, norm_dtype, d, device, epochs, lr, batch):
    Tl = torch.nn.Linear(d, d, bias=True).to(device)
    with torch.no_grad():
        Tl.weight.copy_(torch.eye(d, device=device)); Tl.bias.zero_()
    opt = torch.optim.Adam(Tl.parameters(), lr=lr)
    n = X_tr.shape[0]; logt = torch.log(target_tr.clamp_min(1e-12))
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            logits = _readout(Tl(X_tr[idx]), norm, Wc, cap, norm_dtype)
            logq = F.log_softmax(logits, -1)
            loss = (target_tr[idx] * (logt[idx] - logq)).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return Tl


def _proj_rowspace(Pmat, eps=1e-4):
    """Orthogonal projector onto rowspace(P).  P: (m,d) -> (d,d)."""
    m = Pmat.shape[0]
    G = Pmat @ Pmat.T + eps * np.eye(m)
    return Pmat.T @ np.linalg.inv(G) @ Pmat           # (d,d)


def _kl(p, q):
    p = np.clip(p, 1e-12, None); q = np.clip(q, 1e-12, None)
    return (p * np.log(p / q)).sum(-1)


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--model", default="Qwen/Qwen3.5-9B")
    P.add_argument("--output_dir", default="results")
    P.add_argument("--families", nargs="+", default=["Mess3", "Arch", "Wing", "Strata"])
    P.add_argument("--layers", type=int, nargs="+", default=None,
                   help="layers to test (default: a spread; pass best layers for speed)")
    P.add_argument("--seq_len", type=int, default=20000)
    P.add_argument("--probe_start", type=int, default=15000)
    P.add_argument("--n_seeds", type=int, default=10)
    P.add_argument("--train_frac", type=float, default=0.2)
    P.add_argument("--n_random", type=int, default=5, help="random-subspace control draws")
    P.add_argument("--chunk_size", type=int, default=2048)
    P.add_argument("--tl_epochs", type=int, default=20)
    P.add_argument("--tl_lr", type=float, default=1e-5)
    P.add_argument("--tl_batch", type=int, default=512)
    P.add_argument("--retrain_on_ablated", action="store_true",
                   help="train the lens on ablated activations (stronger test)")
    P.add_argument("--device", default="cuda")
    args = P.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    wrapper, tokenizer = load_model(args.model, args.device)
    device = next(wrapper.model.parameters()).device
    for p in wrapper.model.parameters():
        p.requires_grad_(False)
    norm = _final_norm(wrapper)
    norm_dtype = next(norm.parameters()).dtype if any(True for _ in norm.parameters()) \
                 else next(wrapper.model.parameters()).dtype
    lm_W = wrapper.model.lm_head.weight
    cap = None
    if wrapper.family == "gemma":
        tc = getattr(wrapper.model.config, "text_config", wrapper.model.config)
        cap = getattr(tc, "final_logit_softcapping", None)
    all_layers = list(range(wrapper.n_layers))
    layers = args.layers or all_layers
    d = wrapper.hidden_size
    ms = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "")
    out_path = os.path.join(args.output_dir, f"lens_ablation_{ms}.csv")

    rows = []
    pbar = tqdm(total=len(args.families) * args.n_seeds * len(layers), desc="ablation")
    for fam in args.families:
        cfg = HMMS.get(fam)
        if not cfg or fam not in MAIN_TEXT:
            continue
        param = [p for p in cfg["params"] if cfg["label_fn"](p) == MAIN_TEXT[fam]]
        if not param:
            continue
        param = param[0]; label = cfg["label_fn"](param)
        tok_ids = get_tok_ids(tokenizer, cfg["token_names"])
        Wc = lm_W[tok_ids].detach()
        T = cfg["fn"](*param); T_stack = np.stack(T)
        pi = stationary_distribution(T); M = emission_matrix(T)
        ps = args.probe_start

        for seed in range(args.n_seeds):
            tokens = sample_hmm_sequence(T, pi, args.seq_len, seed=seed).astype(np.int64)
            beliefs = full_bayesian_beliefs(tokens, T_stack, pi)
            oracle = (beliefs @ M)                      # NTP target for the lens
            prompt = tokens_to_prompt(tokens, cfg["token_names"])
            input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
            pos, _ = match_positions(input_ids, tok_ids)
            n = min(len(tokens), len(pos)); pos = pos[:n]
            if n <= ps:
                pbar.update(len(layers)); continue
            acts, _ = extract_activations_chunked(wrapper, input_ids, layers, pos[ps:n],
                                                  args.chunk_size, device)
            y_bel = beliefs[ps:n]; y_orc = oracle[ps:n]
            n_late = len(y_bel)
            rng = np.random.default_rng(seed)
            idx = rng.permutation(n_late); ntr = int(args.train_frac * n_late)
            tr, te = idx[:ntr], idx[ntr:]

            for l in layers:
                X = acts[l].float().cpu().numpy()       # (n_late, d)
                Xtr, Xte = X[tr], X[te]
                mu = Xtr.mean(0, keepdims=True)

                # --- per-seed belief probe on TRAIN ---
                Xtr_b = np.hstack([Xtr, np.ones((len(Xtr), 1))])
                Wb, *_ = np.linalg.lstsq(Xtr_b, y_bel[tr], rcond=None)
                Pmat = Wb[:d].T                          # (m, d)
                m = Pmat.shape[0]
                Prow = _proj_rowspace(Pmat)              # (d, d)

                # --- random + shuffled control subspaces (per seed/layer) ---
                ctrl_projs = {}
                for r in range(args.n_random):
                    R = rng.standard_normal((m, d))
                    R, _ = np.linalg.qr(R.T); R = R[:, :m].T   # orthonormal m x d
                    ctrl_projs[f"random{r}"] = _proj_rowspace(R)
                Psh = Pmat.copy()
                for row in Psh:                          # shuffle features within each row
                    rng.shuffle(row)
                ctrl_projs["shuffled"] = _proj_rowspace(Psh)

                def ablate(Z, Pr):
                    return Z - (Z - mu) @ Pr.T

                # --- conditions: intact + belief + controls ---
                conds = {"intact": Xte, "belief": ablate(Xte, Prow)}
                for name, Pr in ctrl_projs.items():
                    conds[name] = ablate(Xte, Pr)

                # train the lens (frozen: on intact train; retrain: per condition)
                orc_tr = torch.tensor(y_orc[tr], device=device, dtype=torch.float32)
                if not args.retrain_on_ablated:
                    Xtr_t = torch.tensor(Xtr, device=device, dtype=torch.float32)
                    Tl = _train_translator(Xtr_t, orc_tr, norm, Wc, cap, norm_dtype, d,
                                           device, args.tl_epochs, args.tl_lr, args.tl_batch)
                    lenses = {c: Tl for c in conds}      # same lens, different inputs
                else:
                    lenses = {}
                    for c in conds:
                        Xtr_c = ablate(Xtr, Prow if c == "belief"
                                       else (np.eye(d) if c == "intact" else ctrl_projs[c]))
                        Xtr_ct = torch.tensor(Xtr_c, device=device, dtype=torch.float32)
                        lenses[c] = _train_translator(Xtr_ct, orc_tr, norm, Wc, cap, norm_dtype,
                                                      d, device, args.tl_epochs, args.tl_lr, args.tl_batch)

                for c, Z in conds.items():
                    Zt = torch.tensor(Z, device=device, dtype=torch.float32)
                    with torch.no_grad():
                        probs = F.softmax(_readout(lenses[c](Zt), norm, Wc, cap, norm_dtype), -1).cpu().numpy()
                    kl = _kl(y_orc[te], probs).mean()
                    # collapse the random draws into one label at analysis time
                    cond = "random" if c.startswith("random") else c
                    rows.append(dict(hmm=fam, param=label, layer=l, seed=seed,
                                     condition=cond, draw=c, kl_hmm=float(kl)))
                if args.retrain_on_ablated:
                    for Tl in lenses.values():
                        del Tl
                else:
                    del Tl
                torch.cuda.empty_cache()
            del acts; gc.collect(); torch.cuda.empty_cache()
            pbar.update(len(layers))
            pd.DataFrame(rows).to_csv(out_path, index=False)
    pbar.close()
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Done -> {out_path}")


if __name__ == "__main__":
    main()
