"""Train per-layer tuned-lens translators over HMM data and SAVE THE WEIGHTS.

Stripped-down sibling of run_tuned_lens.py whose only job is to persist the affine
translators A_l (weight + bias) so they can be analysed offline:
  - is A_l close to the final-layer translator A_L (identity-up-to-LayerNorm)?
  - does A_l, viewed in belief coordinates, implement the emission map M?

Only the tuned_hmm lens is trained (the one whose target is the HMM oracle NTP),
and by default only the main-text parametrization per family -- that's all the
weight-space analyses need. Use --all_params / --lenses to broaden.

Outputs:
  translators_{model}.npz  with keys  {hmm}__{param}__L{layer}__{lens}__W  (d,d) fp16
                                       {hmm}__{param}__L{layer}__{lens}__b  (d,)  fp16
  and a sidecar translators_{model}_meta.csv listing what was saved.

Usage:
  python run_tunedlens_weights.py --model Qwen/Qwen3.5-9B
  python run_tunedlens_weights.py --model Qwen/Qwen3.5-9B --families Arch --all_params
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


def _final_norm(wrapper):
    m = wrapper.model
    cands = ([m.model.language_model] if wrapper.family == "gemma" else [m.model])
    for base in cands + [m.model]:
        for attr in ("norm", "final_layernorm", "ln_f"):
            if hasattr(base, attr):
                return getattr(base, attr)
    raise AttributeError("Could not locate the final norm; adjust _final_norm().")


def _split(n, train_frac, seed=42):
    perm = np.random.default_rng(seed).permutation(n)
    n_tr = int(round(n * train_frac))
    return perm[:n_tr], perm[n_tr:]


def _readout(h, norm, Wc, cap, norm_dtype):
    normed = norm(h.to(norm_dtype)).float()
    logits = normed @ Wc.float().T
    if cap is not None:
        logits = torch.tanh(logits / cap) * cap
    return logits


def _train_translator(X_tr, target_tr, norm, Wc, cap, norm_dtype, d, device,
                      epochs, lr, batch):
    """Affine translator (init identity) trained to minimise KL(target || lens)."""
    Tlin = torch.nn.Linear(d, d, bias=True).to(device)
    with torch.no_grad():
        Tlin.weight.copy_(torch.eye(d, device=device)); Tlin.bias.zero_()
    opt = torch.optim.Adam(Tlin.parameters(), lr=lr)
    n = X_tr.shape[0]
    logt = torch.log(target_tr.clamp_min(1e-12))
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            logits = _readout(Tlin(X_tr[idx]), norm, Wc, cap, norm_dtype)
            logq = F.log_softmax(logits, dim=-1)
            loss = (target_tr[idx] * (logt[idx] - logq)).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return Tlin


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--model", default="Qwen/Qwen3.5-9B")
    P.add_argument("--output_dir", default="results")
    P.add_argument("--families", nargs="+", default=["Mess3", "Arch", "Wing", "Strata"])
    P.add_argument("--all_params", action="store_true")
    P.add_argument("--lenses", nargs="+", default=["tuned_hmm"],
                   choices=["tuned_hmm", "tuned_concept"],
                   help="which translator targets to train and save")
    P.add_argument("--seq_len", type=int, default=20000)
    P.add_argument("--n_seeds", type=int, default=10)
    P.add_argument("--probe_start", type=int, default=15000)
    P.add_argument("--train_frac", type=float, default=0.2)
    P.add_argument("--layers", type=int, nargs="+", default=None)
    P.add_argument("--chunk_size", type=int, default=2048)
    P.add_argument("--tl_epochs", type=int, default=20)
    P.add_argument("--tl_lr", type=float, default=1e-5)
    P.add_argument("--tl_batch", type=int, default=512)
    P.add_argument("--save_dtype", default="float16", choices=["float16", "float32"])
    P.add_argument("--device", default="cuda")
    args = P.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    wrapper, tokenizer = load_model(args.model, args.device)
    device = next(wrapper.model.parameters()).device
    if device.type == "cpu":
        wrapper.model.float()
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
    layers = args.layers or list(range(wrapper.n_layers))
    d = wrapper.hidden_size
    ms = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "")
    save_np_dtype = np.float16 if args.save_dtype == "float16" else np.float32

    def params_for(fam):
        c = HMMS[fam]
        return c["params"] if args.all_params else [REPRESENTATIVES[fam]]

    store, meta = {}, []
    n_units = sum(len(params_for(f)) for f in args.families if f in HMMS) * len(layers) * len(args.lenses)
    pbar = tqdm(total=n_units, desc="train+save translators")

    for fam in args.families:
        cfg = HMMS.get(fam)
        if not cfg:
            continue
        tok_ids = get_tok_ids(tokenizer, cfg["token_names"])
        Wc = lm_W[tok_ids].detach()

        for param in params_for(fam):
            label = cfg["label_fn"](param)
            T_matrices = cfg["fn"](*param); T_stack = np.stack(T_matrices)
            pi = stationary_distribution(T_matrices); M = emission_matrix(T_matrices)
            ps = args.probe_start

            # gather activations + oracle NTP over the late window
            acts_pool = {l: [] for l in layers}
            oracle_pool = []
            for seed in range(args.n_seeds):
                tokens = sample_hmm_sequence(T_matrices, pi, args.seq_len, seed=seed).astype(np.int64)
                beliefs = full_bayesian_beliefs(tokens, T_stack, pi)
                prompt = tokens_to_prompt(tokens, cfg["token_names"])
                input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
                pos, _ = match_positions(input_ids, tok_ids)
                n = min(len(tokens), len(pos)); pos = pos[:n]
                if n <= ps:
                    continue
                acts, _ = extract_activations_chunked(wrapper, input_ids, layers,
                                                      pos[ps:n], args.chunk_size, device)
                for l in layers:
                    acts_pool[l].append(acts[l].detach().cpu())
                oracle_pool.append((beliefs[ps:n] @ M).astype(np.float32))
                del acts; torch.cuda.empty_cache()

            for l in layers:
                acts_pool[l] = torch.cat(acts_pool[l], dim=0)
            oracle = np.concatenate(oracle_pool, 0)
            N, C = oracle.shape
            tr, te = _split(N, args.train_frac)

            # model's own final concept distribution (only needed for tuned_concept)
            final = None
            if "tuned_concept" in args.lenses:
                with torch.no_grad():
                    h = acts_pool[layers[-1]].to(device)
                    final = F.softmax(_readout(h, norm, Wc, cap, norm_dtype), -1).cpu().numpy()
                    del h; torch.cuda.empty_cache()

            targets = {}
            if "tuned_hmm" in args.lenses:
                targets["tuned_hmm"] = oracle
            if "tuned_concept" in args.lenses:
                targets["tuned_concept"] = final

            for l in layers:
                Xtr = acts_pool[l][tr].to(device)
                for name, tgt in targets.items():
                    tgt_tr = torch.from_numpy(np.ascontiguousarray(tgt[tr])).to(device)
                    Tlin = _train_translator(Xtr, tgt_tr, norm, Wc, cap, norm_dtype, d,
                                             device, args.tl_epochs, args.tl_lr, args.tl_batch)
                    key = f"{fam}__{label}__L{l}__{name}"
                    store[f"{key}__W"] = Tlin.weight.detach().cpu().numpy().astype(save_np_dtype)
                    store[f"{key}__b"] = Tlin.bias.detach().cpu().numpy().astype(save_np_dtype)
                    meta.append(dict(hmm=fam, param=label, layer=l, lens=name,
                                     d=d, C=C, n_train=len(tr), n_test=len(te)))
                    del Tlin, tgt_tr; torch.cuda.empty_cache()
                    pbar.update(1)
                del Xtr; torch.cuda.empty_cache()

            # save the emission matrix + token ids alongside, for the M-recovery analysis
            store[f"{fam}__{label}__M"] = M.astype(np.float32)
            store[f"{fam}__{label}__tok_ids"] = np.asarray(tok_ids, dtype=np.int64)

            del acts_pool; gc.collect(); torch.cuda.empty_cache()
            # checkpoint after each param
            np.savez_compressed(os.path.join(args.output_dir, f"translators_{ms}.npz"), **store)
            pd.DataFrame(meta).to_csv(
                os.path.join(args.output_dir, f"translators_{ms}_meta.csv"), index=False)

    pbar.close()
    np.savez_compressed(os.path.join(args.output_dir, f"translators_{ms}.npz"), **store)
    pd.DataFrame(meta).to_csv(
        os.path.join(args.output_dir, f"translators_{ms}_meta.csv"), index=False)
    print(f"Done. saved {len(meta)} translators -> translators_{ms}.npz")


if __name__ == "__main__":
    main()
