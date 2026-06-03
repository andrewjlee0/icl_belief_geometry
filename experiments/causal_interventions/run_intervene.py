"""Belief-subspace interventions: activation patching + steering with controls.

Faithful port of the paper's Section 6 (Figures 6/52) into this repo's
HF-transformers + hook conventions. Reuses src.model_utils / src.hmm /
src.metrics.probes exactly like run_r2.py.

Outputs: intervene_{model}.csv   (long format: one row per condition/k/layer/draw)

──────────────────────────────────────────────────────────────────────────────
WHY THIS IS ~10x FASTER THAN THE TRANSFORMERLENS VERSION
──────────────────────────────────────────────────────────────────────────────
The original took hours almost entirely because of two things, not the method:

  (1) Every intervention forward ran the FULL post-convergence context
      (thousands of tokens). But the belief state converges within the HMM
      mixing time (tau <= 75 tokens; see paper App. B.2), and the paper's own negative result
      (Sec. 7: k-independence + past-consistent == past-inconsistent) shows the
      read-out is LOCAL to the current position. So a context window of ~1024
      tokens reproduces the belief at the measure position. Model-body compute
      is ~linear in context length and attention is ~quadratic, so trimming
      ~4200 -> ~1024 is the dominant win.

  (2) Logits were computed at every position (L x |vocab|, |vocab|~128k) when
      only the final position is read. We slice the last hidden state and apply
      lm_head at the measure position only, and restrict the softmax to the HMM
      tokens (the paper's "truncate to HMM tokens"), skipping the full-vocab
      logsumexp entirely.

Set --context_window 0 to recover full-context behaviour for validation; you
should confirm baseline KL matches between window=1024 and window=0 on one HMM.

ALIGNMENT CONVENTION (matches run_r2.py, NOT the partner repo):
  activation at the model position of HMM-token index i  -->  beliefs[i]
  (beliefs[i] is the posterior after observing tokens[0..i]); the NTP predicted
  at that position is next_token_probs(beliefs[i]). No +1 shift.
"""
import argparse, gc, os, sys
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs.hmm_configs import HMMS, REPRESENTATIVES
from src.hmm import (stationary_distribution, sample_hmm_sequence,
                     full_bayesian_beliefs, emission_matrix, next_token_probs)
from src.metrics.probes import fit_probe, predict_probe
from src.model_utils import (load_model, tokens_to_prompt, match_positions,
                             get_tok_ids, extract_activations_chunked)

PATCH_CONDS = ["past_consistent", "round_trip", "past_inconsistent", "random"]
STEER_CONDS = ["past_consistent", "past_inconsistent", "random"]


# ── small numeric helpers ─────────────────────────────────────────────────────
def _kl(p, q):
    """KL(p || q) over the HMM-token simplex. p,q: (..., n_tokens)."""
    p = np.clip(p, 1e-12, None); q = np.clip(q, 1e-12, None)
    return float((p * np.log(p / q)).sum(-1))


def _fit(X, Y, device):
    """fit_probe with a CPU fallback (torch.linalg.pinv is unsupported on MPS)."""
    try:
        return fit_probe(X, Y, use_bias=True).to(device)
    except (RuntimeError, NotImplementedError):
        return fit_probe(X.cpu(), Y.cpu(), use_bias=True).to(device)


def _continue_beliefs(start, toks, T_stack):
    """Run the Bayesian filter for |toks| steps from belief `start`.

    Used for the paper-exact past-inconsistent target: `start` is the belief
    after a fresh within-family prefix, `toks` are the source's REAL last-k
    suffix tokens. Returns the k running beliefs (k, n_states); the last one is
    the injected belief at the measure position N.
    """
    b = np.asarray(start, dtype=np.float64).copy()
    out = np.zeros((len(toks), b.shape[0]))
    for s, tok in enumerate(toks):
        b = b @ T_stack[int(tok)]
        z = b.sum()
        if z > 0:
            b = b / z
        out[s] = b
    return out


def _llm_ntp(last_hidden_row, lm_head_w, tok_ids, cap):
    """Truncate-to-HMM-tokens next-token distribution at the measure position.

    last_hidden_row: (B, d) final-position hidden states.
    Returns (B, n_tokens) renormalised over HMM tokens (paper Sec. 4 convention).
    """
    h = last_hidden_row.float().to(lm_head_w.device)
    with torch.no_grad():
        logits = h @ lm_head_w[tok_ids].float().T      # (B, n_tokens) only
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap     # element-wise; Gemma
        return torch.softmax(logits, dim=-1).cpu().numpy()


# ── intervention forward passes (HF hooks) ────────────────────────────────────
def _clean_window(wrapper, window_ids, layers, max_k, device):
    """One clean forward over the window.

    Returns (last_hidden_measure (1,d), {layer: clean_act_tail (max_k, d)}).
    clean_act_tail are the residual activations at the last max_k positions.
    """
    tail = {}
    def mk(li):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            tail[li] = h[0, -max_k:, :].detach().float()
        return fn
    handles = [wrapper.get_layer(l).register_forward_hook(mk(l)) for l in layers]
    with torch.no_grad():
        out = wrapper.forward(window_ids.to(device), use_cache=False)
    for hd in handles: hd.remove()
    return out.last_hidden_state[:, -1, :].detach(), tail


def _intervened_batch(wrapper, window_ids, layer, plan, W, device, dtype):
    """Run all variants for ONE layer in a single batched forward.

    plan: list of dicts {offsets: LongTensor, mode: 'patch'|'steer', value: (k,d)}.
    Row r of the batch is window_ids modified per plan[r] at `layer`.
    Returns last-position hidden states (B, d).
    """
    B = len(plan)
    batch_ids = window_ids.to(device).expand(B, -1).contiguous()
    offs = [p["offsets"].to(device) for p in plan]
    vals = [p["value"].to(device=device, dtype=dtype) for p in plan]
    modes = [p["mode"] for p in plan]

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out      # (B, W, d), in place
        for r in range(B):
            if modes[r] == "patch":
                h[r, offs[r], :] = vals[r]
            else:                                          # steer (add)
                h[r, offs[r], :] = h[r, offs[r], :] + vals[r]
        return None                                        # keep mutated tensor

    handle = wrapper.get_layer(layer).register_forward_hook(hook)
    with torch.no_grad():
        out = wrapper.forward(batch_ids, use_cache=False)
    handle.remove()
    return out.last_hidden_state[:, -1, :].detach()


# ── per-sequence driver ───────────────────────────────────────────────────────
def run_sequence(wrapper, tokenizer, cfg, T_matrices, pi, beliefs_all, seed,
                 layers, k_values, tok_ids, lm_head_w, cap, M, args, device, dtype,
                 pbar, n_planned_layers):
    """Fit encoder/decoder on full-context post-conv positions, then intervene
    on truncated windows. Returns a list of result rows."""
    n_states = cfg["n_states"]
    T_stack = np.stack(T_matrices)
    tokens = sample_hmm_sequence(T_matrices, pi, args.seq_len, seed=seed)
    beliefs = beliefs_all[seed]
    prompt = tokens_to_prompt(tokens, cfg["token_names"])
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
    pos_indices, _ = match_positions(input_ids, tok_ids)
    n_matched = min(len(tokens), len(pos_indices))
    if n_matched <= args.probe_start + 50:
        pbar.update(args.n_eval * n_planned_layers)
        return []

    # ── Fit encoder (act->belief) and decoder (belief->act) per layer ──
    # One chunked full-context pass (KV-cached); identical machinery to run_r2.
    late_idx = np.arange(args.probe_start, n_matched)           # HMM-token indices
    late_pos = pos_indices[late_idx]
    y_late = beliefs[late_idx]
    pbar.set_postfix_str(f"{cfg['_name']} s{seed} | fit enc/dec", refresh=True)
    acts, _ = extract_activations_chunked(wrapper, input_ids, layers, late_pos,
                                          args.chunk_size, device)
    idx_tr, idx_te = train_test_split(np.arange(len(late_idx)),
                                      train_size=args.train_frac, random_state=seed)
    y_tr = torch.tensor(y_late[idx_tr], device=device, dtype=torch.float32)
    enc, dec = {}, {}
    for l in layers:
        X = acts[l]
        if X.numel() == 0:
            continue
        Xtr = X[idx_tr].float()
        enc[l] = _fit(Xtr, y_tr, device)                        # (d+1, n_states)
        dec[l] = _fit(y_tr, Xtr, device)                        # (n_states+1, d)
    del acts; gc.collect(); torch.cuda.empty_cache()
    layers = [l for l in layers if l in dec]

    # ── Choose eval (measure) positions from the held-out split ──
    W = args.context_window if args.context_window > 0 else None
    eval_hmm_idx = late_idx[idx_te]
    min_idx = (W - 1) if W else max(k_values)                   # window/k must fit
    eval_hmm_idx = eval_hmm_idx[(pos_indices[eval_hmm_idx] >= (W - 1 if W else 0)) &
                                (eval_hmm_idx >= max(k_values))]
    rng = np.random.default_rng(seed + 999)
    if len(eval_hmm_idx) > args.n_eval:
        eval_hmm_idx = rng.choice(eval_hmm_idx, args.n_eval, replace=False)
    eval_hmm_idx = np.sort(eval_hmm_idx)

    # Cyclic 1:1 donor assignment for past-inconsistent prefixes: sequence s
    # borrows the next sequence's prefix, then the one after, etc. (mod n_seeds).
    # With --n_draws 1 this is exactly "each uses the next one's" (0->1, ..., 9->0).
    n_seeds = args.n_seeds
    donors = [(seed + j) % n_seeds for j in range(1, args.n_draws + 1)]
    donors = [d for d in donors if d != seed] or [seed]   # exclude self; degenerate fallback
    max_k = max(k_values)
    rows = []

    for ei, i in enumerate(eval_hmm_idx):
        pbar.set_postfix_str(f"{cfg['_name']} s{seed} | pos {ei + 1}/{len(eval_hmm_idx)}")
        mp = int(pos_indices[i])                                 # model measure pos
        w0 = (mp - W + 1) if W else 0
        window_ids = input_ids[:, w0:mp + 1]
        Wlen = window_ids.shape[1]
        last_hidden_c, clean_tail = _clean_window(wrapper, window_ids, layers,
                                                  max_k, device)
        p_clean = _llm_ntp(last_hidden_c, lm_head_w, tok_ids, cap)[0]
        p_factual = (beliefs[i] @ M)                             # NTP for true belief
        baseline_kl = _kl(p_factual, p_clean)
        rows.append(dict(hmm=cfg["_name"], param=cfg["_label"], seed=seed, layer=-1,
                         k=0, intervention="none", condition="unmodified", draw=0,
                         kl_to_target=baseline_kl, kl_to_factual=baseline_kl,
                         kl_to_pi_target=float("nan"), baseline_kl=baseline_kl))

        # Pre-sample condition target beliefs (shared across layers), per k.
        # The suffix tokens (source's real last k) are SHARED across conditions;
        # only the deep-history prefix differs (this is the whole design):
        #   past_consistent  : source's own true beliefs  = filter[real prefix . real suffix]
        #   past_inconsistent: filter[fresh within-family prefix . SAME real suffix],
        #                      i.e. continue the filter from a donor sequence's belief
        #                      at index (i-k) through the source's real last-k tokens.
        #   random           : i.i.d. Dirichlet simplex points (direction control)
        targets = {}     # (cond, k, draw) -> belief seq (k, n_states)
        for k in k_values:
            suffix_toks = tokens[i - k + 1:i + 1]                # real last-k tokens
            targets[("past_consistent", k, 0)] = beliefs[i - k + 1:i + 1]
            for d in range(args.n_draws):
                ds = donors[d % len(donors)]
                entry = beliefs_all[ds][i - k]                   # belief after fresh prefix
                targets[("past_inconsistent", k, d)] = _continue_beliefs(
                    entry, suffix_toks, T_stack)
            for d in range(args.n_draws):                        # random control
                targets[("random", k, d)] = rng.dirichlet(np.ones(n_states), size=k)

        for l in layers:
            We, Wd = enc[l], dec[l]
            ct = clean_tail[l]                                   # (max_k, d), fp32
            plan, meta = [], []                                  # parallel lists

            def add(mode, cond, k, draw, belief_seq, alt_ntp=None):
                bt = torch.tensor(np.asarray(belief_seq), device=device,
                                  dtype=torch.float32)
                dec_t = predict_probe(bt, Wd, use_bias=True)     # (k, d)
                if mode == "steer":
                    src = predict_probe(predict_probe(ct[-k:], We, use_bias=True),
                                        Wd, use_bias=True)       # dec(enc(h_clean))
                    value = dec_t - src
                else:
                    value = dec_t
                offsets = torch.arange(Wlen - k, Wlen, dtype=torch.long)
                plan.append(dict(offsets=offsets, mode=mode, value=value))
                # NTP implied by the injected belief at the MEASURE position:
                p_tgt = (np.asarray(belief_seq)[-1] @ M)
                meta.append((mode, cond, k, draw, p_tgt, alt_ntp))

            for k in k_values:
                # NTP of the matched past-inconsistent target, so the random control
                # can ALSO be scored against the principled "same target optimum".
                pi_ntp = {d: (targets[("past_inconsistent", k, d)][-1] @ M)
                          for d in range(args.n_draws)}
                # patching
                add("patch", "past_consistent", k, 0, targets[("past_consistent", k, 0)])
                rt = predict_probe(ct[-k:], We, use_bias=True).cpu().numpy()  # enc(h)
                add("patch", "round_trip", k, 0, rt)
                for d in range(args.n_draws):
                    add("patch", "past_inconsistent", k, d, targets[("past_inconsistent", k, d)])
                    add("patch", "random", k, d, targets[("random", k, d)], alt_ntp=pi_ntp[d])
                # steering
                add("steer", "past_consistent", k, 0, targets[("past_consistent", k, 0)])
                for d in range(args.n_draws):
                    add("steer", "past_inconsistent", k, d, targets[("past_inconsistent", k, d)])
                    add("steer", "random", k, d, targets[("random", k, d)], alt_ntp=pi_ntp[d])

            lh = _intervened_batch(wrapper, window_ids, l, plan, Wlen, device, dtype)
            p_llm = _llm_ntp(lh, lm_head_w, tok_ids, cap)        # (B, n_tokens)
            for r, (mode, cond, k, draw, p_tgt, p_alt) in enumerate(meta):
                rows.append(dict(hmm=cfg["_name"], param=cfg["_label"], seed=seed,
                                 layer=l, k=k, intervention=mode, condition=cond,
                                 draw=draw,
                                 kl_to_target=_kl(p_tgt, p_llm[r]),
                                 kl_to_factual=_kl(p_factual, p_llm[r]),
                                 kl_to_pi_target=(_kl(p_alt, p_llm[r])
                                                  if p_alt is not None else float("nan")),
                                 baseline_kl=baseline_kl))
            pbar.update(1)
        del clean_tail; torch.cuda.empty_cache()
    pbar.update(args.n_eval * n_planned_layers - len(eval_hmm_idx) * len(layers))
    return rows


def main():
    P = argparse.ArgumentParser(description="Belief-subspace patching + steering")
    P.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    P.add_argument("--output_dir", default="results")
    P.add_argument("--families", nargs="+", default=["Strata", "Wing"])
    P.add_argument("--all_params", action="store_true",
                   help="sweep all params (default: representative param only)")
    P.add_argument("--seq_len", type=int, default=20000)
    P.add_argument("--probe_start", type=int, default=15000,
                   help="enc/dec fit on the last (seq_len-probe_start) tokens; default last 5k")
    P.add_argument("--n_seeds", type=int, default=10)
    P.add_argument("--train_frac", type=float, default=0.2,
                   help="20/80 train/test -> ~1k train positions on the last-5k window")
    P.add_argument("--n_eval", type=int, default=8,
                   help="measure positions per sequence")
    P.add_argument("--context_window", type=int, default=1024,
                   help="tokens before+incl. the measure position; 0 = full context")
    P.add_argument("--k_values", type=int, nargs="+", default=[1, 5, 10])
    P.add_argument("--n_draws", type=int, default=1,
                   help="donor prefixes per source (1 = cyclic 1:1, seq s borrows s+1); "
                        "also sets random-control draws per position")
    P.add_argument("--layers", type=int, nargs="+", default=None)
    P.add_argument("--chunk_size", type=int, default=4096)
    P.add_argument("--device", default="cuda")
    P.add_argument("--smoke", action="store_true",
                   help="tiny local plumbing test (1 seed, 2 layers, short window)")
    args = P.parse_args()

    if args.smoke:                       # fast end-to-end check, NOT scientific output
        args.seq_len = min(args.seq_len, 800); args.probe_start = 200
        args.n_seeds = 1; args.n_eval = 2; args.n_draws = 1
        args.k_values = [1, 5]; args.context_window = 128
        if not args.all_params:
            args.families = args.families[:1]

    os.makedirs(args.output_dir, exist_ok=True)
    wrapper, tokenizer = load_model(args.model, args.device)
    # Derive the real device from the loaded model (handles device_map="auto",
    # CUDA / MPS / CPU). fp16 matmul is unsupported on CPU -> fall back to fp32.
    device = next(wrapper.model.parameters()).device
    if device.type == "cpu":
        wrapper.model.float()
    dtype = next(wrapper.model.parameters()).dtype
    layers = args.layers or list(range(wrapper.n_layers))
    if args.smoke:
        layers = [wrapper.n_layers // 3, 2 * wrapper.n_layers // 3]
    ms = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "")

    lm_head_w = wrapper.model.lm_head.weight
    cap = None
    if wrapper.family == "gemma":
        tc = getattr(wrapper.model.config, "text_config", wrapper.model.config)
        cap = getattr(tc, "final_logit_softcapping", None)

    n_planned = len(layers)
    combos = sum((len(HMMS[f]["params"]) if args.all_params else 1)
                 for f in args.families if f in HMMS)
    pbar = tqdm(total=combos * args.n_seeds * args.n_eval * n_planned,
                desc="interventions", smoothing=0.05)

    all_rows = []
    for hmm_name in args.families:
        cfg = HMMS.get(hmm_name)
        if not cfg:
            continue
        tok_ids = get_tok_ids(tokenizer, cfg["token_names"])
        params = cfg["params"] if args.all_params else [REPRESENTATIVES[hmm_name]]
        for param in params:
            label = cfg["label_fn"](param)
            cfg = {**cfg, "_name": hmm_name, "_label": label}
            T_matrices = cfg["fn"](*param); T_stack = np.stack(T_matrices)
            pi = stationary_distribution(T_matrices)
            M = emission_matrix(T_matrices)                      # (n_states, n_tokens)
            beliefs_all = {}                                     # donor pool (per seed)
            for s in range(args.n_seeds):
                tk = sample_hmm_sequence(T_matrices, pi, args.seq_len, seed=s)
                beliefs_all[s] = full_bayesian_beliefs(tk.astype(np.int64), T_stack, pi)
            tqdm.write(f"===== {hmm_name} {label} =====")
            for seed in range(args.n_seeds):
                all_rows += run_sequence(wrapper, tokenizer, cfg, T_matrices, pi,
                                         beliefs_all, seed, layers, args.k_values,
                                         tok_ids, lm_head_w, cap, M, args, device, dtype,
                                         pbar, n_planned)
                pd.DataFrame(all_rows).to_csv(
                    os.path.join(args.output_dir, f"intervene_{ms}.csv"), index=False)
    pbar.close()
    print("Done.")


if __name__ == "__main__":
    main()
