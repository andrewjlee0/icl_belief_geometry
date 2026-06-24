"""Probe-subspace similarity: how aligned are the learned probes for belief / ntp / log_ntp?

GATING TEST for the reciprocal destruction experiment (steer ntp/log_ntp -> decode belief).
If the ntp probe's subspace is *contained in* the belief probe's subspace, then steering ntp
is a move inside the belief subspace and "did belief change?" can't come out "no" -> the
reciprocal test is vacuous. (Empirically belief|ntp cos2 overlap = 1.0, so this matters.)

For each (model, hmm, param, seed, layer) it fits the three probes on the last-k CLEAN
positions and, for all three pairings {belief|ntp, belief|log_ntp, ntp|log_ntp}, reports
every reasonable similarity between the two learned probes -- SEPARATELY for the encoder
READ subspace (act -> X) and the decoder WRITE subspace (X -> act):

  principal angles  -> mean_cos2 (overlap), mean_cos, max_cos, min_cos, chordal_dist,
                       contain_a_in_b, contain_b_in_a, proj_frob, mat_cosine
  effective ranks of each probe (erank_a / erank_b columns)
  representation     -> cka_linear (linear CKA between the two probes' outputs, role="output")

No steering, no KV passes -> ONE chunked forward per sequence. Cheap.
Output: subspace_sim_{model}.csv  (long form; resumes on (hmm,param,seed)).
"""
import argparse, gc, os, sys
import numpy as np, pandas as pd, torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs.hmm_configs import HMMS, REPRESENTATIVES
from src.hmm import (stationary_distribution, sample_hmm_sequence,
                     full_bayesian_beliefs, emission_matrix, next_token_probs)
from src.model_utils import (load_model, tokens_to_prompt, match_positions,
                             get_tok_ids, extract_activations_chunked)

EPS = 1e-12
QUANTS = ["belief", "ntp", "log_ntp"]
PAIRS = [("belief", "ntp"), ("belief", "log_ntp"), ("ntp", "log_ntp")]


def _vals(q, beliefs, M):
    b = np.asarray(beliefs, np.float64)
    if q == "belief":
        return b
    ntp = b @ M
    return ntp if q == "ntp" else np.log(ntp + EPS)


def _fit_W(A, Y):
    """Least-squares W with Y ~= A_c @ W (bias removed by centering). A:(n,d) Y:(n,m) -> (d,m)."""
    Ac = A - A.mean(0, keepdim=True); Yc = Y - Y.mean(0, keepdim=True)
    try:
        return torch.linalg.pinv(Ac) @ Yc
    except (RuntimeError, NotImplementedError):
        return (torch.linalg.pinv(Ac.cpu()) @ Yc.cpu()).to(A.device)


def _ortho_basis(W, tol=1e-6):
    """Orthonormal basis (d x r) for the column space of W (d x m); r = effective rank."""
    U, S, _ = torch.linalg.svd(W, full_matrices=False)
    r = int((S > S.max() * tol).sum()) if S.numel() and float(S.max()) > 0 else 0
    return U[:, :r], r


def _angles(Qa, Qb):
    """Cosines of the principal angles between subspaces with orthonormal bases Qa, Qb."""
    if Qa.shape[1] == 0 or Qb.shape[1] == 0:
        return torch.zeros(0)
    return torch.clamp(torch.linalg.svdvals(Qa.T @ Qb), 0.0, 1.0)


def _pair_metrics(Qa, Qb, ra, rb):
    c = _angles(Qa, Qb)                            # cos(principal angles), len = min(ra, rb)
    if c.numel() == 0:
        return {}
    c2 = c ** 2; da, db = max(ra, 1), max(rb, 1)
    return {
        "mean_cos2": float(c2.mean()),             # average overlap over min-dim (== decode_shift's)
        "mean_cos": float(c.mean()),
        "max_cos": float(c.max()),                 # best-aligned direction (smallest principal angle)
        "min_cos": float(c.min()),                 # worst-aligned (largest principal angle)
        "chordal_dist": float(torch.sqrt((1 - c2).clamp(min=0).sum())),
        "contain_a_in_b": float(c2.sum() / da),    # fraction of A captured by B (asymmetric)
        "contain_b_in_a": float(c2.sum() / db),
        "proj_frob": float(c2.sum()),              # ||P_A P_B||_F^2 = sum cos^2
        "mat_cosine": float(c2.sum() / np.sqrt(da * db)),   # projection-matrix cosine similarity
    }


def _linear_cka(X, Y):
    """Linear CKA between two (centered) representations X:(n,p), Y:(n,q)."""
    Xc = X - X.mean(0, keepdim=True); Yc = Y - Y.mean(0, keepdim=True)
    xy = float((Xc.T @ Yc).norm() ** 2)
    xx = float((Xc.T @ Xc).norm()); yy = float((Yc.T @ Yc).norm())
    return xy / (xx * yy) if xx > 0 and yy > 0 else float("nan")


def run_sequence(wrapper, tokenizer, cfg, beliefs, tokens, seed, layers, n_fit, tok_ids, M,
                 args, device, pbar):
    prompt = tokens_to_prompt(tokens, cfg["token_names"])
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False)
    pos_indices, _ = match_positions(input_ids, tok_ids)
    n_matched = min(len(tokens), len(pos_indices))
    if n_matched <= n_fit + 10:
        return []
    idx = np.arange(n_matched - n_fit, n_matched)          # last-k HMM-token indices
    abs_pos = pos_indices[idx]
    pbar.set_postfix_str(f"{cfg['_name']} s{seed} | clean pass", refresh=True)
    acts, _ = extract_activations_chunked(wrapper, input_ids, layers, abs_pos, args.chunk_size, device)
    vals = {q: torch.tensor(_vals(q, beliefs[idx], M), device=device, dtype=torch.float32)
            for q in QUANTS}

    rows = []
    base = dict(hmm=cfg["_name"], param=cfg["_label"], seed=seed)
    for l in layers:
        X = acts.get(l)
        if X is None or X.numel() == 0:
            continue
        A = X.to(device).float()
        read, write, erank, preds = {}, {}, {}, {}
        for q in QUANTS:
            Wr = _fit_W(A, vals[q])                        # read:  act -> X   (d, m)
            Vw = _fit_W(vals[q], A)                        # write: X -> act   (m, d)
            Qr, rr = _ortho_basis(Wr); Qw, rw = _ortho_basis(Vw.T)
            read[q] = (Qr, rr); write[q] = (Qw, rw); erank[q] = (rr, rw)
            preds[q] = A @ Wr                              # probe output (n, m) for CKA
        for a, b in PAIRS:
            for role, bd in (("read", read), ("write", write)):
                Qa, ra = bd[a]; Qb, rb = bd[b]
                ridx = 0 if role == "read" else 1
                for name, val in _pair_metrics(Qa, Qb, ra, rb).items():
                    rows.append({**base, "layer": l, "pair": f"{a}|{b}", "role": role,
                                 "metric": name, "value": val,
                                 "erank_a": erank[a][ridx], "erank_b": erank[b][ridx],
                                 "dim_a": vals[a].shape[1], "dim_b": vals[b].shape[1]})
            rows.append({**base, "layer": l, "pair": f"{a}|{b}", "role": "output",
                         "metric": "cka_linear", "value": _linear_cka(preds[a], preds[b]),
                         "erank_a": erank[a][0], "erank_b": erank[b][0],
                         "dim_a": vals[a].shape[1], "dim_b": vals[b].shape[1]})
        del A
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del acts; gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    pbar.update(1)
    return rows


def main():
    P = argparse.ArgumentParser(description="Probe-subspace similarity (gating test for the reciprocal experiment)")
    P.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    P.add_argument("--output_dir", default="results")
    P.add_argument("--families", nargs="+", default=["Mess3", "Arch", "Wing", "Strata", "Spiral"])
    P.add_argument("--all_params", action="store_true")
    P.add_argument("--seq_len", type=int, default=20000)
    P.add_argument("--n_fit", type=int, default=5000, help="last-k clean positions used to fit the probes")
    P.add_argument("--n_seeds", type=int, default=5)
    P.add_argument("--layers", type=int, nargs="+", default=None, help="layers (default: all)")
    P.add_argument("--chunk_size", type=int, default=2048)
    P.add_argument("--device", default="cuda")
    P.add_argument("--smoke", action="store_true")
    args = P.parse_args()

    if args.smoke:
        args.seq_len = 4000; args.n_fit = 1000; args.n_seeds = 2
        if not args.all_params:
            args.families = args.families[:2]

    os.makedirs(args.output_dir, exist_ok=True)
    wrapper, tokenizer = load_model(args.model, args.device)
    device = next(wrapper.model.parameters()).device
    if device.type == "cpu":
        wrapper.model.float()
    layers = args.layers or list(range(wrapper.n_layers))
    if args.smoke:
        layers = [wrapper.n_layers // 3, 2 * wrapper.n_layers // 3]
    ms = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "")

    combos = sum((len(HMMS[f]["params"]) if args.all_params else 1)
                 for f in args.families if f in HMMS)
    pbar = tqdm(total=combos * args.n_seeds, desc="subspace_sim", smoothing=0.05)
    out_csv = os.path.join(args.output_dir, f"subspace_sim_{ms}.csv")
    all_rows, done = [], set()
    if os.path.exists(out_csv):                            # RESUME on (hmm,param,seed)
        prev = pd.read_csv(out_csv)
        all_rows = prev.to_dict("records")
        done = set(zip(prev["hmm"], prev["param"], prev["seed"]))
        tqdm.write(f"resume: {len(done)} (hmm,param,seed) already done -> skipping")

    for hmm_name in args.families:
        cfg0 = HMMS.get(hmm_name)
        if not cfg0:
            continue
        tok_ids = get_tok_ids(tokenizer, cfg0["token_names"])
        params = cfg0["params"] if args.all_params else [REPRESENTATIVES[hmm_name]]
        for param in params:
            label = cfg0["label_fn"](param)
            cfg = {**cfg0, "_name": hmm_name, "_label": label, "_param": param}
            T_matrices = cfg["fn"](*param); T_stack = np.stack(T_matrices)
            pi = stationary_distribution(T_matrices); M = emission_matrix(T_matrices)
            sv = np.linalg.svd(M, compute_uv=False)
            mrank = int((sv > sv.max() * 1e-8).sum())
            mcond = float(sv.max() / sv[sv > 0].min()) if (sv > 0).any() else float("inf")
            tqdm.write(f"===== {hmm_name} {label} | M rank {mrank}/{M.shape[0]} cond {mcond:.2f} =====")
            for seed in range(args.n_seeds):
                if (hmm_name, label, seed) in done:
                    pbar.update(1); continue
                tk = sample_hmm_sequence(T_matrices, pi, args.seq_len, seed=seed).astype(np.int64)
                beliefs = full_bayesian_beliefs(tk, T_stack, pi)
                rows = run_sequence(wrapper, tokenizer, cfg, beliefs, tk, seed, layers,
                                    args.n_fit, tok_ids, M, args, device, pbar)
                for r in rows:
                    r["M_rank"] = mrank; r["M_cond"] = mcond
                all_rows += rows
                pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    pbar.close()
    print("Done.")


if __name__ == "__main__":
    main()
