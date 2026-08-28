"""Probe-subspace stability across sequences, on dumped best-layer activations.

Per (model, hmm, param) and cyclic pair (seed s -> s+1 mod 10), all with matched
training size n_half (=min half-window across the pair):
  within : probe on random half A of seed s vs probe on disjoint half B of seed s
           (the noise ceiling for subspace agreement at this fit quality).
  cross  : probe on half A of seed s vs probe on half A of seed t.
Metrics: principal-angle mean_cos2, min_cos between the two probes' column spaces
(bias row dropped, effective-rank orthonormal bases, tol 1e-6 -- as run_subspace_sim).

Output: model,hmm,param,train_seed,test_seed,kind,mean_cos2,min_cos,r_a,r_b
"""
import argparse, glob, os, re
import numpy as np, pandas as pd, torch


def fit_w(X, Y):
    Xa = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], 1)
    return (torch.linalg.pinv(Xa) @ Y)[:-1]          # drop bias row -> (d, m)


def basis(W, tol=1e-6):
    U, S, _ = torch.linalg.svd(W, full_matrices=False)
    r = int((S > S.max() * tol).sum()) if S.numel() and float(S.max()) > 0 else 0
    return U[:, :r], r


def angles(Wa, Wb):
    Qa, ra = basis(Wa); Qb, rb = basis(Wb)
    if ra == 0 or rb == 0: return np.nan, np.nan, ra, rb
    c = torch.clamp(torch.linalg.svdvals(Qa.T @ Qb), 0, 1)
    return float((c ** 2).mean()), float(c.min()), ra, rb


def main():
    A = argparse.ArgumentParser()
    A.add_argument("--dump_dir", required=True)
    A.add_argument("--out_csv", required=True)
    A.add_argument("--device", default="cuda")
    args = A.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    rows = []
    for mdir in sorted(glob.glob(os.path.join(args.dump_dir, "*"))):
        ms = os.path.basename(mdir)
        by_pp = {}
        for fn in sorted(glob.glob(os.path.join(mdir, "acts_*.npz"))):
            m = re.match(r"acts_(.+?)__(.+?)__s(\d+)\.npz", os.path.basename(fn))
            if m: by_pp.setdefault((m.group(1), m.group(2)), {})[int(m.group(3))] = fn
        for (hmm, label), seeds in sorted(by_pp.items()):
            if len(seeds) < 10: continue
            data = {}
            for s, fn in seeds.items():
                z = np.load(fn)
                data[s] = (torch.tensor(z["acts"], dtype=torch.float32, device=device),
                           torch.tensor(z["beliefs"], dtype=torch.float32, device=device))
            for s in range(10):
                t = (s + 1) % 10
                Xs, Ys = data[s]; Xt, Yt = data[t]
                n_half = min(Xs.shape[0], Xt.shape[0]) // 2
                rng = np.random.default_rng(1000 + s)
                perm_s = rng.permutation(Xs.shape[0])
                A_s, B_s = perm_s[:n_half], perm_s[n_half:2 * n_half]
                A_t = np.random.default_rng(2000 + t).permutation(Xt.shape[0])[:n_half]
                W_sA = fit_w(Xs[A_s], Ys[A_s])
                W_sB = fit_w(Xs[B_s], Ys[B_s])
                W_tA = fit_w(Xt[A_t], Yt[A_t])
                for kind, Wa, Wb in [("within", W_sA, W_sB), ("cross", W_sA, W_tA)]:
                    mc2, mn, ra, rb = angles(Wa, Wb)
                    rows.append(dict(model=ms, hmm=hmm, param=label, train_seed=s,
                                     test_seed=t, kind=kind, mean_cos2=mc2,
                                     min_cos=mn, r_a=ra, r_b=rb))
            del data
            if device == "cuda": torch.cuda.empty_cache()
            print(f"{ms} {hmm} [{label}] done", flush=True)
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
