"""Gap-protected contiguous train-size sweep on dumped best-layer activations.

Test set is FIXED at the last 2,000 window positions. Train blocks are the first
500 / 1,000 / 2,500 positions, so every block ends >= 500 positions before the test
set begins (well beyond the ~100-token correlation scale of the slowest process).
Growing the block therefore adds coverage without adding temporal adjacency.
Also recomputes the belief-distance decile diagnostic on the gapped test set for
Arch a=0.98 / a=0.99 (any remaining gradient cannot be temporal adjacency).

Coverage: all Arch params + the three other representative params, all models.
Outputs: gapsweep.csv (model,hmm,param,n_train,seed,R2) and
         gapsweep_diag.csv (model,param,seed,decile,mindist,frac_unexplained)
"""
import argparse, glob, os, re
import numpy as np, pandas as pd, torch

KEEP = lambda hmm, label: hmm == "Arch" or (hmm, label) in {
    ("Mess3", "a=0.01, x=0.02"), ("Wing", "a=0.98, x=0.4"),
    ("Strata", "a=0.97, t0=0.38, t1=0.54")}
N_TEST = 2000
N_TRAINS = [500, 1000, 2500]


def fit_w(X, Y):
    Xa = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], 1)
    return torch.linalg.pinv(Xa) @ Y


def pred_w(X, W):
    return torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], 1) @ W


def main():
    A = argparse.ArgumentParser()
    A.add_argument("--dump_dir", required=True)
    A.add_argument("--out_dir", required=True)
    A.add_argument("--device", default="cuda")
    args = A.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    rows, diag = [], []
    for mdir in sorted(glob.glob(os.path.join(args.dump_dir, "*"))):
        ms = os.path.basename(mdir)
        for fn in sorted(glob.glob(os.path.join(mdir, "acts_*.npz"))):
            m = re.match(r"acts_(.+?)__(.+?)__s(\d+)\.npz", os.path.basename(fn))
            if not m or not KEEP(m.group(1), m.group(2)): continue
            hmm, label, seed = m.group(1), m.group(2), int(m.group(3))
            z = np.load(fn)
            X = torch.tensor(z["acts"], dtype=torch.float32, device=device)
            Y = torch.tensor(z["beliefs"], dtype=torch.float32, device=device)
            n = X.shape[0]
            ite = np.arange(n - N_TEST, n)
            Y_te = Y[ite]
            ss_tot = float(((Y_te - Y_te.mean(0)) ** 2).sum())
            for ntr in N_TRAINS:
                if ntr + 500 > n - N_TEST: continue
                W = fit_w(X[:ntr], Y[:ntr])
                P = pred_w(X[ite], W)
                r2 = 1.0 - float(((Y_te - P) ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
                rows.append(dict(model=ms, hmm=hmm, param=label, n_train=ntr,
                                 seed=seed, R2=r2))
                if ntr == 1000 and hmm == "Arch" and label in ("a=0.98", "a=0.99"):
                    res = ((Y_te - P) ** 2).sum(1)
                    var = float(((Y_te - Y_te.mean(0)) ** 2).sum(1).mean())
                    d2 = torch.cdist(Y_te, Y[:ntr]).min(1).values
                    q = torch.quantile(d2, torch.linspace(0, 1, 11, device=device))
                    for i in range(10):
                        msk = (d2 >= q[i]) & (d2 <= q[i + 1])
                        if int(msk.sum()) == 0: continue
                        diag.append(dict(model=ms, param=label, seed=seed, decile=i,
                                         mindist=float(d2[msk].mean()),
                                         frac_unexplained=float(res[msk].mean()) / var))
            del X, Y
            if device == "cuda": torch.cuda.empty_cache()
        print(f"{ms} done", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "gapsweep.csv"), index=False)
        pd.DataFrame(diag).to_csv(os.path.join(args.out_dir, "gapsweep_diag.csv"), index=False)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
