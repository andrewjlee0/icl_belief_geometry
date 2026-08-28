"""Probe-protocol comparison on dumped best-layer activations (see run_r2_dump.py).

Protocols per (model, hmm, param) at the published best layer:
  rand20    : random 20% train / 80% test within each seed's window (published protocol,
              sanity anchor; train_test_split(random_state=seed)).
  contig1000: first 1,000 positions train, rest test (the reviewer-facing contiguous split).
  contig500 / contig2500 (Arch only): contiguous-train-size sweep -- if the contig1000
              deficit is train-coverage, it shrinks as the block grows.
  crossseq  : probe fit on seed 0's FULL window, evaluated on seeds 1--9's windows.
              Different sequences entirely -- no within-sequence leakage channel exists.

All fits float32 pinv with bias (the published regularization). Outputs long-form CSV:
  model,hmm,param,protocol,seed,R2   (for crossseq, seed = the held-out test seed)
Plus, for Arch a=0.98/a=0.99: per-position diagnostics binned by belief-space distance
to the contiguous train block (extrapolation signature):
  model,param,seed,decile,mindist,frac_unexplained
"""
import argparse, glob, os, re, sys
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split


def fit_w(X, Y):
    Xa = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], 1)
    return torch.linalg.pinv(Xa) @ Y


def pred_w(X, W):
    return torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], 1) @ W


def r2_of(Y, P):
    ss_res = float(((Y - P) ** 2).sum())
    ss_tot = float(((Y - Y.mean(0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


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
        by_pp = {}
        for fn in sorted(glob.glob(os.path.join(mdir, "acts_*.npz"))):
            m = re.match(r"acts_(.+?)__(.+?)__s(\d+)\.npz", os.path.basename(fn))
            if not m: continue
            by_pp.setdefault((m.group(1), m.group(2)), {})[int(m.group(3))] = fn
        for (hmm, label), seeds in sorted(by_pp.items()):
            data = {}
            for s, fn in seeds.items():
                z = np.load(fn)
                data[s] = (torch.tensor(z["acts"], dtype=torch.float32, device=device),
                           torch.tensor(z["beliefs"], dtype=torch.float32, device=device))
            protos = {"contig500": 500, "contig1000": 1000, "contig2500": 2500} \
                if hmm == "Arch" else {"contig1000": 1000}
            for s, (X, Y) in data.items():
                n = X.shape[0]
                itr, ite = train_test_split(np.arange(n), train_size=0.2, random_state=s)
                W = fit_w(X[itr], Y[itr])
                rows.append(dict(model=ms, hmm=hmm, param=label, protocol="rand20",
                                 seed=s, R2=r2_of(Y[ite], pred_w(X[ite], W))))
                for pname, ntr in protos.items():
                    W = fit_w(X[:ntr], Y[:ntr])
                    P = pred_w(X[ntr:], W)
                    rows.append(dict(model=ms, hmm=hmm, param=label, protocol=pname,
                                     seed=s, R2=r2_of(Y[ntr:], P)))
                    if pname == "contig1000" and hmm == "Arch" and label in ("a=0.98", "a=0.99"):
                        res = ((Y[ntr:] - P) ** 2).sum(1)
                        var = float(((Y[ntr:] - Y[ntr:].mean(0)) ** 2).sum(1).mean())
                        d2 = torch.cdist(Y[ntr:], Y[:ntr]).min(1).values
                        q = torch.quantile(d2, torch.linspace(0, 1, 11, device=device))
                        for i in range(10):
                            msk = (d2 >= q[i]) & (d2 <= q[i + 1])
                            if int(msk.sum()) == 0: continue
                            diag.append(dict(model=ms, param=label, seed=s, decile=i,
                                             mindist=float(d2[msk].mean()),
                                             frac_unexplained=float(res[msk].mean()) / var))
            if 0 in data and len(data) > 1:
                X0, Y0 = data[0]
                W = fit_w(X0, Y0)
                for s, (X, Y) in data.items():
                    if s == 0: continue
                    rows.append(dict(model=ms, hmm=hmm, param=label, protocol="crossseq",
                                     seed=s, R2=r2_of(Y, pred_w(X, W))))
            del data
            if device == "cuda": torch.cuda.empty_cache()
            print(f"{ms} {hmm} [{label}] done", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "r2_protocols.csv"), index=False)
        pd.DataFrame(diag).to_csv(os.path.join(args.out_dir, "arch_extrap_diag.csv"), index=False)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
