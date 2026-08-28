"""Held-out-sequence probes on dumped activations, cyclic pairing: seed s trains,
seed (s+1) mod 10 tests -- 10 train/test pairs per (model, hmm, param), no pooling.

Two train-size variants per pair:
  crosscyc_full: train on ALL of seed s's window (~4,900 positions).
  crosscyc_1k  : train on the published protocol's 1,000 training positions
                 (train_test_split(train_size=0.2, random_state=s)) of seed s --
                 the published probe, transported to a different sequence.
Test is always the FULL window of the held-out seed. Float32 pinv with bias.

Output: model,hmm,param,protocol,train_seed,test_seed,R2
"""
import argparse, glob, os, re
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split


def fit_w(X, Y):
    Xa = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], 1)
    return torch.linalg.pinv(Xa) @ Y


def pred_w(X, W):
    return torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], 1) @ W


def r2_of(Y, P):
    ss_res = float(((Y - P) ** 2).sum()); ss_tot = float(((Y - Y.mean(0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


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
                W = fit_w(Xs, Ys)
                rows.append(dict(model=ms, hmm=hmm, param=label, protocol="crosscyc_full",
                                 train_seed=s, test_seed=t, R2=r2_of(Yt, pred_w(Xt, W))))
                itr, _ = train_test_split(np.arange(Xs.shape[0]), train_size=0.2, random_state=s)
                W = fit_w(Xs[itr], Ys[itr])
                rows.append(dict(model=ms, hmm=hmm, param=label, protocol="crosscyc_1k",
                                 train_seed=s, test_seed=t, R2=r2_of(Yt, pred_w(Xt, W))))
            del data
            if device == "cuda": torch.cuda.empty_cache()
            print(f"{ms} {hmm} [{label}] done", flush=True)
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
