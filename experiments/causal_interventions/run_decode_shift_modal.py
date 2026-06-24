"""Modal launcher for run_decode_shift.py on Qwen3.5-9B.

One GPU container per HMM family (parallel via .starmap). Qwen weights are cached
in a persistent Volume so the smoke run warms the cache for the full run. Each
container writes decode_shift_qwen35_9b.csv to its own results subdir (no cross-
container file collisions), gzips it, and returns the bytes; the local entrypoint
writes them under the repo's results/.

  smoke (default):  modal run .../run_decode_shift_modal.py
  full (4 families): modal run .../run_decode_shift_modal.py --mode full

Scope (full): k=100 (intervene+decode the final 100 tokens at each L -> ~100 R²
points/seed), ALL layers intervened + read out, n_seeds=10, representative param
per family. Edit FULL_ARGS.
"""
import gzip, os, pathlib, subprocess
import modal

REPO = "/Users/andrewjunlee/Library/CloudStorage/OneDrive-UCLAITServices/UCLA/Projects/SPAR"
MODEL = "Qwen/Qwen3.5-9B"
MS = "qwen35_9b"
FAMILIES = ["Mess3", "Arch", "Wing", "Strata"]
FULL_ARGS = ["--k_values", "100", "--chunk_size", "2048"]   # all layers is the script default
                                                             # chunk_size: memory knob, not science

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "accelerate", "huggingface_hub",
                 "numpy", "pandas", "tqdm", "numba", "scikit-learn", "sentencepiece")
    .add_local_dir(f"{REPO}/src", "/spar/src", ignore=["__pycache__"], copy=True)
    .add_local_dir(f"{REPO}/configs", "/spar/configs", ignore=["__pycache__"], copy=True)
    .add_local_dir(f"{REPO}/experiments", "/spar/experiments", ignore=["__pycache__"], copy=True)
)

app = modal.App("spar-decode-shift", image=image)
hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("spar-decode-shift-out", create_if_missing=True)
SCRIPT = "/spar/experiments/causal_interventions/run_decode_shift.py"


@app.function(gpu="A10G", volumes={"/cache": hf_vol, "/out": out_vol}, timeout=10800)
def run_family(family: str, smoke: bool = False):
    env = dict(os.environ, HF_HOME="/cache/hf", HF_HUB_ENABLE_HF_TRANSFER="0")
    outdir = f"/out/{'smoke' if smoke else 'full'}/{family}"
    cmd = ["python", SCRIPT, "--model", MODEL, "--families", family,
           "--device", "cuda", "--output_dir", outdir]
    cmd += ["--smoke"] if smoke else FULL_ARGS
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)
    hf_vol.commit(); out_vol.commit()
    csv = f"{outdir}/decode_shift_{MS}.csv"
    return family, gzip.compress(pathlib.Path(csv).read_bytes())


@app.local_entrypoint()
def main(mode: str = "smoke"):
    out = pathlib.Path(REPO) / "results"; out.mkdir(exist_ok=True)
    if mode == "smoke":
        fam, data = run_family.remote("Mess3", smoke=True)
        p = out / f"decode_shift_{MS}_SMOKE_{fam}.csv.gz"
        p.write_bytes(data)
        print(f"SMOKE OK -> {p} ({len(data)} bytes gz)")
    elif mode == "full":
        import io, pandas as pd
        frames = []
        for fam, data in run_family.starmap([(f, False) for f in FAMILIES]):
            p = out / f"decode_shift_{MS}_{fam}.csv.gz"
            p.write_bytes(data)
            frames.append(pd.read_csv(io.BytesIO(data), compression="gzip"))
            print(f"DONE {fam} -> {p} ({len(data)} bytes gz)")
        # combined CSV (the name plot_decode_shift.py reads), plus a .gz copy
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(out / f"decode_shift_{MS}.csv", index=False)
        combined.to_csv(out / f"decode_shift_{MS}.csv.gz", index=False, compression="gzip")
        print(f"COMBINED -> {out / f'decode_shift_{MS}.csv'} ({len(combined)} rows)")
    else:
        raise SystemExit("mode must be 'smoke' or 'full'")
