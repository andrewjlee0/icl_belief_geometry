"""Modal launcher for run_decode_destroy.py (retrain/destruction test) on Qwen3.5-9B.

One A100-80GB container per family (full 20k context, steered, all layers). Reuses
the hf-cache volume warmed by the decode_shift smoke. Writes per-family CSVs + a
combined decode_destroy_qwen35_9b.csv into the repo's results/.

  smoke:  MODAL_PROFILE=riechers-shai-c10-andrew modal run .../run_decode_destroy_modal.py
  full :  MODAL_PROFILE=riechers-shai-c10-andrew modal run .../run_decode_destroy_modal.py --mode full
"""
import gzip, io, os, pathlib, subprocess
import modal

REPO = "/Users/andrewjunlee/Library/CloudStorage/OneDrive-UCLAITServices/UCLA/Projects/SPAR"
MODEL = "Qwen/Qwen3.5-9B"
MS = "qwen35_9b"
FAMILIES = ["Mess3", "Arch", "Wing", "Strata"]
FULL_ARGS = ["--k", "5000", "--n_train", "1000", "--chunk_size", "2048"]   # all layers = script default

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "accelerate", "huggingface_hub",
                 "numpy", "pandas", "tqdm", "numba", "scikit-learn", "sentencepiece")
    .add_local_dir(f"{REPO}/src", "/spar/src", ignore=["__pycache__"], copy=True)
    .add_local_dir(f"{REPO}/configs", "/spar/configs", ignore=["__pycache__"], copy=True)
    .add_local_dir(f"{REPO}/experiments", "/spar/experiments", ignore=["__pycache__"], copy=True)
)
app = modal.App("spar-decode-destroy", image=image)
hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("spar-decode-destroy-out", create_if_missing=True)
SCRIPT = "/spar/experiments/causal_interventions/run_decode_destroy.py"


@app.function(gpu="A100-80GB", volumes={"/cache": hf_vol, "/out": out_vol}, timeout=14400)
def run_family(family: str, smoke: bool = False):
    env = dict(os.environ, HF_HOME="/cache/hf")
    outdir = f"/out/{'smoke' if smoke else 'full'}/{family}"
    cmd = ["python", SCRIPT, "--model", MODEL, "--families", family,
           "--device", "cuda", "--output_dir", outdir]
    cmd += ["--smoke"] if smoke else FULL_ARGS
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)
    hf_vol.commit(); out_vol.commit()
    return family, gzip.compress(pathlib.Path(f"{outdir}/decode_destroy_{MS}.csv").read_bytes())


@app.local_entrypoint()
def main(mode: str = "smoke"):
    import pandas as pd
    out = pathlib.Path(REPO) / "results"; out.mkdir(exist_ok=True)
    if mode == "smoke":
        fam, data = run_family.remote("Mess3", smoke=True)
        p = out / f"decode_destroy_{MS}_SMOKE_{fam}.csv.gz"; p.write_bytes(data)
        print(f"SMOKE OK -> {p} ({len(data)} bytes gz)")
    elif mode == "full":
        frames = []
        for fam, data in run_family.starmap([(f, False) for f in FAMILIES]):
            p = out / f"decode_destroy_{MS}_{fam}.csv.gz"; p.write_bytes(data)
            frames.append(pd.read_csv(io.BytesIO(data), compression="gzip"))
            print(f"DONE {fam} -> {p} ({len(data)} bytes gz)")
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(out / f"decode_destroy_{MS}.csv", index=False)
        print(f"COMBINED -> {out / f'decode_destroy_{MS}.csv'} ({len(combined)} rows)")
    else:
        raise SystemExit("mode must be 'smoke' or 'full'")
