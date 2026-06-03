#!/bin/bash
# Run the intervention experiment across all 40 HMMs on ONE model (default
# Llama-3.2-3B, the paper's intervention model), work-stealing the four HMM
# families across every available GPU. Each family runs all 10 parametrizations
# x n_seeds sequences; the per-family CSVs are merged into one at the end.
#
# Usage:
#   bash run_interventions_hmms.sh
#   GPUS="0 1 2 3" bash run_interventions_hmms.sh
#   MODEL=Qwen/Qwen3.5-9B bash run_interventions_hmms.sh        # interventions on another model
#   EXTRA_ARGS="--n_eval 3" bash run_interventions_hmms.sh      # tighter error bands
#   DRY_RUN=1 GPUS="0 1 2 3" bash run_interventions_hmms.sh
#
# Env: PROJ_DIR (defaults to this script's dir), MODEL, FAMILIES, RESULTS_DIR,
#      LOG_DIR, HF_HOME, HF_TOKEN, EXTRA_ARGS, GPUS, DRY_RUN.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="${PROJ_DIR:-$SCRIPT_DIR}"
MODEL="${MODEL:-meta-llama/Llama-3.2-3B}"
FAMILIES="${FAMILIES:-Mess3 Arch Wing Strata}"
RESULTS_DIR="${RESULTS_DIR:-${PROJ_DIR}/results}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"
PARTS="${RESULTS_DIR}/_parts"
EXTRA_ARGS="${EXTRA_ARGS:-}"
QUEUE="/tmp/intv_hmm_queue.$$"
LOCK="/tmp/intv_hmm_lock.$$"
rm -rf "$PARTS"; mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$PARTS"

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
[ -z "${HF_TOKEN:-}" ] && echo "WARNING: HF_TOKEN unset; gated models will fail." >&2

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
ms=$(echo "${MODEL##*/}" | tr '[:upper:]' '[:lower:]' | tr '-' '_' | tr -d '.')

GPU_LIST=()
if [ -n "${GPUS:-}" ]; then read -r -a GPU_LIST <<< "$GPUS"
elif command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t GPU_LIST < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
fi
[ "${#GPU_LIST[@]}" -eq 0 ] && GPU_LIST=(0)

printf "%s\n" $FAMILIES > "$QUEUE"          # one family per line (word-split intended)
TOTAL=$(wc -l < "$QUEUE")
MASTER="${LOG_DIR}/intervene_hmms_master.log"
echo "$(timestamp) model=$MODEL | $TOTAL families (~$((TOTAL*10)) HMMs) across ${#GPU_LIST[@]} GPU(s): ${GPU_LIST[*]}" | tee "$MASTER"

gpu_worker() {
    local gpu=$1 fam logfile status
    export CUDA_VISIBLE_DEVICES=$gpu
    while true; do
        fam=$(flock "$LOCK" bash -c 'head -1 "'"$QUEUE"'" && sed -i "1d" "'"$QUEUE"'"')
        [ -z "$fam" ] && break
        logfile="${LOG_DIR}/intervene_${ms}_${fam}.log"
        mkdir -p "${PARTS}/${fam}"
        echo "$(timestamp) [GPU $gpu] START ${fam} (all params)" | tee -a "$MASTER"
        if [ "${DRY_RUN:-0}" = "1" ]; then
            sleep $((RANDOM % 3 + 1)); echo "(dry) ${fam} on GPU $gpu" > "$logfile"; status=0
        else
            python "${PROJ_DIR}/experiments/causal_interventions/run_intervene.py" \
                --model "$MODEL" --families "$fam" --all_params \
                --output_dir "${PARTS}/${fam}" $EXTRA_ARGS > "$logfile" 2>&1
            status=$?
        fi
        if [ "$status" -eq 0 ]; then
            echo "$(timestamp) [GPU $gpu] DONE  ${fam}" | tee -a "$MASTER"
        else
            echo "$(timestamp) [GPU $gpu] FAIL  ${fam} (exit=$status)" | tee -a "$MASTER"
            tail -8 "$logfile" >> "$MASTER"
        fi
    done
    echo "$(timestamp) [GPU $gpu] no more families." | tee -a "$MASTER"
}

cd "$PROJ_DIR" 2>/dev/null || true
for g in "${GPU_LIST[@]}"; do gpu_worker "$g" & done
wait

# ── merge per-family CSVs into one ──
echo "$(timestamp) merging per-family CSVs ..." | tee -a "$MASTER"
python3 - "$PARTS" "$ms" "$RESULTS_DIR" << 'PYEOF' | tee -a "$MASTER"
import sys, glob, os, pandas as pd
parts, ms, out = sys.argv[1], sys.argv[2], sys.argv[3]
files = sorted(glob.glob(os.path.join(parts, "*", f"intervene_{ms}.csv")))
if not files:
    print("  no part files found — nothing merged"); raise SystemExit
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
dst = os.path.join(out, f"intervene_{ms}.csv")
df.to_csv(dst, index=False)
print(f"  merged {len(files)} families -> {dst}")
print(f"  {len(df)} rows | {df.groupby(['hmm','param']).ngroups} HMMs | families: {sorted(df.hmm.unique())}")
PYEOF

echo "$(timestamp) ALL FINISHED. passed=$(grep -c '] DONE ' "$MASTER") failed=$(grep -c '] FAIL ' "$MASTER")" | tee -a "$MASTER"
rm -f "$QUEUE" "$LOCK"
