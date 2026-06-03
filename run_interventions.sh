#!/bin/bash
# ALL models x ALL 40 HMMs. Work-steals (model, family) jobs across every GPU so
# all GPUs stay busy and a failure costs one family, not a whole model. Per-family
# parts are merged into one intervene_<model>.csv per model at the end.
#
# 6 models x 4 families = 24 jobs. On 4 GPUs at ~1h/family that's ~6 waves (~6h).
#
# Usage:
#   bash run_interventions_all.sh
#   GPUS="0 1 2 3" bash run_interventions_all.sh
#   MODELS="meta-llama/Llama-3.2-3B Qwen/Qwen3.5-9B" bash run_interventions_all.sh
#   EXTRA_ARGS="--n_eval 3" bash run_interventions_all.sh
#   DRY_RUN=1 GPUS="0 1 2 3" bash run_interventions_all.sh

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="${PROJ_DIR:-$SCRIPT_DIR}"
MODELS="${MODELS:-Qwen/Qwen3.5-9B Qwen/Qwen3.5-4B meta-llama/Llama-3.1-8B meta-llama/Llama-3.2-3B google/gemma-4-E4B google/gemma-4-E2B}"
FAMILIES="${FAMILIES:-Mess3 Arch Wing Strata}"
RESULTS_DIR="${RESULTS_DIR:-${PROJ_DIR}/results}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"
PARTS="${RESULTS_DIR}/_parts"
EXTRA_ARGS="${EXTRA_ARGS:-}"
QUEUE="/tmp/intv_all_queue.$$"
LOCK="/tmp/intv_all_lock.$$"
rm -rf "$PARTS"; mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$PARTS"

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
[ -z "${HF_TOKEN:-}" ] && echo "WARNING: HF_TOKEN unset; gated models will fail." >&2

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
model_short() { local n="${1##*/}"; echo "$n" | tr '[:upper:]' '[:lower:]' | tr '-' '_' | tr -d '.'; }
MASTER="${LOG_DIR}/intervene_all_master.log"

GPU_LIST=()
if [ -n "${GPUS:-}" ]; then read -r -a GPU_LIST <<< "$GPUS"
elif command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t GPU_LIST < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
fi
[ "${#GPU_LIST[@]}" -eq 0 ] && GPU_LIST=(0)

# queue: one "model|family" job per line
: > "$QUEUE"
for m in $MODELS; do for f in $FAMILIES; do echo "${m}|${f}" >> "$QUEUE"; done; done
TOTAL=$(wc -l < "$QUEUE")
echo "$(timestamp) $TOTAL (model,family) jobs across ${#GPU_LIST[@]} GPU(s): ${GPU_LIST[*]}" | tee "$MASTER"

gpu_worker() {
    local gpu=$1 job model family ms out logfile status
    export CUDA_VISIBLE_DEVICES=$gpu
    while true; do
        job=$(flock "$LOCK" bash -c 'head -1 "'"$QUEUE"'" && sed -i "1d" "'"$QUEUE"'"')
        [ -z "$job" ] && break
        model="${job%|*}"; family="${job#*|}"; ms=$(model_short "$model")
        out="${PARTS}/${ms}/${family}"; mkdir -p "$out"
        logfile="${LOG_DIR}/intervene_${ms}_${family}.log"
        echo "$(timestamp) [GPU $gpu] START ${ms} / ${family}" | tee -a "$MASTER"
        if [ "${DRY_RUN:-0}" = "1" ]; then
            sleep $((RANDOM % 3 + 1)); echo "(dry) ${ms}/${family} on GPU $gpu" > "$logfile"; status=0
        else
            python "${PROJ_DIR}/experiments/causal_interventions/run_intervene.py" \
                --model "$model" --families "$family" --all_params \
                --output_dir "$out" $EXTRA_ARGS > "$logfile" 2>&1
            status=$?
        fi
        if [ "$status" -eq 0 ]; then
            echo "$(timestamp) [GPU $gpu] DONE  ${ms} / ${family}" | tee -a "$MASTER"
        else
            echo "$(timestamp) [GPU $gpu] FAIL  ${ms} / ${family} (exit=$status)" | tee -a "$MASTER"
            tail -8 "$logfile" >> "$MASTER"
        fi
    done
    echo "$(timestamp) [GPU $gpu] no more jobs." | tee -a "$MASTER"
}

cd "$PROJ_DIR" 2>/dev/null || true
for g in "${GPU_LIST[@]}"; do gpu_worker "$g" & done
wait

# ── merge per-family parts into one CSV per model ──
echo "$(timestamp) merging parts per model ..." | tee -a "$MASTER"
for m in $MODELS; do
    ms=$(model_short "$m")
    python3 - "${PARTS}/${ms}" "$ms" "$RESULTS_DIR" << 'PYEOF' | tee -a "$MASTER"
import sys, glob, os, pandas as pd
parts, ms, out = sys.argv[1], sys.argv[2], sys.argv[3]
files = sorted(glob.glob(os.path.join(parts, "*", f"intervene_{ms}.csv")))
if not files:
    print(f"  {ms}: no parts (all families failed?)"); raise SystemExit
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
dst = os.path.join(out, f"intervene_{ms}.csv")
df.to_csv(dst, index=False)
print(f"  {ms}: {len(files)} families -> {dst} | {df.groupby(['hmm','param']).ngroups} HMMs, {len(df)} rows")
PYEOF
done

echo "$(timestamp) ALL FINISHED. passed=$(grep -c '] DONE ' "$MASTER") failed=$(grep -c '] FAIL ' "$MASTER")" | tee -a "$MASTER"
rm -f "$QUEUE" "$LOCK"