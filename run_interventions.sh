#!/bin/bash
# Run causal interventions (run_intervene.py) across all 6 models, work-stealing
# across every available GPU: each GPU pulls the next model from a shared queue
# the moment it finishes its current one. One model = one job (6 jobs total).
# Mirrors the run_remaining.sh pattern.
#
# Usage:
#   bash run_interventions.sh
#   GPUS="0 1 2 3" bash run_interventions.sh            # choose GPUs explicitly
#   EXTRA_ARGS="--all_params" bash run_interventions.sh # pass-through to run_intervene.py
#   DRY_RUN=1 GPUS="0 1 2 3" bash run_interventions.sh  # show scheduling, run nothing
#
# Env: PROJ_DIR, RESULTS_DIR, LOG_DIR, HF_HOME, HF_TOKEN (needed for gated models).

set -u
PROJ_DIR="${PROJ_DIR:-/workspace/icl_belief_geometry}"
RESULTS_DIR="${RESULTS_DIR:-${PROJ_DIR}/results}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
QUEUE="/tmp/intervene_queue.$$"
LOCK="/tmp/intervene_lock.$$"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
[ -z "${HF_TOKEN:-}" ] && echo "WARNING: HF_TOKEN unset; gated models (Llama/Gemma) will fail." >&2

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
model_short() { local n="${1##*/}"; echo "$n" | tr '[:upper:]' '[:lower:]' | tr '-' '_' | tr -d '.'; }

# ── GPUs: explicit $GPUS, else auto-detect via nvidia-smi, else GPU 0 ──
GPU_LIST=()
if [ -n "${GPUS:-}" ]; then
    read -r -a GPU_LIST <<< "$GPUS"
elif command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t GPU_LIST < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
fi
[ "${#GPU_LIST[@]}" -eq 0 ] && GPU_LIST=(0)

# ── Job queue: one model per line ──
cat > "$QUEUE" << 'JOBS'
Qwen/Qwen3.5-9B
Qwen/Qwen3.5-4B
meta-llama/Llama-3.1-8B
meta-llama/Llama-3.2-3B
google/gemma-4-E4B
google/gemma-4-E2B
JOBS

TOTAL=$(wc -l < "$QUEUE")
echo "$(timestamp) $TOTAL models queued across ${#GPU_LIST[@]} GPU(s): ${GPU_LIST[*]}" \
    | tee "${LOG_DIR}/intervene_master.log"

gpu_worker() {
    local gpu=$1 model ms logfile status
    export CUDA_VISIBLE_DEVICES=$gpu
    while true; do
        # atomically pop the first queued model
        model=$(flock "$LOCK" bash -c 'head -1 "'"$QUEUE"'" && sed -i "1d" "'"$QUEUE"'"')
        [ -z "$model" ] && break
        ms=$(model_short "$model")
        logfile="${LOG_DIR}/intervene_${ms}.log"
        echo "$(timestamp) [GPU $gpu] START ${model}" | tee -a "${LOG_DIR}/intervene_master.log"

        if [ "${DRY_RUN:-0}" = "1" ]; then
            sleep $((RANDOM % 3 + 1)); echo "(dry run) ${model} on GPU $gpu" > "$logfile"; status=0
        else
            python "${PROJ_DIR}/experiments/causal_interventions/run_intervene.py" \
                --model "$model" --output_dir "$RESULTS_DIR" $EXTRA_ARGS \
                > "$logfile" 2>&1
            status=$?
        fi

        if [ "$status" -eq 0 ]; then
            echo "$(timestamp) [GPU $gpu] DONE  ${model}" | tee -a "${LOG_DIR}/intervene_master.log"
        else
            echo "$(timestamp) [GPU $gpu] FAIL  ${model} (exit=$status)" | tee -a "${LOG_DIR}/intervene_master.log"
            tail -8 "$logfile" >> "${LOG_DIR}/intervene_master.log"
        fi
    done
    echo "$(timestamp) [GPU $gpu] no more jobs." | tee -a "${LOG_DIR}/intervene_master.log"
}

cd "$PROJ_DIR" 2>/dev/null || true
for g in "${GPU_LIST[@]}"; do gpu_worker "$g" & done
wait

{
    echo ""
    echo "$(timestamp) ALL FINISHED."
    echo "  passed: $(grep -c '] DONE ' "${LOG_DIR}/intervene_master.log")"
    echo "  failed: $(grep -c '] FAIL ' "${LOG_DIR}/intervene_master.log")"
    echo "  per-model logs: ${LOG_DIR}/intervene_<model>.log"
    echo "  results: ${RESULTS_DIR}/intervene_<model>.csv"
} | tee -a "${LOG_DIR}/intervene_master.log"
rm -f "$QUEUE" "$LOCK"
