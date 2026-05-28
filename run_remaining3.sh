#!/bin/bash
PROJ_DIR="/workspace/icl_belief_geometry"
LOG_DIR="${PROJ_DIR}/results/logs"
RESULTS_DIR="${PROJ_DIR}/results"
QUEUE="/tmp/job_queue3"
LOCK="/tmp/job_lock3"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

model_short() {
    local name="$1"
    name="${name##*/}"
    name=$(echo "$name" | tr '[:upper:]' '[:lower:]' | tr '-' '_' | tr -d '.')
    echo "$name"
}

cat > "$QUEUE" << 'JOBS'
google/gemma-4-E4B|run_kl
google/gemma-4-E2B|run_kl
Qwen/Qwen3.5-9B|run_between
Qwen/Qwen3.5-4B|run_between
meta-llama/Llama-3.1-8B|run_between
meta-llama/Llama-3.2-3B|run_between
google/gemma-4-E4B|run_between
google/gemma-4-E2B|run_between
JOBS

TOTAL=$(wc -l < "$QUEUE")
echo "$(timestamp) $TOTAL jobs queued across 3 GPUs" | tee "${LOG_DIR}/master3.log"

gpu_worker() {
    local gpu=$1
    export CUDA_VISIBLE_DEVICES=$gpu
    while true; do
        local job
        job=$(flock "$LOCK" bash -c 'head -1 '"$QUEUE"' && sed -i "1d" '"$QUEUE")
        [ -z "$job" ] && break

        local model="${job%%|*}"
        local experiment="${job##*|}"
        local ms=$(model_short "$model")
        local logfile="${LOG_DIR}/${experiment}_${ms}.log"

        echo "$(timestamp) [GPU $gpu] START ${experiment} ${model}" | tee -a "${LOG_DIR}/master3.log"

        python "${PROJ_DIR}/experiments/belief_probes/${experiment}.py" \
            --model "$model" \
            --output_dir "$RESULTS_DIR" \
            > "$logfile" 2>&1

        local status=$?
        if [ $status -eq 0 ]; then
            echo "$(timestamp) [GPU $gpu] DONE  ${experiment} ${model}" | tee -a "${LOG_DIR}/master3.log"
        else
            echo "$(timestamp) [GPU $gpu] FAIL  ${experiment} ${model} (exit=$status)" | tee -a "${LOG_DIR}/master3.log"
            tail -5 "$logfile" >> "${LOG_DIR}/master3.log"
        fi
    done
    echo "$(timestamp) [GPU $gpu] No more jobs." | tee -a "${LOG_DIR}/master3.log"
}

cd "$PROJ_DIR"
gpu_worker 0 &
gpu_worker 1 &
gpu_worker 2 &
wait

echo "" | tee -a "${LOG_DIR}/master3.log"
echo "═══════════════════════════════════════════════" | tee -a "${LOG_DIR}/master3.log"
echo "$(timestamp) ALL DONE. Summary:" | tee -a "${LOG_DIR}/master3.log"
grep -c "DONE" "${LOG_DIR}/master3.log" | xargs -I{} echo "  Passed: {}" | tee -a "${LOG_DIR}/master3.log"
grep -c "FAIL" "${LOG_DIR}/master3.log" | xargs -I{} echo "  Failed: {}" | tee -a "${LOG_DIR}/master3.log"
echo "  Logs: ${LOG_DIR}/" | tee -a "${LOG_DIR}/master3.log"
echo "═══════════════════════════════════════════════" | tee -a "${LOG_DIR}/master3.log"