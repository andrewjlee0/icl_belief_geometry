#!/bin/bash
# Re-run all 6 KL (consistency after function change) + 2 failed redr2
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
cd /workspace/icl_belief_geometry
R=results; L=results/logs; mkdir -p $L

echo "$(date '+%Y-%m-%d %H:%M:%S') Starting KL batch (all 6 models)"

CUDA_VISIBLE_DEVICES=0 python experiments/belief_probes/run_kl.py --model Qwen/Qwen3.5-9B --output_dir $R > $L/run_kl_qwen35_9b.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python experiments/belief_probes/run_kl.py --model Qwen/Qwen3.5-4B --output_dir $R > $L/run_kl_qwen35_4b.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python experiments/belief_probes/run_kl.py --model meta-llama/Llama-3.1-8B --output_dir $R > $L/run_kl_llama_31_8b.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python experiments/belief_probes/run_kl.py --model meta-llama/Llama-3.2-3B --output_dir $R > $L/run_kl_llama_32_3b.log 2>&1 &
wait
echo "$(date '+%Y-%m-%d %H:%M:%S') KL batch 1 done"

CUDA_VISIBLE_DEVICES=0 python experiments/belief_probes/run_kl.py --model google/gemma-4-E4B --output_dir $R > $L/run_kl_gemma_4_e4b.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python experiments/belief_probes/run_kl.py --model google/gemma-4-E2B --output_dir $R > $L/run_kl_gemma_4_e2b.log 2>&1 &
wait
echo "$(date '+%Y-%m-%d %H:%M:%S') KL batch 2 done"

echo "$(date '+%Y-%m-%d %H:%M:%S') Starting redr2 (one at a time to avoid RAM OOM)"
CUDA_VISIBLE_DEVICES=0 python experiments/belief_probes/run_redr2.py --model meta-llama/Llama-3.1-8B --output_dir $R > $L/run_redr2_llama_31_8b.log 2>&1
echo "$(date '+%Y-%m-%d %H:%M:%S') redr2 Llama 3.1 8B done (exit=$?)"

CUDA_VISIBLE_DEVICES=0 python experiments/belief_probes/run_redr2.py --model meta-llama/Llama-3.2-3B --output_dir $R > $L/run_redr2_llama_32_3b.log 2>&1
echo "$(date '+%Y-%m-%d %H:%M:%S') redr2 Llama 3.2 3B done (exit=$?)"

echo "$(date '+%Y-%m-%d %H:%M:%S') ALL DONE"
