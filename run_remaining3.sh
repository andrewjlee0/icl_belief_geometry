#!/bin/bash
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
cd /workspace/icl_belief_geometry
R=results; L=results/logs; mkdir -p $L

echo "$(date) Starting Gemma KL"
CUDA_VISIBLE_DEVICES=0 python experiments/belief_probes/run_kl.py --model google/gemma-4-E4B --output_dir $R > $L/run_kl_gemma_4_e4b.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python experiments/belief_probes/run_kl.py --model google/gemma-4-E2B --output_dir $R > $L/run_kl_gemma_4_e2b.log 2>&1 &
wait
echo "$(date) Gemma KL done"

echo "$(date) Starting between R²"
CUDA_VISIBLE_DEVICES=0 python experiments/belief_probes/run_between.py --model Qwen/Qwen3.5-9B --output_dir $R > $L/run_between_qwen35_9b.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python experiments/belief_probes/run_between.py --model Qwen/Qwen3.5-4B --output_dir $R > $L/run_between_qwen35_4b.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python experiments/belief_probes/run_between.py --model meta-llama/Llama-3.1-8B --output_dir $R > $L/run_between_llama_31_8b.log 2>&1 &
wait
echo "$(date) Between batch 1 done"

CUDA_VISIBLE_DEVICES=0 python experiments/belief_probes/run_between.py --model meta-llama/Llama-3.2-3B --output_dir $R > $L/run_between_llama_32_3b.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python experiments/belief_probes/run_between.py --model google/gemma-4-E4B --output_dir $R > $L/run_between_gemma_4_e4b.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python experiments/belief_probes/run_between.py --model google/gemma-4-E2B --output_dir $R > $L/run_between_gemma_4_e2b.log 2>&1 &
wait
echo "$(date) Between batch 2 done"

echo "$(date) ALL DONE"
