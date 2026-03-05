#!/usr/bin/env bash

set -euo pipefail

GPU_ID=6
SEED=42
COMPOSED_STRATEGY="with_carry_filtered"
WEAK_MODEL="Qwen/Qwen3-0.6B"
STRONG_MODELS=(
  "Qwen/Qwen3-0.6B"
  "Qwen/Qwen3-4B"
  "Qwen/Qwen3-8B"
)

BASE_OUTPUT_ROOT="results/qwen_w2s_scaling_filtered"
LOG_ROOT="${BASE_OUTPUT_ROOT}/logs"

mkdir -p "${BASE_OUTPUT_ROOT}"
mkdir -p "${LOG_ROOT}"

run_experiment() {
  local strong_model="$1"
  local model_slug="${strong_model##*/}"
  model_slug="${model_slug//./_}"
  local timestamp
  timestamp="$(date +%Y%m%d_%H%M%S)"
  local outdir="${BASE_OUTPUT_ROOT}/${model_slug}_${timestamp}"
  local log_path="${LOG_ROOT}/${model_slug}_${timestamp}.log"

  echo "=== Running strong model: ${strong_model} ==="
  echo "Output dir: ${outdir}"
  echo "Log file: ${log_path}"
  echo "-------------------------------------------"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    python weak_to_strong_addition_experiment_v2.py \
      --output-dir "${outdir}" \
      --composed-strategy "${COMPOSED_STRATEGY}" \
      --weak-model "${WEAK_MODEL}" \
      --strong-model "${strong_model}" \
      --seed "${SEED}" \
      --skip-strong-full \
      --skip-strong-w2s \
      --skip-strong-w2s-pseudo-direct \
      --skip-weak-w2s \
      --skip-weak-w2s-pseudo \
      > "${log_path}" 2>&1

  echo "Completed strong model: ${strong_model}"
  echo
}

for strong_model in "${STRONG_MODELS[@]}"; do
  run_experiment "${strong_model}"
done

echo "All scaling experiments finished."
