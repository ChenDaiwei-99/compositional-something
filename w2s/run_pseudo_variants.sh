#!/usr/bin/env bash

# Run focused pseudo-label experiments for weak/strong variants with component and direct pseudo labeling.
# Uses GPU 0 for "WEAK" jobs and GPU 1 for "STRONG" jobs (override via GPU_WEAK/GPU_STRONG env vars).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_WEAK="${GPU_WEAK:-0}"
GPU_STRONG="${GPU_STRONG:-1}"
OUT_ROOT="$(realpath "${1:-${ROOT_DIR}/results/qwen_w2s_pseudo_runs}")"
LOG_DIR="${OUT_ROOT}/logs"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

run_job() {
  local gpu="$1"
  local name="$2"
  shift 2
  local out_dir="${OUT_ROOT}/${name}"
  local log_file="${LOG_DIR}/${name}.log"
  local wandb_dir="${out_dir}/wandb"
  mkdir -p "${out_dir}" "${wandb_dir}/cache" "${wandb_dir}/config"
  echo ">>> Launching ${name} on GPU ${gpu}, output -> ${out_dir}" >&2
  WANDB_DIR="${wandb_dir}" \
  WANDB_CACHE_DIR="${wandb_dir}/cache" \
  WANDB_CONFIG_DIR="${wandb_dir}/config" \
  WANDB_START_METHOD="${WANDB_START_METHOD_OVERRIDE:-thread}" \
  WANDB_CONSOLE="${WANDB_CONSOLE:-off}" \
  WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-w2s-direct-pseudo}" \
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${ROOT_DIR}/weak_to_strong_addition_experiment_v2.py" \
    --output-dir "${out_dir}" \
    --wandb-mode "${WANDB_MODE:-online}" \
    "$@" >"${log_file}" 2>&1 &
  local pid=$!
  echo "${pid}"
}

wait_for_process() {
  local pid="$1"
  local name="$2"
  echo ">>> Waiting for ${name} (PID: ${pid}) to complete..."
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 5
  done
  echo ">>> ${name} completed."
}

# 1. Weak_W2S_Pseudo (component cache, no carry) on GPU 0.
weak_w2s_pseudo_nc_pid=$(run_job "${GPU_WEAK}" "weak_w2s_pseudo_without_carry" \
  --composed-strategy without_carry \
  --skip-weak \
  --skip-strong-full \
  --skip-strong-w2s \
  --skip-strong-w2s-pseudo \
  --skip-strong-w2s-pseudo-direct \
  --skip-weak-w2s \
  --load-component-pseudo-cache "${ROOT_DIR}/results/qwen_w2s_addition_runs/pseudo_without_carry.json")

# 2. Weak (direct pseudo generation) on GPU 1.
weak_direct_pid=$(run_job "${GPU_STRONG}" "weak_direct" \
  --skip-strong-full \
  --skip-strong-w2s \
  --skip-strong-w2s-pseudo \
  --skip-strong-w2s-pseudo-direct \
  --skip-weak-w2s \
  --skip-weak-w2s-pseudo \
  --composed-strategy with_carry \
  --save-direct-pseudo-cache-with-carry "${OUT_ROOT}/direct_pseudo_with_carry.json" \
  --save-direct-pseudo-cache-without-carry "${OUT_ROOT}/direct_pseudo_without_carry.json")

wait_for_process "${weak_w2s_pseudo_nc_pid}" "weak_w2s_pseudo_without_carry"

# 3. Weak_W2S_Pseudo (component cache, with carry) after weak direct job, reuse pseudo cache.
weak_w2s_pseudo_wc_pid=$(run_job "${GPU_WEAK}" "weak_w2s_pseudo_with_carry" \
  --composed-strategy with_carry \
  --skip-weak \
  --skip-strong-full \
  --skip-strong-w2s \
  --skip-strong-w2s-pseudo \
  --skip-strong-w2s-pseudo-direct \
  --skip-weak-w2s \
  --load-component-pseudo-cache "${ROOT_DIR}/results/qwen_w2s_addition_runs/pseudo_with_carry.json")

wait_for_process "${weak_direct_pid}" "weak_direct"

# 4. Strong_W2S_Pseudo (direct pseudo) with carry on GPU 1.
strong_direct_wc_pid=$(run_job "${GPU_STRONG}" "strong_w2s_pseudo_direct_with_carry" \
  --composed-strategy with_carry \
  --skip-weak \
  --skip-strong-full \
  --skip-strong-w2s \
  --skip-strong-w2s-pseudo \
  --skip-weak-w2s \
  --skip-weak-w2s-pseudo \
  --load-direct-pseudo-cache "${OUT_ROOT}/direct_pseudo_with_carry.json")

wait_for_process "${weak_w2s_pseudo_wc_pid}" "weak_w2s_pseudo_with_carry"

# 5. Strong_W2S_Pseudo (direct pseudo) without carry on GPU 0, reuse direct cache.
strong_direct_nc_pid=$(run_job "${GPU_WEAK}" "strong_w2s_pseudo_direct_without_carry" \
  --composed-strategy without_carry \
  --skip-weak \
  --skip-strong-full \
  --skip-strong-w2s \
  --skip-strong-w2s-pseudo \
  --skip-weak-w2s \
  --skip-weak-w2s-pseudo \
  --load-direct-pseudo-cache "${OUT_ROOT}/direct_pseudo_without_carry.json")

# 6. Weak_W2S_Pseudo (direct pseudo) with carry on GPU 1.
weak_direct_wc_pid=$(run_job "${GPU_STRONG}" "weak_w2s_pseudo_direct_with_carry" \
  --composed-strategy with_carry \
  --skip-weak \
  --skip-strong-full \
  --skip-strong-w2s \
  --skip-strong-w2s-pseudo \
  --skip-weak-w2s \
  --load-direct-pseudo-cache "${OUT_ROOT}/direct_pseudo_with_carry.json")

wait_for_process "${strong_direct_wc_pid}" "strong_w2s_pseudo_direct_with_carry"

# 7. Weak_W2S_Pseudo (direct pseudo) without carry on GPU 1.
weak_direct_nc_pid=$(run_job "${GPU_STRONG}" "weak_w2s_pseudo_direct_without_carry" \
  --composed-strategy without_carry \
  --skip-weak \
  --skip-strong-full \
  --skip-strong-w2s \
  --skip-strong-w2s-pseudo \
  --skip-weak-w2s \
  --load-direct-pseudo-cache "${OUT_ROOT}/direct_pseudo_without_carry.json")

wait_for_process "${strong_direct_nc_pid}" "strong_w2s_pseudo_direct_without_carry"
wait_for_process "${weak_direct_wc_pid}" "weak_w2s_pseudo_direct_with_carry"
wait_for_process "${weak_direct_nc_pid}" "weak_w2s_pseudo_direct_without_carry"

echo ">>> Pseudo-focused experiment batch complete. Logs -> ${LOG_DIR}"
echo ">>> Outputs stored under ${OUT_ROOT}"
