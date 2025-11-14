#!/usr/bin/env bash

# Run weak/strong addition experiments using GPU 1 and GPU 3.
# The script reuses pseudo-label caches so that Strong_W2S_Pseudo jobs
# don't regenerate pseudo labels repeatedly.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_WEAK="${GPU_WEAK:-1}"
GPU_STRONG="${GPU_STRONG:-2}"
OUT_ROOT="$(realpath "${1:-${ROOT_DIR}/results/qwen_w2s_addition_runs}")"
LOG_DIR="${OUT_ROOT}/logs"
PSEUDO_WITH="${OUT_ROOT}/pseudo_with_carry.json"
PSEUDO_WITHOUT="${OUT_ROOT}/pseudo_without_carry.json"
WANDB_MODE="${WANDB_MODE:-online}"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"
rm -f "${PSEUDO_WITH}" "${PSEUDO_WITHOUT}"

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
  WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-w2s-addition}" \
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${ROOT_DIR}/weak_to_strong_addition_experiment_v2.py" \
    --output-dir "${out_dir}" \
    --wandb-mode "${WANDB_MODE}" \
    "$@" >"${log_file}" 2>&1 &
  local pid=$!
  echo "${pid}"
}

# Function to wait for a process to complete
wait_for_process() {
  local pid="$1"
  local name="$2"
  echo ">>> Waiting for ${name} (PID: ${pid}) to complete..."
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 5
  done
  echo ">>> ${name} completed."
}

# 1. Weak-only run (GPU 1).
weak_pid=$(run_job "${GPU_WEAK}" "weak" \
  --skip-strong-full \
  --skip-strong-w2s \
  --skip-strong-w2s-pseudo \
  --composed-strategy without_carry)

# 2. Strong_Full (GPU 3) in parallel.
strong_full_pid=$(run_job "${GPU_STRONG}" "strong_full" \
  --skip-weak \
  --skip-strong-w2s \
  --skip-strong-w2s-pseudo \
  --composed-strategy without_carry)

# Wait for weak run so GPU 1 is free.
wait_for_process "${weak_pid}" "weak"
echo ">>> Weak phase complete."

# 3. Strong_W2S with carry (GPU 1) – also writes pseudo cache for reuse.
w2s_with_pid=$(run_job "${GPU_WEAK}" "strong_w2s_with_carry" \
  --skip-weak \
  --skip-strong-full \
  --skip-strong-w2s-pseudo \
  --composed-strategy with_carry \
  --save-component-pseudo-cache "${PSEUDO_WITH}")

# Wait for strong_full so GPU 3 is free.
wait_for_process "${strong_full_pid}" "strong_full"
echo ">>> Strong_Full phase complete."

# 4. Strong_W2S without carry (GPU 3) – produces pseudo cache for no-carry strategy.
w2s_without_pid=$(run_job "${GPU_STRONG}" "strong_w2s_without_carry" \
  --skip-weak \
  --skip-strong-full \
  --skip-strong-w2s-pseudo \
  --composed-strategy without_carry \
  --save-component-pseudo-cache "${PSEUDO_WITHOUT}")

# Finish both Strong_W2S runs before launching pseudo variants.
wait_for_process "${w2s_with_pid}" "strong_w2s_with_carry"
echo ">>> Strong_W2S (with carry) complete; pseudo cache -> ${PSEUDO_WITH}"

w2s_pseudo_with_pid=$(run_job "${GPU_WEAK}" "strong_w2s_pseudo_with_carry" \
  --skip-weak \
  --skip-strong-full \
  --skip-strong-w2s \
  --load-component-pseudo-cache "${PSEUDO_WITH}" \
  --composed-strategy with_carry)

wait_for_process "${w2s_without_pid}" "strong_w2s_without_carry"
echo ">>> Strong_W2S (without carry) complete; pseudo cache -> ${PSEUDO_WITHOUT}"

wait_for_process "${w2s_pseudo_with_pid}" "strong_w2s_pseudo_with_carry"

w2s_pseudo_without_pid=$(run_job "${GPU_STRONG}" "strong_w2s_pseudo_without_carry" \
  --skip-weak \
  --skip-strong-full \
  --skip-strong-w2s \
  --load-component-pseudo-cache "${PSEUDO_WITHOUT}" \
  --composed-strategy without_carry)

# Wait for pseudo variants.
wait_for_process "${w2s_pseudo_without_pid}" "strong_w2s_pseudo_without_carry"

echo ">>> All experiments completed. Logs -> ${LOG_DIR}"
echo ">>> Results stored under ${OUT_ROOT}"
