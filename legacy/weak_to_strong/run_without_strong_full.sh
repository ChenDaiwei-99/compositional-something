#!/usr/bin/env bash

# Rerun weak and Strong_W2S variants (with/without pseudo) without Strong_Full.
# Generates fresh pseudo-label caches using dedicated weak-only jobs so that
# W2S runs can load them without retraining the weak model.

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
WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-w2s-rerun}"
WANDB_START_METHOD_OVERRIDE="${WANDB_START_METHOD_OVERRIDE:-thread}"

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
  WANDB_START_METHOD="${WANDB_START_METHOD_OVERRIDE}" \
  WANDB_CONSOLE="${WANDB_CONSOLE}" \
  WANDB_RUN_GROUP="${WANDB_RUN_GROUP}" \
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${ROOT_DIR}/weak_to_strong_addition_experiment_v2.py" \
    --output-dir "${out_dir}" \
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

# 1. Train weak baseline once and produce both pseudo caches.
weak_pid=$(run_job "${GPU_WEAK}" "weak" \
  --skip-strong-full \
  --skip-strong-w2s \
  --skip-strong-w2s-pseudo \
  --composed-strategy with_carry \
  --save-component-pseudo-cache-with-carry "${PSEUDO_WITH}" \
  --save-component-pseudo-cache-without-carry "${PSEUDO_WITHOUT}" \
  --wandb-mode "${WANDB_MODE}")

wait_for_process "${weak_pid}" "weak"
echo ">>> Pseudo caches ready."

# 2. Strong_W2S runs (load caches, skip weak training).
w2s_with_pid=$(run_job "${GPU_WEAK}" "strong_w2s_with_carry" \
  --skip-weak \
  --skip-strong-full \
  --skip-strong-w2s-pseudo \
  --composed-strategy with_carry \
  --load-component-pseudo-cache "${PSEUDO_WITH}" \
  --wandb-mode "${WANDB_MODE}")

w2s_without_pid=$(run_job "${GPU_STRONG}" "strong_w2s_without_carry" \
  --skip-weak \
  --skip-strong-full \
  --skip-strong-w2s-pseudo \
  --composed-strategy without_carry \
  --load-component-pseudo-cache "${PSEUDO_WITHOUT}" \
  --wandb-mode "${WANDB_MODE}")

wait_for_process "${w2s_with_pid}" "strong_w2s_with_carry"
wait_for_process "${w2s_without_pid}" "strong_w2s_without_carry"
echo ">>> Strong_W2S baselines complete."

# 3. Strong_W2S_Pseudo runs using the same caches.
w2s_pseudo_with_pid=$(run_job "${GPU_WEAK}" "strong_w2s_pseudo_with_carry" \
  --skip-weak \
  --skip-strong-full \
  --skip-strong-w2s \
  --composed-strategy with_carry \
  --load-component-pseudo-cache "${PSEUDO_WITH}" \
  --wandb-mode "${WANDB_MODE}")

w2s_pseudo_without_pid=$(run_job "${GPU_STRONG}" "strong_w2s_pseudo_without_carry" \
  --skip-weak \
  --skip-strong-full \
  --skip-strong-w2s \
  --composed-strategy without_carry \
  --load-component-pseudo-cache "${PSEUDO_WITHOUT}" \
  --wandb-mode "${WANDB_MODE}")

wait_for_process "${w2s_pseudo_with_pid}" "strong_w2s_pseudo_with_carry"
wait_for_process "${w2s_pseudo_without_pid}" "strong_w2s_pseudo_without_carry"

echo ">>> All requested experiments completed (weak + Strong_W2S variants). Logs -> ${LOG_DIR}"
echo ">>> Results stored under ${OUT_ROOT}"
