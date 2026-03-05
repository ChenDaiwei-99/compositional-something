#!/usr/bin/env bash

# Launch Strong_W2S_Pseudo (component-stitched) runs for Qwen3-4B and Qwen3-8B.
# Uses cached weak pseudo labels and trains with LoRA adapters to fit on 40GB GPUs.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

GPU_Q4B="${GPU_Q4B:-1}"
GPU_Q8B="${GPU_Q8B:-6}"

OUT_ROOT="$(realpath "${1:-${ROOT_DIR}/results/strong_component_pseudo_runs}")"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

PSEUDO_WITH="${ROOT_DIR}/results/qwen_w2s_addition_runs/pseudo_with_carry.json"
PSEUDO_WITHOUT="${ROOT_DIR}/results/qwen_w2s_addition_runs/pseudo_without_carry.json"

for cache_path in "${PSEUDO_WITH}" "${PSEUDO_WITHOUT}"; do
  if [[ ! -f "${cache_path}" ]]; then
    echo "Missing pseudo cache: ${cache_path}" >&2
    exit 1
  fi
done

WANDB_MODE_OPT="${WANDB_MODE_OVERRIDE:-${WANDB_MODE:-online}}"
WANDB_GROUP_OPT="${WANDB_RUN_GROUP:-w2s-strong-component}"

LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

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
  WANDB_RUN_GROUP="${WANDB_GROUP_OPT}" \
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${ROOT_DIR}/weak_to_strong_addition_experiment_v2.py" \
    --output-dir "${out_dir}" \
    --wandb-mode "${WANDB_MODE_OPT}" \
    "$@" >"${log_file}" 2>&1 &
  echo $!
}

wait_for_process() {
  local pid="$1"
  local label="$2"
  echo ">>> Waiting for ${label} (PID: ${pid}) to complete..."
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 5
  done
  echo ">>> ${label} completed."
}

declare -A GPU_PIDS=()
declare -A GPU_LABELS=()

wait_for_gpu() {
  local gpu="$1"
  local pid="${GPU_PIDS[${gpu}]:-}"
  local label="${GPU_LABELS[${gpu}]:-}"
  if [[ -n "${pid}" ]]; then
    wait_for_process "${pid}" "${label}"
    unset "GPU_PIDS[${gpu}]"
    unset "GPU_LABELS[${gpu}]"
  fi
}

launch_variant() {
  local model_slug="$1"
  local model_name="$2"
  local gpu="$3"
  local batch_size="$4"
  local grad_accum="$5"
  local strategy="$6"
  local cache_path
  if [[ "${strategy}" == "with_carry" ]]; then
    cache_path="${PSEUDO_WITH}"
  else
    cache_path="${PSEUDO_WITHOUT}"
  fi
  local run_name="${model_slug}_${strategy}"

  local -a args=(
    --strong-model "${model_name}"
    --skip-weak
    --skip-strong-full
    --skip-strong-w2s
    --skip-strong-w2s-pseudo-direct
    --skip-weak-w2s
    --skip-weak-w2s-pseudo
    --composed-strategy "${strategy}"
    --load-component-pseudo-cache "${cache_path}"
    --bf16
    --use-lora
    --lora-apply-to strong
    --lora-r "${LORA_R}"
    --lora-alpha "${LORA_ALPHA}"
    --lora-dropout "${LORA_DROPOUT}"
    --lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
    --per-device-train-batch-size "${batch_size}"
    --per-device-eval-batch-size "${batch_size}"
    --gradient-accumulation-steps "${grad_accum}"
  )

  local pid
  pid=$(run_job "${gpu}" "${run_name}" "${args[@]}")
  GPU_PIDS["${gpu}"]="${pid}"
  GPU_LABELS["${gpu}"]="${run_name}"
}

launch_variant "qwen3_4b" "Qwen/Qwen3-4B" "${GPU_Q4B}" 2 1 "with_carry"
launch_variant "qwen3_8b" "Qwen/Qwen3-8B" "${GPU_Q8B}" 2 2 "with_carry"
wait_for_gpu "${GPU_Q4B}"
wait_for_gpu "${GPU_Q8B}"

launch_variant "qwen3_4b" "Qwen/Qwen3-4B" "${GPU_Q4B}" 2 1 "without_carry"
launch_variant "qwen3_8b" "Qwen/Qwen3-8B" "${GPU_Q8B}" 2 2 "without_carry"
wait_for_gpu "${GPU_Q4B}"
wait_for_gpu "${GPU_Q8B}"

echo ">>> Strong component pseudo-label batch complete. Logs -> ${LOG_DIR}"
echo ">>> Outputs stored under ${OUT_ROOT}"
