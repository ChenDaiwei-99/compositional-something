#!/usr/bin/env bash
#SBATCH --job-name=selfimp-qwen
#SBATCH --output=logs/selfimp-qwen-%j.out
#SBATCH --error=logs/selfimp-qwen-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=mig
#SBATCH --qos=gpu-short
#SBATCH --time=08:00:00

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
  SCRIPT_DIR="${ROOT_DIR}/w2s"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${ROOT_DIR}"

mkdir -p "${ROOT_DIR}/logs"

TORCH_ENV_PATH="${TORCH_ENV_PATH:-${HOME}/.conda/envs/torch-env}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif [[ -x "${TORCH_ENV_PATH}/bin/python" ]]; then
  PYTHON_BIN="${TORCH_ENV_PATH}/bin/python"
else
  PYTHON_BIN="python"
fi

DEFAULT_LOCAL_MODEL="${ROOT_DIR}/models/Qwen3-0.6B"
MODEL_NAME="${MODEL_NAME:-${DEFAULT_LOCAL_MODEL}}"
SEED="${SEED:-42}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/w2s/self_improvement_runs/qwen_no_growth_${TS}}"

COMMON_ARGS=(
  --model-name "${MODEL_NAME}"
  --output-dir "${OUT_ROOT}"
  --seed "${SEED}"
  --bf16
  --per-device-train-batch-size 1
  --per-device-eval-batch-size 2
  --gradient-accumulation-steps 8
  --num-epochs 1
  --initial-min-digits 2
  --initial-max-digits 4
  --initial-train-per-digit 2000
  --initial-eval-per-digit 120
  --num-expand-rounds 10
  --expand-num-digits 2
  --expand-train-per-digit 1200
  --eval-per-digit 120
  --composed-eval-per-digit 60
  --composed-strategy without_carry
  --composed-refresh-mode dynamic
  --decode-max-new-tokens 48
  --logging-steps 20
)

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  COMMON_ARGS+=(${EXTRA_ARGS})
fi

echo "[INFO] Output dir: ${OUT_ROOT}"
echo "[INFO] Python: ${PYTHON_BIN}"
echo "[INFO] Model: ${MODEL_NAME}"
echo "[INFO] Slurm Job ID: ${SLURM_JOB_ID:-unknown}"
echo "[INFO] Hostname: $(hostname)"
nvidia-smi -L || true

"${PYTHON_BIN}" -c "import torch; print(f'[INFO] torch={torch.__version__} cuda={torch.cuda.is_available()} device_count={torch.cuda.device_count()}')"
MODEL_NAME="${MODEL_NAME}" "${PYTHON_BIN}" - <<'PY'
import os
from transformers import AutoConfig

model_name = os.environ["MODEL_NAME"]
AutoConfig.from_pretrained(model_name, trust_remote_code=True)
print(f"[INFO] model_config_ok={model_name}")
PY

"${PYTHON_BIN}" "${ROOT_DIR}/w2s/self_improvement.py" "${COMMON_ARGS[@]}"

echo "[INFO] Finished Qwen no-growth self-improvement run."
