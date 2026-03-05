#!/usr/bin/env bash
#SBATCH --job-name=selfimp-boundary
#SBATCH --output=logs/selfimp-boundary-%j.out
#SBATCH --error=logs/selfimp-boundary-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=mig
#SBATCH --time=02:59:00

set -euo pipefail

# In Slurm, batch scripts are often staged to a spool path, so BASH_SOURCE[0]
# may not point to the repository location. Prefer the submit directory.
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
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/w2s/self_improvement_runs/mig_boundary_eval_${TS}}"

# Defaults are tuned for a MIG slice; override with EXTRA_ARGS if needed.
COMMON_ARGS=(
  --model-name "${MODEL_NAME}"
  --seed "${SEED}"
  --fp16
  --per-device-train-batch-size 1
  --per-device-eval-batch-size 2
  --gradient-accumulation-steps 8
  --num-epochs 1
  --initial-min-digits 3
  --initial-max-digits 7
  --initial-train-per-digit 1200
  --initial-eval-per-digit 40
  --num-expand-rounds 2
  --expand-num-digits 4
  --expand-train-per-digit 1200
  --eval-per-digit 60
  --composed-eval-per-digit 60
  --decode-max-new-tokens 24
  --logging-steps 20
)

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  COMMON_ARGS+=(${EXTRA_ARGS})
fi

echo "[INFO] Output root: ${OUT_ROOT}"
echo "[INFO] Python: ${PYTHON_BIN}"
echo "[INFO] Model: ${MODEL_NAME}"
"${PYTHON_BIN}" -c "import torch; print(f'[INFO] torch={torch.__version__} cuda={torch.cuda.is_available()}')"
MODEL_NAME="${MODEL_NAME}" "${PYTHON_BIN}" - <<'PY'
import os
from transformers import AutoConfig

model_name = os.environ["MODEL_NAME"]
AutoConfig.from_pretrained(model_name, trust_remote_code=True)
print(f"[INFO] model_config_ok={model_name}")
PY

echo "[INFO] Run 1/2: with_carry (no filtering)"
"${PYTHON_BIN}" "${ROOT_DIR}/w2s/self_improvement.py" \
  --output-dir "${OUT_ROOT}/with_carry" \
  --composed-strategy with_carry \
  "${COMMON_ARGS[@]}"

echo "[INFO] Run 2/2: with_carry_filtered (0% boundary-carry retention)"
"${PYTHON_BIN}" "${ROOT_DIR}/w2s/self_improvement_composition_error_experiment.py" \
  --composition-error-percent 0 \
  -- \
  --output-dir "${OUT_ROOT}/with_carry_filtered_p0" \
  "${COMMON_ARGS[@]}"

echo "[INFO] Finished MIG boundary-carry slice experiment."
