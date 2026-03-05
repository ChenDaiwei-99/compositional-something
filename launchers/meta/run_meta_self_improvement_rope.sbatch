#!/usr/bin/env bash
#SBATCH --job-name=meta-selfimp-rope
#SBATCH --output=artifacts/logs/meta-selfimp-rope-%j.out
#SBATCH --error=artifacts/logs/meta-selfimp-rope-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -euo pipefail

# In Slurm, batch scripts may be staged to a spool path.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
  SCRIPT_DIR="${ROOT_DIR}/launchers/meta"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${ROOT_DIR}"

mkdir -p "${ROOT_DIR}/artifacts/logs"

TORCH_ENV_PATH="${TORCH_ENV_PATH:-${HOME}/.conda/envs/torch-env}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif [[ -x "${TORCH_ENV_PATH}/bin/python" ]]; then
  PYTHON_BIN="${TORCH_ENV_PATH}/bin/python"
else
  PYTHON_BIN="python"
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/meta_self_improvement/cluster_${TS}}"
MAX_PSEUDO_PER_ROUND_PER_DIGIT="${MAX_PSEUDO_PER_ROUND_PER_DIGIT:-1000}"
NEW_PSEUDO_UNIQUE_QUOTA_PER_DIGIT="${NEW_PSEUDO_UNIQUE_QUOTA_PER_DIGIT:-1000}"
MIN_UNIQUE_CANDIDATE_PER_DIGIT="${MIN_UNIQUE_CANDIDATE_PER_DIGIT:-200}"
MIN_UNIQUE_COMPONENT_ACC="${MIN_UNIQUE_COMPONENT_ACC:-0.85}"
UNIQUE_CANDIDATE_AUGMENT_PER_DIGIT="${UNIQUE_CANDIDATE_AUGMENT_PER_DIGIT:-200}"
UNIQUE_CANDIDATE_AUGMENT_MAX_EVENTS_PER_EDGE_DIGIT="${UNIQUE_CANDIDATE_AUGMENT_MAX_EVENTS_PER_EDGE_DIGIT:-4}"
# Keep hard-fail off by default; enable only for strict debugging.
MIN_UNIQUE_CANDIDATE_HARD_FAIL="${MIN_UNIQUE_CANDIDATE_HARD_FAIL:-0}"
SEED_RETENTION_PER_DIGIT="${SEED_RETENTION_PER_DIGIT:-1000}"
VALIDATION_PER_DIGIT="${VALIDATION_PER_DIGIT:-100}"
STALL_BOOTSTRAP_COMPOSED_PER_DIGIT="${STALL_BOOTSTRAP_COMPOSED_PER_DIGIT:-1000}"
STALL_BOOTSTRAP_MAX_EVENTS_PER_EDGE_DIGIT="${STALL_BOOTSTRAP_MAX_EVENTS_PER_EDGE_DIGIT:-3}"
EXPAND_NUM_DIGITS_DIVISION_FACTOR="${EXPAND_NUM_DIGITS_DIVISION_FACTOR:-2}"
FULL_EVAL_INTERVAL="${FULL_EVAL_INTERVAL:-0}"
GROWTH_DISTILL_EPOCHS="${GROWTH_DISTILL_EPOCHS:-2}"
CONTINUATION_SAVE_INTERVAL="${CONTINUATION_SAVE_INTERVAL:-5}"
RESUME_FROM_LATEST_STATE="${RESUME_FROM_LATEST_STATE:-0}"

COMMON_ARGS=(
  --output-dir "${OUT_ROOT}"
  --initial-min-digits 2
  --initial-max-digits 4
  --initial-train-per-digit 3000
  --validation-per-digit "${VALIDATION_PER_DIGIT}"
  --composed-train-per-digit 3000
  --composed-boundary-mode with_carry_filtered
  --expand-num-digits-division-factor "${EXPAND_NUM_DIGITS_DIVISION_FACTOR}"
  --max-pseudo-per-round-per-digit "${MAX_PSEUDO_PER_ROUND_PER_DIGIT}"
  --new-pseudo-unique-quota-per-digit "${NEW_PSEUDO_UNIQUE_QUOTA_PER_DIGIT}"
  --min-unique-candidate-per-digit "${MIN_UNIQUE_CANDIDATE_PER_DIGIT}"
  --min-unique-candidate-min-component-acc "${MIN_UNIQUE_COMPONENT_ACC}"
  --unique-candidate-augment-per-digit "${UNIQUE_CANDIDATE_AUGMENT_PER_DIGIT}"
  --unique-candidate-augment-max-events-per-edge-digit "${UNIQUE_CANDIDATE_AUGMENT_MAX_EVENTS_PER_EDGE_DIGIT}"
  --max-total-rounds 10000
  --stage-configs "96x4x2,128x4x3,160x5x4,192x6x5,224x7x6,256x8x7,288x9x8,320x10x9,352x11x10,384x12x11"
  --num-epochs 3
  --initial-bootstrapping-epochs 8
  --growth-warmup-epochs 5
  --growth-distill-epochs "${GROWTH_DISTILL_EPOCHS}"
  --learning-rate 2e-4
  --saturation-patience 3
  --stall-bootstrap-composed-per-digit "${STALL_BOOTSTRAP_COMPOSED_PER_DIGIT}"
  --stall-bootstrap-max-events-per-edge-digit "${STALL_BOOTSTRAP_MAX_EVENTS_PER_EDGE_DIGIT}"
  --min-rounds-per-frontier-before-growth 3
  --seed-retention-per-digit "${SEED_RETENTION_PER_DIGIT}"
  --initial-mastery-threshold 0.95
  --frontier-mastery-threshold 0.9
  --min-growth-seed-examples 1024
  --bootstrap-new-examples-per-digit 1500
  --full-eval-interval "${FULL_EVAL_INTERVAL}"
  --continuation-save-interval "${CONTINUATION_SAVE_INTERVAL}"
)

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  COMMON_ARGS+=(${EXTRA_ARGS})
fi
case "${MIN_UNIQUE_CANDIDATE_HARD_FAIL,,}" in
  1|true|yes)
    COMMON_ARGS+=(--min-unique-candidate-hard-fail)
    ;;
esac
case "${RESUME_FROM_LATEST_STATE,,}" in
  1|true|yes)
    COMMON_ARGS+=(--resume-from-latest-state)
    ;;
esac

echo "[INFO] Root dir: ${ROOT_DIR}"
echo "[INFO] Output dir: ${OUT_ROOT}"
echo "[INFO] Python: ${PYTHON_BIN}"
echo "[INFO] Expansion: division_factor=${EXPAND_NUM_DIGITS_DIVISION_FACTOR}"
echo "[INFO] Pseudo caps: max_per_digit=${MAX_PSEUDO_PER_ROUND_PER_DIGIT} unique_quota_per_digit=${NEW_PSEUDO_UNIQUE_QUOTA_PER_DIGIT}"
echo "[INFO] Unique floor: min_unique_per_digit=${MIN_UNIQUE_CANDIDATE_PER_DIGIT} augment_per_digit=${UNIQUE_CANDIDATE_AUGMENT_PER_DIGIT} augment_max_events_per_edge_digit=${UNIQUE_CANDIDATE_AUGMENT_MAX_EVENTS_PER_EDGE_DIGIT} hard_fail=${MIN_UNIQUE_CANDIDATE_HARD_FAIL}"
echo "[INFO] Stall bootstrap: composed_per_digit=${STALL_BOOTSTRAP_COMPOSED_PER_DIGIT} max_events_per_edge_digit=${STALL_BOOTSTRAP_MAX_EVENTS_PER_EDGE_DIGIT}"
echo "[INFO] Seed retention: per_digit=${SEED_RETENTION_PER_DIGIT} (set 0 to disable)"
echo "[INFO] Validation: per_digit=${VALIDATION_PER_DIGIT} full_eval_interval=${FULL_EVAL_INTERVAL}"
echo "[INFO] Growth distill epochs: ${GROWTH_DISTILL_EPOCHS}"
echo "[INFO] Continuation: interval=${CONTINUATION_SAVE_INTERVAL} resume_from_latest=${RESUME_FROM_LATEST_STATE}"
echo "[INFO] Slurm Job ID: ${SLURM_JOB_ID:-unknown}"
echo "[INFO] Hostname: $(hostname)"
nvidia-smi -L || true

"${PYTHON_BIN}" -c "import torch; print(f'[INFO] torch={torch.__version__} cuda={torch.cuda.is_available()} device_count={torch.cuda.device_count()}')"

"${PYTHON_BIN}" -m meta.train_meta_self_improvement_rope "${COMMON_ARGS[@]}"

echo "[INFO] Finished meta self-improvement run. Results at: ${OUT_ROOT}"
