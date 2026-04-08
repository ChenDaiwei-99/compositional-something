#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

JOB_SCRIPT="${JOB_SCRIPT:-${ROOT_DIR}/launchers/meta/run_meta_self_improvement_rope.sbatch}"
TS="$(date +%Y%m%d_%H%M%S)"
BASE_OUT="${BASE_OUT:-${ROOT_DIR}/artifacts/runs/meta_self_improvement/compare_legacy_vs_progressive_depth_width_${TS}}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-80}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-160}"
SBATCH_TIME="${SBATCH_TIME:-36:00:00}"
FRONTIER_EXPAND_THRESHOLD="${FRONTIER_EXPAND_THRESHOLD:-0.95}"
FRONTIER_MASTERY_THRESHOLD="${FRONTIER_MASTERY_THRESHOLD:-0.95}"
INITIAL_TRAIN_PER_DIGIT="${INITIAL_TRAIN_PER_DIGIT:-3000}"
COMPOSED_TRAIN_PER_DIGIT="${COMPOSED_TRAIN_PER_DIGIT:-3000}"
VALIDATION_PER_DIGIT="${VALIDATION_PER_DIGIT:-100}"
BOOTSTRAP_NEW_EXAMPLES_PER_DIGIT="${BOOTSTRAP_NEW_EXAMPLES_PER_DIGIT:-1500}"
MAX_PSEUDO_PER_ROUND_PER_DIGIT="${MAX_PSEUDO_PER_ROUND_PER_DIGIT:-1000}"
NEW_PSEUDO_UNIQUE_QUOTA_PER_DIGIT="${NEW_PSEUDO_UNIQUE_QUOTA_PER_DIGIT:-1000}"
MIN_UNIQUE_CANDIDATE_PER_DIGIT="${MIN_UNIQUE_CANDIDATE_PER_DIGIT:-200}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
INITIAL_BOOTSTRAPPING_EPOCHS="${INITIAL_BOOTSTRAPPING_EPOCHS:-8}"
GROWTH_WARMUP_EPOCHS="${GROWTH_WARMUP_EPOCHS:-5}"

LEGACY_STAGE_CONFIGS="${LEGACY_STAGE_CONFIGS:-96x4x2,128x4x3,160x5x4,192x6x5,224x7x6,256x8x7,288x9x8,320x10x9,352x11x10,384x12x11}"
PROGRESSIVE_DEPTH_STAGE_CONFIGS="${PROGRESSIVE_DEPTH_STAGE_CONFIGS:-96x4x2,96x4x3,96x4x4,96x4x5,96x4x6,96x4x7,96x4x8,96x4x9,96x4x10,96x4x11}"
PROGRESSIVE_DEPTH_WIDTH_STAGE_CONFIGS="${PROGRESSIVE_DEPTH_WIDTH_STAGE_CONFIGS:-${LEGACY_STAGE_CONFIGS}}"

mkdir -p "${BASE_OUT}"

submit_arm() {
  local arm_name="$1"
  local scheme="$2"
  local stage_configs="$3"
  local out_root="${BASE_OUT}/${arm_name}"
  local job_id

  job_id="$(
    OUT_ROOT="${out_root}" \
    CAPACITY_GROWTH_SCHEME="${scheme}" \
    STAGE_CONFIGS="${stage_configs}" \
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
    FRONTIER_EXPAND_THRESHOLD="${FRONTIER_EXPAND_THRESHOLD}" \
    FRONTIER_MASTERY_THRESHOLD="${FRONTIER_MASTERY_THRESHOLD}" \
    INITIAL_TRAIN_PER_DIGIT="${INITIAL_TRAIN_PER_DIGIT}" \
    COMPOSED_TRAIN_PER_DIGIT="${COMPOSED_TRAIN_PER_DIGIT}" \
    VALIDATION_PER_DIGIT="${VALIDATION_PER_DIGIT}" \
    BOOTSTRAP_NEW_EXAMPLES_PER_DIGIT="${BOOTSTRAP_NEW_EXAMPLES_PER_DIGIT}" \
    MAX_PSEUDO_PER_ROUND_PER_DIGIT="${MAX_PSEUDO_PER_ROUND_PER_DIGIT}" \
    NEW_PSEUDO_UNIQUE_QUOTA_PER_DIGIT="${NEW_PSEUDO_UNIQUE_QUOTA_PER_DIGIT}" \
    MIN_UNIQUE_CANDIDATE_PER_DIGIT="${MIN_UNIQUE_CANDIDATE_PER_DIGIT}" \
    NUM_EPOCHS="${NUM_EPOCHS}" \
    INITIAL_BOOTSTRAPPING_EPOCHS="${INITIAL_BOOTSTRAPPING_EPOCHS}" \
    GROWTH_WARMUP_EPOCHS="${GROWTH_WARMUP_EPOCHS}" \
    sbatch \
      --time="${SBATCH_TIME}" \
      --export=ALL \
      "${JOB_SCRIPT}" | awk '{print $4}'
  )"

  echo "${job_id}"
}

legacy_job_id="$(submit_arm legacy legacy "${LEGACY_STAGE_CONFIGS}")"
progressive_depth_job_id="$(submit_arm progressive_depth progressive_depth "${PROGRESSIVE_DEPTH_STAGE_CONFIGS}")"
progressive_depth_width_job_id="$(submit_arm progressive_depth_width progressive_depth_width "${PROGRESSIVE_DEPTH_WIDTH_STAGE_CONFIGS}")"

cat > "${BASE_OUT}/submission_info.txt" <<EOF
base_out=${BASE_OUT}
train_batch_size=${TRAIN_BATCH_SIZE}
eval_batch_size=${EVAL_BATCH_SIZE}
sbatch_time=${SBATCH_TIME}
frontier_expand_threshold=${FRONTIER_EXPAND_THRESHOLD}
frontier_mastery_threshold=${FRONTIER_MASTERY_THRESHOLD}
initial_train_per_digit=${INITIAL_TRAIN_PER_DIGIT}
composed_train_per_digit=${COMPOSED_TRAIN_PER_DIGIT}
validation_per_digit=${VALIDATION_PER_DIGIT}
bootstrap_new_examples_per_digit=${BOOTSTRAP_NEW_EXAMPLES_PER_DIGIT}
max_pseudo_per_round_per_digit=${MAX_PSEUDO_PER_ROUND_PER_DIGIT}
new_pseudo_unique_quota_per_digit=${NEW_PSEUDO_UNIQUE_QUOTA_PER_DIGIT}
min_unique_candidate_per_digit=${MIN_UNIQUE_CANDIDATE_PER_DIGIT}
num_epochs=${NUM_EPOCHS}
initial_bootstrapping_epochs=${INITIAL_BOOTSTRAPPING_EPOCHS}
growth_warmup_epochs=${GROWTH_WARMUP_EPOCHS}
legacy_stage_configs=${LEGACY_STAGE_CONFIGS}
progressive_depth_stage_configs=${PROGRESSIVE_DEPTH_STAGE_CONFIGS}
progressive_depth_width_stage_configs=${PROGRESSIVE_DEPTH_WIDTH_STAGE_CONFIGS}
legacy_job_id=${legacy_job_id}
legacy_out_root=${BASE_OUT}/legacy
progressive_depth_job_id=${progressive_depth_job_id}
progressive_depth_out_root=${BASE_OUT}/progressive_depth
progressive_depth_width_job_id=${progressive_depth_width_job_id}
progressive_depth_width_out_root=${BASE_OUT}/progressive_depth_width
submitted_at=$(date --iso-8601=seconds)
EOF

echo "[INFO] Submitted comparison runs:"
echo "  legacy -> ${legacy_job_id}"
echo "  progressive_depth -> ${progressive_depth_job_id}"
echo "  progressive_depth_width -> ${progressive_depth_width_job_id}"
echo "[INFO] Metadata written to ${BASE_OUT}/submission_info.txt"
