#!/usr/bin/env bash

set -euo pipefail

MODELS=(
  "Qwen/Qwen3-0.6B"
  "Qwen/Qwen3-1.7B"
  "Qwen/Qwen3-4B"
  "Qwen/Qwen3-8B"
  "meta-llama/Llama-3.1-8B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "microsoft/Phi-4-mini-instruct"
)

SEEDS=("13" "21" "37")

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${BASE_DIR}/composite_runs"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

NUM_EXAMPLES=1000
BATCH_SIZE=8
MAX_NEW_TOKENS=6

for model in "${MODELS[@]}"; do
  safe_model="${model//\//_}"
  for seed in "${SEEDS[@]}"; do
    report_path="${OUT_DIR}/${safe_model}_seed${seed}.json"
    log_path="${LOG_DIR}/${safe_model}_seed${seed}.log"
    echo "[$(date -Iseconds)] Starting composite run model=${model} seed=${seed}" | tee "${log_path}"
    python "${BASE_DIR}/composite_judge_experiment.py" \
      --model "${model}" \
      --num-examples "${NUM_EXAMPLES}" \
      --seed "${seed}" \
      --batch-size "${BATCH_SIZE}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --report-path "${report_path}" \
      >> "${log_path}" 2>&1
    echo "[$(date -Iseconds)] Finished composite run model=${model} seed=${seed}" | tee -a "${log_path}"
  done
done

echo "All composite jobs completed."

