#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PROPORTIONS=(0 10 20 30 40 50 60 70 80 90 100)
GPU_POOL=(0 1 6)

# Add any common arguments for all runs below (e.g., --weak-train-per-digit ...).
W2S_ARGS=()

declare -A GPU_PIDS=()

launch_job() {
    local gpu="$1"
    local percent="$2"
    local output_dir="results/w2s_err_${percent}"

    echo "[INFO] Launching weak_to_strong (GPU${gpu}) with composition error percent=${percent}"
    if [ "${#W2S_ARGS[@]}" -gt 0 ]; then
        CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT_DIR}/weak_to_strong_composition_error_experiment.py" \
            --composition-error-percent "${percent}" \
            -- --output-dir "${output_dir}" "${W2S_ARGS[@]}" &
    else
        CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT_DIR}/weak_to_strong_composition_error_experiment.py" \
            --composition-error-percent "${percent}" \
            -- --output-dir "${output_dir}" &
    fi
    GPU_PIDS["${gpu}"]=$!
}

refresh_gpu_pool() {
    for gpu in "${GPU_POOL[@]}"; do
        local pid="${GPU_PIDS[$gpu]-}"
        if [[ -n "${pid}" ]]; then
            if ! kill -0 "${pid}" 2>/dev/null; then
                unset GPU_PIDS["${gpu}"]
            fi
        fi
    done
}

for percent in "${PROPORTIONS[@]}"; do
    while true; do
        refresh_gpu_pool
        available_gpu=""
        for gpu in "${GPU_POOL[@]}"; do
            if [[ -z "${GPU_PIDS[$gpu]+x}" ]]; then
                available_gpu="${gpu}"
                break
            fi
        done

        if [[ -n "${available_gpu}" ]]; then
            launch_job "${available_gpu}" "${percent}"
            break
        fi

        # No GPU free yet; wait for any job to finish before looping.
        wait -n || true
    done
done

# Wait for any remaining jobs to finish.
wait

echo "[INFO] Completed weak-to-strong composition error sweep on GPUs ${GPU_POOL[*]}."
