#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ROOT_DIR}"

PROPORTIONS=(40 50 60 70 80 90 100)

# Customize additional arguments for each experiment as needed.
SELF_IMPROVEMENT_ARGS=()
SELF_IMPROVEMENT_GPU_POOL=(6)

declare -A SELF_GPU_PIDS=()

launch_self_job() {
    local gpu="$1"
    local percent="$2"
    local self_output_dir="${ROOT_DIR}/artifacts/runs/self_improvement/error_${percent}"

    echo "[INFO] Launching self_improvement (GPU${gpu}) with composition error percent=${percent}"
    if [ "${#SELF_IMPROVEMENT_ARGS[@]}" -gt 0 ]; then
        CUDA_VISIBLE_DEVICES="${gpu}" python -m w2s.self.self_improvement_composition_error_experiment \
            --composition-error-percent "${percent}" \
            -- --output-dir "${self_output_dir}" "${SELF_IMPROVEMENT_ARGS[@]}" &
    else
        CUDA_VISIBLE_DEVICES="${gpu}" python -m w2s.self.self_improvement_composition_error_experiment \
            --composition-error-percent "${percent}" \
            -- --output-dir "${self_output_dir}" &
    fi
    SELF_GPU_PIDS["${gpu}"]=$!
}

refresh_self_gpu_pool() {
    for gpu in "${SELF_IMPROVEMENT_GPU_POOL[@]}"; do
        local pid="${SELF_GPU_PIDS[$gpu]-}"
        if [[ -n "${pid}" ]]; then
            if ! kill -0 "${pid}" 2>/dev/null; then
                unset SELF_GPU_PIDS["${gpu}"]
            fi
        fi
    done
}

run_self_queue() {
    for percent in "${PROPORTIONS[@]}"; do
        while true; do
            refresh_self_gpu_pool
            available_gpu=""
            for gpu in "${SELF_IMPROVEMENT_GPU_POOL[@]}"; do
                if [[ -z "${SELF_GPU_PIDS[$gpu]+x}" ]]; then
                    available_gpu="${gpu}"
                    break
                fi
            done

            if [[ -n "${available_gpu}" ]]; then
                launch_self_job "${available_gpu}" "${percent}"
                break
            fi

            wait -n
        done
    done

    wait
}

run_self_queue

echo "[INFO] Completed composition error sweep."
