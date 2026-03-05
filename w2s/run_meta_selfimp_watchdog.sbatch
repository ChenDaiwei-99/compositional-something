#!/usr/bin/env bash
#SBATCH --job-name=meta-selfimp-watch
#SBATCH --output=logs/meta-selfimp-watch-%j.out
#SBATCH --error=logs/meta-selfimp-watch-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=cpu
#SBATCH --time=06:30:00

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${ROOT_DIR}"

mkdir -p "${ROOT_DIR}/logs"

echo "[INFO] Watchdog wrapper running on $(hostname)"
echo "[INFO] Initial job id: ${INITIAL_JOB_ID:-}"
echo "[INFO] Max hours: ${MAX_HOURS:-6}"
echo "[INFO] Poll seconds: ${POLL_SECONDS:-45}"
echo "[INFO] Max attempts: ${MAX_ATTEMPTS:-6}"

bash "${ROOT_DIR}/w2s/watch_meta_self_improvement.sh"
