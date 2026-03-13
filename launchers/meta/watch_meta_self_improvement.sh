#!/usr/bin/env bash
set -euo pipefail

# Supervises meta-self-improvement sbatch runs.
# - Polls a job until terminal state.
# - On failure, applies a conservative config fallback and resubmits.
# - Repeats until success or max attempts / deadline reached.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p "${ROOT_DIR}/artifacts/logs" "${ROOT_DIR}/artifacts/runs/meta_self_improvement"

JOB_SCRIPT="${JOB_SCRIPT:-${ROOT_DIR}/launchers/meta/run_meta_self_improvement_rope.sbatch}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-6}"
POLL_SECONDS="${POLL_SECONDS:-45}"
MAX_HOURS="${MAX_HOURS:-6}"
INITIAL_JOB_ID="${INITIAL_JOB_ID:-}"

WATCH_TS="$(date +%Y%m%d_%H%M%S)"
WATCH_LOG="${WATCH_LOG:-${ROOT_DIR}/artifacts/logs/meta-selfimp-watchdog-${WATCH_TS}.log}"
STATUS_FILE="${STATUS_FILE:-${ROOT_DIR}/artifacts/runs/meta_self_improvement/watchdog_status.json}"
RUN_DIRS_FILE="${RUN_DIRS_FILE:-${ROOT_DIR}/artifacts/runs/meta_self_improvement/watchdog_run_dirs.txt}"

deadline_epoch=$(( $(date +%s) + MAX_HOURS * 3600 ))

extra_args="${EXTRA_ARGS:-}"
use_fallback_profile=0

log() {
  local msg="$1"
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[${ts}] ${msg}" | tee -a "${WATCH_LOG}"
}

get_job_state() {
  local jid="$1"
  local state
  state="$(sacct -j "${jid}" --format=State -n -P 2>/dev/null | head -n1 | tr -d '[:space:]')"
  if [[ -z "${state}" ]]; then
    if squeue -j "${jid}" -h >/dev/null 2>&1; then
      state="$(squeue -j "${jid}" -h -o '%T' | head -n1 | tr -d '[:space:]')"
    else
      state="UNKNOWN"
    fi
  fi
  echo "${state}"
}

read_out_dir() {
  local jid="$1"
  local out_file="${ROOT_DIR}/artifacts/logs/meta-selfimp-rope-${jid}.out"
  if [[ ! -f "${out_file}" ]]; then
    return 0
  fi
  local dir
  dir="$(rg -n '^\[INFO\] Output dir:' "${out_file}" | tail -n1 | sed -E 's/^.*Output dir: //')"
  if [[ -n "${dir}" ]]; then
    echo "${dir}"
  fi
}

write_status() {
  local jid="$1"
  local state="$2"
  local attempt="$3"
  local out_dir="$4"
  cat > "${STATUS_FILE}" <<JSON
{
  "timestamp": "$(date '+%Y-%m-%dT%H:%M:%S')",
  "job_id": "${jid}",
  "state": "${state}",
  "attempt": ${attempt},
  "extra_args": "${extra_args}",
  "watch_log": "${WATCH_LOG}",
  "run_dir": "${out_dir}"
}
JSON
}

submit_job() {
  local jid
  if [[ -n "${extra_args}" ]]; then
    jid="$(sbatch --export=ALL,EXTRA_ARGS="${extra_args}" "${JOB_SCRIPT}" | awk '{print $4}')"
  else
    jid="$(sbatch "${JOB_SCRIPT}" | awk '{print $4}')"
  fi
  echo "${jid}"
}

choose_repair_profile() {
  local jid="$1"
  local err_file="${ROOT_DIR}/artifacts/logs/meta-selfimp-rope-${jid}.err"
  local out_file="${ROOT_DIR}/artifacts/logs/meta-selfimp-rope-${jid}.out"

  if [[ -f "${err_file}" ]] && rg -qi 'requires at least two .*stage configs' "${err_file}"; then
    extra_args="--capacity-growth-scheme progressive --stage-configs 96x4x2,96x4x3,96x4x4,96x4x5,96x4x6,96x4x7,96x4x8,96x4x9,96x4x10,96x4x11"
    use_fallback_profile=1
    log "Detected single-stage progressive override. Applying fallback EXTRA_ARGS='${extra_args}'."
    return
  fi

  if [[ -f "${err_file}" ]] && rg -qi 'out of memory|cuda error' "${err_file}"; then
    extra_args="--batch-size 64 --eval-batch-size 128 --capacity-growth-scheme progressive --stage-configs 96x4x2,96x4x3,96x4x4,96x4x5"
    use_fallback_profile=1
    log "Detected OOM. Applying fallback EXTRA_ARGS='${extra_args}'."
    return
  fi

  if [[ -f "${err_file}" ]] && rg -qi 'DUE TO TIME LIMIT|TIME LIMIT' "${err_file}"; then
    extra_args="--num-rounds 4 --batch-size 80 --eval-batch-size 160 --capacity-growth-scheme progressive --stage-configs 96x4x2,96x4x3,96x4x4,96x4x5,96x4x6"
    use_fallback_profile=1
    log "Detected time-limit issue. Applying fallback EXTRA_ARGS='${extra_args}'."
    return
  fi

  if [[ -f "${err_file}" ]] && rg -qi 'No module named|ImportError' "${err_file}"; then
    # Keep script defaults; this path mainly documents detection.
    log "Detected import/module failure. Retrying with same launcher defaults."
    return
  fi

  if [[ -f "${out_file}" ]] && rg -qi '\[INFO\] Dataset sizes:' "${out_file}"; then
    # Job made it into training pipeline; keep same args for retry.
    log "Job entered training pipeline before failing; retrying with same args."
    return
  fi

  if [[ ${use_fallback_profile} -eq 0 ]]; then
    extra_args="--batch-size 80 --eval-batch-size 160 --capacity-growth-scheme progressive"
    use_fallback_profile=1
    log "Applying generic conservative fallback EXTRA_ARGS='${extra_args}'."
  fi
}

summary_quality_ok() {
  local summary_path="$1"
  python - "$summary_path" <<'PY'
import json
import math
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
except Exception:
    print("summary_read_error")
    raise SystemExit(1)

rounds = payload.get("rounds") or []
if not rounds:
    print("summary_empty_rounds")
    raise SystemExit(1)

def to_num(value):
    try:
        x = float(value)
        if math.isnan(x):
            return 0.0
        return x
    except Exception:
        return 0.0

best_initial = max(to_num(r.get("initial_validation_accuracy", 0.0)) for r in rounds)
best_frontier = max(to_num(r.get("frontier_validation_accuracy", 0.0)) for r in rounds)
initial_mastered = bool(payload.get("initial_mastered", False))
stop_reason = str(payload.get("stop_reason", ""))

print(
    f"initial_mastered={initial_mastered} best_initial={best_initial:.4f} "
    f"best_frontier={best_frontier:.4f} stop_reason={stop_reason}"
)

# Conservative quality gate:
#  - initial distribution should be mastered
#  - and best initial accuracy should be high
if initial_mastered and best_initial >= 0.90:
    raise SystemExit(0)
raise SystemExit(1)
PY
}

choose_quality_profile() {
  # Stronger training profile for underfitting runs.
  extra_args="--initial-train-per-digit 3000 --validation-per-digit 160 --initial-bootstrapping-epochs 12 --num-epochs 5 --growth-warmup-epochs 8 --learning-rate 2e-4 --batch-size 80 --eval-batch-size 160 --max-total-rounds 140 --saturation-patience 4 --saturation-delta 0.001 --bootstrap-new-examples-per-digit 500 --min-rounds-per-frontier-before-growth 4 --capacity-growth-scheme progressive"
  use_fallback_profile=1
  log "Applying quality fallback EXTRA_ARGS='${extra_args}'."
}

log "Watchdog started."
log "Job script: ${JOB_SCRIPT}"
log "Max attempts: ${MAX_ATTEMPTS}, poll: ${POLL_SECONDS}s, horizon: ${MAX_HOURS}h"
log "Watch log: ${WATCH_LOG}"
log "Status file: ${STATUS_FILE}"

attempt=0
job_id="${INITIAL_JOB_ID}"

if [[ -n "${job_id}" ]]; then
  log "Using provided INITIAL_JOB_ID=${job_id}."
else
  job_id="$(submit_job)"
  attempt=1
  log "Submitted attempt ${attempt} as JobID=${job_id}."
fi

while true; do
  if [[ $(date +%s) -ge ${deadline_epoch} ]]; then
    log "Reached watchdog deadline; stopping."
    write_status "${job_id}" "DEADLINE_REACHED" "${attempt}" ""
    exit 1
  fi

  state="$(get_job_state "${job_id}")"
  out_dir="$(read_out_dir "${job_id}" || true)"
  write_status "${job_id}" "${state}" "${attempt}" "${out_dir:-}"

  if [[ -n "${out_dir:-}" ]]; then
    echo "${out_dir}" >> "${RUN_DIRS_FILE}"
  fi

  case "${state}" in
    PENDING|RUNNING|CONFIGURING|COMPLETING)
      log "Job ${job_id} state=${state}. Waiting..."
      sleep "${POLL_SECONDS}"
      ;;
    COMPLETED)
      log "Job ${job_id} COMPLETED successfully."
      if [[ -n "${out_dir:-}" ]]; then
        log "Result directory: ${out_dir}"
        if [[ -f "${out_dir}/summary.json" ]]; then
          log "Summary file: ${out_dir}/summary.json"
          if ! summary_quality_ok "${out_dir}/summary.json" | tee -a "${WATCH_LOG}"; then
            if [[ ${attempt} -ge ${MAX_ATTEMPTS} ]]; then
              log "Completed but quality gate failed and max attempts reached."
              write_status "${job_id}" "COMPLETED_LOW_QUALITY_MAX_ATTEMPTS" "${attempt}" "${out_dir:-}"
              exit 1
            fi
            choose_quality_profile
            attempt=$((attempt + 1))
            job_id="$(submit_job)"
            log "Submitted quality-retry attempt ${attempt} as JobID=${job_id}."
            continue
          fi
        fi
      fi
      write_status "${job_id}" "${state}" "${attempt}" "${out_dir:-}"
      exit 0
      ;;
    *)
      log "Job ${job_id} ended with state=${state}."
      if [[ -f "${ROOT_DIR}/artifacts/logs/meta-selfimp-rope-${job_id}.err" ]]; then
        tail -n 40 "${ROOT_DIR}/artifacts/logs/meta-selfimp-rope-${job_id}.err" | sed 's/^/[ERR] /' | tee -a "${WATCH_LOG}"
      fi

      if [[ ${attempt} -ge ${MAX_ATTEMPTS} ]]; then
        log "Max attempts reached (${MAX_ATTEMPTS}). Stopping."
        write_status "${job_id}" "FAILED_MAX_ATTEMPTS" "${attempt}" "${out_dir:-}"
        exit 1
      fi

      choose_repair_profile "${job_id}"
      attempt=$((attempt + 1))
      job_id="$(submit_job)"
      log "Submitted retry attempt ${attempt} as JobID=${job_id}."
      ;;
  esac
done
