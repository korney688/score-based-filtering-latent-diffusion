#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace}"
cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
MODE="${1:-preflight}"
case "${MODE}" in
  audit|preflight|dry-run|status|run) ;;
  *) echo 'Usage: bash scripts/remote/run_imagenet100_noise_seed_replicates.sh [audit|preflight|dry-run|status|run]' >&2; exit 2 ;;
esac
if [[ "${MODE}" == run ]]; then
  # One process owns both replicates. The lock is outside scientific run roots.
  LOCK_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/experiments}/imagenet100_image_space_variable_awgn_v1"
  mkdir -p "${LOCK_BASE}"
  exec 9>"${LOCK_BASE}/.noise_seed_replicates.lock"
  flock -n 9 || { echo 'Replicate launcher already running' >&2; exit 1; }
fi
python scripts/remote/imagenet100_noise_seed_replicates.py "${MODE}"
