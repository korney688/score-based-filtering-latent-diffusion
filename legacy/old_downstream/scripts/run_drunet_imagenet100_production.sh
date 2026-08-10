#!/usr/bin/env bash
set -Eeuo pipefail

DATA_ROOT="${DATA_ROOT:-/workspace/data}"
DATASET_ROOT="${DATA_ROOT}/imagenet100"
OUTPUT_ROOT="experiments/imagenet100/exp_007_drunet"
CHECKPOINT_ROOT="checkpoints/imagenet100/drunet"
FILTERING_ROOT="experiments/imagenet100/exp_005_filtering"
FORCE="${FORCE:-0}"

RUNS=(
  "quantile10_sigma25"
  "topk10_sigma25"
  "full_sigma25"
)

fail() {
  echo "SMOKE/PRODUCTION FAILED: $*" >&2
  exit 1
}

on_error() {
  local exit_code=$?
  echo "DRUNet production launch failed at line ${BASH_LINENO[0]} with exit code ${exit_code}" >&2
  exit "${exit_code}"
}
trap on_error ERR

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || fail "missing file: ${path}"
}

require_dir() {
  local path="$1"
  [[ -d "${path}" ]] || fail "missing directory: ${path}"
}

check_artifacts() {
  local run_name="$1"
  local run_dir="${OUTPUT_ROOT}/${run_name}"

  require_dir "${run_dir}"
  require_file "${run_dir}/config.json"
  require_file "${run_dir}/metrics.json"
  require_file "${run_dir}/results/per_image_metrics.csv"
  require_file "${run_dir}/results/training_history.csv"
  require_file "${CHECKPOINT_ROOT}/${run_name}.pth"
}

guard_existing_outputs() {
  if [[ "${FORCE}" == "1" ]]; then
    echo "FORCE=1: existing DRUNet output directories/checkpoints are allowed. Nothing will be deleted automatically."
    return
  fi

  for run_name in "${RUNS[@]}"; do
    if [[ -e "${OUTPUT_ROOT}/${run_name}" || -e "${CHECKPOINT_ROOT}/${run_name}.pth" ]]; then
      fail "existing artifacts for ${run_name}. Set FORCE=1 to allow continuing without automatic deletion."
    fi
  done
}

echo "DRUNet ImageNet-100 production launch"
echo "data root: ${DATA_ROOT}"
echo "dataset root: ${DATASET_ROOT}"
echo "output root: ${OUTPUT_ROOT}"
echo "checkpoint root: ${CHECKPOINT_ROOT}"
echo "runs: ${RUNS[*]}"
echo "batch_size: 64"
echo "epochs: 15"
echo "sigma: 25/255"
echo

require_dir "${DATASET_ROOT}/train"
require_dir "${DATASET_ROOT}/val"
require_file "${FILTERING_ROOT}/quantile/10/selected_indices.npy"
require_file "${FILTERING_ROOT}/topk/10/selected_indices.npy"
guard_existing_outputs

echo "[1/5] Smoke checks"
python -m compileall -q scripts src

python scripts/train_drunet.py \
  --dataset imagenet100 \
  --list-runs

python scripts/train_drunet.py \
  --dataset imagenet100 \
  --dry-run

echo "[2/5] Training quantile10_sigma25"
python scripts/train_drunet.py \
  --dataset imagenet100 \
  --run quantile10_sigma25 \
  --data-root "${DATA_ROOT}"
check_artifacts "quantile10_sigma25"

echo "[3/5] Training topk10_sigma25"
python scripts/train_drunet.py \
  --dataset imagenet100 \
  --run topk10_sigma25 \
  --data-root "${DATA_ROOT}"
check_artifacts "topk10_sigma25"

echo "[4/5] Training full_sigma25"
python scripts/train_drunet.py \
  --dataset imagenet100 \
  --run full_sigma25 \
  --data-root "${DATA_ROOT}"
check_artifacts "full_sigma25"

echo "[5/5] Evaluation artifact summary"
for run_name in "${RUNS[@]}"; do
  echo "${run_name}:"
  echo "  metrics: ${OUTPUT_ROOT}/${run_name}/metrics.json"
  echo "  per-image metrics: ${OUTPUT_ROOT}/${run_name}/results/per_image_metrics.csv"
  echo "  history: ${OUTPUT_ROOT}/${run_name}/results/training_history.csv"
  echo "  checkpoint: ${CHECKPOINT_ROOT}/${run_name}.pth"
done

echo "DRUNET PRODUCTION PASSED"
