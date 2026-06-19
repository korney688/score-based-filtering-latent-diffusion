#!/usr/bin/env bash

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATASET_ROOT="${IMAGENET100_ROOT:-${PROJECT_ROOT}/data/imagenet100}"
FORCE="${FORCE:-0}"

cd "${PROJECT_ROOT}" || {
  echo "DOWNSTREAM GRID FAILED"
  echo "Cannot cd to PROJECT_ROOT=${PROJECT_ROOT}"
  exit 1
}

export IMAGENET100_ROOT="${DATASET_ROOT}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

FILTERING_ROOT="experiments/imagenet100/exp_005_filtering"
TDNCNN_ROOT="experiments/imagenet100/exp_006_tdncnn"
TDNCNN_CKPT_ROOT="checkpoints/imagenet100/tdncnn"

REQUIRED_FILTERING_SUBSETS=(
  "topk/5"
  "topk/10"
  "topk/15"
  "quantile/5"
  "quantile/10"
  "quantile/15"
)

REUSED_TDNCNN_RUNS=(
  "full"
  "topk_10"
)

NEW_TDNCNN_RUNS=(
  "topk_5"
  "topk_15"
  "quantile_5"
  "quantile_10"
  "quantile_15"
)

fail() {
  local message="$1"
  echo
  echo "DOWNSTREAM GRID FAILED"
  echo "${message}"
  exit 1
}

run_step() {
  local name="$1"
  shift
  echo
  echo "===== ${name} ====="
  "$@"
  local status=$?
  if [[ ${status} -ne 0 ]]; then
    fail "Step failed with exit code ${status}: ${name}"
  fi
  echo "===== ${name}: OK ====="
}

assert_file() {
  local path="$1"
  [[ -f "${path}" ]] || fail "Missing expected file: ${path}"
}

assert_dir() {
  local path="$1"
  [[ -d "${path}" ]] || fail "Missing expected directory: ${path}"
}

print_plan() {
  cat <<EOF
ImageNet-100 downstream grid completion

PROJECT_ROOT=${PROJECT_ROOT}
IMAGENET100_ROOT=${IMAGENET100_ROOT}
FORCE=${FORCE}

This script reuses existing production artifacts and runs only missing TDnCNN downstream experiments.

Reused artifacts:
  ${FILTERING_ROOT}/topk/5/selected_indices.npy
  ${FILTERING_ROOT}/topk/10/selected_indices.npy
  ${FILTERING_ROOT}/topk/15/selected_indices.npy
  ${FILTERING_ROOT}/quantile/5/selected_indices.npy
  ${FILTERING_ROOT}/quantile/10/selected_indices.npy
  ${FILTERING_ROOT}/quantile/15/selected_indices.npy
  ${TDNCNN_CKPT_ROOT}/full.pth
  ${TDNCNN_CKPT_ROOT}/topk_10.pth
  ${TDNCNN_ROOT}/full/
  ${TDNCNN_ROOT}/topk_10/

New TDnCNN run names:
  topk_5
  topk_15
  quantile_5
  quantile_10
  quantile_15

Expected new artifact directories:
  ${TDNCNN_ROOT}/topk_5/
  ${TDNCNN_ROOT}/topk_15/
  ${TDNCNN_ROOT}/quantile_5/
  ${TDNCNN_ROOT}/quantile_10/
  ${TDNCNN_ROOT}/quantile_15/

Expected new checkpoints:
  ${TDNCNN_CKPT_ROOT}/topk_5.pth
  ${TDNCNN_CKPT_ROOT}/topk_15.pth
  ${TDNCNN_CKPT_ROOT}/quantile_5.pth
  ${TDNCNN_CKPT_ROOT}/quantile_10.pth
  ${TDNCNN_CKPT_ROOT}/quantile_15.pth

Estimated remaining duration on Tesla V100-SXM2-32GB:
  topk_5:      ~15-45 min
  topk_15:     ~30-90 min
  quantile_5:  ~15-45 min
  quantile_10: ~20-60 min
  quantile_15: ~30-90 min
  research plots: ~2-15 min
  total: ~2-6 hours

This script does not run encoder training, DDPM training, score validation, or filtering.
It uses TDnCNN production defaults from the current project config.
EOF
}

check_dataset() {
  assert_dir "${IMAGENET100_ROOT}/train"
  assert_dir "${IMAGENET100_ROOT}/val"
}

check_filtering_subsets() {
  assert_dir "${FILTERING_ROOT}"
  for subset in "${REQUIRED_FILTERING_SUBSETS[@]}"; do
    assert_file "${FILTERING_ROOT}/${subset}/selected_indices.npy"
    assert_file "${FILTERING_ROOT}/${subset}/metadata.json"
  done
}

check_reused_tdncnn_artifacts() {
  for run_name in "${REUSED_TDNCNN_RUNS[@]}"; do
    assert_file "${TDNCNN_CKPT_ROOT}/${run_name}.pth"
    assert_file "${TDNCNN_ROOT}/${run_name}/metrics.json"
    assert_file "${TDNCNN_ROOT}/${run_name}/results/per_image_metrics.csv"
  done
}

check_tdncnn_run_definitions() {
  local available
  available="$(python scripts/train_tdncnn.py --dataset imagenet100 --list-runs)"
  for run_name in "${REUSED_TDNCNN_RUNS[@]}" "${NEW_TDNCNN_RUNS[@]}"; do
    if ! grep -Fxq "${run_name}" <<<"${available}"; then
      echo "Available TDnCNN runs:"
      echo "${available}"
      fail "TDnCNN run is not defined: ${run_name}"
    fi
  done
}

run_is_complete() {
  local run_name="$1"
  [[ -f "${TDNCNN_CKPT_ROOT}/${run_name}.pth" && -f "${TDNCNN_ROOT}/${run_name}/metrics.json" ]]
}

run_has_partial_artifacts() {
  local run_name="$1"
  [[ -e "${TDNCNN_CKPT_ROOT}/${run_name}.pth" || -e "${TDNCNN_ROOT}/${run_name}" ]]
}

train_missing_tdncnn_run() {
  local run_name="$1"

  if run_is_complete "${run_name}"; then
    echo "Reusing existing completed TDnCNN run: ${run_name}"
    return
  fi

  if run_has_partial_artifacts "${run_name}" && [[ "${FORCE}" != "1" ]]; then
    fail "Partial TDnCNN artifacts exist for ${run_name}. Set FORCE=1 to allow overwrite/reuse by the training script."
  fi

  python scripts/train_tdncnn.py --dataset imagenet100 --run "${run_name}"
}

check_new_tdncnn_artifacts() {
  for run_name in "${NEW_TDNCNN_RUNS[@]}"; do
    assert_file "${TDNCNN_CKPT_ROOT}/${run_name}.pth"
    assert_file "${TDNCNN_ROOT}/${run_name}/metrics.json"
    assert_file "${TDNCNN_ROOT}/${run_name}/results/per_image_metrics.csv"
  done
}

check_research_artifacts() {
  local comparison_dir="${TDNCNN_ROOT}/comparison_plots/research_style"
  assert_dir "${comparison_dir}"
  assert_file "${comparison_dir}/quality_gap_summary.csv"
  assert_file "${comparison_dir}/quality_gap_summary.md"
}

print_plan

run_step "Check ImageNet-100 dataset layout" check_dataset
run_step "Check existing filtering subsets" check_filtering_subsets
run_step "Check reused TDnCNN full/topk_10 artifacts" check_reused_tdncnn_artifacts
run_step "Check TDnCNN run definitions" check_tdncnn_run_definitions

for run_name in "${NEW_TDNCNN_RUNS[@]}"; do
  run_step "TDnCNN ${run_name}" train_missing_tdncnn_run "${run_name}"
done
run_step "Check new TDnCNN grid artifacts" check_new_tdncnn_artifacts

run_step "Regenerate research plots" \
  python scripts/generate_research_plots.py --dataset imagenet100 --stage all --strict false
run_step "Check downstream research artifacts" check_research_artifacts

cat <<EOF

DOWNSTREAM GRID PASSED

Completed publication-ready downstream grid:
  full
  topk_5
  topk_10
  topk_15
  quantile_5
  quantile_10
  quantile_15

Artifacts:
  ${TDNCNN_CKPT_ROOT}/
  ${TDNCNN_ROOT}/
EOF
