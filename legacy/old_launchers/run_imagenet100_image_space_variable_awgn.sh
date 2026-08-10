#!/usr/bin/env bash
set -Eeuo pipefail

CURRENT_STAGE="init"
on_error() {
  local status=$?
  echo "IMAGE-SPACE VARIABLE-AWGN RUN FAILED stage=${CURRENT_STAGE} line=${BASH_LINENO[0]} status=${status}" >&2
  exit "${status}"
}
trap on_error ERR

PROJECT_ROOT="${PROJECT_ROOT:-/workspace}"
DATA_ROOT="${DATA_ROOT:-/workspace/data/imagenet100}"
OUTPUT_BASE="${OUTPUT_BASE:-/workspace/experiments}"
RUN_ID="${RUN_ID:-imagenet100_image_space_awgn_$(date +%Y%m%d_%H%M%S)}"
CONFIG_PATH="${CONFIG_PATH:-configs/experiments/imagenet100_image_space_variable_awgn_v1.yaml}"
EXTERNAL_BENCHMARK_ROOT="${EXTERNAL_BENCHMARK_ROOT:-/workspace/data/external_benchmarks}"
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
REAL_SMOKE="${REAL_SMOKE:-0}"
if [[ "${REAL_SMOKE}" == "1" && "${RUN_ID}" != *_real_smoke ]]; then
  RUN_ID="${RUN_ID}_real_smoke"
fi
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${OUTPUT_BASE}/imagenet100_image_space_variable_awgn_v1/${RUN_ID}}"
PIPELINE_STAGE="${PIPELINE_STAGE:-all}"
START_STAGE="${START_STAGE:-}"
STOP_AFTER_STAGE="${STOP_AFTER_STAGE:-}"

STAGES=(
  preflight
  prepare
  ddpm_train
  score_validate
  score_select
  filter
  downstream_full
  downstream_topk
  downstream_quantile
  downstream_random
  evaluate
  external_eval
  summary
)

cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

LOG_DIR="${EXPERIMENT_ROOT}/00_protocol/logs"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/launcher.log") 2>&1

fail() {
  echo "IMAGE-SPACE VARIABLE-AWGN RUN FAILED: $*" >&2
  exit 1
}

contains_stage() {
  local needle="$1"
  local stage
  for stage in "${STAGES[@]}"; do
    [[ "${stage}" == "${needle}" ]] && return 0
  done
  return 1
}

stage_rel_dir() {
  case "$1" in
    prepare) echo "01_corpus" ;;
    preflight) echo "00_protocol/preflight" ;;
    ddpm_train) echo "02_ddpm" ;;
    score_validate) echo "03_score_validation" ;;
    score_select) echo "04_selected_score_config" ;;
    filter) echo "05_filtering" ;;
    downstream_full) echo "06_downstream/full" ;;
    downstream_topk) echo "06_downstream/topk_10" ;;
    downstream_quantile) echo "06_downstream/quantile_10" ;;
    downstream_random) echo "06_downstream/random_10" ;;
    evaluate) echo "07_evaluation" ;;
    external_eval) echo "08_external_benchmarks" ;;
    summary) echo "09_summary" ;;
    *) fail "unknown stage $1" ;;
  esac
}

required_artifact() {
  case "$1" in
    prepare) echo "${EXPERIMENT_ROOT}/01_corpus/corpus_manifest.csv" ;;
    preflight) echo "${EXPERIMENT_ROOT}/00_protocol/preflight/preflight_report.json" ;;
    ddpm_train) echo "${EXPERIMENT_ROOT}/02_ddpm/training_history.csv" ;;
    score_validate) echo "${EXPERIMENT_ROOT}/03_score_validation/candidate_metrics.csv" ;;
    score_select) echo "${EXPERIMENT_ROOT}/04_selected_score_config/selected_score_config.yaml" ;;
    filter) echo "${EXPERIMENT_ROOT}/05_filtering/scores_full.csv" ;;
    downstream_full) echo "${EXPERIMENT_ROOT}/06_downstream/full/best.pt" ;;
    downstream_topk) echo "${EXPERIMENT_ROOT}/06_downstream/topk_10/best.pt" ;;
    downstream_quantile) echo "${EXPERIMENT_ROOT}/06_downstream/quantile_10/best.pt" ;;
    downstream_random) echo "${EXPERIMENT_ROOT}/06_downstream/random_10/best.pt" ;;
    evaluate) echo "${EXPERIMENT_ROOT}/07_evaluation/summary_table.csv" ;;
    external_eval) echo "${EXPERIMENT_ROOT}/08_external_benchmarks/combined_table.csv" ;;
    summary) echo "${EXPERIMENT_ROOT}/09_summary/final_summary.md" ;;
  esac
}

stage_complete() {
  local stage="$1"
  local marker="${EXPERIMENT_ROOT}/$(stage_rel_dir "${stage}")/_SUCCESS.json"
  local artifact
  artifact="$(required_artifact "${stage}")"
  [[ -f "${marker}" && -f "${artifact}" ]]
}

selected_ok() {
  [[ -f "${EXPERIMENT_ROOT}/04_selected_score_config/_SUCCESS.json" && -f "${EXPERIMENT_ROOT}/04_selected_score_config/selected_score_config.yaml" ]]
}

preflight() {
  CURRENT_STAGE="preflight"
  echo "ImageNet-100 image-space variable-AWGN launcher"
  echo "PROJECT_ROOT=${PROJECT_ROOT}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "OUTPUT_BASE=${OUTPUT_BASE}"
  echo "EXPERIMENT_ROOT=${EXPERIMENT_ROOT}"
  echo "CONFIG_PATH=${CONFIG_PATH}"
  echo "EXTERNAL_BENCHMARK_ROOT=${EXTERNAL_BENCHMARK_ROOT}"
  echo "RUN_ID=${RUN_ID}"
  echo "PIPELINE_STAGE=${PIPELINE_STAGE}"
  echo "START_STAGE=${START_STAGE}"
  echo "STOP_AFTER_STAGE=${STOP_AFTER_STAGE}"
  echo "RESUME=${RESUME}"
  echo "DRY_RUN=${DRY_RUN}"
  echo "SMOKE=${SMOKE}"
  echo "REAL_SMOKE=${REAL_SMOKE}"

  [[ -f "${CONFIG_PATH}" ]] || fail "missing config: ${CONFIG_PATH}"
  bash -n "$0"
  [[ "${EXPERIMENT_ROOT}" != *"/experiments/imagenet100/exp_"* ]] || fail "old ImageNet experiment root is not allowed: ${EXPERIMENT_ROOT}"
  [[ "${CONFIG_PATH}" != *"train_latent_DDPM"* ]] || fail "latent DDPM config is not allowed"

  python - <<'PY'
import importlib
mods = [
    "torch",
    "numpy",
    "pandas",
    "src.imagenet100_image_space_variable_awgn",
]
for name in mods:
    importlib.import_module(name)
print("python imports: ok")
PY

  local workload_cmd=(
    python scripts/imagenet100_image_space_variable_awgn.py
    --config "${CONFIG_PATH}"
    --run-id "${RUN_ID}"
    --experiment-root "${EXPERIMENT_ROOT}"
    --stage preflight
    --data-root "${DATA_ROOT}"
    --external-benchmark-root "${EXTERNAL_BENCHMARK_ROOT}"
    --dry-run
  )
  [[ "${SMOKE}" == "1" ]] && workload_cmd+=(--smoke)
  [[ "${REAL_SMOKE}" == "1" ]] && workload_cmd+=(--real-smoke)
  echo "Resolved workload preview:"
  "${workload_cmd[@]}"

  if [[ "${SMOKE}" != "1" && "${DRY_RUN}" != "1" ]]; then
    [[ -d "${DATA_ROOT}/train" ]] || fail "missing train split: ${DATA_ROOT}/train"
    [[ -d "${DATA_ROOT}/val" ]] || fail "missing val split: ${DATA_ROOT}/val"
    python - <<PY
from pathlib import Path
import torch
root = Path("${DATA_ROOT}")
train_classes = sorted(p for p in (root / "train").iterdir() if p.is_dir())
val_classes = sorted(p for p in (root / "val").iterdir() if p.is_dir())
print("train classes:", len(train_classes))
print("val classes:", len(val_classes))
train_count = sum(1 for p in (root / "train").rglob("*") if p.is_file())
val_count = sum(1 for p in (root / "val").rglob("*") if p.is_file())
print("train samples:", train_count)
print("val samples:", val_count)
if len(train_classes) != 100 or len(val_classes) != 100:
    raise SystemExit(f"Expected 100 classes, got train={len(train_classes)} val={len(val_classes)}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for full mode")
print("cuda:", torch.cuda.get_device_name(0))
PY
  fi

  if [[ "${PIPELINE_STAGE}" == "external_eval" || "${START_STAGE}" == "external_eval" || "${PIPELINE_STAGE}" == "all" ]]; then
    if [[ "${SMOKE}" != "1" && "${DRY_RUN}" != "1" ]]; then
      [[ -d "${EXTERNAL_BENCHMARK_ROOT}" ]] || fail "missing external benchmark root: ${EXTERNAL_BENCHMARK_ROOT}"
    fi
  fi

  df -h "${OUTPUT_BASE}" || true
}

selected_stages() {
  if [[ "${PIPELINE_STAGE}" != "all" ]]; then
    contains_stage "${PIPELINE_STAGE}" || fail "unsupported PIPELINE_STAGE=${PIPELINE_STAGE}"
    echo "${PIPELINE_STAGE}"
    return
  fi

  local emitting=1
  if [[ -n "${START_STAGE}" ]]; then
    contains_stage "${START_STAGE}" || fail "unsupported START_STAGE=${START_STAGE}"
    emitting=0
  fi
  local stage
  for stage in "${STAGES[@]}"; do
    if [[ "${emitting}" == "0" && "${stage}" == "${START_STAGE}" ]]; then
      emitting=1
    fi
    if [[ "${emitting}" == "1" ]]; then
      echo "${stage}"
    fi
    if [[ -n "${STOP_AFTER_STAGE}" && "${stage}" == "${STOP_AFTER_STAGE}" ]]; then
      break
    fi
  done
}

run_stage() {
  local stage="$1"
  CURRENT_STAGE="${stage}"
  if [[ "${RESUME}" == "1" && "${DRY_RUN}" != "1" ]] && stage_complete "${stage}"; then
    echo "[resume] skipping completed stage ${stage}"
    return
  fi
  if [[ "${stage}" == "filter" || "${stage}" == downstream_* ]]; then
    if [[ "${DRY_RUN}" != "1" ]] && ! selected_ok; then
      fail "${stage} is blocked until selected_score_config.yaml exists with successful score_select marker"
    fi
  fi
  local cmd=(
    python scripts/imagenet100_image_space_variable_awgn.py
    --config "${CONFIG_PATH}"
    --run-id "${RUN_ID}"
    --experiment-root "${EXPERIMENT_ROOT}"
    --stage "${stage}"
    --data-root "${DATA_ROOT}"
    --external-benchmark-root "${EXTERNAL_BENCHMARK_ROOT}"
  )
  [[ "${RESUME}" == "1" ]] && cmd+=(--resume)
  [[ "${DRY_RUN}" == "1" ]] && cmd+=(--dry-run)
  [[ "${SMOKE}" == "1" ]] && cmd+=(--smoke)
  [[ "${REAL_SMOKE}" == "1" ]] && cmd+=(--real-smoke)
  echo
  echo "===== ${stage} ====="
  echo "${cmd[*]}"
  "${cmd[@]}"
  if [[ "${DRY_RUN}" != "1" ]] && ! stage_complete "${stage}"; then
    fail "stage ${stage} finished without valid marker/artifact"
  fi
  echo "===== ${stage}: OK ====="
}

preflight

for stage in $(selected_stages); do
  run_stage "${stage}"
done

echo
echo "IMAGE-SPACE VARIABLE-AWGN RUN PASSED"
echo "RUN_ID=${RUN_ID}"
echo "EXPERIMENT_ROOT=${EXPERIMENT_ROOT}"
echo "selected_score_config=${EXPERIMENT_ROOT}/04_selected_score_config/selected_score_config.yaml"
echo "scores_full=${EXPERIMENT_ROOT}/05_filtering/scores_full.csv"
echo "final_summary=${EXPERIMENT_ROOT}/09_summary/final_summary.md"
