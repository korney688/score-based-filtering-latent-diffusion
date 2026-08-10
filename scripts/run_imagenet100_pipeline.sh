#!/usr/bin/env bash
set -Eeuo pipefail

CURRENT_STAGE="init"
on_error() {
  local status=$?
  echo "IMAGENET100 PIPELINE FAILED stage=${CURRENT_STAGE} line=${BASH_LINENO[0]} status=${status}" >&2
  exit "${status}"
}
trap on_error ERR

PROJECT_ROOT="${PROJECT_ROOT:-/workspace}"
DATA_ROOT="${DATA_ROOT:-/workspace/data/imagenet100}"
OUTPUT_BASE="${OUTPUT_BASE:-/workspace/experiments}"
RUN_ID="${RUN_ID:-imagenet100_image_space_awgn_v1_prod_20260809_01}"
CONFIG_PATH="${CONFIG_PATH:-configs/experiments/imagenet100.yaml}"
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
REAL_SMOKE="${REAL_SMOKE:-0}"
PIPELINE_STAGE="${PIPELINE_STAGE:-all}"
START_STAGE="${START_STAGE:-}"
STOP_AFTER_STAGE="${STOP_AFTER_STAGE:-}"

if [[ "${REAL_SMOKE}" == "1" && "${RUN_ID}" != *_real_smoke ]]; then
  RUN_ID="${RUN_ID}_real_smoke"
fi

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${OUTPUT_BASE}/imagenet100/${RUN_ID}}"

STAGES=(
  prepare
  ddpm_train
  score_validate
  score_select
  filter
  tdncnn_full
  tdncnn_topk
  tdncnn_quantile
  tdncnn_random
  tdncnn_evaluate
  drunet_full
  drunet_topk
  drunet_quantile
  drunet_random
  drunet_evaluate
  summary
)

cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

LOG_DIR="${EXPERIMENT_ROOT}/00_protocol/logs"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/launcher_full.log") 2>&1

fail() {
  echo "IMAGENET100 PIPELINE FAILED: $*" >&2
  exit 1
}

contains_stage() {
  local needle="$1"
  local stage
  [[ "${needle}" == "all" || "${needle}" == "preflight" ]] && return 0
  for stage in "${STAGES[@]}"; do
    [[ "${stage}" == "${needle}" ]] && return 0
  done
  return 1
}

stage_rel_dir() {
  case "$1" in
    prepare) echo "01_corpus" ;;
    ddpm_train) echo "02_ddpm" ;;
    score_validate) echo "03_score_validation" ;;
    score_select) echo "04_selected_score_config" ;;
    filter) echo "05_filtering" ;;
    tdncnn_full) echo "06_downstream/tdncnn/full" ;;
    tdncnn_topk) echo "06_downstream/tdncnn/topk_10" ;;
    tdncnn_quantile) echo "06_downstream/tdncnn/quantile_10" ;;
    tdncnn_random) echo "06_downstream/tdncnn/random_10" ;;
    tdncnn_evaluate) echo "07_evaluation/tdncnn" ;;
    drunet_full) echo "06_downstream/drunet/full" ;;
    drunet_topk) echo "06_downstream/drunet/topk_10" ;;
    drunet_quantile) echo "06_downstream/drunet/quantile_10" ;;
    drunet_random) echo "06_downstream/drunet/random_10" ;;
    drunet_evaluate) echo "07_evaluation/drunet" ;;
    summary) echo "09_summary" ;;
    preflight) echo "00_protocol/preflight" ;;
    *) fail "unknown stage $1" ;;
  esac
}

required_artifact() {
  case "$1" in
    prepare) echo "${EXPERIMENT_ROOT}/01_corpus/corpus_manifest.csv" ;;
    ddpm_train) echo "${EXPERIMENT_ROOT}/02_ddpm/training_history.csv" ;;
    score_validate) echo "${EXPERIMENT_ROOT}/03_score_validation/candidate_metrics.csv" ;;
    score_select) echo "${EXPERIMENT_ROOT}/04_selected_score_config/selected_score_config.yaml" ;;
    filter) echo "${EXPERIMENT_ROOT}/05_filtering/scores_full.csv" ;;
    tdncnn_full) echo "${EXPERIMENT_ROOT}/06_downstream/tdncnn/full/best.pt" ;;
    tdncnn_topk) echo "${EXPERIMENT_ROOT}/06_downstream/tdncnn/topk_10/best.pt" ;;
    tdncnn_quantile) echo "${EXPERIMENT_ROOT}/06_downstream/tdncnn/quantile_10/best.pt" ;;
    tdncnn_random) echo "${EXPERIMENT_ROOT}/06_downstream/tdncnn/random_10/best.pt" ;;
    tdncnn_evaluate) echo "${EXPERIMENT_ROOT}/07_evaluation/tdncnn/summary_table.csv" ;;
    drunet_full) echo "${EXPERIMENT_ROOT}/06_downstream/drunet/full/best.pt" ;;
    drunet_topk) echo "${EXPERIMENT_ROOT}/06_downstream/drunet/topk_10/best.pt" ;;
    drunet_quantile) echo "${EXPERIMENT_ROOT}/06_downstream/drunet/quantile_10/best.pt" ;;
    drunet_random) echo "${EXPERIMENT_ROOT}/06_downstream/drunet/random_10/best.pt" ;;
    drunet_evaluate) echo "${EXPERIMENT_ROOT}/07_evaluation/drunet/summary_table.csv" ;;
    summary) echo "${EXPERIMENT_ROOT}/09_summary/final_summary.md" ;;
    preflight) echo "${EXPERIMENT_ROOT}/00_protocol/preflight/preflight_report.json" ;;
    *) fail "unknown artifact stage $1" ;;
  esac
}

stage_complete() {
  local stage="$1"
  local marker="${EXPERIMENT_ROOT}/$(stage_rel_dir "${stage}")/_SUCCESS.json"
  local artifact
  artifact="$(required_artifact "${stage}")"
  [[ -f "${marker}" && -f "${artifact}" ]]
}

print_plan() {
  cat <<EOF
ImageNet-100 current pipeline production run

PROJECT_ROOT=${PROJECT_ROOT}
DATA_ROOT=${DATA_ROOT}
OUTPUT_BASE=${OUTPUT_BASE}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT}
CONFIG_PATH=${CONFIG_PATH}
RUN_ID=${RUN_ID}
PIPELINE_STAGE=${PIPELINE_STAGE}
START_STAGE=${START_STAGE}
STOP_AFTER_STAGE=${STOP_AFTER_STAGE}
RESUME=${RESUME}
DRY_RUN=${DRY_RUN}
SMOKE=${SMOKE}
REAL_SMOKE=${REAL_SMOKE}

Stage graph:
  prepare -> ddpm_train -> score_validate -> score_select -> filter
  -> tdncnn_full -> tdncnn_topk -> tdncnn_quantile -> tdncnn_random -> tdncnn_evaluate
  -> drunet_full -> drunet_topk -> drunet_quantile -> drunet_random -> drunet_evaluate
  -> summary
EOF
}

run_python_stage() {
  local stage="$1"
  shift
  local cmd=(
    python scripts/run_imagenet100_pipeline.py
    --config "${CONFIG_PATH}"
    --run-id "${RUN_ID}"
    --experiment-root "${EXPERIMENT_ROOT}"
    --stage "${stage}"
    --data-root "${DATA_ROOT}"
  )
  [[ "${RESUME}" == "1" ]] && cmd+=(--resume)
  [[ "${DRY_RUN}" == "1" ]] && cmd+=(--dry-run)
  [[ "${SMOKE}" == "1" ]] && cmd+=(--smoke)
  [[ "${REAL_SMOKE}" == "1" ]] && cmd+=(--real-smoke)
  cmd+=("$@")
  echo "${cmd[*]}"
  "${cmd[@]}"
}

preflight() {
  CURRENT_STAGE="preflight"
  print_plan
  [[ -f "${CONFIG_PATH}" ]] || fail "missing config: ${CONFIG_PATH}"
  bash -n "$0"
  [[ "${EXPERIMENT_ROOT}" == "${OUTPUT_BASE}/imagenet100/"* ]] || \
    fail "experiment root must stay under ${OUTPUT_BASE}/imagenet100"

  python - <<'PY'
import importlib
for name in ["torch", "numpy", "pandas", "src.imagenet100_pipeline", "src.DRUNet_image", "src.TDnCNN_image"]:
    importlib.import_module(name)
print("python imports: ok")
PY

  echo "Resolved production workload preview:"
  run_python_stage preflight --dry-run

  if [[ "${SMOKE}" != "1" && "${DRY_RUN}" != "1" ]]; then
    [[ -d "${DATA_ROOT}/train" ]] || fail "missing train split: ${DATA_ROOT}/train"
    [[ -d "${DATA_ROOT}/val" ]] || fail "missing val split: ${DATA_ROOT}/val"
    python - <<PY
from pathlib import Path
import torch
root = Path("${DATA_ROOT}")
train_classes = sorted(p for p in (root / "train").iterdir() if p.is_dir())
val_classes = sorted(p for p in (root / "val").iterdir() if p.is_dir())
train_count = sum(1 for p in (root / "train").rglob("*") if p.is_file())
val_count = sum(1 for p in (root / "val").rglob("*") if p.is_file())
print("train classes:", len(train_classes))
print("val classes:", len(val_classes))
print("train samples:", train_count)
print("val samples:", val_count)
if len(train_classes) != 100 or len(val_classes) != 100:
    raise SystemExit(f"Expected exactly 100 classes, got train={len(train_classes)} val={len(val_classes)}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for production run")
print("cuda:", torch.cuda.get_device_name(0))
PY
  fi
  mkdir -p "${OUTPUT_BASE}"
  [[ -w "${OUTPUT_BASE}" ]] || fail "output base is not writable: ${OUTPUT_BASE}"
  df -h "${OUTPUT_BASE}" || true
}

validate_filtering_gate() {
  [[ -f "${EXPERIMENT_ROOT}/03_score_validation/_SUCCESS.json" ]] || fail "missing score_validate success marker"
  [[ -f "${EXPERIMENT_ROOT}/04_selected_score_config/_SUCCESS.json" ]] || fail "missing score_select success marker"
  [[ -f "${EXPERIMENT_ROOT}/04_selected_score_config/selected_score_config.yaml" ]] || fail "missing selected_score_config.yaml"
  [[ -f "${EXPERIMENT_ROOT}/05_filtering/_SUCCESS.json" ]] || fail "missing filter success marker"
  [[ -f "${EXPERIMENT_ROOT}/05_filtering/scores_full.csv" ]] || fail "missing scores_full.csv"
  for name in full topk_10 quantile_10 random_10; do
    [[ -f "${EXPERIMENT_ROOT}/05_filtering/subset_indices/${name}_indices.npy" ]] || fail "missing subset indices: ${name}"
    [[ -f "${EXPERIMENT_ROOT}/05_filtering/subset_manifests/${name}_manifest.csv" ]] || fail "missing subset manifest: ${name}"
  done
  python - <<PY
import json
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
root = Path("${EXPERIMENT_ROOT}")
run_id = "${RUN_ID}"
for rel in ["03_score_validation/_SUCCESS.json", "04_selected_score_config/_SUCCESS.json", "05_filtering/_SUCCESS.json"]:
    payload = json.loads((root / rel).read_text(encoding="utf-8"))
    if payload.get("run_id") != run_id:
        raise SystemExit(f"marker RUN_ID mismatch in {rel}: {payload.get('run_id')} != {run_id}")
selected = OmegaConf.to_container(OmegaConf.load(root / "04_selected_score_config/selected_score_config.yaml"), resolve=True)
if str(selected.get("run_id")) != run_id:
    raise SystemExit("selected_score_config RUN_ID mismatch")
selected_hash = str(selected.get("config_hash", ""))
for rel in ["04_selected_score_config/_SUCCESS.json", "05_filtering/_SUCCESS.json"]:
    marker_hash = str(json.loads((root / rel).read_text(encoding="utf-8")).get("config_hash", ""))
    if selected_hash and marker_hash and marker_hash != selected_hash:
        raise SystemExit(f"config hash mismatch between selected config and {rel}")
sizes = {name: int(np.load(root / "05_filtering" / "subset_indices" / f"{name}_indices.npy").shape[0]) for name in ["topk_10", "quantile_10", "random_10"]}
if len(set(sizes.values())) != 1:
    raise SystemExit(f"filtered subset sizes differ: {sizes}")
print("filtering gate: ok", sizes)
PY
}

selected_stages() {
  if [[ "${PIPELINE_STAGE}" == "preflight" ]]; then
    echo "preflight"
    return
  fi
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
  case "${stage}" in
    tdncnn_*|drunet_*) [[ "${DRY_RUN}" == "1" ]] || validate_filtering_gate ;;
  esac
  if [[ "${stage}" == "preflight" ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
      return
    fi
    run_python_stage preflight
    return
  fi
  if [[ "${RESUME}" == "1" && "${DRY_RUN}" != "1" ]] && stage_complete "${stage}"; then
    echo "[resume] skipping completed stage ${stage}"
    return
  fi
  echo
  echo "===== ${stage} ====="
  run_python_stage "${stage}"
  if [[ "${DRY_RUN}" != "1" ]] && ! stage_complete "${stage}"; then
    fail "stage ${stage} finished without valid marker/artifact"
  fi
  echo "===== ${stage}: OK ====="
}

contains_stage "${PIPELINE_STAGE}" || fail "unsupported PIPELINE_STAGE=${PIPELINE_STAGE}"
preflight

if [[ "${PIPELINE_STAGE}" != "preflight" ]]; then
  for stage in $(selected_stages); do
    run_stage "${stage}"
  done
fi

echo
echo "IMAGE-SPACE VARIABLE-AWGN FULL RUN PASSED"
echo "RUN_ID=${RUN_ID}"
echo "EXPERIMENT_ROOT=${EXPERIMENT_ROOT}"
echo "selected_score_config=${EXPERIMENT_ROOT}/04_selected_score_config/selected_score_config.yaml"
echo "scores_full=${EXPERIMENT_ROOT}/05_filtering/scores_full.csv"
echo "tdncnn_internal=${EXPERIMENT_ROOT}/07_evaluation/tdncnn/summary_table.csv"
echo "drunet_internal=${EXPERIMENT_ROOT}/07_evaluation/drunet/summary_table.csv"
echo "combined_internal=${EXPERIMENT_ROOT}/07_evaluation/combined/internal_combined_table.csv"
echo "final_summary=${EXPERIMENT_ROOT}/09_summary/final_summary.md"
