#!/usr/bin/env bash

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace}"
DATASET_ROOT="${IMAGENET100_ROOT:-/workspace/data/imagenet100}"
RUN_ID="${RUN_ID:-full_$(date +%Y%m%d_%H%M%S)}"
FORCE="${FORCE:-0}"

cd "${PROJECT_ROOT}" || {
  echo "FULL RUN FAILED"
  echo "Cannot cd to PROJECT_ROOT=${PROJECT_ROOT}"
  exit 1
}

export IMAGENET100_ROOT="${DATASET_ROOT}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

BASELINE_RUN="latent_ddpm_baseline_ae_noise_consistency_imagenet100_${RUN_ID}"
INDUCED_RUN="latent_ddpm_induced_ae_noise_consistency_imagenet100_${RUN_ID}"
BASELINE_DDPM_DIR="checkpoints/imagenet100/ddpm/${BASELINE_RUN}"
INDUCED_DDPM_DIR="checkpoints/imagenet100/ddpm/${INDUCED_RUN}"

ENCODER_DIR="checkpoints/imagenet100/autoencoders/noise_consistency_large_latent256"
ENCODER_OUTPUT_DIR="outputs/imagenet100/autoencoders/noise_consistency_large_latent256"
ENCODER_CKPT="${ENCODER_DIR}/autoencoder_checkpoint.pt"
ENCODER_STATE="${ENCODER_DIR}/E.pt"

ENCODER_VALIDATION_ROOT="experiments/imagenet100/exp_002_encoder_validation"
SCORE_VALIDATION_ROOT="experiments/imagenet100/exp_003_latent_ddpm_validation"
FILTERING_ROOT="experiments/imagenet100/exp_005_filtering"
TDNCNN_ROOT="experiments/imagenet100/exp_006_tdncnn"

CANONICAL_EXPERIMENT_DIRS=(
  "${ENCODER_VALIDATION_ROOT}"
  "${SCORE_VALIDATION_ROOT}"
  "${FILTERING_ROOT}"
  "${TDNCNN_ROOT}"
)

fail() {
  local message="$1"
  echo
  echo "FULL RUN FAILED"
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

assert_any_file_glob() {
  local pattern="$1"
  compgen -G "${pattern}" >/dev/null || fail "Missing expected file matching: ${pattern}"
}

print_plan() {
  cat <<EOF
ImageNet-100 production full run

PROJECT_ROOT=${PROJECT_ROOT}
IMAGENET100_ROOT=${IMAGENET100_ROOT}
RUN_ID=${RUN_ID}
FORCE=${FORCE}
GPU target: Tesla V100-SXM2-32GB

WARNING:
  This script uses canonical experiment directories:
    ${ENCODER_VALIDATION_ROOT}
    ${SCORE_VALIDATION_ROOT}
    ${FILTERING_ROOT}
    ${TDNCNN_ROOT}

  If any canonical experiment directory already exists, the run fails fast unless FORCE=1.
  FORCE=1 allows the run to proceed but does not delete or clean existing directories.
  Existing files may be overwritten by individual pipeline stages, and some stages may still
  refuse non-empty output directories according to their own internal guards.

Pipeline:
  1. GPU sanity check
  2. Dataset sanity check
  3. Encoder Training
     Command: python scripts/train_encoder.py noise-consistency --dataset imagenet100
     Expected duration on Tesla V100 32GB: ~2-5 hours
  4. Encoder Validation
     Command: python scripts/evaluate_encoder.py compare-encoders --dataset imagenet100
     Expected duration: ~5-20 minutes
  5. Latent DDPM Baseline
     Command: python scripts/main.py dataset=imagenet100 task=train_latent_DDPM train_latent_DDPM.latent_noise_mode=baseline train_latent_DDPM.id_label=_${RUN_ID}
     Expected duration: ~4-10 hours
  6. Latent DDPM Induced
     Command: python scripts/main.py dataset=imagenet100 task=train_latent_DDPM train_latent_DDPM.latent_noise_mode=induced train_latent_DDPM.id_label=_${RUN_ID}
     Expected duration: ~6-14 hours
  7. Score Validation
     Command: python scripts/evaluate_latent_ddpm_score.py --dataset imagenet100 --modes baseline induced --baseline-run-dir ${BASELINE_DDPM_DIR} --induced-run-dir ${INDUCED_DDPM_DIR}
     Expected duration: ~5-20 minutes
  8. Filtering
     Command: python scripts/main.py dataset=imagenet100 task=filter_dataset filter_dataset.ddpm_branch=induced dataset.latent_ddpm_validation.runs.induced.checkpoint_run=${INDUCED_RUN}
     Expected duration: ~20-90 minutes
  9. TDnCNN full
     Command: python scripts/train_tdncnn.py --dataset imagenet100 --run full
     Expected duration: ~1-4 hours
  10. TDnCNN topk_10
      Command: python scripts/train_tdncnn.py --dataset imagenet100 --run topk_10
      Expected duration: ~20-60 minutes
  11. Research Plots
      Command: python scripts/generate_research_plots.py --dataset imagenet100 --stage all --strict false
      Expected duration: ~2-15 minutes

Artifact directories:
  ${ENCODER_DIR}
  ${ENCODER_OUTPUT_DIR}
  ${BASELINE_DDPM_DIR}
  ${INDUCED_DDPM_DIR}
  checkpoints/imagenet100/tdncnn
  ${ENCODER_VALIDATION_ROOT}
  ${SCORE_VALIDATION_ROOT}
  ${FILTERING_ROOT}
  ${TDNCNN_ROOT}

Expected checkpoints:
  ${ENCODER_CKPT}
  ${ENCODER_STATE}
  ${BASELINE_DDPM_DIR}/best_model.pth or ${BASELINE_DDPM_DIR}/epoch_*.pth
  ${INDUCED_DDPM_DIR}/best_model.pth or ${INDUCED_DDPM_DIR}/epoch_*.pth
  checkpoints/imagenet100/tdncnn/full.pth
  checkpoints/imagenet100/tdncnn/topk_10.pth

Differences from DGX smoke:
  - uses real ImageNet-100 only
  - no --fast_dev_run
  - no --epochs 1
  - no max_samples limits
  - no topk_10_smoke
  - no batch_size=2 override
  - no num_workers=0 override
  - uses production defaults from configs
  - filtering runs the configured top_k/quantile percentage sweep
  - TDnCNN runs production full and topk_10
EOF
}

guard_canonical_dirs() {
  local existing=()
  for path in "${CANONICAL_EXPERIMENT_DIRS[@]}"; do
    if [[ -e "${path}" ]]; then
      existing+=("${path}")
    fi
  done

  if [[ ${#existing[@]} -gt 0 && "${FORCE}" != "1" ]]; then
    echo
    echo "Canonical experiment directories already exist:"
    printf '  %s\n' "${existing[@]}"
    echo
    echo "Refusing to start production run to avoid mixing or overwriting artifacts."
    echo "Set FORCE=1 to allow the run to proceed without deleting anything."
    exit 1
  fi

  if [[ ${#existing[@]} -gt 0 && "${FORCE}" == "1" ]]; then
    echo
    echo "FORCE=1 is set. Existing canonical experiment directories will be reused:"
    printf '  %s\n' "${existing[@]}"
    echo "No directories will be deleted automatically."
  fi
}

gpu_sanity_check() {
  python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print("cuda device count:", torch.cuda.device_count())
print("cuda device 0:", torch.cuda.get_device_name(0))
x = torch.ones(1, device="cuda")
print("cuda tensor check:", float(x.item()))
PY
}

dataset_sanity_check() {
  python - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["IMAGENET100_ROOT"])
train = root / "train"
val = root / "val"
print("IMAGENET100_ROOT:", root)
print("train:", train)
print("val:", val)
if not train.is_dir():
    raise SystemExit(f"Missing train directory: {train}")
if not val.is_dir():
    raise SystemExit(f"Missing val directory: {val}")
train_classes = sorted(p for p in train.iterdir() if p.is_dir())
val_classes = sorted(p for p in val.iterdir() if p.is_dir())
print("train classes:", len(train_classes))
print("val classes:", len(val_classes))
if not train_classes:
    raise SystemExit("No train class directories found")
if not val_classes:
    raise SystemExit("No val class directories found")
PY
}

check_encoder_artifacts() {
  assert_file "${ENCODER_CKPT}"
  assert_file "${ENCODER_STATE}"
  assert_file "${ENCODER_OUTPUT_DIR}/metrics.json"
  assert_file "${ENCODER_OUTPUT_DIR}/reconstruction_grid.png"
}

check_encoder_validation_artifacts() {
  assert_dir "${ENCODER_VALIDATION_ROOT}"
  assert_file "${ENCODER_VALIDATION_ROOT}/summary.json"
  assert_file "${ENCODER_VALIDATION_ROOT}/report/encoder_validation_report.md"
}

check_ddpm_artifacts() {
  local run_dir="$1"
  assert_dir "${run_dir}"
  if [[ ! -f "${run_dir}/best_model.pth" ]] && ! compgen -G "${run_dir}/epoch_*.pth" >/dev/null; then
    fail "Missing DDPM checkpoint in: ${run_dir}"
  fi
  assert_file "${run_dir}/DDPM_metrics.csv"
}

check_score_validation_artifacts() {
  assert_dir "${SCORE_VALIDATION_ROOT}"
  assert_file "${SCORE_VALIDATION_ROOT}/summary.json"
  assert_file "${SCORE_VALIDATION_ROOT}/report/latent_ddpm_validation_report.md"
  assert_file "${SCORE_VALIDATION_ROOT}/metrics/summary.csv"
}

check_filtering_artifacts() {
  assert_dir "${FILTERING_ROOT}"
  assert_file "${FILTERING_ROOT}/summary.json"
  assert_file "${FILTERING_ROOT}/topk/10/selected_indices.npy"
  assert_file "${FILTERING_ROOT}/topk/10/scores.csv"
  assert_file "${FILTERING_ROOT}/topk/10/metadata.json"
}

check_tdncnn_artifacts() {
  assert_dir "${TDNCNN_ROOT}"
  assert_file "checkpoints/imagenet100/tdncnn/full.pth"
  assert_file "checkpoints/imagenet100/tdncnn/topk_10.pth"
  assert_file "${TDNCNN_ROOT}/full/metrics.json"
  assert_file "${TDNCNN_ROOT}/topk_10/metrics.json"
}

check_research_plot_artifacts() {
  local filtering_research_dir="${FILTERING_ROOT}/plots/research_style"
  local ddpm_research_dir="${SCORE_VALIDATION_ROOT}/plots/research_style"
  local tdncnn_research_dir="${TDNCNN_ROOT}/comparison_plots/research_style"

  if [[ -f "${filtering_research_dir}/filtering_research_plots_manifest.json" ]]; then
    echo "Found filtering research manifest: ${filtering_research_dir}/filtering_research_plots_manifest.json"
    return
  fi

  if [[ -d "${filtering_research_dir}" || -d "${ddpm_research_dir}" || -d "${tdncnn_research_dir}" ]]; then
    echo "Found at least one research_style directory:"
    [[ -d "${filtering_research_dir}" ]] && echo "  ${filtering_research_dir}"
    [[ -d "${ddpm_research_dir}" ]] && echo "  ${ddpm_research_dir}"
    [[ -d "${tdncnn_research_dir}" ]] && echo "  ${tdncnn_research_dir}"
    return
  fi

  fail "Missing research plots manifest and research_style directories"
}

print_plan
guard_canonical_dirs

run_step "GPU sanity check" gpu_sanity_check
run_step "Dataset sanity check" dataset_sanity_check

run_step "Encoder training" \
  python scripts/train_encoder.py noise-consistency --dataset imagenet100
run_step "Check encoder artifacts" check_encoder_artifacts

run_step "Encoder validation" \
  python scripts/evaluate_encoder.py compare-encoders --dataset imagenet100
run_step "Check encoder validation artifacts" check_encoder_validation_artifacts

run_step "Latent DDPM baseline" \
  python scripts/main.py \
    dataset=imagenet100 \
    task=train_latent_DDPM \
    train_latent_DDPM.latent_noise_mode=baseline \
    train_latent_DDPM.id_label="_${RUN_ID}"
run_step "Check baseline DDPM artifacts" check_ddpm_artifacts "${BASELINE_DDPM_DIR}"

run_step "Latent DDPM induced" \
  python scripts/main.py \
    dataset=imagenet100 \
    task=train_latent_DDPM \
    train_latent_DDPM.latent_noise_mode=induced \
    train_latent_DDPM.id_label="_${RUN_ID}"
run_step "Check induced DDPM artifacts" check_ddpm_artifacts "${INDUCED_DDPM_DIR}"

run_step "Score validation" \
  python scripts/evaluate_latent_ddpm_score.py \
    --dataset imagenet100 \
    --modes baseline induced \
    --baseline-run-dir "${BASELINE_DDPM_DIR}" \
    --induced-run-dir "${INDUCED_DDPM_DIR}"
run_step "Check score validation artifacts" check_score_validation_artifacts

run_step "Filtering" \
  python scripts/main.py \
    dataset=imagenet100 \
    task=filter_dataset \
    filter_dataset.ddpm_branch=induced \
    dataset.latent_ddpm_validation.runs.induced.checkpoint_run="${INDUCED_RUN}"
run_step "Check filtering artifacts" check_filtering_artifacts

run_step "TDnCNN full" \
  python scripts/train_tdncnn.py --dataset imagenet100 --run full

run_step "TDnCNN topk_10" \
  python scripts/train_tdncnn.py --dataset imagenet100 --run topk_10
run_step "Check TDnCNN artifacts" check_tdncnn_artifacts

run_step "Research plots" \
  python scripts/generate_research_plots.py --dataset imagenet100 --stage all --strict false
run_step "Check research plot artifacts" check_research_plot_artifacts

cat <<EOF

FULL RUN PASSED

RUN_ID=${RUN_ID}

Final artifact directories:
  ${ENCODER_DIR}
  ${ENCODER_OUTPUT_DIR}
  ${BASELINE_DDPM_DIR}
  ${INDUCED_DDPM_DIR}
  checkpoints/imagenet100/tdncnn
  ${ENCODER_VALIDATION_ROOT}
  ${SCORE_VALIDATION_ROOT}
  ${FILTERING_ROOT}
  ${TDNCNN_ROOT}
EOF
