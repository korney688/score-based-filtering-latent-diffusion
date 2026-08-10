#!/usr/bin/env bash

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace}"
DATASET_ROOT="${IMAGENET100_ROOT:-/workspace/data/imagenet100}"
RUN_ID="${RUN_ID:-imagenet100_noisy_pg_$(date +%Y%m%d_%H%M%S)}"
FORCE="${FORCE:-0}"
PIPELINE_STAGE="${PIPELINE_STAGE:-full}"

cd "${PROJECT_ROOT}" || {
  echo "NOISY FULL RUN FAILED"
  echo "Cannot cd to PROJECT_ROOT=${PROJECT_ROOT}"
  exit 1
}

export IMAGENET100_ROOT="${DATASET_ROOT}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

NOISY_HYDRA_OVERRIDE="dataset.dataset_mode=noisy"
NOISY_CLI_ARGS=(--dataset-mode noisy)

BASELINE_RUN="latent_ddpm_baseline_ae_noise_consistency_imagenet100_${RUN_ID}"
INDUCED_RUN="latent_ddpm_induced_ae_noise_consistency_imagenet100_${RUN_ID}"
BASELINE_DDPM_DIR="checkpoints/imagenet100/ddpm/${BASELINE_RUN}"
INDUCED_DDPM_DIR="checkpoints/imagenet100/ddpm/${INDUCED_RUN}"

ENCODER_DIR="checkpoints/imagenet100/autoencoders/noise_consistency_large_latent256"
ENCODER_OUTPUT_DIR="outputs/imagenet100/autoencoders/noise_consistency_large_latent256"
ENCODER_CKPT="${ENCODER_DIR}/autoencoder_checkpoint.pt"
ENCODER_STATE="${ENCODER_DIR}/E.pt"

ENCODER_VALIDATION_ROOT="experiments/imagenet100/exp_002_encoder_validation_${RUN_ID}"
SCORE_VALIDATION_ROOT="experiments/imagenet100/exp_003_latent_ddpm_validation_${RUN_ID}"
FILTERING_ROOT="experiments/imagenet100/exp_005_filtering_${RUN_ID}"
TDNCNN_ROOT="experiments/imagenet100/exp_006_tdncnn"
TDNCNN_CKPT_ROOT="checkpoints/imagenet100/tdncnn"
DRUNET_ROOT="experiments/imagenet100/exp_007_drunet"
DRUNET_CKPT_ROOT="checkpoints/imagenet100/drunet"
DRUNET_CKPT="${DRUNET_CKPT_ROOT}/full_sigma25.pth"

DOWNSTREAM_FILTERING_ROOT="${DOWNSTREAM_FILTERING_ROOT:-${FILTERING_ROOT}}"
TDNCNN_DOWNSTREAM_ROOT="${TDNCNN_DOWNSTREAM_ROOT:-experiments/imagenet100/exp_006_tdncnn_${RUN_ID}}"
TDNCNN_DOWNSTREAM_CKPT_ROOT="${TDNCNN_DOWNSTREAM_CKPT_ROOT:-checkpoints/imagenet100/tdncnn_${RUN_ID}}"
DRUNET_DOWNSTREAM_ROOT="${DRUNET_DOWNSTREAM_ROOT:-experiments/imagenet100/exp_007_drunet_${RUN_ID}}"
DRUNET_DOWNSTREAM_CKPT_ROOT="${DRUNET_DOWNSTREAM_CKPT_ROOT:-checkpoints/imagenet100/drunet_${RUN_ID}}"

USED_COMMANDS=()

CANONICAL_EXPERIMENT_DIRS=(
  "${ENCODER_VALIDATION_ROOT}"
  "${SCORE_VALIDATION_ROOT}"
  "${FILTERING_ROOT}"
  "${TDNCNN_ROOT}"
  "${TDNCNN_CKPT_ROOT}"
  "${DRUNET_ROOT}"
  "${DRUNET_CKPT_ROOT}"
)

fail() {
  local message="$1"
  echo
  echo "NOISY FULL RUN FAILED"
  echo "${message}"
  exit 1
}

run_step() {
  local name="$1"
  shift
  echo
  echo "===== ${name} ====="
  USED_COMMANDS+=("$*")
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
ImageNet-100 Poisson-Gaussian noisy production full run

PROJECT_ROOT=${PROJECT_ROOT}
IMAGENET100_ROOT=${IMAGENET100_ROOT}
RUN_ID=${RUN_ID}
PIPELINE_STAGE=${PIPELINE_STAGE}
FORCE=${FORCE}
GPU target: Tesla V100-SXM2-32GB

Noisy override:
  Hydra: ${NOISY_HYDRA_OVERRIDE}
  CLI: ${NOISY_CLI_ARGS[*]}

Pipeline:
  1. GPU sanity check
  2. Dataset + noisy corruption sanity check
  3. Encoder validation
     Command: python scripts/evaluate_encoder.py compare-encoders --dataset imagenet100 --dataset-mode noisy --output-root ${ENCODER_VALIDATION_ROOT}
  4. Encoder selection artifact check
     Selected checkpoint: ${ENCODER_CKPT}
  5. Latent DDPM baseline
     Command: python scripts/main.py dataset=imagenet100 dataset.dataset_mode=noisy task=train_latent_DDPM train_latent_DDPM.latent_noise_mode=baseline train_latent_DDPM.id_label=_${RUN_ID}
  6. Latent DDPM encoder-induced
     Command: python scripts/main.py dataset=imagenet100 dataset.dataset_mode=noisy task=train_latent_DDPM train_latent_DDPM.latent_noise_mode=induced train_latent_DDPM.id_label=_${RUN_ID}
  7. Score validation
     Command: python scripts/evaluate_latent_ddpm_score.py --dataset imagenet100 --dataset-mode noisy --modes baseline induced --baseline-run-dir ${BASELINE_DDPM_DIR} --induced-run-dir ${INDUCED_DDPM_DIR} --output-root ${SCORE_VALIDATION_ROOT}
  8. Filtering, induced branch
     Command: python scripts/main.py dataset=imagenet100 dataset.dataset_mode=noisy task=filter_dataset filter_dataset.ddpm_branch=induced dataset.latent_ddpm_validation.runs.induced.checkpoint_run=${INDUCED_RUN} filter_dataset.output_root=${FILTERING_ROOT}
  9. TDnCNN training + ImageNet validation
     Command: python scripts/train_tdncnn.py --dataset imagenet100 --dataset-mode noisy --run full
  10. DRUNet training + ImageNet validation
      Command: python scripts/train_drunet.py --dataset imagenet100 --dataset-mode noisy --run full_sigma25

Artifact directories:
  ${ENCODER_DIR}
  ${ENCODER_OUTPUT_DIR}
  ${BASELINE_DDPM_DIR}
  ${INDUCED_DDPM_DIR}
  ${ENCODER_VALIDATION_ROOT}
  ${SCORE_VALIDATION_ROOT}
  ${FILTERING_ROOT}
  ${TDNCNN_ROOT}
  ${TDNCNN_CKPT_ROOT}
  ${DRUNET_ROOT}
  ${DRUNET_CKPT_ROOT}
EOF
}

print_downstream_only_plan() {
  cat <<EOF
ImageNet-100 Poisson-Gaussian noisy downstream-only run

PROJECT_ROOT=${PROJECT_ROOT}
IMAGENET100_ROOT=${IMAGENET100_ROOT}
RUN_ID=${RUN_ID}
PIPELINE_STAGE=${PIPELINE_STAGE}

This mode skips encoder validation, encoder selection, DDPM training, score validation, filtering, and full downstream.

Noisy filtering outputs:
  ${DOWNSTREAM_FILTERING_ROOT}/topk/10/selected_indices.npy
  ${DOWNSTREAM_FILTERING_ROOT}/quantile/10/selected_indices.npy

Downstream jobs:
  TDnCNN topk_10
  TDnCNN quantile_10
  DRUNet topk10_sigma25
  DRUNet quantile10_sigma25

Checkpoint outputs:
  ${TDNCNN_DOWNSTREAM_CKPT_ROOT}/topk_10.pth
  ${TDNCNN_DOWNSTREAM_CKPT_ROOT}/quantile_10.pth
  ${DRUNET_DOWNSTREAM_CKPT_ROOT}/topk10_sigma25.pth
  ${DRUNET_DOWNSTREAM_CKPT_ROOT}/quantile10_sigma25.pth

Metrics outputs:
  ${TDNCNN_DOWNSTREAM_ROOT}/topk_10/metrics.json
  ${TDNCNN_DOWNSTREAM_ROOT}/quantile_10/metrics.json
  ${DRUNET_DOWNSTREAM_ROOT}/topk10_sigma25/metrics.json
  ${DRUNET_DOWNSTREAM_ROOT}/quantile10_sigma25/metrics.json
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
    echo "Noisy experiment directories already exist:"
    printf '  %s\n' "${existing[@]}"
    echo
    echo "Refusing to start production run to avoid mixing or overwriting artifacts."
    echo "Set FORCE=1 to allow the run to proceed without deleting anything."
    exit 1
  fi

  if [[ ${#existing[@]} -gt 0 && "${FORCE}" == "1" ]]; then
    echo
    echo "FORCE=1 is set. Existing noisy experiment directories will be reused:"
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

dataset_and_noisy_sanity_check() {
  python - <<'PY'
import os
from pathlib import Path

from src.dataset_registry import apply_dataset_runtime_overrides, corruption_enabled, load_dataset_config

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

cfg = apply_dataset_runtime_overrides(load_dataset_config("imagenet100"), mode="noisy")
cfg["local_root"] = str(root)
corruption = cfg.get("corruption", {})
print("dataset_mode:", cfg.get("dataset_mode"))
print("corruption:", corruption)
if cfg.get("dataset_mode") != "noisy":
    raise SystemExit("dataset_mode is not noisy")
if not corruption.get("enabled", False):
    raise SystemExit("dataset.corruption.enabled is not true")
if str(corruption.get("type")) != "poisson_gaussian":
    raise SystemExit(f"Unexpected corruption type: {corruption.get('type')}")
if not corruption_enabled(cfg, train=True) or not corruption_enabled(cfg, train=False):
    raise SystemExit("Poisson-Gaussian corruption is not enabled for both train and val")
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

check_downstream_filtering_artifacts() {
  assert_dir "${DOWNSTREAM_FILTERING_ROOT}"
  assert_file "${DOWNSTREAM_FILTERING_ROOT}/summary.json"
  assert_file "${DOWNSTREAM_FILTERING_ROOT}/topk/10/selected_indices.npy"
  assert_file "${DOWNSTREAM_FILTERING_ROOT}/topk/10/scores.csv"
  assert_file "${DOWNSTREAM_FILTERING_ROOT}/topk/10/metadata.json"
  assert_file "${DOWNSTREAM_FILTERING_ROOT}/quantile/10/selected_indices.npy"
  assert_file "${DOWNSTREAM_FILTERING_ROOT}/quantile/10/scores.csv"
  assert_file "${DOWNSTREAM_FILTERING_ROOT}/quantile/10/metadata.json"
}

check_tdncnn_artifacts() {
  assert_dir "${TDNCNN_ROOT}"
  assert_file "${TDNCNN_CKPT_ROOT}/full.pth"
  assert_file "${TDNCNN_ROOT}/full/metrics.json"
  assert_file "${TDNCNN_ROOT}/full/results/per_image_metrics.csv"
}

check_drunet_artifacts() {
  assert_dir "${DRUNET_ROOT}"
  assert_file "${DRUNET_CKPT}"
  assert_file "${DRUNET_ROOT}/full_sigma25/metrics.json"
  assert_file "${DRUNET_ROOT}/full_sigma25/results/per_image_metrics.csv"
}

check_downstream_tdncnn_artifacts() {
  local run_name
  for run_name in topk_10 quantile_10; do
    assert_file "${TDNCNN_DOWNSTREAM_CKPT_ROOT}/${run_name}.pth"
    assert_file "${TDNCNN_DOWNSTREAM_ROOT}/${run_name}/metrics.json"
    assert_file "${TDNCNN_DOWNSTREAM_ROOT}/${run_name}/results/per_image_metrics.csv"
  done
}

check_downstream_drunet_artifacts() {
  local run_name
  for run_name in topk10_sigma25 quantile10_sigma25; do
    assert_file "${DRUNET_DOWNSTREAM_CKPT_ROOT}/${run_name}.pth"
    assert_file "${DRUNET_DOWNSTREAM_ROOT}/${run_name}/metrics.json"
    assert_file "${DRUNET_DOWNSTREAM_ROOT}/${run_name}/results/per_image_metrics.csv"
  done
}

print_used_commands() {
  echo
  echo "Used commands:"
  local command
  for command in "${USED_COMMANDS[@]}"; do
    echo "  ${command}"
  done
}

run_downstream_only() {
  print_downstream_only_plan

  run_step "Check existing noisy filtering artifacts" check_downstream_filtering_artifacts

  run_step "TDnCNN noisy topk_10 training and ImageNet validation" \
    python scripts/train_tdncnn.py \
      --dataset imagenet100 \
      "${NOISY_CLI_ARGS[@]}" \
      --filtering-root "${DOWNSTREAM_FILTERING_ROOT}" \
      --output-root "${TDNCNN_DOWNSTREAM_ROOT}" \
      --checkpoint-root "${TDNCNN_DOWNSTREAM_CKPT_ROOT}" \
      --run topk_10

  run_step "TDnCNN noisy quantile_10 training and ImageNet validation" \
    python scripts/train_tdncnn.py \
      --dataset imagenet100 \
      "${NOISY_CLI_ARGS[@]}" \
      --filtering-root "${DOWNSTREAM_FILTERING_ROOT}" \
      --output-root "${TDNCNN_DOWNSTREAM_ROOT}" \
      --checkpoint-root "${TDNCNN_DOWNSTREAM_CKPT_ROOT}" \
      --run quantile_10

  run_step "Check noisy TDnCNN filtered artifacts" check_downstream_tdncnn_artifacts

  run_step "DRUNet noisy topk10_sigma25 training and ImageNet validation" \
    python scripts/train_drunet.py \
      --dataset imagenet100 \
      "${NOISY_CLI_ARGS[@]}" \
      --filtering-root "${DOWNSTREAM_FILTERING_ROOT}" \
      --output-root "${DRUNET_DOWNSTREAM_ROOT}" \
      --checkpoint-root "${DRUNET_DOWNSTREAM_CKPT_ROOT}" \
      --run topk10_sigma25

  run_step "DRUNet noisy quantile10_sigma25 training and ImageNet validation" \
    python scripts/train_drunet.py \
      --dataset imagenet100 \
      "${NOISY_CLI_ARGS[@]}" \
      --filtering-root "${DOWNSTREAM_FILTERING_ROOT}" \
      --output-root "${DRUNET_DOWNSTREAM_ROOT}" \
      --checkpoint-root "${DRUNET_DOWNSTREAM_CKPT_ROOT}" \
      --run quantile10_sigma25

  run_step "Check noisy DRUNet filtered artifacts" check_downstream_drunet_artifacts

  print_used_commands

  cat <<EOF

NOISY DOWNSTREAM-ONLY RUN PASSED

RUN_ID=${RUN_ID}

Filtering outputs used:
  ${DOWNSTREAM_FILTERING_ROOT}/topk/10/selected_indices.npy
  ${DOWNSTREAM_FILTERING_ROOT}/quantile/10/selected_indices.npy

Checkpoint paths:
  ${TDNCNN_DOWNSTREAM_CKPT_ROOT}/topk_10.pth
  ${TDNCNN_DOWNSTREAM_CKPT_ROOT}/quantile_10.pth
  ${DRUNET_DOWNSTREAM_CKPT_ROOT}/topk10_sigma25.pth
  ${DRUNET_DOWNSTREAM_CKPT_ROOT}/quantile10_sigma25.pth

Metrics paths:
  ${TDNCNN_DOWNSTREAM_ROOT}/topk_10/metrics.json
  ${TDNCNN_DOWNSTREAM_ROOT}/quantile_10/metrics.json
  ${DRUNET_DOWNSTREAM_ROOT}/topk10_sigma25/metrics.json
  ${DRUNET_DOWNSTREAM_ROOT}/quantile10_sigma25/metrics.json
EOF
}

case "${PIPELINE_STAGE}" in
  full|downstream_only)
    ;;
  *)
    fail "Unsupported PIPELINE_STAGE=${PIPELINE_STAGE}. Use full or downstream_only."
    ;;
esac

if [[ "${PIPELINE_STAGE}" == "downstream_only" ]]; then
  run_downstream_only
  exit 0
fi

print_plan
guard_canonical_dirs

run_step "GPU sanity check" gpu_sanity_check
run_step "Dataset and noisy corruption sanity check" dataset_and_noisy_sanity_check

run_step "Encoder validation" \
  python scripts/evaluate_encoder.py compare-encoders \
    --dataset imagenet100 \
    "${NOISY_CLI_ARGS[@]}" \
    --output-root "${ENCODER_VALIDATION_ROOT}"
run_step "Check encoder validation artifacts" check_encoder_validation_artifacts
run_step "Encoder selection artifact check" check_encoder_artifacts

run_step "Latent DDPM baseline" \
  python scripts/main.py \
    dataset=imagenet100 \
    "${NOISY_HYDRA_OVERRIDE}" \
    task=train_latent_DDPM \
    train_latent_DDPM.latent_noise_mode=baseline \
    train_latent_DDPM.id_label="_${RUN_ID}"
run_step "Check baseline DDPM artifacts" check_ddpm_artifacts "${BASELINE_DDPM_DIR}"

run_step "Latent DDPM encoder-induced" \
  python scripts/main.py \
    dataset=imagenet100 \
    "${NOISY_HYDRA_OVERRIDE}" \
    task=train_latent_DDPM \
    train_latent_DDPM.latent_noise_mode=induced \
    train_latent_DDPM.id_label="_${RUN_ID}"
run_step "Check induced DDPM artifacts" check_ddpm_artifacts "${INDUCED_DDPM_DIR}"

run_step "Score validation" \
  python scripts/evaluate_latent_ddpm_score.py \
    --dataset imagenet100 \
    "${NOISY_CLI_ARGS[@]}" \
    --modes baseline induced \
    --baseline-run-dir "${BASELINE_DDPM_DIR}" \
    --induced-run-dir "${INDUCED_DDPM_DIR}" \
    --output-root "${SCORE_VALIDATION_ROOT}"
run_step "Check score validation artifacts" check_score_validation_artifacts

run_step "Filtering" \
  python scripts/main.py \
    dataset=imagenet100 \
    "${NOISY_HYDRA_OVERRIDE}" \
    task=filter_dataset \
    filter_dataset.ddpm_branch=induced \
    dataset.latent_ddpm_validation.runs.induced.checkpoint_run="${INDUCED_RUN}" \
    filter_dataset.output_root="${FILTERING_ROOT}"
run_step "Check filtering artifacts" check_filtering_artifacts

run_step "TDnCNN full training and ImageNet validation" \
  python scripts/train_tdncnn.py --dataset imagenet100 "${NOISY_CLI_ARGS[@]}" --run full
run_step "Check TDnCNN artifacts" check_tdncnn_artifacts

run_step "DRUNet full training and ImageNet validation" \
  python scripts/train_drunet.py --dataset imagenet100 "${NOISY_CLI_ARGS[@]}" --run full_sigma25
run_step "Check DRUNet artifacts" check_drunet_artifacts

print_used_commands

cat <<EOF

NOISY FULL RUN PASSED

RUN_ID=${RUN_ID}

Checkpoint paths:
  ${ENCODER_CKPT}
  ${ENCODER_STATE}
  ${BASELINE_DDPM_DIR}/best_model.pth or ${BASELINE_DDPM_DIR}/epoch_*.pth
  ${INDUCED_DDPM_DIR}/best_model.pth or ${INDUCED_DDPM_DIR}/epoch_*.pth
  ${TDNCNN_CKPT_ROOT}/full.pth
  ${DRUNET_CKPT}

Output directories:
  ${ENCODER_OUTPUT_DIR}
  ${ENCODER_VALIDATION_ROOT}
  ${BASELINE_DDPM_DIR}
  ${INDUCED_DDPM_DIR}
  ${SCORE_VALIDATION_ROOT}
  ${FILTERING_ROOT}
  ${TDNCNN_ROOT}/full
  ${DRUNET_ROOT}/full_sigma25
EOF
