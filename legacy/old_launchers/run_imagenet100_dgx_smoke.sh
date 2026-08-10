#!/usr/bin/env bash

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace}"
DATASET_ROOT="${IMAGENET100_ROOT:-/workspace/data/imagenet100}"
RUN_ID="${RUN_ID:-dgx_smoke_$(date +%Y%m%d_%H%M%S)}"

cd "${PROJECT_ROOT}" || {
  echo "SMOKE FAILED"
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
ENCODER_CKPT="checkpoints/imagenet100/autoencoders/noise_consistency_large_latent256/autoencoder_checkpoint.pt"
ENCODER_VALIDATION_ROOT="experiments/imagenet100/exp_002_encoder_validation_${RUN_ID}"
SCORE_VALIDATION_ROOT="experiments/imagenet100/exp_003_latent_ddpm_validation_${RUN_ID}"
FILTERING_ROOT="experiments/imagenet100/exp_005_filtering_${RUN_ID}"

fail() {
  local message="$1"
  echo
  echo "SMOKE FAILED"
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

gpu_sanity_check() {
  python - <<'PY'
import sys
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
  assert_file "checkpoints/imagenet100/autoencoders/noise_consistency_large_latent256/E.pt"
  assert_file "outputs/imagenet100/autoencoders/noise_consistency_large_latent256/metrics.json"
}

check_encoder_validation_artifacts() {
  assert_dir "${ENCODER_VALIDATION_ROOT}"
  assert_file "${ENCODER_VALIDATION_ROOT}/summary.json"
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
}

check_filtering_artifacts() {
  local filter_dir="${FILTERING_ROOT}/topk/10"
  assert_dir "${filter_dir}"
  assert_file "${filter_dir}/scores.csv"
  assert_file "${filter_dir}/selected_indices.npy"
  assert_file "${filter_dir}/metadata.json"
  assert_file "${FILTERING_ROOT}/summary.json"
}

echo "ImageNet-100 DGX smoke run"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "IMAGENET100_ROOT=${IMAGENET100_ROOT}"
echo "RUN_ID=${RUN_ID}"
echo "BASELINE_RUN=${BASELINE_RUN}"
echo "INDUCED_RUN=${INDUCED_RUN}"

run_step "GPU sanity check" gpu_sanity_check
run_step "Dataset sanity check" dataset_sanity_check

run_step "Encoder training smoke" \
  python scripts/train_encoder.py noise-consistency \
    --dataset imagenet100 \
    --variant large \
    --latent-dim 256 \
    --epochs 1 \
    --fast_dev_run
run_step "Check encoder artifacts" check_encoder_artifacts

run_step "Encoder validation smoke" \
  python scripts/evaluate_encoder.py compare-encoders \
    --dataset imagenet100 \
    --num-samples 8 \
    --batch-size 4 \
    --output-root "${ENCODER_VALIDATION_ROOT}"
run_step "Check encoder validation artifacts" check_encoder_validation_artifacts

run_step "Latent DDPM baseline smoke" \
  python scripts/main.py \
    dataset=imagenet100 \
    task=train_latent_DDPM \
    train_latent_DDPM.latent_noise_mode=baseline \
    train_latent_DDPM.id_label="_${RUN_ID}" \
    train_latent_DDPM.n_epochs=1 \
    train_latent_DDPM.batch_size=2 \
    train_latent_DDPM.num_workers=0 \
    train_latent_DDPM.patience=1 \
    train_latent_DDPM.score_stats.enabled=true \
    train_latent_DDPM.score_stats.max_score_stat_samples=16 \
    train_latent_DDPM.score_stats.score_stat_every_n_epochs=1
run_step "Check baseline DDPM artifacts" check_ddpm_artifacts "${BASELINE_DDPM_DIR}"

run_step "Latent DDPM induced smoke" \
  python scripts/main.py \
    dataset=imagenet100 \
    task=train_latent_DDPM \
    train_latent_DDPM.latent_noise_mode=induced \
    train_latent_DDPM.id_label="_${RUN_ID}" \
    train_latent_DDPM.n_epochs=1 \
    train_latent_DDPM.batch_size=2 \
    train_latent_DDPM.num_workers=0 \
    train_latent_DDPM.patience=1 \
    train_latent_DDPM.score_stats.enabled=true \
    train_latent_DDPM.score_stats.max_score_stat_samples=16 \
    train_latent_DDPM.score_stats.score_stat_every_n_epochs=1
run_step "Check induced DDPM artifacts" check_ddpm_artifacts "${INDUCED_DDPM_DIR}"

run_step "Score validation smoke" \
  python scripts/evaluate_latent_ddpm_score.py \
    --dataset imagenet100 \
    --num-samples 8 \
    --batch-size 4 \
    --modes baseline induced \
    --baseline-run-dir "${BASELINE_DDPM_DIR}" \
    --induced-run-dir "${INDUCED_DDPM_DIR}" \
    --output-root "${SCORE_VALIDATION_ROOT}"
run_step "Check score validation artifacts" check_score_validation_artifacts

run_step "Filtering top-k 10 smoke" \
  python scripts/main.py \
    dataset=imagenet100 \
    task=filter_dataset \
    filter_dataset.ddpm_branch=induced \
    dataset.latent_ddpm_validation.runs.induced.checkpoint_run="${INDUCED_RUN}" \
    "filter_dataset.filter_modes=[top_k]" \
    "filter_dataset.filter_percentages=[10]" \
    filter_dataset.max_samples=16 \
    filter_dataset.batch_size=4 \
    filter_dataset.grid_n_images=8 \
    filter_dataset.noisy_grid_n_images=4 \
    filter_dataset.overwrite=true \
    filter_dataset.output_root="${FILTERING_ROOT}"
run_step "Check filtering artifacts" check_filtering_artifacts

echo
echo "SMOKE PASSED"
