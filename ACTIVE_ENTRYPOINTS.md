# Active Entrypoints

This document records the public entrypoints for the current MNIST latent-DDPM research protocol.

## 1. Encoder Training

- `scripts/train_encoder.py`

Modes:

- `full`
- `noise-consistency`
- `representation`
- `vae`

## 2. Encoder Validation And Selection

- `scripts/evaluate_encoder.py`

Modes:

- `compare-encoders`
- `noise-geometry`
- `score-validation`
- `all`

Implementation modules:

- `src/evaluation/encoder_validation.py`
- `src/evaluation/encoder_score_validation.py`

## 3. Latent-DDPM Training

- `scripts/main.py task=train_latent_DDPM`

DDPM noise modes:

- `train_latent_DDPM.latent_noise_mode=baseline`
- `train_latent_DDPM.latent_noise_mode=induced`

Implementation:

- `scripts/train_ddpm.py`

## 4. Frozen-Encoder Latent-DDPM Score Validation

- `scripts/evaluate_latent_ddpm_score.py`

Implementation:

- `src/evaluation/latent_ddpm_score_validation.py`

Scope:

- fixed selected `noise_consistency` encoder
- trained baseline latent-DDPM checkpoint
- trained induced latent-DDPM checkpoint
- score behavior against sampled sigma
- target-noise distribution and covariance diagnostics

## 5. Score Calibration Only If Needed

There is no active calibration entrypoint right now. `src/evaluation/score_calibration.py` is kept only for future adaptation to the new latent-DDPM score validation outputs.

## 6. Score-Based Filtering

- `scripts/main.py task=filter_dataset`
- `scripts/evaluate_pipeline.py filtering-analysis`

Implementation:

- `scripts/filter_dataset.py`
- `src/evaluation/filtering_evaluation.py`

## 7. TDnCNN Downstream Validation

- `scripts/evaluate_pipeline.py downstream-validation`
- `scripts/train_tdncnn.py`

Implementation:

- `scripts/internal/run_TDnCNN_image_suite.py`
- `scripts/internal/train_TDnCNN_image.py`
- `src/TDnCNN_image.py`
- `src/tdncnn_datasets.py`

## Hydra Status

`scripts/main.py` remains active for:

- `train_latent_DDPM`
- `filter_dataset`
