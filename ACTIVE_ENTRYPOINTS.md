# Active Entrypoints

This document records the public entrypoints for the approved MNIST latent-DDPM research protocol.

No model behavior, dataset logic, metrics, configs, or training logic is changed by this document.

## 1. Encoder Training

Public entrypoint:

- `scripts/train_encoder.py`

Preserved implementation scripts:

- `scripts/internal/train_autoencoder_baseline_mnist.py`
- `scripts/internal/train_autoencoder_noise_consistency_mnist.py`
- `scripts/internal/train_autoencoder_representation_mnist.py`
- `scripts/internal/train_autoencoder_vae_mnist.py`

## 2. Encoder Validation And Encoder Selection

Public entrypoint:

- `scripts/evaluate_encoder.py`

Protocol modes:

- `compare-encoders`
- `noise-geometry`
- `score-validation`
- `all`

Preserved implementation modules:

- `src/evaluation/encoder_validation.py`
- `src/evaluation/encoder_score_validation.py`

## 3. Baseline Vs Aligned Latent-DDPM Training

Hydra entrypoint:

- `scripts/main.py task=train_latent_DDPM`

Core implementation:

- `scripts/train_ddpm.py`

## 4. Score Validation

Public entrypoint:

- `scripts/evaluate_score.py score-validation`

Encoder-score validation entrypoint:

- `scripts/evaluate_encoder.py score-validation`

Baseline check:

- `scripts/evaluate_score.py baseline-check`

Preserved implementation modules:

- `src/evaluation/score_validation.py`

## 5. Score Calibration Only If Needed

Public entrypoint:

- `scripts/evaluate_score.py calibration`

Preserved implementation module:

- `src/evaluation/score_calibration.py`

## 6. Score-Based Filtering

Hydra entrypoint:

- `scripts/main.py task=filter_dataset`

Public analysis entrypoint:

- `scripts/evaluate_pipeline.py filtering-analysis`

Preserved implementation files:

- `scripts/filter_dataset.py`
- `src/evaluation/filtering_evaluation.py`
- `src/filter_mnist_top_k.py`
- `src/filter_mnist_qq.py`
- `src/filters.py`

## 7. TDnCNN Downstream Validation

Public validation entrypoint:

- `scripts/evaluate_pipeline.py downstream-validation`

Training suite entrypoint:

- `scripts/train_tdncnn.py`

Preserved implementation files:

- `scripts/internal/run_TDnCNN_image_suite.py`
- `scripts/internal/train_TDnCNN_image.py`
- `src/TDnCNN_image.py`
- `src/tdncnn_datasets.py`

## Hydra Status

`scripts/main.py` is still needed and kept because it is currently the active Hydra entrypoint for:

- `train_latent_DDPM`
- `filter_dataset`

Hydra is not removed.
