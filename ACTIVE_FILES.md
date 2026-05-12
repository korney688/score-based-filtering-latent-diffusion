# Active Files

This document records the active files for the current MNIST latent-DDPM research protocol.

## 1. Encoder Training

Entrypoint:

- `scripts/train_encoder.py`

Required files:

- `scripts/internal/train_autoencoder_baseline_mnist.py`
- `scripts/internal/train_autoencoder_noise_consistency_mnist.py`
- `scripts/internal/train_autoencoder_representation_mnist.py`
- `scripts/internal/train_autoencoder_vae_mnist.py`
- `src/autoencoder.py`
- `src/autoencoder_noise_consistency.py`
- `src/autoencoder_representation.py`
- `src/autoencoder_vae.py`

## 2. Encoder Validation And Selection

Entrypoint:

- `scripts/evaluate_encoder.py`

Required files:

- `src/evaluation/encoder_validation.py`
- `src/autoencoder.py`
- `src/autoencoder_noise_consistency.py`
- `src/autoencoder_representation.py`
- `src/autoencoder_vae.py`

## 3. Latent-DDPM Training

Entrypoints:

- `scripts/main.py`
- `scripts/train_ddpm.py`

Required files:

- `configs/config.yaml`
- `configs/train_latent_DDPM/default.yaml`
- `configs/hydra/hydra_config.yaml`
- `src/DDPM_model.py`
- `src/Unet_model.py`
- `src/autoencoder.py`
- `src/autoencoder_noise_consistency.py`
- `src/datasets.py`
- `src/tools.py`

Dependency graph:

```text
scripts.main -> scripts.train_ddpm, scripts.filter_dataset
scripts.train_ddpm -> src.DDPM_model, src.datasets, src.tools
src.DDPM_model -> src.Unet_model, src.autoencoder, src.autoencoder_noise_consistency
```

## 4. Frozen-Encoder Latent-DDPM Score Validation

Entrypoint:

- `scripts/evaluate_latent_ddpm_score.py`

Required files:

- `src/evaluation/latent_ddpm_score_validation.py`
- `src/DDPM_model.py`
- `src/Unet_model.py`
- `src/autoencoder.py`
- `src/autoencoder_noise_consistency.py`

Dependency graph:

```text
scripts.evaluate_latent_ddpm_score -> src.evaluation.latent_ddpm_score_validation
src.evaluation.latent_ddpm_score_validation -> src.DDPM_model
src.DDPM_model -> src.Unet_model, src.autoencoder, src.autoencoder_noise_consistency
```

## 5. Score Calibration Only If Needed

No active entrypoint is currently defined.

Kept for future adaptation:

- `src/evaluation/score_calibration.py`

## 6. Score-Based Filtering

Entrypoints:

- `scripts/main.py task=filter_dataset`

Required files:

- `scripts/filter_dataset.py`
- `configs/filter_dataset/default.yaml`
- `src/DDPM_model.py`
- `src/Unet_model.py`
- `src/autoencoder.py`
- `src/autoencoder_noise_consistency.py`
- `src/datasets.py`
- `src/filters.py`
- `src/tools.py`

## 7. TDnCNN Downstream Validation

Entrypoints:

- `scripts/train_tdncnn.py`

Required files:

- `scripts/internal/run_TDnCNN_image_suite.py`
- `scripts/internal/train_TDnCNN_image.py`
- `scripts/internal/tdncnn_image_runs_config.py`
- `src/TDnCNN_image.py`
- `src/tdncnn_datasets.py`

## Active Python Files

- `scripts/evaluate_encoder.py`
- `scripts/evaluate_latent_ddpm_score.py`
- `scripts/filter_dataset.py`
- `scripts/main.py`
- `scripts/train_ddpm.py`
- `scripts/train_encoder.py`
- `scripts/train_tdncnn.py`
- `scripts/internal/run_TDnCNN_image_suite.py`
- `scripts/internal/tdncnn_image_runs_config.py`
- `scripts/internal/train_autoencoder_baseline_mnist.py`
- `scripts/internal/train_autoencoder_noise_consistency_mnist.py`
- `scripts/internal/train_autoencoder_representation_mnist.py`
- `scripts/internal/train_autoencoder_vae_mnist.py`
- `scripts/internal/train_TDnCNN_image.py`
- `src/DDPM_model.py`
- `src/TDnCNN_image.py`
- `src/Unet_model.py`
- `src/autoencoder.py`
- `src/autoencoder_noise_consistency.py`
- `src/autoencoder_representation.py`
- `src/autoencoder_vae.py`
- `src/datasets.py`
- `src/evaluation/__init__.py`
- `src/evaluation/encoder_validation.py`
- `src/evaluation/latent_ddpm_score_validation.py`
- `src/evaluation/score_calibration.py`
- `src/filters.py`
- `src/tdncnn_datasets.py`
- `src/tools.py`
