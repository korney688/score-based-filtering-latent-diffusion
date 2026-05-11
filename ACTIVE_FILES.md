# Active Files

This document records the active files and transitive dependencies for the approved MNIST latent-DDPM research protocol.

## 1. Encoder Training

Public entrypoint:

- `scripts/train_encoder.py`

Required files:

- `scripts/train_encoder.py`
- `scripts/internal/train_autoencoder_baseline_mnist.py`
- `scripts/internal/train_autoencoder_noise_consistency_mnist.py`
- `scripts/internal/train_autoencoder_representation_mnist.py`
- `scripts/internal/train_autoencoder_vae_mnist.py`
- `src/autoencoder.py`
- `src/autoencoder_noise_consistency.py`
- `src/autoencoder_representation.py`
- `src/autoencoder_vae.py`

Dependency graph:

```text
scripts.train_encoder -> scripts.internal.train_autoencoder_baseline_mnist, scripts.internal.train_autoencoder_noise_consistency_mnist, scripts.internal.train_autoencoder_representation_mnist, scripts.internal.train_autoencoder_vae_mnist
scripts.internal.train_autoencoder_baseline_mnist -> src.autoencoder
scripts.internal.train_autoencoder_noise_consistency_mnist -> src.autoencoder_noise_consistency
scripts.internal.train_autoencoder_representation_mnist -> src.autoencoder_representation
scripts.internal.train_autoencoder_vae_mnist -> src.autoencoder_vae
```

## 2. Encoder Validation And Encoder Selection

Public entrypoint:

- `scripts/evaluate_encoder.py`

Required files:

- `scripts/evaluate_encoder.py`
- `src/evaluation/encoder_score_validation.py`
- `src/evaluation/encoder_validation.py`
- `src/autoencoder.py`
- `src/autoencoder_noise_consistency.py`
- `src/autoencoder_representation.py`
- `src/autoencoder_vae.py`
- `src/Unet_model.py`

Dependency graph:

```text
scripts.evaluate_encoder -> src.evaluation.encoder_validation, src.evaluation.encoder_score_validation
src.evaluation.encoder_validation -> src.autoencoder, src.autoencoder_noise_consistency, src.autoencoder_representation, src.autoencoder_vae
src.evaluation.encoder_score_validation -> src.Unet_model, src.autoencoder, src.autoencoder_noise_consistency, src.autoencoder_representation, src.autoencoder_vae
```

## 3. Baseline Vs Aligned Latent-DDPM Training

Entrypoints:

- `scripts/main.py`
- `scripts/train_ddpm.py`

Required files:

- `scripts/main.py`
- `scripts/train_ddpm.py`
- `scripts/filter_dataset.py`
- `src/DDPM_model.py`
- `src/Unet_model.py`
- `src/autoencoder.py`
- `src/datasets.py`
- `src/tools.py`

Dependency graph:

```text
scripts.main -> scripts.train_ddpm, scripts.filter_dataset
scripts.train_ddpm -> src.DDPM_model, src.datasets, src.tools
src.DDPM_model -> src.Unet_model, src.autoencoder
```

## 4. Score Validation

Public entrypoint:

- `scripts/evaluate_score.py`

Required files:

- `scripts/evaluate_score.py`
- `src/evaluation/score_validation.py`
- `src/DDPM_model.py`
- `src/Unet_model.py`
- `src/autoencoder.py`

Dependency graph:

```text
scripts.evaluate_score -> src.evaluation.score_validation, src.evaluation.score_calibration
src.evaluation.score_validation -> src.DDPM_model
src.DDPM_model -> src.Unet_model, src.autoencoder
```

## 5. Score Calibration Only If Needed

Public entrypoint:

- `scripts/evaluate_score.py calibration`

Required files:

- `src/evaluation/score_calibration.py`
- `src/Unet_model.py`
- `src/autoencoder.py`

Dependency graph:

```text
src.evaluation.score_calibration -> src.Unet_model, src.autoencoder
```

## 6. Score-Based Filtering

Entrypoints:

- `scripts/main.py task=filter_dataset`
- `scripts/evaluate_pipeline.py filtering-analysis`

Required files:

- `scripts/filter_dataset.py`
- `scripts/evaluate_pipeline.py`
- `src/evaluation/filtering_evaluation.py`
- `src/filter_mnist_top_k.py`
- `src/filter_mnist_qq.py`
- `src/DDPM_model.py`
- `src/Unet_model.py`
- `src/autoencoder.py`
- `src/datasets.py`
- `src/filters.py`
- `src/tools.py`

Dependency graph:

```text
scripts.filter_dataset -> src.DDPM_model, src.datasets, src.filters, src.tools
scripts.evaluate_pipeline -> src.evaluation.filtering_evaluation, scripts.internal.run_TDnCNN_image_suite
src.filter_mnist_qq -> src.DDPM_model
src.filter_mnist_top_k -> src.DDPM_model
src.DDPM_model -> src.Unet_model, src.autoencoder
```

## 7. TDnCNN Downstream Validation

Entrypoints:

- `scripts/train_tdncnn.py`
- `scripts/evaluate_pipeline.py downstream-validation`

Required files:

- `scripts/train_tdncnn.py`
- `scripts/evaluate_pipeline.py`
- `scripts/internal/run_TDnCNN_image_suite.py`
- `scripts/internal/train_TDnCNN_image.py`
- `scripts/internal/tdncnn_image_runs_config.py`
- `src/TDnCNN_image.py`
- `src/tdncnn_datasets.py`

Dependency graph:

```text
scripts.train_tdncnn -> scripts.internal.run_TDnCNN_image_suite
scripts.evaluate_pipeline -> scripts.internal.run_TDnCNN_image_suite, src.evaluation.filtering_evaluation
scripts.internal.run_TDnCNN_image_suite -> scripts.internal.tdncnn_image_runs_config, scripts.internal.train_TDnCNN_image
scripts.internal.train_TDnCNN_image -> src.TDnCNN_image, src.tdncnn_datasets
```

## Active Python Files

- `scripts/evaluate_encoder.py`
- `scripts/evaluate_pipeline.py`
- `scripts/evaluate_score.py`
- `scripts/filter_dataset.py`
- `scripts/main.py`
- `scripts/internal/run_TDnCNN_image_suite.py`
- `scripts/internal/tdncnn_image_runs_config.py`
- `scripts/internal/train_autoencoder_baseline_mnist.py`
- `scripts/internal/train_autoencoder_noise_consistency_mnist.py`
- `scripts/internal/train_autoencoder_representation_mnist.py`
- `scripts/internal/train_autoencoder_vae_mnist.py`
- `scripts/internal/train_TDnCNN_image.py`
- `scripts/train_ddpm.py`
- `scripts/train_encoder.py`
- `scripts/train_tdncnn.py`
- `src/DDPM_model.py`
- `src/TDnCNN_image.py`
- `src/Unet_model.py`
- `src/autoencoder.py`
- `src/autoencoder_noise_consistency.py`
- `src/autoencoder_representation.py`
- `src/autoencoder_vae.py`
- `src/datasets.py`
- `src/evaluation/encoder_validation.py`
- `src/evaluation/filtering_evaluation.py`
- `src/evaluation/score_calibration.py`
- `src/evaluation/score_validation.py`
- `src/filter_mnist_qq.py`
- `src/filter_mnist_top_k.py`
- `src/filters.py`
- `src/tdncnn_datasets.py`
- `src/tools.py`
