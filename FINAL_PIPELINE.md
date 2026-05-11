# Final Pipeline

This is the approved MNIST latent-DDPM research protocol after controlled cleanup.

No experiment logic, model behavior, metrics, configs, dataset generation, filtering logic, or TDnCNN image logic was changed.

## 1. Encoder Training

Protocol entrypoint:

```bash
python scripts/train_encoder.py full
```

Variants:

```bash
python scripts/train_encoder.py noise-consistency
python scripts/train_encoder.py representation
python scripts/train_encoder.py vae
```

Preserved implementations:

- `scripts/internal/train_autoencoder_baseline_mnist.py`
- `scripts/internal/train_autoencoder_noise_consistency_mnist.py`
- `scripts/internal/train_autoencoder_representation_mnist.py`
- `scripts/internal/train_autoencoder_vae_mnist.py`

## 2. Encoder Validation And Encoder Selection

Protocol entrypoint:

```bash
python scripts/evaluate_encoder.py compare-encoders
```

Modes:

```bash
python scripts/evaluate_encoder.py noise-geometry
python scripts/evaluate_encoder.py all
```

Preserved calculations:

- `src/evaluation/encoder_validation.py`

## 3. Baseline Vs Aligned Latent-DDPM Training

Hydra entrypoint:

```bash
python scripts/main.py task=train_latent_DDPM
```

Core implementation:

- `scripts/train_ddpm.py`
- `src/DDPM_model.py`
- `src/Unet_model.py`
- `src/autoencoder.py`
- `src/datasets.py`

## 4. Score Validation

Protocol entrypoint:

```bash
python scripts/evaluate_encoder.py score-validation
```

Dataset score-analysis entrypoint:

```bash
python scripts/evaluate_score.py score-validation
```

Baseline check:

```bash
python scripts/evaluate_score.py baseline-check
```

Preserved calculations:

- `src/evaluation/encoder_score_validation.py`
- `src/evaluation/score_validation.py`

## 5. Score Calibration Only If Needed

Protocol entrypoint:

```bash
python scripts/evaluate_score.py calibration
```

Preserved calculation:

- `src/evaluation/score_calibration.py`

Calibration remains an optional protocol step, not a required active stage unless score validation indicates it is needed.

## 6. Score-Based Filtering

Hydra entrypoint:

```bash
python scripts/main.py task=filter_dataset
```

Filtering analysis:

```bash
python scripts/evaluate_pipeline.py filtering-analysis
```

Preserved filtering logic:

- `scripts/filter_dataset.py`
- `src/evaluation/filtering_evaluation.py`
- `src/filters.py`
- `src/filter_mnist_top_k.py`
- `src/filter_mnist_qq.py`
- `src/DDPM_model.py`

## 7. TDnCNN Downstream Validation

Protocol entrypoint:

```bash
python scripts/evaluate_pipeline.py downstream-validation
```

Training suite entrypoint:

```bash
python scripts/train_tdncnn.py
```

Preserved implementations:

- `scripts/internal/run_TDnCNN_image_suite.py`
- `scripts/internal/train_TDnCNN_image.py`
- `src/TDnCNN_image.py`
- `src/tdncnn_datasets.py`

## Hydra Status

`scripts/main.py` is still kept because it is the active Hydra entrypoint for:

- `train_latent_DDPM`
- `filter_dataset`

Hydra is not removed.

## Verification

Required verification command:

```bash
python -m compileall -q scripts src
```

No long training should be run as part of cleanup verification.
