# Final Pipeline

This is the current MNIST latent-DDPM research protocol after cleanup.

The DDPM score validation stage is separated from encoder validation: by Stage 2 the encoder is already selected and frozen.

## 1. Encoder Training

Entrypoint:

```bash
python scripts/train_encoder.py full
```

Individual variants:

```bash
python scripts/train_encoder.py noise-consistency
python scripts/train_encoder.py representation
python scripts/train_encoder.py vae
```

Implementation:

- `scripts/train_encoder.py`
- `scripts/internal/train_autoencoder_baseline_mnist.py`
- `scripts/internal/train_autoencoder_noise_consistency_mnist.py`
- `scripts/internal/train_autoencoder_representation_mnist.py`
- `scripts/internal/train_autoencoder_vae_mnist.py`

## 2. Encoder Validation And Selection

Entrypoint:

```bash
python scripts/evaluate_encoder.py compare-encoders
```

Other encoder-validation modes:

```bash
python scripts/evaluate_encoder.py noise-geometry
python scripts/evaluate_encoder.py score-validation
python scripts/evaluate_encoder.py all
```

Scope:

- compare candidate encoders
- measure reconstruction quality
- measure latent noise geometry
- optionally compare score behavior for encoder selection

Implementation:

- `src/evaluation/encoder_validation.py`
- `src/evaluation/encoder_score_validation.py`

## 3. Latent-DDPM Training

Hydra entrypoint:

```bash
python scripts/main.py task=train_latent_DDPM
```

Baseline and induced DDPM runs:

```bash
python scripts/main.py task=train_latent_DDPM train_latent_DDPM.latent_noise_mode=baseline
python scripts/main.py task=train_latent_DDPM train_latent_DDPM.latent_noise_mode=induced
```

Noise modes:

- `baseline`: `z_noisy = z + sigma * eps_z`
- `induced`: `z_noisy = E(x + sigma * eps_x)`

Default outputs:

- `outputs/ddpm/latent_ddpm_baseline_ae_noise_consistency_mnist`
- `outputs/ddpm/latent_ddpm_induced_ae_noise_consistency_mnist`

Implementation:

- `scripts/train_ddpm.py`
- `src/DDPM_model.py`
- `src/Unet_model.py`

## 4. Frozen-Encoder Latent-DDPM Score Validation

This is the active Stage 2 DDPM validation.

Entrypoint:

```bash
python scripts/evaluate_latent_ddpm_score.py
```

Implementation:

- `src/evaluation/latent_ddpm_score_validation.py`

Scope:

- use one fixed selected encoder: `noise_consistency`
- load the trained baseline DDPM checkpoint
- load the trained induced DDPM checkpoint
- compute `score = ||eps_pred||^2`
- compare score with sampled external `sigma`
- compare baseline and induced target-noise geometry

Outputs:

- `experiments/exp_003_aligned_latent_ddpm/baseline`
- `experiments/exp_003_aligned_latent_ddpm/induced`
- `experiments/exp_003_aligned_latent_ddpm/comparison`

Diagnostics:

- Pearson/Spearman correlation between score and sigma
- score scatter plots
- mean score by sigma bins
- recomputed validation loss
- normalized validation loss
- target-noise norm distributions
- KL, Jensen-Shannon, and Wasserstein diagnostics
- covariance spectrum, anisotropy, and Frobenius distance

## 5. Score Calibration Only If Needed

Calibration is not part of the active validation path right now. If needed, it should be reconnected to the outputs of `scripts/evaluate_latent_ddpm_score.py`.

Existing implementation kept for future adaptation:

- `src/evaluation/score_calibration.py`

## 6. Score-Based Filtering

Hydra entrypoint:

```bash
python scripts/main.py task=filter_dataset
```

Filtering analysis:

```bash
python scripts/evaluate_pipeline.py filtering-analysis
```

Implementation:

- `scripts/filter_dataset.py`
- `src/evaluation/filtering_evaluation.py`
- `src/filters.py`
- `src/filter_mnist_top_k.py`
- `src/filter_mnist_qq.py`

## 7. TDnCNN Downstream Validation

Validation entrypoint:

```bash
python scripts/evaluate_pipeline.py downstream-validation
```

Training suite entrypoint:

```bash
python scripts/train_tdncnn.py
```

Implementation:

- `scripts/internal/run_TDnCNN_image_suite.py`
- `scripts/internal/train_TDnCNN_image.py`
- `src/TDnCNN_image.py`
- `src/tdncnn_datasets.py`

## Hydra Status

`scripts/main.py` remains the Hydra entrypoint for:

- `train_latent_DDPM`
- `filter_dataset`

## Verification

Use:

```bash
python -m compileall -q scripts src
```

Do not run long training jobs as cleanup verification.
