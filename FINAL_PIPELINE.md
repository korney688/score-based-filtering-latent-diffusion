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
python scripts/evaluate_encoder.py all
```

Scope:

- compare candidate encoders
- measure reconstruction quality
- measure latent noise geometry

Implementation:

- `src/evaluation/encoder_validation.py`

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

Stage 3 uses the MNIST train split only. It does not create a new image dataset.

Protocol:

```text
MNIST train image
-> online Gaussian noising
-> selected encoder
-> latent DDPM
-> score = ||eps_pred||^2
-> index selection
```

Supported DDPM branches:

- `filter_dataset.ddpm_branch=baseline`
- `filter_dataset.ddpm_branch=induced`

Supported filtering modes:

- `filter_dataset.filter_mode=top_k`
- `filter_dataset.filter_mode=quantile`

Example commands:

```bash
python scripts/main.py task=filter_dataset filter_dataset.ddpm_branch=baseline filter_dataset.filter_mode=top_k filter_dataset.keep_ratio=0.1
python scripts/main.py task=filter_dataset filter_dataset.ddpm_branch=baseline filter_dataset.filter_mode=quantile filter_dataset.quantile_low=0.0 filter_dataset.quantile_high=0.1
python scripts/main.py task=filter_dataset filter_dataset.ddpm_branch=induced filter_dataset.filter_mode=top_k filter_dataset.keep_ratio=0.1
python scripts/main.py task=filter_dataset filter_dataset.ddpm_branch=induced filter_dataset.filter_mode=quantile filter_dataset.quantile_low=0.0 filter_dataset.quantile_high=0.1
```

Implementation:

- `scripts/filter_dataset.py`
- `src/filters.py`

Outputs:

- `experiments/exp_005_filtering/<branch>/<mode>/scores.csv`
- `experiments/exp_005_filtering/<branch>/<mode>/selected_indices.npy`
- `experiments/exp_005_filtering/<branch>/<mode>/best_noisy_grid.png`
- `experiments/exp_005_filtering/<branch>/<mode>/worst_noisy_grid.png`
- `experiments/exp_005_filtering/<branch>/<mode>/best_clean_noisy_grid.png`
- `experiments/exp_005_filtering/<branch>/<mode>/worst_clean_noisy_grid.png`
- `experiments/exp_005_filtering/<branch>/<mode>/best_samples_grid.png`
- `experiments/exp_005_filtering/<branch>/<mode>/worst_samples_grid.png`
- `experiments/exp_005_filtering/<branch>/<mode>/selected_samples_grid.png`
- `experiments/exp_005_filtering/<branch>/<mode>/rejected_samples_grid.png`
- `experiments/exp_005_filtering/<branch>/<mode>/config.yaml`
- `experiments/exp_005_filtering/<branch>/<mode>/metadata.json`

## 7. TDnCNN Downstream Validation

TDnCNN uses torchvision MNIST directly. Training noise is generated online, and Stage 3 `selected_indices.npy` files affect only the MNIST train split. The MNIST test split is never filtered.

Run all configured downstream experiments:

```bash
python scripts/train_tdncnn.py
```

Run one downstream experiment:

```bash
python scripts/train_tdncnn.py --run full
python scripts/train_tdncnn.py --run baseline_topk_10
python scripts/train_tdncnn.py --run induced_topk_10
python scripts/train_tdncnn.py --run baseline_quantile_q0_q10
python scripts/train_tdncnn.py --run induced_quantile_q0_q10
```

Implementation:

- `scripts/internal/run_TDnCNN_image_suite.py`
- `scripts/internal/tdncnn_image_runs_config.py`
- `scripts/internal/train_TDnCNN_image.py`
- `src/TDnCNN_image.py`
- `src/tdncnn_datasets.py`

Outputs:

- `experiments/exp_006_tdncnn/full`
- `experiments/exp_006_tdncnn/baseline_topk_10`
- `experiments/exp_006_tdncnn/induced_topk_10`
- `experiments/exp_006_tdncnn/baseline_quantile_q0_q10`
- `experiments/exp_006_tdncnn/induced_quantile_q0_q10`

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
