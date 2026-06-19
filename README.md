# Dataset-Scoped Latent-DDPM Filtering Project

Research pipeline for dataset-scoped encoder training, latent-DDPM training and validation, score-based train-set filtering, and downstream TDnCNN denoising validation.

Active dataset slugs are `mnist`, `cifar10`, and `imagenet100`.

## Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

On Windows with the local virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Dataset Configs

Dataset configs live in `configs/dataset/`:

- `mnist.yaml`: `1 x 28 x 28`, normalized with mean/std `[0.5]`.
- `cifar10.yaml`: `3 x 32 x 32`, normalized with mean/std `[0.5, 0.5, 0.5]`.
- `imagenet100.yaml`: `3 x 64 x 64`, ImageFolder-compatible, normalized with mean/std `[0.5, 0.5, 0.5]`.

Noise-consistency encoder defaults:

- MNIST: `noise_consistency_small`, `latent_dim=16`.
- CIFAR-10: `noise_consistency_large`, `latent_dim=128`.
- ImageNet-100: `noise_consistency_large`, `latent_dim=256`.

Dataset-scoped artifact roots:

```text
experiments/<dataset>/
checkpoints/<dataset>/
outputs/<dataset>/
```

## ImageNet-100 Research Run

ImageNet-100 is the active ImageNet-derived research benchmark. It keeps the same protocol:

```text
Encoder Validation
-> Encoder Selection
-> Baseline vs Induced Latent-DDPM
-> Score Validation
-> Score-Based Filtering
-> TDnCNN Downstream Validation
```

Expected local data layout:

```text
data/imagenet100/
├── train/<class_name>/*.JPEG
└── val/<class_name>/*.JPEG
```

Class folders may use either local names such as `class_000` or ImageNet synset names such as `n01440764`. No downloader is provided; the dataset must already be prepared in an ImageFolder-compatible layout. The validation split must already be arranged under `val/<class_name>/`.

Future manual run sequence:

```bash
# 1. Dry-run encoder config
python scripts/train_encoder.py noise-consistency --dataset imagenet100 --variant large --latent-dim 256 --epochs 0

# 2. Train encoder manually
python scripts/train_encoder.py noise-consistency --dataset imagenet100 --variant large --latent-dim 256

# 3. Validate encoder
python scripts/evaluate_encoder.py compare-encoders --dataset imagenet100

# 4. Train latent-DDPM baseline and induced manually
python scripts/main.py dataset=imagenet100 task=train_latent_DDPM train_latent_DDPM.latent_noise_mode=baseline
python scripts/main.py dataset=imagenet100 task=train_latent_DDPM train_latent_DDPM.latent_noise_mode=induced

# 5. Validate score
python scripts/evaluate_latent_ddpm_score.py --dataset imagenet100

# 6. Filter dataset
python scripts/main.py dataset=imagenet100 task=filter_dataset filter_dataset.ddpm_branch=induced

# 7. TDnCNN downstream validation
python scripts/train_tdncnn.py --dataset imagenet100 --run full
python scripts/train_tdncnn.py --dataset imagenet100 --run topk_10
```

ImageNet-100 smoke commands for later manual checks:

```bash
python scripts/evaluate_encoder.py compare-encoders --dataset imagenet100 --num-samples 8 --batch-size 4
python scripts/main.py dataset=imagenet100 task=filter_dataset filter_dataset.ddpm_branch=induced filter_dataset.max_samples=16 filter_dataset.batch_size=4 filter_dataset.grid_n_images=8 filter_dataset.noisy_grid_n_images=4 filter_dataset.output_root=experiments/imagenet100/exp_005_filtering_induced_smoke filter_dataset.overwrite=true
python scripts/train_tdncnn.py --dataset imagenet100 --list-runs
python scripts/train_tdncnn.py --dataset imagenet100 --run topk_10_smoke --epochs 1 --batch-size 2 --max-train-samples 16 --max-test-samples 8
```

Expected ImageNet-100 artifact roots:

```text
checkpoints/imagenet100/autoencoders/noise_consistency_large_latent256/
outputs/imagenet100/autoencoders/noise_consistency_large_latent256/
checkpoints/imagenet100/ddpm/latent_ddpm_baseline_ae_noise_consistency_imagenet100/
checkpoints/imagenet100/ddpm/latent_ddpm_induced_ae_noise_consistency_imagenet100/
experiments/imagenet100/exp_002_encoder_validation/
experiments/imagenet100/exp_003_latent_ddpm_validation/
experiments/imagenet100/exp_005_filtering/
experiments/imagenet100/exp_006_tdncnn/
```

## Main Pipeline

### 1. Train Encoders

MNIST supports all historical encoder variants:

```bash
python scripts/train_encoder.py baseline --dataset mnist
python scripts/train_encoder.py noise-consistency --dataset mnist --variant small
python scripts/train_encoder.py representation --dataset mnist
python scripts/train_encoder.py vae --dataset mnist
```

CIFAR-10 and ImageNet-100 use the noise-consistency autoencoder:

```bash
python scripts/train_encoder.py noise-consistency --dataset cifar10 --variant large --latent-dim 128
python scripts/train_encoder.py noise-consistency --dataset imagenet100 --variant large --latent-dim 256
```

Smoke-test commands without full training:

```bash
python scripts/train_encoder.py noise-consistency --dataset mnist --variant small --epochs 0
python scripts/train_encoder.py noise-consistency --dataset cifar10 --variant large --latent-dim 128 --epochs 0
python scripts/train_encoder.py noise-consistency --dataset imagenet100 --variant large --latent-dim 256 --epochs 0
```

### 2. Validate And Select Encoder

```bash
python scripts/evaluate_encoder.py compare-encoders --dataset mnist
python scripts/evaluate_encoder.py compare-encoders --dataset cifar10
python scripts/evaluate_encoder.py compare-encoders --dataset imagenet100
```

Expected output:

```text
experiments/<dataset>/exp_002_encoder_validation/
├── metrics/
├── latent_geometry/
├── covariance/
├── eigenspectrum/
├── reconstructions/
├── report/
└── summary.json
```

### 3. Train Latent-DDPM

Hydra entrypoint:

```bash
python scripts/main.py task=train_latent_DDPM
```

ImageNet-100 latent-DDPM commands:

```bash
python scripts/main.py dataset=imagenet100 task=train_latent_DDPM train_latent_DDPM.latent_noise_mode=baseline
python scripts/main.py dataset=imagenet100 task=train_latent_DDPM train_latent_DDPM.latent_noise_mode=induced
```

The selected ImageNet-100 encoder checkpoint is configured as:

```text
checkpoints/imagenet100/autoencoders/noise_consistency_large_latent256/autoencoder_checkpoint.pt
```

`train_latent_DDPM.max_samples` is not part of the current DDPM training config. Use `train_latent_DDPM.smoke_model_only=true` for configuration/model-shape checks, or reduce `train_latent_DDPM.batch_size` for manual experiments.

### 4. Validate Latent-DDPM Score

```bash
python scripts/evaluate_latent_ddpm_score.py --dataset mnist
python scripts/evaluate_latent_ddpm_score.py --dataset cifar10
python scripts/evaluate_latent_ddpm_score.py --dataset imagenet100
```

This stage keeps the score definition unchanged:

```text
score = ||eps_pred||^2
```

Expected output:

```text
experiments/<dataset>/exp_003_latent_ddpm_validation/
├── metrics/
├── score_validation/
├── score_distributions/
├── noise_prediction/
├── covariance/
├── report/
└── summary.json
```

### 5. Score-Based Filtering

```bash
python scripts/main.py dataset=imagenet100 task=filter_dataset filter_dataset.ddpm_branch=baseline
python scripts/main.py dataset=imagenet100 task=filter_dataset filter_dataset.ddpm_branch=induced
```

The default config runs both `top_k` and `quantile` filtering at `5%`, `10%`, and `15%`.

Filtering modes:

- `top_k`: selects the lowest-score samples globally. Lower score means a more typical sample for the current score definition.
- `quantile`: performs stratified sampling over score quantile bins, selecting the configured fraction from each bin. This preserves score-distribution coverage and avoids collapsing the subset into only the lowest-score region.

The active `quantile` mode is not a single quantile interval. It is a QQ-spread / stratified quantile sampling strategy inherited from the laboratory filtering protocol.

Expected output:

```text
experiments/imagenet100/exp_005_filtering/
├── topk/
│   ├── 5/
│   ├── 10/
│   └── 15/
├── quantile/
│   ├── 5/
│   ├── 10/
│   └── 15/
├── metrics/
├── plots/
├── report/
└── summary.json
```

Per filtering directory:

- `scores.csv`
- `selected_indices.npy`
- `metadata.json`
- `config.yaml`
- `score_histogram.png`
- `best_samples_grid.png`
- `worst_samples_grid.png`
- `selected_samples_grid.png`
- `rejected_samples_grid.png`
- `best_noisy_grid.png`
- `worst_noisy_grid.png`
- `best_clean_noisy_grid.png`
- `worst_clean_noisy_grid.png`

For `filter_mode=quantile`, `metadata.json` records
`algorithm=quantile_spread`, `keep_ratio`, `min_points_per_bin`, `seed`,
`n_bins`, and the actual selected count. Existing `quantile_low` and
`quantile_high` fields are legacy compatibility fields and are not used by the
active quantile algorithm.

### 6. TDnCNN Downstream Validation

List configured runs:

```bash
python scripts/train_tdncnn.py --dataset mnist --list-runs
python scripts/train_tdncnn.py --dataset cifar10 --list-runs
python scripts/train_tdncnn.py --dataset imagenet100 --list-runs
```

ImageNet-100 TDnCNN commands:

```bash
python scripts/train_tdncnn.py --dataset imagenet100 --run topk_10_smoke --epochs 1 --batch-size 2 --max-train-samples 16 --max-test-samples 8
python scripts/train_tdncnn.py --dataset imagenet100 --run topk_10
python scripts/train_tdncnn.py --dataset imagenet100 --run full
```

ImageNet-100 TDnCNN routing:

- filtered indices: `experiments/imagenet100/exp_005_filtering/topk/10/selected_indices.npy`
- outputs: `experiments/imagenet100/exp_006_tdncnn/topk_10/`
- checkpoint: `checkpoints/imagenet100/tdncnn/topk_10.pth`

TDnCNN on 64x64 images should use smaller smoke batch sizes than MNIST/CIFAR-10.

## Research-Style Plots

The reporting layer builds laboratory-report-style figures from existing artifacts only. It does not train models, run filtering, or change the score definition.

Entrypoint:

```bash
python scripts/generate_research_plots.py --dataset cifar10 --stage all
python scripts/generate_research_plots.py --dataset imagenet100 --stage all
python scripts/generate_research_plots.py --dataset imagenet100 --stage filtering
python scripts/generate_research_plots.py --dataset imagenet100 --stage tdncnn
```

Supported stages:

- `filtering`: score distribution plots for Top-K, QQ-spread Quantile, random 10%, and selected/rejected examples.
- `ddpm`: score/latent training dynamics when per-epoch score stats exist, with DDPM loss fallback.
- `tdncnn`: PSNR/MSE/SSIM/LPIPS CDF comparisons, median-gap tables, difficult-subset plots, and qualitative denoising examples when per-image metrics exist.

Outputs are saved under:

```text
experiments/<dataset>/exp_005_filtering/plots/research_style/
experiments/<dataset>/exp_003_latent_ddpm_validation/plots/research_style/
experiments/<dataset>/exp_006_tdncnn/comparison_plots/research_style/
experiments/<dataset>/exp_006_tdncnn/qualitative/research_style/
```

Required inputs are existing `scores.csv`, `selected_indices.npy`, filtering grids, DDPM `DDPM_metrics.csv` or optional `score_training_dynamics.csv`, and TDnCNN `results/per_image_metrics.csv` for CDF plots. Missing inputs are skipped with warnings when `--strict false`.

Mapping from the wireless-denoising report style:

```text
Lab SNR distribution plots
-> score distribution plots

Lab score/SNR dynamics over DDPM training
-> score/latent dynamics over latent-DDPM training

Lab capacity CDF and gap comparison
-> PSNR/MSE/SSIM/LPIPS CDF and gap comparison

Lab PDP/PAP diversity examples
-> selected/rejected image examples and denoising/error-map grids
```

## Project Layout

```text
configs/      Hydra configs, including dataset configs
data/         Reusable local data assets
experiments/  Dataset-scoped experiment artifacts
checkpoints/  Dataset-scoped model checkpoints
notebooks/    Analysis notebooks and exported reports
outputs/      Dataset-scoped generated outputs
scripts/      Public entrypoints and internal runners
src/          Model, dataset, filtering, and evaluation code
docs/         Supporting documentation
```

## Verification

Use fast checks only:

```bash
python -m compileall -q scripts src
python scripts/train_tdncnn.py --dataset imagenet100 --list-runs
```

Do not run full encoder training, latent-DDPM training, score validation on the full dataset, full filtering, or TDnCNN training as setup verification.
