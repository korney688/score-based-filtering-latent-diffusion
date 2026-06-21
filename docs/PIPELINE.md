# Active Pipeline

This document describes the active dataset-scoped research pipeline.

MNIST remains the complete end-to-end protocol. CIFAR-10 remains active for noise-consistency encoder training, encoder validation, latent-DDPM training/validation, score-based filtering, and TDnCNN downstream validation. ImageNet-100 is the active ImageNet-derived research benchmark and now includes DRUNet as an additional downstream denoiser.

## 1. Dataset Layer

Dataset configs:

- `configs/dataset/mnist.yaml`
- `configs/dataset/cifar10.yaml`
- `configs/dataset/imagenet100.yaml`

Shared dataset utilities:

- `src/dataset_registry.py`

Current data shapes:

- MNIST: `[B, 1, 28, 28]`
- CIFAR-10: `[B, 3, 32, 32]`
- ImageNet-100: `[B, 3, 64, 64]`

All active configs normalize images to `[-1, 1]` with mean/std `0.5`.

Noise-consistency encoder defaults:

- MNIST: `noise_consistency_small`, `latent_dim=16`
- CIFAR-10: `noise_consistency_large`, `latent_dim=128`
- ImageNet-100: `noise_consistency_large`, `latent_dim=256`

Artifact roots are dataset-scoped:

```text
checkpoints/<dataset>/
outputs/<dataset>/
experiments/<dataset>/
```

## 2. ImageNet-100 Research Run

ImageNet-100 is the active ImageNet-derived research benchmark. It is not a new task and does not change the hypothesis or protocol:

```text
Encoder Validation
-> Encoder Selection
-> Baseline vs Induced Latent-DDPM
-> Score Validation
-> Score-Based Filtering
-> Downstream Denoising Validation (TDnCNN / DRUNet)
-> External Benchmark Evaluation
```

Use one dataset slug everywhere:

```text
imagenet100
```

Expected local ImageFolder-compatible layout:

```text
data/imagenet100/
├── train/<class_name>/*.JPEG
└── val/<class_name>/*.JPEG
```

The class folder names may be synthetic names such as `class_000` or real ImageNet synset names such as `n01440764`. No downloader is implemented. If `data/imagenet100/train` or `data/imagenet100/val` is missing, the dataset registry raises a clear `FileNotFoundError`.

## 3. Encoder Training

Entrypoint:

```bash
python scripts/train_encoder.py {baseline,noise-consistency,representation,vae}
```

Implementation:

- `scripts/train_encoder.py`
- `scripts/internal/train_autoencoder_noise_consistency_mnist.py`
- `src/autoencoder_noise_consistency.py`
- `src/dataset_registry.py`

MNIST supports all historical variants:

```bash
python scripts/train_encoder.py baseline --dataset mnist
python scripts/train_encoder.py noise-consistency --dataset mnist --variant small
python scripts/train_encoder.py representation --dataset mnist
python scripts/train_encoder.py vae --dataset mnist
```

CIFAR-10 and ImageNet-100 use the large noise-consistency architecture:

```bash
python scripts/train_encoder.py noise-consistency --dataset cifar10 --variant large --latent-dim 128
python scripts/train_encoder.py noise-consistency --dataset imagenet100 --variant large --latent-dim 256
```

ImageNet-100 dry-run command:

```bash
python scripts/train_encoder.py noise-consistency --dataset imagenet100 --variant large --latent-dim 256 --epochs 0
```

The dry run builds dataloaders and performs a model forward pass, but it does not run training epochs.

ImageNet-100 model/data parameters come from `configs/dataset/imagenet100.yaml`:

- `in_channels=3`
- `out_channels=3`
- `image_size=64`
- `num_classes=100`
- `mean=[0.5, 0.5, 0.5]`
- `std=[0.5, 0.5, 0.5]`

Expected checkpoint and output locations:

```text
checkpoints/imagenet100/autoencoders/noise_consistency_large_latent256/
outputs/imagenet100/autoencoders/noise_consistency_large_latent256/
```

`NoiseConsistencyAELarge` supports `image_size=64` through the existing config-driven spatial-size logic.

## 4. Encoder Validation And Selection

Entrypoint:

```bash
python scripts/evaluate_encoder.py compare-encoders --dataset mnist
python scripts/evaluate_encoder.py compare-encoders --dataset cifar10
python scripts/evaluate_encoder.py compare-encoders --dataset imagenet100
```

ImageNet-100 smoke command:

```bash
python scripts/evaluate_encoder.py compare-encoders --dataset imagenet100 --num-samples 8 --batch-size 4
```

Implementation:

- `scripts/evaluate_encoder.py`
- `src/evaluation/encoder_validation.py`

Expected ImageNet-100 output:

```text
experiments/imagenet100/exp_002_encoder_validation/
├── metrics/
├── latent_geometry/
├── covariance/
├── eigenspectrum/
├── reconstructions/
├── report/
└── summary.json
```

## 5. Latent-DDPM Training

Hydra entrypoint:

```bash
python scripts/main.py task=train_latent_DDPM
```

Implementation:

- `scripts/main.py`
- `scripts/train_ddpm.py`
- `src/DDPM_model.py`
- `src/Unet_model.py`
- `configs/train_latent_DDPM/default.yaml`

Noise modes:

- `baseline`: `z_noisy = z + sigma * eps_z`
- `induced`: `z_noisy = E(x + sigma * eps_x)`

ImageNet-100 commands:

```bash
python scripts/main.py dataset=imagenet100 task=train_latent_DDPM train_latent_DDPM.latent_noise_mode=baseline
python scripts/main.py dataset=imagenet100 task=train_latent_DDPM train_latent_DDPM.latent_noise_mode=induced
```

Expected ImageNet-100 checkpoint paths:

```text
checkpoints/imagenet100/ddpm/latent_ddpm_baseline_ae_noise_consistency_imagenet100/
checkpoints/imagenet100/ddpm/latent_ddpm_induced_ae_noise_consistency_imagenet100/
```

The selected encoder checkpoint is configured in `configs/dataset/imagenet100.yaml`:

```text
checkpoints/imagenet100/autoencoders/noise_consistency_large_latent256/autoencoder_checkpoint.pt
```

`train_latent_DDPM.max_samples` is not part of the current DDPM training config. Use `train_latent_DDPM.smoke_model_only=true` for configuration/model-shape checks, or reduce `train_latent_DDPM.batch_size` for manual experiments.

## 6. Frozen-Encoder Latent-DDPM Score Validation

Entrypoint:

```bash
python scripts/evaluate_latent_ddpm_score.py --dataset mnist
python scripts/evaluate_latent_ddpm_score.py --dataset cifar10
python scripts/evaluate_latent_ddpm_score.py --dataset imagenet100
```

Implementation:

- `scripts/evaluate_latent_ddpm_score.py`
- `src/evaluation/latent_ddpm_score_validation.py`

The score definition is unchanged:

```text
score = ||eps_pred||^2
```

Expected ImageNet-100 output:

```text
experiments/imagenet100/exp_003_latent_ddpm_validation/
├── metrics/
├── score_validation/
├── score_distributions/
├── noise_prediction/
├── covariance/
├── report/
└── summary.json
```

## 7. Score-Based Filtering

Hydra entrypoint:

```bash
python scripts/main.py task=filter_dataset
```

Implementation:

- `scripts/filter_dataset.py`
- `src/filters.py`
- `configs/filter_dataset/default.yaml`

Supported branches:

- `filter_dataset.ddpm_branch=baseline`
- `filter_dataset.ddpm_branch=induced`

Filtering modes:

- `top_k`: selects the lowest-score samples globally. Lower score means a more typical sample for the current score definition.
- `quantile`: performs stratified sampling over score quantile bins, selecting the configured fraction from each bin. This preserves score-distribution coverage and avoids collapsing the subset into only the lowest-score region.

The active `quantile` mode is not a single quantile interval. It is a QQ-spread / stratified quantile sampling strategy inherited from the laboratory filtering protocol. Legacy `quantile_low` and `quantile_high` config fields are kept only for compatibility and are not used by active `filter_mode=quantile`.

ImageNet-100 commands:

```bash
python scripts/main.py dataset=imagenet100 task=filter_dataset filter_dataset.ddpm_branch=baseline
python scripts/main.py dataset=imagenet100 task=filter_dataset filter_dataset.ddpm_branch=induced
```

ImageNet-100 smoke command for future manual checks:

```bash
python scripts/main.py dataset=imagenet100 task=filter_dataset filter_dataset.ddpm_branch=induced filter_dataset.max_samples=16 filter_dataset.batch_size=4 filter_dataset.grid_n_images=8 filter_dataset.noisy_grid_n_images=4 filter_dataset.output_root=experiments/imagenet100/exp_005_filtering_induced_smoke filter_dataset.overwrite=true
```

Expected research output:

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

## 8. TDnCNN Downstream Validation

Entrypoint:

```bash
python scripts/train_tdncnn.py
```

Implementation:

- `scripts/train_tdncnn.py`
- `scripts/internal/run_TDnCNN_image_suite.py`
- `scripts/internal/tdncnn_image_runs_config.py`
- `scripts/internal/train_TDnCNN_image.py`
- `src/TDnCNN_image.py`
- `src/tdncnn_datasets.py`

List configured runs:

```bash
python scripts/train_tdncnn.py --dataset imagenet100 --list-runs
```

ImageNet-100 run configs:

- `full`
- `topk_10`
- `topk_10_smoke`

Commands:

```bash
python scripts/train_tdncnn.py --dataset imagenet100 --run topk_10_smoke --epochs 1 --batch-size 2 --max-train-samples 16 --max-test-samples 8
python scripts/train_tdncnn.py --dataset imagenet100 --run topk_10
python scripts/train_tdncnn.py --dataset imagenet100 --run full
```

Routing:

- filtered indices: `experiments/imagenet100/exp_005_filtering/topk/10/selected_indices.npy`
- outputs: `experiments/imagenet100/exp_006_tdncnn/topk_10/`
- checkpoint: `checkpoints/imagenet100/tdncnn/topk_10.pth`

TDnCNN on 64x64 images should use smaller smoke batch sizes than MNIST/CIFAR-10.

## 9. DRUNet Downstream Validation

DRUNet is integrated as a second downstream denoiser for the ImageNet-100 score-filtering experiment. It does not change encoder training, latent-DDPM training, score computation, score validation, filtering, or TDnCNN outputs.

Entrypoint:

```bash
python scripts/train_drunet.py
```

Implementation:

- `scripts/train_drunet.py`
- `scripts/internal/run_DRUNet_image_suite.py`
- `scripts/internal/drunet_image_runs_config.py`
- `scripts/internal/train_DRUNet_image.py`
- `src/DRUNet_image.py`
- `src/external/drunet/network_unet.py`
- `src/external/drunet/basicblock.py`

Model:

- official DPIR DRUNet `UNetRes`;
- wrapped by `OfficialDRUNetAdapter`;
- public project API remains `forward(x, sigma=None)`;
- the adapter creates a noise-level map and calls the official network with `[B, 4, H, W]` input.

Production ImageNet-100 protocol:

- fixed denoising noise: `sigma = 25 / 255`;
- image size: `64 x 64`;
- optimizer: Adam;
- learning rate: `1e-4`;
- scheduler: none;
- seed: `42`;
- epochs: `15`;
- batch size: `64`.

Production run configs:

- `full_sigma25`
- `quantile10_sigma25`
- `topk10_sigma25`

Commands:

```bash
python scripts/train_drunet.py --dataset imagenet100 --list-runs
python scripts/train_drunet.py --dataset imagenet100 --dry-run
python scripts/train_drunet.py --dataset imagenet100 --run quantile10_sigma25 --data-root /workspace/data
python scripts/train_drunet.py --dataset imagenet100 --run topk10_sigma25 --data-root /workspace/data
python scripts/train_drunet.py --dataset imagenet100 --run full_sigma25 --data-root /workspace/data
```

Unified launch script:

```bash
bash scripts/run_drunet_imagenet100_production.sh
```

Routing:

- full run: ImageNet-100 train partition;
- quantile run: `experiments/imagenet100/exp_005_filtering/quantile/10/selected_indices.npy`;
- top-k run: `experiments/imagenet100/exp_005_filtering/topk/10/selected_indices.npy`;
- outputs: `experiments/imagenet100/exp_007_drunet/<run>/`;
- checkpoints: `checkpoints/imagenet100/drunet/<run>.pth`.

Per-run artifacts:

- `config.json`
- `metrics.json`
- `results/metrics_history.csv`
- `results/training_history.csv`
- `results/per_image_metrics.csv`
- `results/loss_curve.png`
- `results/qualitative_triplets.png`
- `<run>_example.png`
- checkpoint `.pth`

Console logging mirrors TDnCNN and reports per epoch:

```text
train_loss
val_loss
psnr
ssim
lpips
fid
```

No separate internal validation command is required. Validation metrics and per-image metrics are written by the DRUNet training runner after each production run.

## 10. External Benchmark Evaluation

External benchmark evaluation is a publication-style inference-only stage for already trained DRUNet checkpoints. It is independent from encoder training, latent-DDPM training, score validation, score-based filtering, and DRUNet training.

Supported benchmarks:

- `Kodak24`
- `CBSD68`
- `Urban100`

Expected read-only dataset root:

```text
data/external_benchmarks/
|-- Kodak24/
|-- CBSD68/
`-- Urban100/
```

Dataset preparation/validation utility:

```bash
python scripts/prepare_external_benchmarks.py
python scripts/prepare_external_benchmarks.py --strict
```

This validates folder presence, counts image files, checks RGB format, and writes:

```text
data/external_benchmarks/dataset_manifest.json
```

Evaluation entrypoint:

```bash
python scripts/evaluate_external.py \
  --checkpoint checkpoints/imagenet100/drunet/full_sigma25.pth \
  --dataset Kodak24 \
  --sigma 25

python scripts/evaluate_external.py \
  --checkpoint checkpoints/imagenet100/drunet/full_sigma25.pth \
  --all-benchmarks \
  --sigma 25
```

Production checkpoints evaluated by this stage:

```text
checkpoints/imagenet100/drunet/full_sigma25.pth
checkpoints/imagenet100/drunet/topk10_sigma25.pth
checkpoints/imagenet100/drunet/quantile10_sigma25.pth
```

Publication comparison commands:

```bash
python scripts/evaluate_external.py \
  --checkpoint checkpoints/imagenet100/drunet/full_sigma25.pth \
  --all-benchmarks \
  --sigma 25

python scripts/evaluate_external.py \
  --checkpoint checkpoints/imagenet100/drunet/topk10_sigma25.pth \
  --all-benchmarks \
  --sigma 25

python scripts/evaluate_external.py \
  --checkpoint checkpoints/imagenet100/drunet/quantile10_sigma25.pth \
  --all-benchmarks \
  --sigma 25
```

The resulting comparison covers:

```text
Kodak24  x Full / Top-K 10% / Quantile-spread 10%
CBSD68   x Full / Top-K 10% / Quantile-spread 10%
Urban100 x Full / Top-K 10% / Quantile-spread 10%
```

Noise protocol:

- Gaussian noise generated on the fly;
- supported sigma values: `15`, `25`, `50`;
- primary protocol: `sigma=25`, internally `25/255`;
- clean ground-truth images are unchanged.

Inference protocol:

- loads a trained DRUNet checkpoint;
- uses `OfficialDRUNetAdapter`;
- pads images to a multiple of 8 before inference;
- crops predictions back to the original resolution;
- does not resize benchmark images;
- does not crop benchmark images;
- does not augment benchmark images.

Metrics:

- PSNR;
- SSIM;
- optional LPIPS with `--lpips` when pretrained weights are available.

Artifacts:

```text
experiments/external_benchmarks/<checkpoint_name>/
|-- summary.json
|-- Kodak24/
|   |-- metrics.csv
|   |-- summary.json
|   |-- report.md
|   `-- qualitative/
|-- CBSD68/
|   |-- metrics.csv
|   |-- summary.json
|   |-- report.md
|   `-- qualitative/
`-- Urban100/
    |-- metrics.csv
    |-- summary.json
    |-- report.md
    `-- qualitative/
```

Qualitative outputs include ground truth, noisy input, denoised output, and an absolute error map for representative images.

Quick checks:

```bash
python scripts/evaluate_external.py --help
python scripts/prepare_external_benchmarks.py
```

Do not run full benchmark evaluation as infrastructure verification.

## 11. Research-Style Plotting Layer

Entrypoint:

```bash
python scripts/generate_research_plots.py --dataset cifar10 --stage all
python scripts/generate_research_plots.py --dataset imagenet100 --stage all
python scripts/generate_research_plots.py --dataset imagenet100 --stage filtering
python scripts/generate_research_plots.py --dataset imagenet100 --stage tdncnn
```

Behavior:

- reads existing experiment artifacts only;
- does not run encoder training, DDPM training, filtering, TDnCNN training, or DRUNet training;
- writes PNG and PDF figures where possible;
- skips missing inputs with warnings by default;
- raises on missing required inputs only with `--strict true`.

Implemented modules:

- `src/evaluation/research_plot_style.py`
- `src/evaluation/research_filtering_plots.py`
- `src/evaluation/research_training_dynamics_plots.py`
- `src/evaluation/research_tdncnn_plots.py`
- `src/evaluation/research_qualitative_plots.py`

Filtering figures use:

```text
experiments/<dataset>/exp_005_filtering/**/scores.csv
experiments/<dataset>/exp_005_filtering/**/selected_indices.npy
experiments/<dataset>/exp_005_filtering/**/metadata.json
```

Outputs:

```text
experiments/<dataset>/exp_005_filtering/plots/research_style/
```

This stage writes score distribution plots for `topk_{5,10,15}`, QQ-spread `quantile_{5,10,15}`, `random_10`, and an overlay. The random baseline is generated only for reporting with `seed=42` and saved as `random_10_indices.npy`; it does not change the filtering protocol.

DDPM dynamics figures use:

```text
checkpoints/<dataset>/ddpm/**/DDPM_metrics.csv
checkpoints/<dataset>/ddpm/**/score_training_dynamics.csv
```

Outputs:

```text
experiments/<dataset>/exp_003_latent_ddpm_validation/plots/research_style/
experiments/<dataset>/exp_003_latent_ddpm_validation/metrics/score_training_dynamics.csv
```

Future DDPM runs can enable optional score-stat logging through `train_latent_DDPM.score_stats.enabled=true`. This records score mean/median/std/skewness/kurtosis and latent-norm summaries on a fixed validation subset. It does not change checkpoint selection.

TDnCNN CDF figures use:

```text
experiments/<dataset>/exp_006_tdncnn/<run_name>/results/per_image_metrics.csv
```

Outputs:

```text
experiments/<dataset>/exp_006_tdncnn/comparison_plots/research_style/
experiments/<dataset>/exp_006_tdncnn/qualitative/research_style/
```

Future TDnCNN runs save per-image metrics with:

```text
sample_index
run_name
mse
psnr
ssim
lpips
noisy_mse
noisy_psnr
noisy_ssim
noisy_lpips
```

DRUNet production runs save the same downstream metric schema under:

```text
experiments/<dataset>/exp_007_drunet/<run_name>/metrics.json
experiments/<dataset>/exp_007_drunet/<run_name>/results/per_image_metrics.csv
experiments/<dataset>/exp_007_drunet/<run_name>/results/training_history.csv
```

The active ImageNet-100 DRUNet run names are:

```text
full_sigma25
quantile10_sigma25
topk10_sigma25
```

Image-domain mapping from the wireless report:

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

Wireless-specific quantities such as PDP, PAP, capacity, UL/DL, rank, fixed noise power, and SNR in dB are not used in this image-domain reporting layer.

## 12. Full ImageNet-100 Manual Sequence

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

# 8. DRUNet downstream validation
python scripts/train_drunet.py --dataset imagenet100 --run quantile10_sigma25 --data-root /workspace/data
python scripts/train_drunet.py --dataset imagenet100 --run topk10_sigma25 --data-root /workspace/data
python scripts/train_drunet.py --dataset imagenet100 --run full_sigma25 --data-root /workspace/data

# 9. External benchmark evaluation for trained DRUNet checkpoints
python scripts/prepare_external_benchmarks.py --strict
python scripts/evaluate_external.py --checkpoint checkpoints/imagenet100/drunet/full_sigma25.pth --all-benchmarks --sigma 25
```

## 13. Verification

Allowed quick checks:

```bash
python -m compileall -q scripts src
python scripts/train_tdncnn.py --dataset imagenet100 --list-runs
python scripts/train_drunet.py --dataset imagenet100 --list-runs
python scripts/train_drunet.py --dataset imagenet100 --dry-run
python scripts/evaluate_external.py --help
python scripts/prepare_external_benchmarks.py
python scripts/generate_research_plots.py --dataset cifar10 --stage filtering --strict false
python scripts/generate_research_plots.py --dataset imagenet100 --stage filtering --strict false
```

Do not run full encoder training, latent-DDPM training, full score validation, full filtering, TDnCNN training, DRUNet training, or full external benchmark evaluation as infrastructure verification.
