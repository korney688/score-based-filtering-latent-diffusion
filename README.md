# Score-Based Filtering in Latent Diffusion Pipelines

Research pipeline for encoder training, latent-DDPM score estimation, score-based training-data filtering, and downstream denoising validation with TDnCNN and DRUNet.

The project supports dataset-scoped experiments for `mnist`, `cifar10`, and `imagenet100`. The active ImageNet-derived benchmark is `imagenet100`.

## Pipeline

```text
Encoder Training
-> Encoder Validation
-> Baseline vs Induced Latent-DDPM
-> Score Validation
-> Score-Based Filtering
-> Downstream Denoising Validation (TDnCNN / DRUNet)
-> External Benchmark Evaluation
-> Research Artifacts
```

Dataset-scoped artifacts are written under:

```text
checkpoints/<dataset>/
experiments/<dataset>/
outputs/<dataset>/
```

## Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

On Windows with the local virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Fast verification only:

```bash
python -m compileall -q scripts src
python scripts/train_tdncnn.py --dataset imagenet100 --list-runs
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

## ImageNet-100 Data

Expected local ImageFolder layout:

```text
data/imagenet100/
|-- train/<class_name>/*.JPEG
`-- val/<class_name>/*.JPEG
```

Class folders may use either local names such as `class_000` or ImageNet synset names such as `n01440764`. The dataset must be prepared before training. A helper downloader for the selected HuggingFace ImageNet-100 source is available:

```bash
python scripts/download_imagenet100_hf.py --output-dir data/imagenet100
```

For local pipeline smoke checks without the full ImageNet-100 dataset, a CIFAR-10-based ImageFolder subset can be prepared separately:

```bash
python scripts/prepare_smoke_imagenet_from_cifar10.py
```

## Main Commands

Train noise-consistency encoders:

```bash
python scripts/train_encoder.py noise-consistency --dataset cifar10 --variant large --latent-dim 128
python scripts/train_encoder.py noise-consistency --dataset imagenet100 --variant large --latent-dim 256
```

Validate encoder:

```bash
python scripts/evaluate_encoder.py compare-encoders --dataset imagenet100
```

Train latent-DDPM baseline and induced branches:

```bash
python scripts/main.py dataset=imagenet100 task=train_latent_DDPM train_latent_DDPM.latent_noise_mode=baseline
python scripts/main.py dataset=imagenet100 task=train_latent_DDPM train_latent_DDPM.latent_noise_mode=induced
```

Validate latent-DDPM scores:

```bash
python scripts/evaluate_latent_ddpm_score.py --dataset imagenet100
```

Run score-based filtering:

```bash
python scripts/main.py dataset=imagenet100 task=filter_dataset filter_dataset.ddpm_branch=baseline
python scripts/main.py dataset=imagenet100 task=filter_dataset filter_dataset.ddpm_branch=induced
```

Run TDnCNN downstream validation:

```bash
python scripts/train_tdncnn.py --dataset imagenet100 --list-runs
python scripts/train_tdncnn.py --dataset imagenet100 --run full
python scripts/train_tdncnn.py --dataset imagenet100 --run topk_10
```

Run DRUNet downstream validation:

```bash
python scripts/train_drunet.py --dataset imagenet100 --list-runs
python scripts/train_drunet.py --dataset imagenet100 --dry-run
python scripts/train_drunet.py --dataset imagenet100 --run quantile10_sigma25 --data-root /workspace/data
python scripts/train_drunet.py --dataset imagenet100 --run topk10_sigma25 --data-root /workspace/data
python scripts/train_drunet.py --dataset imagenet100 --run full_sigma25 --data-root /workspace/data
```

DRUNet uses the official DPIR `UNetRes` through `OfficialDRUNetAdapter`. The project API remains `forward(x, sigma=None)`, while the adapter internally concatenates the RGB image with a noise-level map. The production DRUNet protocol uses fixed image-denoising noise `sigma=25/255`, batch size `64`, Adam with learning rate `1e-4`, and 15 epochs.

Evaluate trained DRUNet checkpoints on external denoising benchmarks:

```bash
python scripts/prepare_external_benchmarks.py
python scripts/evaluate_external.py --checkpoint checkpoints/imagenet100/drunet/full_sigma25.pth --dataset Kodak24 --sigma 25
python scripts/evaluate_external.py --checkpoint checkpoints/imagenet100/drunet/full_sigma25.pth --all-benchmarks --sigma 25
```

External benchmarks are read-only folders under `data/external_benchmarks/{Kodak24,CBSD68,Urban100}`. The external stage evaluates already trained DRUNet checkpoints for `full_sigma25`, `topk10_sigma25`, and `quantile10_sigma25`; it does not retrain models or rerun filtering. Evaluation uses original image resolution, on-the-fly Gaussian noise, and DRUNet inference with padding to multiples of 8. Each benchmark run writes `metrics.csv`, `summary.json`, `report.md`, and qualitative images under `experiments/external_benchmarks/<checkpoint_name>/<benchmark>/`.

Typical external benchmark commands:

```bash
python scripts/evaluate_external.py --checkpoint checkpoints/imagenet100/drunet/full_sigma25.pth --all-benchmarks --sigma 25
python scripts/evaluate_external.py --checkpoint checkpoints/imagenet100/drunet/topk10_sigma25.pth --all-benchmarks --sigma 25
python scripts/evaluate_external.py --checkpoint checkpoints/imagenet100/drunet/quantile10_sigma25.pth --all-benchmarks --sigma 25
```

## Filtering Modes

The default filtering config runs `top_k` and `quantile` filtering at `5%`, `10%`, and `15%`.

- `top_k`: selects the lowest-score samples globally. Lower score means a more typical sample for the current score definition.
- `quantile`: performs stratified sampling over score quantile bins, selecting the configured fraction from each bin. This preserves score-distribution coverage and avoids collapsing the subset into only the lowest-score region.

The active `quantile` mode is not a single quantile interval. It is a QQ-spread / stratified quantile sampling strategy.

Per filtering directory, expected files include:

```text
scores.csv
selected_indices.npy
metadata.json
config.yaml
score_histogram.png
best_samples_grid.png
worst_samples_grid.png
selected_samples_grid.png
rejected_samples_grid.png
```

For `filter_mode=quantile`, `metadata.json` records `algorithm=quantile_spread`, `keep_ratio`, `min_points_per_bin`, `seed`, `n_bins`, and the actual selected count.

## Research Artifacts

The reporting layer generates analysis artifacts from existing experiment outputs. It does not train models, run filtering, or change the score definition.

Entrypoint:

```bash
python scripts/generate_research_plots.py --dataset imagenet100 --stage all
python scripts/generate_research_plots.py --dataset imagenet100 --stage filtering
python scripts/generate_research_plots.py --dataset imagenet100 --stage tdncnn
```

Generated artifacts include score-distribution figures, selected/rejected sample grids, DDPM training dynamics when available, downstream metric comparisons, summary tables, and qualitative denoising examples.

Outputs are saved under:

```text
experiments/<dataset>/exp_005_filtering/plots/research_style/
experiments/<dataset>/exp_003_latent_ddpm_validation/plots/research_style/
experiments/<dataset>/exp_006_tdncnn/comparison_plots/research_style/
experiments/<dataset>/exp_006_tdncnn/qualitative/research_style/
experiments/<dataset>/exp_007_drunet/
experiments/external_benchmarks/
```

Required inputs are existing `scores.csv`, `selected_indices.npy`, filtering grids, DDPM `DDPM_metrics.csv` or optional `score_training_dynamics.csv`, and downstream `results/per_image_metrics.csv` files from TDnCNN or DRUNet runs. Missing inputs are skipped with warnings when `--strict false`.

## Docker

The repository includes a Dockerfile for DGX and CUDA-enabled runs. The container opens `bash` by default and does not start training automatically.

Build locally:

```bash
docker build -t latent-ddpm-score-filtering:latest .
```

Example run with mounted project artifacts:

```bash
docker run --rm -it --gpus all \
  -v /path/to/data:/workspace/data \
  -v /path/to/checkpoints:/workspace/checkpoints \
  -v /path/to/experiments:/workspace/experiments \
  -v /path/to/outputs:/workspace/outputs \
  latent-ddpm-score-filtering:latest
```

## Automation Scripts

ImageNet-100 DGX scripts are provided for reproducible manual launches:

```bash
scripts/run_imagenet100_dgx_smoke.sh
scripts/run_imagenet100_full.sh
scripts/run_imagenet100_downstream_grid.sh
scripts/run_drunet_imagenet100_production.sh
```

They are intended to be inspected before execution and run manually inside the prepared environment.

## Project Layout

```text
configs/      Hydra configs, including dataset configs
scripts/      Public entrypoints and internal runners
src/          Model, dataset, filtering, and evaluation code
tests/        Fast unit-level checks
docs/         Supporting documentation
data/         Local datasets, not committed
checkpoints/  Local model checkpoints, not committed
experiments/  Local experiment artifacts, not committed
outputs/      Local generated outputs, not committed
```

