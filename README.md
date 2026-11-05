# DDPM score-based filtering for image denoising

This repository contains the code for my research on training data selection using diffusion models. The main question is whether selecting images by a DDPM-based score can improve the performance of an image denoising model.

The experiments use three datasets: Colored MNIST, CIFAR-10 and ImageNet-100. I train an image-space DDPM to estimate a score for each training image and use these scores to select smaller training subsets.

Four training sets are compared:
- **Full** — the complete training dataset.
- **Top-K** — images with the lowest DDPM scores.
- **Quantile-spread** — images sampled from different parts of the score distribution.
- **Random** — randomly selected images.

For the three filtering methods, I use 10%, 20% and 30% of the training data. The selected subsets are used to train two denoising networks, TDnCNN and DRUNet.

The experiment follows these steps:

1. Add Gaussian noise to the training images.
2. Train a DDPM and choose a checkpoint and timestep for scoring.
3. Calculate scores and select training subsets.
4. Train TDnCNN and DRUNet on each subset.
5. Compare denoising quality with training on the full dataset.

The score definition, noise settings and training parameters are described in [Experiment protocol](docs/experiment_protocol.md).

## Setup

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Training requires a CUDA GPU. The project can also be run in Docker.

### Datasets

| Dataset | Location |
|---|---|
| Colored MNIST | `data/` (torchvision MNIST) |
| CIFAR-10 | `data/` (torchvision CIFAR-10) |
| ImageNet-100 | `data/imagenet100/train/` and `data/imagenet100/val/` |
| Kodak24, CBSD68, Urban100 | `data/external_benchmarks/` |
| SIDD Validation sRGB | `data/sidd/validation/` |

Datasets should be downloaded separately. ImageNet-100 can be prepared using `scripts/download_imagenet100_hf.py`.

Trained checkpoints and experiment results are stored outside Git.

## Running experiments

The main scripts are:

- `scripts/run_colored_mnist_pipeline.py`
- `scripts/run_cifar10_pipeline.py`
- `scripts/run_imagenet100_pipeline.py`

Each script takes a configuration file, a run ID and an output directory.

Examples for seed 42:

**Colored MNIST**

```bash
python scripts/run_colored_mnist_pipeline.py --config configs/experiments/colored_mnist_seed42.yaml --run-id colored_seed42 --experiment-root experiments/colored_mnist/colored_seed42 --stage full_grid --dry-run
```

**CIFAR-10**

```bash
python scripts/run_cifar10_pipeline.py --config configs/experiments/cifar10_seed42.yaml --run-id cifar10_001_final_20260901 --experiment-root experiments/cifar10/cifar10_001_final_20260901 --stage internal_all --dry-run
```

**ImageNet-100**

```bash
python scripts/run_imagenet100_pipeline.py --config configs/experiments/imagenet100_seed42.yaml --run-id imagenet100_image_space_variable_awgn_001 --experiment-root experiments/imagenet100_image_space_variable_awgn_v1/imagenet100_image_space_variable_awgn_001 --stage internal_all --dry-run
```

The `--dry-run` option shows the planned stages. Remove it to run the experiment. For an actual run, use an empty output directory.

### Experiments with different noise seeds

I use three corpus-noise seeds: **42, 43 and 44**. The other random seeds and training parameters remain fixed so that the results can be compared.

The configuration files are in `configs/experiments/`.

The replicate scripts are:

```bash
python scripts/run_cifar10_noise_seed_replicates.py dry-run
python scripts/run_colored_mnist_noise_seed_replicates.py dry-run
python scripts/remote/imagenet100_noise_seed_replicates.py dry-run
```

Use `preflight` to check the data and saved baseline before starting the replicates, then `run` to execute them.

The replicate scripts require the completed baseline experiments and their saved metadata. Colored MNIST additionally uses an earlier reference run to check its evaluation data. A new individual run can be started using the main commands above.

The CIFAR-10 and Colored MNIST replicate launchers run on Windows. The ImageNet-100 launcher runs on Linux.

## Evaluation and results

For internal evaluation, I compare TDnCNN and DRUNet on all three datasets using PSNR, SSIM, MSE and LPIPS where available.

External evaluation uses Kodak24, CBSD68 and Urban100 with Gaussian noise at σ = 25/255. I also evaluate ImageNet-100 denoisers on real noisy images from SIDD. The [SIDD evaluation protocol](docs/sidd_evaluation.md) describes that experiment.

The final comparisons use the three corpus-noise seeds. For each seed, I calculate the PSNR difference between a filtered subset and the corresponding Full model.

### Tables and figures

The final tables can be generated from saved experiment results:

```bash
python scripts/build_final_experiment_results.py --artifact-root <artifact-project> --output-dir results/publication/tables_3seeds
python scripts/final_imagenet_external.py aggregate --artifact-root <artifact-project> --source-root <artifact-project>/results_final/imagenet100_external --output-dir results/publication/imagenet100_external
```

To generate the final figures:

```bash
python scripts/plot_final_results.py --results-root results/publication --scope all --dataset all --output-dir results/figures
```

The plots show the mean PSNR difference over three seeds. Shaded regions indicate the minimum and maximum values.

The aggregation scripts use the saved experiment outputs and metadata. The full ImageNet-100 aggregation also requires the archived results from the remote runs.

To export numerical tables for sharing:

```bash
python scripts/build_consultant_report.py export-publication --results-root results/publication --output-dir results/consultant-share
```

## Docker

A Dockerfile is included for running the experiments in a CUDA environment.

Build the image:

```bash
docker build -t research-environment:latest .
```

Start a container with the dataset and experiment directories mounted:

```bash
docker run --rm -it --gpus device=0 --cpus 12 --memory 64g --shm-size 2g --mount type=bind,source=/absolute/path/to/data,target=/workspace/data,readonly --mount type=bind,source=/absolute/path/to/experiments,target=/workspace/experiments research-environment:latest bash
```

The ImageNet-100 pipeline can be run inside the container using the commands above.

For the ImageNet-100 noise-seed replicates, use:

```bash
bash scripts/remote/run_imagenet100_noise_seed_replicates.sh preflight
bash scripts/remote/run_imagenet100_noise_seed_replicates.sh run
```

These commands require a completed seed-42 baseline in the same experiment directory.

## Repository structure

- `src/` — DDPM, denoising models, filtering methods and dataset pipelines.
- `scripts/` — experiment launchers, evaluation, result aggregation and plotting.
- `configs/` — dataset and experiment configurations for each noise seed.
- `docs/` — experimental protocol and SIDD evaluation details.

The DRUNet implementation is based on the official architecture and is included under `src/external/drunet/`.