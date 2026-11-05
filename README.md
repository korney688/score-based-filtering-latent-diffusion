# DDPM score-based filtering for image denoising

This project studies whether selecting training images by an image-space DDPM score improves denoising. It compares Colored MNIST, CIFAR-10 and ImageNet-100 with TDnCNN and official DRUNet models. Each architecture uses Full, Top-K, Quantile-spread and Random subsets at 10%, 20% and 30%.

The pipeline is fixed corpus corruption -> DDPM training -> Score Validation and checkpoint/timestep selection -> filtering -> downstream denoising -> evaluation. See the [experiment protocol](docs/experiment_protocol.md) for formulas, budgets and comparison rules, and the [SIDD protocol](docs/sidd_evaluation.md) for native noisy/GT evaluation.

## Setup and data

Run commands from the repository root in an isolated Python environment:

```sh
python -m pip install -r requirements.txt
```

Local checks use Python 3.10 on Windows; Docker uses Python 3.11 with PyTorch 2.7.1/CUDA 11.8. Exact scientific dependency pins are in requirements.txt. Full experiments require CUDA and separately supplied LPIPS pretrained weights. Local tests and their dev configuration are excluded from Git.

| Input | Expected location |
|---|---|
| Colored MNIST | torchvision MNIST under `data/`; RGB color/background seeds are both 42 |
| CIFAR-10 | torchvision CIFAR-10 under `data/`; dataset settings in `configs/dataset/cifar10.yaml` |
| ImageNet-100 | ImageFolder `data/imagenet100/{train,val}/`, 100 classes, 130000/5000 images |
| Synthetic external benchmarks | clean RGB images in `data/external_benchmarks/{Kodak24,CBSD68,Urban100}/` |
| SIDD | native Validation sRGB noisy/GT MAT files in `data/sidd/validation/` |

Production configs disable automatic downloads. `scripts/download_imagenet100_hf.py --help` describes optional ImageNet preparation. Data, checkpoints and experiment results are external inputs.

## Run an experiment

Each primary CLI requires `--run-id` and `--experiment-root`. Start with a new root and inspect the plan with `--dry-run`. Removing that flag starts the requested experiment.

```sh
python scripts/run_colored_mnist_pipeline.py --config configs/experiments/colored_mnist_seed42.yaml --run-id colored_seed42 --experiment-root experiments/colored_mnist/colored_seed42 --stage full_grid --dry-run
python scripts/run_cifar10_pipeline.py --config configs/experiments/cifar10_seed42.yaml --run-id cifar10_001_final_20260901 --experiment-root experiments/cifar10/cifar10_001_final_20260901 --stage internal_all --dry-run
python scripts/run_imagenet100_pipeline.py --config configs/experiments/imagenet100_seed42.yaml --run-id imagenet100_image_space_variable_awgn_001 --experiment-root experiments/imagenet100_image_space_variable_awgn_v1/imagenet100_image_space_variable_awgn_001 --stage internal_all --dry-run
```

These stages cover the internal experiment. CIFAR's `all` stage also includes external evaluation; use `internal_all` for the internal grid. Resume requires matching run, source and config identity. Keep the source snapshot used for an existing run; do not edit its ownership markers.

## Corpus-noise replicates

Configs under `configs/experiments/` provide corpus seeds 42/43/44. Each dataset uses `<dataset>_seed42.yaml` for seed42; `colored_mnist.yaml` is its driver's equality template. Other seeds, budgets and scientific parameters stay fixed.

```sh
python scripts/run_cifar10_noise_seed_replicates.py dry-run
python scripts/run_colored_mnist_noise_seed_replicates.py dry-run
python scripts/remote/imagenet100_noise_seed_replicates.py dry-run
```

Use each driver's `preflight` before `run`. Colored 43/44 must use their driver: it keeps validation-ladder seed42 separate from corpus seed. Direct primary CLI calls with those two YAML files omit that adapter. CIFAR and Colored launchers require Windows (`msvcrt` locks); the ImageNet driver uses Linux and its shell launcher uses `flock`.

## Reproducibility limits

Individual primary experiments need data and dependencies. The replicate drivers additionally require baseline evidence, and final aggregation expects the registered artifact layout:

| Dataset | Required baseline and replicate roots |
|---|---|
| CIFAR-10 | `experiments/cifar10/cifar10_001_final_20260901` and siblings ending `_seed43`, `_seed44` |
| ImageNet-100 | `experiments/imagenet100_image_space_variable_awgn_v1/imagenet100_image_space_variable_awgn_001` and siblings ending `_seed43`, `_seed44` |
| Colored MNIST | driver outputs `experiments/colored_mnist/colored_mnist_001_seed42_canonical`, `_seed43`, `_seed44`; preflight separately reads `experiments/colored_mnist/colored_mnist_001` |

CIFAR/ImageNet need a completed seed42 run: resolved configs, split/evaluation indices, corpus or evaluation noise manifests, cached evaluation tensors where used, and all 20 branch metadata/history files. ImageNet also checks its protocol summary and final summary. Default free-space checks are 20 GiB for CIFAR, 200 GiB for ImageNet and 30 GiB for Colored MNIST.

Colored preflight needs the reference root's resolved config, all 10000 test indices, evaluation noise manifest and cached input/target tensors. It checks the same MNIST data, non-corpus seeds and colorization. It recomputes all three runs and reuses no baseline training checkpoints. A clean clone cannot produce the driver-owned triple without that reference evidence; a direct seed42 primary run does not create the required replicate-owner metadata.

The internal final collector needs the default Colored/CIFAR roots plus `results_final/imagenet100/seed42`, `seed43`, `seed44`. Each ImageNet archive needs `DOWNLOAD_VERIFIED.json`, configs, filtering indices, evaluation results, actual figures, summaries and complete branch metadata/history. It also needs `imagenet100_remote_inventory.json` and `imagenet100_remote_completeness.json` in `results_final/`. The collector uses exact layouts rather than searching arbitrary runs.

`scripts/collect_imagenet_final_results.py` supplies remote inventory and verified downloads. The remote completeness JSON must be supplied separately; no CLI here regenerates it. A new local ImageNet run needs a separately validated export adapter to become this archive layout. External aggregation also needs all 120 per-observation/provenance combinations and the 30 matching DRUNet checkpoints. Do not fabricate transfer or ownership markers to satisfy these requirements.

## Tables, figures and sharing

Given an artifact project with the layout above, write outputs to new directories:

```sh
python scripts/build_final_experiment_results.py --artifact-root <artifact-project> --output-dir results/publication/tables_3seeds
python scripts/final_imagenet_external.py aggregate --artifact-root <artifact-project> --source-root <artifact-project>/results_final/imagenet100_external --output-dir results/publication/imagenet100_external
python scripts/plot_final_results.py --results-root results/publication --scope all --dataset all --output-dir results/figures
```

Internal tables contain 180 per-seed rows and 60 means; external tables contain 120 rows and 40 means. Plots show the mean of three paired Delta PSNR values with a seed min-max band. Output includes PNG/PDF/SVG and numeric plot tables. Plotting checks table manifests; older external bundles need `--verify-source-artifacts` or reaggregation into a new directory.

`final_imagenet_external.py --help` exposes audit, checkpoint verification/download, evaluate and aggregate. `evaluate_external.py` supports standalone synthetic evaluation. The consultant script's `validate` action checks saved tables; its optional Windows watcher builds reports without starting inference. Its validators are also used by plotting.

Internal CSV/Excel/provenance bundles retain absolute source/checkpoint paths for validation and resume. Keep them private. Export sanitized numerical tables separately:

```sh
python scripts/build_consultant_report.py export-publication --results-root results/publication --output-dir results/consultant-share
```

The export preserves numbers and missing LPIPS, omits private paths, and uses relative manifest names. It does not replace the original artifact bundle. Figure validation records relative input-table names.

## Docker

Build the scientific runtime from the public pinned PyTorch image:

```sh
docker build -t research-environment:latest .
```

Use the same image for the ImageNet-100 pipeline and its corpus-noise replicates. Start a shell with data mounted read-only and persistent experiment storage. Replace the host paths and select the intended GPU:

```sh
docker run --rm -it --gpus device=0 --cpus 12 --memory 64g --shm-size 2g --mount type=bind,source=/absolute/path/to/data,target=/workspace/data,readonly --mount type=bind,source=/absolute/path/to/experiments,target=/workspace/experiments research-environment:latest bash
```

Inside the container, run the ImageNet-100 command from [Run an experiment](#run-an-experiment). Paths are relative to /workspace; data/imagenet100 is the mounted ImageFolder. Keep --dry-run to inspect the plan; remove it to produce the seed42 baseline.

After the required baseline is complete, inspect and launch seed43/44 sequentially with the Linux driver:

```sh
bash scripts/remote/run_imagenet100_noise_seed_replicates.sh dry-run
bash scripts/remote/run_imagenet100_noise_seed_replicates.sh preflight
bash scripts/remote/run_imagenet100_noise_seed_replicates.sh run
```

The same data and experiment mounts must remain available. DATA_ROOT, OUTPUT_BASE and SEED42_ROOT can override the container defaults; baseline and replicate roots must remain distinct siblings. The launcher uses flock to prevent overlapping replicate processes. For resume, set RESUME=1 for preflight and run using the same image/source/config identity. Keep existing owner markers and checkpoints unchanged. Baseline evidence and storage requirements are listed in [Reproducibility limits](#reproducibility-limits).

## Source layout

`src/` contains the three pipelines, DDPM/denoiser models, filtering and evaluation helpers. `scripts/` contains CLIs, replicate drivers, aggregation and plotting; `scripts/remote/` contains optional transport and Linux launchers. `configs/` contains dataset settings, the named seed configurations and SIDD settings. The official DRUNet architecture is vendored under `src/external/drunet/`.
