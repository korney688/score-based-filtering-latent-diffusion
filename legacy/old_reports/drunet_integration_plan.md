# DRUNet Integration Plan

This note documents how to add DRUNet as a second downstream denoiser for the
ImageNet-100 score-filtering experiments without changing the encoder, latent
DDPM, filtering, or TDnCNN pipeline.

## Current TDnCNN downstream interface

The current downstream layer is organized around:

- `scripts/train_tdncnn.py`: public entrypoint.
- `scripts/internal/run_TDnCNN_image_suite.py`: CLI runner with `--dataset`,
  `--run`, `--list-runs`, epoch/batch-size overrides, and smoke sample limits.
- `scripts/internal/tdncnn_image_runs_config.py`: run definitions and artifact
  roots.
- `scripts/internal/train_TDnCNN_image.py`: train/validate/checkpoint/report
  implementation.
- `src/TDnCNN_image.py`: model, training/evaluation metrics, PSNR/SSIM/LPIPS/FID
  helpers.
- `src/tdncnn_datasets.py`: reusable ImageFolder/CIFAR/MNIST data loading,
  filtered subset application, and online Gaussian noise generation.

The runner calls a single `run_experiment(...)` function with:

- `experiment_name`
- `mode` as `full` or `filtered`
- optional `filtered_indices_path`
- output/checkpoint roots
- dataset name and data root
- noise range, seed, batch size, epochs, and smoke limits

## Produced TDnCNN artifacts

For each run, TDnCNN currently writes:

- `config.json`
- `metrics.json`
- `<run>_example.png`
- `results/metrics_history.csv`
- `results/loss_curve.png`
- `results/metrics_grid.png`
- `results/per_image_metrics.csv`
- `results/qualitative_triplets.png`
- `results/error_maps.png`
- checkpoint under `checkpoints/<dataset>/tdncnn/<run>.pth`

The ImageNet-100 production grid is under:

- `experiments/imagenet100/exp_006_tdncnn/`
- `checkpoints/imagenet100/tdncnn/`

## Reusable pieces for DRUNet

DRUNet can reuse these project components directly:

- `src.tdncnn_datasets.create_dataloaders` for full/filtered train splits and
  online noisy image pairs.
- Existing `selected_indices.npy` from
  `experiments/imagenet100/exp_005_filtering/topk/10/` and
  `experiments/imagenet100/exp_005_filtering/quantile/10/`.
- Metric helpers from `src.TDnCNN_image` for PSNR, SSIM, LPIPS, and FID.
- The same output convention: `config.json`, `metrics.json`,
  `results/per_image_metrics.csv`, `results/metrics_history.csv`, qualitative
  grids, and model checkpoints.

## DRUNet integration boundary

The project now contains `src/DRUNet_image.py` with a placeholder model and a
`build_drunet(...)` factory. This is intentionally not the official DRUNet. The
official implementation should be added manually later, then wired into
`build_drunet(...)` without changing the upstream score-filtering artifacts.

Prepared production runs:

- `full`
- `quantile_10`
- `topk_10`

Prepared future artifact roots:

- `checkpoints/imagenet100/drunet/`
- `experiments/imagenet100/exp_007_drunet/full/`
- `experiments/imagenet100/exp_007_drunet/quantile_10/`
- `experiments/imagenet100/exp_007_drunet/topk_10/`
