# DRUNet Production Preflight

Date: 2026-06-20

## Production Runs

The final production run set is exactly:

| Run | Training subset | Selected indices |
| --- | --- | --- |
| `full_sigma25` | Full ImageNet-100 training partition | None |
| `quantile10_sigma25` | Quantile-spread 10% subset | `experiments/imagenet100/exp_005_filtering/quantile/10/selected_indices.npy` |
| `topk10_sigma25` | Top-K 10% subset | `experiments/imagenet100/exp_005_filtering/topk/10/selected_indices.npy` |

The local selected-index files were checked and both exist.

## Fixed Sigma Protocol

The previous DRUNet infrastructure used a random uniform sigma range. The
production protocol now uses a fixed denoising noise level:

```text
sigma_mode: fixed
fixed_sigma: 25 / 255 = 0.09803921568627451
sigma_min: 0.09803921568627451
sigma_max: 0.09803921568627451
```

This disables multi-sigma training for the production comparison and matches
the classic image-denoising protocol.

## Shared Conditions

All three runs share:

| Parameter | Value |
| --- | --- |
| Dataset | ImageNet-100 |
| Data root in container | `/workspace/data` |
| Denoiser | official DPIR DRUNet `UNetRes` |
| Input API | `forward(x, sigma=None)` through `OfficialDRUNetAdapter` |
| Sigma | fixed `25/255` |
| Epochs | 15 |
| Batch size | 64 |
| Optimizer | Adam |
| Learning rate | `1.0e-4` |
| Weight decay | `0.0` |
| Scheduler | none |
| Seed | 42 |
| Num workers | 0 |
| Output root | `experiments/imagenet100/exp_007_drunet` |
| Checkpoint root | `checkpoints/imagenet100/drunet` |

The only intended experimental variable is the training subset composition.

## What Differs

| Run | Difference |
| --- | --- |
| `full_sigma25` | Uses every image in the ImageNet-100 training partition. |
| `quantile10_sigma25` | Uses the saved quantile-spread 10% selected indices. |
| `topk10_sigma25` | Uses the saved Top-K 10% selected indices. |

No score computation, filtering, encoder, latent-DDPM, score-validation, TDnCNN,
or plotting configuration is part of this experiment.

## Config Files

Because the project already has `configs/downstream/drunet.yaml`, the
production per-run configs are stored under an equivalent directory:

```text
configs/downstream/drunet_production/full_sigma25.yaml
configs/downstream/drunet_production/quantile10_sigma25.yaml
configs/downstream/drunet_production/topk10_sigma25.yaml
```

The executable run definitions are synchronized in:

```text
scripts/internal/drunet_image_runs_config.py
```

## Expected Artifacts

Each completed run should create:

```text
experiments/imagenet100/exp_007_drunet/<run>/config.json
experiments/imagenet100/exp_007_drunet/<run>/metrics.json
experiments/imagenet100/exp_007_drunet/<run>/results/per_image_metrics.csv
experiments/imagenet100/exp_007_drunet/<run>/results/training_history.csv
experiments/imagenet100/exp_007_drunet/<run>/results/qualitative_triplets.png
checkpoints/imagenet100/drunet/<run>.pth
```

The run also creates `results/metrics_history.csv`, `results/loss_curve.png`,
`<run>_example.png`, and a `qualitative/` directory.

## Checks Performed

Executed without training:

```powershell
.venv\Scripts\python.exe -m compileall -q scripts src
.venv\Scripts\python.exe scripts\train_drunet.py --dataset imagenet100 --list-runs
.venv\Scripts\python.exe scripts\train_drunet.py --dataset imagenet100 --dry-run
```

Observed run list:

```text
full_sigma25
quantile10_sigma25
topk10_sigma25
```

Observed dry-run parameters:

```text
batch_size: 64
epochs: 15
lr: 0.0001
weight_decay: 0.0
scheduler: null
seed: 42
sigma_mode: fixed
fixed_sigma: 0.09803921568627451
```

Both required selected-index files exist locally:

```text
experiments/imagenet100/exp_005_filtering/quantile/10/selected_indices.npy
experiments/imagenet100/exp_005_filtering/topk/10/selected_indices.npy
```

## Launch Commands

Inside the container:

```bash
python scripts/train_drunet.py --dataset imagenet100 --run full_sigma25 --data-root /workspace/data
python scripts/train_drunet.py --dataset imagenet100 --run quantile10_sigma25 --data-root /workspace/data
python scripts/train_drunet.py --dataset imagenet100 --run topk10_sigma25 --data-root /workspace/data
```

## Conclusion

The project is ready to launch:

```text
full_sigma25
quantile10_sigma25
topk10_sigma25
```

No additional infrastructure changes are required for these production DRUNet
runs. Training was not started during this preflight.
