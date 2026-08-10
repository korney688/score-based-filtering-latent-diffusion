# DRUNet ImageNet-100 Production Runbook

This runbook is for the final DRUNet downstream denoising comparison on the
remote GPU host. It assumes the Docker image and ImageNet-100 volume are already
available.

## Where to Run

Run inside the project container on the DGX/V100 environment:

```text
GPU: Tesla V100-SXM2-32GB
CPU: 12
RAM: 64 GB
working directory: /workspace
dataset root: /workspace/data/imagenet100
```

The dataset must use ImageFolder layout:

```text
/workspace/data/imagenet100/train/<class>/*.jpg
/workspace/data/imagenet100/val/<class>/*.jpg
```

Required filtering artifacts:

```text
experiments/imagenet100/exp_005_filtering/quantile/10/selected_indices.npy
experiments/imagenet100/exp_005_filtering/topk/10/selected_indices.npy
```

## Launch Command

From `/workspace`:

```bash
bash scripts/run_drunet_imagenet100_production.sh
```

If a previous partial run already created output directories or checkpoints and
you intentionally want to continue without automatic deletion:

```bash
FORCE=1 bash scripts/run_drunet_imagenet100_production.sh
```

`FORCE=1` does not delete anything. It only disables the fail-fast guard for
existing output paths.

## Execution Order

The script runs in this order:

```text
1. Smoke checks
2. quantile10_sigma25
3. topk10_sigma25
4. full_sigma25
5. artifact summary
```

The filtered 10% runs are executed before the full run so that shorter
experiments finish first and provide early downstream evidence.

## Fixed Production Protocol

All three runs use identical settings except for the training subset:

```text
model: official DPIR DRUNet UNetRes
resolution: 64x64
noise: fixed sigma = 25/255
epochs: 15
batch size: 64
optimizer: Adam
learning rate: 1e-4
weight decay: 0.0
scheduler: none
seed: 42
```

Runs:

```text
full_sigma25
quantile10_sigma25
topk10_sigma25
```

## Evaluation

No separate evaluation command is required.

The DRUNet training runner performs validation after each epoch and writes the
final evaluation artifacts at the end of each run:

```text
metrics.json
results/per_image_metrics.csv
results/training_history.csv
```

Metrics include:

```text
PSNR
SSIM
LPIPS
FID, if torch-fidelity is available
validation_loss
```

If `torch-fidelity` is unavailable, FID is written as `NaN` by the existing
metric path.

## Result Locations

Output directories:

```text
experiments/imagenet100/exp_007_drunet/quantile10_sigma25/
experiments/imagenet100/exp_007_drunet/topk10_sigma25/
experiments/imagenet100/exp_007_drunet/full_sigma25/
```

Checkpoints:

```text
checkpoints/imagenet100/drunet/quantile10_sigma25.pth
checkpoints/imagenet100/drunet/topk10_sigma25.pth
checkpoints/imagenet100/drunet/full_sigma25.pth
```

Final metric files:

```text
experiments/imagenet100/exp_007_drunet/<run>/metrics.json
experiments/imagenet100/exp_007_drunet/<run>/results/per_image_metrics.csv
experiments/imagenet100/exp_007_drunet/<run>/results/training_history.csv
```

Qualitative outputs:

```text
experiments/imagenet100/exp_007_drunet/<run>/<run>_example.png
experiments/imagenet100/exp_007_drunet/<run>/results/qualitative_triplets.png
```

## Success Criteria

The run is successful when the script prints:

```text
DRUNET PRODUCTION PASSED
```

and each run has:

```text
config.json
metrics.json
results/per_image_metrics.csv
results/training_history.csv
checkpoint .pth
```

Any failed command stops the script immediately.
