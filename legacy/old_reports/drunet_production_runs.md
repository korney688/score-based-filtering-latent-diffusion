# DRUNet ImageNet-100 Production Runs

This document defines the final DRUNet downstream denoising runs for the
ImageNet-100 score-filtering experiment. These commands are intended to be run
inside the project container on the remote GPU host. Do not run them during
preflight.

## Shared Protocol

Goal: test whether score-based filtering can reduce the training set size
without a proportional downstream denoising-quality loss.

Only one experimental variable changes:

```text
training subset composition
```

Fixed conditions for all runs:

```text
dataset: ImageNet-100
data root inside container: /workspace/data
image root used by loader: /workspace/data/imagenet100
model: official DPIR DRUNet UNetRes through OfficialDRUNetAdapter
noise protocol: fixed sigma
sigma: 25 / 255 = 0.09803921568627451
epochs: 15
batch size: 64
optimizer: Adam
learning rate: 1.0e-4
weight decay: 0.0
scheduler: none
seed: 42
num_workers: 0
output root: experiments/imagenet100/exp_007_drunet
checkpoint root: checkpoints/imagenet100/drunet
```

Batch size 64 is selected from the DRUNet memory preflight for a ~33M
parameter DRUNet on 64x64 ImageNet-100 images on a Tesla V100 32GB.

## Run 1: Full Dataset

Run name:

```text
full_sigma25
```

Training data:

```text
/workspace/data/imagenet100/train
```

Selected indices:

```text
none
```

Expected outputs:

```text
experiments/imagenet100/exp_007_drunet/full_sigma25/
checkpoints/imagenet100/drunet/full_sigma25.pth
```

Command:

```bash
python scripts/train_drunet.py \
  --dataset imagenet100 \
  --run full_sigma25 \
  --data-root /workspace/data
```

## Run 2: Quantile-Spread 10%

Run name:

```text
quantile10_sigma25
```

Training data:

```text
/workspace/data/imagenet100/train
```

Selected indices:

```text
experiments/imagenet100/exp_005_filtering/quantile/10/selected_indices.npy
```

Expected outputs:

```text
experiments/imagenet100/exp_007_drunet/quantile10_sigma25/
checkpoints/imagenet100/drunet/quantile10_sigma25.pth
```

Command:

```bash
python scripts/train_drunet.py \
  --dataset imagenet100 \
  --run quantile10_sigma25 \
  --data-root /workspace/data
```

## Run 3: Top-K 10%

Run name:

```text
topk10_sigma25
```

Training data:

```text
/workspace/data/imagenet100/train
```

Selected indices:

```text
experiments/imagenet100/exp_005_filtering/topk/10/selected_indices.npy
```

Expected outputs:

```text
experiments/imagenet100/exp_007_drunet/topk10_sigma25/
checkpoints/imagenet100/drunet/topk10_sigma25.pth
```

Command:

```bash
python scripts/train_drunet.py \
  --dataset imagenet100 \
  --run topk10_sigma25 \
  --data-root /workspace/data
```

## Expected Artifact Files

Each completed run should create:

```text
experiments/imagenet100/exp_007_drunet/<run>/
├── config.json
├── metrics.json
├── <run>_example.png
├── qualitative/
└── results/
    ├── metrics_history.csv
    ├── training_history.csv
    ├── loss_curve.png
    ├── per_image_metrics.csv
    └── qualitative_triplets.png

checkpoints/imagenet100/drunet/<run>.pth
```

## Preflight Commands

These commands do not train models:

```bash
python -m compileall -q scripts src
python scripts/train_drunet.py --dataset imagenet100 --list-runs
python scripts/train_drunet.py --dataset imagenet100 --dry-run
python scripts/train_drunet.py --dataset imagenet100 --init-model
python scripts/train_drunet.py \
  --dataset imagenet100 \
  --run full_sigma25 \
  --init-dataloader \
  --data-root /workspace/data \
  --max-train-samples 16 \
  --max-test-samples 8 \
  --batch-size 2
```
