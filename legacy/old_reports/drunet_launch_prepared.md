# DRUNet Launch Prepared

Date: 2026-06-20

Final launch infrastructure for DRUNet ImageNet-100 production downstream
runs has been prepared. No training, DGX job, filtering, encoder, DDPM, TDnCNN,
or plotting job was started.

## Production Configs

Executable DRUNet run definitions:

```text
scripts/internal/drunet_image_runs_config.py
```

Human-readable shared config:

```text
configs/downstream/drunet.yaml
```

Per-run production configs:

```text
configs/downstream/drunet_production/full_sigma25.yaml
configs/downstream/drunet_production/quantile10_sigma25.yaml
configs/downstream/drunet_production/topk10_sigma25.yaml
```

All three production runs now use:

```text
batch_size: 64
epochs: 15
sigma: 25/255
learning_rate: 1e-4
optimizer: Adam
weight_decay: 0.0
scheduler: null
seed: 42
```

The only experimental variable is the training subset:

```text
full ImageNet-100
quantile-spread 10%
top-k 10%
```

## Launch Script

Created:

```text
scripts/run_drunet_imagenet100_production.sh
```

The script performs:

```text
1. compile smoke check
2. run discovery
3. dry-run validation
4. quantile10_sigma25 training
5. topk10_sigma25 training
6. full_sigma25 training
7. artifact checks and summary
```

It also checks:

```text
/workspace/data/imagenet100/train
/workspace/data/imagenet100/val
experiments/imagenet100/exp_005_filtering/quantile/10/selected_indices.npy
experiments/imagenet100/exp_005_filtering/topk/10/selected_indices.npy
```

The script fails fast if expected output directories/checkpoints already exist,
unless:

```bash
FORCE=1
```

is set by the operator.

## Runbook

Created:

```text
docs/drunet_production_runbook.md
```

The runbook documents where to launch, execution order, expected result paths,
metric files, and success criteria.

## Final Artifact Paths

Expected output directories:

```text
experiments/imagenet100/exp_007_drunet/quantile10_sigma25/
experiments/imagenet100/exp_007_drunet/topk10_sigma25/
experiments/imagenet100/exp_007_drunet/full_sigma25/
```

Expected checkpoints:

```text
checkpoints/imagenet100/drunet/quantile10_sigma25.pth
checkpoints/imagenet100/drunet/topk10_sigma25.pth
checkpoints/imagenet100/drunet/full_sigma25.pth
```

Expected final metrics:

```text
experiments/imagenet100/exp_007_drunet/<run>/metrics.json
experiments/imagenet100/exp_007_drunet/<run>/results/per_image_metrics.csv
experiments/imagenet100/exp_007_drunet/<run>/results/training_history.csv
```

Expected qualitative artifacts:

```text
experiments/imagenet100/exp_007_drunet/<run>/<run>_example.png
experiments/imagenet100/exp_007_drunet/<run>/results/qualitative_triplets.png
```

## Evaluation Status

No separate evaluation command is needed. The current DRUNet runner saves:

```text
metrics.json
per_image_metrics.csv
training_history.csv
PSNR
SSIM
LPIPS
FID when torch-fidelity is installed
```

as part of the training run.

## Launch Command

Inside `/workspace` on the remote GPU container:

```bash
bash scripts/run_drunet_imagenet100_production.sh
```

The launch infrastructure is ready for:

```text
quantile10_sigma25
topk10_sigma25
full_sigma25
```
