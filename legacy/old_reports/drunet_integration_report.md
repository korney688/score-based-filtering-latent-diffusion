# DRUNet Integration Report

## Created files

- `src/DRUNet_image.py`
- `scripts/train_drunet.py`
- `scripts/internal/drunet_image_runs_config.py`
- `scripts/internal/run_DRUNet_image_suite.py`
- `scripts/internal/train_DRUNet_image.py`
- `configs/downstream/drunet.yaml`
- `docs/drunet_integration_plan.md`
- `docs/drunet_import_files.md`
- `docs/drunet_integration_report.md`

## Changed files

No existing encoder, latent-DDPM, filtering, TDnCNN, or experiment-output files
were modified.

## Prepared run definitions

The DRUNet runner is prepared for the ImageNet-100 downstream grid:

- `full`
- `quantile_10`
- `topk_10`

Filtered runs reuse existing production filtering artifacts:

- `experiments/imagenet100/exp_005_filtering/quantile/10/selected_indices.npy`
- `experiments/imagenet100/exp_005_filtering/topk/10/selected_indices.npy`

## Future artifact roots

- `checkpoints/imagenet100/drunet/`
- `experiments/imagenet100/exp_007_drunet/full/`
- `experiments/imagenet100/exp_007_drunet/quantile_10/`
- `experiments/imagenet100/exp_007_drunet/topk_10/`

## Smoke commands

List configured DRUNet runs:

```bash
python scripts/train_drunet.py --dataset imagenet100 --list-runs
```

Validate production run paths without training:

```bash
python scripts/train_drunet.py --dataset imagenet100 --dry-run
```

Construct the placeholder DRUNet model:

```bash
python scripts/train_drunet.py --dataset imagenet100 --init-model
```

Construct dataloaders against the local CIFAR-derived smoke ImageFolder:

```bash
python scripts/train_drunet.py --dataset imagenet100 --run full --init-dataloader --data-root data/smoke_imagenet --max-train-samples 16 --max-test-samples 8 --batch-size 2
```

The script intentionally refuses real training while `src/DRUNet_image.py` still
uses the placeholder model, unless `--allow-placeholder-training` is explicitly
provided for local plumbing experiments.

## Verification performed

The following allowed checks were executed without training:

```bash
python -m compileall -q scripts src
python scripts/train_drunet.py --dataset imagenet100 --list-runs
python scripts/train_drunet.py --dataset imagenet100 --dry-run
python scripts/train_drunet.py --dataset imagenet100 --init-model
python scripts/train_drunet.py --dataset imagenet100 --run full --init-dataloader --data-root data/smoke_imagenet --max-train-samples 16 --max-test-samples 8 --batch-size 2
```

Observed run discovery:

```text
full
quantile_10
topk_10
```

Observed model initialization:

```text
drunet_model=DRUNetPlaceholder
parameters=114307
placeholder=True
```

Observed smoke dataloader initialization:

```text
Train objects: 16
Test objects: 8
train_batches=8
test_batches=4
```

## Official DRUNet insertion point

Manual official import should happen in:

```text
src/external/drunet/
```

Then update:

```text
src/DRUNet_image.py
```

Specifically, replace the `official=True` branch in `build_drunet(...)` with the
official DRUNet constructor while preserving the project-facing denoiser API.
