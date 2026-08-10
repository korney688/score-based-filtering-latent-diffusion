# DRUNet Official Import Report

Date: 2026-06-20

## Imported Files

The manually provided DPIR files were found and kept as the only vendored
official DRUNet sources:

```text
src/external/drunet/network_unet.py
src/external/drunet/basicblock.py
```

Added package marker:

```text
src/external/drunet/__init__.py
```

No other DPIR or KAIR files were imported.

## Import Fixes

The DPIR repository-relative import in `network_unet.py` was changed from:

```python
import models.basicblock as B
```

to:

```python
from . import basicblock as B
```

This keeps the official network files local to `src.external.drunet`.

## Project Files Changed

```text
src/DRUNet_image.py
src/external/drunet/network_unet.py
src/external/drunet/__init__.py
scripts/internal/drunet_image_runs_config.py
scripts/internal/run_DRUNet_image_suite.py
scripts/internal/train_DRUNet_image.py
configs/downstream/drunet.yaml
docs/drunet_import_report.md
```

The DRUNet-specific config was updated to select the official adapter by
default. Run names, selected-index paths, output roots, checkpoint roots, and
artifact layout remain unchanged.

## Official Adapter

Implemented:

```text
OfficialDRUNetAdapter
```

Location:

```text
src/DRUNet_image.py
```

The adapter wraps DPIR:

```python
UNetRes(
    in_nc=4,
    out_nc=3,
    nc=[64, 128, 256, 512],
    nb=4,
    act_mode="R",
    downsample_mode="strideconv",
    upsample_mode="convtranspose",
)
```

The public project API is preserved:

```python
forward(x, sigma=None)
```

For official DRUNet, `sigma=None` raises a clear error:

```text
Official DRUNet requires a noise level: call forward(x, sigma=...).
```

Supported sigma inputs:

```text
float
tensor[B]
tensor[B, 1]
tensor[B, 1, 1, 1]
```

All are converted internally to:

```text
[B, 1, H, W]
```

and concatenated with the RGB input:

```text
[B, 3, H, W] + [B, 1, H, W] -> [B, 4, H, W]
```

## Sigma-Aware DRUNet Training Path

Official DRUNet cannot be trained through the TDnCNN helper functions because
those call:

```python
model(x_noisy)
```

and do not expose the sampled sigma. To keep shared TDnCNN code unchanged,
`scripts/internal/train_DRUNet_image.py` now uses DRUNet-local equivalents:

```text
OnlineNoisyDatasetWithSigma
create_drunet_dataloaders
train_one_epoch
evaluate
```

These return/use:

```text
x_noisy, x_clean, sigma
```

and call:

```python
model(x_noisy, sigma=sigma)
```

No shared dataset code, TDnCNN code, filtering code, score code, or artifact
generation contract was changed.

## Smoke Checks

Commands executed:

```powershell
.venv\Scripts\python.exe -m compileall -q scripts src
.venv\Scripts\python.exe scripts\train_drunet.py --dataset imagenet100 --dry-run
.venv\Scripts\python.exe scripts\train_drunet.py --dataset imagenet100 --init-model
.venv\Scripts\python.exe scripts\train_drunet.py --dataset imagenet100 --run full --init-dataloader --data-root data\smoke_imagenet --max-train-samples 16 --max-test-samples 8 --batch-size 2
```

Results:

```text
compileall: passed
dry-run: passed; runs resolved: full, quantile_10, topk_10
init-model: OfficialDRUNetAdapter
parameters: 32640960
placeholder: False
init-dataloader: train_batches=8, test_batches=4
```

Forward-pass check:

```text
input shape:  [2, 3, 64, 64]
sigma:        25/255
output shape: [2, 3, 64, 64]
```

Sigma compatibility check:

```text
float sigma:           output [2, 3, 64, 64]
tensor[B]:             output [2, 3, 64, 64]
tensor[B, 1]:          output [2, 3, 64, 64]
tensor[B, 1, 1, 1]:    output [2, 3, 64, 64]
sigma=None:            ValueError with explicit message
```

No training, DGX run, filtering, encoder, DDPM, TDnCNN, or plotting job was
started.

## Compatibility Confirmation

Unchanged:

```text
encoder pipeline
latent DDPM pipeline
score computation
score validation
filtering artifacts
TDnCNN implementation
research plotting
existing experiment outputs
```

`scripts/train_drunet.py` remains the public entrypoint. The DRUNet run set is
still:

```text
full
quantile_10
topk_10
```

The selected-index paths still point to existing filtering artifacts:

```text
experiments/imagenet100/exp_005_filtering/quantile/10/selected_indices.npy
experiments/imagenet100/exp_005_filtering/topk/10/selected_indices.npy
```

## Readiness

The DRUNet infrastructure is now ready for future downstream runs:

```text
full
quantile_10
topk_10
```

No additional infrastructure changes are required before launching those runs.
Only normal training-time considerations remain: GPU memory, dataset mount, and
experiment output storage.
