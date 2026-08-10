# External Benchmark Evaluation Readiness Report

Date: 2026-06-20

This audit covers the external DRUNet denoising benchmark evaluation layer only.
No model training, full benchmark evaluation, encoder, DDPM, filtering, or
DRUNet production run was executed.

## Dataset Discovery

Root:

```text
data/external_benchmarks/
```

Detected datasets:

| Dataset | Exists | Images | Formats |
| --- | --- | ---: | --- |
| Kodak24 | yes | 24 | PNG |
| CBSD68 | yes | 68 | PNG |
| Urban100 | yes | 100 | PNG |

Strict manifest validation passed:

```text
Kodak24: exists=True images=24 valid_rgb=24 invalid=0
CBSD68: exists=True images=68 valid_rgb=68 invalid=0
Urban100: exists=True images=100 valid_rgb=100 invalid=0
```

Manifest:

```text
data/external_benchmarks/dataset_manifest.json
```

## File Format Handling

`scripts/evaluate_external.py` supports:

```text
.jpg
.jpeg
.png
.bmp
.tif
.tiff
.webp
```

Images are loaded with PIL and converted through:

```python
image.convert("RGB")
```

This means PNG and JPG/JPEG are supported, and grayscale or palette images are
converted to RGB tensors before evaluation.

## Checkpoint Interface

Expected evaluation checkpoint:

```text
checkpoints/imagenet100/drunet/<run>.pth
```

The evaluator accepts either:

```text
raw state_dict
```

or:

```text
{"model_state_dict": state_dict}
```

It also strips a leading `module.` prefix if the checkpoint was saved from
`DataParallel`.

Compatibility with `train_drunet.py`: compatible without conversion. The DRUNet
training code saves:

```python
torch.save(model.state_dict(), checkpoint_path)
```

and the external evaluator loads that exact format into `OfficialDRUNetAdapter`
with `strict=True`.

## Metrics

Implemented metrics:

| Metric | Implementation |
| --- | --- |
| PSNR | `20 * log10(1 / sqrt(mse + 1e-8))`, data range `[0, 1]` |
| SSIM | existing `src.TDnCNN_image.compute_ssim_item`, skimage SSIM, RGB channel axis |
| LPIPS | optional via `--lpips`; disabled by default and skipped if unavailable |

LPIPS is not required for readiness and is not computed unless explicitly
requested.

## Inference Pipeline

The external evaluator:

```text
1. Loads the original RGB image.
2. Generates Gaussian noise on the fly.
3. Keeps the clean image unchanged.
4. Pads the noisy image to a multiple of 8.
5. Runs DRUNet inference with sigma = sigma / 255.
6. Crops prediction back to the original H x W.
7. Computes metrics at original resolution.
```

Confirmed:

```text
no resize
no benchmark crop
pad_to_multiple_of_8 exists
crop_to_original exists
```

## Output Structure

Per benchmark output:

```text
experiments/external_benchmarks/<checkpoint_name>/<benchmark>/
├── metrics.csv
├── summary.json
├── report.md
└── qualitative/
    ├── *_gt.png
    ├── *_noisy.png
    ├── *_denoised.png
    └── *_error_map.png
```

Aggregate output:

```text
experiments/external_benchmarks/<checkpoint_name>/summary.json
```

Dry-run artifact check created:

```text
experiments/external_benchmarks/dry_run/Kodak24/metrics.csv
experiments/external_benchmarks/dry_run/Kodak24/summary.json
experiments/external_benchmarks/dry_run/Kodak24/report.md
experiments/external_benchmarks/dry_run/Kodak24/qualitative/kodim01_gt.png
experiments/external_benchmarks/dry_run/Kodak24/qualitative/kodim01_noisy.png
experiments/external_benchmarks/dry_run/Kodak24/qualitative/kodim01_denoised.png
experiments/external_benchmarks/dry_run/Kodak24/qualitative/kodim01_error_map.png
```

## Dry-Run Mode

Added:

```bash
python scripts/evaluate_external.py --dataset Kodak24 --dry-run --device cpu
```

Dry-run behavior:

```text
processes only the first image;
does not require a checkpoint;
uses an identity denoiser;
writes the same artifact schema as the real evaluator;
marks summary.json with "dry_run": true.
```

Observed dry-run result:

```text
Kodak24: images=1 sigma=25 psnr=20.2763 ssim=0.508376
```

This is an infrastructure check only and is not a benchmark result.

## Checks Performed

```powershell
.venv\Scripts\python.exe -m compileall -q scripts src
.venv\Scripts\python.exe scripts\evaluate_external.py --help
.venv\Scripts\python.exe scripts\prepare_external_benchmarks.py --strict
.venv\Scripts\python.exe scripts\evaluate_external.py --dataset Kodak24 --dry-run --device cpu
```

All checks passed.

## Readiness Conclusion

The external benchmark evaluation layer is ready for checkpoint-based DRUNet
evaluation on:

```text
Kodak24
CBSD68
Urban100
```

No checkpoint conversion is expected for checkpoints produced by
`scripts/train_drunet.py`.
