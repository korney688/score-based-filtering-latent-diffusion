# Worklog: ImageNet-100 noisy pipeline, 2026-07-03

This file is a handoff note for reopening the project later. It summarizes the noisy ImageNet-100 work done today and the latest production-script correction.

## Goal

Prepare an experimental `dataset_mode=noisy` mode for the existing ImageNet-100 pipeline.

Constraints from the task:

- Do not create a separate new pipeline.
- Do not change model architectures.
- Do not change the research setup.
- Keep the existing clean pipeline as the baseline.
- Change only the input data source/protocol: clean ImageNet-100 images become on-the-fly physically corrupted images in Dataset/DataLoader.

## Dataset mode

`configs/dataset/imagenet100.yaml` now has:

```yaml
dataset_mode: clean

corruption:
  enabled: true
  type: poisson_gaussian
  poisson_scale: 30.0
  gaussian_sigma: 0.01
  seed: 42
  apply_to: all
```

The clean mode remains the default. Noisy mode is enabled with:

```bash
dataset.dataset_mode=noisy
```

or for scripts that expose CLI flags:

```bash
--dataset-mode noisy
```

## Corruption protocol

The physical corruption protocol is Poisson-Gaussian:

```python
lambda_ = 30.0
sigma_r = 0.01

k = torch.poisson(lambda_ * x)
x_poisson = k / lambda_
x_noisy = x_poisson + torch.randn_like(x) * sigma_r
x_noisy = torch.clamp(x_noisy, 0.0, 1.0)
```

Implementation is in `src/dataset_registry.py`:

- `PoissonGaussianCorruption`
- `CorruptedDataset`
- `dataset_mode(...)`
- `corruption_enabled(...)`
- `apply_dataset_runtime_overrides(...)`
- support for `return_clean_for_corruption`

Important behavior: in noisy mode, downstream denoiser datasets can return `(x_noisy, x_clean)` so the physical corruption is applied once and the clean image remains the supervised target.

## Files touched for noisy integration

Main files involved in the noisy-mode integration:

- `configs/dataset/imagenet100.yaml`
- `src/dataset_registry.py`
- `src/evaluation/encoder_validation.py`
- `src/evaluation/latent_ddpm_score_validation.py`
- `src/tdncnn_datasets.py`
- `scripts/internal/train_DRUNet_image.py`
- `scripts/internal/train_TDnCNN_image.py`
- `scripts/internal/run_TDnCNN_image_suite.py`
- `scripts/internal/run_DRUNet_image_suite.py`
- `scripts/internal/tdncnn_image_runs_config.py`
- `scripts/internal/drunet_image_runs_config.py`
- `scripts/internal/train_autoencoder_noise_consistency_mnist.py`
- `scripts/run_imagenet100_full_noisy.sh`

## Double-noise checks

The intended final protocol:

- Dataset adds physical Poisson-Gaussian noise exactly once.
- Encoder Validation in `dataset_mode=noisy` uses noisy input from Dataset and must not add its old image Gaussian noise on top.
- TDnCNN in `dataset_mode=noisy` must not add the old training Gaussian as another physical input corruption.
- DRUNet receives noisy images from Dataset; if the base sample is already `(x_noisy, x_clean)`, its wrapper adds only sigma-conditioning, not another physical noisy image.
- DDPM diffusion noise is unchanged because it is model noise, not physical image corruption.

## Smoke checks already run

Local smoke checks completed during the work:

- Python compile for modified modules.
- Dataset smoke on `data/smoke_imagenet`: train/val load correctly, batch shape `(2, 3, 64, 64)`.
- Exact Poisson-Gaussian formula check returned `max diff = 0.0`.
- Encoder validation with noisy input passed.
- Direct DDPM train step on noisy batch passed.
- Score validation with noisy override and smoke checkpoints passed.
- Filtering through `scripts/main.py` with noisy override passed.
- TDnCNN forward on noisy dataset passed.
- DRUNet dataloader/forward on noisy dataset passed.

## New production script

Created:

```bash
scripts/run_imagenet100_full_noisy.sh
```

Do not overwrite or repurpose:

```bash
scripts/run_imagenet100_full.sh
```

The clean script is intended to remain the clean experiment reference.

The noisy script keeps the clean sequence and changes the dataset mode only:

```bash
NOISY_HYDRA_OVERRIDE="dataset.dataset_mode=noisy"
NOISY_CLI_ARGS=(--dataset-mode noisy)
RUN_ID="${RUN_ID:-imagenet100_noisy_pg_$(date +%Y%m%d_%H%M%S)}"
```

The noisy production sequence is:

1. GPU sanity check
2. Dataset and noisy corruption sanity check
3. Encoder Validation
4. Encoder Selection artifact check
5. Latent DDPM baseline
6. Latent DDPM encoder-induced
7. Score Validation
8. Filtering using the encoder-induced branch
9. TDnCNN full training and ImageNet validation
10. DRUNet full training and ImageNet validation

The script validates before running:

- dataset root exists;
- `dataset.dataset_mode=noisy`;
- `dataset.corruption.enabled=true`;
- `dataset.corruption.type=poisson_gaussian`;
- `poisson_scale=30.0`;
- `gaussian_sigma=0.01`;
- expected checkpoints exist;
- run-specific outputs do not collide unless `FORCE=1` is set.

Syntax check passed:

```bash
bash -n scripts/run_imagenet100_full_noisy.sh
```

## Latest correction: external benchmarks removed

Important correction from the latest user request:

External benchmark evaluation is NOT part of the production ImageNet pipeline. It must remain a separate local workflow after DRUNet training.

Removed from `scripts/run_imagenet100_full_noisy.sh`:

- `prepare_external_benchmarks.py`
- `evaluate_external.py`
- external benchmark sanity checks
- artifact checks for `Kodak24`, `CBSD68`, `Urban100`

The noisy production script now ends after internal ImageNet downstream training/validation:

```text
TDnCNN full training and ImageNet validation
DRUNet full training and ImageNet validation
```

After removal, this check returned no matches:

```powershell
Select-String -Path scripts\run_imagenet100_full_noisy.sh -Pattern "external|evaluate_external|prepare_external|Kodak|CBSD|Urban" -CaseSensitive:$false
```

## Full server launch command

Default server launch:

```bash
PROJECT_ROOT=/workspace \
IMAGENET100_ROOT=/workspace/data/imagenet100 \
bash scripts/run_imagenet100_full_noisy.sh
```

With explicit run id:

```bash
RUN_ID=imagenet100_noisy_pg_YYYYMMDD_HHMMSS \
PROJECT_ROOT=/workspace \
IMAGENET100_ROOT=/workspace/data/imagenet100 \
bash scripts/run_imagenet100_full_noisy.sh
```

## Docker commands

Prepared commands:

```bash
docker build -t imagenet100-noisy-pg:latest .
docker push <registry>/imagenet100-noisy-pg:latest
```

`docker push` is needed only if the remote server pulls the image from a registry.

## Expected runtime

Use the clean ImageNet-100 production runtime as the baseline. The noisy mode should add only DataLoader overhead for on-the-fly Poisson-Gaussian corruption.

## Notes for the next session

- Do not add external benchmark evaluation back into `scripts/run_imagenet100_full_noisy.sh`.
- Do not add a second physical noise source for `dataset_mode=noisy`.
- Keep DDPM diffusion noise unchanged.
- Keep architecture, optimizer, scheduler, number of epochs, checkpoint naming, and research setup unchanged.
- If editing production launch behavior, prefer editing only `scripts/run_imagenet100_full_noisy.sh`.
- Be careful with the dirty worktree: some files were already modified during this task, and `scripts/run_imagenet100_full.sh` should not be changed unless explicitly requested.
