# Encoder Architectures

## NoiseConsistencyAESmall

`NoiseConsistencyAESmall` is the original noise-consistency autoencoder architecture.

It is kept for backward compatibility with the MNIST pipeline:

- old MNIST checkpoints keep the same state-dict key structure;
- `NoiseConsistencyAutoencoder` remains a backward-compatible name for this small architecture;
- existing DDPM, filtering, and score-validation code can continue to import `NoiseConsistencyAutoencoder`.

Shape pattern:

```text
encoder: image -> Conv2d(in, 8) -> Conv2d(8, 16) -> latent
decoder: latent -> 16 x H/4 x W/4 -> ConvTranspose2d(16, 8) -> ConvTranspose2d(8, out)
```

MNIST default:

```yaml
encoder:
  name: noise_consistency_small
  latent_dim: 16
```

## NoiseConsistencyAELarge

`NoiseConsistencyAELarge` is the expanded CIFAR-10 adaptation architecture.

It keeps the same research setup:

- same latent representation type;
- same reconstruction loss;
- same noise-consistency loss;
- same training loop and optimizer setup.

Only model capacity is increased.

Shape pattern:

```text
encoder: image -> Conv2d(in, 32) -> Conv2d(32, 64) -> latent
decoder: latent -> 64 x H/4 x W/4 -> ConvTranspose2d(64, 32) -> ConvTranspose2d(32, out)
```

CIFAR-10 default:

```yaml
encoder:
  name: noise_consistency_large
  latent_dim: 128
  checkpoint_run: noise_consistency_large_latent128
```

## Training Commands

MNIST Small:

```powershell
.\.venv\Scripts\python.exe scripts/train_encoder.py noise-consistency --dataset mnist --variant small
```

CIFAR-10 Large:

```powershell
.\.venv\Scripts\python.exe scripts/train_encoder.py noise-consistency --dataset cifar10 --variant large --latent-dim 128
```

Latent dimension can be overridden explicitly:

```powershell
.\.venv\Scripts\python.exe scripts/train_encoder.py noise-consistency --dataset cifar10 --variant large --latent-dim 128
```

Equivalent override style:

```powershell
.\.venv\Scripts\python.exe scripts/train_encoder.py noise-consistency dataset=cifar10 variant=large latent_dim=128
```

## Artifact Directories

New noise-consistency runs are architecture- and latent-dimension-scoped:

```text
checkpoints/<dataset>/autoencoders/noise_consistency_small_latent16/
checkpoints/<dataset>/autoencoders/noise_consistency_large_latent128/
outputs/<dataset>/autoencoders/noise_consistency_small_latent16/
outputs/<dataset>/autoencoders/noise_consistency_large_latent128/
```

Historical paths such as `ae_noise_consistency_mnist/` and `ae_noise_consistency_cifar10/` are left untouched.

## Validation And DDPM Use

The selected CIFAR-10 encoder is used by both Encoder Validation and Latent-DDPM:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_encoder.py compare-encoders --dataset cifar10
.\.venv\Scripts\python.exe scripts/main.py dataset=cifar10 task=train_latent_DDPM train_latent_DDPM.latent_noise_mode=baseline
.\.venv\Scripts\python.exe scripts/evaluate_latent_ddpm_score.py --dataset cifar10
```

For repeated DDPM runs, add `train_latent_DDPM.id_label=...` to avoid reusing an existing output directory.
