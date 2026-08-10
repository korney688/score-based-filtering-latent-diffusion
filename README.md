# Score-Based Filtering for Image Denoising

This repository studies score-based filtering for downstream image denoising under controlled per-image corruption.

## Current Pipeline

```text
Clean dataset
-> controlled per-image corruption
-> image-space DDPM
-> DDPM score validation
-> selected checkpoint + timestep + MC K
-> full-corpus scoring
-> Filtering: Full / Top-K 10% / Quantile-spread 10% / Random 10%
-> downstream denoising
-> internal evaluation
-> external evaluation where applicable
-> summary
```

Current domains:

- MNIST
- CIFAR-10
- ImageNet-100

Encoder and latent-DDPM experiments are historical work and are preserved under `legacy/`.

## Protocol

Current filtering uses exactly one subset fraction:

```text
Full            = 100% training pool
Top-K           = 10%, minimum DDPM score, deterministic tie-break by original_index
Quantile-spread = 10%, exact shared selector
Random          = 10%, deterministic sample without replacement
```

Current score validation uses:

```text
PSNR ladder = [10, 15, 20, 25, 30, 35, 40] dB
score       = ||epsilon_theta(x_t, t)||^2
gate        = PASS required before filtering/downstream
```

Downstream denoisers use validation-based early stopping where supported by the current domain pipeline. Evaluation uses `best.pt`, selected by validation loss.

## Quick Start

```bash
python scripts/run_mnist_pipeline.py --help
python scripts/run_cifar10_pipeline.py --help
python scripts/run_imagenet100_pipeline.py --help
python scripts/run_external_evaluation.py --help
```

Canonical configs:

```text
configs/experiments/mnist.yaml
configs/experiments/cifar10.yaml
configs/experiments/imagenet100.yaml
```

New runs write to:

```text
experiments/<domain>/<run_id>/
```

## Smoke / Dry Run

MNIST synthetic smoke:

```bash
python scripts/run_mnist_pipeline.py --stage smoke_checks --smoke
```

CIFAR-10 dry-run:

```bash
python scripts/run_cifar10_pipeline.py --stage validation_all --smoke --dry-run
```

ImageNet-100 dry-run:

```bash
python scripts/run_imagenet100_pipeline.py \
  --config configs/experiments/imagenet100.yaml \
  --run-id imagenet100_smoke \
  --experiment-root experiments/imagenet100/imagenet100_smoke \
  --stage preflight \
  --smoke \
  --dry-run
```

## External Evaluation

External benchmarks for CIFAR-10 and ImageNet-100 use:

```text
Kodak24 / CBSD68 / Urban100
AWGN sigma = 25 / 255
seed = 42
```

Run:

```bash
python scripts/run_external_evaluation.py --help
```

## Repository Structure

```text
configs/experiments/      current domain configs
scripts/                  canonical launchers and helper CLIs
src/                      current implementation modules
tests/                    lightweight current tests
legacy/                   historical encoder/latent code and old launchers
```

## Legacy

Previous research investigated encoder-based latent DDPM variants, including noise-consistency autoencoders and induced latent noise. Those components are not part of the current image-space pipeline. They are retained under `legacy/` for provenance and reproducibility of earlier experiments.
