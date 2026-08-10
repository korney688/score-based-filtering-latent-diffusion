# Documentation

Supporting documentation for the dataset-scoped latent-DDPM filtering project.

## Files

- `PIPELINE.md` - detailed active pipeline, entrypoints, dataset scope, and run commands.
- `ENCODERS.md` - encoder architecture names, compatibility notes, and Small/Large commands.

The root `README.md` is the primary quick-start entrypoint for setup, active commands, report locations, and the project layout.

## Current Scope

- MNIST: complete active protocol.
- CIFAR-10: active for `NoiseConsistencyAELarge` training, encoder validation, latent-DDPM training/validation, score-based filtering, and the first TDnCNN downstream validation path for induced `topk_10` filtering.
