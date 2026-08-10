# Legacy Research Branches

This directory contains historical code from earlier versions of the project.

The current research pipeline uses image-space DDPMs with deterministic variable per-image AWGN corruption, validated DDPM scores, filtering, and TDnCNN/DRUNet downstream denoisers.

The encoder-based latent DDPM pipeline is no longer part of the current scientific protocol. Its code is preserved here only for reproducibility of earlier experiments and for historical reference.

Current runtime code should not import modules from `legacy/`.

The historical CIFAR e18/t100/MC16 result is classified as `LEGACY_LATENT_SCORE_RESULT`. It is preserved for provenance and comparison, not as a current image-space DDPM result.
