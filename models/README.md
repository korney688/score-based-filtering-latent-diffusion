# Models Directory

Canonical model artifacts for the approved MNIST latent-DDPM protocol.

Current layout:

- `autoencoders/` - selected encoder/autoencoder checkpoints.
- `ddpm/` - latent-DDPM checkpoints.
- `tdncnn/` - TDnCNN image-denoiser checkpoints.

Model files should be named deterministically, including model family, protocol role, and version where needed.

Examples:

- `autoencoder_baseline_v1.pth`
- `ddpm_baseline_latent_v1.pth`
- `ddpm_aligned_latent_v1.pth`
- `tdncnn_filtering_topk40_v1.pth`

Do not store exploratory plots or metric dumps here.
