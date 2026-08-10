# Methodology

Current experiments compare denoising performance after selecting training samples with DDPM-derived scores.

The current domains are MNIST, CIFAR-10, and ImageNet-100. Each domain uses controlled per-image Gaussian corruption, image-space DDPM score validation, and the same filtering baselines:

- Full
- Top-K 10%
- Quantile-spread 10%
- Random 10%

Encoder-based latent DDPM variants are legacy experiments and are not part of the current protocol.
