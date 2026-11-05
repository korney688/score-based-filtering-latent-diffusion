# Experiment protocol

## Corpus and DDPM

The datasets are Colored MNIST (deterministic RGB color and abstract background), CIFAR-10 and ImageNet-100. Image-space DDPMs predict epsilon; latent encoders are outside this protocol. Images are represented in [0,1] for corruption and denoising, and mapped to [-1,1] for DDPM input.

For each image, draw a target quality and fix the corruption by image identity and corpus seed:

```text
target_PSNR ~ Uniform(10, 40) dB
sigma = 10^(-target_PSNR / 20)
x_corpus = clip(x_clean + sigma * noise, 0, 1), noise ~ N(0, I)
```

Clipping can change the measured PSNR. Corpus noise is separate from task, forward-diffusion and evaluation noise. Colored MNIST color/background seeds stay 42 for all corpus seeds.

All three DDPMs use T=1000 and linear betas from 0.0001 to 0.02. Forward diffusion and training are:

```text
alpha_bar_t = product_{s<=t}(1 - beta_s)
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
loss = mean((epsilon_theta(x_t, t) - epsilon)^2)
```

The full dataset-specific architecture, split and scan settings are in the experiment YAML files. Seed43/44 configs keep other scientific settings fixed. Their drivers check the permitted metadata and resume-policy differences.

## Score Validation and filtering

For a selected checkpoint and timestep, score an image by mean squared L2 epsilon energy over K forward-noise samples:

```text
score(x, t) = (1/K) * sum_k sum_pixels_channels(epsilon_theta(x_t^(k), t)^2)
```

Average energies, not epsilon predictions. CIFAR also records the pixel-mean form; its fixed image size preserves the ranking. Report timesteps start at 1. The configured scan is [1,2,5,10,20,50,100,200,500], with coarse K=4 and final K=16. Top candidates are refined using the same validation images and PSNR ladder [10,15,20,25,30,35,40] dB. Dataset-specific acceptance gates and confirmation logic must pass before filtering or downstream training.

Top-K keeps the lowest scores. Quantile-spread allocates exactly K samples across equal-frequency score-rank bins using largest-remainder quotas, then samples within bins with a fixed seed. Random samples without replacement from ordered original indices. Ties use deterministic image/bin order. Every subset has the exact requested size and unique original indices.

Each architecture has ten branches: Full, then Top-K, Quantile-spread and Random at 10%, 20% and 30%. Corpus seeds are 42/43/44; split, model, subset, task and evaluation seeds stay fixed. The Colored MNIST replicate driver separately fixes validation-ladder seed42. Use that driver for corpus seeds43/44.

## Downstream training

TDnCNN and official DRUNet learn to remove fresh task noise while keeping the fixed corrupted corpus image as the target:

```text
x_target = x_corpus
x_input = clip(x_corpus + task_noise, 0, 1)
```

Task sigma is sampled uniformly from 0.1 to 0.8. Variants share initial-state and validation/evaluation settings within their dataset and architecture. Validation data do not enter the training subsets.

| Dataset | DDPM epochs | TDnCNN step cap | DRUNet step cap | Validation interval | Warmup |
|---|---:|---:|---:|---:|---:|
| Colored MNIST | 20 | 20000 | 20000 | 100 | 100 |
| CIFAR-10 | 40 | 60000 | 50000 | 500 | 1000 |
| ImageNet-100 | 40 | 60000 | 50000 | 500 | 1000 |

Early stopping minimizes validation loss with patience=10 and min_delta=0.0001. Save every true validation minimum in best.pt, including improvements below min_delta. Warmup/min_delta control patience, not checkpoint selection. Evaluation loads best.pt with a compatible model state.

## Evaluation and aggregation

All variants use the same fixed test noise and clean targets. Colored/CIFAR internal evaluation uses uniform sigma in [0.1,0.8]; ImageNet uses fixed AWGN25/255. Colored MNIST LPIPS is missing because it was not computed; it is not zero.

The final external grid evaluates ImageNet-100 DRUNet only: ten variants, three corpus seeds, and Kodak24, CBSD68, Urban100 and SIDD (120 combinations). Synthetic benchmarks use RGB without resizing, shared AWGN25/255 and evaluation seed42. Reflect padding to a multiple of 8 is cropped back after prediction. SIDD uses native noisy/GT pairs; 25/255 is DRUNet conditioning rather than added corruption. See [SIDD evaluation](sidd_evaluation.md).

Metrics are PSNR, SSIM, LPIPS and MSE. Pair each filtered Delta PSNR with Full from the same dataset, architecture and corpus seed. Aggregate three-seed means and sample standard deviation (ddof=1), preserving missing values. Final validators require the full grids, matching input/checkpoint metadata and complete per-image or per-block observations. Plot bands show seed min-max, not standard deviation. Runtime inputs and artifact-layout limits are described in the [README](../README.md#reproducibility-limits).
