# Experiment protocol

## Datasets and noise

The experiments use Colored MNIST, CIFAR-10 and ImageNet-100. Colored MNIST is created from MNIST with fixed colors and backgrounds.

Each image is corrupted with Gaussian noise. The noise level is chosen using a target PSNR between 10 and 40 dB:

```text
target_PSNR ~ Uniform(10, 40) dB
sigma = 10^(-target_PSNR / 20)
x_corpus = clip(x_clean + sigma * noise, 0, 1)
noise ~ N(0, I)
```

Noise is generated using the image index and corpus seed, so the same experiment can be repeated. Clipping may change the measured PSNR.

The corpus seeds are 42, 43 and 44. Other random seeds are kept fixed. For Colored MNIST, the colors and backgrounds are also fixed across the three runs.

## DDPM training and scoring

An image-space DDPM is trained to predict Gaussian noise. Images are scaled to [-1, 1] before entering the DDPM.

All three datasets use 1000 diffusion steps with a linear beta schedule from 0.0001 to 0.02.

Forward diffusion is defined as:

```text
alpha_bar_t = product_{s<=t}(1 - beta_s)

x_t = sqrt(alpha_bar_t) * x_0
    + sqrt(1 - alpha_bar_t) * epsilon
```

The model is trained using the epsilon prediction loss:

```text
loss = mean((epsilon_theta(x_t, t) - epsilon)^2)
```

After training, I evaluate several DDPM checkpoints and timesteps to choose the settings for scoring.

For each image, the score is calculated as the average squared norm of the predicted noise over K samples:

```text
score(x, t) = (1/K) * sum_k ||epsilon_theta(x_t^(k), t)||^2
```

The score is averaged over 16 noise samples for the final evaluation. The checkpoint and timestep are selected using validation images with different PSNR levels.

## Training subset selection

Four strategies are compared:

- **Full:** all training images.
- **Top-K:** images with the lowest DDPM scores.
- **Quantile-spread:** images selected across different parts of the score distribution.
- **Random:** images selected randomly without replacement.

For Quantile-spread, images are divided into equal-frequency score bins. Samples are selected from these bins using a fixed seed, with the exact required subset size.

The selected fractions are 10%, 20% and 30%. Together with Full, this gives ten training variants for each denoising model.

## Denoiser training

I train TDnCNN and DRUNet on the selected subsets.

During training, additional Gaussian noise is applied to the corrupted corpus images:

```text
x_target = x_corpus
x_input = clip(x_corpus + task_noise, 0, 1)
```

The task noise level is sampled uniformly from 0.1 to 0.8.

All subsets use the same model initialization and validation settings for a given dataset and architecture. Validation images are kept separate from the training subsets.

The main training settings are:

| Dataset | DDPM epochs | TDnCNN max steps | DRUNet max steps | Validation interval |
|---|---:|---:|---:|---:|
| Colored MNIST | 20 | 20,000 | 20,000 | 100 |
| CIFAR-10 | 40 | 60,000 | 50,000 | 500 |
| ImageNet-100 | 40 | 60,000 | 50,000 | 500 |

Early stopping uses validation loss with patience=10 and min_delta=0.0001. Training stops after the allowed number of validation checks without sufficient improvement. The checkpoint with the lowest validation loss is saved as `best.pt`.

## Evaluation

The models are evaluated using PSNR, SSIM, MSE and LPIPS where available.

Colored MNIST and CIFAR-10 use evaluation noise with sigma sampled uniformly from 0.1 to 0.8. ImageNet-100 uses fixed Gaussian noise with sigma=25/255.

External evaluation uses Kodak24, CBSD68, Urban100 and SIDD. The final external comparison evaluates DRUNet trained on ImageNet-100. SIDD uses real noisy/clean image pairs, as described in the [SIDD evaluation protocol](sidd_evaluation.md).

For each corpus seed, I compare the PSNR of a filtered model with the Full model trained using the same seed:

```text
Delta PSNR = PSNR_filtered - PSNR_full
```

The final tables report the mean and sample standard deviation over seeds 42, 43 and 44. Figures show the mean Delta PSNR, with shaded regions indicating the minimum and maximum values across seeds.

LPIPS was not calculated for Colored MNIST and is left missing in the final tables.