# ImageNet-Scale Validation with DRUNet

Source:

```text
experiments/imagenet100/exp_007_drunet
```

No training, validation rerun, inference rerun, or filtering was performed for
this material collection.

## A. Verified Runs

| Run | Status | Run directory | Checkpoint | Epochs | Selected images | Ratio | Completion timestamp |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `full_sigma25` | complete | `experiments/imagenet100/exp_007_drunet/full_sigma25` | `checkpoints/imagenet100/drunet/full_sigma25.pth` | 15 | 120,581 | 100.000% | 2026-06-21 07:49:50 |
| `topk10_sigma25` | metrics complete; checkpoint not load-verified | `experiments/imagenet100/exp_007_drunet/topk10_sigma25` | `checkpoints/imagenet100/drunet/topk10_sigma25.pth` | 15 | 12,058 | 10.000% | 2026-06-20 23:53:01 |
| `quantile10_sigma25` | complete | `experiments/imagenet100/exp_007_drunet/quantile10_sigma25` | `checkpoints/imagenet100/drunet/quantile10_sigma25.pth` | 15 | 12,047 | 9.991% | 2026-06-20 22:30:36 |

Checkpoint load audit:

```text
full_sigma25: loaded as state_dict, 32,640,960 parameters
quantile10_sigma25: loaded as state_dict, 32,640,960 parameters
topk10_sigma25: torch.load failed with "PytorchStreamReader failed reading zip archive"
```

The Top-K run has complete `training_history.csv`, `metrics_history.csv`,
`metrics.json`, and `per_image_metrics.csv`, but the local checkpoint file is
not currently load-verified and may need to be retransferred before future
inference.

## B. Exact Metrics Table

The table uses the final row of `results/training_history.csv`, because it
contains epoch-level validation aggregates over the validation loader.

| Training subset | Selected images | Actual ratio | Validation loss | PSNR | SSIM | LPIPS | FID |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full dataset | 120,581 | 100.000% | 0.000973 | 30.5173 | 0.9165 | 0.0326 | 0.2548 |
| Top-K 10% | 12,058 | 10.000% | 0.001412 | 28.7949 | 0.8729 | 0.0449 | 0.2032 |
| Quantile-spread 10% | 12,047 | 9.991% | 0.001383 | 28.8536 | 0.8722 | 0.0466 | 0.2300 |

LaTeX table:

```latex
\begin{table}[t]
\centering
\caption{ImageNet-100 downstream denoising results with DRUNet at $\sigma=25/255$.}
\label{tab:drunet_imagenet100}
\begin{tabular}{lrrrrrr}
\hline
Training subset & Selected & Ratio & PSNR $\uparrow$ & SSIM $\uparrow$ & LPIPS $\downarrow$ & FID $\downarrow$ \\
\hline
Full dataset & 120{,}581 & 100.000\% & 30.5173 & 0.9165 & 0.0326 & 0.2548 \\
Top-K 10\% & 12{,}058 & 10.000\% & 28.7949 & 0.8729 & 0.0449 & 0.2032 \\
Quantile-spread 10\% & 12{,}047 & 9.991\% & 28.8536 & 0.8722 & 0.0466 & 0.2300 \\
\hline
\end{tabular}
\end{table}
```

## C. Differences Relative To Full

| Training subset | Data reduction | Δ validation loss | Δ PSNR | Δ SSIM | Δ LPIPS | Δ FID |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Top-K 10% | 90.000% | +0.000439 | -1.7224 | -0.0436 | +0.0122 | -0.0516 |
| Quantile-spread 10% | 90.009% | +0.000409 | -1.6637 | -0.0443 | +0.0139 | -0.0248 |

Filtered-run comparison:

```text
Best filtered PSNR: Quantile-spread 10%
Best filtered validation loss: Quantile-spread 10%
Best filtered SSIM: Top-K 10%
Best filtered LPIPS: Top-K 10%
Best filtered FID: Top-K 10%
```

Article-ready observations:

1. The full-dataset DRUNet achieved the best validation loss, PSNR, SSIM, and LPIPS.
2. Both filtered subsets used approximately 10% of the ImageNet-100 training partition, corresponding to about a 90% data reduction.
3. Quantile-spread 10% gave a slightly higher PSNR and lower validation loss than Top-K 10%.
4. Top-K 10% gave slightly better SSIM, LPIPS, and FID than Quantile-spread 10%.
5. Neither filtered subset dominated across all reported metrics.

## D. Qualitative Figure Status

Requested output:

```text
experiments/imagenet100/exp_007_drunet/qualitative/research_style/drunet_denoising_comparison.png
experiments/imagenet100/exp_007_drunet/qualitative/research_style/drunet_denoising_comparison.pdf
```

Status:

```text
not created
```

Reason: the available production qualitative grids do not use identical noisy
images across runs. They share the same clean images, but the noisy panels
differ between `full_sigma25`, `topk10_sigma25`, and `quantile10_sigma25`.
Creating a five-column figure with a shared `Noisy` column from these artifacts
would therefore be misleading.

Existing qualitative artifacts:

```text
experiments/imagenet100/exp_007_drunet/full_sigma25/results/qualitative_triplets.png
experiments/imagenet100/exp_007_drunet/topk10_sigma25/results/qualitative_triplets.png
experiments/imagenet100/exp_007_drunet/quantile10_sigma25/results/qualitative_triplets.png
```

Compatibility check on the first three rows:

```text
clean panels: identical across runs
noisy panels: not identical across runs
```

A compliant qualitative comparison requires existing artifacts generated from a
shared noisy validation set, or a new inference-only visualization pass using
the same noisy inputs for all three checkpoints.

## E. Article-Ready Paragraph

The full-dataset DRUNet achieved the best validation loss, PSNR, SSIM, and
LPIPS on ImageNet-100 at $\sigma=25/255$. Both filtered subsets reduced the
training set by approximately 90%, with a corresponding reduction in PSNR and
SSIM relative to the full-data model. Among the filtered runs,
Quantile-spread 10% produced the higher PSNR and lower validation loss, whereas
Top-K 10% produced slightly better SSIM, LPIPS, and FID. Thus, the filtered
subsets did not dominate each other consistently across all metrics.

## F. Qualitative Figure Caption

No compliant qualitative figure was generated from the available artifacts,
because the saved noisy inputs are not identical across runs. If a common-noise
visualization artifact is generated later, the caption can be:

```latex
\caption{Qualitative ImageNet-100 denoising comparison for DRUNet at
$\sigma=25/255$. Each row shows the same noisy input and clean target, followed
by predictions from models trained on the full dataset, the Top-K 10\% subset,
and the Quantile-spread 10\% subset.}
```
