# DRUNet Memory Preflight

Date: 2026-06-20

Target production environment:

```text
GPU: Tesla V100-SXM2-32GB
CPU: 12
RAM: 64 GB
model: official DPIR DRUNet UNetRes
image size: 64x64
training precision: FP32
noise protocol: fixed sigma = 25 / 255
runs: full_sigma25, quantile10_sigma25, topk10_sigma25
```

No training, epoch, DGX job, filtering, encoder, DDPM, TDnCNN, or production
run was started for this preflight.

## Method

The local CUDA device is not the target DGX GPU:

```text
local CUDA: NVIDIA GeForce GTX 1050
target GPU: Tesla V100-SXM2-32GB
```

Therefore, this is an analytical V100 memory estimate rather than a measured
DGX sweep.

Observed model facts:

```text
parameters: 32,640,960
parameter memory FP32: 124.5 MiB
```

Static training state estimate:

```text
weights + gradients + Adam exp_avg + Adam exp_avg_sq
= 4 x parameter memory
= about 498 MiB
```

Activation accounting was estimated from one local CPU forward shape pass with
`OfficialDRUNetAdapter`, input `[1, 3, 64, 64]`, and `sigma=25/255`.

Observed leaf-module forward outputs:

```text
leaf output activation footprint per sample: about 47.2 MiB FP32
conv output footprint per sample: about 30.9 MiB FP32
```

For training-mode memory, the estimate uses a conservative multiplier:

```text
peak estimate =
  1.2 x (
    parameters + gradients + Adam states
    + 3 x leaf activation footprint x batch_size
    + 2 GiB cuDNN/workspace allowance
  )
```

This is intentionally conservative because autograd stores intermediate tensors
and skip connections, and the CUDA allocator/cuDNN workspace can add overhead.

## Dataset Sizes

Full training size was read from production filtering `scores.csv`.
Filtered sizes were read from the existing `selected_indices.npy` files.

| Run | Images per epoch | Source |
| --- | ---: | --- |
| `full_sigma25` | 120,581 | `experiments/imagenet100/exp_005_filtering/topk/10/scores.csv` |
| `quantile10_sigma25` | 12,047 | `experiments/imagenet100/exp_005_filtering/quantile/10/selected_indices.npy` |
| `topk10_sigma25` | 12,058 | `experiments/imagenet100/exp_005_filtering/topk/10/selected_indices.npy` |

## Memory Estimate

| Batch size | Forward fits | Forward+backward fits | Estimated training peak | V100 32GB headroom | Recommendation |
| ---: | --- | --- | ---: | ---: | --- |
| 4 | yes | yes | 3.6 GiB | 28.4 GiB | Too small; wastes available memory. |
| 8 | yes | yes | 4.3 GiB | 27.7 GiB | Safe, but still very conservative. |
| 16 | yes | yes | 5.6 GiB | 26.4 GiB | Safe. |
| 32 | yes | yes | 8.3 GiB | 23.7 GiB | Safe and reasonable. |
| 64 | yes | yes | 13.6 GiB | 18.4 GiB | Recommended production default. |
| 128 | yes, analytically | likely yes, analytically | 24.2 GiB | 7.8 GiB | Possible, but less robust without measured DGX sweep. |

Forward-only inference memory is far below the forward+backward estimate for
all listed batch sizes. The limiting case is training mode with activations,
gradients, and Adam state.

## Steps Per Epoch

| Batch size | `full_sigma25` | `quantile10_sigma25` | `topk10_sigma25` |
| ---: | ---: | ---: | ---: |
| 4 | 30,146 | 3,012 | 3,015 |
| 8 | 15,073 | 1,506 | 1,508 |
| 16 | 7,537 | 753 | 754 |
| 32 | 3,769 | 377 | 377 |
| 64 | 1,885 | 189 | 189 |
| 128 | 943 | 95 | 95 |

## Expected Training-Time Impact

Relative to batch size 4, step count decreases approximately linearly:

| Batch size | Approximate step-count reduction vs batch 4 |
| ---: | ---: |
| 8 | 2.0x fewer steps |
| 16 | 4.0x fewer steps |
| 32 | 8.0x fewer steps |
| 64 | 16.0x fewer steps |
| 128 | 32.0x fewer steps |

Actual wall-clock speedup will be smaller than the step-count reduction because
larger batches can reduce per-step efficiency and data loading may contribute
overhead. Still, batch size 4 would make the full run unnecessarily long:

```text
batch 4:   30,146 steps/epoch for full_sigma25
batch 64:   1,885 steps/epoch for full_sigma25
batch 128:    943 steps/epoch for full_sigma25
```

## Does Batch Size 4 Make Sense?

No. For DRUNet at 64x64 on a V100 32GB, batch size 4 is a smoke-safe value, not
a good production value. The estimate leaves more than 28 GiB unused at batch 4.

Batch size 128 is analytically plausible, but it leaves a narrower margin for:

```text
cuDNN algorithm workspace
CUDA allocator fragmentation
LPIPS/FID validation models
driver/runtime overhead
unmodeled tensor saves
```

Batch size 64 is the best production default from this preflight: it reduces
the number of optimization steps by about 16x relative to batch 4 while keeping
an estimated 18 GiB memory margin on V100 32GB.

## Recommendation

Recommended production batch size:

```text
batch_size = 64
```

Do not keep:

```text
batch_size = 4
```

unless the goal is only a very conservative smoke run.

Suggested launch approach:

```text
1. Update only DRUNet production batch size from 4 to 64.
2. Keep epochs, optimizer, scheduler, sigma, learning rate, seed, and run set unchanged.
3. On DGX, optionally run a single manual memory sanity check before overnight training.
```

This preflight does not change the production configs. It only recommends the
batch-size update.
