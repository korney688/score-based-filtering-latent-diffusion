# DRUNet Official Implementation Audit

This note audits official DRUNet sources before importing the model into the
ImageNet-100 downstream denoising pipeline. No training code is changed at this
stage.

## Scope

Current project contract for downstream denoisers:

```python
forward(x, sigma=None)
```

This contract must remain stable. Official DRUNet code does not expose this
exact API directly; it conditions on the noise level by receiving a concatenated
noise map as an extra input channel.

## Sources Reviewed

| Source | Repository | Relevant files | Activity | Integration complexity | Dependency footprint | Suitability |
| --- | --- | --- | --- | --- | --- | --- |
| DPIR DRUNet | `cszn/DPIR` | `models/network_unet.py`, `models/basicblock.py`, `main_dpir_denoising.py` | Model-specific repository; latest push observed via GitHub API: `2022-11-21` | Low if only the network files are copied | `torch`, `numpy`; no project-wide DPIR utilities required for training | Best primary source because DPIR is the original DRUNet denoiser context |
| KAIR DRUNet | `cszn/KAIR` | `models/network_unet.py`, `models/basicblock.py`, `models/select_network.py` | Active toolbox; latest push observed via GitHub API: `2024-10-02` | Medium if imported as toolbox; low if only network files are copied | Same core network dependencies, but full KAIR includes many unrelated restoration modules | Good maintained toolbox source, but larger than needed |
| Other Kai Zhang DRUNet sources | No separate newer canonical DRUNet-only implementation was needed for this project | N/A | N/A | N/A | N/A | DPIR and KAIR are the relevant official sources for this import decision |

Recommendation: import the minimal DPIR DRUNet network implementation, not the
full DPIR or KAIR repository. DPIR is the clearest primary source for the
DRUNet denoiser and explicitly demonstrates the noise-map conditioning used at
inference. KAIR can remain a fallback reference because it exposes the same
`UNetRes` family through a broader image-restoration toolbox.

## Model Entry Point

### DPIR

Source: `cszn/DPIR/models/network_unet.py`

Entry-point class:

```python
class UNetRes(nn.Module)
```

Constructor signature at source line 68:

```python
UNetRes(
    in_nc=1,
    out_nc=1,
    nc=[64, 128, 256, 512],
    nb=4,
    act_mode="R",
    downsample_mode="strideconv",
    upsample_mode="convtranspose",
)
```

The color DRUNet denoiser is instantiated in `cszn/DPIR/main_dpir_denoising.py`
with `in_nc=n_channels+1`, `out_nc=n_channels`, `nc=[64, 128, 256, 512]`,
`nb=4`, `act_mode="R"`, `downsample_mode="strideconv"`,
`upsample_mode="convtranspose"` at source lines 92-93.

### KAIR

Source: `cszn/KAIR/models/network_unet.py`

Entry-point class:

```python
class UNetRes(nn.Module)
```

Constructor signature at source line 22:

```python
UNetRes(
    in_nc=3,
    out_nc=3,
    nc=[64, 128, 256, 512],
    nb=4,
    act_mode="R",
    downsample_mode="strideconv",
    upsample_mode="convtranspose",
    bias=True,
)
```

KAIR resolves `net_type == "drunet"` to `models.network_unet.UNetRes` in
`cszn/KAIR/models/select_network.py` at source lines 174-184.

## Minimal Import Tree

The network dependency tree was checked from the source files:

```text
network_unet.py
└── basicblock.py
    ├── torch
    ├── torch.nn
    └── torch.nn.functional
```

Minimal files to copy manually:

| File | Required | Why |
| --- | --- | --- |
| `models/network_unet.py` | Yes | Defines `UNetRes`, the DRUNet model entry point. |
| `models/basicblock.py` | Yes | Provides `conv`, `ResBlock`, `sequential`, downsample and upsample blocks used by `UNetRes`. |
| `main_dpir_denoising.py` | No | Useful reference for noise-map construction, but not needed in the project runtime. |
| `models/select_network.py` | No | KAIR factory helper; unnecessary if the project adapter instantiates `UNetRes` directly. |
| DPIR/KAIR utility files | No | Not needed for ImageNet-100 downstream training/evaluation because the project already owns dataloading, metrics, checkpoints, and artifacts. |
| Official license/readme attribution | Recommended | Keep attribution for copied source files. This is documentation/compliance, not runtime code. |

Recommended local destination for copied source:

```text
src/external/drunet/network_unet.py
src/external/drunet/basicblock.py
src/external/drunet/__init__.py
```

The import inside `network_unet.py` must be adjusted from:

```python
import models.basicblock as B
```

to a local package import such as:

```python
from . import basicblock as B
```

## Actual Forward Scheme

The official model forward is:

```python
model(x0)
```

For `UNetRes`, the source `forward(self, x0)` starts at:

| Source | Line |
| --- | --- |
| DPIR `models/network_unet.py` | 105 |
| KAIR `models/network_unet.py` | 59 |

There is no separate `sigma` argument in `UNetRes.forward`.

The DPIR inference script constructs a noise-conditioned input first:

```text
image tensor + noise-level map -> concatenated input -> UNetRes
```

Observed in `cszn/DPIR/main_dpir_denoising.py`:

| Line | Behavior |
| --- | --- |
| 92-93 | Instantiates `UNetRes` with `in_nc=n_channels+1` and `out_nc=n_channels`. |
| 129 | Concatenates image tensor and a repeated scalar noise-level map along the channel dimension. |
| 137 | Calls `model(img_L)` on the concatenated tensor. |

## Noise-Level Conditioning

For RGB denoising, the effective tensor shapes are:

```text
image:
[B, 3, H, W]

noise map:
[B, 1, H, W]

official DRUNet input:
[B, 4, H, W]

official DRUNet output:
[B, 3, H, W]
```

DPIR forms the noise map as a constant spatial channel. In its test script the
noise level is normalized by `255.0` before concatenation. For this project,
the existing TDnCNN/DRUNet downstream code already uses normalized image tensors
and sigma values; the adapter should pass sigma in the same normalized scale as
the training noise protocol.

If `sigma` is scalar, the adapter should expand it to `[B, 1, H, W]`.
If `sigma` is per-image with shape `[B]` or `[B, 1]`, the adapter should reshape
and expand it to `[B, 1, H, W]`.
If `sigma=None`, the adapter can either use a configured default sigma or raise
a clear error for official DRUNet mode. For training, sigma should always be
provided by the existing noisy-pair protocol.

## Compatibility With Project API

The current project API can be preserved with an adapter:

```text
Project calls:
    y = model(x, sigma)

Adapter:
    1. validate x has shape [B, 3, H, W]
    2. normalize/reshape sigma to [B, 1, 1, 1]
    3. expand sigma map to [B, 1, H, W]
    4. concatenate [x, sigma_map] -> [B, 4, H, W]
    5. call official UNetRes(input_4ch)
    6. return denoised RGB tensor [B, 3, H, W]
```

Suggested adapter location:

```text
src/DRUNet_image.py
```

Suggested internal class shape:

```text
OfficialDRUNetAdapter
├── self.network = UNetRes(in_nc=4, out_nc=3, ...)
└── forward(x, sigma=None)
```

The public `build_drunet(...)` factory can keep returning an object with
`forward(x, sigma=None)`.

## Integration Impact

Expected changes for the later import step:

| Area | Change needed |
| --- | --- |
| `src/DRUNet_image.py` | Replace placeholder construction with `OfficialDRUNetAdapter`; keep `forward(x, sigma=None)`. |
| `src/external/drunet/` | Add copied official `network_unet.py` and `basicblock.py` with adjusted relative imports. |
| `configs/downstream/drunet.yaml` | Optionally add model parameters such as `nc`, `nb`, and `official: true`. |

Expected unchanged areas:

| Area | Reason |
| --- | --- |
| `scripts/train_drunet.py` | Already delegates to the DRUNet runner. |
| DRUNet run configs | Already define `full`, `quantile_10`, and `topk_10`. |
| Filtering artifacts | DRUNet consumes existing selected indices only. |
| Score computation | Unrelated to downstream denoiser import. |
| Encoder/DDPM pipeline | No dependency on DRUNet. |
| TDnCNN implementation | DRUNet is a parallel downstream denoiser. |
| Artifact generation | Existing DRUNet runner already mirrors TDnCNN-style outputs. |

Conclusion: the current DRUNet infrastructure should remain valid without
changes to dataloading, filtering, evaluation metrics, or artifact layout. The
only required runtime integration work is replacing the placeholder with an
adapter around official `UNetRes`.

## Final Recommendation

Import the DPIR implementation of DRUNet:

```text
cszn/DPIR/models/network_unet.py
cszn/DPIR/models/basicblock.py
```

Use `UNetRes` as the official model body with:

```text
in_nc=4
out_nc=3
nc=[64, 128, 256, 512]
nb=4
act_mode="R"
downsample_mode="strideconv"
upsample_mode="convtranspose"
```

Wrap it in a local adapter preserving:

```python
forward(x, sigma=None)
```

This is the smallest import that preserves the current project design, avoids
pulling in a full external toolbox, keeps the score-filtering pipeline intact,
and matches the original DRUNet noise-map conditioning protocol.

## Source Links

- DPIR repository: https://github.com/cszn/DPIR
- DPIR `network_unet.py`: https://github.com/cszn/DPIR/blob/master/models/network_unet.py
- DPIR denoising script: https://github.com/cszn/DPIR/blob/master/main_dpir_denoising.py
- DPIR `basicblock.py`: https://github.com/cszn/DPIR/blob/master/models/basicblock.py
- KAIR repository: https://github.com/cszn/KAIR
- KAIR `network_unet.py`: https://github.com/cszn/KAIR/blob/master/models/network_unet.py
- KAIR `select_network.py`: https://github.com/cszn/KAIR/blob/master/models/select_network.py
