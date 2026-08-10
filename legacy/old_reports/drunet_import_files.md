# DRUNet Manual Import Files

No external DRUNet repository is downloaded or vendored automatically by this
project. To enable real DRUNet training, manually copy the official model files
from the upstream DRUNet repository into a local module and update the adapter.

## Files to import manually

From the official DRUNet / DPIR codebase, copy the model definition files that
define the DRUNet network, typically:

- `models/network_unet.py`
- any adjacent utility modules imported by that file, for example basic block
  definitions or sequential/container helpers if the official version requires
  them.

The exact file list depends on the official repository revision. Use the
official DRUNet source as the authority and keep the original license header.

## Destination in this project

Recommended local destination:

```text
src/external/drunet/
```

Example structure:

```text
src/external/drunet/
├── __init__.py
├── network_unet.py
└── basicblock.py        # only if required by the official source
```

## Adapter update

After copying the official files, update:

```text
src/DRUNet_image.py
```

Required changes:

1. Import the official DRUNet class from `src.external.drunet`.
2. Replace the `official=True` branch in `build_drunet(...)` with construction
   of the official model.
3. Preserve the project-facing `forward(x, sigma=None)` contract.
4. If the official model expects a noise-level map concatenated to the image,
   adapt the wrapper so callers can still pass either only `x` or `(x, sigma)`.
5. Leave `DRUNetPlaceholder` available for smoke model-construction checks.

## Imports likely to fix

Official DRUNet files often use repository-relative imports such as:

```python
from models.basicblock import ...
```

Change those to project-local imports, for example:

```python
from src.external.drunet.basicblock import ...
```

Do not modify encoder, latent-DDPM, filtering, or TDnCNN code while importing
DRUNet.
