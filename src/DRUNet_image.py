from __future__ import annotations

import torch
from torch import nn

from src.external.drunet.network_unet import UNetRes


class DRUNetPlaceholder(nn.Module):
    """Interface-compatible placeholder for the official DRUNet implementation.

    The project does not vendor external DRUNet code automatically. This module
    provides the downstream API and tensor contract so run/config/evaluation
    infrastructure can be validated before manually importing the official model.
    """

    is_placeholder = True

    def __init__(self, in_channels: int = 3, features: int = 64, num_layers: int = 5):
        super().__init__()
        if num_layers < 3:
            raise ValueError(f"num_layers must be >= 3, got {num_layers}")

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers - 2):
            layers.extend(
                [
                    nn.Conv2d(features, features, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.Conv2d(features, in_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor | None = None) -> torch.Tensor:
        # Official DRUNet is noise-level conditioned. The placeholder accepts the
        # same optional argument but does not use it.
        residual = self.net(x)
        return x - residual


class OfficialDRUNetAdapter(nn.Module):
    """Adapter preserving the project DRUNet API for the official DPIR UNetRes.

    Official DRUNet receives the noise level as an extra spatial map channel,
    not as a separate forward argument. This adapter keeps the public project
    contract ``forward(x, sigma=None)`` and prepares the 4-channel input.
    """

    is_placeholder = False

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int | None = None,
        nc: list[int] | tuple[int, ...] | None = None,
        nb: int = 4,
        act_mode: str = "R",
        downsample_mode: str = "strideconv",
        upsample_mode: str = "convtranspose",
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError(f"Official color DRUNet expects in_channels=3, got {in_channels}")
        out_channels = in_channels if out_channels is None else out_channels
        nc = [64, 128, 256, 512] if nc is None else list(nc)
        self.in_channels = in_channels
        self.net = UNetRes(
            in_nc=in_channels + 1,
            out_nc=out_channels,
            nc=nc,
            nb=nb,
            act_mode=act_mode,
            downsample_mode=downsample_mode,
            upsample_mode=upsample_mode,
        )

    @staticmethod
    def _sigma_to_map(sigma: float | torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(sigma):
            sigma_tensor = torch.tensor(float(sigma), dtype=x.dtype, device=x.device)
        else:
            sigma_tensor = sigma.to(device=x.device, dtype=x.dtype)

        batch_size, _, height, width = x.shape
        if sigma_tensor.ndim == 0:
            sigma_tensor = sigma_tensor.view(1, 1, 1, 1).expand(batch_size, 1, 1, 1)
        elif sigma_tensor.ndim == 1:
            if sigma_tensor.shape[0] not in {1, batch_size}:
                raise ValueError(f"sigma shape {tuple(sigma_tensor.shape)} is not compatible with batch size {batch_size}")
            sigma_tensor = sigma_tensor.view(-1, 1, 1, 1)
        elif sigma_tensor.ndim == 2:
            if sigma_tensor.shape[-1] != 1 or sigma_tensor.shape[0] not in {1, batch_size}:
                raise ValueError(f"sigma shape {tuple(sigma_tensor.shape)} must be [B, 1] or [1, 1]")
            sigma_tensor = sigma_tensor.view(-1, 1, 1, 1)
        elif sigma_tensor.ndim == 4:
            if sigma_tensor.shape[1] != 1:
                raise ValueError(f"sigma map must have one channel, got shape {tuple(sigma_tensor.shape)}")
            if sigma_tensor.shape[0] not in {1, batch_size}:
                raise ValueError(f"sigma map batch size {sigma_tensor.shape[0]} is not compatible with {batch_size}")
            if sigma_tensor.shape[2:] not in {(1, 1), (height, width)}:
                raise ValueError(
                    f"sigma map spatial shape {tuple(sigma_tensor.shape[2:])} must be (1, 1) or {(height, width)}"
                )
        else:
            raise ValueError(f"Unsupported sigma shape: {tuple(sigma_tensor.shape)}")

        return sigma_tensor.expand(batch_size, 1, height, width)

    def forward(self, x: torch.Tensor, sigma: float | torch.Tensor | None = None) -> torch.Tensor:
        if sigma is None:
            raise ValueError("Official DRUNet requires a noise level: call forward(x, sigma=...).")
        if x.ndim != 4:
            raise ValueError(f"x must have shape [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"x must have {self.in_channels} channels, got {x.shape[1]}")

        sigma_map = self._sigma_to_map(sigma, x)
        conditioned_input = torch.cat([x, sigma_map], dim=1)
        return self.net(conditioned_input)


def build_drunet(
    in_channels: int = 3,
    features: int = 64,
    num_layers: int = 5,
    official: bool = False,
    nc: list[int] | tuple[int, ...] | None = None,
    nb: int = 4,
    act_mode: str = "R",
    downsample_mode: str = "strideconv",
    upsample_mode: str = "convtranspose",
) -> nn.Module:
    """Build a DRUNet-compatible denoiser.

    ``official=False`` keeps the lightweight placeholder for plumbing checks.
    ``official=True`` builds the official DPIR UNetRes behind a project API
    adapter that accepts ``forward(x, sigma=None)``.
    """

    if official:
        return OfficialDRUNetAdapter(
            in_channels=in_channels,
            nc=nc,
            nb=nb,
            act_mode=act_mode,
            downsample_mode=downsample_mode,
            upsample_mode=upsample_mode,
        )
    return DRUNetPlaceholder(in_channels=in_channels, features=features, num_layers=num_layers)
