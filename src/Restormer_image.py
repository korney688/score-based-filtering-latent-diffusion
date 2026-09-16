from __future__ import annotations

import torch
from torch import nn

from src.external.restormer.restormer_arch import Restormer


class RestormerAdapter(nn.Module):
    """Adapter for using the official Restormer as a project denoiser."""

    is_placeholder = False

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int | None = None,
        dim: int = 48,
        num_blocks: list[int] | tuple[int, ...] = (4, 6, 6, 8),
        num_refinement_blocks: int = 4,
        heads: list[int] | tuple[int, ...] = (1, 2, 4, 8),
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        layer_norm_type: str = "WithBias",
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        if out_channels != in_channels:
            raise ValueError(
                "For the standard Restormer denoising setup, "
                "out_channels must equal in_channels."
            )

        if len(num_blocks) != 4:
            raise ValueError("num_blocks must contain exactly 4 values.")

        if len(heads) != 4:
            raise ValueError("heads must contain exactly 4 values.")

        self.in_channels = in_channels

        self.net = Restormer(
            inp_channels=in_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=list(num_blocks),
            num_refinement_blocks=num_refinement_blocks,
            heads=list(heads),
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=layer_norm_type,
            dual_pixel_task=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        sigma: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"x must have shape [B, C, H, W], got {tuple(x.shape)}"
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"x must have {self.in_channels} channels, "
                f"got {x.shape[1]}"
            )

        height, width = x.shape[-2:]

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                "Restormer requires image height and width "
                f"to be divisible by 8, got {height}x{width}."
            )

        # sigma is accepted only to keep a denoiser-compatible API.
        # Standard Restormer does not use an explicit noise-level input.
        return self.net(x)


def build_restormer(
    in_channels: int = 3,
    out_channels: int | None = None,
    dim: int = 48,
    num_blocks: list[int] | tuple[int, ...] = (4, 6, 6, 8),
    num_refinement_blocks: int = 4,
    heads: list[int] | tuple[int, ...] = (1, 2, 4, 8),
    ffn_expansion_factor: float = 2.66,
    bias: bool = False,
    layer_norm_type: str = "WithBias",
) -> nn.Module:
    return RestormerAdapter(
        in_channels=in_channels,
        out_channels=out_channels,
        dim=dim,
        num_blocks=num_blocks,
        num_refinement_blocks=num_refinement_blocks,
        heads=heads,
        ffn_expansion_factor=ffn_expansion_factor,
        bias=bias,
        layer_norm_type=layer_norm_type,
    )