import math

import torch
import torch.nn as nn
import torch.nn.functional as F


KernelSize = int | tuple[int, int]


def zero_module(module: nn.Module) -> nn.Module:
    # Start this layer from zero output for a smoother residual block
    for p in module.parameters():
        p.detach().zero_()
    return module


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    # Convert integer diffusion steps into sinusoidal vectors
    half = dim // 2
    device = timesteps.device

    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=device) / half
    )
    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class _ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        kernel_size: KernelSize,
        groups: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            padding = (kernel_size - 1) // 2
        else:
            padding = tuple((k - 1) // 2 for k in kernel_size)

        gn_groups = min(groups, in_channels // 4) if in_channels > 4 else 1

        self.norm1 = nn.GroupNorm(gn_groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels),
        )

        gn_groups_out = min(groups, out_channels // 4) if out_channels > 4 else 1
        self.norm2 = nn.GroupNorm(gn_groups_out, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = zero_module(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        )

        if self.in_channels != self.out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        # Add the timestep information to every spatial position.
        t_vec = self.time_emb_proj(time_emb)
        h = h + t_vec[:, :, None, None]

        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        return self.skip_proj(x) + h


class _EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        kernel_size: KernelSize,
        stride: KernelSize,
    ) -> None:
        super().__init__()
        self.block = _ResBlock(in_channels, out_channels, time_dim, kernel_size)
        self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Return both the downsampled tensor and the skip connection
        skip = self.block(x, time_emb)
        x = self.max_pool(skip)
        return x, skip


class _DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        kernel_size: KernelSize,
        stride: KernelSize,
    ) -> None:
        super().__init__()
        self.transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=stride, stride=stride)
        self.block = _ResBlock(in_channels, out_channels, time_dim, kernel_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        x = self.transpose(x)

        # Align tensor sizes before concatenating with the skip connection.
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(
            x,
            [
                diff_w // 2,
                diff_w - diff_w // 2,
                diff_h // 2,
                diff_h - diff_h // 2,
            ],
        )

        x = torch.cat([x, skip], dim=1)
        return self.block(x, time_emb)


class _Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        kernel_size: KernelSize,
    ) -> None:
        super().__init__()
        self.block = _ResBlock(in_channels, out_channels, time_dim, kernel_size)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        return self.block(x, time_emb)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_dim: int = 64,
        time_dim: int = 256,
        kernel_sizes: list[KernelSize] | None = None,
        strides: list[KernelSize] | None = None,
    ) -> None:
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [3, 3]
        if strides is None:
            strides = [2, 2]
        if len(kernel_sizes) != len(strides):
            raise ValueError("kernel_sizes and strides must have the same length.")

        self.time_dim = time_dim

        # The number of encoder blocks is controlled by the number of strides.
        deep_lvl = len(strides)
        self.features = [base_dim * (2**i) for i in range(deep_lvl)]

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        current_in = in_channels
        for feature, kernel_size, stride in zip(self.features, kernel_sizes, strides):
            self.encoders.append(
                _EncoderBlock(
                    in_channels=current_in,
                    out_channels=feature,
                    time_dim=time_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            )
            current_in = feature

        last_kernel = kernel_sizes[-1]
        self.bottleneck = _Bottleneck(
            in_channels=self.features[-1],
            out_channels=self.features[-1] * 2,
            time_dim=time_dim,
            kernel_size=last_kernel,
        )

        reversed_features = list(reversed(self.features))
        reversed_kernels = list(reversed(kernel_sizes))
        reversed_strides = list(reversed(strides))

        for feature, kernel_size, stride in zip(reversed_features, reversed_kernels, reversed_strides):
            self.decoders.append(
                _DecoderBlock(
                    in_channels=feature * 2,
                    out_channels=feature,
                    time_dim=time_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            )

        # Map the last hidden channels back to the requested output channels.
        self.final_conv = nn.Conv2d(base_dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        skip_connections: list[torch.Tensor] = []
        time_emb = timestep_embedding(timesteps, dim=self.time_dim)

        for encoder in self.encoders:
            x, skip = encoder(x, time_emb)
            skip_connections.append(skip)

        x = self.bottleneck(x, time_emb)

        for decoder, skip in zip(self.decoders, reversed(skip_connections)):
            x = decoder(x, skip, time_emb)

        return self.final_conv(x)
