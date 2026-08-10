from __future__ import annotations

from typing import Any

import torch
from torch import nn


class NoiseConsistencyBase(nn.Module):
    """Shared AE behavior: image reconstruction plus latent noise consistency."""

    architecture_name = "noise_consistency_base"

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

    def noise_consistency_loss(self, x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
        eps = torch.randn_like(x)
        x_noisy = (x + sigma * eps).clamp(-1.0, 1.0)
        z_clean = self.encode(x)
        z_noisy = self.encode(x_noisy)
        return torch.mean((z_noisy - z_clean) ** 2) / (sigma**2)


class NoiseConsistencyEncoderSmall(nn.Module):
    """Original lightweight encoder used by the MNIST pipeline."""

    def __init__(
        self,
        latent_dim: int = 16,
        in_channels: int = 1,
        image_size: int = 28,
    ) -> None:
        super().__init__()
        encoded_spatial_size = encoded_spatial_size_after_two_downsamples(image_size)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * encoded_spatial_size * encoded_spatial_size, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NoiseConsistencyAESmall(NoiseConsistencyBase):
    """Original noise-consistency autoencoder architecture."""

    architecture_name = "noise_consistency_small"

    def __init__(
        self,
        latent_dim: int = 16,
        in_channels: int = 1,
        out_channels: int | None = None,
        image_size: int = 28,
    ) -> None:
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        encoded_spatial_size = _validate_two_downsample_image_size(image_size, self.__class__.__name__)

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size
        self.encoded_spatial_size = encoded_spatial_size

        self.encoder = NoiseConsistencyEncoderSmall(
            latent_dim=latent_dim,
            in_channels=in_channels,
            image_size=image_size,
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16 * encoded_spatial_size * encoded_spatial_size),
            nn.ReLU(),
            nn.Unflatten(1, (16, encoded_spatial_size, encoded_spatial_size)),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    @classmethod
    def from_dataset_config(cls, dataset_cfg: Any, latent_dim: int = 16) -> "NoiseConsistencyAESmall":
        return cls(
            latent_dim=latent_dim,
            in_channels=int(_cfg_get(dataset_cfg, "in_channels", _cfg_get(dataset_cfg, "channels", 1))),
            out_channels=int(_cfg_get(dataset_cfg, "out_channels", _cfg_get(dataset_cfg, "channels", 1))),
            image_size=int(_cfg_get(dataset_cfg, "image_size", 28)),
        )


class NoiseConsistencyEncoderLarge(nn.Module):
    """Wider encoder for CIFAR-10 noise-consistency adaptation studies."""

    def __init__(
        self,
        latent_dim: int = 64,
        in_channels: int = 3,
        image_size: int = 32,
    ) -> None:
        super().__init__()
        encoded_spatial_size = encoded_spatial_size_after_two_downsamples(image_size)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * encoded_spatial_size * encoded_spatial_size, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NoiseConsistencyAELarge(NoiseConsistencyBase):
    """Expanded noise-consistency autoencoder with the same training objective."""

    architecture_name = "noise_consistency_large"

    def __init__(
        self,
        latent_dim: int = 64,
        in_channels: int = 3,
        out_channels: int | None = None,
        image_size: int = 32,
    ) -> None:
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        encoded_spatial_size = _validate_two_downsample_image_size(image_size, self.__class__.__name__)

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size
        self.encoded_spatial_size = encoded_spatial_size

        self.encoder = NoiseConsistencyEncoderLarge(
            latent_dim=latent_dim,
            in_channels=in_channels,
            image_size=image_size,
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * encoded_spatial_size * encoded_spatial_size),
            nn.ReLU(),
            nn.Unflatten(1, (64, encoded_spatial_size, encoded_spatial_size)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    @classmethod
    def from_dataset_config(cls, dataset_cfg: Any, latent_dim: int = 64) -> "NoiseConsistencyAELarge":
        return cls(
            latent_dim=latent_dim,
            in_channels=int(_cfg_get(dataset_cfg, "in_channels", _cfg_get(dataset_cfg, "channels", 3))),
            out_channels=int(_cfg_get(dataset_cfg, "out_channels", _cfg_get(dataset_cfg, "channels", 3))),
            image_size=int(_cfg_get(dataset_cfg, "image_size", 32)),
        )


class NoiseConsistencyAutoencoder(NoiseConsistencyAESmall):
    """Backward-compatible name for the original small architecture."""


NoiseConsistencyEncoder = NoiseConsistencyEncoderSmall


NOISE_CONSISTENCY_AE_REGISTRY = {
    "noise_consistency_small": NoiseConsistencyAESmall,
    "small": NoiseConsistencyAESmall,
    "noise_consistency_large": NoiseConsistencyAELarge,
    "large": NoiseConsistencyAELarge,
}


def build_noise_consistency_autoencoder(
    architecture: str,
    dataset_cfg: Any,
    latent_dim: int,
) -> NoiseConsistencyBase:
    key = normalize_noise_consistency_architecture_name(architecture)
    return NOISE_CONSISTENCY_AE_REGISTRY[key].from_dataset_config(dataset_cfg, latent_dim=latent_dim)


def normalize_noise_consistency_architecture_name(name: str | None) -> str:
    key = str(name or "noise_consistency_small").strip().lower().replace("-", "_")
    if key in {"small", "noise_consistency_small"}:
        return "noise_consistency_small"
    if key in {"large", "noise_consistency_large"}:
        return "noise_consistency_large"
    raise ValueError(
        f"Unsupported noise-consistency architecture: {name}. "
        "Supported: noise_consistency_small, noise_consistency_large."
    )


def short_noise_consistency_architecture_name(name: str | None) -> str:
    normalized = normalize_noise_consistency_architecture_name(name)
    return normalized.removeprefix("noise_consistency_")


def encoded_spatial_size_after_two_downsamples(image_size: int) -> int:
    size = int(image_size)
    for _ in range(2):
        size = (size + 1) // 2
    return size


def _validate_two_downsample_image_size(image_size: int, class_name: str) -> int:
    encoded_spatial_size = encoded_spatial_size_after_two_downsamples(image_size)
    decoded_size = encoded_spatial_size * 4
    if decoded_size != image_size:
        raise ValueError(
            f"{class_name} currently supports image sizes that are exactly reconstructed "
            f"by two stride-2 transpose convolutions. Got image_size={image_size}, "
            f"decoded_size={decoded_size}."
        )
    return encoded_spatial_size


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)
