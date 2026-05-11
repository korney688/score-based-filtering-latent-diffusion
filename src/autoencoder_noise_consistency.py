from __future__ import annotations

import torch
from torch import nn


class NoiseConsistencyEncoder(nn.Module):
    """Encoder used as a standalone image-to-latent mapping after autoencoder training"""
    def __init__(self, latent_dim: int = 16) -> None:
        super().__init__()
        # Downsample a 28x28 grayscale MNIST image into a compact latent vector
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NoiseConsistencyAutoencoder(nn.Module):
    """Autoencoder that trains the encoder with reconstruction loss
     and latent noise-consistency regularization"""
    def __init__(self, latent_dim: int = 16) -> None:
        super().__init__()
        self.encoder = NoiseConsistencyEncoder(latent_dim=latent_dim)
        # Reconstruct the image from the latent vector back to the 28x28 image grid
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (16, 7, 7)),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # autoencoder pass image -> latent vector -> reconstructed image
        z = self.encode(x)
        return self.decode(z)

    def noise_consistency_loss(self, x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
        # Add controlled Gaussian noise in image space.
        eps = torch.randn_like(x)
        x_noisy = (x + sigma * eps).clamp(-1.0, 1.0)

        # Compare latent representations of the clean and noisy versions
        z_clean = self.encode(x)
        z_noisy = self.encode(x_noisy)

        # Penalize latent changes caused by input noise, normalized by noise strength
        return torch.mean((z_noisy - z_clean) ** 2) / (sigma**2)
