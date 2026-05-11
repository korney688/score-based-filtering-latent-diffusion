from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class RepresentationEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 16,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        # Use pretrained ResNet18 features when torchvision weights are available
        weights = None
        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
            except Exception:
                weights = None

        try:
            backbone = models.resnet18(weights=weights)
        except Exception:
            backbone = models.resnet18(weights=None)
            weights = None

        # Remove ResNet's classification head and keep only feature extraction
        self.backbone_pretrained = weights is not None
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity() # only feature vector
        self.backbone = backbone # safe as backbone

        # Optionally freeze the representation backbone and train only the projector/decoder first
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Map high-dimensional ResNet features into the compact latent space used by the pipeline
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        # Convert MNIST tensors from [-1, 1] grayscale to ImageNet-normalized RGB
        x = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (x - mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(self._prepare_input(x))
        return self.projector(features)


class RepresentationAutoencoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 16,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = RepresentationEncoder(
            latent_dim=latent_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
        # Decode the compact latent vector back into a 28x28 grayscale reconstruction
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
        # autoencoder pass: image -> representation latent vector -> reconstruction
        z = self.encode(x)
        return self.decode(z)
