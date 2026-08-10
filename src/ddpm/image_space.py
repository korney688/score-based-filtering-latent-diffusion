from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from src.Unet_model import UNet


def _cfg_get(cfg: Any | None, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class ImageSpaceDDPM(nn.Module):
    """Image-space epsilon-prediction DDPM wrapper.

    Checkpoints in the current CIFAR validation pipeline store only
    ``model.state_dict()``, so this wrapper keeps the trainable UNet under the
    same ``.model`` attribute used by the previous runner.
    """

    prediction_type = "epsilon"
    data_domain = "image_space"

    def __init__(
        self,
        nn_model: nn.Module,
        *,
        n_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__()
        self.n_steps = int(n_steps)
        self.device = torch.device(device)
        self.model = nn_model.to(self.device)

        betas = torch.linspace(float(beta_start), float(beta_end), self.n_steps, device=self.device)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))

    def diffusion_noise_std(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1.0 - self.alphas_cumprod[t]).view(-1, 1, 1, 1)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_0)
        alpha_bar_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        return torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1.0 - alpha_bar_t) * noise

    def train_step(self, x_0: torch.Tensor) -> torch.Tensor:
        batch_size = int(x_0.shape[0])
        x_0 = x_0.to(self.device)
        t = torch.randint(0, self.n_steps, (batch_size,), device=self.device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        eps_pred = self.model(x_t, t)
        return F.mse_loss(eps_pred, noise)


def build_image_space_ddpm_model(
    *,
    base_dim: int = 16,
    deep: int = 3,
    device: str | torch.device = "cpu",
    dataset_cfg: Any | None = None,
    in_channels: int | None = None,
    n_steps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> ImageSpaceDDPM:
    channels = int(in_channels or _cfg_get(dataset_cfg, "in_channels", _cfg_get(dataset_cfg, "channels", 3)))
    kernel_sizes = [3 for _ in range(int(deep))]
    strides = [2 for _ in range(int(deep))]
    nn_model = UNet(
        in_channels=channels,
        out_channels=channels,
        base_dim=int(base_dim),
        time_dim=128,
        kernel_sizes=kernel_sizes,
        strides=strides,
    )
    return ImageSpaceDDPM(
        nn_model,
        n_steps=int(n_steps),
        beta_start=float(beta_start),
        beta_end=float(beta_end),
        device=device,
    )
