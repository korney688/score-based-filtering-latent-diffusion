from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.Unet_model import UNet
from src.autoencoder import SimpleAE


class DDPM(nn.Module):
    def __init__(self, NN_model, n_steps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        super().__init__()

        self.n_steps = n_steps
        self.device = device
        self.model = NN_model.to(device)

        self.ae = SimpleAE().to(device)
        ae_weights_path = Path(__file__).resolve().parents[1] / "models" / "autoencoder.pth"
        self.ae.load_state_dict(torch.load(ae_weights_path, map_location=device))
        self.ae.eval()
        for param in self.ae.parameters():
            param.requires_grad = False

        self.latent_dim = self.ae.encoder[-1].out_features

        self.betas = torch.linspace(beta_start, beta_end, n_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def _encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.ae.encode(x)
        z = z.view(z.shape[0], z.shape[1], 1, 1)
        return z

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        z = z.view(z.shape[0], self.latent_dim)
        with torch.no_grad():
            x_recon = self.ae.decode(z)
        return x_recon

    def q_sample(self, x_0, t, noise=None):
        """
        Прямой процесс диффузии: добавляем шум к данным.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        alpha_bar_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise

        return x_t

    def get_score(self, x, t):
        """
        Вычисляет score function в латентном пространстве.
        """
        x = x.to(self.device)
        z_0 = self._encode_to_latent(x)
        print("z shape:", z_0.shape)

        noise = torch.randn_like(z_0)
        z_t = self.q_sample(z_0, t, noise)
        noise_pred = self.model(z_t, t)

        alpha_bar_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        score = -noise_pred / torch.sqrt(1 - alpha_bar_t)

        return score

    def train_step(self, x_0):
        """
        Один шаг обучения в латентном пространстве.
        """
        batch_size = x_0.shape[0]
        x_0 = x_0.to(self.device)

        t = torch.randint(0, self.n_steps, (batch_size,), device=self.device)

        z_0 = self._encode_to_latent(x_0)
        print("z shape:", z_0.shape)

        noise = torch.randn_like(z_0)
        z_t = self.q_sample(z_0, t, noise)
        noise_pred = self.model(z_t, t)

        loss = nn.functional.mse_loss(noise_pred, noise)

        return loss


def build_DDPM_model(base_dim: int = 16, deep: int = 3, device: str = "cpu") -> DDPM:
    """Инициализация UNet и DDPM для латентного пространства."""
    latent_dim = SimpleAE().encoder[-1].out_features

    aniso_kernel = (1, 1)
    aniso_stride = (1, 1)

    kernels = []
    strides = []

    for _ in range(deep):
        kernels.append(aniso_kernel)
        strides.append(aniso_stride)

    NN_model = UNet(
        in_channels=latent_dim,
        out_channels=latent_dim,
        base_dim=base_dim,
        time_dim=128,
        residual=True,
        kernel_sizes=kernels,
        strides=strides,
    )

    DDPM_model = DDPM(
        NN_model,
        n_steps=10000,
        beta_start=1e-4,
        beta_end=0.02,
        device=device,
    )

    return DDPM_model.to(device)
