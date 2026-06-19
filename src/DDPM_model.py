from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.Unet_model import UNet
from src.autoencoder import SimpleAE
from src.autoencoder_noise_consistency import build_noise_consistency_autoencoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AE_CHECKPOINT_PATH = (
    PROJECT_ROOT / "checkpoints" / "mnist" / "autoencoders" / "ae_noise_consistency_mnist" / "autoencoder_checkpoint.pt"
)


def _extract_state_dict(maybe_state: Any) -> dict[str, torch.Tensor]:
    # Some checkpoints save weights directly, others put them under "state_dict".
    if isinstance(maybe_state, dict) and "state_dict" in maybe_state and isinstance(maybe_state["state_dict"], dict):
        return maybe_state["state_dict"]
    if isinstance(maybe_state, dict):
        return maybe_state
    raise ValueError("Unsupported checkpoint format: expected a state dict-like object")


def _encode_deterministic(autoencoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # Use only the latent vector if the encoder returns extra values
    encoded = autoencoder.encode(x)
    if isinstance(encoded, tuple):
        return encoded[0]
    return encoded


def _cfg_get(cfg: Any | None, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def autoencoder_input_shape(dataset_cfg: Any | None = None) -> tuple[int, int]:
    in_channels = int(_cfg_get(dataset_cfg, "in_channels", _cfg_get(dataset_cfg, "channels", 1)))
    image_size = int(_cfg_get(dataset_cfg, "image_size", 28))
    return in_channels, image_size


def instantiate_autoencoder(
    kind: str,
    device: str | torch.device,
    dataset_cfg: Any | None = None,
    architecture: str | None = None,
    latent_dim: int | None = None,
) -> nn.Module:
    # Choose which autoencoder architecture should be used for the latent space
    if kind in {"baseline", "simple"}:
        autoencoder = SimpleAE().to(device)
    elif kind == "noise_consistency":
        encoder_cfg = _cfg_get(dataset_cfg, "encoder", {})
        resolved_architecture = architecture or _cfg_get(encoder_cfg, "name", "noise_consistency_small")
        resolved_latent_dim = int(latent_dim or _cfg_get(encoder_cfg, "latent_dim", 16))
        autoencoder = build_noise_consistency_autoencoder(
            architecture=resolved_architecture,
            dataset_cfg=dataset_cfg,
            latent_dim=resolved_latent_dim,
        ).to(device)
    else:
        raise ValueError(f"Unsupported latent-DDPM autoencoder kind: {kind}")
    return autoencoder


def _infer_noise_consistency_config_from_state_dict(state_dict: dict[str, torch.Tensor]) -> tuple[str | None, int | None]:
    architecture = None
    first_conv = state_dict.get("encoder.net.0.weight")
    if first_conv is not None:
        out_channels = int(first_conv.shape[0])
        if out_channels == 32:
            architecture = "noise_consistency_large"
        elif out_channels == 8:
            architecture = "noise_consistency_small"

    latent_dim = None
    latent_weight = state_dict.get("encoder.net.5.weight")
    if latent_weight is not None:
        latent_dim = int(latent_weight.shape[0])

    return architecture, latent_dim


def freeze_autoencoder(autoencoder: nn.Module) -> nn.Module:
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False
    return autoencoder


def load_frozen_autoencoder(
    kind: str,
    checkpoint_path: str | Path,
    device: str | torch.device,
    dataset_cfg: Any | None = None,
) -> nn.Module:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing autoencoder checkpoint: {checkpoint_path}")

    # Load the saved autoencoder weights before using it inside DDPM
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_dict = checkpoint if isinstance(checkpoint, dict) else {}
    state_dict = checkpoint_dict.get("model_state_dict") if "model_state_dict" in checkpoint_dict else _extract_state_dict(checkpoint)
    inferred_architecture, inferred_latent_dim = _infer_noise_consistency_config_from_state_dict(state_dict)
    architecture = checkpoint_dict.get("architecture", inferred_architecture)
    latent_dim = checkpoint_dict.get("latent_dim", inferred_latent_dim)

    autoencoder = instantiate_autoencoder(
        kind=kind,
        device=device,
        dataset_cfg=dataset_cfg,
        architecture=architecture,
        latent_dim=latent_dim,
    )
    if "model_state_dict" in checkpoint_dict:
        autoencoder.load_state_dict(state_dict)
    else:
        autoencoder.load_state_dict(state_dict)

    # Freeze the autoencoder: DDPM trains only the noise prediction model
    return freeze_autoencoder(autoencoder)


def infer_latent_dim(
    autoencoder: nn.Module,
    device: str | torch.device,
    dataset_cfg: Any | None = None,
) -> int:
    # Pass one fake image through the encoder to find the latent vector size
    in_channels, image_size = autoencoder_input_shape(dataset_cfg)
    with torch.no_grad():
        dummy = torch.zeros(1, in_channels, image_size, image_size, device=device)
        z = _encode_deterministic(autoencoder, dummy)
    return int(z.flatten(start_dim=1).shape[1])


class DDPM(nn.Module):
    def __init__(
        self,
        NN_model: nn.Module,
        autoencoder: nn.Module,
        latent_dim: int,
        latent_noise_mode: str = "baseline",
        n_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cuda",
    ):
        super().__init__()

        if latent_noise_mode not in {"baseline", "induced"}:
            raise ValueError(f"Unsupported latent_noise_mode: {latent_noise_mode}")

        self.n_steps = n_steps
        self.device = device
        self.model = NN_model.to(device)
        self.ae = autoencoder.to(device)
        self.latent_dim = latent_dim
        self.latent_noise_mode = latent_noise_mode

        # Linear beta schedule for the forward diffusion process
        self.betas = torch.linspace(beta_start, beta_end, n_steps).to(device)
        self.alphas = 1.0 - self.betas
        # alpha_bar_t is the product of all alpha values up to step t
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def _latent_noise_std(self, t: torch.Tensor) -> torch.Tensor:
        # Standard deviation of noise at timestep t.
        return torch.sqrt(1.0 - self.alphas_cumprod[t]).view(-1, 1, 1, 1)

    def _encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        # Encode images to latent vectors and reshape them like 1x1 feature maps for UNet
        with torch.no_grad():
            z = _encode_deterministic(self.ae, x)
        return z.view(z.shape[0], z.shape[1], 1, 1)

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        # Convert a latent vector back to image space using the frozen decoder
        z = z.view(z.shape[0], self.latent_dim)
        with torch.no_grad():
            x_recon = self.ae.decode(z)
        return x_recon

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        # Standard DDPM forward process: mix the clean sample with Gaussian noise.
        if noise is None:
            noise = torch.randn_like(x_0)

        alpha_bar_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t

    def _baseline_latent_noise(self, z_0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Add Gaussian noise directly in latent space.
        sigma = self._latent_noise_std(t)
        eps_z = torch.randn_like(z_0)
        z_noisy = z_0 + sigma * eps_z
        return z_noisy, eps_z

    def _induced_latent_noise(self, x_0: torch.Tensor, z_0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Add noise in image space first, then encode the noisy image to latent space.
        sigma = self._latent_noise_std(t)
        eps_x = torch.randn_like(x_0)
        x_noisy = x_0 + sigma * eps_x
        z_noisy = self._encode_to_latent(x_noisy)
        # This is the effective latent noise caused by image-space corruption.
        target_noise = (z_noisy - z_0) / sigma.clamp_min(1e-8)
        return z_noisy, target_noise

    def _make_noisy_latent_batch(
        self,
        x_0: torch.Tensor,
        z_0: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Select how noisy latent examples should be created for training.
        if self.latent_noise_mode == "baseline":
            return self._baseline_latent_noise(z_0, t)
        if self.latent_noise_mode == "induced":
            return self._induced_latent_noise(x_0, z_0, t)
        raise ValueError(f"Unsupported latent_noise_mode: {self.latent_noise_mode}")

    def get_score(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Estimate the score function from the model's predicted noise.
        x = x.to(self.device)
        z_0 = self._encode_to_latent(x)
        z_t, _ = self._make_noisy_latent_batch(x, z_0, t)
        noise_pred = self.model(z_t, t)

        sigma = self._latent_noise_std(t)
        score = -noise_pred / sigma.clamp_min(1e-8)
        return score

    def train_step(self, x_0: torch.Tensor) -> torch.Tensor:
        # One DDPM training step: sample time, add noise, predict noise, compute MSE.
        batch_size = x_0.shape[0]
        x_0 = x_0.to(self.device)
        t = torch.randint(0, self.n_steps, (batch_size,), device=self.device)

        z_0 = self._encode_to_latent(x_0)
        z_t, target_noise = self._make_noisy_latent_batch(x_0, z_0, t)
        noise_pred = self.model(z_t, t)
        return F.mse_loss(noise_pred, target_noise)


def build_DDPM_model(
    base_dim: int = 16,
    deep: int = 3,
    device: str = "cpu",
    latent_noise_mode: str = "baseline",
    autoencoder_kind: str = "baseline",
    autoencoder_checkpoint_path: str | Path = DEFAULT_AE_CHECKPOINT_PATH,
    dataset_cfg: Any | None = None,
    autoencoder: nn.Module | None = None,
) -> DDPM:
    # Load a pretrained autoencoder that defines the latent space.
    if autoencoder is None:
        autoencoder = load_frozen_autoencoder(
            kind=autoencoder_kind,
            checkpoint_path=autoencoder_checkpoint_path,
            device=device,
            dataset_cfg=dataset_cfg,
        )
    else:
        autoencoder = freeze_autoencoder(autoencoder.to(device))
    latent_dim = infer_latent_dim(autoencoder, device, dataset_cfg=dataset_cfg)

    aniso_kernel = (1, 1)
    aniso_stride = (1, 1)

    # The latent tensor has spatial size 1x1, so all UNet kernels and strides are 1x1.
    kernels = []
    strides = []

    for _ in range(deep):
        kernels.append(aniso_kernel)
        strides.append(aniso_stride)

    # UNet predicts the noise in latent space.
    NN_model = UNet(
        in_channels=latent_dim,
        out_channels=latent_dim,
        base_dim=base_dim,
        time_dim=128,
        kernel_sizes=kernels,
        strides=strides,
    )

    # Wrap the frozen autoencoder and the trainable UNet into one DDPM module.
    DDPM_model = DDPM(
        NN_model,
        autoencoder=autoencoder,
        latent_dim=latent_dim,
        latent_noise_mode=latent_noise_mode,
        n_steps=10000,
        beta_start=1e-4,
        beta_end=0.02,
        device=device,
    )

    return DDPM_model.to(device)
