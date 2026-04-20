from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.DDPM_model import build_DDPM_model


device = "cpu"
project_root = Path(__file__).resolve().parents[1]

# --- модель ---
model = build_DDPM_model(base_dim=16, deep=3, device=device)

checkpoint = torch.load(
    project_root / "experiments" / "exp_001" / "models" / "DDPM_model" / "best_model.pth",
    map_location=device,
)
model.model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# --- загрузка CLEAN изображения ---
with h5py.File(
    project_root
    / "experiments"
    / "exp_001"
    / "data"
    / "dataset_clean_mnist"
    / "dataset_clean_mnist.h5",
    "r",
) as f:
    img = f["dataset"][0]


# --- preprocessing ---
img = img.astype(np.float32)
img = (img - 0.5) / 0.5
img = np.expand_dims(img, axis=0)
x = torch.tensor(img).unsqueeze(0).float().to(device)


# --- переход в латентное пространство ---
with torch.no_grad():
    z = model.ae.encode(x)
z = z.view(z.shape[0], z.shape[1], 1, 1)
print("z shape:", z.shape)


# --- DDPM шаг в латентном пространстве ---
t = torch.tensor([20], device=device)
noise = torch.randn_like(z)
z_t = model.q_sample(z, t, noise)

with torch.no_grad():
    noise_pred = model.model(z_t, t)


# --- денойзинг и декодирование обратно в изображение ---
z_denoised = z_t - noise_pred
z_denoised = z_denoised.view(z_denoised.shape[0], model.latent_dim)

with torch.no_grad():
    x_recon = model.ae.decode(z_denoised)


# --- в numpy ---
clean = x[0, 0].cpu().numpy()
noisy_latent = z_t[0].mean(dim=0).cpu().numpy()
denoised = x_recon[0, 0].cpu().numpy()


# --- визуализация ---
plt.figure(figsize=(9, 3))

plt.subplot(1, 3, 1)
plt.title("Clean")
plt.imshow(clean, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Noisy latent")
plt.imshow(noisy_latent, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Decoded")
plt.imshow(denoised, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()


if __name__ == "__main__":
    pass
