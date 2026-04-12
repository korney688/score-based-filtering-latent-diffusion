import torch
from src.DDPM_model import build_DDPM_model
import h5py
import numpy as np
import matplotlib.pyplot as plt

device = 'cpu'

model = build_DDPM_model(base_dim=16, deep=3, device=device)

checkpoint = torch.load(r"E:\ITMO\НИР Хватов\2 семместр\project\experiments\exp_001\models\DDPM_model\best_model.pth", map_location=device)
model.model.load_state_dict(checkpoint['model_state_dict'])

model.eval()


with h5py.File("dataset_noisy.h5", "r") as f:
    img = f["dataset"][0]  # (28, 28)

# preprocessing как в dataset
img = (img - 0.5) / 0.5
img = np.expand_dims(img, axis=0)
img = np.repeat(img, 2, axis=0)
img = np.expand_dims(img, axis=1)

x = torch.tensor(img).unsqueeze(0).float()  # [1, 2, 1, 28, 28]

# Прогон
t = torch.randint(0, model.n_steps, (1,))

with torch.no_grad():
    noise_pred = model.model(x, t)


noise_img = noise_pred[0, 0, 0].cpu().numpy()
noise_true = torch.randn_like(x)[0,0,0].numpy()
denoised = x - noise_pred
img = denoised[0,0,0].cpu().numpy()

plt.imshow(img, cmap='gray')
plt.title("Approx denoised")
plt.show()
