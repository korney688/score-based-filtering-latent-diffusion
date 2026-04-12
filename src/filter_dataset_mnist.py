import torch
import h5py
import numpy as np
from src.DDPM_model import build_DDPM_model

device = 'cpu'

# === загрузка модели ===
model = build_DDPM_model(base_dim=16, deep=3, device=device)
checkpoint = torch.load(r"E:\ITMO\НИР Хватов\2 семместр\project\experiments\exp_001\models\DDPM_model\best_model.pth", map_location=device)
model.model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# === загрузка данных ===
with h5py.File(r"E:\ITMO\НИР Хватов\2 семместр\project\experiments\exp_001\data\dataset_noisy\dataset_noisy.h5", "r") as f:
    data = f["dataset"][:]

scores = []

for i in range(len(data)):
    img = data[i]

    # preprocessing как в dataset
    img = (img - 0.5) / 0.5
    img = np.expand_dims(img, 0)
    img = np.repeat(img, 2, axis=0)
    img = np.expand_dims(img, 1)

    x = torch.tensor(img).unsqueeze(0).float()

    t = torch.randint(0, model.n_steps, (1,))

    with torch.no_grad():
        noise_pred = model.model(x, t)

    score = noise_pred.pow(2).mean().item()
    scores.append(score)

scores = np.array(scores)

k = int(0.2 * len(scores))  # 20%
idx = np.argsort(scores)[:k]

filtered = data[idx]

with h5py.File("filtered_dataset.h5", "w") as f:
    f.create_dataset("dataset", data=filtered)
    f.create_dataset("scores", data=scores)
    f.create_dataset("selected_indices", data=idx)

print("original:", data.shape)
print("filtered:", filtered.shape)
print("saved scores:", scores.shape)
print("saved indices:", idx.shape)