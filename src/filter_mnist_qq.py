import torch
import h5py
import numpy as np
from src.DDPM_model import build_DDPM_model

MODEL_PATH = r"E:\ITMO\НИР Хватов\2 семместр\project\experiments\exp_001\models\DDPM_model\best_model.pth"
DATA_PATH = r"E:\ITMO\НИР Хватов\2 семместр\project\experiments\exp_001\data\dataset_noisy\dataset_noisy.h5"
OUTPUT_PATH = "filtered_dataset_qq.h5"

LOW_Q = 0.05
HIGH_Q = 0.95

device = 'cpu'

model = build_DDPM_model(base_dim=16, deep=3, device=device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with h5py.File(DATA_PATH, "r") as f:
    data = f["dataset"][:]

scores = []

print("Calculating scores...")

for i in range(len(data)):
    img = data[i]

    # preprocessing
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

low_thr = np.quantile(scores, LOW_Q)
high_thr = np.quantile(scores, HIGH_Q)

mask = (scores >= low_thr) & (scores <= high_thr)

filtered = data[mask]

print("Original:", data.shape)
print("Filtered:", filtered.shape)
print("Score range:", low_thr, high_thr)

with h5py.File(OUTPUT_PATH, "w") as f:
    f.create_dataset("dataset", data=filtered)

print("Saved:", OUTPUT_PATH)