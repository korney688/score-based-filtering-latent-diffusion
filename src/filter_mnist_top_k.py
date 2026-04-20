from pathlib import Path

import h5py
import numpy as np
import torch

from src.DDPM_model import build_DDPM_model


project_root = Path(__file__).resolve().parents[1]
model_path = project_root / "experiments" / "exp_001" / "models" / "DDPM_model" / "best_model.pth"
data_path = project_root / "experiments" / "exp_001" / "data" / "dataset_noisy_var" / "dataset_noisy_var.h5"
output_dir = project_root / "src" / "filtered_mnist_topk"

device = "cpu"
t_values = [10, 30, 50]
topk_fractions = [0.2, 0.4, 0.6]
torch.manual_seed(0)


def save_filter_info(info_path: Path, selected_count: int, total_count: int, fraction: float) -> None:
    info_path.write_text(
        "\n".join(
            [
                f"filter_type=top_k",
                f"fraction={fraction}",
                f"selected_count={selected_count}",
                f"total_count={total_count}",
            ]
        ),
        encoding="utf-8",
    )


model = build_DDPM_model(base_dim=16, deep=3, device=device)
checkpoint = torch.load(model_path, map_location=device)
model.model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
output_dir.mkdir(parents=True, exist_ok=True)

with h5py.File(data_path, "r") as f:
    data = f["dataset"][:]

scores = []

for i in range(len(data)):
    img = data[i]

    img = img.astype(np.float32)
    img = (img - 0.5) / 0.5
    img = np.expand_dims(img, 0)

    x = torch.tensor(img).unsqueeze(0).float().to(device)

    if x.shape != (1, 1, 28, 28):
        raise ValueError(f"Expected x shape [1, 1, 28, 28], got {tuple(x.shape)}")

    with torch.no_grad():
        z = model.ae.encode(x)
    z = z.view(z.shape[0], z.shape[1], 1, 1)

    score_accum = 0.0

    for t_val in t_values:
        t = torch.tensor([t_val], device=device)
        noise = torch.randn_like(z)
        z_t = model.q_sample(z, t, noise)

        with torch.no_grad():
            noise_pred = model.model(z_t, t)

        score_accum += noise_pred.pow(2).mean().item()

    score = score_accum / len(t_values)
    scores.append(score)

scores = np.array(scores)

print("Score stats:")
print("min:", scores.min())
print("max:", scores.max())
print("mean:", scores.mean())

for fraction in topk_fractions:
    k = int(fraction * len(scores))
    idx = np.argsort(scores)[-k:]
    filtered = data[idx]

    fraction_label = int(fraction * 100)
    output_path = output_dir / f"filtered_dataset_topk_{fraction_label}pct.h5"
    info_path = output_dir / f"filtered_dataset_topk_{fraction_label}pct_info.txt"

    with h5py.File(output_path, "w") as f:
        f.create_dataset("dataset", data=filtered)
        f.create_dataset("scores", data=scores)
        f.create_dataset("selected_indices", data=idx)

    save_filter_info(
        info_path=info_path,
        selected_count=len(idx),
        total_count=len(scores),
        fraction=fraction,
    )

    print(f"top-k {fraction_label}%")
    print("original:", data.shape)
    print("filtered:", filtered.shape)
    print("saved:", output_path)
    print("info:", info_path)
