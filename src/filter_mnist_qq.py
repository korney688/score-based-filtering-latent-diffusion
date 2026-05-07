from pathlib import Path

import h5py
import numpy as np
import torch

from src.DDPM_model import build_DDPM_model


project_root = Path(__file__).resolve().parents[1]
model_path = project_root / "experiments" / "exp_001" / "models" / "DDPM_model" / "best_model.pth"
data_path = project_root / "experiments" / "exp_001" / "data" / "dataset_noisy_var" / "dataset_noisy_var.h5"
output_dir = project_root / "outputs" / "final_results" / "filtering" / "qq"

quantile_ranges = [
    (0.80, 1.00),
    (0.60, 1.00),
    (0.40, 1.00),
]

device = "cpu"
t_values = [10, 30, 50]
torch.manual_seed(0)


def save_filter_info(
    info_path: Path,
    selected_count: int,
    total_count: int,
    low_q: float,
    high_q: float,
) -> None:
    info_path.write_text(
        "\n".join(
            [
                "filter_type=qq",
                f"low_quantile={low_q}",
                f"high_quantile={high_q}",
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

for low_q, high_q in quantile_ranges:
    low_thr = np.quantile(scores, low_q)
    high_thr = np.quantile(scores, high_q)

    mask = (scores >= low_thr) & (scores <= high_thr)
    filtered = data[mask]
    selected_count = int(mask.sum())

    low_label = int(low_q * 100)
    high_label = int(high_q * 100)
    output_path = output_dir / f"filtered_dataset_qq_upper_{low_label}_{high_label}.h5"
    info_path = output_dir / f"filtered_dataset_qq_upper_{low_label}_{high_label}_info.txt"

    with h5py.File(output_path, "w") as f:
        f.create_dataset("dataset", data=filtered)
        f.create_dataset("scores", data=scores)
        f.create_dataset("selected_indices", data=np.where(mask)[0])

    save_filter_info(
        info_path=info_path,
        selected_count=selected_count,
        total_count=len(scores),
        low_q=low_q,
        high_q=high_q,
    )

    print(f"qq {low_label}-{high_label}")
    print("original:", data.shape)
    print("filtered:", filtered.shape)
    print("saved:", output_path)
    print("info:", info_path)
