from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.autoencoder import SimpleAE
from src.autoencoder_noise_consistency import NoiseConsistencyAutoencoder
from src.autoencoder_representation import RepresentationAutoencoder
from src.autoencoder_vae import VariationalAutoencoder


ENCODER_VALIDATION_ROOT = PROJECT_ROOT / "experiments" / "exp_002_encoder_validation"
GEOMETRY_OUTPUT_DIR = ENCODER_VALIDATION_ROOT / "geometry"
RECONSTRUCTION_OUTPUT_DIR = ENCODER_VALIDATION_ROOT / "reconstruction"
BATCH_SIZE = 128
NUM_SAMPLES = 5000
SEED = 42
SIGMA_MIN = 0.1
SIGMA_MAX = 0.8
NUM_BINS = 10


ENCODER_SPECS = {
    "baseline": {
        "kind": "baseline",
        "checkpoint_path": PROJECT_ROOT / "outputs" / "ae_baseline_mnist" / "autoencoder_checkpoint.pt",
        "encoder_path": PROJECT_ROOT / "outputs" / "ae_baseline_mnist" / "E.pt",
    },
    "representation": {
        "kind": "representation",
        "checkpoint_path": PROJECT_ROOT / "outputs" / "ae_representation_mnist" / "autoencoder_checkpoint.pt",
        "encoder_path": PROJECT_ROOT / "outputs" / "ae_representation_mnist" / "E.pt",
    },
    "noise_consistency": {
        "kind": "noise_consistency",
        "checkpoint_path": PROJECT_ROOT / "outputs" / "ae_noise_consistency_mnist" / "autoencoder_checkpoint.pt",
        "encoder_path": PROJECT_ROOT / "outputs" / "ae_noise_consistency_mnist" / "E.pt",
    },
    "vae": {
        "kind": "vae",
        "checkpoint_path": PROJECT_ROOT / "outputs" / "ae_vae_mnist" / "autoencoder_checkpoint.pt",
        "encoder_path": PROJECT_ROOT / "outputs" / "ae_vae_mnist" / "E.pt",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Protocol stage-1 encoder validation.")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output-dir", type=Path, default=RECONSTRUCTION_OUTPUT_DIR)
    parser.add_argument("--geometry-output-dir", type=Path, default=GEOMETRY_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _extract_state_dict(maybe_state: Any) -> dict[str, torch.Tensor]:
    if isinstance(maybe_state, dict) and "state_dict" in maybe_state and isinstance(maybe_state["state_dict"], dict):
        return maybe_state["state_dict"]
    if isinstance(maybe_state, dict):
        return maybe_state
    raise ValueError("Unsupported checkpoint format: expected a state dict-like object.")


def load_autoencoder(kind: str, checkpoint_path: Path, encoder_path: Path, device: torch.device) -> torch.nn.Module:
    if kind == "baseline":
        model = SimpleAE().to(device)
    elif kind == "representation":
        model = RepresentationAutoencoder(pretrained=False).to(device)
    elif kind == "noise_consistency":
        model = NoiseConsistencyAutoencoder().to(device)
    elif kind == "vae":
        model = VariationalAutoencoder().to(device)
    else:
        raise ValueError(f"Unknown encoder kind: {kind}")

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        checkpoint_dict = checkpoint if isinstance(checkpoint, dict) else {}
        if "model_state_dict" in checkpoint_dict:
            model.load_state_dict(checkpoint_dict["model_state_dict"])
        else:
            model.load_state_dict(_extract_state_dict(checkpoint))
    elif encoder_path.exists():
        encoder_state = _extract_state_dict(torch.load(encoder_path, map_location=device))
        if kind == "vae" and {"encoder", "fc_mu", "fc_logvar"}.issubset(encoder_state):
            model.encoder.load_state_dict(encoder_state["encoder"])
            model.fc_mu.load_state_dict(encoder_state["fc_mu"])
            model.fc_logvar.load_state_dict(encoder_state["fc_logvar"])
        else:
            model.encoder.load_state_dict(encoder_state)
    else:
        raise FileNotFoundError(f"Missing encoder checkpoint: {checkpoint_path} or {encoder_path}")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def encode_deterministic(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    encoded = model.encode(x)
    if isinstance(encoded, tuple):
        return encoded[0]
    return encoded


def reconstruct_deterministic(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    output = model(x)
    if isinstance(output, tuple):
        return output[0]
    return output


def build_loader(num_samples: int, batch_size: int) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.MNIST(
        root=str(PROJECT_ROOT / "data"),
        train=False,
        download=False,
        transform=transform,
    )
    subset_indices = np.arange(min(num_samples, len(dataset)), dtype=np.int64)
    subset = torch.utils.data.Subset(dataset, subset_indices.tolist())
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)


def flatten_latent(z: torch.Tensor) -> torch.Tensor:
    if z.ndim == 2:
        return z
    if z.ndim >= 3:
        return z.flatten(start_dim=1)
    raise ValueError(f"Unexpected latent shape: {tuple(z.shape)}")


def sample_sigma(batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.empty(batch_size, 1, 1, 1, device=device).uniform_(SIGMA_MIN, SIGMA_MAX)


def save_geometry_scatter(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(df["sigma"], df["score_latent"], s=8, alpha=0.25)
    plt.xlabel("sigma")
    plt.ylabel("score_latent = ||delta_z||^2")
    plt.title("sigma vs score_latent")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_binned_curve(bin_centers: np.ndarray, bin_means: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(bin_centers, bin_means, marker="o")
    plt.xlabel("sigma bin center")
    plt.ylabel("mean(score_latent)")
    plt.title("Mean score_latent by sigma bin")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def reconstruction_ssim_simple(x: np.ndarray, y: np.ndarray) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    x_var = float(x.var())
    y_var = float(y.var())
    xy_cov = float(((x - x_mean) * (y - y_mean)).mean())
    numerator = (2.0 * x_mean * y_mean + c1) * (2.0 * xy_cov + c2)
    denominator = (x_mean**2 + y_mean**2 + c1) * (x_var + y_var + c2)
    return float(numerator / denominator)


def psnr_from_mse(mse_value: float) -> float:
    if mse_value <= 0:
        return float("inf")
    return float(20.0 * math.log10(2.0 / math.sqrt(mse_value)))


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


@torch.no_grad()
def evaluate_reconstruction(
    name: str,
    model: torch.nn.Module,
    loader: DataLoader,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    model_dir = output_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)

    total_mse = 0.0
    total_items = 0
    ssim_values: list[float] = []

    for x, _ in loader:
        x = x.to(device)
        reconstruction = reconstruct_deterministic(model, x) * 2.0 - 1.0
        mse_per_item = (reconstruction - x).flatten(start_dim=1).pow(2).mean(dim=1)
        total_mse += float(mse_per_item.sum().item())
        total_items += int(x.shape[0])

        x_np = x.detach().cpu().numpy()
        rec_np = reconstruction.detach().cpu().numpy()
        for item_idx in range(x_np.shape[0]):
            ssim_values.append(reconstruction_ssim_simple(x_np[item_idx, 0], rec_np[item_idx, 0]))

    reconstruction_mse = total_mse / total_items
    metrics = {
        "encoder": name,
        "reconstruction_mse": reconstruction_mse,
        "psnr": psnr_from_mse(reconstruction_mse),
        "ssim": float(np.mean(ssim_values)),
        "num_samples": total_items,
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


@torch.no_grad()
def evaluate_geometry(
    name: str,
    encoder: torch.nn.Module,
    loader: DataLoader,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    model_dir = output_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)

    sigma_values: list[np.ndarray] = []
    score_latent_values: list[np.ndarray] = []
    delta_z_norm_values: list[np.ndarray] = []

    for x, _ in loader:
        x = x.to(device)
        sigma = sample_sigma(x.shape[0], device)
        epsilon = torch.randn_like(x)
        x_noisy = x + sigma * epsilon

        z_clean = flatten_latent(encode_deterministic(encoder, x))
        z_noisy = flatten_latent(encode_deterministic(encoder, x_noisy))
        delta_z = z_noisy - z_clean

        score_latent = delta_z.pow(2).sum(dim=1)
        delta_z_norm = delta_z / sigma.view(-1, 1)

        sigma_values.append(sigma.view(-1).cpu().numpy().astype(np.float32))
        score_latent_values.append(score_latent.cpu().numpy().astype(np.float32))
        delta_z_norm_values.append(delta_z_norm.cpu().numpy().astype(np.float32))

    sigma_np = np.concatenate(sigma_values, axis=0)
    sigma_sq_np = sigma_np**2
    score_latent_np = np.concatenate(score_latent_values, axis=0)
    delta_z_norm_np = np.concatenate(delta_z_norm_values, axis=0)

    pearson_score_sigma2 = float(pd.Series(score_latent_np).corr(pd.Series(sigma_sq_np), method="pearson"))
    pearson_score_sigma = float(pd.Series(score_latent_np).corr(pd.Series(sigma_np), method="pearson"))
    spearman_score_sigma = float(pd.Series(score_latent_np).corr(pd.Series(sigma_np), method="spearman"))

    bin_edges = np.linspace(SIGMA_MIN, SIGMA_MAX, NUM_BINS + 1)
    bin_indices = np.clip(np.digitize(sigma_np, bin_edges) - 1, 0, NUM_BINS - 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_means = np.array(
        [
            float(score_latent_np[bin_indices == bin_idx].mean()) if np.any(bin_indices == bin_idx) else float("nan")
            for bin_idx in range(NUM_BINS)
        ],
        dtype=np.float32,
    )

    delta_z_norm_mean = delta_z_norm_np.mean(axis=0)
    delta_z_norm_var = delta_z_norm_np.var(axis=0)
    avg_variance = float(delta_z_norm_var.mean())

    cov = np.cov(delta_z_norm_np, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    min_eig = float(np.min(eigvals))
    max_eig = float(np.max(eigvals))
    anisotropy = float(max_eig / max(min_eig, 1e-12))

    df = pd.DataFrame(
        {
            "sigma": sigma_np.astype(np.float32),
            "score_latent": score_latent_np.astype(np.float32),
        }
    )
    csv_path = model_dir / "data.csv"
    scatter_path = model_dir / "sigma_vs_score_latent_scatter.png"
    binned_path = model_dir / "sigma_bins_mean_score_latent.png"
    df.to_csv(csv_path, index=False)
    save_geometry_scatter(df, scatter_path)
    save_binned_curve(bin_centers, bin_means, binned_path)

    metrics = {
        "encoder": name,
        "pearson_corr_score_sigma": pearson_score_sigma,
        "pearson_corr_score_sigma2": pearson_score_sigma2,
        "spearman_corr_score_sigma": spearman_score_sigma,
        "mean_var": avg_variance,
        "anisotropy": anisotropy,
        "delta_z_norm_mean": delta_z_norm_mean.tolist(),
        "delta_z_norm_variance_per_coordinate": delta_z_norm_var.tolist(),
        "covariance_matrix": cov.tolist(),
        "covariance_eigenvalues": eigvals.tolist(),
        "sigma_bin_edges": bin_edges.tolist(),
        "sigma_bin_centers": bin_centers.tolist(),
        "mean_score_latent_per_bin": bin_means.tolist(),
        "min_eigenvalue": min_eig,
        "max_eigenvalue": max_eig,
        "csv_path": str(csv_path),
        "scatter_path": str(scatter_path),
        "binned_path": str(binned_path),
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def run_reconstruction_validation(
    output_dir: Path = RECONSTRUCTION_OUTPUT_DIR,
    num_samples: int = NUM_SAMPLES,
    batch_size: int = BATCH_SIZE,
    device_arg: str | None = None,
) -> None:
    set_seed(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_arg or ("cuda" if torch.cuda.is_available() else "cpu"))
    loader = build_loader(num_samples=num_samples, batch_size=batch_size)

    rows = []
    for name, spec in ENCODER_SPECS.items():
        print(f"reconstruction_encoder={name}")
        model = load_autoencoder(
            kind=spec["kind"],
            checkpoint_path=spec["checkpoint_path"],
            encoder_path=spec["encoder_path"],
            device=device,
        )
        rows.append(evaluate_reconstruction(name, model, loader, output_dir, device))

    summary_df = pd.DataFrame(rows)
    summary_path = output_dir / "reconstruction_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    (output_dir / "reconstruction_summary.md").write_text(dataframe_to_markdown(summary_df) + "\n", encoding="utf-8")
    print(f"reconstruction_summary={summary_path}")


def noise_geometry_main(
    output_dir: Path = GEOMETRY_OUTPUT_DIR,
    num_samples: int = NUM_SAMPLES,
    batch_size: int = BATCH_SIZE,
    device_arg: str | None = None,
) -> None:
    set_seed(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_arg or ("cuda" if torch.cuda.is_available() else "cpu"))
    loader = build_loader(num_samples=num_samples, batch_size=batch_size)

    rows = []
    for name, spec in ENCODER_SPECS.items():
        print(f"geometry_encoder={name}")
        model = load_autoencoder(
            kind=spec["kind"],
            checkpoint_path=spec["checkpoint_path"],
            encoder_path=spec["encoder_path"],
            device=device,
        )
        rows.append(evaluate_geometry(name, model, loader, output_dir, device))

    summary_df = pd.DataFrame(
        [
            {
                "encoder": row["encoder"],
                "pearson_corr_score_sigma": row["pearson_corr_score_sigma"],
                "pearson_corr_score_sigma2": row["pearson_corr_score_sigma2"],
                "spearman_corr_score_sigma": row["spearman_corr_score_sigma"],
                "mean_var": row["mean_var"],
                "anisotropy": row["anisotropy"],
            }
            for row in rows
        ]
    )
    summary_path = output_dir / "geometry_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    (output_dir / "geometry_summary.md").write_text(dataframe_to_markdown(summary_df) + "\n", encoding="utf-8")
    print(f"geometry_summary={summary_path}")


def main() -> None:
    args = parse_args()
    run_reconstruction_validation(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device_arg=args.device,
    )
    noise_geometry_main(
        output_dir=args.geometry_output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device_arg=args.device,
    )


if __name__ == "__main__":
    main()
