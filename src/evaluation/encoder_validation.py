from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.autoencoder import SimpleAE
from src.autoencoder_noise_consistency import build_noise_consistency_autoencoder
from src.autoencoder_representation import RepresentationAutoencoder
from src.autoencoder_vae import VariationalAutoencoder
from src.dataset_registry import DATASET_SPECS, build_torchvision_split, dataset_name


BATCH_SIZE = 128
NUM_SAMPLES = 5000
SEED = 42
SIGMA_MIN = 0.1
SIGMA_MAX = 0.8
NUM_BINS = 10


@dataclass(frozen=True)
class EncoderValidationContext:
    dataset_cfg: dict[str, Any]
    dataset_slug: str
    root: Path
    reconstruction_dir: Path
    geometry_dir: Path
    metrics_dir: Path
    plots_dir: Path
    covariance_dir: Path
    eigenspectrum_dir: Path
    report_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Protocol stage-1 encoder validation.")
    parser.add_argument("mode_arg", nargs="?", choices=["compare-encoders", "noise-geometry", "all"])
    parser.add_argument("--mode", default=None, choices=["compare-encoders", "noise-geometry", "all"])
    parser.add_argument("--dataset", default="mnist", choices=sorted(DATASET_SPECS))
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--geometry-output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    args.mode = args.mode or args.mode_arg or "compare-encoders"
    return args


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_dataset_config(slug: str) -> dict[str, Any]:
    config_path = PROJECT_ROOT / "configs" / "dataset" / f"{slug}.yaml"
    cfg = OmegaConf.load(config_path)
    dataset_cfg = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(dataset_cfg, dict):
        raise TypeError(f"Dataset config must be a mapping: {config_path}")
    return dataset_cfg


def build_context(args: argparse.Namespace) -> EncoderValidationContext:
    dataset_cfg = load_dataset_config(args.dataset)
    slug = dataset_name(dataset_cfg)
    root = args.output_root or PROJECT_ROOT / "experiments" / slug / "exp_002_encoder_validation"
    return EncoderValidationContext(
        dataset_cfg=dataset_cfg,
        dataset_slug=slug,
        root=root,
        reconstruction_dir=args.output_dir or root / "reconstructions",
        geometry_dir=args.geometry_output_dir or root / "latent_geometry",
        metrics_dir=root / "metrics",
        plots_dir=root / "plots",
        covariance_dir=root / "covariance",
        eigenspectrum_dir=root / "eigenspectrum",
        report_dir=root / "report",
    )


def ensure_context_dirs(context: EncoderValidationContext) -> None:
    for path in [
        context.root,
        context.reconstruction_dir,
        context.geometry_dir,
        context.metrics_dir,
        context.plots_dir,
        context.covariance_dir,
        context.eigenspectrum_dir,
        context.report_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def _extract_state_dict(maybe_state: Any) -> dict[str, torch.Tensor]:
    if isinstance(maybe_state, dict) and "state_dict" in maybe_state and isinstance(maybe_state["state_dict"], dict):
        return maybe_state["state_dict"]
    if isinstance(maybe_state, dict):
        return maybe_state
    raise ValueError("Unsupported checkpoint format: expected a state dict-like object.")


def encoder_validation_candidates(dataset_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    validation_cfg = dataset_cfg.get("encoder_validation", {})
    candidates = validation_cfg.get("candidates", {}) if isinstance(validation_cfg, dict) else {}
    if not isinstance(candidates, dict) or not candidates:
        encoder_cfg = dataset_cfg.get("encoder", {}) if isinstance(dataset_cfg.get("encoder", {}), dict) else {}
        architecture = encoder_cfg.get("name", "noise_consistency_small")
        latent_dim = int(encoder_cfg.get("latent_dim", 16))
        candidates = {
            "noise_consistency": {
                "kind": "noise_consistency",
                "architecture": architecture,
                "latent_dim": latent_dim,
                "checkpoint_run": f"{architecture}_latent{latent_dim}",
            }
        }
    return candidates


def resolve_checkpoint_paths(context: EncoderValidationContext, spec: dict[str, Any]) -> tuple[Path, Path]:
    autoencoder_root = PROJECT_ROOT / "checkpoints" / context.dataset_slug / "autoencoders"
    checkpoint_run = spec.get("checkpoint_run")
    if not checkpoint_run:
        architecture = spec.get("architecture")
        latent_dim = spec.get("latent_dim")
        if architecture and latent_dim:
            checkpoint_run = f"{architecture}_latent{int(latent_dim)}"
        else:
            raise ValueError(f"Missing checkpoint_run in encoder validation spec: {spec}")
    run_dir = autoencoder_root / str(checkpoint_run)
    return run_dir / "autoencoder_checkpoint.pt", run_dir / "E.pt"


def _checkpoint_metadata(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    if not checkpoint_path.exists():
        return {}
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        return checkpoint
    return {}


def load_autoencoder(
    spec: dict[str, Any],
    dataset_cfg: dict[str, Any],
    checkpoint_path: Path,
    encoder_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    kind = str(spec["kind"])
    if kind == "baseline":
        model = SimpleAE().to(device)
    elif kind == "representation":
        model = RepresentationAutoencoder(pretrained=False).to(device)
    elif kind == "noise_consistency":
        checkpoint_meta = _checkpoint_metadata(checkpoint_path, device)
        architecture = spec.get("architecture") or checkpoint_meta.get("architecture") or dataset_cfg.get("encoder", {}).get("name")
        latent_dim = int(spec.get("latent_dim") or checkpoint_meta.get("latent_dim") or dataset_cfg.get("encoder", {}).get("latent_dim", 16))
        model = build_noise_consistency_autoencoder(
            architecture=architecture,
            dataset_cfg=dataset_cfg,
            latent_dim=latent_dim,
        ).to(device)
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


def build_loader(dataset_cfg: dict[str, Any], num_samples: int, batch_size: int) -> DataLoader:
    dataset = build_torchvision_split(
        dataset_cfg=dataset_cfg,
        train=False,
        data_root=PROJECT_ROOT / "data",
        transform_profile="normalized",
        download=bool(dataset_cfg.get("download", False)),
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


def save_covariance_heatmap(cov: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(cov, cmap="viridis")
    plt.colorbar(label="covariance")
    plt.title("Latent noise covariance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_eigenspectrum(eigvals: np.ndarray, output_path: Path) -> None:
    sorted_vals = np.sort(eigvals)[::-1]
    plt.figure(figsize=(7, 5))
    plt.plot(np.arange(1, len(sorted_vals) + 1), sorted_vals, marker="o", linewidth=1)
    plt.xlabel("eigenvalue index")
    plt.ylabel("eigenvalue")
    plt.title("Latent covariance eigenspectrum")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def denormalize_to_unit_interval(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def save_reconstruction_grid(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    output_path: Path,
    max_items: int = 8,
) -> None:
    num_items = min(max_items, original.shape[0])
    original = denormalize_to_unit_interval(original[:num_items]).cpu()
    reconstructed = denormalize_to_unit_interval(reconstructed[:num_items]).cpu()
    fig, axes = plt.subplots(2, num_items, figsize=(2 * num_items, 4))
    if num_items == 1:
        axes = np.array(axes).reshape(2, 1)
    for idx in range(num_items):
        if original.shape[1] == 1:
            axes[0, idx].imshow(original[idx, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
            axes[1, idx].imshow(reconstructed[idx, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        else:
            axes[0, idx].imshow(original[idx].permute(1, 2, 0).numpy(), vmin=0.0, vmax=1.0)
            axes[1, idx].imshow(reconstructed[idx].permute(1, 2, 0).numpy(), vmin=0.0, vmax=1.0)
        axes[0, idx].axis("off")
        axes[1, idx].axis("off")
    axes[0, 0].set_ylabel("clean")
    axes[1, 0].set_ylabel("recon")
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
    context: EncoderValidationContext,
    device: torch.device,
) -> dict[str, Any]:
    model_dir = context.reconstruction_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)

    total_mse = 0.0
    total_items = 0
    ssim_values: list[float] = []
    example_original = None
    example_reconstruction = None

    for x, _ in loader:
        x = x.to(device)
        reconstruction = reconstruct_deterministic(model, x) * 2.0 - 1.0
        mse_per_item = (reconstruction - x).flatten(start_dim=1).pow(2).mean(dim=1)
        total_mse += float(mse_per_item.sum().item())
        total_items += int(x.shape[0])

        if example_original is None:
            example_original = x.detach().cpu()
            example_reconstruction = reconstruction.detach().cpu()

        x_np = x.detach().cpu().numpy()
        rec_np = reconstruction.detach().cpu().numpy()
        for item_idx in range(x_np.shape[0]):
            ssim_values.append(reconstruction_ssim_simple(x_np[item_idx], rec_np[item_idx]))

    reconstruction_mse = total_mse / total_items
    grid_path = model_dir / "reconstruction_grid.png"
    if example_original is not None and example_reconstruction is not None:
        save_reconstruction_grid(example_original, example_reconstruction, grid_path)

    metrics = {
        "encoder": name,
        "dataset": context.dataset_slug,
        "reconstruction_mse": reconstruction_mse,
        "psnr": psnr_from_mse(reconstruction_mse),
        "ssim": float(np.mean(ssim_values)),
        "num_samples": total_items,
        "reconstruction_grid": str(grid_path),
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


@torch.no_grad()
def evaluate_geometry(
    name: str,
    encoder: torch.nn.Module,
    loader: DataLoader,
    context: EncoderValidationContext,
    device: torch.device,
) -> dict[str, Any]:
    model_dir = context.geometry_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)
    covariance_model_dir = context.covariance_dir / name
    eigenspectrum_model_dir = context.eigenspectrum_dir / name
    covariance_model_dir.mkdir(parents=True, exist_ok=True)
    eigenspectrum_model_dir.mkdir(parents=True, exist_ok=True)

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
    covariance_path = covariance_model_dir / "covariance_matrix.png"
    eigenspectrum_path = eigenspectrum_model_dir / "eigenspectrum.png"
    df.to_csv(csv_path, index=False)
    save_geometry_scatter(df, scatter_path)
    save_binned_curve(bin_centers, bin_means, binned_path)
    save_covariance_heatmap(cov, covariance_path)
    save_eigenspectrum(eigvals, eigenspectrum_path)

    metrics = {
        "encoder": name,
        "dataset": context.dataset_slug,
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
        "covariance_plot_path": str(covariance_path),
        "eigenspectrum_plot_path": str(eigenspectrum_path),
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _load_models_for_context(
    context: EncoderValidationContext,
    device: torch.device,
) -> list[tuple[str, dict[str, Any], torch.nn.Module]]:
    models = []
    for name, spec in encoder_validation_candidates(context.dataset_cfg).items():
        checkpoint_path, encoder_path = resolve_checkpoint_paths(context, spec)
        print(f"encoder={name} checkpoint={checkpoint_path}")
        model = load_autoencoder(
            spec=spec,
            dataset_cfg=context.dataset_cfg,
            checkpoint_path=checkpoint_path,
            encoder_path=encoder_path,
            device=device,
        )
        models.append((name, spec, model))
    return models


def write_summary(context: EncoderValidationContext, reconstruction_rows: list[dict[str, Any]], geometry_rows: list[dict[str, Any]]) -> None:
    summary = {
        "dataset": context.dataset_slug,
        "root": str(context.root),
        "reconstruction_summary": str(context.metrics_dir / "reconstruction_summary.csv"),
        "geometry_summary": str(context.metrics_dir / "geometry_summary.csv"),
        "encoders": [row["encoder"] for row in reconstruction_rows or geometry_rows],
    }
    (context.root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        f"# Encoder Validation Summary: {context.dataset_slug}",
        "",
        "## Reconstruction",
    ]
    if reconstruction_rows:
        report_lines.append(dataframe_to_markdown(pd.DataFrame(reconstruction_rows)))
    report_lines.extend(["", "## Latent Geometry"])
    if geometry_rows:
        compact_geometry = pd.DataFrame(
            [
                {
                    "encoder": row["encoder"],
                    "pearson_corr_score_sigma": row["pearson_corr_score_sigma"],
                    "pearson_corr_score_sigma2": row["pearson_corr_score_sigma2"],
                    "spearman_corr_score_sigma": row["spearman_corr_score_sigma"],
                    "mean_var": row["mean_var"],
                    "anisotropy": row["anisotropy"],
                }
                for row in geometry_rows
            ]
        )
        report_lines.append(dataframe_to_markdown(compact_geometry))
    (context.report_dir / "encoder_validation_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def run_reconstruction_validation(
    context: EncoderValidationContext,
    num_samples: int,
    batch_size: int,
    device_arg: str | None,
) -> list[dict[str, Any]]:
    set_seed(SEED)
    ensure_context_dirs(context)
    device = torch.device(device_arg or ("cuda" if torch.cuda.is_available() else "cpu"))
    loader = build_loader(context.dataset_cfg, num_samples=num_samples, batch_size=batch_size)

    rows = []
    for name, _, model in _load_models_for_context(context, device):
        print(f"reconstruction_encoder={name}")
        rows.append(evaluate_reconstruction(name, model, loader, context, device))

    summary_df = pd.DataFrame(rows)
    summary_path = context.metrics_dir / "reconstruction_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    (context.metrics_dir / "reconstruction_summary.md").write_text(dataframe_to_markdown(summary_df) + "\n", encoding="utf-8")
    print(f"reconstruction_summary={summary_path}")
    return rows


def run_geometry_validation(
    context: EncoderValidationContext,
    num_samples: int,
    batch_size: int,
    device_arg: str | None,
) -> list[dict[str, Any]]:
    set_seed(SEED)
    ensure_context_dirs(context)
    device = torch.device(device_arg or ("cuda" if torch.cuda.is_available() else "cpu"))
    loader = build_loader(context.dataset_cfg, num_samples=num_samples, batch_size=batch_size)

    rows = []
    for name, _, model in _load_models_for_context(context, device):
        print(f"geometry_encoder={name}")
        rows.append(evaluate_geometry(name, model, loader, context, device))

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
    summary_path = context.metrics_dir / "geometry_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    (context.metrics_dir / "geometry_summary.md").write_text(dataframe_to_markdown(summary_df) + "\n", encoding="utf-8")
    print(f"geometry_summary={summary_path}")
    return rows


def noise_geometry_main() -> None:
    args = parse_args()
    context = build_context(args)
    geometry_rows = run_geometry_validation(
        context=context,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device_arg=args.device,
    )
    write_summary(context, reconstruction_rows=[], geometry_rows=geometry_rows)


def main() -> None:
    args = parse_args()
    context = build_context(args)
    reconstruction_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    if args.mode in {"compare-encoders", "all"}:
        reconstruction_rows = run_reconstruction_validation(
            context=context,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            device_arg=args.device,
        )
        geometry_rows = run_geometry_validation(
            context=context,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            device_arg=args.device,
        )
    elif args.mode == "noise-geometry":
        geometry_rows = run_geometry_validation(
            context=context,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            device_arg=args.device,
        )
    write_summary(context, reconstruction_rows=reconstruction_rows, geometry_rows=geometry_rows)


if __name__ == "__main__":
    main()
