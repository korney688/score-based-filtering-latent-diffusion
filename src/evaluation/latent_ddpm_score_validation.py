from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.DDPM_model import build_DDPM_model
from src.dataset_registry import DATASET_SPECS, build_torchvision_split, dataset_name


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BATCH_SIZE = 128
NUM_SAMPLES = 2000
SEED = 42
SIGMA_MIN = 0.1
SIGMA_MAX = 0.8
NUM_BINS = 10


@dataclass(frozen=True)
class LatentDDPMValidationContext:
    dataset_cfg: dict[str, Any]
    dataset_slug: str
    root: Path
    metrics_dir: Path
    score_validation_dir: Path
    score_distributions_dir: Path
    noise_prediction_dir: Path
    covariance_dir: Path
    report_dir: Path
    autoencoder_checkpoint_path: Path


@dataclass(frozen=True)
class ModeOutputDirs:
    metrics_path: Path
    score_validation_dir: Path
    score_distributions_dir: Path
    noise_prediction_dir: Path
    covariance_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 A/B latent-DDPM score validation")
    parser.add_argument("--dataset", default="mnist", choices=sorted(DATASET_SPECS))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--modes", nargs="+", choices=["baseline", "induced"], default=None)
    parser.add_argument("--baseline-run-dir", type=Path, default=None)
    parser.add_argument("--induced-run-dir", type=Path, default=None)
    parser.add_argument("--autoencoder-checkpoint-path", type=Path, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    # Keep sigma sampling and random noise reproducible.
    torch.manual_seed(seed)
    np.random.seed(seed)


def resolve_device(device_arg: str | None) -> torch.device:
    # Use the requested device, otherwise prefer CUDA when it is available
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset_config(slug: str) -> dict[str, Any]:
    config_path = PROJECT_ROOT / "configs" / "dataset" / f"{slug}.yaml"
    cfg = OmegaConf.load(config_path)
    dataset_cfg = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(dataset_cfg, dict):
        raise TypeError(f"Dataset config must be a mapping: {config_path}")
    return dataset_cfg


def _cfg_get(cfg: Any | None, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def default_autoencoder_checkpoint_path(dataset_cfg: dict[str, Any], dataset_slug: str) -> Path:
    encoder_cfg = _cfg_get(dataset_cfg, "encoder", {})
    checkpoint_run = _cfg_get(encoder_cfg, "checkpoint_run", None)
    if checkpoint_run is None:
        architecture = _cfg_get(encoder_cfg, "name", "noise_consistency_small")
        latent_dim = int(_cfg_get(encoder_cfg, "latent_dim", 16))
        checkpoint_run = f"{architecture}_latent{latent_dim}"
    return PROJECT_ROOT / "checkpoints" / dataset_slug / "autoencoders" / str(checkpoint_run) / "autoencoder_checkpoint.pt"


def build_context(args: argparse.Namespace) -> LatentDDPMValidationContext:
    dataset_cfg = load_dataset_config(args.dataset)
    slug = dataset_name(dataset_cfg)
    root = args.output_root or PROJECT_ROOT / "experiments" / slug / "exp_003_latent_ddpm_validation"
    autoencoder_checkpoint_path = args.autoencoder_checkpoint_path or default_autoencoder_checkpoint_path(dataset_cfg, slug)
    return LatentDDPMValidationContext(
        dataset_cfg=dataset_cfg,
        dataset_slug=slug,
        root=root,
        metrics_dir=root / "metrics",
        score_validation_dir=root / "score_validation",
        score_distributions_dir=root / "score_distributions",
        noise_prediction_dir=root / "noise_prediction",
        covariance_dir=root / "covariance",
        report_dir=root / "report",
        autoencoder_checkpoint_path=autoencoder_checkpoint_path,
    )


def ensure_context_dirs(context: LatentDDPMValidationContext) -> None:
    for path in [
        context.root,
        context.metrics_dir,
        context.score_validation_dir,
        context.score_distributions_dir,
        context.noise_prediction_dir,
        context.covariance_dir,
        context.report_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)


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


def validation_run_specs(dataset_cfg: dict[str, Any], dataset_slug: str) -> dict[str, dict[str, Any]]:
    validation_cfg = dataset_cfg.get("latent_ddpm_validation", {})
    runs = validation_cfg.get("runs", {}) if isinstance(validation_cfg, dict) else {}
    if isinstance(runs, dict) and runs:
        return runs

    ddpm_root = PROJECT_ROOT / "checkpoints" / dataset_slug / "ddpm"
    return {
        "baseline": {
            "latent_noise_mode": "baseline",
            "checkpoint_run": f"latent_ddpm_baseline_ae_noise_consistency_{dataset_slug}",
            "default_run_dir": ddpm_root / f"latent_ddpm_baseline_ae_noise_consistency_{dataset_slug}",
        },
        "induced": {
            "latent_noise_mode": "induced",
            "checkpoint_run": f"latent_ddpm_induced_ae_noise_consistency_{dataset_slug}",
            "default_run_dir": ddpm_root / f"latent_ddpm_induced_ae_noise_consistency_{dataset_slug}",
        },
    }


def resolve_run_dir(context: LatentDDPMValidationContext, mode: str, spec: dict[str, Any], args: argparse.Namespace) -> Path:
    override = args.baseline_run_dir if mode == "baseline" else args.induced_run_dir
    if override is not None:
        return override
    if "run_dir" in spec:
        return Path(str(spec["run_dir"]))
    if "default_run_dir" in spec:
        return Path(spec["default_run_dir"])
    checkpoint_run = spec.get("checkpoint_run")
    if checkpoint_run is None:
        checkpoint_run = f"latent_ddpm_{mode}_ae_noise_consistency_{context.dataset_slug}"
    return PROJECT_ROOT / "checkpoints" / context.dataset_slug / "ddpm" / str(checkpoint_run)


def selected_run_specs(
    context: LatentDDPMValidationContext,
    args: argparse.Namespace,
) -> dict[str, tuple[dict[str, Any], Path]]:
    specs = validation_run_specs(context.dataset_cfg, context.dataset_slug)
    modes = args.modes or list(specs.keys())
    selected: dict[str, tuple[dict[str, Any], Path]] = {}
    for mode in modes:
        if mode not in specs:
            raise ValueError(f"Missing latent-DDPM validation run spec for mode={mode}, dataset={context.dataset_slug}")
        spec = specs[mode]
        selected[mode] = (spec, resolve_run_dir(context, mode, spec, args))
    return selected


def latest_checkpoint_path(run_dir: Path) -> Path:
    # Prefer the last epoch checkpoint, fallback to best_model if needed
    epoch_paths = sorted(run_dir.glob("epoch_*.pth"))
    if epoch_paths:
        return epoch_paths[-1]

    best_path = run_dir / "best_model.pth"
    if best_path.exists():
        return best_path

    raise FileNotFoundError(f"Missing DDPM checkpoint in {run_dir}")


def load_training_metrics(run_dir: Path) -> dict[str, float | str]:
    # Read historical train/validation losses saved during DDPM training
    metrics_path = run_dir / "DDPM_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing DDPM metrics: {metrics_path}")

    df = pd.read_csv(metrics_path)
    best_idx = df["val_loss"].idxmin()
    best_row = df.loc[best_idx]
    last_row = df.iloc[-1]
    return {
        "metrics_path": str(metrics_path),
        "best_epoch": float(best_row["epoch"]),
        "best_val_loss": float(best_row["val_loss"]),
        "last_epoch": float(last_row["epoch"]),
        "last_val_loss": float(last_row["val_loss"]),
    }


def resolve_autoencoder_checkpoint_from_ddpm_checkpoint(
    checkpoint: dict[str, Any],
    context: LatentDDPMValidationContext,
) -> Path:
    checkpoint_autoencoder_path = checkpoint.get("autoencoder_checkpoint_path")
    if checkpoint_autoencoder_path:
        candidate = Path(str(checkpoint_autoencoder_path))
        if candidate.exists():
            return candidate
    return context.autoencoder_checkpoint_path


def load_stage2_model(
    mode: str,
    checkpoint_path: Path,
    context: LatentDDPMValidationContext,
    device: torch.device,
):
    # Rebuild the DDPM architecture and load only the trained UNet weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_mode = checkpoint.get("latent_noise_mode")
    if checkpoint_mode is not None and checkpoint_mode != mode:
        # Do not accidentally evaluate an induced checkpoint as baseline or vice versa
        raise ValueError(
            f"Checkpoint mode mismatch for {checkpoint_path}: "
            f"expected {mode}, got {checkpoint_mode}"
        )

    ddpm_params = checkpoint.get("DDPM_params", {})
    base_dim = int(ddpm_params.get("base_dim", 16))
    deep = int(ddpm_params.get("deep", 3))
    autoencoder_checkpoint_path = resolve_autoencoder_checkpoint_from_ddpm_checkpoint(checkpoint, context)

    model = build_DDPM_model(
        base_dim=base_dim,
        deep=deep,
        device=str(device),
        latent_noise_mode=mode,
        autoencoder_kind=checkpoint.get("autoencoder_kind", "noise_consistency"),
        autoencoder_checkpoint_path=autoencoder_checkpoint_path,
        dataset_cfg=context.dataset_cfg,
    )
    model.model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, autoencoder_checkpoint_path


def sigma_to_t(model, sigma_flat: torch.Tensor) -> torch.Tensor:
    # Map external sigma to the closest DDPM noise level sqrt(1 - alpha_bar_t)
    schedule = torch.sqrt(1.0 - model.alphas_cumprod)
    sigma = sigma_flat.detach().clamp(float(schedule[0]), float(schedule[-1]))
    idx = torch.searchsorted(schedule, sigma)
    idx = idx.clamp(0, schedule.shape[0] - 1)
    prev_idx = (idx - 1).clamp(0, schedule.shape[0] - 1)

    cur_err = (schedule[idx] - sigma).abs()
    prev_err = (schedule[prev_idx] - sigma).abs()
    return torch.where(prev_err < cur_err, prev_idx, idx).long()


def save_scatter(df: pd.DataFrame, x_col: str, y_col: str, output_path: Path, title: str) -> None:
    # Save a scatter plot for quick visual score diagnostics
    plt.figure(figsize=(7, 5))
    plt.scatter(df[x_col], df[y_col], s=8, alpha=0.25)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_hist(values: np.ndarray, output_path: Path, title: str, xlabel: str) -> None:
    # Save a distribution plot for scores or target-noise norms
    plt.figure(figsize=(7, 5))
    plt.hist(values, bins=50, alpha=0.85)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.title(title)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_binned_curve(bin_centers: np.ndarray, bin_means: np.ndarray, output_path: Path, title: str) -> None:
    # Plot the average score inside sigma bins.
    plt.figure(figsize=(7, 5))
    plt.plot(bin_centers, bin_means, marker="o")
    plt.xlabel("sigma bin center")
    plt.ylabel("mean(score)")
    plt.title(title)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_eigenvalues(eigvals: np.ndarray, output_path: Path, title: str) -> None:
    # Visualize the covariance spectrum of the target latent noise
    plt.figure(figsize=(7, 5))
    plt.plot(np.arange(1, len(eigvals) + 1), eigvals, marker="o")
    plt.xlabel("eigenvalue index")
    plt.ylabel("eigenvalue")
    plt.title(title)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_covariance_heatmap(cov: np.ndarray, output_path: Path, title: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(cov, cmap="viridis")
    plt.colorbar(label="covariance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = [str(col) for col in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    widths = [
        max([len(columns[idx]), *(len(row[idx]) for row in rows)] or [len(columns[idx])])
        for idx in range(len(columns))
    ]
    header = "| " + " | ".join(columns[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    body = ["| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def score_bin_stats(sigma: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Group score values by sigma intervals
    bin_edges = np.linspace(SIGMA_MIN, SIGMA_MAX, NUM_BINS + 1)
    bin_indices = np.clip(np.digitize(sigma, bin_edges) - 1, 0, NUM_BINS - 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_means = np.array(
        [float(score[bin_indices == idx].mean()) if np.any(bin_indices == idx) else float("nan") for idx in range(NUM_BINS)],
        dtype=np.float32,
    )
    return bin_edges, bin_centers, bin_means


def covariance_diagnostics(target_noise: np.ndarray) -> dict[str, Any]:
    # Measure how isotropic or anisotropic the target latent noise is
    cov = np.cov(target_noise, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    min_eig = float(np.min(eigvals))
    max_eig = float(np.max(eigvals))
    return {
        "mean_per_coordinate": target_noise.mean(axis=0).tolist(),
        "variance_per_coordinate": target_noise.var(axis=0).tolist(),
        "covariance_matrix": cov.tolist(),
        "covariance_eigenvalues": eigvals.tolist(),
        "min_eigenvalue": min_eig,
        "max_eigenvalue": max_eig,
        "anisotropy": float(max_eig / max(min_eig, 1e-12)),
    }


def mode_output_dirs(context: LatentDDPMValidationContext, mode: str) -> ModeOutputDirs:
    dirs = ModeOutputDirs(
        metrics_path=context.metrics_dir / f"{mode}_metrics.json",
        score_validation_dir=context.score_validation_dir / mode,
        score_distributions_dir=context.score_distributions_dir / mode,
        noise_prediction_dir=context.noise_prediction_dir / mode,
        covariance_dir=context.covariance_dir / mode,
    )
    for path in [
        dirs.score_validation_dir,
        dirs.score_distributions_dir,
        dirs.noise_prediction_dir,
        dirs.covariance_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    return dirs


@torch.no_grad()
def evaluate_mode(
    mode: str,
    model,
    loader: DataLoader,
    training_metrics: dict[str, float | str],
    outputs: ModeOutputDirs,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray]:
    # Validate one DDPM mode: baseline or induced.

    sigma_values: list[np.ndarray] = []
    score_values: list[np.ndarray] = []
    target_values: list[np.ndarray] = []
    prediction_loss_sum = 0.0
    prediction_loss_count = 0

    for x, _ in loader:
        # Sample an external degradation level for every image.
        x = x.to(device)
        batch_size = x.shape[0]
        sigma = torch.empty(batch_size, device=device).uniform_(SIGMA_MIN, SIGMA_MAX)
        t = sigma_to_t(model, sigma)

        # Reuse the same noisy-latent construction that was used in training
        z_0 = model._encode_to_latent(x)
        z_t, target_noise = model._make_noisy_latent_batch(x, z_0, t)
        eps_pred = model.model(z_t, t)
        # The scalar score is the squared norm of predicted noise
        score = eps_pred.flatten(start_dim=1).pow(2).sum(dim=1)
        prediction_error = (eps_pred - target_noise).pow(2)
        prediction_loss_sum += float(prediction_error.sum().item())
        prediction_loss_count += int(prediction_error.numel())

        sigma_values.append(sigma.cpu().numpy().astype(np.float32))
        score_values.append(score.cpu().numpy().astype(np.float32))
        target_values.append(target_noise.flatten(start_dim=1).cpu().numpy().astype(np.float32))

    sigma_np = np.concatenate(sigma_values, axis=0)
    score_np = np.concatenate(score_values, axis=0)
    target_np = np.concatenate(target_values, axis=0)

    target_noise_mse_vs_zero = float(np.mean(target_np**2))
    target_noise_std = float(np.std(target_np))
    # This recomputes validation loss on the current validation sample
    recomputed_val_loss = prediction_loss_sum / max(prediction_loss_count, 1)
    best_val_loss = float(training_metrics["best_val_loss"])
    last_val_loss = float(training_metrics["last_val_loss"])

    pearson = float(pd.Series(score_np).corr(pd.Series(sigma_np), method="pearson"))
    spearman = float(pd.Series(score_np).corr(pd.Series(sigma_np), method="spearman"))
    bin_edges, bin_centers, bin_means = score_bin_stats(sigma_np, score_np)

    df = pd.DataFrame(
        {
            "sigma": sigma_np,
            "score": score_np,
            "target_noise_norm": np.linalg.norm(target_np, axis=1).astype(np.float32),
        }
    )
    df.to_csv(outputs.score_validation_dir / "score_samples.csv", index=False)

    # Save per-mode plots and metrics.
    save_scatter(df, "sigma", "score", outputs.score_validation_dir / "score_vs_sigma.png", f"{mode}: score vs sigma")
    save_hist(score_np, outputs.score_distributions_dir / "score_distribution.png", f"{mode}: score distribution", "score")
    save_hist(
        df["target_noise_norm"].to_numpy(),
        outputs.noise_prediction_dir / "target_noise_norm_distribution.png",
        f"{mode}: target noise norm",
        "||target_noise||",
    )
    save_binned_curve(
        bin_centers,
        bin_means,
        outputs.score_validation_dir / "sigma_bins_mean_score.png",
        f"{mode}: mean score by sigma bin",
    )

    cov_metrics = covariance_diagnostics(target_np)
    cov_matrix = np.asarray(cov_metrics["covariance_matrix"])
    eigvals = np.asarray(cov_metrics["covariance_eigenvalues"])
    np.savetxt(outputs.covariance_dir / "covariance_matrix.csv", cov_matrix, delimiter=",")
    save_covariance_heatmap(cov_matrix, outputs.covariance_dir / "covariance_matrix.png", f"{mode}: target covariance")
    save_eigenvalues(eigvals, outputs.covariance_dir / "covariance_eigenvalues.png", f"{mode}: target covariance spectrum")

    metrics = {
        "mode": mode,
        "raw_pearson_score_sigma": pearson,
        "raw_spearman_score_sigma": spearman,
        "pearson_score_sigma": pearson,
        "spearman_score_sigma": spearman,
        "target_noise_std": target_noise_std,
        "target_noise_mse_vs_zero": target_noise_mse_vs_zero,
        "recomputed_val_loss": recomputed_val_loss,
        "normalized_recomputed_val_loss": recomputed_val_loss / max(target_noise_mse_vs_zero, 1e-12),
        "normalized_best_val_loss": best_val_loss / max(target_noise_mse_vs_zero, 1e-12),
        "normalized_last_val_loss": last_val_loss / max(target_noise_mse_vs_zero, 1e-12),
        "training_metrics": training_metrics,
        "sigma_bin_edges": bin_edges.tolist(),
        "sigma_bin_centers": bin_centers.tolist(),
        "mean_score_per_bin": bin_means.tolist(),
        "covariance": cov_metrics,
        "num_samples": int(len(df)),
    }
    outputs.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics, target_np


def histogram_prob(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    # Convert a histogram into probabilities for divergence metrics
    hist, _ = np.histogram(values, bins=bin_edges)
    prob = hist.astype(np.float64)
    prob /= max(prob.sum(), 1.0)
    return prob


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    # KL(p || q) with a small epsilon to avoid log(0).
    p_safe = p + eps
    q_safe = q + eps
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    # Symmetric divergence based on two KL terms.
    m = 0.5 * (p + q)
    return float(0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m))


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    # Simple 1D Wasserstein distance between two empirical distributions
    a_sorted = np.sort(a.reshape(-1))
    b_sorted = np.sort(b.reshape(-1))
    n = min(len(a_sorted), len(b_sorted))
    return float(np.mean(np.abs(a_sorted[:n] - b_sorted[:n])))


def save_comparison_hist(a: np.ndarray, b: np.ndarray, output_path: Path, title: str, xlabel: str) -> None:
    # Overlay baseline and induced distributions on one plot
    plt.figure(figsize=(7, 5))
    plt.hist(a, bins=50, alpha=0.55, label="baseline", density=True)
    plt.hist(b, bins=50, alpha=0.55, label="induced", density=True)
    plt.xlabel(xlabel)
    plt.ylabel("density")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_comparison(
    mode_metrics: dict[str, dict[str, Any]],
    target_noise: dict[str, np.ndarray],
    context: LatentDDPMValidationContext,
) -> dict[str, Any] | None:
    # Compare the target-noise distributions of baseline and induced DDPMs
    if not {"baseline", "induced"}.issubset(mode_metrics.keys()):
        return None

    comparison_dir = context.score_distributions_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    baseline_norm = np.linalg.norm(target_noise["baseline"], axis=1)
    induced_norm = np.linalg.norm(target_noise["induced"], axis=1)
    min_value = float(min(baseline_norm.min(), induced_norm.min()))
    max_value = float(max(baseline_norm.max(), induced_norm.max()))
    bin_edges = np.linspace(min_value, max_value, 51)
    p = histogram_prob(baseline_norm, bin_edges)
    q = histogram_prob(induced_norm, bin_edges)

    baseline_cov = np.cov(target_noise["baseline"], rowvar=False)
    induced_cov = np.cov(target_noise["induced"], rowvar=False)
    cov_diff = baseline_cov - induced_cov

    # Distribution-level diagnostics for the two target-noise regimes
    comparison = {
        "target_noise_norm_kl_baseline_to_induced": kl_divergence(p, q),
        "target_noise_norm_kl_induced_to_baseline": kl_divergence(q, p),
        "target_noise_norm_js_divergence": js_divergence(p, q),
        "target_noise_norm_wasserstein": wasserstein_1d(baseline_norm, induced_norm),
        "covariance_frobenius_distance": float(np.linalg.norm(cov_diff, ord="fro")),
        "baseline_target_noise_mse_vs_zero": mode_metrics["baseline"]["target_noise_mse_vs_zero"],
        "induced_target_noise_mse_vs_zero": mode_metrics["induced"]["target_noise_mse_vs_zero"],
        "baseline_normalized_best_val_loss": mode_metrics["baseline"]["normalized_best_val_loss"],
        "induced_normalized_best_val_loss": mode_metrics["induced"]["normalized_best_val_loss"],
    }
    (context.metrics_dir / "distribution_diagnostics.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    pd.DataFrame([comparison]).to_csv(context.metrics_dir / "comparison_metrics.csv", index=False)

    save_comparison_hist(
        baseline_norm,
        induced_norm,
        comparison_dir / "target_noise_norm_comparison.png",
        "Target noise norm distribution mismatch",
        "||target_noise||",
    )
    return comparison


def write_summary(
    context: LatentDDPMValidationContext,
    mode_metrics: dict[str, dict[str, Any]],
    comparison: dict[str, Any] | None,
    run_metadata: dict[str, dict[str, str]],
) -> None:
    rows = []
    for mode, metrics in mode_metrics.items():
        row = {
            "dataset": context.dataset_slug,
            "mode": mode,
            "raw_pearson_score_sigma": metrics["raw_pearson_score_sigma"],
            "raw_spearman_score_sigma": metrics["raw_spearman_score_sigma"],
            "target_noise_std": metrics["target_noise_std"],
            "target_noise_mse_vs_zero": metrics["target_noise_mse_vs_zero"],
            "recomputed_val_loss": metrics["recomputed_val_loss"],
            "normalized_recomputed_val_loss": metrics["normalized_recomputed_val_loss"],
            "best_val_loss": metrics["training_metrics"]["best_val_loss"],
            "last_val_loss": metrics["training_metrics"]["last_val_loss"],
            "normalized_best_val_loss": metrics["normalized_best_val_loss"],
            "normalized_last_val_loss": metrics["normalized_last_val_loss"],
            "covariance_min_eigenvalue": metrics["covariance"]["min_eigenvalue"],
            "covariance_max_eigenvalue": metrics["covariance"]["max_eigenvalue"],
            "covariance_anisotropy": metrics["covariance"]["anisotropy"],
            **{
                f"mean_score_sigma_bin_{idx + 1:02d}": value
                for idx, value in enumerate(metrics["mean_score_per_bin"])
            },
        }
        if comparison is not None:
            row.update(comparison)
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_csv = context.metrics_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    summary_markdown = dataframe_to_markdown(summary_df)
    (context.metrics_dir / "summary.md").write_text(summary_markdown + "\n", encoding="utf-8")

    summary = {
        "dataset": context.dataset_slug,
        "root": str(context.root),
        "autoencoder_checkpoint_path": str(context.autoencoder_checkpoint_path),
        "modes": list(mode_metrics.keys()),
        "runs": run_metadata,
        "summary_csv": str(summary_csv),
        "comparison_available": comparison is not None,
    }
    (context.root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        f"# Latent-DDPM Validation Summary: {context.dataset_slug}",
        "",
        "## Runs",
    ]
    for mode, metadata in run_metadata.items():
        report_lines.extend(
            [
                f"- {mode}: {metadata['run_dir']}",
                f"  checkpoint: {metadata['checkpoint_path']}",
                f"  autoencoder: {metadata['autoencoder_checkpoint_path']}",
            ]
        )
    report_lines.extend(["", "## Metrics", summary_markdown])
    if comparison is None:
        report_lines.extend(["", "Comparison diagnostics were not written because both baseline and induced modes were not selected."])
    (context.report_dir / "latent_ddpm_validation_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> None:
    # Run selected DDPM modes and write per-mode plus comparison outputs
    args = parse_args()
    set_seed(args.seed)
    context = build_context(args)
    ensure_context_dirs(context)
    device = resolve_device(args.device)
    loader = build_loader(context.dataset_cfg, num_samples=args.num_samples, batch_size=args.batch_size)
    run_specs = selected_run_specs(context, args)

    mode_metrics: dict[str, dict[str, Any]] = {}
    target_noise: dict[str, np.ndarray] = {}
    run_metadata: dict[str, dict[str, str]] = {}

    for mode, (spec, run_dir) in run_specs.items():
        # Load the trained checkpoint and evaluate this mode independently
        checkpoint_path = latest_checkpoint_path(run_dir)
        training_metrics = load_training_metrics(run_dir)
        model, autoencoder_checkpoint_path = load_stage2_model(
            mode=str(spec.get("latent_noise_mode", mode)),
            checkpoint_path=checkpoint_path,
            context=context,
            device=device,
        )
        outputs = mode_output_dirs(context, mode)
        metrics, target_np = evaluate_mode(
            mode=mode,
            model=model,
            loader=loader,
            training_metrics=training_metrics,
            outputs=outputs,
            device=device,
        )
        metrics["dataset"] = context.dataset_slug
        metrics["run_dir"] = str(run_dir)
        metrics["checkpoint_path"] = str(checkpoint_path)
        metrics["autoencoder_checkpoint_path"] = str(autoencoder_checkpoint_path)
        outputs.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        mode_metrics[mode] = metrics
        target_noise[mode] = target_np
        run_metadata[mode] = {
            "run_dir": str(run_dir),
            "checkpoint_path": str(checkpoint_path),
            "autoencoder_checkpoint_path": str(autoencoder_checkpoint_path),
        }

    comparison = write_comparison(mode_metrics, target_noise, context)
    write_summary(context, mode_metrics, comparison, run_metadata)
    print(f"latent_ddpm_score_validation_output={context.root}")


if __name__ == "__main__":
    main()
