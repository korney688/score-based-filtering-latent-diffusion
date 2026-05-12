from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.DDPM_model import build_DDPM_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "exp_003_aligned_latent_ddpm"
DDPM_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ddpm"
# Stage 2 uses one fixed encoder selected after encoder validation
AUTOENCODER_CHECKPOINT_PATH = (
    PROJECT_ROOT / "models" / "autoencoders" / "ae_noise_consistency_mnist" / "autoencoder_checkpoint.pt"
)

BATCH_SIZE = 128
NUM_SAMPLES = 2000
SEED = 42
SIGMA_MIN = 0.1
SIGMA_MAX = 0.8
NUM_BINS = 10

RUN_SPECS = {
    # Compare two DDPMs trained with the same encoder but different latent noise modes
    "baseline": {
        "latent_noise_mode": "baseline",
        "default_run_dir": DDPM_OUTPUT_ROOT / "latent_ddpm_baseline_ae_noise_consistency_mnist",
    },
    "induced": {
        "latent_noise_mode": "induced",
        "default_run_dir": DDPM_OUTPUT_ROOT / "latent_ddpm_induced_ae_noise_consistency_mnist",
    },
}


def parse_args() -> argparse.Namespace:
    # Allow rerunning the validation with another output folder or checkpoint folders
    parser = argparse.ArgumentParser(description="Stage 2 A/B latent-DDPM score validation")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--baseline-run-dir",
        type=Path,
        default=RUN_SPECS["baseline"]["default_run_dir"],
    )
    parser.add_argument(
        "--induced-run-dir",
        type=Path,
        default=RUN_SPECS["induced"]["default_run_dir"],
    )
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


def build_loader(num_samples: int, batch_size: int) -> DataLoader:
    # Use MNIST test images for validation; labels are ignored later
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


def load_stage2_model(mode: str, checkpoint_path: Path, device: torch.device):
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

    model = build_DDPM_model(
        base_dim=base_dim,
        deep=deep,
        device=str(device),
        latent_noise_mode=mode,
        autoencoder_kind="noise_consistency",
        autoencoder_checkpoint_path=AUTOENCODER_CHECKPOINT_PATH,
    )
    model.model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


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


@torch.no_grad()
def evaluate_mode(
    mode: str,
    model,
    loader: DataLoader,
    training_metrics: dict[str, float | str],
    output_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray]:
    # Validate one DDPM mode: baseline or induced.
    output_dir.mkdir(parents=True, exist_ok=True)

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
    df.to_csv(output_dir / "score_samples.csv", index=False)

    # Save per-mode plots and metrics.
    save_scatter(df, "sigma", "score", output_dir / "score_vs_sigma.png", f"{mode}: score vs sigma")
    save_hist(score_np, output_dir / "score_distribution.png", f"{mode}: score distribution", "score")
    save_hist(df["target_noise_norm"].to_numpy(), output_dir / "target_noise_norm_distribution.png", f"{mode}: target noise norm", "||target_noise||")
    save_binned_curve(bin_centers, bin_means, output_dir / "sigma_bins_mean_score.png", f"{mode}: mean score by sigma bin")

    cov_metrics = covariance_diagnostics(target_np)
    save_eigenvalues(np.asarray(cov_metrics["covariance_eigenvalues"]), output_dir / "covariance_eigenvalues.png", f"{mode}: target covariance spectrum")

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
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
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
    output_dir: Path,
) -> None:
    # Compare the target-noise distributions of baseline and induced DDPMs
    output_dir.mkdir(parents=True, exist_ok=True)

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
    (output_dir / "distribution_diagnostics.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    pd.DataFrame([comparison]).to_csv(output_dir / "comparison_metrics.csv", index=False)

    summary_df = pd.DataFrame(
        # One compact comparison table with per-mode metrics and shared diagnostics
        [
            {
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
                "target_noise_norm_kl_baseline_to_induced": comparison["target_noise_norm_kl_baseline_to_induced"],
                "target_noise_norm_kl_induced_to_baseline": comparison["target_noise_norm_kl_induced_to_baseline"],
                "target_noise_norm_js_divergence": comparison["target_noise_norm_js_divergence"],
                "target_noise_norm_wasserstein": comparison["target_noise_norm_wasserstein"],
                "covariance_frobenius_distance": comparison["covariance_frobenius_distance"],
                **{
                    f"mean_score_sigma_bin_{idx + 1:02d}": value
                    for idx, value in enumerate(metrics["mean_score_per_bin"])
                },
            }
            for mode, metrics in mode_metrics.items()
        ]
    )
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    save_comparison_hist(
        baseline_norm,
        induced_norm,
        output_dir / "target_noise_norm_comparison.png",
        "Target noise norm distribution mismatch",
        "||target_noise||",
    )


def main() -> None:
    # Run both DDPM modes and write per-mode plus comparison outputs
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    loader = build_loader(num_samples=args.num_samples, batch_size=args.batch_size)
    run_dirs = {
        "baseline": args.baseline_run_dir,
        "induced": args.induced_run_dir,
    }

    mode_metrics: dict[str, dict[str, Any]] = {}
    target_noise: dict[str, np.ndarray] = {}

    for mode, spec in RUN_SPECS.items():
        # Load the trained checkpoint and evaluate this mode independently
        run_dir = run_dirs[mode]
        checkpoint_path = latest_checkpoint_path(run_dir)
        training_metrics = load_training_metrics(run_dir)
        model = load_stage2_model(mode=spec["latent_noise_mode"], checkpoint_path=checkpoint_path, device=device)
        metrics, target_np = evaluate_mode(
            mode=mode,
            model=model,
            loader=loader,
            training_metrics=training_metrics,
            output_dir=args.output_root / mode,
            device=device,
        )
        metrics["checkpoint_path"] = str(checkpoint_path)
        (args.output_root / mode / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        mode_metrics[mode] = metrics
        target_noise[mode] = target_np

    write_comparison(mode_metrics, target_noise, args.output_root / "comparison")
    print(f"latent_ddpm_score_validation_output={args.output_root}")


if __name__ == "__main__":
    main()
