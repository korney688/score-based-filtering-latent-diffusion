from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.DDPM_model import build_DDPM_model


DEFAULT_T_VALUES = (10, 30, 50)
FOREGROUND_THRESHOLD = 0.1
ENTROPY_BINS = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extended score analysis for MNIST noisy dataset and trained DDPM.",
    )
    parser.add_argument(
        "--ddpm-path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "models" / "ddpm_mnist_5000.pt",
    )
    parser.add_argument(
        "--clean-path",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "exp_001" / "data" / "dataset_clean_mnist" / "dataset_clean_mnist.h5",
    )
    parser.add_argument(
        "--noisy-path",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "exp_001" / "data" / "dataset_noisy_var" / "dataset_noisy_var.h5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "score_analysis",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--t-values",
        type=int,
        nargs="+",
        default=list(DEFAULT_T_VALUES),
    )
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def load_h5_data(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5_file:
        return {key: h5_file[key][:] for key in h5_file.keys()}


def normalize_for_ddpm(batch: np.ndarray) -> torch.Tensor:
    batch = batch.astype(np.float32)
    batch = (batch - 0.5) / 0.5
    batch = np.expand_dims(batch, axis=1)
    return torch.from_numpy(batch)


def clean_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in output_dir.iterdir():
        if child.is_file():
            child.unlink()


@torch.no_grad()
def compute_scores_and_latents(
    noisy_images: np.ndarray,
    ddpm_model,
    device: torch.device,
    batch_size: int,
    t_values: list[int],
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    latent_norm_batches: list[np.ndarray] = []
    score_per_t_batches: dict[int, list[np.ndarray]] = {t_value: [] for t_value in t_values}

    for start in range(0, len(noisy_images), batch_size):
        end = min(start + batch_size, len(noisy_images))
        batch_np = noisy_images[start:end]
        x = normalize_for_ddpm(batch_np).to(device=device, dtype=torch.float32)

        z = ddpm_model.ae.encode(x)
        z_view = z.view(z.shape[0], z.shape[1], 1, 1)
        latent_norm_batches.append(z.norm(dim=1).cpu().numpy().astype(np.float32))

        for t_value in t_values:
            t = torch.full((x.shape[0],), t_value, device=device, dtype=torch.long)
            noise = torch.randn_like(z_view)
            z_t = ddpm_model.q_sample(z_view, t, noise)
            noise_pred = ddpm_model.model(z_t, t)
            score_t = noise_pred.pow(2).flatten(1).mean(dim=1)
            score_per_t_batches[t_value].append(score_t.cpu().numpy().astype(np.float32))

    latent_norm = np.concatenate(latent_norm_batches, axis=0)
    score_per_t = {
        t_value: np.concatenate(score_per_t_batches[t_value], axis=0)
        for t_value in t_values
    }
    score_matrix = np.stack([score_per_t[t_value] for t_value in t_values], axis=1)
    score = score_matrix.mean(axis=1).astype(np.float32)
    return score, latent_norm, score_per_t


def compute_entropy(images: np.ndarray, bins: int = ENTROPY_BINS) -> np.ndarray:
    entropy = np.zeros(images.shape[0], dtype=np.float32)
    for idx, image in enumerate(images):
        hist, _ = np.histogram(image, bins=bins, range=(0.0, 1.0))
        prob = hist.astype(np.float64)
        prob = prob / max(prob.sum(), 1.0)
        prob = prob[prob > 0]
        entropy[idx] = float(-(prob * np.log(prob + 1e-12)).sum())
    return entropy


def compute_gradient_energy(images: np.ndarray) -> np.ndarray:
    dx = images[:, :, 1:] - images[:, :, :-1]
    dy = images[:, 1:, :] - images[:, :-1, :]
    dx_energy = np.mean(dx ** 2, axis=(1, 2), dtype=np.float32)
    dy_energy = np.mean(dy ** 2, axis=(1, 2), dtype=np.float32)
    return (dx_energy + dy_energy).astype(np.float32)


def build_features_dataframe(
    index: np.ndarray,
    label: np.ndarray,
    sigma: np.ndarray,
    score: np.ndarray,
    latent_norm: np.ndarray,
    score_per_t: dict[int, np.ndarray],
    noisy_images: np.ndarray,
) -> pd.DataFrame:
    ordered_t = sorted(score_per_t.keys())
    score_matrix = np.stack([score_per_t[t_value] for t_value in ordered_t], axis=1)
    score_mean = score_matrix.mean(axis=1)
    score_std = score_matrix.std(axis=1)
    score_cv = score_std / (score_mean + 1e-8)

    mean_intensity = noisy_images.mean(axis=(1, 2), dtype=np.float32)
    std_intensity = noisy_images.std(axis=(1, 2), dtype=np.float32)
    foreground_mass = (noisy_images > FOREGROUND_THRESHOLD).sum(axis=(1, 2)).astype(np.float32)
    entropy = compute_entropy(noisy_images)
    gradient_energy = compute_gradient_energy(noisy_images)

    data = {
        "index": index.astype(np.int64),
        "label": label.astype(np.int64),
        "sigma": sigma.astype(np.float32),
        "score": score.astype(np.float32),
        "latent_norm": latent_norm.astype(np.float32),
        "score_over_latent": (score / (latent_norm + 1e-8)).astype(np.float32),
        "score_over_latent2": (score / (latent_norm ** 2 + 1e-8)).astype(np.float32),
    }

    for t_value in ordered_t:
        data[f"score_t{t_value}"] = score_per_t[t_value].astype(np.float32)

    data["score_mean"] = score_mean.astype(np.float32)
    data["score_std"] = score_std.astype(np.float32)
    data["score_cv"] = score_cv.astype(np.float32)
    data["mean_intensity"] = mean_intensity.astype(np.float32)
    data["std_intensity"] = std_intensity.astype(np.float32)
    data["foreground_mass"] = foreground_mass.astype(np.float32)
    data["entropy"] = entropy.astype(np.float32)
    data["gradient_energy"] = gradient_energy.astype(np.float32)

    return pd.DataFrame(data)


def compute_feature_correlations(df: pd.DataFrame) -> pd.DataFrame:
    excluded = {"index", "label", "sigma"}
    rows: list[dict[str, float | str]] = []
    for column in df.columns:
        if column in excluded:
            continue
        rows.append(
            {
                "feature": column,
                "pearson_vs_sigma": float(df[column].corr(df["sigma"], method="pearson")),
                "spearman_vs_sigma": float(df[column].corr(df["sigma"], method="spearman")),
                "abs_pearson_vs_sigma": float(abs(df[column].corr(df["sigma"], method="pearson"))),
            }
        )
    corr_df = pd.DataFrame(rows).sort_values("abs_pearson_vs_sigma", ascending=False).reset_index(drop=True)
    return corr_df


def save_scatter(df: pd.DataFrame, x_col: str, y_col: str, output_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(df[x_col], df[y_col], s=7, alpha=0.25)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} vs {x_col}")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_boxplot(df: pd.DataFrame, value_column: str, output_path: Path) -> None:
    data = [df.loc[df["label"] == label, value_column].to_numpy() for label in range(10)]
    plt.figure(figsize=(9, 5))
    plt.boxplot(data, tick_labels=[str(label) for label in range(10)], showfliers=False)
    plt.xlabel("label")
    plt.ylabel(value_column)
    plt.title(f"{value_column} by label")
    plt.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_summary(
    output_path: Path,
    feature_corr_df: pd.DataFrame,
    df: pd.DataFrame,
) -> None:
    top5 = feature_corr_df.head(5)
    compare_features = ["score", "score_over_latent", "score_cv"]
    compare_df = feature_corr_df[feature_corr_df["feature"].isin(compare_features)].copy()
    compare_df = compare_df.set_index("feature").loc[compare_features].reset_index()

    lines = [
        "top5_features_by_abs_pearson_vs_sigma:",
        top5.to_string(index=False, float_format=lambda value: f"{value:.6f}"),
        "",
        "comparison_score_variants:",
        compare_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"),
        "",
        "all_feature_correlations:",
        feature_corr_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"),
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    ensure_file_exists(args.ddpm_path, "DDPM checkpoint")
    ensure_file_exists(args.clean_path, "clean dataset")
    ensure_file_exists(args.noisy_path, "noisy dataset")

    clean_output_dir(args.output_dir)

    clean_h5 = load_h5_data(args.clean_path)
    noisy_h5 = load_h5_data(args.noisy_path)

    clean_images = clean_h5["dataset"].astype(np.float32)
    noisy_images = noisy_h5["dataset"].astype(np.float32)
    labels = noisy_h5["label"].astype(np.int64)
    indices = noisy_h5["index"].astype(np.int64)
    sigma = noisy_h5["noise_std"].astype(np.float32)

    if clean_images.shape != noisy_images.shape:
        raise ValueError(f"Clean/noisy shape mismatch: {clean_images.shape} vs {noisy_images.shape}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ddpm_checkpoint = torch.load(args.ddpm_path, map_location=device)
    ddpm_params = ddpm_checkpoint.get("DDPM_params", {})
    base_dim = int(ddpm_params.get("base_dim", 16))
    deep = int(ddpm_params.get("deep", 3))

    ddpm_model = build_DDPM_model(base_dim=base_dim, deep=deep, device=str(device))
    ddpm_model.model.load_state_dict(ddpm_checkpoint["model_state_dict"])
    ddpm_model.eval()

    score, latent_norm, score_per_t = compute_scores_and_latents(
        noisy_images=noisy_images,
        ddpm_model=ddpm_model,
        device=device,
        batch_size=args.batch_size,
        t_values=args.t_values,
    )

    df = build_features_dataframe(
        index=indices,
        label=labels,
        sigma=sigma,
        score=score,
        latent_norm=latent_norm,
        score_per_t=score_per_t,
        noisy_images=noisy_images,
    )
    score_stats_path = args.output_dir / "score_stats.csv"
    df.to_csv(score_stats_path, index=False)

    feature_corr_df = compute_feature_correlations(df)
    feature_corr_path = args.output_dir / "feature_correlations.csv"
    feature_corr_df.to_csv(feature_corr_path, index=False)

    save_scatter(df, "sigma", "score", args.output_dir / "score_vs_sigma.png")
    save_scatter(df, "sigma", "score_over_latent", args.output_dir / "score_over_latent_vs_sigma.png")
    save_scatter(df, "sigma", "score_cv", args.output_dir / "score_cv_vs_sigma.png")
    save_scatter(df, "sigma", "latent_norm", args.output_dir / "latent_norm_vs_sigma.png")
    save_boxplot(df, "score", args.output_dir / "score_by_label_boxplot.png")
    save_boxplot(df, "score_over_latent", args.output_dir / "score_over_latent_by_label_boxplot.png")

    with h5py.File(args.output_dir / "score_samples.h5", "w") as h5_file:
        h5_file.create_dataset("x_clean", data=clean_images.astype(np.float32))
        h5_file.create_dataset("x_noisy", data=noisy_images.astype(np.float32))
        h5_file.create_dataset("label", data=labels.astype(np.int64))
        h5_file.create_dataset("index", data=indices.astype(np.int64))
        h5_file.create_dataset("sigma", data=sigma.astype(np.float32))
        h5_file.create_dataset("score", data=df["score"].to_numpy(dtype=np.float32))
        h5_file.create_dataset("latent_norm", data=df["latent_norm"].to_numpy(dtype=np.float32))

    write_summary(
        output_path=args.output_dir / "correlations.txt",
        feature_corr_df=feature_corr_df,
        df=df,
    )

    top5 = feature_corr_df.head(5)
    compare = feature_corr_df[feature_corr_df["feature"].isin(["score", "score_over_latent", "score_cv"])]
    compare = compare.set_index("feature").loc[["score", "score_over_latent", "score_cv"]].reset_index()

    print("Top-5 features by |Pearson(feature, sigma)|:")
    print(top5.to_string(index=False))
    print()
    print("Comparison: score vs score_over_latent vs score_cv")
    print(compare.to_string(index=False))
    print()
    print(f"Saved score stats: {score_stats_path}")
    print(f"Saved feature correlations: {feature_corr_path}")
    print(f"Saved outputs to: {args.output_dir}")




# Protocol mode: baseline-check.
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

baseline_check_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(baseline_check_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(baseline_check_PROJECT_ROOT))

from src.DDPM_model import build_DDPM_model


baseline_check_DEFAULT_T_VALUES = (10, 30, 50)


def baseline_check_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity-check for DDPM score definitions.")
    parser.add_argument(
        "--ddpm-path",
        type=Path,
        default=baseline_check_PROJECT_ROOT / "outputs" / "models" / "ddpm_mnist_5000.pt",
    )
    parser.add_argument(
        "--clean-path",
        type=Path,
        default=baseline_check_PROJECT_ROOT / "experiments" / "exp_001" / "data" / "dataset_clean_mnist" / "dataset_clean_mnist.h5",
    )
    parser.add_argument(
        "--noisy-path",
        type=Path,
        default=baseline_check_PROJECT_ROOT / "experiments" / "exp_001" / "data" / "dataset_noisy_var" / "dataset_noisy_var.h5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=baseline_check_PROJECT_ROOT / "outputs" / "score_analysis",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--matched-t-max", type=int, default=2000)
    parser.add_argument(
        "--t-values",
        type=int,
        nargs="+",
        default=list(baseline_check_DEFAULT_T_VALUES),
    )
    return parser.baseline_check_parse_args()


def baseline_check_resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def baseline_check_ensure_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def baseline_check_load_h5_data(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5_file:
        return {key: h5_file[key][:] for key in h5_file.keys()}


def baseline_check_normalize_for_ddpm(batch: np.ndarray) -> torch.Tensor:
    batch = batch.astype(np.float32)
    batch = (batch - 0.5) / 0.5
    batch = np.expand_dims(batch, axis=1)
    return torch.from_numpy(batch)


def baseline_check_corr(a: np.ndarray, b: np.ndarray, method: str = "pearson") -> float:
    series_a = pd.Series(a)
    series_b = pd.Series(b)
    return float(series_a.baseline_check_corr(series_b, method=method))


@torch.no_grad()
def baseline_check_encode_latents(images: np.ndarray, ddpm_model, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    z_batches: list[np.ndarray] = []
    latent_norm_batches: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        x = baseline_check_normalize_for_ddpm(images[start:end]).to(device=device, dtype=torch.float32)
        z = ddpm_model.ae.encode(x)
        z_batches.append(z.cpu().numpy().astype(np.float32))
        latent_norm_batches.append(z.norm(dim=1).cpu().numpy().astype(np.float32))
    return np.concatenate(z_batches, axis=0), np.concatenate(latent_norm_batches, axis=0)


@torch.no_grad()
def baseline_check_ddpm_native_regime(
    z0_np: np.ndarray,
    ddpm_model,
    device: torch.device,
    batch_size: int,
    t_values: list[int],
) -> tuple[list[dict[str, float]], dict[str, float]]:
    per_t_rows: list[dict[str, float]] = []
    global_eps_energy_pred: list[np.ndarray] = []
    global_eps_energy_true: list[np.ndarray] = []
    global_t: list[np.ndarray] = []

    for t_value in t_values:
        mse_values: list[np.ndarray] = []
        cos_values: list[np.ndarray] = []
        eps_energy_pred_values: list[np.ndarray] = []
        eps_energy_true_values: list[np.ndarray] = []

        for start in range(0, len(z0_np), batch_size):
            end = min(start + batch_size, len(z0_np))
            z0 = torch.from_numpy(z0_np[start:end]).to(device=device, dtype=torch.float32)
            z0_view = z0.view(z0.shape[0], z0.shape[1], 1, 1)
            eps = torch.randn_like(z0_view)
            t = torch.full((z0.shape[0],), t_value, device=device, dtype=torch.long)
            z_t = ddpm_model.q_sample(z0_view, t, eps)
            eps_pred = ddpm_model.model(z_t, t)

            mse = F.mse_loss(eps_pred, eps, reduction="none").flatten(1).mean(dim=1)
            cos = F.cosine_similarity(eps_pred.flatten(1), eps.flatten(1), dim=1)
            eps_energy_pred = eps_pred.pow(2).flatten(1).mean(dim=1)
            eps_energy_true = eps.pow(2).flatten(1).mean(dim=1)

            mse_values.append(mse.cpu().numpy().astype(np.float32))
            cos_values.append(cos.cpu().numpy().astype(np.float32))
            eps_energy_pred_values.append(eps_energy_pred.cpu().numpy().astype(np.float32))
            eps_energy_true_values.append(eps_energy_true.cpu().numpy().astype(np.float32))

        mse_all = np.concatenate(mse_values, axis=0)
        cos_all = np.concatenate(cos_values, axis=0)
        eps_energy_pred_all = np.concatenate(eps_energy_pred_values, axis=0)
        eps_energy_true_all = np.concatenate(eps_energy_true_values, axis=0)

        per_t_rows.append(
            {
                "t": float(t_value),
                "mse_eps_pred_vs_eps": float(mse_all.mean()),
                "cosine_eps_pred_vs_eps": float(cos_all.mean()),
                "corr_eps_energy_pred_vs_true": baseline_check_corr(eps_energy_pred_all, eps_energy_true_all, method="pearson"),
            }
        )

        global_eps_energy_pred.append(eps_energy_pred_all)
        global_eps_energy_true.append(eps_energy_true_all)
        global_t.append(np.full_like(eps_energy_pred_all, t_value, dtype=np.float32))

    eps_energy_pred_global = np.concatenate(global_eps_energy_pred, axis=0)
    eps_energy_true_global = np.concatenate(global_eps_energy_true, axis=0)
    t_global = np.concatenate(global_t, axis=0)

    summary = {
        "global_corr_eps_energy_pred_vs_true": baseline_check_corr(eps_energy_pred_global, eps_energy_true_global, "pearson"),
        "global_corr_eps_energy_pred_vs_t": baseline_check_corr(eps_energy_pred_global, t_global, "pearson"),
        "global_spearman_eps_energy_pred_vs_t": baseline_check_corr(eps_energy_pred_global, t_global, "spearman"),
    }
    return per_t_rows, summary


@torch.no_grad()
def baseline_check_fixed_t_external_regime(
    z_noisy_np: np.ndarray,
    sigma: np.ndarray,
    ddpm_model,
    device: torch.device,
    batch_size: int,
    t_values: list[int],
) -> dict[str, float]:
    eps_energy_all: list[np.ndarray] = []
    score_energy_all: list[np.ndarray] = []
    t_all: list[np.ndarray] = []

    alphas_cumprod = ddpm_model.alphas_cumprod.detach().cpu().numpy()

    for t_value in t_values:
        eps_energy_values: list[np.ndarray] = []
        score_energy_values: list[np.ndarray] = []

        alpha_bar_t = float(alphas_cumprod[t_value])
        score_scale = np.sqrt(max(1.0 - alpha_bar_t, 1e-12))

        for start in range(0, len(z_noisy_np), batch_size):
            end = min(start + batch_size, len(z_noisy_np))
            z = torch.from_numpy(z_noisy_np[start:end]).to(device=device, dtype=torch.float32)
            z_view = z.view(z.shape[0], z.shape[1], 1, 1)
            noise = torch.randn_like(z_view)
            t = torch.full((z.shape[0],), t_value, device=device, dtype=torch.long)
            z_t = ddpm_model.q_sample(z_view, t, noise)
            eps_pred = ddpm_model.model(z_t, t)

            eps_energy = eps_pred.pow(2).flatten(1).mean(dim=1)
            score_energy = (eps_pred / score_scale).pow(2).flatten(1).mean(dim=1)

            eps_energy_values.append(eps_energy.cpu().numpy().astype(np.float32))
            score_energy_values.append(score_energy.cpu().numpy().astype(np.float32))

        eps_energy_t = np.concatenate(eps_energy_values, axis=0)
        score_energy_t = np.concatenate(score_energy_values, axis=0)

        eps_energy_all.append(eps_energy_t)
        score_energy_all.append(score_energy_t)
        t_all.append(np.full_like(eps_energy_t, t_value, dtype=np.float32))

    eps_energy_all_np = np.concatenate(eps_energy_all, axis=0)
    score_energy_all_np = np.concatenate(score_energy_all, axis=0)
    sigma_all = np.tile(sigma.astype(np.float32), len(t_values))
    t_all_np = np.concatenate(t_all, axis=0)

    return {
        "corr_eps_energy_vs_t": baseline_check_corr(eps_energy_all_np, t_all_np, "pearson"),
        "corr_score_energy_vs_t": baseline_check_corr(score_energy_all_np, t_all_np, "pearson"),
        "corr_eps_energy_vs_sigma": baseline_check_corr(eps_energy_all_np, sigma_all, "pearson"),
        "corr_score_energy_vs_sigma": baseline_check_corr(score_energy_all_np, sigma_all, "pearson"),
        "spearman_eps_energy_vs_sigma": baseline_check_corr(eps_energy_all_np, sigma_all, "spearman"),
        "spearman_score_energy_vs_sigma": baseline_check_corr(score_energy_all_np, sigma_all, "spearman"),
    }


def baseline_check_build_sigma_to_t_map(ddpm_model, sigma: np.ndarray, matched_t_max: int) -> np.ndarray:
    alpha_bar = ddpm_model.alphas_cumprod.detach().cpu().numpy()[:matched_t_max]
    ddpm_noise_level = np.sqrt(np.clip(1.0 - alpha_bar, 0.0, None))
    diff = np.abs(ddpm_noise_level[None, :] - sigma[:, None])
    matched_t = diff.argmin(axis=1).astype(np.int64)
    return matched_t


@torch.no_grad()
def baseline_check_matched_t_external_regime(
    z_noisy_np: np.ndarray,
    sigma: np.ndarray,
    latent_norm: np.ndarray,
    matched_t: np.ndarray,
    ddpm_model,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    score_values: list[np.ndarray] = []
    scaled_score_values: list[np.ndarray] = []
    used_t_values: list[np.ndarray] = []

    alpha_bar = ddpm_model.alphas_cumprod.detach().cpu().numpy()

    for start in range(0, len(z_noisy_np), batch_size):
        end = min(start + batch_size, len(z_noisy_np))
        z = torch.from_numpy(z_noisy_np[start:end]).to(device=device, dtype=torch.float32)
        t_np = matched_t[start:end]
        t = torch.from_numpy(t_np).to(device=device, dtype=torch.long)
        z_view = z.view(z.shape[0], z.shape[1], 1, 1)
        noise = torch.randn_like(z_view)
        z_t = ddpm_model.q_sample(z_view, t, noise)
        eps_pred = ddpm_model.model(z_t, t)

        score = eps_pred.pow(2).flatten(1).mean(dim=1)
        scale = np.sqrt(np.clip(1.0 - alpha_bar[t_np], 1e-12, None)).astype(np.float32)
        scale_t = torch.from_numpy(scale).to(device=device, dtype=torch.float32).view(-1, 1, 1, 1)
        scaled_score = (eps_pred / scale_t).pow(2).flatten(1).mean(dim=1)

        score_values.append(score.cpu().numpy().astype(np.float32))
        scaled_score_values.append(scaled_score.cpu().numpy().astype(np.float32))
        used_t_values.append(t_np.astype(np.float32))

    score_np = np.concatenate(score_values, axis=0)
    scaled_score_np = np.concatenate(scaled_score_values, axis=0)
    used_t_np = np.concatenate(used_t_values, axis=0)
    score_over_latent_np = score_np / (latent_norm + 1e-8)

    return {
        "corr_matched_t_vs_sigma": baseline_check_corr(used_t_np, sigma, "pearson"),
        "corr_score_vs_sigma": baseline_check_corr(score_np, sigma, "pearson"),
        "corr_scaled_score_vs_sigma": baseline_check_corr(scaled_score_np, sigma, "pearson"),
        "corr_score_over_latent_vs_sigma": baseline_check_corr(score_over_latent_np, sigma, "pearson"),
        "spearman_score_vs_sigma": baseline_check_corr(score_np, sigma, "spearman"),
        "spearman_scaled_score_vs_sigma": baseline_check_corr(scaled_score_np, sigma, "spearman"),
        "spearman_score_over_latent_vs_sigma": baseline_check_corr(score_over_latent_np, sigma, "spearman"),
        "matched_t_min": float(used_t_np.min()),
        "matched_t_max": float(used_t_np.max()),
        "matched_t_mean": float(used_t_np.mean()),
    }


def baseline_check_write_report(
    output_path: Path,
    native_rows: list[dict[str, float]],
    native_summary: dict[str, float],
    fixed_summary: dict[str, float],
    matched_summary: dict[str, float],
) -> None:
    native_df = pd.DataFrame(native_rows)
    lines = [
        "Sanity Check Score Report",
        "",
        "[1] DDPM-native regime",
        native_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"),
        "",
        "native_summary:",
    ]
    lines.extend(f"{key}={value:.6f}" for key, value in native_summary.items())
    lines.extend(
        [
            "",
            "[2] True score scaling",
        ]
    )
    lines.extend(f"{key}={value:.6f}" for key, value in fixed_summary.items())
    lines.extend(
        [
            "",
            "[3] External-noise matching",
            "matched_t heuristic: nearest DDPM noise level sqrt(1 - alpha_bar_t) to external sigma",
        ]
    )
    lines.extend(f"{key}={value:.6f}" for key, value in matched_summary.items())
    output_path.write_text("\n".join(lines), encoding="utf-8")


def baseline_check_main() -> None:
    args = baseline_check_parse_args()
    device = baseline_check_resolve_device(args.device)

    baseline_check_ensure_file_exists(args.ddpm_path, "DDPM checkpoint")
    baseline_check_ensure_file_exists(args.clean_path, "clean dataset")
    baseline_check_ensure_file_exists(args.noisy_path, "noisy dataset")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clean_h5 = baseline_check_load_h5_data(args.clean_path)
    noisy_h5 = baseline_check_load_h5_data(args.noisy_path)

    clean_images = clean_h5["dataset"].astype(np.float32)
    noisy_images = noisy_h5["dataset"].astype(np.float32)
    sigma = noisy_h5["noise_std"].astype(np.float32)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ddpm_checkpoint = torch.load(args.ddpm_path, map_location=device)
    ddpm_params = ddpm_checkpoint.get("DDPM_params", {})
    base_dim = int(ddpm_params.get("base_dim", 16))
    deep = int(ddpm_params.get("deep", 3))

    ddpm_model = build_DDPM_model(base_dim=base_dim, deep=deep, device=str(device))
    ddpm_model.model.load_state_dict(ddpm_checkpoint["model_state_dict"])
    ddpm_model.eval()

    z0_np, _ = baseline_check_encode_latents(clean_images, ddpm_model, device, args.batch_size)
    z_noisy_np, latent_norm = baseline_check_encode_latents(noisy_images, ddpm_model, device, args.batch_size)

    native_rows, native_summary = baseline_check_ddpm_native_regime(
        z0_np=z0_np,
        ddpm_model=ddpm_model,
        device=device,
        batch_size=args.batch_size,
        t_values=args.t_values,
    )
    fixed_summary = baseline_check_fixed_t_external_regime(
        z_noisy_np=z_noisy_np,
        sigma=sigma,
        ddpm_model=ddpm_model,
        device=device,
        batch_size=args.batch_size,
        t_values=args.t_values,
    )
    matched_t = baseline_check_build_sigma_to_t_map(ddpm_model, sigma=sigma, matched_t_max=args.matched_t_max)
    matched_summary = baseline_check_matched_t_external_regime(
        z_noisy_np=z_noisy_np,
        sigma=sigma,
        latent_norm=latent_norm,
        matched_t=matched_t,
        ddpm_model=ddpm_model,
        device=device,
        batch_size=args.batch_size,
    )

    report_path = args.output_dir / "sanity_check_score.txt"
    baseline_check_write_report(
        output_path=report_path,
        native_rows=native_rows,
        native_summary=native_summary,
        fixed_summary=fixed_summary,
        matched_summary=matched_summary,
    )

    print(f"saved_report={report_path}")
    print("[1] native_summary")
    for key, value in native_summary.items():
        print(f"{key}={value:.6f}")
    print("[2] fixed_summary")
    for key, value in fixed_summary.items():
        print(f"{key}={value:.6f}")
    print("[3] matched_summary")
    for key, value in matched_summary.items():
        print(f"{key}={value:.6f}")




if __name__ == "__main__":
    main()
