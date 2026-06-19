from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.Unet_model import UNet
from src.autoencoder import SimpleAE
from src.dataset_registry import build_torchvision_split


DATASET_NAME = "mnist"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latent score calibration experiment: (S1..Sk) -> sigma_hat")
    parser.add_argument(
        "--encoder-path",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / DATASET_NAME / "autoencoders" / "ae_baseline_mnist" / "E.pt",
    )
    parser.add_argument(
        "--ae-checkpoint-path",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / DATASET_NAME / "autoencoders" / "ae_baseline_mnist" / "autoencoder_checkpoint.pt",
    )
    parser.add_argument(
        "--ddpm-path",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / DATASET_NAME / "ddpm" / "ddpm_mnist_baseline.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / DATASET_NAME / "latent_score_calibration",
    )
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--noise-repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma-min", type=float, default=0.1)
    parser.add_argument("--sigma-max", type=float, default=0.8)
    parser.add_argument("--timesteps", type=int, nargs="+", default=[50, 100, 200, 400, 700])
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def clean_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in output_dir.iterdir():
        if child.is_file():
            child.unlink()


def _extract_state_dict(maybe_state: Any) -> dict[str, torch.Tensor]:
    if isinstance(maybe_state, dict) and "state_dict" in maybe_state and isinstance(maybe_state["state_dict"], dict):
        return maybe_state["state_dict"]
    if isinstance(maybe_state, dict):
        return maybe_state
    raise ValueError("Unsupported checkpoint format: expected a state dict-like object.")


def load_encoder(
    encoder_path: Path,
    ae_checkpoint_path: Path,
    device: torch.device,
) -> SimpleAE:
    model = SimpleAE().to(device)
    if encoder_path.exists():
        encoder_state = _extract_state_dict(torch.load(encoder_path, map_location=device))
        model.encoder.load_state_dict(encoder_state)
    elif ae_checkpoint_path.exists():
        checkpoint = torch.load(ae_checkpoint_path, map_location=device)
        checkpoint_dict = checkpoint if isinstance(checkpoint, dict) else {}
        if "model_state_dict" in checkpoint_dict:
            model.load_state_dict(checkpoint_dict["model_state_dict"])
        else:
            model.load_state_dict(_extract_state_dict(checkpoint))
    else:
        raise FileNotFoundError(
            f"Missing encoder checkpoint. Expected one of: {encoder_path} or {ae_checkpoint_path}"
        )

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_baseline_ddpm(
    ddpm_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.Tensor]:
    if not ddpm_path.exists():
        raise FileNotFoundError(f"Missing DDPM checkpoint: {ddpm_path}")

    checkpoint = torch.load(ddpm_path, map_location=device)
    base_dim = int(checkpoint.get("base_dim", 16))
    time_dim = int(checkpoint.get("time_dim", 128))
    n_steps = int(checkpoint.get("n_steps", 1000))

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_dim=base_dim,
        time_dim=time_dim,
        kernel_sizes=[3, 3, 3],
        strides=[2, 2, 2],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    betas = torch.linspace(1e-4, 0.02, n_steps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    noise_std_schedule = torch.sqrt(1.0 - alphas_cumprod)
    return model, noise_std_schedule


def build_loader(num_samples: int, batch_size: int) -> DataLoader:
    dataset = build_torchvision_split(
        dataset_cfg=DATASET_NAME,
        train=False,
        data_root=PROJECT_ROOT / "data",
        transform_profile="normalized",
        download=False,
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


@torch.no_grad()
def build_calibration_dataset(
    loader: DataLoader,
    ae: SimpleAE,
    ddpm_model: torch.nn.Module,
    noise_std_schedule: torch.Tensor,
    timesteps: list[int],
    sigma_min: float,
    sigma_max: float,
    noise_repeats: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    raw_mapped_rows: list[np.ndarray] = []

    for x, _ in loader:
        x = x.to(device)
        batch_size = x.shape[0]

        for _ in range(noise_repeats):
            sigma = torch.empty(batch_size, device=device).uniform_(sigma_min, sigma_max)
            eps_x = torch.randn_like(x)
            x_sigma = x + sigma.view(-1, 1, 1, 1) * eps_x

            # Match inversion experiment (latent_noise_mismatch, mode B):
            # z_noisy = E(x + sigma * eps_x), no extra DDPM q_sample noise on z.
            z_noisy = flatten_latent(ae.encode(x_sigma))
            x_for_ddpm = ae.decode(z_noisy)
            x_for_ddpm = x_for_ddpm * 2.0 - 1.0

            per_t_scores: list[torch.Tensor] = []
            for t_value in timesteps:
                t = torch.full((batch_size,), int(t_value), device=device, dtype=torch.long)
                eps_pred = ddpm_model(x_for_ddpm, t)
                s_t = eps_pred.flatten(start_dim=1).pow(2).sum(dim=1)
                per_t_scores.append(s_t)

            # Also keep exact legacy raw score with sigma->t mapping (single score per sample).
            t_sigma = sigma_to_t(sigma, noise_std_schedule)
            eps_pred_sigma = ddpm_model(x_for_ddpm, t_sigma)
            raw_mapped = eps_pred_sigma.flatten(start_dim=1).pow(2).sum(dim=1)

            x_rows.append(torch.stack(per_t_scores, dim=1).cpu().numpy().astype(np.float32))
            y_rows.append(sigma.cpu().numpy().astype(np.float32))
            raw_mapped_rows.append(raw_mapped.cpu().numpy().astype(np.float32))

    x_all = np.concatenate(x_rows, axis=0)
    y_all = np.concatenate(y_rows, axis=0)
    raw_mapped_all = np.concatenate(raw_mapped_rows, axis=0)
    return x_all, y_all, raw_mapped_all


def sigma_to_t(sigma_flat: torch.Tensor, noise_std_schedule: torch.Tensor) -> torch.Tensor:
    sigma = sigma_flat.detach().clamp(float(noise_std_schedule[0]), float(noise_std_schedule[-1]))
    idx = torch.searchsorted(noise_std_schedule, sigma)
    idx = idx.clamp(0, noise_std_schedule.shape[0] - 1)
    prev_idx = (idx - 1).clamp(0, noise_std_schedule.shape[0] - 1)

    cur_err = (noise_std_schedule[idx] - sigma).abs()
    prev_err = (noise_std_schedule[prev_idx] - sigma).abs()
    choose_prev = prev_err < cur_err
    return torch.where(choose_prev, prev_idx, idx).long()


def split_train_test(
    n_items: int,
    train_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = n_items
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    train_n = int(n * train_ratio)
    train_idx = idx[:train_n]
    test_idx = idx[train_n:]
    return train_idx, test_idx


@dataclass
class StandardScalerNP:
    mean_: np.ndarray
    std_: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "StandardScalerNP":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean_=mean, std_=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_


@dataclass
class IsotonicRegressorNP:
    x_thresholds: np.ndarray
    y_thresholds: np.ndarray
    increasing: bool = True

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, increasing: bool = True) -> "IsotonicRegressorNP":
        x_work = np.asarray(x, dtype=np.float64).reshape(-1)
        y_work = np.asarray(y, dtype=np.float64).reshape(-1)

        if not increasing:
            y_work = -y_work

        order = np.argsort(x_work)
        x_sorted = x_work[order]
        y_sorted = y_work[order]

        # Pool Adjacent Violators (equal weights)
        v = y_sorted.copy()
        w = np.ones_like(v)
        i = 0
        while i < len(v) - 1:
            if v[i] <= v[i + 1] + 1e-12:
                i += 1
                continue
            total_w = w[i] + w[i + 1]
            avg = (v[i] * w[i] + v[i + 1] * w[i + 1]) / total_w
            v[i] = avg
            w[i] = total_w
            v = np.delete(v, i + 1)
            w = np.delete(w, i + 1)
            x_sorted = np.delete(x_sorted, i + 1)
            if i > 0:
                i -= 1

        # Expand block values to step thresholds
        # Recompute with block boundaries using a second pass.
        x_sorted = np.asarray(np.sort(x_work), dtype=np.float64)
        y_sorted = y_work[np.argsort(x_work)]
        blocks = [[x_sorted[i], x_sorted[i], y_sorted[i], 1.0] for i in range(len(x_sorted))]
        j = 0
        while j < len(blocks) - 1:
            if blocks[j][2] <= blocks[j + 1][2] + 1e-12:
                j += 1
                continue
            wsum = blocks[j][3] + blocks[j + 1][3]
            yavg = (blocks[j][2] * blocks[j][3] + blocks[j + 1][2] * blocks[j + 1][3]) / wsum
            merged = [blocks[j][0], blocks[j + 1][1], yavg, wsum]
            blocks[j] = merged
            del blocks[j + 1]
            if j > 0:
                j -= 1

        x_thr = np.array([b[1] for b in blocks], dtype=np.float64)
        y_thr = np.array([b[2] for b in blocks], dtype=np.float64)
        if not increasing:
            y_thr = -y_thr
        return cls(x_thresholds=x_thr, y_thresholds=y_thr, increasing=increasing)

    def predict(self, x: np.ndarray) -> np.ndarray:
        xq = np.asarray(x, dtype=np.float64).reshape(-1)
        idx = np.searchsorted(self.x_thresholds, xq, side="left")
        idx = np.clip(idx, 0, len(self.y_thresholds) - 1)
        return self.y_thresholds[idx].astype(np.float32)


@dataclass
class LinearRegressorNP:
    weights: np.ndarray
    bias: float

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray) -> "LinearRegressorNP":
        x_aug = np.hstack([x, np.ones((x.shape[0], 1), dtype=x.dtype)])
        coef, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
        return cls(weights=coef[:-1], bias=float(coef[-1]))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights + self.bias


@dataclass
class RidgeRegressorNP:
    weights: np.ndarray
    bias: float

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, alpha: float) -> "RidgeRegressorNP":
        x_mean = x.mean(axis=0)
        y_mean = float(y.mean())
        xc = x - x_mean
        yc = y - y_mean
        xtx = xc.T @ xc
        reg = alpha * np.eye(xtx.shape[0], dtype=xtx.dtype)
        weights = np.linalg.solve(xtx + reg, xc.T @ yc)
        bias = float(y_mean - x_mean @ weights)
        return cls(weights=weights, bias=bias)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights + self.bias


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(pd.Series(a).corr(pd.Series(b), method="pearson"))


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def save_scatter(x: np.ndarray, y: np.ndarray, x_label: str, y_label: str, title: str, output_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, s=8, alpha=0.25)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_calibration_curve(
    sigma_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_path: Path,
    sigma_min: float,
    sigma_max: float,
    n_bins: int = 10,
) -> None:
    edges = np.linspace(sigma_min, sigma_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(sigma_true, edges) - 1, 0, n_bins - 1)

    plt.figure(figsize=(7, 5))
    plt.plot(centers, centers, linestyle="--", label="ideal: sigma_hat=sigma")
    for name, pred in predictions.items():
        mean_pred = np.array(
            [float(pred[idx == i].mean()) if np.any(idx == i) else np.nan for i in range(n_bins)],
            dtype=np.float32,
        )
        plt.plot(centers, mean_pred, marker="o", label=name)
    plt.xlabel("sigma (bin center)")
    plt.ylabel("mean sigma_hat in bin")
    plt.title("Calibration Curve")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def sign_label(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        values = []
        for h in headers:
            v = row[h]
            if isinstance(v, float):
                values.append(f"{v:.6f}")
            else:
                values.append(str(v))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    clean_output_dir(args.output_dir)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae = load_encoder(args.encoder_path, args.ae_checkpoint_path, device)
    ddpm_model, noise_std_schedule = load_baseline_ddpm(args.ddpm_path, device)
    loader = build_loader(num_samples=args.num_samples, batch_size=args.batch_size)

    x_all, y_all, raw_mapped_all = build_calibration_dataset(
        loader=loader,
        ae=ae,
        ddpm_model=ddpm_model,
        noise_std_schedule=noise_std_schedule,
        timesteps=[int(t) for t in args.timesteps],
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
        noise_repeats=int(args.noise_repeats),
        device=device,
    )

    feature_columns = [f"S_t{int(t)}" for t in args.timesteps]
    feature_df_all = pd.DataFrame(x_all, columns=feature_columns)
    feature_df_all["sigma"] = y_all
    feature_df_all["raw_score_sigma_mapped_t"] = raw_mapped_all

    per_timestep_rows: list[dict[str, float | int | str]] = []
    for t in args.timesteps:
        col = f"S_t{int(t)}"
        p = pearson_corr(feature_df_all[col].to_numpy(), y_all)
        s = spearman_corr(feature_df_all[col].to_numpy(), y_all)
        m = float(feature_df_all[col].mean())
        st = float(feature_df_all[col].std())
        per_timestep_rows.append(
            {
                "timestep": int(t),
                "pearson": p,
                "spearman": s,
                "mean": m,
                "std": st,
                "corr_sign": sign_label(p),
            }
        )

    per_timestep_df = pd.DataFrame(per_timestep_rows)
    per_timestep_df.to_csv(args.output_dir / "per_timestep_diagnostics.csv", index=False)

    for t in args.timesteps:
        col = f"S_t{int(t)}"
        save_scatter(
            x=y_all,
            y=feature_df_all[col].to_numpy(),
            x_label="sigma",
            y_label=col,
            title=f"sigma vs {col}",
            output_path=args.output_dir / f"sigma_vs_{col}.png",
        )

    train_idx, test_idx = split_train_test(
        n_items=len(y_all),
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
    )
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]
    raw_train = raw_mapped_all[train_idx]
    raw_test = raw_mapped_all[test_idx]

    # Baseline raw score (mean across timesteps)
    x_test = x_all[test_idx]
    raw_score_test = x_test.mean(axis=1)
    raw_pearson = pearson_corr(raw_score_test, y_test)
    raw_spearman = spearman_corr(raw_score_test, y_test)
    raw_mapped_pearson = pearson_corr(raw_test, y_test)
    raw_mapped_spearman = spearman_corr(raw_test, y_test)

    metrics_rows: list[dict[str, float | str]] = []
    metrics_rows.append(
        {
            "model": "baseline_raw_sigma_mapped_t",
            "scaled": "none",
            "pearson": raw_mapped_pearson,
            "spearman": raw_mapped_spearman,
            "mse": np.nan,
            "corr_sign": sign_label(raw_mapped_pearson),
        }
    )
    metrics_rows.append(
        {
            "model": "baseline_raw_mean_score",
            "scaled": "none",
            "pearson": raw_pearson,
            "spearman": raw_spearman,
            "mse": np.nan,
            "corr_sign": sign_label(raw_pearson),
        }
    )
    for t in args.timesteps:
        col = f"S_t{int(t)}"
        s_t_test = x_test[:, list(args.timesteps).index(t)]
        metrics_rows.append(
            {
                "model": f"baseline_raw_{col}",
                "scaled": "none",
                "pearson": pearson_corr(s_t_test, y_test),
                "spearman": spearman_corr(s_t_test, y_test),
                "mse": np.nan,
                "corr_sign": sign_label(pearson_corr(s_t_test, y_test)),
            }
        )

    # Train calibrators only if inversion is present after reproducing old score setup.
    inversion_present = (
        (raw_mapped_pearson < -0.2)
        or (raw_mapped_spearman < -0.2)
        or (float(per_timestep_df["pearson"].min()) < -0.2)
        or (float(per_timestep_df["spearman"].min()) < -0.2)
    )

    best_linear_pred = None
    best_ridge_pred = None
    best_row = None
    model_preds_for_curve: dict[str, np.ndarray] = {}

    if inversion_present:
        x1_train = raw_train.reshape(-1, 1)
        x1_test = raw_test.reshape(-1, 1)
        scaler_1d = StandardScalerNP.fit(x1_train)
        x1_train_scaled = scaler_1d.transform(x1_train)
        x1_test_scaled = scaler_1d.transform(x1_test)

        lin = LinearRegressorNP.fit(x1_train, y_train)
        lin_pred = lin.predict(x1_test)

        ridge = RidgeRegressorNP.fit(x1_train, y_train, alpha=float(args.ridge_alpha))
        ridge_pred = ridge.predict(x1_test)

        iso = IsotonicRegressorNP.fit(raw_train, y_train, increasing=False)
        iso_pred = iso.predict(raw_test)

        poly2_train = np.hstack([x1_train_scaled, x1_train_scaled ** 2])
        poly2_test = np.hstack([x1_test_scaled, x1_test_scaled ** 2])
        poly2_ridge = RidgeRegressorNP.fit(poly2_train, y_train, alpha=float(args.ridge_alpha))
        poly2_ridge_pred = poly2_ridge.predict(poly2_test)

        def add_metrics(name: str, pred: np.ndarray) -> None:
            metrics_rows.append(
                {
                    "model": name,
                    "scaled": "n/a",
                    "pearson": pearson_corr(pred, y_test),
                    "spearman": spearman_corr(pred, y_test),
                    "mse": mse(pred, y_test),
                    "corr_sign": sign_label(pearson_corr(pred, y_test)),
                }
            )

        add_metrics("linear_regression_raw_mapped", lin_pred)
        add_metrics("ridge_regression_raw_mapped", ridge_pred)
        add_metrics("isotonic_regression_raw_mapped", iso_pred)
        add_metrics("poly2_ridge_raw_mapped", poly2_ridge_pred)

        metrics_df = pd.DataFrame(metrics_rows)
        best_linear_pred = lin_pred
        best_ridge_pred = ridge_pred
        model_preds_for_curve = {
            "linear": lin_pred,
            "ridge": ridge_pred,
            "isotonic": iso_pred,
            "poly2_ridge": poly2_ridge_pred,
        }
        non_baseline = metrics_df[
            ~metrics_df["model"].str.startswith("baseline_raw_")
        ].copy()
        best_idx = non_baseline["mse"].astype(float).idxmin()
        best_row = non_baseline.loc[best_idx]
    else:
        metrics_df = pd.DataFrame(metrics_rows)

    metrics_df.to_csv(args.output_dir / "metrics.csv", index=False)

    save_scatter(
        x=raw_test,
        y=y_test,
        x_label="raw_score (sigma->t mapped)",
        y_label="sigma",
        title="Raw Score vs Sigma",
        output_path=args.output_dir / "raw_score_vs_sigma.png",
    )
    if inversion_present and best_linear_pred is not None and best_ridge_pred is not None:
        save_scatter(
            x=y_test,
            y=best_linear_pred,
            x_label="sigma",
            y_label="sigma_hat",
            title="Sigma_hat vs Sigma (Linear on raw_mapped_score)",
            output_path=args.output_dir / "sigma_hat_vs_sigma_linear.png",
        )
        save_scatter(
            x=y_test,
            y=best_ridge_pred,
            x_label="sigma",
            y_label="sigma_hat",
            title="Sigma_hat vs Sigma (Ridge on raw_mapped_score)",
            output_path=args.output_dir / "sigma_hat_vs_sigma_ridge.png",
        )
        if "isotonic" in model_preds_for_curve:
            save_scatter(
                x=y_test,
                y=model_preds_for_curve["isotonic"],
                x_label="sigma",
                y_label="sigma_hat",
                title="Sigma_hat vs Sigma (Isotonic, increasing=False)",
                output_path=args.output_dir / "sigma_hat_vs_sigma_isotonic.png",
            )
        save_calibration_curve(
            sigma_true=y_test,
            predictions=model_preds_for_curve,
            output_path=args.output_dir / "calibration_curve.png",
            sigma_min=float(args.sigma_min),
            sigma_max=float(args.sigma_max),
            n_bins=10,
        )

    # Save dataset
    dataset_df = pd.DataFrame(x_all, columns=feature_columns)
    dataset_df["sigma"] = y_all
    dataset_df["raw_score_sigma_mapped_t"] = raw_mapped_all
    split_mask = np.zeros(len(y_all), dtype=np.int32)
    train_count = int(len(y_all) * float(args.train_ratio))
    split_mask[train_idx] = 1
    dataset_df["is_train"] = split_mask
    dataset_df.to_csv(args.output_dir / "calibration_dataset.csv", index=False)
    np.savez(
        args.output_dir / "calibration_dataset.npz",
        X=x_all.astype(np.float32),
        y=y_all.astype(np.float32),
        timesteps=np.asarray(args.timesteps, dtype=np.int64),
    )

    # Markdown report
    report_lines = [
        "# Latent Score Calibration",
        "",
        "## Setup",
        f"- encoder_path: `{args.encoder_path}`",
        f"- ddpm_path: `{args.ddpm_path}`",
        f"- split: MNIST test, num_samples={args.num_samples}, noise_repeats={args.noise_repeats}",
        f"- timesteps: {list(map(int, args.timesteps))}",
        "",
        "## Baseline (raw score)",
        "- raw_score (legacy inversion setup) = ||eps_pred(x_from_E(x_sigma), t_sigma)||^2, where t_sigma is mapped from sigma",
        f"- Pearson(raw_score_sigma_mapped_t, sigma): {raw_mapped_pearson:.6f} ({sign_label(raw_mapped_pearson)})",
        f"- Spearman(raw_score_sigma_mapped_t, sigma): {raw_mapped_spearman:.6f}",
        "- auxiliary raw_score = mean(S_i), where S_i = ||eps_pred(x_from_E(x_sigma), t_i)||^2",
        f"- Pearson(raw_score, sigma): {raw_pearson:.6f} ({sign_label(raw_pearson)})",
        f"- Spearman(raw_score, sigma): {raw_spearman:.6f}",
        "",
        "## Per-timestep raw score diagnostics",
        "Columns: timestep | pearson | spearman | mean | std | corr_sign",
        dataframe_to_markdown(per_timestep_df),
        "",
        "## Calibrator Metrics",
        dataframe_to_markdown(metrics_df),
        "",
        "## Conclusion",
        f"- inversion_present: {inversion_present}",
    ]
    if inversion_present and best_row is not None:
        report_lines.extend(
            [
                f"- Best calibrator: `{best_row['model']}` (scaled={best_row['scaled']})",
                f"- Best Pearson: {float(best_row['pearson']):.6f}",
                f"- Best Spearman: {float(best_row['spearman']):.6f}",
                f"- Best MSE: {float(best_row['mse']):.6f}",
            ]
        )
    else:
        report_lines.extend(
            [
                "",
                "Calibration is not meaningful because score features do not preserve sigma information in this setup.",
            ]
        )

    (args.output_dir / "metrics.md").write_text("\n".join(report_lines), encoding="utf-8")

    run_config = {
        "encoder_path": str(args.encoder_path),
        "ae_checkpoint_path": str(args.ae_checkpoint_path),
        "ddpm_path": str(args.ddpm_path),
        "num_samples": int(args.num_samples),
        "noise_repeats": int(args.noise_repeats),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "sigma_min": float(args.sigma_min),
        "sigma_max": float(args.sigma_max),
        "timesteps": [int(t) for t in args.timesteps],
        "ridge_alpha": float(args.ridge_alpha),
        "train_ratio": float(args.train_ratio),
        "device": str(device),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print(f"output_dir={args.output_dir}")
    print(f"inversion_present={inversion_present}")
    print(f"raw_mapped_pearson={raw_mapped_pearson:.6f}")
    print(f"raw_mapped_spearman={raw_mapped_spearman:.6f}")
    print(f"raw_pearson={raw_pearson:.6f}")
    print(f"raw_spearman={raw_spearman:.6f}")
    if inversion_present and best_row is not None:
        print(f"best_model={best_row['model']} scaled={best_row['scaled']}")
        print(f"best_pearson={float(best_row['pearson']):.6f}")
        print(f"best_spearman={float(best_row['spearman']):.6f}")
        print(f"best_mse={float(best_row['mse']):.6f}")
    else:
        print("best_model=skipped")


if __name__ == "__main__":
    main()
