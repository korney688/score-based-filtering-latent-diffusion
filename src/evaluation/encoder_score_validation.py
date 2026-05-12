"""Legacy Stage 1 helper for comparing several encoders through DDPM scores.

This is not the final frozen-encoder DDPM validation. The active Stage 2 DDPM
score validation lives in src.evaluation.latent_ddpm_score_validation.
"""

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
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.DDPM_model import build_DDPM_model
from src.autoencoder import SimpleAE
from src.autoencoder_noise_consistency import NoiseConsistencyAutoencoder
from src.autoencoder_representation import RepresentationAutoencoder
from src.autoencoder_vae import VariationalAutoencoder


OUTPUT_DIR = PROJECT_ROOT / "experiments" / "exp_002_encoder_validation" / "score_validation"
DDPM_RUN_DIR = PROJECT_ROOT / "outputs" / "ddpm" / "latent_ddpm_baseline_ae_noise_consistency_mnist"
DDPM_PATH = DDPM_RUN_DIR / "epoch_030.pth"
AUTOENCODER_CHECKPOINT_PATH = (
    PROJECT_ROOT / "models" / "autoencoders" / "ae_noise_consistency_mnist" / "autoencoder_checkpoint.pt"
)
BATCH_SIZE = 128
FAST_NUM_SAMPLES = 500
FULL_NUM_SAMPLES = 2000
FAST_NOISE_REPEATS = 1
FULL_NOISE_REPEATS = 3
SEED = 42
SIGMA_MIN = 0.1
SIGMA_MAX = 0.8
TRAIN_RATIO = 0.8


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
    parser = argparse.ArgumentParser(description="Quick score comparison for trained MNIST encoders.")
    parser.add_argument("--fast_eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--noise-repeats", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
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


def load_ddpm_model(ddpm_path: Path, device: torch.device) -> tuple[torch.nn.Module, torch.Tensor]:
    if not ddpm_path.exists():
        raise FileNotFoundError(f"Missing DDPM checkpoint: {ddpm_path}")

    checkpoint = torch.load(ddpm_path, map_location=device)
    ddpm_params = checkpoint.get("DDPM_params", {})
    base_dim = int(ddpm_params.get("base_dim", 16))
    deep = int(ddpm_params.get("deep", 3))
    latent_noise_mode = checkpoint.get("latent_noise_mode", "baseline")
    autoencoder_kind = checkpoint.get("autoencoder_kind", "noise_consistency")
    autoencoder_checkpoint_path = checkpoint.get("autoencoder_checkpoint_path", str(AUTOENCODER_CHECKPOINT_PATH))

    model = build_DDPM_model(
        base_dim=base_dim,
        deep=deep,
        device=str(device),
        latent_noise_mode=latent_noise_mode,
        autoencoder_kind=autoencoder_kind,
        autoencoder_checkpoint_path=autoencoder_checkpoint_path,
    )
    model.model.load_state_dict(checkpoint["model_state_dict"])
    model.model.eval()
    for param in model.model.parameters():
        param.requires_grad = False

    noise_std_schedule = torch.sqrt(1.0 - model.alphas_cumprod)
    return model, noise_std_schedule


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


def sigma_to_t(sigma_flat: torch.Tensor, noise_std_schedule: torch.Tensor) -> torch.Tensor:
    sigma = sigma_flat.detach().clamp(float(noise_std_schedule[0]), float(noise_std_schedule[-1]))
    idx = torch.searchsorted(noise_std_schedule, sigma)
    idx = idx.clamp(0, noise_std_schedule.shape[0] - 1)
    prev_idx = (idx - 1).clamp(0, noise_std_schedule.shape[0] - 1)

    cur_err = (noise_std_schedule[idx] - sigma).abs()
    prev_err = (noise_std_schedule[prev_idx] - sigma).abs()
    choose_prev = prev_err < cur_err
    return torch.where(choose_prev, prev_idx, idx).long()


@torch.no_grad()
def build_raw_score_dataset(
    loader: DataLoader,
    ae: torch.nn.Module,
    ddpm_model,
    noise_std_schedule: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
    noise_repeats: int,
    device: torch.device,
) -> pd.DataFrame:
    sigma_all: list[np.ndarray] = []
    score_all: list[np.ndarray] = []

    for x, _ in loader:
        x = x.to(device)
        batch_size = x.shape[0]

        for _ in range(noise_repeats):
            sigma = torch.empty(batch_size, device=device).uniform_(sigma_min, sigma_max)
            eps_x = torch.randn_like(x)
            x_sigma = x + sigma.view(-1, 1, 1, 1) * eps_x

            z_noisy = flatten_latent(encode_deterministic(ae, x_sigma))
            z_noisy_view = z_noisy.view(z_noisy.shape[0], z_noisy.shape[1], 1, 1)

            t_sigma = sigma_to_t(sigma, noise_std_schedule)
            eps_pred = ddpm_model.model(z_noisy_view, t_sigma)
            raw_score = eps_pred.flatten(start_dim=1).pow(2).sum(dim=1)

            sigma_all.append(sigma.cpu().numpy().astype(np.float32))
            score_all.append(raw_score.cpu().numpy().astype(np.float32))

    return pd.DataFrame(
        {
            "sigma": np.concatenate(sigma_all, axis=0),
            "raw_score_sigma_mapped_t": np.concatenate(score_all, axis=0),
        }
    )


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
        blocks = [[x_sorted[i], x_sorted[i], y_sorted[i], 1.0] for i in range(len(x_sorted))]

        j = 0
        while j < len(blocks) - 1:
            if blocks[j][2] <= blocks[j + 1][2] + 1e-12:
                j += 1
                continue
            wsum = blocks[j][3] + blocks[j + 1][3]
            yavg = (blocks[j][2] * blocks[j][3] + blocks[j + 1][2] * blocks[j + 1][3]) / wsum
            blocks[j] = [blocks[j][0], blocks[j + 1][1], yavg, wsum]
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


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(pd.Series(a).corr(pd.Series(b), method="pearson"))


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def sign_label(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def split_train_test(n_items: int, train_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_items)
    train_n = int(n_items * train_ratio)
    return idx[:train_n], idx[train_n:]


def save_scatter(x: np.ndarray, y: np.ndarray, x_label: str, y_label: str, title: str, output_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, s=8, alpha=0.35)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


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


def evaluate_encoder(
    name: str,
    spec: dict[str, Any],
    loader: DataLoader,
    ddpm_model: torch.nn.Module,
    noise_std_schedule: torch.Tensor,
    output_dir: Path,
    noise_repeats: int,
    device: torch.device,
) -> dict[str, Any]:
    model_dir = output_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)

    ae = load_autoencoder(
        kind=spec["kind"],
        checkpoint_path=spec["checkpoint_path"],
        encoder_path=spec["encoder_path"],
        device=device,
    )
    df = build_raw_score_dataset(
        loader=loader,
        ae=ae,
        ddpm_model=ddpm_model,
        noise_std_schedule=noise_std_schedule,
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
        noise_repeats=noise_repeats,
        device=device,
    )
    data_path = model_dir / "raw_scores.csv"
    df.to_csv(data_path, index=False)

    sigma = df["sigma"].to_numpy()
    raw_score = df["raw_score_sigma_mapped_t"].to_numpy()
    raw_pearson = pearson_corr(raw_score, sigma)
    raw_spearman = spearman_corr(raw_score, sigma)

    train_idx, test_idx = split_train_test(len(df), TRAIN_RATIO, SEED)
    iso = IsotonicRegressorNP.fit(raw_score[train_idx], sigma[train_idx], increasing=False)
    sigma_hat = iso.predict(raw_score[test_idx])
    sigma_test = sigma[test_idx]

    calibrated_pearson = pearson_corr(sigma_hat, sigma_test)
    calibrated_spearman = spearman_corr(sigma_hat, sigma_test)
    calibrated_mse = mse(sigma_hat, sigma_test)

    sigma_vs_score_path = model_dir / "sigma_vs_score_scatter.png"
    sigma_hat_path = model_dir / "sigma_hat_vs_sigma.png"
    save_scatter(
        x=sigma,
        y=raw_score,
        x_label="sigma",
        y_label="raw_score_sigma_mapped_t",
        title=f"{name}: sigma vs raw score",
        output_path=sigma_vs_score_path,
    )
    save_scatter(
        x=sigma_test,
        y=sigma_hat,
        x_label="sigma",
        y_label="sigma_hat",
        title=f"{name}: sigma_hat vs sigma (isotonic, decreasing)",
        output_path=sigma_hat_path,
    )

    metrics = {
        "encoder": name,
        "raw_pearson": raw_pearson,
        "raw_spearman": raw_spearman,
        "sign": sign_label(raw_pearson),
        "calibrated_pearson": calibrated_pearson,
        "calibrated_spearman": calibrated_spearman,
        "mse": calibrated_mse,
        "data_path": str(data_path),
        "sigma_vs_score_scatter": str(sigma_vs_score_path),
        "sigma_hat_vs_sigma": str(sigma_hat_path),
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def write_text_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    signs = summary_df["sign"].tolist()
    same_sign = len(set(signs)) == 1
    inversion_rows = summary_df[summary_df["sign"] == "negative"]
    best_calibrated = summary_df.sort_values("calibrated_spearman", ascending=False).iloc[0]
    strongest_raw = summary_df.reindex(summary_df["raw_spearman"].abs().sort_values(ascending=False).index).iloc[0]

    lines = [
        "# Quick Encoder Score Comparison",
        "",
        dataframe_to_markdown(summary_df),
        "",
        "## Short Summary",
        f"- Inversion is {'preserved for all encoders' if len(inversion_rows) == len(summary_df) else f'present for {len(inversion_rows)}/{len(summary_df)} encoders'}.",
        f"- Raw correlation sign is {'the same across encoders' if same_sign else 'not the same across encoders'}: {', '.join(signs)}.",
        f"- Strongest raw Spearman magnitude: {strongest_raw['encoder']} ({strongest_raw['raw_spearman']:.6f}).",
        f"- Isotonic calibrator works best by calibrated Spearman for: {best_calibrated['encoder']} ({best_calibrated['calibrated_spearman']:.6f}, MSE={best_calibrated['mse']:.6f}).",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    num_samples = args.num_samples
    if num_samples is None:
        num_samples = FAST_NUM_SAMPLES if args.fast_eval else FULL_NUM_SAMPLES
    noise_repeats = args.noise_repeats
    if noise_repeats is None:
        noise_repeats = FAST_NOISE_REPEATS if args.fast_eval else FULL_NOISE_REPEATS

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    loader = build_loader(num_samples=num_samples, batch_size=BATCH_SIZE)
    ddpm_model, noise_std_schedule = load_ddpm_model(DDPM_PATH, device)

    rows = []
    for name, spec in ENCODER_SPECS.items():
        print(f"evaluating={name}")
        rows.append(
            evaluate_encoder(
                name=name,
                spec=spec,
                loader=loader,
                ddpm_model=ddpm_model,
                noise_std_schedule=noise_std_schedule,
                output_dir=output_dir,
                noise_repeats=noise_repeats,
                device=device,
            )
        )

    summary_df = pd.DataFrame(rows)[
        [
            "encoder",
            "raw_pearson",
            "raw_spearman",
            "sign",
            "calibrated_pearson",
            "calibrated_spearman",
            "mse",
        ]
    ]
    csv_path = output_dir / "quick_summary.csv"
    md_path = output_dir / "quick_summary.md"
    summary_path = output_dir / "summary.md"
    summary_df.to_csv(csv_path, index=False)
    md_path.write_text(dataframe_to_markdown(summary_df) + "\n", encoding="utf-8")
    write_text_summary(summary_df, summary_path)

    run_config = {
        "fast_eval": args.fast_eval,
        "num_samples": num_samples,
        "noise_repeats": noise_repeats,
        "ddpm_path": str(DDPM_PATH),
        "sigma_min": SIGMA_MIN,
        "sigma_max": SIGMA_MAX,
        "train_ratio": TRAIN_RATIO,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print()
    print(dataframe_to_markdown(summary_df))
    print()
    print("plots:")
    for name in ENCODER_SPECS:
        print(f"- {name}: {output_dir / name / 'sigma_vs_score_scatter.png'}")
        print(f"- {name}: {output_dir / name / 'sigma_hat_vs_sigma.png'}")
    print()
    print(f"quick_summary_csv={csv_path}")
    print(f"quick_summary_md={md_path}")
    print(f"summary_md={summary_path}")


if __name__ == "__main__":
    main()
