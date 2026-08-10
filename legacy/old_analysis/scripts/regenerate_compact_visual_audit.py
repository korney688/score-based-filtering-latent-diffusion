from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.TDnCNN_image import DnCNN_2D


DPI = 190
BRANCHES = ["full", "topk_10", "quantile_spread_10", "random_10"]
CIFAR_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate compact visual-only audit panels.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=["mnist", "cifar10"], default=["cifar10", "mnist"])
    parser.add_argument("--no-train", action="store_true", help="Required guard. This script never trains.")
    return parser.parse_args()


def run_root(output_root: Path, dataset: str) -> Path:
    root = output_root / f"{dataset}_psnr10-40_s42" / "main"
    if not root.exists():
        raise FileNotFoundError(root)
    return root


def clean_dataset(dataset: str, train: bool):
    transform = transforms.ToTensor()
    data_root = PROJECT_ROOT / "data"
    if dataset == "mnist":
        return datasets.MNIST(data_root, train=train, download=False, transform=transform)
    return datasets.CIFAR10(data_root, train=train, download=False, transform=transform)


def class_name(dataset: str, class_id: int) -> str:
    return str(class_id) if dataset == "mnist" else CIFAR_CLASSES[int(class_id)]


def load_train_frames(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(root / "01_corpus" / "train_manifest.csv")
    scores = pd.read_csv(root / "03_filtering" / "scores_and_subsets.csv")
    merged = scores.merge(
        manifest[
            [
                "original_index",
                "cache_pos",
                "class_id",
                "target_psnr_db",
                "realized_psnr_db_postclip",
                "realized_snr_db_postclip",
                "sigma",
                "signal_power",
                "clipped_fraction",
            ]
        ],
        on="original_index",
        how="left",
        suffixes=("_score", ""),
    )
    if "class_id_score" in merged.columns:
        merged["class_id"] = merged["class_id"].fillna(merged["class_id_score"])
    if "target_psnr_db_score" in merged.columns:
        merged["target_psnr_db"] = merged["target_psnr_db"].fillna(merged["target_psnr_db_score"])
    return manifest, scores, merged


def subset_mask(df: pd.DataFrame, subset: str) -> pd.Series:
    if subset == "full":
        return pd.Series(True, index=df.index)
    return df[f"is_{subset}"].astype(bool)


def tensor_to_image(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().float().clamp(0.0, 1.0)
    if x.ndim == 4:
        x = x[0]
    if x.shape[0] == 1:
        return x[0].numpy()
    return x.permute(1, 2, 0).numpy()


def show_image(ax, x: torch.Tensor, dataset: str) -> None:
    arr = tensor_to_image(x)
    if arr.ndim == 2:
        ax.imshow(arr, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    else:
        ax.imshow(arr, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])


def save_figure(fig, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def write_cifar_filtering_distributions(root: Path) -> None:
    _, _, df = load_train_frames(root)
    audit_dir = root / "07_visual_audit"
    comparisons = [
        ("Top-K 10%", "topk_10"),
        ("Quantile 10%", "quantile_spread_10"),
        ("Random 10%", "random_10"),
    ]
    metrics = [
        ("realized_psnr_db_postclip", "realized PSNR, dB"),
        ("sigma", "sigma"),
    ]
    bins = {metric: np.linspace(float(df[metric].min()), float(df[metric].max()), 55) for metric, _ in metrics}

    fig, axes = plt.subplots(2, 3, figsize=(18.5, 8.2), dpi=DPI, constrained_layout=True)
    for row, (metric, ylabel) in enumerate(metrics):
        for col, (title, subset) in enumerate(comparisons):
            ax = axes[row, col]
            sub = df[subset_mask(df, subset)]
            ax.hist(df[metric], bins=bins[metric], density=True, alpha=0.35, label=f"Full n={len(df)}")
            ax.hist(sub[metric], bins=bins[metric], density=True, alpha=0.58, label=f"{title} n={len(sub)}")
            full_med = float(df[metric].median())
            sub_med = float(sub[metric].median())
            ax.axvline(full_med, color="tab:blue", linestyle="--", linewidth=1.8)
            ax.axvline(sub_med, color="tab:orange", linestyle="--", linewidth=1.8)
            ax.text(
                0.04,
                0.92,
                f"median delta {sub_med - full_med:+.3f}",
                transform=ax.transAxes,
                fontsize=12,
                va="top",
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2},
            )
            ax.set_xlim(float(df[metric].min()), float(df[metric].max()))
            ax.set_title(title if row == 0 else "", fontsize=15)
            ax.set_xlabel(ylabel, fontsize=12)
            ax.set_ylabel("Density", fontsize=12)
            ax.legend(fontsize=12)
            ax.tick_params(labelsize=11)
    save_figure(fig, audit_dir / "03_filtering_distributions_compact.png")


def write_cifar_score_examples(root: Path) -> None:
    dataset = "cifar10"
    train = clean_dataset(dataset, train=True)
    corpus = torch.load(root / "01_corpus" / "train_corpus.pt", map_location="cpu").float()
    _, _, df = load_train_frames(root)
    audit_dir = root / "07_visual_audit"
    groups = [
        ("lowest score", df.nsmallest(6, "score_value")),
        ("highest score", df.nlargest(6, "score_value")),
        ("highest realized PSNR", df.nlargest(6, "realized_psnr_db_postclip")),
        ("lowest realized PSNR", df.nsmallest(6, "realized_psnr_db_postclip")),
    ]
    fig, axes = plt.subplots(4, 12, figsize=(24, 12.5), dpi=DPI, constrained_layout=True)
    for row_idx, (group_name, rows) in enumerate(groups):
        for ex_idx, row in enumerate(rows.itertuples(index=False)):
            clean, _ = train[int(row.original_index)]
            corrupted = corpus[int(row.cache_pos)]
            clean_ax = axes[row_idx, ex_idx * 2]
            corpus_ax = axes[row_idx, ex_idx * 2 + 1]
            show_image(clean_ax, clean, dataset)
            show_image(corpus_ax, corrupted, dataset)
            clean_ax.set_title("Clean", fontsize=13)
            corpus_ax.set_title("Corpus", fontsize=13)
            clean_ax.set_ylabel(group_name, fontsize=15)
            corpus_ax.set_xlabel(
                f"{class_name(dataset, int(row.class_id))}\n"
                f"score {float(row.score_value):.3g}\n"
                f"PSNR {float(row.realized_psnr_db_postclip):.2f}\n"
                f"sigma {float(row.sigma):.3f}, clip {float(row.clipped_fraction):.2f}",
                fontsize=12,
            )
    save_figure(fig, audit_dir / "08_score_selection_examples_compact.png")


def load_model(path: Path, channels: int) -> DnCNN_2D:
    model = DnCNN_2D(in_channels=channels)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def infer_selected(root: Path, sample_indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    inputs = torch.load(root / "01_corpus" / "test_eval_input.pt", map_location="cpu").float()
    targets = torch.load(root / "01_corpus" / "test_eval_target.pt", map_location="cpu").float()
    selected_input = inputs[sample_indices]
    outputs: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for branch in BRANCHES:
            model = load_model(root / "04_downstream" / branch / "best.pt", int(inputs.shape[1]))
            outputs[branch] = model(selected_input).clamp(0.0, 1.0)
    return selected_input, targets[sample_indices], outputs


def metric_text(metrics: pd.DataFrame, subset: str, sample_index: int, dataset: str) -> str:
    row = metrics[(metrics["training_subset"] == subset) & (metrics["sample_index"] == sample_index)]
    if row.empty:
        return "metrics UNKNOWN"
    r = row.iloc[0]
    text = f"PSNR {float(r.psnr):.2f}\nSSIM {float(r.ssim):.3f}\nMSE {float(r.mse):.4f}"
    if dataset == "cifar10":
        text += f"\nLPIPS {float(r.lpips):.3f}"
    return text


def write_downstream_panel(root: Path, output_name: str, sample_indices: list[int], row_labels: list[str], dataset: str) -> None:
    audit_dir = root / "07_visual_audit"
    metrics = pd.read_csv(root / "05_evaluation" / "per_image_metrics.csv")
    inputs, targets, outputs = infer_selected(root, sample_indices)
    columns = ["Clean", "Noisy input", "Full", "Top-K", "Quantile", "Random"]
    branch_for_col: dict[int, str] = {2: "full", 3: "topk_10", 4: "quantile_spread_10", 5: "random_10"}

    fig, axes = plt.subplots(3, 6, figsize=(22, 9.5), dpi=DPI, constrained_layout=True)
    for r, sample_index in enumerate(sample_indices):
        show_image(axes[r, 0], targets[r], dataset)
        axes[r, 0].set_title(columns[0], fontsize=15)
        axes[r, 0].set_ylabel(f"{row_labels[r]}\nidx {sample_index}", fontsize=14)
        axes[r, 0].set_xlabel("target", fontsize=12)

        show_image(axes[r, 1], inputs[r], dataset)
        axes[r, 1].set_title(columns[1], fontsize=15)
        axes[r, 1].set_xlabel(metric_text(metrics, "noisy_input", sample_index, dataset), fontsize=12)

        for c in range(2, 6):
            branch = branch_for_col[c]
            show_image(axes[r, c], outputs[branch][r], dataset)
            axes[r, c].set_title(columns[c], fontsize=15)
            axes[r, c].set_xlabel(metric_text(metrics, branch, sample_index, dataset), fontsize=12)
    save_figure(fig, audit_dir / output_name)


def nearest_sample_by_quantile(df: pd.DataFrame, q: float) -> int:
    value = float(df["psnr"].quantile(q))
    idx = (df["psnr"] - value).abs().idxmin()
    return int(df.loc[idx, "sample_index"])


def write_cifar_downstream_panels(root: Path) -> None:
    metrics = pd.read_csv(root / "05_evaluation" / "per_image_metrics.csv")
    noisy = metrics[metrics["training_subset"] == "noisy_input"].copy()
    representative = [
        nearest_sample_by_quantile(noisy, 0.90),
        nearest_sample_by_quantile(noisy, 0.50),
        nearest_sample_by_quantile(noisy, 0.10),
    ]
    write_downstream_panel(
        root,
        "11_downstream_representative_examples.png",
        representative,
        ["easy test", "median test", "hard test"],
        "cifar10",
    )

    pivot = metrics.pivot(index="sample_index", columns="training_subset", values="psnr")
    delta = pivot["topk_10"] - pivot["full"]
    extremes = [int(delta.idxmax()), int((delta - 0.0).abs().idxmin()), int(delta.idxmin())]
    labels = [
        f"max delta\n{float(delta.loc[extremes[0]]):+.2f} dB",
        f"near zero\n{float(delta.loc[extremes[1]]):+.2f} dB",
        f"min delta\n{float(delta.loc[extremes[2]]):+.2f} dB",
    ]
    write_downstream_panel(root, "12_topk_vs_full_extremes.png", extremes, labels, "cifar10")


def mnist_foreground_fraction(train) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx in range(len(train)):
        image, _ = train[idx]
        rows.append({"original_index": idx, "foreground_fraction": float((image > 0.1).float().mean().item())})
    return pd.DataFrame(rows)


def write_mnist_score_quality_compact(root: Path) -> None:
    dataset = "mnist"
    train = clean_dataset(dataset, train=True)
    _, _, df = load_train_frames(root)
    df = df.merge(mnist_foreground_fraction(train), on="original_index", how="left")
    y_cols = [
        ("target_psnr_db", "target PSNR"),
        ("realized_psnr_db_postclip", "realized PSNR"),
        ("sigma", "sigma"),
        ("signal_power", "signal power"),
        ("foreground_fraction", "foreground fraction"),
        ("clipped_fraction", "clipped fraction"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 9.5), dpi=DPI, constrained_layout=True)
    for ax, (col, label) in zip(axes.flatten(), y_cols):
        hb = ax.hexbin(df["score_value"], df[col], gridsize=60, mincnt=1, bins="log", cmap="viridis")
        pear = pearsonr(df["score_value"], df[col])
        spear = spearmanr(df["score_value"], df[col])
        ax.set_title(
            f"{label}\nPearson {pear.statistic:.3f}, Spearman {spear.statistic:.3f}",
            fontsize=15,
        )
        ax.set_xlabel("score_value (Top-K uses minimum)", fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.tick_params(labelsize=11)
        cbar = fig.colorbar(hb, ax=ax)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label("log count", fontsize=12)
    save_figure(fig, root / "07_visual_audit" / "05_score_vs_quality_compact.png")


def main() -> None:
    args = parse_args()
    if not args.no_train:
        raise RuntimeError("Pass --no-train. This script is visual-only and never trains.")
    output_root = args.output_root if args.output_root.is_absolute() else PROJECT_ROOT / args.output_root
    if "cifar10" in args.datasets:
        cifar_root = run_root(output_root, "cifar10")
        write_cifar_filtering_distributions(cifar_root)
        write_cifar_score_examples(cifar_root)
        write_cifar_downstream_panels(cifar_root)
    if "mnist" in args.datasets:
        write_mnist_score_quality_compact(run_root(output_root, "mnist"))


if __name__ == "__main__":
    main()
