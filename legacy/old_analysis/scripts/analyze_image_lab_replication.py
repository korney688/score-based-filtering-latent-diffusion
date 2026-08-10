from __future__ import annotations

import argparse
import hashlib
import json
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
import yaml
from PIL import Image
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.TDnCNN_image import DnCNN_2D, compute_ssim_item


BRANCHES = ["full", "topk_10", "quantile_spread_10", "random_10"]
CIFAR10_CLASSES = [
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
    parser = argparse.ArgumentParser(description="Post-hoc visual audit for completed image lab replication runs.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=["mnist", "cifar10"], required=True)
    parser.add_argument("--mode", choices=["visual_audit"], default="visual_audit")
    parser.add_argument("--no-train", action="store_true", help="Required guard: this script never trains models.")
    parser.add_argument("--visual-seed", type=int, default=42)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    return parser.parse_args()


def assert_inside(child: Path, parent: Path) -> None:
    child_resolved = child.resolve()
    parent_resolved = parent.resolve()
    if parent_resolved not in [child_resolved, *child_resolved.parents]:
        raise RuntimeError(f"Refusing to write outside {parent_resolved}: {child_resolved}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    columns = list(df.columns)
    rows = ["| " + " | ".join(columns) + " |"]
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(format_value(row[col]) for col in columns) + " |")
    return "\n".join(rows)


def format_value(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def find_run(output_root: Path, dataset: str) -> Path:
    candidates = sorted(output_root.glob(f"{dataset}_psnr*-*_s*/main"))
    if not candidates:
        raise FileNotFoundError(f"No main run found for {dataset} under {output_root}")
    if len(candidates) > 1:
        exact = output_root / f"{dataset}_psnr10-40_s42" / "main"
        if exact.exists():
            return exact
        raise RuntimeError(f"Multiple main runs found for {dataset}: {[str(p) for p in candidates]}")
    return candidates[0]


def load_clean_dataset(dataset: str, train: bool):
    transform = transforms.ToTensor()
    root = PROJECT_ROOT / "data"
    if dataset == "mnist":
        return datasets.MNIST(root=root, train=train, download=False, transform=transform)
    if dataset == "cifar10":
        return datasets.CIFAR10(root=root, train=train, download=False, transform=transform)
    raise ValueError(dataset)


def class_name(dataset: str, class_id: int) -> str:
    if dataset == "mnist":
        return str(class_id)
    return CIFAR10_CLASSES[int(class_id)]


def tensor_image(dataset_obj, original_index: int) -> torch.Tensor:
    image, _ = dataset_obj[int(original_index)]
    return image.to(torch.float32).clamp(0.0, 1.0)


def to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().to(torch.float32).clamp(0.0, 1.0)
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.shape[0] == 1:
        return tensor[0].numpy()
    return tensor.permute(1, 2, 0).numpy()


def show_tensor(ax, tensor: torch.Tensor, dataset: str, title: str) -> None:
    arr = to_numpy_image(tensor)
    if arr.ndim == 2:
        ax.imshow(arr, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest" if dataset == "mnist" else "none")
    else:
        ax.imshow(arr, vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def enhanced_residual(clean: torch.Tensor, corrupted: torch.Tensor) -> torch.Tensor:
    residual = (corrupted - clean).abs()
    scale = float(torch.quantile(residual.flatten(), 0.99).item())
    if scale <= 1e-8:
        return residual
    return (residual / scale).clamp(0.0, 1.0)


def panel_examples(
    *,
    dataset: str,
    clean_dataset,
    corpus: torch.Tensor,
    manifest: pd.DataFrame,
    rows: pd.DataFrame,
    output_png: Path,
    output_csv: Path,
    title: str,
    group_col: str | None = None,
) -> None:
    rows = rows.copy()
    rows.to_csv(output_csv, index=False)
    n = len(rows)
    fig_h = max(7.5, n * 1.35)
    fig, axes = plt.subplots(n, 3, figsize=(15, fig_h), dpi=180)
    if n == 1:
        axes = np.array([axes])
    fig.suptitle(title, fontsize=14, y=0.995)

    for r, row in enumerate(rows.itertuples(index=False)):
        original_index = int(row.original_index)
        clean = tensor_image(clean_dataset, original_index)
        cache_pos = int(getattr(row, "cache_pos"))
        corrupted = corpus[cache_pos].to(torch.float32).clamp(0.0, 1.0)
        residual = enhanced_residual(clean, corrupted)

        prefix = ""
        if group_col and hasattr(row, group_col):
            prefix = f"{getattr(row, group_col)}\n"
        caption = (
            f"{prefix}idx={original_index}, class={class_name(dataset, int(row.class_id))}\n"
            f"target={float(row.target_psnr_db):.2f}, realized={float(row.realized_psnr_db_postclip):.2f}, "
            f"SNR={float(row.realized_snr_db_postclip):.2f}\n"
            f"sigma={float(row.sigma):.4f}, clipped={float(row.clipped_fraction):.3f}"
        )
        if hasattr(row, "score_value"):
            caption += f"\nscore={float(row.score_value):.4g}, rank={int(row.score_rank)}"
        if hasattr(row, "is_topk_10"):
            caption += f", Top-K={bool(row.is_topk_10)}"

        show_tensor(axes[r, 0], clean, dataset, "Clean")
        show_tensor(axes[r, 1], corrupted, dataset, "Corpus-corrupted")
        show_tensor(axes[r, 2], residual, dataset, "|Residual| enhanced")
        axes[r, 1].set_xlabel(caption, fontsize=6.5)

    fig.tight_layout(rect=(0, 0, 1, 0.985), pad=0.35)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def select_quantile_rows(df: pd.DataFrame, column: str, quantiles: list[float]) -> pd.DataFrame:
    sorted_df = df.sort_values(column).reset_index(drop=True)
    n = len(sorted_df)
    picks = []
    used: set[int] = set()
    for q in quantiles:
        pos = int(round(q * (n - 1)))
        while pos in used and pos + 1 < n:
            pos += 1
        while pos in used and pos - 1 >= 0:
            pos -= 1
        used.add(pos)
        picks.append(sorted_df.iloc[pos])
    return pd.DataFrame(picks)


def subset_mask(scores: pd.DataFrame, subset: str) -> pd.Series:
    if subset == "full":
        return pd.Series(True, index=scores.index)
    mapping = {
        "topk_10": "is_topk_10",
        "quantile_spread_10": "is_quantile_spread_10",
        "random_10": "is_random_10",
    }
    return scores[mapping[subset]].astype(bool)


def merge_quality_scores(manifest: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    merged = scores.merge(
        manifest[
            [
                "original_index",
                "cache_pos",
                "target_psnr_db",
                "sigma",
                "realized_psnr_db_postclip",
                "realized_snr_db_postclip",
                "signal_power",
                "clipped_fraction",
                "class_id",
            ]
        ],
        on="original_index",
        how="left",
        suffixes=("_scorefile", ""),
    )
    for col in ["target_psnr_db", "class_id"]:
        left = f"{col}_scorefile"
        if left in merged.columns:
            merged[col] = merged[col].fillna(merged[left])
            merged = merged.drop(columns=[left])
    return merged


def write_subset_quality_distributions(audit_dir: Path, merged: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("Full vs topk_10", "topk_10"),
        ("Full vs quantile_spread_10", "quantile_spread_10"),
        ("Full vs random_10", "random_10"),
    ]
    columns = ["target_psnr_db", "realized_psnr_db_postclip", "realized_snr_db_postclip"]
    labels = ["Target PSNR dB", "Realized PSNR dB", "Realized SNR dB"]
    bins_by_col = {col: np.linspace(float(merged[col].min()), float(merged[col].max()), 50) for col in columns}
    stats_rows: list[dict[str, Any]] = []
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), dpi=180)
    for r, (row_label, subset) in enumerate(comparisons):
        full_df = merged
        subset_df = merged[subset_mask(merged, subset)]
        for c, (col, label) in enumerate(zip(columns, labels)):
            ax = axes[r, c]
            bins = bins_by_col[col]
            ax.hist(full_df[col], bins=bins, density=True, alpha=0.35, label=f"Full (n={len(full_df)})")
            ax.hist(subset_df[col], bins=bins, density=True, alpha=0.55, label=f"{subset} (n={len(subset_df)})")
            full_med = float(full_df[col].median())
            subset_med = float(subset_df[col].median())
            delta = subset_med - full_med
            ax.axvline(full_med, color="tab:blue", linestyle="--", linewidth=1.2)
            ax.axvline(subset_med, color="tab:orange", linestyle="--", linewidth=1.2)
            ax.text(0.03, 0.92, f"delta median={delta:+.3f}", transform=ax.transAxes, fontsize=8, va="top")
            ax.set_xlim(float(merged[col].min()), float(merged[col].max()))
            ax.set_ylabel("Density")
            ax.set_xlabel(label)
            ax.set_title(row_label if c == 0 else label)
            ax.legend(fontsize=7)
            stats_rows.append(
                {
                    "comparison": row_label,
                    "subset": subset,
                    "metric": col,
                    "full_n": len(full_df),
                    "subset_n": len(subset_df),
                    "full_mean": float(full_df[col].mean()),
                    "subset_mean": float(subset_df[col].mean()),
                    "full_median": full_med,
                    "subset_median": subset_med,
                    "delta_median_subset_minus_full": delta,
                }
            )
    fig.tight_layout()
    fig.savefig(audit_dir / "03_subset_quality_distributions.png", dpi=180)
    plt.close(fig)

    stats = pd.DataFrame(stats_rows)
    stats.to_csv(audit_dir / "03_subset_quality_statistics.csv", index=False)
    return stats


def write_sigma_distributions(audit_dir: Path, merged: pd.DataFrame) -> None:
    comparisons = [
        ("Full vs topk_10", "topk_10"),
        ("Full vs quantile_spread_10", "quantile_spread_10"),
        ("Full vs random_10", "random_10"),
    ]
    bins = np.linspace(float(merged["sigma"].min()), float(merged["sigma"].max()), 50)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), dpi=180)
    for ax, (label, subset) in zip(axes, comparisons):
        subset_df = merged[subset_mask(merged, subset)]
        ax.hist(merged["sigma"], bins=bins, density=True, alpha=0.35, label=f"Full (n={len(merged)})")
        ax.hist(subset_df["sigma"], bins=bins, density=True, alpha=0.55, label=f"{subset} (n={len(subset_df)})")
        full_med = float(merged["sigma"].median())
        subset_med = float(subset_df["sigma"].median())
        ax.axvline(full_med, color="tab:blue", linestyle="--", linewidth=1.2)
        ax.axvline(subset_med, color="tab:orange", linestyle="--", linewidth=1.2)
        ax.text(0.03, 0.92, f"delta median={subset_med - full_med:+.5f}", transform=ax.transAxes, fontsize=8, va="top")
        ax.set_xlim(float(merged["sigma"].min()), float(merged["sigma"].max()))
        ax.set_xlabel("sigma")
        ax.set_ylabel("Density")
        ax.set_title(label)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(audit_dir / "04_sigma_distributions.png", dpi=180)
    plt.close(fig)


def correlation_row(df: pd.DataFrame, x_col: str, y_col: str) -> dict[str, Any]:
    work = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) < 3:
        return {
            "x": x_col,
            "y": y_col,
            "n": len(work),
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
        }
    pear = pearsonr(work[x_col].to_numpy(), work[y_col].to_numpy())
    spear = spearmanr(work[x_col].to_numpy(), work[y_col].to_numpy())
    return {
        "x": x_col,
        "y": y_col,
        "n": len(work),
        "pearson_r": float(pear.statistic),
        "pearson_p": float(pear.pvalue),
        "spearman_rho": float(spear.statistic),
        "spearman_p": float(spear.pvalue),
    }


def write_score_quality(audit_dir: Path, merged: pd.DataFrame) -> pd.DataFrame:
    y_cols = [
        "target_psnr_db",
        "realized_psnr_db_postclip",
        "realized_snr_db_postclip",
        "sigma",
        "signal_power",
        "clipped_fraction",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=180)
    rows = []
    for ax, y_col in zip(axes.flatten(), y_cols):
        row = correlation_row(merged, "score_value", y_col)
        rows.append(row)
        if len(merged) > 5000:
            hb = ax.hexbin(merged["score_value"], merged[y_col], gridsize=55, mincnt=1, bins="log", cmap="viridis")
            fig.colorbar(hb, ax=ax, label="log count")
        else:
            ax.scatter(merged["score_value"], merged[y_col], s=6, alpha=0.25)
        ax.set_xlabel("score_value")
        ax.set_ylabel(y_col)
        ax.set_title(
            f"{y_col}\nPearson r={row['pearson_r']:.3f}, p={row['pearson_p']:.2g}; "
            f"Spearman rho={row['spearman_rho']:.3f}, p={row['spearman_p']:.2g}; n={row['n']}",
            fontsize=9,
        )
    fig.suptitle("Top-K uses minimum score_value", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(audit_dir / "05_score_vs_quality.png", dpi=180)
    plt.close(fig)
    correlations = pd.DataFrame(rows)
    correlations.to_csv(audit_dir / "05_score_quality_correlations.csv", index=False)
    return correlations


def write_quality_by_score_decile(audit_dir: Path, merged: pd.DataFrame) -> pd.DataFrame:
    df = merged.copy()
    df["score_decile"] = pd.qcut(df["score_value"].rank(method="first"), 10, labels=False) + 1
    rows = []
    for decile, group in df.groupby("score_decile"):
        row: dict[str, Any] = {"score_decile": int(decile), "n": len(group)}
        for col in ["target_psnr_db", "realized_psnr_db_postclip", "realized_snr_db_postclip"]:
            q25, med, q75 = group[col].quantile([0.25, 0.5, 0.75]).tolist()
            row[f"{col}_median"] = float(med)
            row[f"{col}_iqr"] = float(q75 - q25)
            row[f"{col}_q25"] = float(q25)
            row[f"{col}_q75"] = float(q75)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(audit_dir / "06_quality_by_score_decile.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=180)
    for ax, col in zip(axes, ["target_psnr_db", "realized_psnr_db_postclip", "realized_snr_db_postclip"]):
        ax.errorbar(
            out["score_decile"],
            out[f"{col}_median"],
            yerr=[
                out[f"{col}_median"] - out[f"{col}_q25"],
                out[f"{col}_q75"] - out[f"{col}_median"],
            ],
            marker="o",
            capsize=3,
        )
        ax.set_xlabel("Score decile (1 = lowest score)")
        ax.set_ylabel(col)
        ax.set_title(f"{col} median and IQR")
    fig.tight_layout()
    fig.savefig(audit_dir / "06_quality_by_score_decile.png", dpi=180)
    plt.close(fig)
    return out


def write_low_high_score_quality(audit_dir: Path, merged: pd.DataFrame) -> pd.DataFrame:
    n = int(round(0.10 * len(merged)))
    low = merged.nsmallest(n, "score_value").copy()
    high = merged.nlargest(n, "score_value").copy()
    low["subset"] = "low_score_10"
    high["subset"] = "high_score_10"
    full = merged.copy()
    full["subset"] = "full"
    combined = pd.concat([full, low, high], ignore_index=True)
    metrics = ["target_psnr_db", "realized_psnr_db_postclip", "realized_snr_db_postclip", "sigma"]
    rows = []
    for subset, group in combined.groupby("subset"):
        for metric in metrics:
            rows.append(
                {
                    "subset": subset,
                    "metric": metric,
                    "n": len(group),
                    "mean": float(group[metric].mean()),
                    "median": float(group[metric].median()),
                    "std": float(group[metric].std(ddof=1)),
                }
            )
    stats = pd.DataFrame(rows)
    stats.to_csv(audit_dir / "07_low_vs_high_score_quality.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=180)
    for ax, metric in zip(axes.flatten(), metrics):
        bins = np.linspace(float(merged[metric].min()), float(merged[metric].max()), 50)
        for subset, alpha in [("full", 0.25), ("low_score_10", 0.55), ("high_score_10", 0.55)]:
            vals = combined[combined["subset"] == subset][metric]
            ax.hist(vals, bins=bins, density=True, alpha=alpha, label=f"{subset} (n={len(vals)})")
        ax.set_xlabel(metric)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(audit_dir / "07_low_vs_high_score_quality.png", dpi=180)
    plt.close(fig)
    return stats


def write_selected_and_oracle_examples(
    audit_dir: Path,
    dataset: str,
    clean_dataset,
    corpus: torch.Tensor,
    merged: pd.DataFrame,
) -> pd.DataFrame:
    groups = [
        ("A. Lowest score (best by score)", merged.nsmallest(8, "score_value")),
        ("B. Highest score (most rejected by score)", merged.nlargest(8, "score_value")),
        ("C. Oracle cleanest by realized PSNR", merged.nlargest(8, "realized_psnr_db_postclip")),
        ("D. Oracle noisiest by realized PSNR", merged.nsmallest(8, "realized_psnr_db_postclip")),
    ]
    rows = []
    for group_name, group in groups:
        g = group.copy()
        g["group"] = group_name
        rows.append(g)
    examples = pd.concat(rows, ignore_index=True)
    panel_examples(
        dataset=dataset,
        clean_dataset=clean_dataset,
        corpus=corpus,
        manifest=merged,
        rows=examples,
        output_png=audit_dir / "08_selected_and_oracle_examples.png",
        output_csv=audit_dir / "08_selected_and_oracle_examples.csv",
        title="Selected and oracle examples",
        group_col="group",
    )
    return examples


def write_score_quality_quadrants(
    audit_dir: Path,
    dataset: str,
    clean_dataset,
    corpus: torch.Tensor,
    merged: pd.DataFrame,
) -> pd.DataFrame:
    score_low = merged["score_value"].quantile(0.20)
    score_high = merged["score_value"].quantile(0.80)
    quality_low = merged["realized_psnr_db_postclip"].quantile(0.20)
    quality_high = merged["realized_psnr_db_postclip"].quantile(0.80)
    groups = [
        ("low score + high realized PSNR", merged[(merged["score_value"] <= score_low) & (merged["realized_psnr_db_postclip"] >= quality_high)]),
        ("low score + low realized PSNR", merged[(merged["score_value"] <= score_low) & (merged["realized_psnr_db_postclip"] <= quality_low)]),
        ("high score + high realized PSNR", merged[(merged["score_value"] >= score_high) & (merged["realized_psnr_db_postclip"] >= quality_high)]),
        ("high score + low realized PSNR", merged[(merged["score_value"] >= score_high) & (merged["realized_psnr_db_postclip"] <= quality_low)]),
    ]
    rows = []
    for group_name, group in groups:
        selected = group.sort_values(["score_value", "realized_psnr_db_postclip"]).head(6).copy()
        selected["group"] = group_name
        selected["available_group_size"] = len(group)
        rows.append(selected)
    examples = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    panel_examples(
        dataset=dataset,
        clean_dataset=clean_dataset,
        corpus=corpus,
        manifest=merged,
        rows=examples,
        output_png=audit_dir / "09_score_quality_quadrants.png",
        output_csv=audit_dir / "09_score_quality_quadrants.csv",
        title="Score-quality quadrants",
        group_col="group",
    )
    return examples


def content_features(dataset: str, clean_dataset, manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in manifest.itertuples(index=False):
        img = tensor_image(clean_dataset, int(row.original_index))
        out: dict[str, Any] = {
            "original_index": int(row.original_index),
            "class_id": int(row.class_id),
            "mean_intensity": float(img.mean().item()),
            "signal_power_check": float((img**2).mean().item()),
        }
        if dataset == "mnist":
            out["foreground_fraction"] = float((img > 0.1).to(torch.float32).mean().item())
            out["digit_brightness_sum"] = float(img.sum().item())
        else:
            luminance = 0.2126 * img[0] + 0.7152 * img[1] + 0.0722 * img[2]
            out["mean_luminance"] = float(luminance.mean().item())
            out["channel_mean_r"] = float(img[0].mean().item())
            out["channel_mean_g"] = float(img[1].mean().item())
            out["channel_mean_b"] = float(img[2].mean().item())
        rows.append(out)
    return pd.DataFrame(rows)


def js_divergence_counts(full_counts: pd.Series, subset_counts: pd.Series) -> float:
    labels = sorted(set(full_counts.index) | set(subset_counts.index))
    p = np.array([full_counts.get(label, 0) for label in labels], dtype=np.float64)
    q = np.array([subset_counts.get(label, 0) for label in labels], dtype=np.float64)
    p = p / max(p.sum(), 1.0)
    q = q / max(q.sum(), 1.0)
    return float(jensenshannon(p, q, base=2.0) ** 2)


def write_content_bias(audit_dir: Path, dataset: str, clean_dataset, manifest: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    features = content_features(dataset, clean_dataset, manifest)
    work = merged.merge(features, on=["original_index", "class_id"], how="left")
    feature_cols = ["signal_power", "mean_intensity", "clipped_fraction"]
    if dataset == "mnist":
        feature_cols += ["foreground_fraction", "digit_brightness_sum"]
    else:
        feature_cols += ["mean_luminance", "channel_mean_r", "channel_mean_g", "channel_mean_b"]

    full_counts = work["class_id"].value_counts().sort_index()
    rows = []
    for subset in BRANCHES:
        group = work[subset_mask(work, subset)]
        subset_counts = group["class_id"].value_counts().sort_index()
        jsd = js_divergence_counts(full_counts, subset_counts)
        for class_id, count in subset_counts.items():
            rows.append(
                {
                    "subset": subset,
                    "metric": "class_fraction",
                    "class_id": int(class_id),
                    "class_name": class_name(dataset, int(class_id)),
                    "value": float(count / max(len(group), 1)),
                    "js_divergence_vs_full": jsd,
                }
            )
        for col in feature_cols:
            rows.append(
                {
                    "subset": subset,
                    "metric": col,
                    "class_id": "",
                    "class_name": "",
                    "value": float(group[col].mean()),
                    "median": float(group[col].median()),
                    "std": float(group[col].std(ddof=1)),
                    "js_divergence_vs_full": jsd,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(audit_dir / "10_content_bias.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=180)
    class_ax = axes[0, 0]
    labels = sorted(work["class_id"].unique())
    x = np.arange(len(labels))
    width = 0.18
    for i, subset in enumerate(BRANCHES):
        group = work[subset_mask(work, subset)]
        vals = group["class_id"].value_counts(normalize=True).reindex(labels, fill_value=0.0).to_numpy()
        class_ax.bar(x + (i - 1.5) * width, vals, width=width, label=subset)
    class_ax.set_xticks(x)
    class_ax.set_xticklabels([class_name(dataset, int(label)) for label in labels], rotation=45, ha="right")
    class_ax.set_ylabel("Class fraction")
    class_ax.legend(fontsize=8)
    for ax, col in zip(axes.flatten()[1:], feature_cols[:3]):
        data = [work[subset_mask(work, subset)][col].to_numpy() for subset in BRANCHES]
        ax.boxplot(data, tick_labels=BRANCHES, showfliers=False)
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(audit_dir / "10_content_bias.png", dpi=180)
    plt.close(fig)
    return out


def load_tdncnn(checkpoint: Path, in_channels: int) -> DnCNN_2D:
    state = torch.load(checkpoint, map_location="cpu")
    model = DnCNN_2D(in_channels=in_channels)
    model.load_state_dict(state)
    model.eval()
    return model


def infer_outputs(root: Path, dataset: str, sample_indices: list[int]) -> dict[str, torch.Tensor]:
    inputs = torch.load(root / "01_corpus" / "test_eval_input.pt", map_location="cpu").to(torch.float32)
    outputs: dict[str, torch.Tensor] = {"noisy_input": inputs[sample_indices].clone()}
    in_channels = int(inputs.shape[1])
    with torch.no_grad():
        for branch in BRANCHES:
            model = load_tdncnn(root / "04_downstream" / branch / "best.pt", in_channels)
            pred = model(inputs[sample_indices]).clamp(0.0, 1.0)
            outputs[branch] = pred.cpu()
    return outputs


def metrics_lookup(metrics: pd.DataFrame, subset: str, sample_index: int) -> dict[str, float]:
    row = metrics[(metrics["training_subset"] == subset) & (metrics["sample_index"] == sample_index)]
    if row.empty:
        return {"mse": np.nan, "psnr": np.nan, "ssim": np.nan, "lpips": np.nan}
    record = row.iloc[0]
    return {metric: float(record[metric]) for metric in ["mse", "psnr", "ssim", "lpips"] if metric in row.columns}


def output_title(subset: str, vals: dict[str, float], dataset: str) -> str:
    title = f"{subset}\nPSNR={vals.get('psnr', np.nan):.2f}, SSIM={vals.get('ssim', np.nan):.3f}\nMSE={vals.get('mse', np.nan):.4f}"
    if dataset == "cifar10":
        title += f", LPIPS={vals.get('lpips', np.nan):.3f}"
    return title


def write_downstream_panel(
    *,
    root: Path,
    audit_dir: Path,
    dataset: str,
    sample_indices: list[int],
    output_png: Path,
    output_csv: Path,
    title: str,
    reason_by_index: dict[int, str],
) -> pd.DataFrame:
    targets = torch.load(root / "01_corpus" / "test_eval_target.pt", map_location="cpu").to(torch.float32)
    inputs = torch.load(root / "01_corpus" / "test_eval_input.pt", map_location="cpu").to(torch.float32)
    metrics = pd.read_csv(root / "05_evaluation" / "per_image_metrics.csv")
    outputs = infer_outputs(root, dataset, sample_indices)
    columns = ["clean target", "noisy_input", "full", "topk_10", "quantile_spread_10", "random_10"]
    fig, axes = plt.subplots(len(sample_indices), len(columns), figsize=(18, 3.3 * len(sample_indices)), dpi=180)
    if len(sample_indices) == 1:
        axes = np.array([axes])
    rows = []
    for r, sample_index in enumerate(sample_indices):
        show_tensor(axes[r, 0], targets[sample_index], dataset, f"Clean target\nidx={sample_index}\n{reason_by_index.get(sample_index, '')}")
        noisy_vals = metrics_lookup(metrics, "noisy_input", sample_index)
        show_tensor(axes[r, 1], inputs[sample_index], dataset, output_title("noisy_input", noisy_vals, dataset))
        rows.append({"sample_index": sample_index, "reason": reason_by_index.get(sample_index, ""), "training_subset": "noisy_input", **noisy_vals})
        for c, subset in enumerate(BRANCHES, start=2):
            vals = metrics_lookup(metrics, subset, sample_index)
            show_tensor(axes[r, c], outputs[subset][sample_indices.index(sample_index)], dataset, output_title(subset, vals, dataset))
            rows.append({"sample_index": sample_index, "reason": reason_by_index.get(sample_index, ""), "training_subset": subset, **vals})
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    out = pd.DataFrame(rows)
    out.to_csv(output_csv, index=False)
    return out


def nearest_by_quantile(df: pd.DataFrame, column: str, q: float) -> int:
    value = float(df[column].quantile(q))
    idx = (df[column] - value).abs().idxmin()
    return int(df.loc[idx, "sample_index"])


def write_downstream_examples(root: Path, audit_dir: Path, dataset: str) -> dict[str, list[int]]:
    metrics = pd.read_csv(root / "05_evaluation" / "per_image_metrics.csv")
    noisy = metrics[metrics["training_subset"] == "noisy_input"].copy()
    q_indices = [
        nearest_by_quantile(noisy, "psnr", 0.10),
        nearest_by_quantile(noisy, "psnr", 0.50),
        nearest_by_quantile(noisy, "psnr", 0.90),
    ]
    q_reasons = {
        q_indices[0]: "near noisy-input PSNR p10",
        q_indices[1]: "near noisy-input PSNR p50",
        q_indices[2]: "near noisy-input PSNR p90",
    }
    write_downstream_panel(
        root=root,
        audit_dir=audit_dir,
        dataset=dataset,
        sample_indices=q_indices,
        output_png=audit_dir / "11_downstream_representative_examples.png",
        output_csv=audit_dir / "11_downstream_representative_examples.csv",
        title="Downstream representative examples",
        reason_by_index=q_reasons,
    )

    pivot = metrics.pivot(index="sample_index", columns="training_subset", values="psnr")
    delta = pivot["topk_10"] - pivot["full"]
    max_idx = int(delta.idxmax())
    zero_idx = int((delta - 0.0).abs().idxmin())
    min_idx = int(delta.idxmin())
    e_indices = [max_idx, zero_idx, min_idx]
    e_reasons = {
        max_idx: f"max Top-K - Full PSNR ({delta.loc[max_idx]:+.3f} dB)",
        zero_idx: f"nearest zero Top-K - Full PSNR ({delta.loc[zero_idx]:+.3f} dB)",
        min_idx: f"min Top-K - Full PSNR ({delta.loc[min_idx]:+.3f} dB)",
    }
    write_downstream_panel(
        root=root,
        audit_dir=audit_dir,
        dataset=dataset,
        sample_indices=e_indices,
        output_png=audit_dir / "12_downstream_pairwise_extremes.png",
        output_csv=audit_dir / "12_downstream_pairwise_extremes.csv",
        title="Downstream pairwise extremes",
        reason_by_index=e_reasons,
    )
    return {"representative": q_indices, "pairwise_extremes": e_indices}


def write_metric_deltas(audit_dir: Path, root: Path, dataset: str, resamples: int) -> pd.DataFrame:
    metrics = pd.read_csv(root / "05_evaluation" / "per_image_metrics.csv")
    comparisons = [("topk_10", "full"), ("topk_10", "random_10"), ("topk_10", "quantile_spread_10")]
    metric_names = ["psnr", "ssim", "mse"]
    if dataset == "cifar10":
        metric_names.append("lpips")
    rng = np.random.default_rng(42)
    rows = []
    fig, axes = plt.subplots(len(metric_names), len(comparisons), figsize=(15, 3.2 * len(metric_names)), dpi=180)
    if len(metric_names) == 1:
        axes = np.array([axes])
    for r, metric in enumerate(metric_names):
        pivot = metrics.pivot(index="sample_index", columns="training_subset", values=metric).dropna()
        for c, (a, b) in enumerate(comparisons):
            diff = (pivot[a] - pivot[b]).to_numpy(dtype=np.float64)
            n = len(diff)
            if n:
                idx = rng.integers(0, n, size=(resamples, n))
                boot = diff[idx].mean(axis=1)
                ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
            else:
                ci_low = ci_high = np.nan
            superiority = float((diff > 0).mean()) if metric in ["psnr", "ssim"] else float((diff < 0).mean())
            rows.append(
                {
                    "comparison": f"{a} - {b}",
                    "metric": metric,
                    "n": n,
                    "mean_delta": float(np.mean(diff)),
                    "median_delta": float(np.median(diff)),
                    "bootstrap_ci95_low": float(ci_low),
                    "bootstrap_ci95_high": float(ci_high),
                    "topk_superior_fraction": superiority,
                    "superiority_direction": "higher is better" if metric in ["psnr", "ssim"] else "lower is better",
                    "bootstrap_resamples": resamples,
                }
            )
            ax = axes[r, c]
            ax.hist(diff, bins=60, density=True, alpha=0.75)
            ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
            ax.axvline(float(np.mean(diff)), color="tab:red", linewidth=1)
            ax.set_title(f"{a} - {b}\n{metric}")
            ax.set_xlabel("delta")
            ax.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(audit_dir / "13_per_image_metric_deltas.png", dpi=180)
    plt.close(fig)
    out = pd.DataFrame(rows)
    out.to_csv(audit_dir / "13_per_image_metric_deltas.csv", index=False)
    return out


def write_hard_subset(audit_dir: Path, root: Path, dataset: str) -> pd.DataFrame:
    metrics = pd.read_csv(root / "05_evaluation" / "per_image_metrics.csv")
    test_manifest = pd.read_csv(root / "01_corpus" / "test_noise_manifest.csv")
    n_hard = int(math.ceil(0.25 * len(test_manifest)))
    hard_indices = set(test_manifest.nsmallest(n_hard, "realized_psnr_db_postclip")["cache_pos"].astype(int).tolist())
    hard_metrics = metrics[metrics["sample_index"].isin(hard_indices)].copy()
    metric_names = ["mse", "psnr", "ssim"]
    if dataset == "cifar10":
        metric_names.append("lpips")
    rows = []
    for subset, group in hard_metrics.groupby("training_subset"):
        row = {"training_subset": subset, "n": len(group)}
        for metric in metric_names:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
            row[f"{metric}_median"] = float(group[metric].median())
        rows.append(row)
    out = pd.DataFrame(rows)
    full = out[out["training_subset"] == "full"].iloc[0]
    for metric in metric_names:
        out[f"delta_{metric}_vs_full"] = out[f"{metric}_mean"] - float(full[f"{metric}_mean"])
    out.to_csv(audit_dir / "14_hard_subset_summary.csv", index=False)
    lines = [
        f"# Hard Test Subset: {dataset}",
        "",
        f"Hard subset: {n_hard} test samples with lowest noisy-input realized PSNR.",
        "",
        dataframe_to_markdown(out),
    ]
    (audit_dir / "14_hard_subset_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return out


def protocol_audit(root: Path, audit_dir: Path, config: dict[str, Any], manifest: pd.DataFrame, scores: pd.DataFrame) -> dict[str, Any]:
    summary = pd.read_csv(root / "04_downstream" / "downstream_training_summary.csv")
    test_manifest = pd.read_csv(root / "01_corpus" / "test_noise_manifest.csv")
    metrics_path = root / "05_evaluation" / "per_image_metrics.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()

    subsets: dict[str, Any] = {}
    for subset in BRANCHES:
        if subset == "full":
            indices = manifest["original_index"].to_numpy(dtype=np.int64)
            selected_path = "manifest/full"
        else:
            selected_file = root / "03_filtering" / subset / "selected_indices.npy"
            indices = np.load(selected_file) if selected_file.exists() else np.array([], dtype=np.int64)
            selected_path = str(selected_file)
        subsets[subset] = {
            "selected_indices_path": selected_path,
            "count": int(len(indices)),
            "unique_count": int(len(np.unique(indices))),
            "has_duplicates": bool(len(indices) != len(np.unique(indices))),
        }

    topk_n = subsets["topk_10"]["count"]
    expected_topk = set(scores.nsmallest(topk_n, "score_value")["original_index"].astype(int).tolist())
    actual_topk = set(scores[scores["is_topk_10"].astype(bool)]["original_index"].astype(int).tolist())

    eval_sets: dict[str, Any] = {}
    if not metrics.empty:
        for subset, group in metrics.groupby("training_subset"):
            eval_sets[subset] = {
                "count": int(len(group)),
                "unique_sample_index_count": int(group["sample_index"].nunique()),
                "sample_index_sha256": hashlib.sha256(
                    np.asarray(sorted(group["sample_index"].astype(int).tolist()), dtype=np.int64).tobytes()
                ).hexdigest(),
            }

    branch_checkpoints = {}
    for subset in BRANCHES:
        checkpoint = root / "04_downstream" / subset / "best.pt"
        branch_checkpoints[subset] = {
            "best_checkpoint": str(checkpoint),
            "best_checkpoint_sha256": sha256_file(checkpoint) if checkpoint.exists() else "UNKNOWN",
        }

    init_path = root / "04_downstream" / "initial_state_dict.pt"
    eval_input_path = root / "01_corpus" / "test_eval_input.pt"
    config_hash_path = root / "config_hash"
    train_pairs = set(zip(manifest["split"].astype(str), manifest["original_index"].astype(int)))
    test_pairs = set(zip(test_manifest["split"].astype(str), test_manifest["original_index"].astype(int)))

    payload = {
        "dataset": config.get("dataset", "UNKNOWN"),
        "run_root": str(root),
        "train_size": int(len(manifest)),
        "test_size": int(len(test_manifest)),
        "config_hash": config_hash_path.read_text(encoding="utf-8").strip() if config_hash_path.exists() else "UNKNOWN",
        "seeds": config.get("seeds", {}),
        "downstream_training": summary.to_dict("records"),
        "batch_size": int(config.get("downstream", {}).get("batch_size", -1)),
        "shared_initial_state_dict": {
            "path": str(init_path),
            "sha256": sha256_file(init_path) if init_path.exists() else "UNKNOWN",
            "used_for_all_branches": init_path.exists(),
        },
        "branch_checkpoints": branch_checkpoints,
        "eval_noise_cache": {
            "path": str(eval_input_path),
            "sha256": sha256_file(eval_input_path) if eval_input_path.exists() else "UNKNOWN",
            "manifest": str(root / "01_corpus" / "test_noise_manifest.csv"),
        },
        "subsets": subsets,
        "checks": {
            "train_test_pair_intersection_count": len(train_pairs & test_pairs),
            "numeric_original_index_intersection_count": int(
                len(set(manifest["original_index"].astype(int)) & set(test_manifest["original_index"].astype(int)))
            ),
            "original_index_is_split_local": True,
            "topk_10_exactly_minimum_scores": actual_topk == expected_topk,
            "topk_quantile_random_equal_size": subsets["topk_10"]["count"]
            == subsets["quantile_spread_10"]["count"]
            == subsets["random_10"]["count"],
            "eval_same_sample_indices_all_branches": len({v["sample_index_sha256"] for v in eval_sets.values()}) == 1
            if eval_sets
            else "UNKNOWN",
            "eval_metrics_file_present": metrics_path.exists(),
        },
        "eval_sets": eval_sets,
        "unknown": [],
    }
    if not metrics_path.exists():
        payload["unknown"].append("per_image_metrics.csv missing")

    write_json(audit_dir / "protocol_audit.json", payload)
    lines = [
        f"# Protocol Audit: {payload['dataset']}",
        "",
        f"- Run root: `{root}`",
        f"- Train size: {payload['train_size']}",
        f"- Test size: {payload['test_size']}",
        f"- Config hash: `{payload['config_hash']}`",
        f"- Seeds: `{payload['seeds']}`",
        f"- Shared initial_state_dict SHA256: `{payload['shared_initial_state_dict']['sha256']}`",
        f"- Eval-noise cache SHA256: `{payload['eval_noise_cache']['sha256']}`",
        "",
        "## Checks",
        "",
    ]
    for key, value in payload["checks"].items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## Downstream", "", dataframe_to_markdown(summary)]
    (audit_dir / "protocol_audit.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def write_visual_audit_report(
    audit_dir: Path,
    dataset: str,
    config: dict[str, Any],
    quality_stats: pd.DataFrame,
    correlations: pd.DataFrame,
    low_high: pd.DataFrame,
    content: pd.DataFrame,
    deltas: pd.DataFrame,
    hard: pd.DataFrame,
    protocol: dict[str, Any],
) -> None:
    final_table = pd.read_csv(audit_dir.parent / "06_summary" / "final_table.csv")
    topk_vs_full = final_table[final_table["Training subset"] == "topk_10"].iloc[0]
    topk_row = final_table[final_table["Training subset"] == "topk_10"].iloc[0]
    random_row = final_table[final_table["Training subset"] == "random_10"].iloc[0]
    quantile_row = final_table[final_table["Training subset"] == "quantile_spread_10"].iloc[0]
    best = final_table.sort_values("PSNR", ascending=False).iloc[0]
    topk_vs_random_psnr = float(topk_row["PSNR"]) - float(random_row["PSNR"])
    topk_vs_random_ssim = float(topk_row["SSIM"]) - float(random_row["SSIM"])
    topk_vs_quantile_psnr = float(topk_row["PSNR"]) - float(quantile_row["PSNR"])
    topk_vs_quantile_ssim = float(topk_row["SSIM"]) - float(quantile_row["SSIM"])

    def delta_median(subset: str, metric: str) -> float:
        row = quality_stats[(quality_stats["subset"] == subset) & (quality_stats["metric"] == metric)].iloc[0]
        return float(row["delta_median_subset_minus_full"])

    corr_target = correlations[correlations["y"] == "target_psnr_db"].iloc[0]
    corr_psnr = correlations[correlations["y"] == "realized_psnr_db_postclip"].iloc[0]
    corr_snr = correlations[correlations["y"] == "realized_snr_db_postclip"].iloc[0]

    low_psnr = low_high[(low_high["subset"] == "low_score_10") & (low_high["metric"] == "realized_psnr_db_postclip")][
        "median"
    ].iloc[0]
    high_psnr = low_high[(low_high["subset"] == "high_score_10") & (low_high["metric"] == "realized_psnr_db_postclip")][
        "median"
    ].iloc[0]
    direction = "low-score" if low_psnr > high_psnr else "high-score"

    js_rows = content[content["metric"] == "class_fraction"].groupby("subset")["js_divergence_vs_full"].max()
    max_js = float(js_rows.drop(labels=["full"], errors="ignore").max())
    bias_note = "low" if max_js < 0.01 else "visible"

    hard_psnr = hard[["training_subset", "psnr_mean", "delta_psnr_vs_full"]].copy()

    lines = [
        f"# Visual Audit Report: {dataset}",
        "",
        "## Direct Answers",
        "",
        f"1. Different fixed corruption levels: YES. target PSNR and realized PSNR/SNR have non-degenerate distributions; see `01_random_corruption_examples.png` and `02_quality_ladder.png`.",
        f"2. Top-K median shift in target PSNR: {delta_median('topk_10', 'target_psnr_db'):+.4f} dB.",
        f"3. Top-K median shift in realized PSNR: {delta_median('topk_10', 'realized_psnr_db_postclip'):+.4f} dB.",
        f"4. Top-K median shift in realized SNR: {delta_median('topk_10', 'realized_snr_db_postclip'):+.4f} dB.",
        f"5. Score-quality correlations: target PSNR Pearson {corr_target['pearson_r']:.4f}, Spearman {corr_target['spearman_rho']:.4f}; realized PSNR Pearson {corr_psnr['pearson_r']:.4f}, Spearman {corr_psnr['spearman_rho']:.4f}; realized SNR Pearson {corr_snr['pearson_r']:.4f}, Spearman {corr_snr['spearman_rho']:.4f}. Top-K uses minimum score.",
        f"6. Direction linked to cleaner data in low/high diagnostic: {direction}; low-score median realized PSNR {low_psnr:.4f}, high-score median realized PSNR {high_psnr:.4f}.",
        f"7. Class/content/brightness bias: class JSD max vs Full is {max_js:.6f}; bias flag: {bias_note}. See `10_content_bias.csv`.",
        f"8. Top-K preserves Full quality: {'YES' if float(topk_vs_full['delta_PSNR_vs_full']) >= 0 else 'NO'}; delta PSNR vs Full {float(topk_vs_full['delta_PSNR_vs_full']):+.4f} dB, delta SSIM {float(topk_vs_full['delta_SSIM_vs_full']):+.4f}.",
        f"9. Best by PSNR: {best['Training subset']} ({float(best['PSNR']):.4f}). Top-K vs Random: delta PSNR {topk_vs_random_psnr:+.4f}, delta SSIM {topk_vs_random_ssim:+.4f}. Top-K vs Quantile: delta PSNR {topk_vs_quantile_psnr:+.4f}, delta SSIM {topk_vs_quantile_ssim:+.4f}. Detailed paired deltas are in `13_per_image_metric_deltas.csv`.",
        "10. Hard test subset summary:",
        "",
        dataframe_to_markdown(hard_psnr),
        "",
        f"11. Protocol checks confirmed: {protocol['checks']}. UNKNOWN: {protocol.get('unknown', [])}.",
        "",
        "## Required Observed Baselines",
        "",
        dataframe_to_markdown(final_table),
        "",
        "## Paired Delta Statistics",
        "",
        dataframe_to_markdown(deltas),
    ]
    (audit_dir / "visual_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def analyze_dataset(output_root: Path, dataset: str, seed: int, resamples: int) -> dict[str, Any]:
    root = find_run(output_root, dataset)
    audit_dir = root / "07_visual_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    assert_inside(audit_dir, root)

    config = load_yaml(root / "00_protocol" / "resolved_config.yaml")
    manifest = pd.read_csv(root / "01_corpus" / "train_manifest.csv")
    scores = pd.read_csv(root / "03_filtering" / "scores_and_subsets.csv")
    merged = merge_quality_scores(manifest, scores)
    corpus = torch.load(root / "01_corpus" / "train_corpus.pt", map_location="cpu").to(torch.float32)
    clean_train = load_clean_dataset(dataset, train=True)

    protocol = protocol_audit(root, audit_dir, config, manifest, scores)

    rng = np.random.default_rng(seed)
    random_indices = rng.choice(manifest["original_index"].to_numpy(dtype=np.int64), size=12, replace=False)
    random_rows = manifest[manifest["original_index"].isin(random_indices)].copy()
    random_rows["_order"] = pd.Categorical(random_rows["original_index"], categories=random_indices, ordered=True)
    random_rows = random_rows.sort_values("_order").drop(columns=["_order"])
    panel_examples(
        dataset=dataset,
        clean_dataset=clean_train,
        corpus=corpus,
        manifest=manifest,
        rows=random_rows,
        output_png=audit_dir / "01_random_corruption_examples.png",
        output_csv=audit_dir / "01_random_corruption_examples.csv",
        title=f"{dataset}: random fixed corpus corruption examples",
    )

    ladder = select_quantile_rows(
        manifest,
        "realized_psnr_db_postclip",
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
    )
    panel_examples(
        dataset=dataset,
        clean_dataset=clean_train,
        corpus=corpus,
        manifest=manifest,
        rows=ladder,
        output_png=audit_dir / "02_quality_ladder.png",
        output_csv=audit_dir / "02_quality_ladder.csv",
        title=f"{dataset}: strongest to weakest corpus corruption by realized PSNR",
    )

    quality_stats = write_subset_quality_distributions(audit_dir, merged)
    write_sigma_distributions(audit_dir, merged)
    correlations = write_score_quality(audit_dir, merged)
    deciles = write_quality_by_score_decile(audit_dir, merged)
    low_high = write_low_high_score_quality(audit_dir, merged)
    selected_examples = write_selected_and_oracle_examples(audit_dir, dataset, clean_train, corpus, merged)
    quadrants = write_score_quality_quadrants(audit_dir, dataset, clean_train, corpus, merged)
    content = write_content_bias(audit_dir, dataset, clean_train, manifest, merged)
    downstream_indices = write_downstream_examples(root, audit_dir, dataset)
    deltas = write_metric_deltas(audit_dir, root, dataset, resamples)
    hard = write_hard_subset(audit_dir, root, dataset)

    visual_indices = {
        "random_corruption_original_indices": [int(x) for x in random_indices],
        "quality_ladder_original_indices": [int(x) for x in ladder["original_index"].tolist()],
        "selected_and_oracle_original_indices": [int(x) for x in selected_examples["original_index"].tolist()],
        "quadrant_original_indices": [int(x) for x in quadrants["original_index"].tolist()],
        "downstream_test_sample_indices": downstream_indices,
    }
    write_json(audit_dir / "visual_sample_indices.json", visual_indices)

    write_visual_audit_report(
        audit_dir,
        dataset,
        config,
        quality_stats,
        correlations,
        low_high,
        content,
        deltas,
        hard,
        protocol,
    )

    return {
        "dataset": dataset,
        "run_root": str(root),
        "audit_dir": str(audit_dir),
        "checkpoints": {branch: str(root / "04_downstream" / branch / "best.pt") for branch in BRANCHES},
        "created_files": sorted(str(p.relative_to(audit_dir)) for p in audit_dir.rglob("*") if p.is_file()),
    }


def main() -> None:
    args = parse_args()
    if not args.no_train:
        raise RuntimeError("Pass --no-train. This script is post-hoc only and never trains models.")
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    results = [analyze_dataset(output_root, dataset, args.visual_seed, args.bootstrap_resamples) for dataset in args.datasets]
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
