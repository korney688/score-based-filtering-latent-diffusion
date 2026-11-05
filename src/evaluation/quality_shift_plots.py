from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DPI = 180


def _metric_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"None of the metric columns exist: {candidates}")


def quality_metric_specs(scores: pd.DataFrame) -> list[tuple[str, str]]:
    return [
        (_metric_column(scores, ("target_psnr_db", "target_psnr")), "target PSNR"),
        (_metric_column(scores, ("realized_psnr_db_postclip", "realized_psnr_db", "realized_psnr")), "realized PSNR"),
        (_metric_column(scores, ("realized_snr_db_postclip", "realized_snr_db", "realized_snr")), "realized SNR"),
    ]


def _subset_frame(scores: pd.DataFrame, indices: np.ndarray) -> pd.DataFrame:
    selected = set(np.asarray(indices, dtype=np.int64).tolist())
    return scores[scores["original_index"].isin(selected)].copy()


def plot_quality_shift_3x3(
    scores: pd.DataFrame,
    subsets: Mapping[str, np.ndarray],
    output_path: str | Path,
    *,
    fraction_label: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = quality_metric_specs(scores)
    rows = [("Top-K", subsets["topk"]), ("Quantile", subsets["quantile"]), ("Random", subsets["random"])]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
    for row_idx, (name, indices) in enumerate(rows):
        sub = _subset_frame(scores, indices)
        for col_idx, (col, label) in enumerate(metrics):
            full_vals = scores[col].to_numpy(dtype=np.float64)
            sub_vals = sub[col].to_numpy(dtype=np.float64)
            bins = np.histogram_bin_edges(full_vals[np.isfinite(full_vals)], bins=35)
            ax = axes[row_idx, col_idx]
            ax.hist(full_vals, bins=bins, density=True, alpha=0.35, label=f"Full (n={len(scores)})")
            ax.hist(sub_vals, bins=bins, density=True, alpha=0.55, label=f"{name} (n={len(sub)})")
            full_median = float(np.nanmedian(full_vals))
            sub_median = float(np.nanmedian(sub_vals))
            ax.axvline(full_median, color="#1f77b4", linestyle="--", linewidth=1.2)
            ax.axvline(sub_median, color="#ff7f0e", linestyle="--", linewidth=1.2)
            ax.set_xlim(float(np.nanmin(full_vals)), float(np.nanmax(full_vals)))
            ax.set_title(f"{name} {fraction_label}: {label}\ndelta median {sub_median - full_median:+.3f}, n={len(sub)}")
            ax.set_xlabel(f"{label}, dB" if "PSNR" in label or "SNR" in label else label)
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=DEFAULT_DPI)
    plt.close(fig)


def plot_ecdf_realized_psnr(
    scores: pd.DataFrame,
    subsets: Mapping[str, np.ndarray],
    output_path: str | Path,
    *,
    fraction_label: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    col = _metric_column(scores, ("realized_psnr_db_postclip", "realized_psnr_db", "realized_psnr"))
    series = {"Full": scores["original_index"].to_numpy(dtype=np.int64), **subsets}
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    for name, indices in series.items():
        vals = np.sort(_subset_frame(scores, np.asarray(indices, dtype=np.int64))[col].to_numpy(dtype=np.float64))
        if len(vals) == 0:
            continue
        ax.plot(vals, np.arange(1, len(vals) + 1) / len(vals), label=name)
    ax.set_title(f"realized PSNR ECDF {fraction_label}")
    ax.set_xlabel("realized PSNR, dB")
    ax.set_ylabel("ECDF")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=DEFAULT_DPI)
    plt.close(fig)


def plot_subset_fraction_metric(results: pd.DataFrame, metric: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_col = metric.upper() if metric.upper() in results.columns else metric
    fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
    full = results[results["strategy"].str.lower() == "full"]
    if not full.empty:
        y = float(full.iloc[0][metric_col])
        ax.axhline(y, color="black", linewidth=1.2, linestyle="--", label="Full")
        ax.scatter([100], [y], color="black", s=36)
    for strategy, label in [("topk", "Top-K"), ("quantile", "Quantile"), ("random", "Random")]:
        sub = results[results["strategy"].str.lower() == strategy].sort_values("fraction")
        if sub.empty:
            continue
        ax.plot(sub["fraction"] * 100.0, sub[metric_col], marker="o", linewidth=1.8, label=label)
    ax.set_xticks([10, 20, 30, 100])
    ax.set_xlabel("subset fraction, %")
    ax.set_ylabel(metric_col)
    ax.set_title(f"subset fraction vs {metric_col}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=DEFAULT_DPI)
    plt.close(fig)
