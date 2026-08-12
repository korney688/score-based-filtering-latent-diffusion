from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity as skimage_ssim
from torchvision.utils import make_grid, save_image


EPS = 1e-12


def select_topk_exact(score_df: pd.DataFrame, n_subset: int) -> np.ndarray:
    selected = (
        score_df.sort_values(["score_value", "original_index"], ascending=[True, True])
        .head(int(n_subset))["original_index"]
        .to_numpy(dtype=np.int64)
    )
    return np.sort(selected.astype(np.int64, copy=False))


def select_random_exact(score_df: pd.DataFrame, n_subset: int, seed: int) -> np.ndarray:
    original_indices = np.sort(score_df["original_index"].to_numpy(dtype=np.int64))
    rng = np.random.default_rng(seed)
    selected = rng.choice(original_indices, size=int(n_subset), replace=False)
    return np.sort(selected.astype(np.int64, copy=False))


def select_quantile_spread_exact(
    score_df: pd.DataFrame,
    n_subset: int,
    *,
    seed: int,
    min_points_per_bin: int = 30,
) -> np.ndarray:
    if score_df.empty:
        raise ValueError("score_df must not be empty")
    if n_subset <= 0 or n_subset > len(score_df):
        raise ValueError(f"n_subset must be in [1, {len(score_df)}], got {n_subset}")
    sorted_df = score_df.sort_values(["score_value", "original_index"], ascending=[True, True]).reset_index(drop=True)
    num_bins = max(1, len(sorted_df) // max(1, int(min_points_per_bin)))
    num_bins = min(num_bins, int(n_subset), len(sorted_df))
    position_bins = np.array_split(np.arange(len(sorted_df), dtype=np.int64), num_bins)

    exact_float = np.asarray([n_subset * len(bin_positions) / len(sorted_df) for bin_positions in position_bins])
    quotas = np.floor(exact_float).astype(np.int64)
    remainder = int(n_subset - quotas.sum())
    if remainder > 0:
        fractional = exact_float - quotas
        order = np.lexsort((np.arange(len(fractional)), -fractional))
        quotas[order[:remainder]] += 1

    rng = np.random.default_rng(seed)
    selected_positions: list[np.ndarray] = []
    for bin_positions, quota in zip(position_bins, quotas):
        if quota <= 0:
            continue
        chosen = rng.choice(bin_positions, size=int(quota), replace=False)
        selected_positions.append(chosen)
    if not selected_positions:
        return np.asarray([], dtype=np.int64)
    positions = np.concatenate(selected_positions)
    selected = sorted_df.iloc[positions]["original_index"].to_numpy(dtype=np.int64)
    return np.sort(selected.astype(np.int64, copy=False))


def build_scores_and_subsets(
    score_df: pd.DataFrame,
    *,
    subset_seed: int,
    include_topk5: bool,
    quantile_min_points_per_bin: int = 30,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    df = score_df.copy()
    n_full = len(df)
    n_subset = int(round(0.10 * n_full))
    n_subset = max(1, min(n_subset, n_full))
    topk_10 = select_topk_exact(df, n_subset)
    quantile_10 = select_quantile_spread_exact(
        df,
        n_subset,
        seed=subset_seed,
        min_points_per_bin=quantile_min_points_per_bin,
    )
    random_10 = select_random_exact(df, n_subset, seed=subset_seed)

    subsets = {
        "full": np.sort(df["original_index"].to_numpy(dtype=np.int64)),
        "topk_10": topk_10,
        "quantile_spread_10": quantile_10,
        "random_10": random_10,
    }

    for name, selected in subsets.items():
        if name == "full":
            continue
        df[f"is_{name}"] = df["original_index"].isin(set(selected.tolist()))
    ranked = df.sort_values(["score_value", "original_index"], ascending=[True, True]).reset_index()
    rank_by_index = pd.Series(np.arange(1, len(ranked) + 1, dtype=np.int64), index=ranked["index"])
    df["score_rank"] = rank_by_index.loc[df.index].to_numpy(dtype=np.int64)
    return df, subsets


def save_subset_indices(subsets: dict[str, np.ndarray], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, indices in subsets.items():
        if name == "full":
            continue
        run_dir = output_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        np.save(run_dir / "selected_indices.npy", np.sort(indices.astype(np.int64, copy=False)))
        (run_dir / "metadata.json").write_text(
            json.dumps({"subset": name, "num_selected_samples": int(len(indices))}, indent=2),
            encoding="utf-8",
        )


def _subset_quality(df: pd.DataFrame, selected: np.ndarray) -> pd.DataFrame:
    selected_set = set(np.asarray(selected, dtype=np.int64).tolist())
    return df[df["original_index"].isin(selected_set)].copy()


def write_quality_diagnostics(
    score_subset_df: pd.DataFrame,
    subsets: dict[str, np.ndarray],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full = score_subset_df.copy()
    bins = np.histogram_bin_edges(full["realized_snr_db"].to_numpy(), bins=40)
    x_min = float(np.nanmin(full["realized_snr_db"].to_numpy()))
    x_max = float(np.nanmax(full["realized_snr_db"].to_numpy()))

    rows: list[dict[str, Any]] = []
    plot_names = [name for name in ("topk_10", "quantile_spread_10", "random_10") if name in subsets]
    fig, axes = plt.subplots(len(plot_names), 1, figsize=(8, max(3, 2.6 * len(plot_names))), squeeze=False)
    for ax, name in zip(axes[:, 0], plot_names):
        selected_df = _subset_quality(full, subsets[name])
        ax.hist(full["realized_snr_db"], bins=bins, density=True, alpha=0.35, label=f"Full (n={len(full)})")
        ax.hist(
            selected_df["realized_snr_db"],
            bins=bins,
            density=True,
            alpha=0.55,
            label=f"{name} (n={len(selected_df)})",
        )
        full_median = float(full["realized_snr_db"].median())
        selected_median = float(selected_df["realized_snr_db"].median())
        ax.axvline(full_median, linestyle="--", linewidth=1.3, color="#1f77b4", label="Full median")
        ax.axvline(selected_median, linestyle="--", linewidth=1.3, color="#ff7f0e", label=f"{name} median")
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel("realized SNR, dB")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        for source_name, source_df in (("full", full), (name, selected_df)):
            for value in source_df["realized_snr_db"].to_numpy():
                rows.append({"comparison": name, "subset": source_name, "metric": "realized_snr_db", "value": float(value)})
            for value in source_df["realized_psnr_db"].to_numpy():
                rows.append({"comparison": name, "subset": source_name, "metric": "realized_psnr_db", "value": float(value)})
    fig.tight_layout()
    fig.savefig(output_dir / "quality_distributions.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(output_dir / "quality_distribution_values.csv", index=False)


def write_score_vs_quality(score_subset_df: pd.DataFrame, output_dir: str | Path) -> dict[str, float]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = score_subset_df.copy()
    pearson_psnr = float(df["score_value"].corr(df["realized_psnr_db"], method="pearson"))
    spearman_psnr = float(df["score_value"].corr(df["realized_psnr_db"], method="spearman"))
    pearson_snr = float(df["score_value"].corr(df["realized_snr_db"], method="pearson"))
    spearman_snr = float(df["score_value"].corr(df["realized_snr_db"], method="spearman"))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["score_value"], df["realized_snr_db"], s=8, alpha=0.35, label="realized SNR")
    ax.set_xlabel("score_value = sum(eps_pred ** 2)")
    ax.set_ylabel("realized SNR, dB")
    ax.set_title("Score vs Quality")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "score_vs_quality.png", dpi=180)
    plt.close(fig)

    metrics = {
        "pearson_score_realized_psnr": pearson_psnr,
        "spearman_score_realized_psnr": spearman_psnr,
        "pearson_score_realized_snr": pearson_snr,
        "spearman_score_realized_snr": spearman_snr,
        "expected_interpretation": "Top-K selects minimum score; negative score-quality correlation means Top-K tends to pick higher-quality samples.",
    }
    (output_dir / "score_vs_quality_correlations.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p = p / max(float(p.sum()), EPS)
    q = q / max(float(q.sum()), EPS)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log((p + EPS) / (m + EPS))) + 0.5 * np.sum(q * np.log((q + EPS) / (m + EPS))))


def write_class_balance(score_subset_df: pd.DataFrame, subsets: dict[str, np.ndarray], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = sorted(int(c) for c in score_subset_df["class_id"].unique())
    full_counts = score_subset_df["class_id"].value_counts().reindex(classes, fill_value=0).to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for subset_name, indices in subsets.items():
        subset_df = _subset_quality(score_subset_df, indices)
        counts = subset_df["class_id"].value_counts().reindex(classes, fill_value=0).to_numpy(dtype=np.float64)
        js = _js_divergence(full_counts, counts)
        missing = [int(cls) for cls, count in zip(classes, counts) if count == 0]
        for cls, count in zip(classes, counts):
            rows.append(
                {
                    "subset": subset_name,
                    "class_id": int(cls),
                    "count": int(count),
                    "fraction": float(count / max(float(counts.sum()), 1.0)),
                    "js_divergence_vs_full": js,
                    "missing_class_warning": bool(missing),
                }
            )
    balance_df = pd.DataFrame(rows)
    balance_df.to_csv(output_dir / "class_balance.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / max(1, len(subsets))
    x = np.arange(len(classes))
    for offset, subset_name in enumerate(subsets):
        values = balance_df[balance_df["subset"] == subset_name].sort_values("class_id")["fraction"].to_numpy()
        ax.bar(x + offset * width, values, width=width, label=subset_name)
    ax.set_xticks(x + width * (len(subsets) - 1) / 2)
    ax.set_xticklabels([str(cls) for cls in classes])
    ax.set_xlabel("class_id")
    ax.set_ylabel("fraction")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "class_balance.png", dpi=180)
    plt.close(fig)


def write_quality_examples(
    *,
    clean_dataset,
    corpus_tensor_path: str | Path,
    manifest_path: str | Path,
    score_subset_df: pd.DataFrame | None,
    output_dir: str | Path,
    max_items_per_group: int = 4,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus = torch.load(Path(corpus_tensor_path), map_location="cpu").to(torch.float32)
    manifest = pd.read_csv(manifest_path)
    cache_pos_by_original = {
        int(row.original_index): int(row.cache_pos)
        for row in manifest[["original_index", "cache_pos"]].itertuples(index=False)
    }
    groups: list[pd.DataFrame] = []
    quality_col = "realized_snr_db" if "realized_snr_db" in manifest.columns else "realized_snr_db_postclip"
    sorted_quality = manifest.sort_values(quality_col)
    groups.append(sorted_quality.head(max_items_per_group))
    groups.append(sorted_quality.tail(max_items_per_group))
    if score_subset_df is not None and not score_subset_df.empty:
        groups.append(score_subset_df.sort_values("score_value").head(max_items_per_group))
        groups.append(score_subset_df.sort_values("score_value").tail(max_items_per_group))

    images: list[torch.Tensor] = []
    for group in groups:
        for row in group.itertuples(index=False):
            clean = clean_dataset[int(row.original_index)][0]
            if not isinstance(clean, torch.Tensor):
                continue
            cache_pos = int(getattr(row, "cache_pos", cache_pos_by_original[int(row.original_index)]))
            images.append(clean.to(torch.float32).clamp(0.0, 1.0))
            images.append(corpus[cache_pos].to(torch.float32).clamp(0.0, 1.0))
    if images:
        grid = make_grid(torch.stack(images, dim=0), nrow=2, padding=2)
        save_image(grid, output_dir / "quality_examples.png")


def mse_psnr_ssim(pred: torch.Tensor, target: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = pred.detach().cpu().clamp(0.0, 1.0)
    target = target.detach().cpu().clamp(0.0, 1.0)
    mse = ((pred - target) ** 2).flatten(start_dim=1).mean(dim=1).numpy()
    psnr = 10.0 * np.log10(1.0 / np.maximum(mse, EPS))
    pred_np = pred.numpy()
    target_np = target.numpy()
    ssim_values = []
    for idx in range(pred_np.shape[0]):
        if pred_np.shape[1] == 1:
            ssim_values.append(float(skimage_ssim(target_np[idx, 0], pred_np[idx, 0], data_range=1.0)))
        else:
            ssim_values.append(float(skimage_ssim(target_np[idx], pred_np[idx], data_range=1.0, channel_axis=0)))
    return mse.astype(np.float64), psnr.astype(np.float64), np.asarray(ssim_values, dtype=np.float64)


def aggregate_metric_rows(per_image_df: pd.DataFrame) -> pd.DataFrame:
    metric_names = ["mse", "psnr", "ssim", "lpips"]
    rows = []
    for subset, group in per_image_df.groupby("training_subset", sort=False):
        row: dict[str, Any] = {"training_subset": subset, "num_images": int(len(group))}
        for metric in metric_names:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean()) if values.notna().any() else float("nan")
            row[f"{metric}_std"] = float(values.std(ddof=0)) if values.notna().any() else float("nan")
            row[f"{metric}_median"] = float(values.median()) if values.notna().any() else float("nan")
        rows.append(row)
    aggregate = pd.DataFrame(rows)
    full = aggregate[aggregate["training_subset"] == "full"]
    if not full.empty:
        full_row = full.iloc[0]
        for metric in ("psnr", "ssim", "mse", "lpips"):
            col = f"{metric}_mean"
            if col in aggregate:
                aggregate[f"delta_{metric.upper()}_vs_full"] = aggregate[col] - float(full_row[col])
    return aggregate
