from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity as skimage_ssim


EPS = 1e-12


def select_topk_exact(score_df: pd.DataFrame, n_subset: int) -> np.ndarray:
    # Lower score is cleaner; original_index breaks ties deterministically.
    selected = (
        score_df.sort_values(["score_value", "original_index"], ascending=[True, True])
        .head(int(n_subset))["original_index"]
        .to_numpy(dtype=np.int64)
    )
    return np.sort(selected.astype(np.int64, copy=False))


def select_random_exact(score_df: pd.DataFrame, n_subset: int, seed: int) -> np.ndarray:
    # Sort before RNG sampling so input row order cannot change the subset.
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
    # Use equal-size rank bins to cover the score range, including ties.
    position_bins = np.array_split(np.arange(len(sorted_df), dtype=np.int64), num_bins)

    ideal_quotas = np.asarray([n_subset * len(bin_positions) / len(sorted_df) for bin_positions in position_bins])
    # Allocate exactly K samples by largest remainder; break ties by bin order.
    quotas = np.floor(ideal_quotas).astype(np.int64)
    remainder = int(n_subset - quotas.sum())
    if remainder > 0:
        fractional_quotas = ideal_quotas - quotas
        order = np.lexsort((np.arange(len(fractional_quotas)), -fractional_quotas))
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
    selected = np.sort(selected.astype(np.int64, copy=False))
    if len(selected) != int(n_subset):
        raise AssertionError(f"Quantile-spread selected {len(selected)} samples, expected {int(n_subset)}")
    if len(np.unique(selected)) != len(selected):
        raise AssertionError("Quantile-spread returned duplicate original indices")
    return selected


def fraction_tag(fraction: float) -> str:
    percent = int(round(float(fraction) * 100.0))
    return f"p{percent:d}"


def subset_size_for_fraction(n_full: int, fraction: float) -> int:
    if n_full <= 0:
        raise ValueError("n_full must be positive")
    n_subset = int(round(float(fraction) * int(n_full)))
    return max(1, min(n_subset, int(n_full)))


def build_subset_grid(
    score_df: pd.DataFrame,
    *,
    fractions: Iterable[float],
    subset_seed: int,
    quantile_min_points_per_bin: int = 30,
) -> dict[str, np.ndarray]:
    df = score_df.copy()
    n_full = len(df)
    full = np.sort(df["original_index"].to_numpy(dtype=np.int64))
    subsets: dict[str, np.ndarray] = {"full": full}
    # Use the same subset seed for every fraction.
    for fraction in fractions:
        tag = fraction_tag(float(fraction))
        n_subset = subset_size_for_fraction(n_full, float(fraction))
        subsets[f"topk_{tag}"] = select_topk_exact(df, n_subset)
        subsets[f"quantile_spread_{tag}"] = select_quantile_spread_exact(
            df,
            n_subset,
            seed=subset_seed,
            min_points_per_bin=quantile_min_points_per_bin,
        )
        subsets[f"random_{tag}"] = select_random_exact(df, n_subset, seed=subset_seed)
    return subsets


def build_scores_and_subsets(
    score_df: pd.DataFrame,
    *,
    subset_seed: int,
    include_topk5: bool,
    quantile_min_points_per_bin: int = 30,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    df = score_df.copy()
    subsets = build_subset_grid(
        df,
        fractions=[0.10],
        subset_seed=subset_seed,
        quantile_min_points_per_bin=quantile_min_points_per_bin,
    )
    subsets["topk_10"] = subsets["topk_p10"]
    subsets["quantile_spread_10"] = subsets["quantile_spread_p10"]
    subsets["random_10"] = subsets["random_p10"]

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
