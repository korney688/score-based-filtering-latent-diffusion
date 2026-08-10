import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

log = logging.getLogger(__name__)


def sigma_to_t(model, sigma: torch.Tensor) -> torch.Tensor:
    # Use the same sigma-to-timestep mapping as Stage 2 score validation.
    schedule = torch.sqrt(1.0 - model.alphas_cumprod)
    sigma = sigma.detach().clamp(float(schedule[0]), float(schedule[-1]))
    idx = torch.searchsorted(schedule, sigma)
    idx = idx.clamp(0, schedule.shape[0] - 1)
    prev_idx = (idx - 1).clamp(0, schedule.shape[0] - 1)

    cur_err = (schedule[idx] - sigma).abs()
    prev_err = (schedule[prev_idx] - sigma).abs()
    return torch.where(prev_err < cur_err, prev_idx, idx).long()


def _make_noisy_latent_and_image(ddpm_model, x: torch.Tensor, z_0: torch.Tensor, t: torch.Tensor):
    # Build the noisy latent explicitly so we can keep the matching noisy image for diagnostics.
    sigma = ddpm_model._latent_noise_std(t)
    if ddpm_model.latent_noise_mode == "baseline":
        eps_z = torch.randn_like(z_0)
        z_noisy = z_0 + sigma * eps_z
        eps_x = torch.randn_like(x)
        x_noisy = x + sigma * eps_x
        return z_noisy, eps_z, x_noisy, sigma

    if ddpm_model.latent_noise_mode == "induced":
        eps_x = torch.randn_like(x)
        x_noisy = x + sigma * eps_x
        z_noisy = ddpm_model._encode_to_latent(x_noisy)
        target_noise = (z_noisy - z_0) / sigma.clamp_min(1e-8)
        return z_noisy, target_noise, x_noisy, sigma

    raise ValueError(f"Unsupported latent_noise_mode: {ddpm_model.latent_noise_mode}")


def _update_ranked_visual_samples(records: dict[str, list[dict]], batch_records: list[dict], n_images: int) -> None:
    # Keep only the current best and worst examples so we do not store the whole dataset as images.
    records["best"].extend(batch_records)
    records["worst"].extend(batch_records)
    records["best"] = sorted(records["best"], key=lambda item: item["score"])[:n_images]
    records["worst"] = sorted(records["worst"], key=lambda item: item["score"], reverse=True)[:n_images]


def compute_latent_ddpm_scores(
    dataset,
    ddpm_model,
    device: str,
    batch_size: int,
    num_workers: int,
    sigma_min: float,
    sigma_max: float,
    visual_n_images: int = 10,
) -> tuple[pd.DataFrame, dict[str, list[dict]]]:
    # Score every train sample with the Stage 2 definition: ||eps_pred||^2.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device == "cuda",
    )

    ddpm_model.eval()
    rows = []
    offset = 0
    visual_samples = {"best": [], "worst": []}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Scoring train split"):
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device)
            current_batch_size = x.shape[0]

            sigma = torch.empty(current_batch_size, device=device).uniform_(sigma_min, sigma_max)
            t = sigma_to_t(ddpm_model, sigma)

            z_0 = ddpm_model._encode_to_latent(x)
            z_t, _, x_noisy, actual_sigma = _make_noisy_latent_and_image(ddpm_model, x, z_0, t)
            eps_pred = ddpm_model.model(z_t, t)
            score = eps_pred.flatten(start_dim=1).pow(2).sum(dim=1)

            batch_indices = np.arange(offset, offset + current_batch_size, dtype=np.int64)
            score_np = score.cpu().numpy().astype(np.float32)
            sigma_np = actual_sigma.flatten().cpu().numpy().astype(np.float32)
            rows.append(
                pd.DataFrame(
                    {
                        "dataset_index": batch_indices,
                        "score": score_np,
                        "sigma": sigma_np,
                    }
                )
            )
            clean_cpu = x.detach().cpu()
            noisy_cpu = x_noisy.detach().cpu()
            batch_visual_records = [
                {
                    "dataset_index": int(batch_indices[item_idx]),
                    "score": float(score_np[item_idx]),
                    "clean": clean_cpu[item_idx],
                    "noisy": noisy_cpu[item_idx],
                }
                for item_idx in range(current_batch_size)
            ]
            _update_ranked_visual_samples(visual_samples, batch_visual_records, visual_n_images)
            offset += current_batch_size

    score_table = pd.concat(rows, ignore_index=True)
    log.info("Computed scores for %s train samples", len(score_table))
    return score_table, visual_samples


def select_lowest_top_k(score_table: pd.DataFrame, keep_ratio: float) -> np.ndarray:
    # Lower score means a more typical sample for the current filtering protocol.
    if not 0 < keep_ratio <= 1:
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")

    keep_count = max(1, int(len(score_table) * keep_ratio))
    selected = score_table.nsmallest(keep_count, "score")["dataset_index"].to_numpy(dtype=np.int64)
    return np.sort(selected)


def select_quantile_interval_legacy(
    score_table: pd.DataFrame,
    quantile_low: float,
    quantile_high: float,
) -> np.ndarray:
    # Keep samples whose scores fall inside an explicit score quantile interval.
    if not 0 <= quantile_low < quantile_high <= 1:
        raise ValueError(
            f"Expected 0 <= quantile_low < quantile_high <= 1, got {quantile_low}, {quantile_high}"
        )

    low_value = float(score_table["score"].quantile(quantile_low))
    high_value = float(score_table["score"].quantile(quantile_high))
    mask = (score_table["score"] >= low_value) & (score_table["score"] <= high_value)
    selected = score_table.loc[mask, "dataset_index"].to_numpy(dtype=np.int64)
    return np.sort(selected)


def quantile_spread_num_bins(score_count: int, min_points_per_bin: int = 30) -> int:
    if min_points_per_bin <= 0:
        raise ValueError(f"min_points_per_bin must be positive, got {min_points_per_bin}")
    return max(1, int(score_count) // int(min_points_per_bin))


def select_quantile_spread(
    score_table: pd.DataFrame,
    keep_ratio: float,
    min_points_per_bin: int = 30,
    seed: int = 42,
) -> np.ndarray:
    # QQ-spread filtering: sample the same fraction inside score quantile bins.
    if not 0 < keep_ratio <= 1:
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
    if score_table.empty:
        raise ValueError("score_table must not be empty")
    if "score" not in score_table.columns:
        raise ValueError("score_table must contain a 'score' column")
    if "dataset_index" not in score_table.columns:
        raise ValueError("score_table must contain a 'dataset_index' column")

    scores = score_table["score"].to_numpy()
    dataset_indices = score_table["dataset_index"].to_numpy(dtype=np.int64)
    num_bins = quantile_spread_num_bins(len(scores), min_points_per_bin)
    bin_edges = np.percentile(scores, np.linspace(0, 100, num_bins + 1))
    bin_ids = np.digitize(scores, bin_edges[1:-1], right=True)
    rng = np.random.default_rng(seed)

    selected: list[np.ndarray] = []
    for bin_id in range(num_bins):
        bin_positions = np.flatnonzero(bin_ids == bin_id)
        if len(bin_positions) == 0:
            continue
        keep_count = max(1, int(keep_ratio * len(bin_positions)))
        chosen_positions = rng.choice(bin_positions, size=keep_count, replace=False)
        selected.append(dataset_indices[chosen_positions])

    if not selected:
        return np.asarray([], dtype=np.int64)
    return np.sort(np.concatenate(selected).astype(np.int64, copy=False))


def select_indices(
    score_table: pd.DataFrame,
    filter_mode: str,
    keep_ratio: float,
    quantile_low: float | None = None,
    quantile_high: float | None = None,
    quantile_min_points_per_bin: int = 30,
    quantile_seed: int = 42,
) -> np.ndarray:
    if filter_mode == "top_k":
        return select_lowest_top_k(score_table, keep_ratio)
    if filter_mode == "quantile":
        return select_quantile_spread(
            score_table=score_table,
            keep_ratio=keep_ratio,
            min_points_per_bin=quantile_min_points_per_bin,
            seed=quantile_seed,
        )
    raise ValueError(f"Unsupported filter_mode: {filter_mode}")


def _load_clean_images(dataset, indices: np.ndarray) -> torch.Tensor:
    # Images are normalized to [-1, 1], so convert them back to [0, 1] for viewing.
    images = []
    for dataset_index in indices:
        sample = dataset[int(dataset_index)]
        image = sample[0] if isinstance(sample, (tuple, list)) else sample
        images.append((image * 0.5 + 0.5).clamp(0.0, 1.0))
    return torch.stack(images, dim=0)


def _save_image_grid(dataset, indices: np.ndarray, output_path: Path, nrow: int) -> None:
    if len(indices) == 0:
        log.warning("No samples available for grid: %s", output_path)
        return

    images = _load_clean_images(dataset, indices)
    grid = make_grid(images, nrow=nrow, padding=2)
    save_image(grid, output_path)


def _prepare_image_for_grid(image: torch.Tensor) -> torch.Tensor:
    return (image * 0.5 + 0.5).clamp(0.0, 1.0)


def _save_tensor_grid(images: list[torch.Tensor], output_path: Path, nrow: int) -> None:
    if not images:
        log.warning("No samples available for grid: %s", output_path)
        return

    image_tensor = torch.stack([_prepare_image_for_grid(image) for image in images], dim=0)
    grid = make_grid(image_tensor, nrow=nrow, padding=2)
    save_image(grid, output_path)


def _save_clean_noisy_pair_grid(records: list[dict], output_path: Path) -> None:
    if not records:
        log.warning("No samples available for grid: %s", output_path)
        return

    images = []
    for record in records:
        images.append(_prepare_image_for_grid(record["clean"]))
        images.append(_prepare_image_for_grid(record["noisy"]))
    grid = make_grid(torch.stack(images, dim=0), nrow=2, padding=2)
    save_image(grid, output_path)


def save_noisy_filtering_grids(visual_samples: dict[str, list[dict]], output_dir: str | Path) -> list[str]:
    # These are the main Stage 3 visual diagnostics: noisy images from the scoring pass.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    grid_specs = [
        ("best_noisy_grid.png", [record["noisy"] for record in visual_samples.get("best", [])], 10),
        ("worst_noisy_grid.png", [record["noisy"] for record in visual_samples.get("worst", [])], 10),
    ]

    for filename, images, nrow in grid_specs:
        output_path = output_dir / filename
        _save_tensor_grid(images, output_path, nrow=nrow)
        if images:
            saved_files.append(filename)

    pair_specs = [
        ("best_clean_noisy_grid.png", visual_samples.get("best", [])),
        ("worst_clean_noisy_grid.png", visual_samples.get("worst", [])),
    ]
    for filename, records in pair_specs:
        output_path = output_dir / filename
        _save_clean_noisy_pair_grid(records, output_path)
        if records:
            saved_files.append(filename)

    return saved_files


def save_filtering_grids(
    dataset,
    scores_df: pd.DataFrame,
    selected_indices: np.ndarray,
    output_dir: str | Path,
    n_images: int = 64,
) -> list[str]:
    # Save clean examples for a quick visual check of the filtering result.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_df = scores_df.copy()
    if "selected" not in scores_df.columns:
        selected_set = set(np.asarray(selected_indices, dtype=np.int64).tolist())
        scores_df["selected"] = scores_df["dataset_index"].isin(selected_set)

    nrow = max(1, int(np.sqrt(n_images)))
    saved_files: list[str] = []

    grid_specs = {
        "best_samples_grid.png": scores_df.sort_values("score", ascending=True),
        "worst_samples_grid.png": scores_df.sort_values("score", ascending=False),
        "selected_samples_grid.png": scores_df[scores_df["selected"]].sort_values("score", ascending=True),
        "rejected_samples_grid.png": scores_df[~scores_df["selected"]].sort_values("score", ascending=False),
    }

    for filename, rows in grid_specs.items():
        grid_indices = rows["dataset_index"].head(n_images).to_numpy(dtype=np.int64)
        output_path = output_dir / filename
        _save_image_grid(dataset, grid_indices, output_path, nrow=nrow)
        if len(grid_indices) > 0:
            saved_files.append(filename)

    return saved_files
