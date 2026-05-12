import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
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


def compute_latent_ddpm_scores(
    dataset,
    ddpm_model,
    device: str,
    batch_size: int,
    num_workers: int,
    sigma_min: float,
    sigma_max: float,
) -> pd.DataFrame:
    # Score every MNIST train sample with the Stage 2 definition: ||eps_pred||^2.
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

    with torch.no_grad():
        for batch in tqdm(loader, desc="Scoring MNIST train"):
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device)
            current_batch_size = x.shape[0]

            sigma = torch.empty(current_batch_size, device=device).uniform_(sigma_min, sigma_max)
            t = sigma_to_t(ddpm_model, sigma)

            z_0 = ddpm_model._encode_to_latent(x)
            z_t, _ = ddpm_model._make_noisy_latent_batch(x, z_0, t)
            eps_pred = ddpm_model.model(z_t, t)
            score = eps_pred.flatten(start_dim=1).pow(2).sum(dim=1)

            batch_indices = np.arange(offset, offset + current_batch_size, dtype=np.int64)
            rows.append(
                pd.DataFrame(
                    {
                        "dataset_index": batch_indices,
                        "score": score.cpu().numpy().astype(np.float32),
                        "sigma": sigma.cpu().numpy().astype(np.float32),
                    }
                )
            )
            offset += current_batch_size

    score_table = pd.concat(rows, ignore_index=True)
    log.info("Computed scores for %s MNIST train samples", len(score_table))
    return score_table


def select_lowest_top_k(score_table: pd.DataFrame, keep_ratio: float) -> np.ndarray:
    # Lower score means a more typical sample for the current filtering protocol.
    if not 0 < keep_ratio <= 1:
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")

    keep_count = max(1, int(len(score_table) * keep_ratio))
    selected = score_table.nsmallest(keep_count, "score")["dataset_index"].to_numpy(dtype=np.int64)
    return np.sort(selected)


def select_quantile_range(score_table: pd.DataFrame, quantile_low: float, quantile_high: float) -> np.ndarray:
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


def select_indices(
    score_table: pd.DataFrame,
    filter_mode: str,
    keep_ratio: float,
    quantile_low: float,
    quantile_high: float,
) -> np.ndarray:
    if filter_mode == "top_k":
        return select_lowest_top_k(score_table, keep_ratio)
    if filter_mode == "quantile":
        return select_quantile_range(score_table, quantile_low, quantile_high)
    raise ValueError(f"Unsupported filter_mode: {filter_mode}")
