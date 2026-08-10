import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.DDPM_model import build_DDPM_model
from src.dataset_registry import build_torchvision_split, dataset_display_name
from src.filters import (
    compute_latent_ddpm_scores,
    quantile_spread_num_bins,
    save_filtering_grids,
    save_noisy_filtering_grids,
    select_indices,
)
from src.tools import set_seed

log = logging.getLogger(__name__)


def build_train_dataset(cfg: DictConfig):
    # Use the original train split; filtering saves indices, not images.
    dataset_cfg = cfg.get("dataset", None)
    if dataset_cfg is None:
        raise ValueError("filter_dataset.dataset config is required")
    data_root = dataset_cfg.get("paths", {}).get("data_root", str(PROJECT_ROOT / "data"))
    dataset = build_torchvision_split(
        dataset_cfg=dataset_cfg,
        train=True,
        data_root=to_absolute_path(data_root),
        transform_profile="normalized",
        download=bool(dataset_cfg.get("download", False)),
    )
    max_samples = cfg.get("max_samples", None)
    if max_samples is not None:
        max_items = min(int(max_samples), len(dataset))
        return Subset(dataset, list(range(max_items)))
    return dataset


def _cfg_get(cfg: Any | None, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def resolve_run_dir(cfg: DictConfig) -> Path:
    branch = cfg.ddpm_branch
    if branch not in {"baseline", "induced"}:
        raise ValueError(f"Unsupported ddpm_branch: {branch}")

    dataset_cfg = cfg.get("dataset", {})
    dataset_slug = _cfg_get(dataset_cfg, "slug", None)
    if dataset_slug is None:
        raise ValueError("filter_dataset.dataset.slug is required")
    validation_cfg = _cfg_get(dataset_cfg, "latent_ddpm_validation", {})
    runs = _cfg_get(validation_cfg, "runs", {})
    run_spec = _cfg_get(runs, branch, {})
    checkpoint_run = _cfg_get(run_spec, "checkpoint_run", None)
    if checkpoint_run is None:
        checkpoint_run = f"latent_ddpm_{branch}_ae_noise_consistency_{dataset_slug}"
    ddpm_root = _cfg_get(_cfg_get(dataset_cfg, "paths", {}), "ddpm_root", None)
    if ddpm_root is None:
        raise ValueError("filter_dataset.dataset.paths.ddpm_root is required")
    return Path(to_absolute_path(str(ddpm_root))) / str(checkpoint_run)


def latest_checkpoint_path(run_dir: Path) -> Path:
    epoch_paths = sorted(run_dir.glob("epoch_*.pth"))
    if epoch_paths:
        return epoch_paths[-1]
    best_path = run_dir / "best_model.pth"
    if best_path.exists():
        return best_path
    raise FileNotFoundError(f"Missing DDPM checkpoint in {run_dir}")


def resolve_checkpoint_path(cfg: DictConfig) -> Path:
    # The DDPM branch selects the checkpoint used for scoring.
    checkpoint_root = resolve_run_dir(cfg)
    checkpoint_name = str(cfg.get("checkpoint_name", "latest"))
    if checkpoint_name == "latest":
        checkpoint_path = latest_checkpoint_path(checkpoint_root)
    else:
        checkpoint_path = checkpoint_root / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing DDPM checkpoint: {checkpoint_path}")
    return checkpoint_path


def load_ddpm_from_checkpoint(cfg: DictConfig, checkpoint_path: Path, device: str):
    # Rebuild the same latent-DDPM wrapper and load only the trainable UNet weights.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_params = checkpoint.get("DDPM_params", {})

    base_dim = checkpoint_params.get("base_dim", cfg.model.base_dim)
    deep = checkpoint_params.get("deep", cfg.model.deep)
    latent_noise_mode = checkpoint.get("latent_noise_mode", cfg.ddpm_branch)
    autoencoder_kind = checkpoint.get("autoencoder_kind", cfg.model.autoencoder_kind)
    cfg_autoencoder_checkpoint_path = to_absolute_path(cfg.model.autoencoder_checkpoint_path)
    autoencoder_checkpoint_path = checkpoint.get(
        "autoencoder_checkpoint_path",
        cfg_autoencoder_checkpoint_path,
    )
    if not Path(autoencoder_checkpoint_path).exists() and Path(cfg_autoencoder_checkpoint_path).exists():
        log.warning(
            "Checkpoint autoencoder path is unavailable after artifact migration: %s. "
            "Using configured path instead: %s",
            autoencoder_checkpoint_path,
            cfg_autoencoder_checkpoint_path,
        )
        autoencoder_checkpoint_path = cfg_autoencoder_checkpoint_path

    if latent_noise_mode != cfg.ddpm_branch:
        raise ValueError(
            f"Checkpoint latent_noise_mode={latent_noise_mode!r} does not match "
            f"ddpm_branch={cfg.ddpm_branch!r}"
        )

    ddpm_model = build_DDPM_model(
        base_dim=base_dim,
        deep=deep,
        device=device,
        latent_noise_mode=latent_noise_mode,
        autoencoder_kind=autoencoder_kind,
        autoencoder_checkpoint_path=autoencoder_checkpoint_path,
        dataset_cfg=cfg.get("dataset"),
    )
    ddpm_model.model.load_state_dict(checkpoint["model_state_dict"])
    ddpm_model.eval()
    return ddpm_model


def _percent_dir_name(percent: float) -> str:
    return f"{percent:g}"


def build_output_dir(cfg: DictConfig, filter_mode: str, percent: float) -> Path:
    # Keep every Stage 3 run under one experiment root.
    output_root = Path(to_absolute_path(cfg.output_root))
    if filter_mode == "top_k":
        mode_dir = "topk"
    elif filter_mode == "quantile":
        mode_dir = "quantile"
    else:
        raise ValueError(f"Unsupported filter_mode: {filter_mode}")
    return output_root / mode_dir / _percent_dir_name(percent)


def filtering_runs_from_config(cfg: DictConfig) -> list[dict[str, float | str]]:
    modes = list(cfg.get("filter_modes", []))
    percentages = list(cfg.get("filter_percentages", []))
    if not modes or not percentages:
        modes = [str(cfg.filter_mode)]
        if cfg.filter_mode == "top_k":
            percentages = [float(cfg.keep_ratio) * 100.0]
        elif cfg.filter_mode == "quantile":
            percentages = [float(cfg.keep_ratio) * 100.0]
        else:
            raise ValueError(f"Unsupported filter_mode: {cfg.filter_mode}")

    runs = []
    for mode in modes:
        for percent in percentages:
            percent_value = float(percent)
            if percent_value <= 0 or percent_value > 100:
                raise ValueError(f"Filtering percent must be in (0, 100], got {percent_value}")
            ratio = percent_value / 100.0
            if mode == "top_k":
                runs.append(
                    {
                        "filter_mode": "top_k",
                        "percent": percent_value,
                        "keep_ratio": ratio,
                        "quantile_low": 0.0,
                        "quantile_high": ratio,
                    }
                )
            elif mode == "quantile":
                runs.append(
                    {
                        "filter_mode": "quantile",
                        "percent": percent_value,
                        "keep_ratio": ratio,
                        "quantile_low": 0.0,
                        "quantile_high": ratio,
                    }
                )
            else:
                raise ValueError(f"Unsupported filter_mode: {mode}")
    return runs


def save_score_histogram(score_table: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))
    plt.hist(score_table["score"].to_numpy(), bins=50, alpha=0.85)
    plt.xlabel("score")
    plt.ylabel("count")
    plt.title("Score distribution")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_summary_plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [f"{row.filter_mode}:{row.percent:g}" for row in summary_df.itertuples(index=False)]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, summary_df["num_selected_samples"].to_numpy())
    plt.xlabel("filter run")
    plt.ylabel("selected samples")
    plt.title("Filtering selection sizes")
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = [str(col) for col in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    widths = [
        max([len(columns[idx]), *(len(row[idx]) for row in rows)] or [len(columns[idx])])
        for idx in range(len(columns))
    ]
    header = "| " + " | ".join(columns[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    body = ["| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def write_outputs(
    output_dir: Path,
    cfg: DictConfig,
    dataset,
    checkpoint_path: Path,
    score_table,
    selected_indices: np.ndarray,
    visual_samples,
    filter_mode: str,
    keep_ratio: float,
    quantile_low: float,
    quantile_high: float,
    percent: float,
) -> None:
    # Save all artifacts needed to reproduce and reuse the selection.
    output_dir.mkdir(parents=True, exist_ok=True)

    score_table = score_table.copy()
    score_table["selected"] = score_table["dataset_index"].isin(selected_indices)

    score_table.to_csv(output_dir / "scores.csv", index=False)
    np.save(output_dir / "selected_indices.npy", selected_indices)
    OmegaConf.save(config=cfg, f=output_dir / "config.yaml", resolve=True)
    save_score_histogram(score_table, output_dir / "score_histogram.png")
    noisy_grid_files = save_noisy_filtering_grids(
        visual_samples=visual_samples,
        output_dir=output_dir,
    )
    clean_grid_files = save_filtering_grids(
        dataset=dataset,
        scores_df=score_table,
        selected_indices=selected_indices,
        output_dir=output_dir,
        n_images=cfg.grid_n_images,
    )

    metadata = {
        "dataset_slug": str(cfg.get("dataset", {}).get("slug")),
        "ddpm_branch": cfg.ddpm_branch,
        "checkpoint_path": str(checkpoint_path),
        "filter_mode": filter_mode,
        "filtering_mode": filter_mode,
        "filtering_percent": float(percent),
        "keep_ratio": float(keep_ratio),
        "quantile_low": float(quantile_low),
        "quantile_high": float(quantile_high),
        "seed": int(cfg.seed),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "score_definition": "score = ||eps_pred||^2",
        "dataset": f"torchvision {dataset_display_name(cfg.get('dataset'))} train split",
        "num_scored_samples": int(len(score_table)),
        "num_selected_samples": int(len(selected_indices)),
        "n_selected": int(len(selected_indices)),
        "visual_grids": noisy_grid_files + clean_grid_files,
        "main_visual_grids": noisy_grid_files,
        "noisy_grid_source": "saved during the same scoring pass",
    }
    if filter_mode == "quantile":
        min_points_per_bin = int(cfg.get("quantile_min_points_per_bin", 30))
        metadata.update(
            {
                "algorithm": "quantile_spread",
                "min_points_per_bin": min_points_per_bin,
                "seed": int(cfg.get("quantile_seed", cfg.seed)),
                "n_bins": quantile_spread_num_bins(len(score_table), min_points_per_bin),
                "note": (
                    "Stratified sampling over score quantile bins; "
                    "preserves score-distribution coverage."
                ),
                "legacy_quantile_low": float(quantile_low),
                "legacy_quantile_high": float(quantile_high),
            }
        )
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def write_experiment_summary(root: Path, cfg: DictConfig, checkpoint_path: Path, rows: list[dict[str, Any]]) -> None:
    metrics_dir = root / "metrics"
    plots_dir = root / "plots"
    report_dir = root / "report"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(metrics_dir / "filtering_summary.csv", index=False)
    (metrics_dir / "filtering_summary.md").write_text(dataframe_to_markdown(summary_df) + "\n", encoding="utf-8")
    save_summary_plot(summary_df, plots_dir / "selected_counts.png")

    summary = {
        "dataset": str(cfg.get("dataset", {}).get("slug")),
        "ddpm_branch": str(cfg.ddpm_branch),
        "checkpoint_path": str(checkpoint_path),
        "score_definition": "score = ||eps_pred||^2",
        "runs": rows,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        f"# Filtering Summary: {summary['dataset']}",
        "",
        f"- DDPM branch: {summary['ddpm_branch']}",
        f"- DDPM checkpoint: {checkpoint_path}",
        "",
        "## Runs",
        dataframe_to_markdown(summary_df),
    ]
    (report_dir / "filtering_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def filter_dataset(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.seed, device)

    checkpoint_path = resolve_checkpoint_path(cfg)
    output_root = Path(to_absolute_path(cfg.output_root))
    runs = filtering_runs_from_config(cfg)
    for run in runs:
        output_dir = build_output_dir(cfg, str(run["filter_mode"]), float(run["percent"]))
        if output_dir.exists() and any(output_dir.iterdir()) and not cfg.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")

    log.info("Stage 3 filtering branch: %s", cfg.ddpm_branch)
    log.info("Stage 3 filtering runs: %s", runs)
    log.info("DDPM checkpoint: %s", checkpoint_path)

    dataset = build_train_dataset(cfg)
    ddpm_model = load_ddpm_from_checkpoint(cfg, checkpoint_path, device)

    score_table, visual_samples = compute_latent_ddpm_scores(
        dataset=dataset,
        ddpm_model=ddpm_model,
        device=device,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        sigma_min=cfg.sigma_min,
        sigma_max=cfg.sigma_max,
        visual_n_images=cfg.noisy_grid_n_images,
    )
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_score_histogram(score_table, plots_dir / "score_distribution.png")
    summary_rows: list[dict[str, Any]] = []
    for run in runs:
        filter_mode = str(run["filter_mode"])
        percent = float(run["percent"])
        keep_ratio = float(run["keep_ratio"])
        quantile_low = float(run["quantile_low"])
        quantile_high = float(run["quantile_high"])
        output_dir = build_output_dir(cfg, filter_mode, percent)
        selected_indices = select_indices(
            score_table=score_table,
            filter_mode=filter_mode,
            keep_ratio=keep_ratio,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
            quantile_min_points_per_bin=int(cfg.get("quantile_min_points_per_bin", 30)),
            quantile_seed=int(cfg.get("quantile_seed", cfg.seed)),
        )
        write_outputs(
            output_dir,
            cfg,
            dataset,
            checkpoint_path,
            score_table,
            selected_indices,
            visual_samples,
            filter_mode=filter_mode,
            keep_ratio=keep_ratio,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
            percent=percent,
        )
        summary_rows.append(
            {
                "dataset": str(cfg.get("dataset", {}).get("slug")),
                "ddpm_branch": str(cfg.ddpm_branch),
                "filter_mode": filter_mode,
                "percent": percent,
                "output_dir": str(output_dir),
                "num_scored_samples": int(len(score_table)),
                "num_selected_samples": int(len(selected_indices)),
                "score_mean": float(score_table["score"].mean()),
                "score_std": float(score_table["score"].std()),
                "score_min": float(score_table["score"].min()),
                "score_max": float(score_table["score"].max()),
            }
        )
        log.info("Saved Stage 3 filtering outputs to %s", output_dir)

    write_experiment_summary(output_root, cfg, checkpoint_path, summary_rows)
    log.info("Saved Stage 3 filtering summary to %s", output_root)
