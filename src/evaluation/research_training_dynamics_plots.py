from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.research_plot_style import (
    FIGSIZE_DYNAMICS_3X2,
    REFERENCE_LINE_STYLE,
    REFERENCE_LINE_WIDTH,
    apply_lab_report_style,
    dataset_experiment_root,
    ensure_dir,
    safe_read_csv,
    save_figure,
)

log = logging.getLogger(__name__)


def _read_ddpm_histories(dataset: str) -> pd.DataFrame:
    ddpm_root = Path("checkpoints") / dataset / "ddpm"
    rows = []
    if not ddpm_root.exists():
        log.warning("Missing DDPM checkpoint root: %s", ddpm_root)
        return pd.DataFrame()
    for metrics_path in ddpm_root.glob("*/DDPM_metrics.csv"):
        df = safe_read_csv(metrics_path)
        if df is None or df.empty:
            continue
        run_name = metrics_path.parent.name
        mode = "induced" if "induced" in run_name else "baseline" if "baseline" in run_name else run_name
        df = df.copy()
        df["run_name"] = run_name
        df["latent_noise_mode"] = mode
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _read_score_stat_histories(dataset: str) -> pd.DataFrame:
    ddpm_root = Path("checkpoints") / dataset / "ddpm"
    rows = []
    if not ddpm_root.exists():
        return pd.DataFrame()
    for metrics_path in ddpm_root.glob("*/score_training_dynamics.csv"):
        df = safe_read_csv(metrics_path)
        if df is None or df.empty:
            continue
        if "run_name" not in df.columns:
            df["run_name"] = metrics_path.parent.name
        if "latent_noise_mode" not in df.columns:
            run_name = metrics_path.parent.name
            df["latent_noise_mode"] = "induced" if "induced" in run_name else "baseline" if "baseline" in run_name else run_name
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _plot_line(ax, df: pd.DataFrame, column: str, label_suffix: str = "") -> bool:
    if column not in df.columns:
        return False
    plotted = False
    for mode, mode_df in df.groupby("latent_noise_mode"):
        clean = mode_df.dropna(subset=[column])
        if clean.empty:
            continue
        ax.plot(clean["epoch"], clean[column], label=f"{mode}{label_suffix}")
        plotted = True
    return plotted


def _best_epoch_from_score_stats(df: pd.DataFrame) -> int | None:
    candidates = [column for column in ("score_skewness", "score_kurtosis") if column in df.columns]
    if not candidates or df.empty:
        return None
    work = df.copy()
    objective = np.zeros(len(work), dtype=np.float64)
    used = 0
    for column in candidates:
        values = work[column].abs().to_numpy(dtype=np.float64)
        if np.isfinite(values).any():
            objective += np.nan_to_num(values, nan=np.nanmax(values[np.isfinite(values)]))
            used += 1
    if used == 0:
        return None
    return int(work.iloc[int(np.argmin(objective))]["epoch"])


def write_training_dynamics_plots(
    dataset: str,
    input_root: str | Path = "experiments",
    output_subdir: str = "research_style",
    strict: bool = False,
) -> list[Path]:
    validation_root = dataset_experiment_root(input_root, dataset) / "exp_003_latent_ddpm_validation"
    metrics_dir = ensure_dir(validation_root / "metrics")
    output_dir = ensure_dir(validation_root / "plots" / output_subdir)
    if not validation_root.exists():
        message = f"Missing latent-DDPM validation root: {validation_root}"
        if strict:
            raise FileNotFoundError(message)
        log.warning(message)

    ddpm_history = _read_ddpm_histories(dataset)
    score_history = _read_score_stat_histories(dataset)
    if score_history.empty:
        log.warning("No per-epoch score statistics found for %s; using DDPM train/validation losses where possible", dataset)
        dynamics = ddpm_history.copy()
    elif ddpm_history.empty:
        dynamics = score_history.copy()
    else:
        dynamics = score_history.merge(
            ddpm_history[["epoch", "latent_noise_mode", "train_loss", "val_loss"]],
            on=["epoch", "latent_noise_mode"],
            how="left",
            suffixes=("", "_ddpm"),
        )

    if dynamics.empty:
        message = f"No DDPM dynamics files found for dataset={dataset}"
        if strict:
            raise FileNotFoundError(message)
        log.warning(message)
        return []

    dynamics.to_csv(metrics_dir / "score_training_dynamics.csv", index=False)

    best_epoch = _best_epoch_from_score_stats(score_history)
    if best_epoch is not None:
        (metrics_dir / "best_score_epoch.json").write_text(json.dumps({"best_score_epoch": best_epoch}, indent=2), encoding="utf-8")

    apply_lab_report_style()
    fig, axes = plt.subplots(3, 2, figsize=FIGSIZE_DYNAMICS_3X2)
    axes = axes.flatten()
    specs = [
        ("Score: Mean and Median", ("score_mean", "score_median")),
        ("Score: Standard deviation", ("score_std",)),
        ("Latent norm: Mean and Median", ("latent_norm_mean", "latent_norm_median")),
        ("Latent norm: Standard deviation", ("latent_norm_std",)),
        ("Score: Skewness", ("score_skewness",)),
        ("Score: Kurtosis", ("score_kurtosis",)),
    ]
    fallback_specs = {
        2: ("DDPM Train Loss", ("train_loss",)),
        3: ("DDPM Validation Loss", ("val_loss",)),
    }

    for idx, (title, columns) in enumerate(specs):
        ax = axes[idx]
        plotted = False
        for column in columns:
            plotted = _plot_line(ax, dynamics, column, label_suffix=f" {column}") or plotted
        if not plotted and idx in fallback_specs:
            title, columns = fallback_specs[idx]
            for column in columns:
                plotted = _plot_line(ax, dynamics, column) or plotted
        if not plotted:
            ax.text(0.5, 0.5, "missing data", ha="center", va="center", transform=ax.transAxes)
            log.warning("Skipping dynamics panel '%s': required columns are missing", title)
        if best_epoch is not None:
            ax.axvline(best_epoch, color="red", linestyle=REFERENCE_LINE_STYLE, linewidth=REFERENCE_LINE_WIDTH)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend(loc="best") if plotted else None

    fig.tight_layout()
    save_figure(fig, output_dir / "score_training_dynamics")
    return sorted(output_dir.glob("score_training_dynamics.*"))
