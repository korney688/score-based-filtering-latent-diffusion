from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.research_plot_style import (
    FIGSIZE_DISTRIBUTION,
    dataframe_to_markdown,
    apply_lab_report_style,
    dataset_experiment_root,
    ensure_dir,
    safe_read_csv,
    save_figure,
)

log = logging.getLogger(__name__)

METRICS = ("psnr", "mse", "ssim", "lpips")
HIGHER_IS_BETTER = {"psnr": True, "ssim": True, "mse": False, "lpips": False}
RUN_LABELS = {
    "full": "denoised_v1 (full)",
    "topk_5": "denoised_v2 (top_k 5%)",
    "topk_10": "denoised_v3 (top_k 10%)",
    "topk_15": "denoised_v4 (top_k 15%)",
    "random_10": "random baseline 10%",
    "quantile_5": "denoised_v5 (Quantile 5%)",
    "quantile_10": "denoised_v6 (Quantile 10%)",
    "quantile_15": "denoised_v7 (Quantile 15%)",
}


def _cdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    clean = values.dropna().to_numpy(dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return np.asarray([]), np.asarray([])
    x = np.sort(clean)
    y = np.arange(1, x.size + 1, dtype=np.float64) / x.size
    return x, y


def _load_per_image_tables(tdncnn_root: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    if not tdncnn_root.exists():
        log.warning("Missing TDnCNN root: %s", tdncnn_root)
        return tables
    for run_dir in sorted(path for path in tdncnn_root.iterdir() if path.is_dir()):
        metrics_path = run_dir / "results" / "per_image_metrics.csv"
        df = safe_read_csv(metrics_path)
        if df is None or df.empty:
            log.warning("Skipping TDnCNN CDF run %s: missing per_image_metrics.csv", run_dir.name)
            continue
        if "run_name" not in df.columns:
            df["run_name"] = run_dir.name
        tables[run_dir.name] = df
    return tables


def _plot_metric_cdf(metric: str, tables: dict[str, pd.DataFrame], output_dir: Path, output_stem: str | None = None) -> None:
    apply_lab_report_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_DISTRIBUTION)
    plotted = False

    noisy_column = f"noisy_{metric}"
    noisy_df = next((df for df in tables.values() if noisy_column in df.columns), None)
    if noisy_df is not None:
        x, y = _cdf(noisy_df[noisy_column])
        if x.size:
            ax.plot(x, y, label="noisy")
            plotted = True

    for run_name, df in tables.items():
        if metric not in df.columns:
            log.warning("Run %s has no %s column; skipping this CDF curve", run_name, metric)
            continue
        x, y = _cdf(df[metric])
        if not x.size:
            continue
        ax.plot(x, y, label=RUN_LABELS.get(run_name, run_name))
        plotted = True

    if not plotted:
        log.warning("No CDF data available for metric=%s", metric)
        plt.close(fig)
        return
    ax.set_xlabel(metric.upper() if metric != "lpips" else "LPIPS")
    ax.set_ylabel("CDF")
    ax.set_title(f"{metric.upper()} CDF Comparison" if metric != "lpips" else "LPIPS CDF Comparison")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir / (output_stem or f"{metric}_cdf_comparison"))


def _median_gap_rows(tables: dict[str, pd.DataFrame], difficult_indices: np.ndarray | None = None) -> list[dict[str, object]]:
    rows = []
    for metric in METRICS:
        noisy_column = f"noisy_{metric}"
        noisy_median = None
        noisy_df = next((df for df in tables.values() if noisy_column in df.columns), None)
        if noisy_df is not None:
            source = noisy_df
            if difficult_indices is not None and "sample_index" in source.columns:
                source = source[source["sample_index"].isin(difficult_indices)]
            if noisy_column in source.columns and not source[noisy_column].dropna().empty:
                noisy_median = float(source[noisy_column].median())

        ideal_median = None
        for run_name, df in tables.items():
            source = df
            if difficult_indices is not None and "sample_index" in source.columns:
                source = source[source["sample_index"].isin(difficult_indices)]
            if metric not in source.columns or source[metric].dropna().empty:
                continue
            median_value = float(source[metric].median())
            higher_better = HIGHER_IS_BETTER[metric]
            improvement = None
            if noisy_median is not None:
                improvement = median_value - noisy_median if higher_better else noisy_median - median_value
            relative = None
            if improvement is not None and noisy_median not in {None, 0.0}:
                relative = improvement / abs(noisy_median) * 100.0
            gap = None
            if ideal_median is not None:
                gap = ideal_median - median_value if higher_better else median_value - ideal_median
            rows.append(
                {
                    "metric": metric,
                    "run_name": run_name,
                    "label": RUN_LABELS.get(run_name, run_name),
                    "median_value": median_value,
                    "gap_to_ideal": gap,
                    "improvement_vs_noisy": improvement,
                    "relative_improvement_percent": relative,
                    "n_samples": int(len(source)),
                }
            )
    return rows


def _write_gap_summary(rows: list[dict[str, object]], output_dir: Path, stem: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(dataframe_to_markdown(df), encoding="utf-8")
    return df


def _difficult_subset(tables: dict[str, pd.DataFrame], output_dir: Path) -> np.ndarray | None:
    source = next((df for df in tables.values() if "sample_index" in df.columns and "noisy_psnr" in df.columns), None)
    if source is None or source.empty:
        log.warning("Skipping difficult subset plots: noisy_psnr per-image metrics are unavailable")
        return None
    threshold = float(source["noisy_psnr"].quantile(0.20))
    difficult = source[source["noisy_psnr"] <= threshold]["sample_index"].to_numpy(dtype=np.int64)
    apply_lab_report_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_DISTRIBUTION)
    ax.hist(source["noisy_psnr"].dropna().to_numpy(), bins=40, density=True, alpha=0.35, label="all noisy PSNR")
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.0, label="bottom 20% threshold")
    ax.set_xlabel("Noisy PSNR")
    ax.set_ylabel("Density")
    ax.set_title("Difficult Subset Distribution")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir / "difficult_subset_distribution")
    return difficult


def write_tdncnn_research_plots(
    dataset: str,
    input_root: str | Path = "experiments",
    output_subdir: str = "research_style",
    strict: bool = False,
) -> list[Path]:
    tdncnn_root = dataset_experiment_root(input_root, dataset) / "exp_006_tdncnn"
    output_dir = ensure_dir(tdncnn_root / "comparison_plots" / output_subdir)
    tables = _load_per_image_tables(tdncnn_root)
    if not tables:
        message = f"No TDnCNN per-image metrics found under {tdncnn_root}"
        if strict:
            raise FileNotFoundError(message)
        log.warning(message)
        return []

    for metric in METRICS:
        _plot_metric_cdf(metric, tables, output_dir)
    gap_df = _write_gap_summary(_median_gap_rows(tables), output_dir, "quality_gap_summary")

    difficult = _difficult_subset(tables, output_dir)
    if difficult is not None and len(difficult) > 0:
        difficult_tables = {
            run: df[df["sample_index"].isin(difficult)].copy() if "sample_index" in df.columns else df.iloc[0:0].copy()
            for run, df in tables.items()
        }
        _plot_metric_cdf("psnr", difficult_tables, output_dir, output_stem="difficult_subset_psnr_cdf_comparison")
        _write_gap_summary(_median_gap_rows(tables, difficult), output_dir, "difficult_subset_gap_summary")
    else:
        log.warning("Difficult subset block skipped for dataset=%s", dataset)

    return sorted(output_dir.glob("*"))
