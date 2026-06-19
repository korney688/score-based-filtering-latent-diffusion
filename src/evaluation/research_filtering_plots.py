from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.evaluation.research_plot_style import (
    FIGSIZE_DISTRIBUTION,
    HIST_ALPHA,
    LINE_WIDTH,
    REFERENCE_LINE_STYLE,
    REFERENCE_LINE_WIDTH,
    apply_lab_report_style,
    dataset_experiment_root,
    ensure_dir,
    format_percent_label,
    safe_read_csv,
    save_figure,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FilteringRun:
    mode_dir: str
    percent: int
    run_dir: Path
    scores: pd.DataFrame
    selected_indices: np.ndarray

    @property
    def label(self) -> str:
        mode = "top_k" if self.mode_dir == "topk" else self.mode_dir
        return f"filtered, {mode}_{self.percent}%"

    @property
    def output_stem(self) -> str:
        return f"score_distribution_{self.mode_dir}_{self.percent}"


def _load_selected_indices(path: Path) -> np.ndarray | None:
    if not path.exists():
        log.warning("Missing selected indices: %s", path)
        return None
    try:
        return np.load(path).astype(np.int64)
    except Exception as error:
        log.warning("Could not read selected indices %s: %s", path, error)
        return None


def _load_filtering_run(filtering_root: Path, mode_dir: str, percent: int) -> FilteringRun | None:
    run_dir = filtering_root / mode_dir / str(percent)
    scores = safe_read_csv(run_dir / "scores.csv")
    if scores is None:
        scores = safe_read_csv(filtering_root / "scores.csv")
    selected_indices = _load_selected_indices(run_dir / "selected_indices.npy")
    if scores is None or selected_indices is None:
        log.warning("Skipping filtering run %s/%s because required files are missing", mode_dir, percent)
        return None
    if "score" not in scores.columns:
        log.warning("Skipping %s: scores.csv has no 'score' column", run_dir)
        return None
    return FilteringRun(mode_dir=mode_dir, percent=percent, run_dir=run_dir, scores=scores, selected_indices=selected_indices)


def _selected_scores(run: FilteringRun) -> pd.Series:
    if "selected" in run.scores.columns:
        selected = run.scores[run.scores["selected"].astype(bool)]["score"]
        if not selected.empty:
            return selected
    if "dataset_index" not in run.scores.columns:
        log.warning("%s scores.csv has no dataset_index column; using first selected-count scores", run.run_dir)
        return run.scores["score"].head(len(run.selected_indices))
    selected_set = set(run.selected_indices.tolist())
    return run.scores[run.scores["dataset_index"].isin(selected_set)]["score"]


def _plot_distribution(full_scores: pd.Series, selected_scores: pd.Series, selected_label: str, output_stem: Path) -> None:
    apply_lab_report_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_DISTRIBUTION)
    bins = 50
    ax.hist(full_scores.to_numpy(), bins=bins, density=True, histtype="stepfilled", alpha=HIST_ALPHA, color="#1f77b4", label="full (sampled)")
    ax.hist(selected_scores.to_numpy(), bins=bins, density=True, histtype="stepfilled", alpha=HIST_ALPHA, color="#ff7f0e", label=selected_label)
    full_median = float(full_scores.median())
    selected_median = float(selected_scores.median())
    ax.axvline(full_median, color="#1f77b4", linestyle=REFERENCE_LINE_STYLE, linewidth=REFERENCE_LINE_WIDTH, label="full median")
    ax.axvline(selected_median, color="#ff7f0e", linestyle=REFERENCE_LINE_STYLE, linewidth=REFERENCE_LINE_WIDTH, label="selected median")
    ax.set_xlabel("Score norm")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_stem)


def _plot_overlay(runs: list[FilteringRun], output_stem: Path) -> None:
    available = [run for run in runs if not _selected_scores(run).empty]
    if not available:
        log.warning("No filtering runs available for score distribution overlay")
        return
    full_scores = available[0].scores["score"]
    apply_lab_report_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_DISTRIBUTION)
    ax.hist(full_scores.to_numpy(), bins=60, density=True, histtype="step", linewidth=LINE_WIDTH, color="#1f77b4", label="full (sampled)")
    colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf"]
    for color, run in zip(colors, available):
        selected = _selected_scores(run)
        ax.hist(selected.to_numpy(), bins=60, density=True, histtype="step", linewidth=LINE_WIDTH, color=color, label=run.label)
    ax.set_xlabel("Score norm")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution Overlay")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_stem)


def _random_baseline(full_scores_df: pd.DataFrame, output_dir: Path, seed: int = 42, percent: int = 10) -> tuple[np.ndarray, pd.Series] | None:
    if "score" not in full_scores_df.columns:
        return None
    n_samples = max(1, int(len(full_scores_df) * percent / 100.0))
    rng = np.random.default_rng(seed)
    selected_positions = np.sort(rng.choice(len(full_scores_df), size=n_samples, replace=False))
    if "dataset_index" in full_scores_df.columns:
        random_indices = full_scores_df.iloc[selected_positions]["dataset_index"].to_numpy(dtype=np.int64)
    else:
        random_indices = selected_positions.astype(np.int64)
    ensure_dir(output_dir)
    np.save(output_dir / "random_10_indices.npy", random_indices)
    return random_indices, full_scores_df.iloc[selected_positions]["score"]


def write_filtering_research_plots(
    dataset: str,
    input_root: str | Path = "experiments",
    output_subdir: str = "research_style",
    strict: bool = False,
) -> list[Path]:
    filtering_root = dataset_experiment_root(input_root, dataset) / "exp_005_filtering"
    output_dir = ensure_dir(filtering_root / "plots" / output_subdir)
    if not filtering_root.exists():
        message = f"Missing filtering root: {filtering_root}"
        if strict:
            raise FileNotFoundError(message)
        log.warning(message)
        return []

    saved_before = set(output_dir.glob("*"))
    runs: list[FilteringRun] = []
    for mode_dir in ("topk", "quantile"):
        for percent in (5, 10, 15):
            run = _load_filtering_run(filtering_root, mode_dir, percent)
            if run is not None:
                selected = _selected_scores(run)
                if selected.empty:
                    log.warning("Skipping %s: no selected scores found", run.run_dir)
                    continue
                _plot_distribution(run.scores["score"], selected, run.label, output_dir / run.output_stem)
                runs.append(run)

    if runs:
        full_scores_df = runs[0].scores
        random_baseline = _random_baseline(full_scores_df, output_dir)
        if random_baseline is not None:
            _, random_scores = random_baseline
            _plot_distribution(full_scores_df["score"], random_scores, "filtered, random_10%", output_dir / "score_distribution_random_10")
        _plot_overlay(runs, output_dir / "score_distribution_overlay")
    else:
        log.warning("No filtering score tables found under %s", filtering_root)

    manifest = {
        "dataset": dataset,
        "filtering_root": str(filtering_root),
        "output_dir": str(output_dir),
        "available_runs": [f"{run.mode_dir}_{run.percent}" for run in runs],
        "random_baseline": str(output_dir / "random_10_indices.npy") if (output_dir / "random_10_indices.npy").exists() else None,
    }
    (output_dir / "filtering_research_plots_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return sorted(set(output_dir.glob("*")) - saved_before)
