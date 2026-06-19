from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

log = logging.getLogger(__name__)

DPI = 300
FIGSIZE_DISTRIBUTION = (6.0, 4.0)
FIGSIZE_CDF_2X2 = (11.0, 8.0)
FIGSIZE_DYNAMICS_3X2 = (12.0, 8.0)
GRID_ALPHA = 0.35
LINE_WIDTH = 1.4
HIST_ALPHA = 0.35
REFERENCE_LINE_STYLE = "--"
REFERENCE_LINE_WIDTH = 1.0


def apply_lab_report_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": GRID_ALPHA,
            "lines.linewidth": LINE_WIDTH,
            "figure.constrained_layout.use": False,
        }
    )


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig, output_path_without_ext: str | Path) -> list[Path]:
    output_path_without_ext = Path(output_path_without_ext)
    ensure_dir(output_path_without_ext.parent)
    saved_paths = []
    for suffix in (".png", ".pdf"):
        output_path = output_path_without_ext.with_suffix(suffix)
        try:
            fig.savefig(output_path, bbox_inches="tight", dpi=DPI)
            saved_paths.append(output_path)
        except Exception as error:  # pragma: no cover - backend dependent
            log.warning("Could not save figure %s: %s", output_path, error)
    plt.close(fig)
    return saved_paths


def format_percent_label(value: Any) -> str:
    percent = float(value)
    if percent.is_integer():
        return f"{int(percent)}%"
    return f"{percent:g}%"


def safe_read_csv(path: str | Path) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        log.warning("Missing CSV: %s", path)
        return None
    try:
        return pd.read_csv(path)
    except Exception as error:
        log.warning("Could not read CSV %s: %s", path, error)
        return None


def dataset_experiment_root(input_root: str | Path, dataset: str) -> Path:
    root = Path(input_root)
    if root.name == dataset:
        return root
    return root / dataset


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "| status |\n| --- |\n| no rows |\n"
    headers = [str(column) for column in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"
