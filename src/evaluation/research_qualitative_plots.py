from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from src.evaluation.research_plot_style import apply_lab_report_style, dataset_experiment_root, ensure_dir, save_figure

log = logging.getLogger(__name__)


def _read_image(path: Path):
    if not path.exists():
        log.warning("Missing qualitative image: %s", path)
        return None
    try:
        return mpimg.imread(path)
    except Exception as error:
        log.warning("Could not read qualitative image %s: %s", path, error)
        return None


def _save_image_strip(images: list[tuple[str, Path]], output_stem: Path, title: str) -> bool:
    loaded = [(label, _read_image(path)) for label, path in images]
    loaded = [(label, image) for label, image in loaded if image is not None]
    if not loaded:
        log.warning("Skipping qualitative figure %s: no source images available", output_stem)
        return False
    apply_lab_report_style()
    fig, axes = plt.subplots(1, len(loaded), figsize=(4.0 * len(loaded), 4.0))
    if len(loaded) == 1:
        axes = [axes]
    for ax, (label, image) in zip(axes, loaded):
        ax.imshow(image)
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, output_stem)
    return True


def write_filtering_qualitative_plots(
    dataset: str,
    input_root: str | Path = "experiments",
    output_subdir: str = "research_style",
    strict: bool = False,
) -> list[Path]:
    filtering_root = dataset_experiment_root(input_root, dataset) / "exp_005_filtering"
    output_dir = ensure_dir(filtering_root / "plots" / output_subdir)
    run_dir = filtering_root / "topk" / "10"
    ok = _save_image_strip(
        [
            ("selected low-score samples", run_dir / "selected_samples_grid.png"),
            ("rejected high-score samples", run_dir / "rejected_samples_grid.png"),
            ("best clean/noisy", run_dir / "best_clean_noisy_grid.png"),
            ("worst clean/noisy", run_dir / "worst_clean_noisy_grid.png"),
        ],
        output_dir / "selected_vs_rejected_examples",
        "Selected vs Rejected Examples",
    )
    if strict and not ok:
        raise FileNotFoundError(f"Missing filtering qualitative grids under {run_dir}")
    return sorted(output_dir.glob("selected_vs_rejected_examples.*"))


def write_tdncnn_qualitative_plots(
    dataset: str,
    input_root: str | Path = "experiments",
    output_subdir: str = "research_style",
    strict: bool = False,
) -> list[Path]:
    tdncnn_root = dataset_experiment_root(input_root, dataset) / "exp_006_tdncnn"
    output_dir = ensure_dir(tdncnn_root / "qualitative" / output_subdir)
    comparison_runs = [
        ("full", "full"),
        ("topk_5", "top_k 5%"),
        ("topk_10", "top_k 10%"),
        ("topk_15", "top_k 15%"),
        ("quantile_5", "Quantile 5%"),
        ("quantile_10", "Quantile 10%"),
        ("quantile_15", "Quantile 15%"),
    ]
    run_images = []
    for run_name, label in comparison_runs:
        run_images.append((label, tdncnn_root / run_name / f"{run_name}_example.png"))
    ok = _save_image_strip(run_images, output_dir / "denoising_examples_comparison", "Denoising Examples Comparison")
    if strict and not ok:
        raise FileNotFoundError(f"Missing TDnCNN qualitative examples under {tdncnn_root}")
    # Error-map grids are written by future runs when per-image qualitative saving is enabled.
    error_images = [(label, tdncnn_root / run / "results" / "error_maps.png") for run, label in comparison_runs]
    _save_image_strip(error_images, output_dir / "error_maps_comparison", "Error Maps Comparison")
    return sorted(output_dir.glob("*comparison.*"))
