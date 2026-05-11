from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "noisy_dataset_path": PROJECT_ROOT / "experiments" / "exp_001" / "data" / "dataset_noisy_var" / "dataset_noisy_var.h5",
    "filter_runs": [
        {
            "name": "topk_20pct",
            "filter_type": "top-k",
            "filter_share": "20%",
            "filtered_path": PROJECT_ROOT / "outputs" / "final_results" / "filtering" / "topk" / "filtered_dataset_topk_20pct.h5",
            "output_dir": PROJECT_ROOT / "outputs" / "final_results" / "filtering" / "topk" / "analysis_topk_20pct",
        },
        {
            "name": "topk_40pct",
            "filter_type": "top-k",
            "filter_share": "40%",
            "filtered_path": PROJECT_ROOT / "outputs" / "final_results" / "filtering" / "topk" / "filtered_dataset_topk_40pct.h5",
            "output_dir": PROJECT_ROOT / "outputs" / "final_results" / "filtering" / "topk" / "analysis_topk_40pct",
        },
        {
            "name": "topk_60pct",
            "filter_type": "top-k",
            "filter_share": "60%",
            "filtered_path": PROJECT_ROOT / "outputs" / "final_results" / "filtering" / "topk" / "filtered_dataset_topk_60pct.h5",
            "output_dir": PROJECT_ROOT / "outputs" / "final_results" / "filtering" / "topk" / "analysis_topk_60pct",
        },
        {
            "name": "qq_upper_80_100",
            "filter_type": "qq",
            "filter_share": "upper 80-100 quantile",
            "filtered_path": PROJECT_ROOT / "outputs" / "final_results" / "filtering" / "qq" / "filtered_dataset_qq_upper_80_100.h5",
            "output_dir": PROJECT_ROOT / "outputs" / "final_results" / "filtering" / "qq" / "analysis_qq_upper_80_100",
        },
        {
            "name": "qq_upper_60_100",
            "filter_type": "qq",
            "filter_share": "upper 60-100 quantile",
            "filtered_path": PROJECT_ROOT / "outputs" / "final_results" / "filtering" / "qq" / "filtered_dataset_qq_upper_60_100.h5",
            "output_dir": PROJECT_ROOT / "outputs" / "final_results" / "filtering" / "qq" / "analysis_qq_upper_60_100",
        },
        {
            "name": "qq_upper_40_100",
            "filter_type": "qq",
            "filter_share": "upper 40-100 quantile",
            "filtered_path": PROJECT_ROOT / "outputs" / "final_results" / "filtering" / "qq" / "filtered_dataset_qq_upper_40_100.h5",
            "output_dir": PROJECT_ROOT / "outputs" / "final_results" / "filtering" / "qq" / "analysis_qq_upper_40_100",
        },
    ],
}


def load_data(noisy_path: Path, filtered_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(noisy_path, "r") as noisy_file:
        noisy_data = noisy_file["dataset"][:]

    with h5py.File(filtered_path, "r") as filtered_file:
        scores = filtered_file["scores"][:]
        selected_indices = filtered_file["selected_indices"][:]

    return noisy_data, scores, selected_indices


def save_best_vs_worst(
    noisy_data: np.ndarray,
    scores: np.ndarray,
    output_path: Path,
    filter_type: str,
    filter_share: str,
) -> None:
    idx_sorted = np.argsort(scores)
    worst_idx = idx_sorted[:16]
    best_idx = idx_sorted[-16:]

    fig, axes = plt.subplots(4, 8, figsize=(12, 6))

    for plot_idx, data_idx in enumerate(worst_idx):
        row = 0 if plot_idx < 8 else 1
        col = plot_idx % 8
        axes[row, col].imshow(noisy_data[data_idx], cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, col].axis("off")

    for plot_idx, data_idx in enumerate(best_idx):
        row = 2 if plot_idx < 8 else 3
        col = plot_idx % 8
        axes[row, col].imshow(noisy_data[data_idx], cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, col].axis("off")

    fig.suptitle(
        f"{filter_type} | {filter_share}\n"
        f"Верх: худшие по score | Низ: лучшие по score"
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def save_score_hist(
    scores: np.ndarray,
    output_path: Path,
    filter_type: str,
    filter_share: str,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(scores, bins=50)
    plt.xlabel("score")
    plt.ylabel("count")
    plt.title(f"Histogram | {filter_type} | {filter_share}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_score_scatter(
    scores: np.ndarray,
    output_path: Path,
    filter_type: str,
    filter_share: str,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(range(len(scores)), scores, s=5)
    plt.xlabel("image index")
    plt.ylabel("score")
    plt.title(f"Score scatter | {filter_type} | {filter_share}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def print_stats(scores: np.ndarray, selected_indices: np.ndarray, filter_name: str) -> None:
    print(f"Analysis: {filter_name}")
    print("Score stats:")
    print("min:", float(scores.min()))
    print("max:", float(scores.max()))
    print("mean:", float(scores.mean()))
    print("std:", float(scores.std()))
    print("selected count:", int(len(selected_indices)))


def analyze_single_run(noisy_path: Path, run_config: dict) -> None:
    filtered_path = Path(run_config["filtered_path"])
    output_dir = Path(run_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    noisy_data, scores, selected_indices = load_data(noisy_path, filtered_path)

    print_stats(scores, selected_indices, run_config["name"])

    save_score_hist(
        scores,
        output_dir / "score_hist.png",
        run_config["filter_type"],
        run_config["filter_share"],
    )
    save_score_scatter(
        scores,
        output_dir / "score_scatter.png",
        run_config["filter_type"],
        run_config["filter_share"],
    )
    save_best_vs_worst(
        noisy_data,
        scores,
        output_dir / "best_vs_worst.png",
        run_config["filter_type"],
        run_config["filter_share"],
    )

    print("saved outputs to:", output_dir)


def main() -> None:
    noisy_path = Path(CONFIG["noisy_dataset_path"])

    for run_config in CONFIG["filter_runs"]:
        filtered_path = Path(run_config["filtered_path"])
        if not filtered_path.exists():
            print(f"skip missing filter file: {filtered_path}")
            continue

        analyze_single_run(noisy_path, run_config)


if __name__ == "__main__":
    main()
