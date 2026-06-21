from __future__ import annotations

from pathlib import Path

from src.dataset_registry import DATASET_SPECS

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _single_filter_run(filter_mode: str, percent: int, filtering_root: Path) -> dict[str, object]:
    if filter_mode not in {"topk", "quantile"}:
        raise ValueError(f"Unsupported filter mode: {filter_mode}")

    experiment_prefix = "topk" if filter_mode == "topk" else "quantile"
    filter_mode_label = "top_k" if filter_mode == "topk" else "quantile"
    ratio = percent / 100.0
    return {
        "experiment_name": f"{experiment_prefix}{percent}_sigma25",
        "mode": "filtered",
        "filtered_indices_path": filtering_root / filter_mode / str(percent) / "selected_indices.npy",
        "output_dir_name": f"{experiment_prefix}{percent}_sigma25",
        "branch": "induced",
        "filter_mode": filter_mode_label,
        "filter_ratio": ratio,
        "suite": "production_sigma25",
    }


def build_train_config(dataset_name: str = "imagenet100") -> dict[str, object]:
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {sorted(DATASET_SPECS)}")

    filtering_root = PROJECT_ROOT / "experiments" / dataset_name / "exp_005_filtering"
    runs = [
        {
            "experiment_name": "full_sigma25",
            "mode": "full",
            "filtered_indices_path": None,
            "output_dir_name": "full_sigma25",
            "branch": "full",
            "filter_mode": "full",
            "filter_ratio": 1.0,
            "suite": "production_sigma25",
        },
        _single_filter_run("quantile", 10, filtering_root),
        _single_filter_run("topk", 10, filtering_root),
    ]

    return {
        "dataset_name": dataset_name,
        "download": False,
        "data_root": PROJECT_ROOT / "data",
        "output_root": PROJECT_ROOT / "experiments" / dataset_name / "exp_007_drunet",
        "checkpoint_root": PROJECT_ROOT / "checkpoints" / dataset_name / "drunet",
        "batch_size": 64 if dataset_name == "imagenet100" else 16,
        "epochs": 15,
        "lr": 1e-4,
        "weight_decay": 0.0,
        "scheduler": None,
        "seed": 42,
        "sigma_mode": "fixed",
        "fixed_sigma": 25.0 / 255.0,
        "sigma_min": 25.0 / 255.0,
        "sigma_max": 25.0 / 255.0,
        "num_workers": 0,
        "in_channels": int(DATASET_SPECS[dataset_name]["channels"]),
        "model": {
            "features": 64,
            "num_layers": 5,
            "official": True,
            "nc": [64, 128, 256, 512],
            "nb": 4,
            "act_mode": "R",
            "downsample_mode": "strideconv",
            "upsample_mode": "convtranspose",
        },
        "runs": runs,
    }
