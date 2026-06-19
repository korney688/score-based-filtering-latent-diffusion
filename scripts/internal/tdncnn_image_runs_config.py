from pathlib import Path

from src.dataset_registry import DATASET_SPECS

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FILTER_SIZE_RATIOS = [0.1, 0.2, 0.4, 0.6, 0.8]


def _ratio_suffix(ratio: float) -> str:
    return f"{int(round(ratio * 100))}"


def _ratio_path_value(ratio: float) -> str:
    return f"{ratio:g}"


def _topk_run(branch: str, ratio: float, filtering_root: Path) -> dict[str, object]:
    suffix = _ratio_suffix(ratio)
    return {
        "experiment_name": f"{branch}_topk_{suffix}",
        "mode": "filtered",
        "filtered_indices_path": filtering_root / branch / f"top_k_{_ratio_path_value(ratio)}" / "selected_indices.npy",
        "output_dir_name": f"{branch}_topk_{suffix}",
        "branch": branch,
        "filter_mode": "top_k",
        "filter_ratio": ratio,
        "quantile_low": None,
        "quantile_high": None,
        "suite": "filter_size",
    }


def _quantile_run(branch: str, ratio: float, filtering_root: Path) -> dict[str, object]:
    suffix = _ratio_suffix(ratio)
    return {
        "experiment_name": f"{branch}_quantile_q0_q{suffix}",
        "mode": "filtered",
        "filtered_indices_path": filtering_root
        / branch
        / f"quantile_0_{_ratio_path_value(ratio)}"
        / "selected_indices.npy",
        "output_dir_name": f"{branch}_quantile_q0_q{suffix}",
        "branch": branch,
        "filter_mode": "quantile",
        "filter_ratio": ratio,
        "quantile_low": 0.0,
        "quantile_high": ratio,
        "suite": "filter_size",
    }


def _single_topk10_run(filtering_root: Path, smoke: bool = False) -> dict[str, object]:
    return {
        "experiment_name": "topk_10_smoke" if smoke else "topk_10",
        "mode": "filtered",
        "filtered_indices_path": filtering_root / "topk" / "10" / "selected_indices.npy",
        "output_dir_name": "topk_10_smoke" if smoke else "topk_10",
        "branch": "induced",
        "filter_mode": "top_k",
        "filter_ratio": 0.10,
        "quantile_low": None,
        "quantile_high": None,
        "suite": "filter_size",
    }


def _single_filter_run(filter_mode: str, percent: int, filtering_root: Path) -> dict[str, object]:
    if filter_mode not in {"topk", "quantile"}:
        raise ValueError(f"Unsupported filter mode: {filter_mode}")

    experiment_prefix = "topk" if filter_mode == "topk" else "quantile"
    filter_mode_label = "top_k" if filter_mode == "topk" else "quantile"
    ratio = percent / 100.0
    return {
        "experiment_name": f"{experiment_prefix}_{percent}",
        "mode": "filtered",
        "filtered_indices_path": filtering_root / filter_mode / str(percent) / "selected_indices.npy",
        "output_dir_name": f"{experiment_prefix}_{percent}",
        "branch": "induced",
        "filter_mode": filter_mode_label,
        "filter_ratio": ratio,
        "quantile_low": None,
        "quantile_high": None,
        "suite": "filter_size",
    }


def _imagenet100_filter_grid_runs(filtering_root: Path) -> list[dict[str, object]]:
    return [
        _single_filter_run("topk", 5, filtering_root),
        _single_filter_run("topk", 10, filtering_root),
        _single_filter_run("topk", 15, filtering_root),
        _single_filter_run("quantile", 5, filtering_root),
        _single_filter_run("quantile", 10, filtering_root),
        _single_filter_run("quantile", 15, filtering_root),
        _single_topk10_run(filtering_root, smoke=True),
    ]


def build_train_config(dataset_name: str = "mnist") -> dict[str, object]:
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {sorted(DATASET_SPECS)}")

    filtering_root = PROJECT_ROOT / "experiments" / dataset_name / "exp_005_filtering"
    if dataset_name == "imagenet100":
        filter_size_runs = _imagenet100_filter_grid_runs(filtering_root)
    elif dataset_name == "cifar10":
        filter_size_runs = [
            _single_topk10_run(filtering_root, smoke=False),
            _single_topk10_run(filtering_root, smoke=True),
        ]
    else:
        filter_size_runs = [
            run
            for ratio in FILTER_SIZE_RATIOS
            for run in (
                _topk_run("baseline", ratio, filtering_root),
                _topk_run("induced", ratio, filtering_root),
                _quantile_run("baseline", ratio, filtering_root),
                _quantile_run("induced", ratio, filtering_root),
            )
        ]

    return {
        "dataset_name": dataset_name,
        "download": False,
        "data_root": PROJECT_ROOT / "data",
        "output_root": PROJECT_ROOT / "experiments" / dataset_name / "exp_006_tdncnn",
        "checkpoint_root": PROJECT_ROOT / "checkpoints" / dataset_name / "tdncnn",
        "batch_size": 16 if dataset_name == "imagenet100" else 32,
        "epochs": 15,
        "lr": 1e-3,
        "seed": 42,
        "sigma_min": 0.1,
        "sigma_max": 0.8,
        "num_workers": 0,
        "in_channels": int(DATASET_SPECS[dataset_name]["channels"]),
        "runs": [
            {
                "experiment_name": "full",
                "mode": "full",
                "filtered_indices_path": None,
                "output_dir_name": "full",
                "branch": "full",
                "filter_mode": "full",
                "filter_ratio": 1.0,
                "quantile_low": None,
                "quantile_high": None,
                "suite": "baseline",
            },
            *filter_size_runs,
        ],
    }


TRAIN_CONFIG = build_train_config("mnist")
