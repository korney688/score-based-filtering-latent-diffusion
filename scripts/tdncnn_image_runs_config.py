from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_CONFIG = {
    "clean_path": PROJECT_ROOT / "experiments" / "exp_001" / "data" / "dataset_clean_mnist" / "dataset_clean_mnist.h5",
    "noisy_path": PROJECT_ROOT / "experiments" / "exp_001" / "data" / "dataset_noisy_var" / "dataset_noisy_var.h5",
    "output_root": PROJECT_ROOT / "models" / "tdncnn_image_suite",
    "batch_size": 32,
    "split": 0.8,
    "epochs": 15,
    "lr": 1e-3,
    "seed": 42,
    "runs": [
        {
            "experiment_name": "baseline_all",
            "mode": "baseline",
            "filtered_indices_path": None,
            "output_dir_name": "baseline_all",
        },
        {
            "experiment_name": "topk_20pct",
            "mode": "filtered",
            "filtered_indices_path": PROJECT_ROOT / "src" / "filtered_mnist_topk" / "filtered_dataset_topk_20pct.h5",
            "output_dir_name": "topk_20pct",
        },
        {
            "experiment_name": "topk_40pct",
            "mode": "filtered",
            "filtered_indices_path": PROJECT_ROOT / "src" / "filtered_mnist_topk" / "filtered_dataset_topk_40pct.h5",
            "output_dir_name": "topk_40pct",
        },
        {
            "experiment_name": "topk_60pct",
            "mode": "filtered",
            "filtered_indices_path": PROJECT_ROOT / "src" / "filtered_mnist_topk" / "filtered_dataset_topk_60pct.h5",
            "output_dir_name": "topk_60pct",
        },
        {
            "experiment_name": "qq_upper_80_100",
            "mode": "filtered",
            "filtered_indices_path": PROJECT_ROOT / "src" / "filtered_mnist_qq" / "filtered_dataset_qq_upper_80_100.h5",
            "output_dir_name": "qq_upper_80_100",
        },
        {
            "experiment_name": "qq_upper_60_100",
            "mode": "filtered",
            "filtered_indices_path": PROJECT_ROOT / "src" / "filtered_mnist_qq" / "filtered_dataset_qq_upper_60_100.h5",
            "output_dir_name": "qq_upper_60_100",
        },
        {
            "experiment_name": "qq_upper_40_100",
            "mode": "filtered",
            "filtered_indices_path": PROJECT_ROOT / "src" / "filtered_mnist_qq" / "filtered_dataset_qq_upper_40_100.h5",
            "output_dir_name": "qq_upper_40_100",
        },
    ],
}
