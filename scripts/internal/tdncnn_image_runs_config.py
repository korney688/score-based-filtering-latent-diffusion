from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FILTERING_ROOT = PROJECT_ROOT / "experiments" / "exp_005_filtering"

TRAIN_CONFIG = {
    "data_root": PROJECT_ROOT / "data",
    "output_root": PROJECT_ROOT / "experiments" / "exp_006_tdncnn",
    "batch_size": 32,
    "epochs": 15,
    "lr": 1e-3,
    "seed": 42,
    "sigma_min": 0.1,
    "sigma_max": 0.8,
    "num_workers": 0,
    "runs": [
        {
            "experiment_name": "full",
            "mode": "full",
            "filtered_indices_path": None,
            "output_dir_name": "full",
        },
        {
            "experiment_name": "baseline_topk_10",
            "mode": "filtered",
            "filtered_indices_path": FILTERING_ROOT / "baseline" / "top_k_0.1" / "selected_indices.npy",
            "output_dir_name": "baseline_topk_10",
        },
        {
            "experiment_name": "induced_topk_10",
            "mode": "filtered",
            "filtered_indices_path": FILTERING_ROOT / "induced" / "top_k_0.1" / "selected_indices.npy",
            "output_dir_name": "induced_topk_10",
        },
        {
            "experiment_name": "baseline_quantile_q0_q10",
            "mode": "filtered",
            "filtered_indices_path": FILTERING_ROOT / "baseline" / "quantile_0_0.1" / "selected_indices.npy",
            "output_dir_name": "baseline_quantile_q0_q10",
        },
        {
            "experiment_name": "induced_quantile_q0_q10",
            "mode": "filtered",
            "filtered_indices_path": FILTERING_ROOT / "induced" / "quantile_0_0.1" / "selected_indices.npy",
            "output_dir_name": "induced_quantile_q0_q10",
        },
    ],
}
