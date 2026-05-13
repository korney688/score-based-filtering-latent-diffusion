import argparse
from pathlib import Path

from scripts.internal.tdncnn_image_runs_config import TRAIN_CONFIG
from scripts.internal.train_TDnCNN_image import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TDnCNN downstream MNIST denoising experiments.")
    parser.add_argument(
        "--run",
        choices=[run_cfg["experiment_name"] for run_cfg in TRAIN_CONFIG["runs"]] + ["all"],
        default="all",
        help="Run one configured experiment or all experiments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(TRAIN_CONFIG["data_root"])
    output_root = Path(TRAIN_CONFIG["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    for run_cfg in TRAIN_CONFIG["runs"]:
        if args.run != "all" and run_cfg["experiment_name"] != args.run:
            continue

        filtered_indices_path = run_cfg["filtered_indices_path"]
        if filtered_indices_path is not None:
            filtered_indices_path = Path(filtered_indices_path)
            if not filtered_indices_path.exists():
                print(f"skip missing filtered file: {filtered_indices_path}")
                continue

        run_experiment(
            experiment_name=run_cfg["experiment_name"],
            mode=run_cfg["mode"],
            filtered_indices_path=filtered_indices_path,
            output_dir=output_root / run_cfg["output_dir_name"],
            data_root=data_root,
            batch_size=TRAIN_CONFIG["batch_size"],
            epochs=TRAIN_CONFIG["epochs"],
            lr=TRAIN_CONFIG["lr"],
            seed=TRAIN_CONFIG["seed"],
            sigma_min=TRAIN_CONFIG["sigma_min"],
            sigma_max=TRAIN_CONFIG["sigma_max"],
            num_workers=TRAIN_CONFIG["num_workers"],
        )


if __name__ == "__main__":
    main()
