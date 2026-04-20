from pathlib import Path

from scripts.tdncnn_image_runs_config import TRAIN_CONFIG
from scripts.train_TDnCNN_image import run_experiment


def main() -> None:
    clean_path = Path(TRAIN_CONFIG["clean_path"])
    noisy_path = Path(TRAIN_CONFIG["noisy_path"])
    output_root = Path(TRAIN_CONFIG["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    for run_cfg in TRAIN_CONFIG["runs"]:
        filtered_indices_path = run_cfg["filtered_indices_path"]
        if filtered_indices_path is not None:
            filtered_indices_path = Path(filtered_indices_path)
            if not filtered_indices_path.exists():
                print(f"skip missing filtered file: {filtered_indices_path}")
                continue

        run_experiment(
            experiment_name=run_cfg["experiment_name"],
            mode=run_cfg["mode"],
            clean_path=clean_path,
            noisy_path=noisy_path,
            filtered_indices_path=filtered_indices_path,
            output_dir=output_root / run_cfg["output_dir_name"],
            batch_size=TRAIN_CONFIG["batch_size"],
            split=TRAIN_CONFIG["split"],
            epochs=TRAIN_CONFIG["epochs"],
            lr=TRAIN_CONFIG["lr"],
            seed=TRAIN_CONFIG["seed"],
        )


if __name__ == "__main__":
    main()
