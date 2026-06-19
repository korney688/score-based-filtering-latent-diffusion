import argparse
from pathlib import Path

from src.dataset_registry import DATASET_SPECS
from scripts.internal.tdncnn_image_runs_config import build_train_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TDnCNN downstream denoising experiments.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_SPECS),
        default="mnist",
        help="Dataset config to use.",
    )
    parser.add_argument(
        "--run",
        default="all",
        help="Run one configured experiment or all experiments.",
    )
    parser.add_argument(
        "--suite",
        choices=["all", "filter_size"],
        default="all",
        help="Restrict --run all to a configured run suite.",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="Print configured run names and exit.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override configured epoch count.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override configured batch size.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit train split for smoke tests.")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Limit test split for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_config = build_train_config(args.dataset)
    run_names = [run_cfg["experiment_name"] for run_cfg in train_config["runs"]]
    if args.run != "all" and args.run not in run_names:
        raise ValueError(f"Unknown run {args.run!r} for dataset={args.dataset}. Available runs: {run_names}")

    if args.list_runs:
        for run_cfg in train_config["runs"]:
            print(run_cfg["experiment_name"])
        return

    data_root = Path(train_config["data_root"])
    output_root = Path(train_config["output_root"])
    checkpoint_root = Path(train_config["checkpoint_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    from scripts.internal.train_TDnCNN_image import run_experiment

    for run_cfg in train_config["runs"]:
        if args.run != "all" and run_cfg["experiment_name"] != args.run:
            continue
        if args.run == "all" and args.suite != "all" and run_cfg.get("suite") != args.suite:
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
            checkpoint_dir=checkpoint_root,
            dataset_name=train_config["dataset_name"],
            download=bool(train_config["download"]),
            in_channels=int(train_config["in_channels"]),
            batch_size=args.batch_size if args.batch_size is not None else train_config["batch_size"],
            epochs=args.epochs if args.epochs is not None else train_config["epochs"],
            lr=train_config["lr"],
            seed=train_config["seed"],
            sigma_min=train_config["sigma_min"],
            sigma_max=train_config["sigma_max"],
            num_workers=train_config["num_workers"],
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
        )


if __name__ == "__main__":
    main()
