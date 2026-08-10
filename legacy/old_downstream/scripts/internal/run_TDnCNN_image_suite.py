import argparse
import json
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
    parser.add_argument("--dataset-mode", choices=["clean", "noisy"], default=None)
    parser.add_argument("--poisson-scale", type=float, default=None)
    parser.add_argument("--gaussian-sigma", type=float, default=None)
    parser.add_argument("--filtering-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate run definitions and paths without training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corruption_overrides = {}
    if args.poisson_scale is not None:
        corruption_overrides["poisson_scale"] = args.poisson_scale
    if args.gaussian_sigma is not None:
        corruption_overrides["gaussian_sigma"] = args.gaussian_sigma
    train_config = build_train_config(
        args.dataset,
        dataset_mode=args.dataset_mode,
        corruption_overrides=corruption_overrides,
        filtering_root=args.filtering_root,
        output_root=args.output_root,
        checkpoint_root=args.checkpoint_root,
    )
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

    selected_runs = [
        run_cfg
        for run_cfg in train_config["runs"]
        if (args.run == "all" or run_cfg["experiment_name"] == args.run)
        and (args.run != "all" or args.suite == "all" or run_cfg.get("suite") == args.suite)
    ]
    for run_cfg in selected_runs:
        filtered_indices_path = run_cfg["filtered_indices_path"]
        if filtered_indices_path is not None and not Path(filtered_indices_path).exists():
            raise FileNotFoundError(f"Missing filtered indices for {run_cfg['experiment_name']}: {filtered_indices_path}")

    if args.dry_run:
        payload = {
            "dataset_name": train_config["dataset_name"],
            "dataset_mode": train_config.get("dataset_cfg", {}).get("dataset_mode"),
            "corruption": train_config.get("dataset_cfg", {}).get("corruption"),
            "filtering_root": str(train_config["filtering_root"]),
            "data_root": str(data_root),
            "output_root": str(output_root),
            "checkpoint_root": str(checkpoint_root),
            "batch_size": int(train_config["batch_size"]),
            "epochs": int(train_config["epochs"]),
            "lr": float(train_config["lr"]),
            "runs": [
                {
                    "experiment_name": run_cfg["experiment_name"],
                    "mode": run_cfg["mode"],
                    "filtered_indices_path": str(run_cfg["filtered_indices_path"])
                    if run_cfg["filtered_indices_path"] is not None
                    else None,
                    "output_dir": str(output_root / str(run_cfg["output_dir_name"])),
                    "checkpoint_path": str(checkpoint_root / f"{run_cfg['experiment_name']}.pth"),
                }
                for run_cfg in selected_runs
            ],
        }
        print(json.dumps(payload, indent=2))
        return

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

        run_experiment(
            experiment_name=run_cfg["experiment_name"],
            mode=run_cfg["mode"],
            filtered_indices_path=filtered_indices_path,
            output_dir=output_root / run_cfg["output_dir_name"],
            data_root=data_root,
            checkpoint_dir=checkpoint_root,
            dataset_name=train_config["dataset_name"],
            dataset_cfg=train_config.get("dataset_cfg"),
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
