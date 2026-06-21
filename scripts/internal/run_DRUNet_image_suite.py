from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.internal.drunet_image_runs_config import build_train_config
from src.dataset_registry import DATASET_SPECS
from src.DRUNet_image import build_drunet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DRUNet downstream denoising experiments.")
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="imagenet100")
    parser.add_argument("--run", default="all", help="Run one configured experiment or all experiments.")
    parser.add_argument("--suite", choices=["all", "baseline", "filter_size", "production_sigma25"], default="all")
    parser.add_argument("--list-runs", action="store_true", help="Print configured run names and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Validate run definitions and paths without training.")
    parser.add_argument("--init-model", action="store_true", help="Construct the DRUNet model and exit.")
    parser.add_argument("--init-dataloader", action="store_true", help="Construct dataloaders and exit.")
    parser.add_argument("--data-root", type=Path, default=None, help="Override data root, useful for smoke_imagenet.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument(
        "--allow-placeholder-training",
        action="store_true",
        help="Allow training the lightweight fallback model. Intended only for local plumbing tests.",
    )
    return parser.parse_args()


def selected_runs(train_config: dict[str, object], run_name: str, suite: str) -> list[dict[str, object]]:
    runs = list(train_config["runs"])
    run_names = [str(run_cfg["experiment_name"]) for run_cfg in runs]
    if run_name != "all" and run_name not in run_names:
        raise ValueError(f"Unknown run {run_name!r}. Available runs: {run_names}")
    selected = []
    for run_cfg in runs:
        if run_name != "all" and run_cfg["experiment_name"] != run_name:
            continue
        if run_name == "all" and suite != "all" and run_cfg.get("suite") != suite:
            continue
        selected.append(run_cfg)
    return selected


def validate_filtered_paths(runs: list[dict[str, object]]) -> None:
    for run_cfg in runs:
        path = run_cfg.get("filtered_indices_path")
        if path is None:
            continue
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing filtered indices for {run_cfg['experiment_name']}: {path}")


def print_run_plan(train_config: dict[str, object], runs: list[dict[str, object]], data_root: Path) -> None:
    payload = {
        "dataset_name": train_config["dataset_name"],
        "data_root": str(data_root),
        "output_root": str(train_config["output_root"]),
        "checkpoint_root": str(train_config["checkpoint_root"]),
        "batch_size": int(train_config["batch_size"]),
        "epochs": int(train_config["epochs"]),
        "lr": float(train_config["lr"]),
        "weight_decay": float(train_config["weight_decay"]),
        "scheduler": train_config["scheduler"],
        "seed": int(train_config["seed"]),
        "sigma_mode": str(train_config["sigma_mode"]),
        "fixed_sigma": float(train_config["fixed_sigma"]),
        "runs": [
            {
                "experiment_name": run_cfg["experiment_name"],
                "mode": run_cfg["mode"],
                "filtered_indices_path": str(run_cfg["filtered_indices_path"])
                if run_cfg["filtered_indices_path"] is not None
                else None,
                "output_dir": str(Path(train_config["output_root"]) / str(run_cfg["output_dir_name"])),
            }
            for run_cfg in runs
        ],
    }
    print(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    train_config = build_train_config(args.dataset)
    runs = selected_runs(train_config, args.run, args.suite)
    data_root = args.data_root or Path(train_config["data_root"])

    if args.list_runs:
        for run_cfg in train_config["runs"]:
            print(run_cfg["experiment_name"])
        return

    validate_filtered_paths(runs)

    if args.dry_run:
        print_run_plan(train_config, runs, data_root)
        return

    model_cfg = dict(train_config["model"])
    model = build_drunet(
        in_channels=int(train_config["in_channels"]),
        features=int(model_cfg["features"]),
        num_layers=int(model_cfg["num_layers"]),
        official=bool(model_cfg["official"]),
        nc=model_cfg.get("nc"),
        nb=int(model_cfg.get("nb", 4)),
        act_mode=str(model_cfg.get("act_mode", "R")),
        downsample_mode=str(model_cfg.get("downsample_mode", "strideconv")),
        upsample_mode=str(model_cfg.get("upsample_mode", "convtranspose")),
    )
    if args.init_model:
        param_count = sum(param.numel() for param in model.parameters())
        print(f"drunet_model={model.__class__.__name__}")
        print(f"parameters={param_count}")
        print(f"placeholder={bool(getattr(model, 'is_placeholder', False))}")
        return

    if args.init_dataloader:
        from scripts.internal.train_DRUNet_image import create_drunet_dataloaders

        first_run = runs[0]
        filtered_indices_path = first_run["filtered_indices_path"]
        if filtered_indices_path is not None:
            filtered_indices_path = Path(filtered_indices_path)
        train_loader, test_loader = create_drunet_dataloaders(
            filtered_indices=filtered_indices_path,
            batch_size=args.batch_size if args.batch_size is not None else int(train_config["batch_size"]),
            mode=str(first_run["mode"]),
            seed=int(train_config["seed"]),
            data_root=data_root,
            dataset_name=str(train_config["dataset_name"]),
            download=bool(train_config["download"]),
            sigma_min=float(train_config["sigma_min"]),
            sigma_max=float(train_config["sigma_max"]),
            fixed_sigma=float(train_config["fixed_sigma"]) if train_config.get("fixed_sigma") is not None else None,
            num_workers=int(train_config["num_workers"]),
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
        )
        print(f"train_batches={len(train_loader)}")
        print(f"test_batches={len(test_loader)}")
        return

    if getattr(model, "is_placeholder", False) and not args.allow_placeholder_training:
        raise RuntimeError(
            "This run resolved to the lightweight fallback model, which is not intended for production training. "
            "Use --dry-run, --init-model, or --init-dataloader for smoke checks."
        )

    from scripts.internal.train_DRUNet_image import run_experiment

    output_root = Path(train_config["output_root"])
    checkpoint_root = Path(train_config["checkpoint_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    for run_cfg in runs:
        filtered_indices_path = run_cfg["filtered_indices_path"]
        if filtered_indices_path is not None:
            filtered_indices_path = Path(filtered_indices_path)
        run_experiment(
            experiment_name=str(run_cfg["experiment_name"]),
            mode=str(run_cfg["mode"]),
            filtered_indices_path=filtered_indices_path,
            output_dir=output_root / str(run_cfg["output_dir_name"]),
            data_root=data_root,
            checkpoint_dir=checkpoint_root,
            dataset_name=str(train_config["dataset_name"]),
            download=bool(train_config["download"]),
            in_channels=int(train_config["in_channels"]),
            batch_size=args.batch_size if args.batch_size is not None else int(train_config["batch_size"]),
            epochs=args.epochs if args.epochs is not None else int(train_config["epochs"]),
            lr=float(train_config["lr"]),
            weight_decay=float(train_config["weight_decay"]),
            scheduler=train_config["scheduler"],
            seed=int(train_config["seed"]),
            sigma_mode=str(train_config["sigma_mode"]),
            fixed_sigma=float(train_config["fixed_sigma"]) if train_config.get("fixed_sigma") is not None else None,
            sigma_min=float(train_config["sigma_min"]),
            sigma_max=float(train_config["sigma_max"]),
            num_workers=int(train_config["num_workers"]),
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
            model_config=model_cfg,
        )


if __name__ == "__main__":
    main()
