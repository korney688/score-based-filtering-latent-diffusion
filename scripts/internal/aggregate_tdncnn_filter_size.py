import argparse
import ast
import csv
import json
import struct
from pathlib import Path

from scripts.internal.tdncnn_image_runs_config import build_train_config
from src.dataset_registry import DATASET_SPECS


METRIC_COLUMNS = ["PSNR", "SSIM", "LPIPS", "FID", "validation_loss"]
GROUP_ORDER = [
    ("full", "full", "full baseline"),
    ("baseline", "top_k", "baseline top-k"),
    ("induced", "top_k", "induced top-k"),
    ("baseline", "quantile", "baseline quantile"),
    ("induced", "quantile", "induced quantile"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate TDnCNN filter-size experiment results.")
    parser.add_argument(
        "--dataset",
        choices=["mnist", "cifar10"],
        default="mnist",
        help="Dataset config to aggregate.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="TDnCNN experiment output root.",
    )
    return parser.parse_args()


def _read_last_history_row(history_path: Path) -> dict[str, float]:
    if not history_path.exists():
        return {}

    with history_path.open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        return {}
    return {key: float(value) for key, value in rows[-1].items() if key != "epoch" and value != ""}


def _num_train_samples(run_cfg: dict[str, object], dataset_name: str) -> int | None:
    filtered_indices_path = run_cfg["filtered_indices_path"]
    if filtered_indices_path is None:
        return int(DATASET_SPECS[dataset_name]["train_size"])

    path = Path(filtered_indices_path)
    if not path.exists():
        return None
    return _read_npy_first_dim(path)


def _read_npy_first_dim(path: Path) -> int:
    with path.open("rb") as npy_file:
        if npy_file.read(6) != b"\x93NUMPY":
            raise ValueError(f"Not a .npy file: {path}")
        major = npy_file.read(1)[0]
        npy_file.read(1)
        if major == 1:
            header_len = struct.unpack("<H", npy_file.read(2))[0]
        else:
            header_len = struct.unpack("<I", npy_file.read(4))[0]
        header = npy_file.read(header_len).decode("latin1")

    shape = ast.literal_eval(header)["shape"]
    return int(shape[0])


def _row_for_run(run_cfg: dict[str, object], output_root: Path, dataset_name: str) -> dict[str, object] | None:
    run_name = str(run_cfg["experiment_name"])
    run_dir = output_root / str(run_cfg["output_dir_name"])
    metrics_path = run_dir / "metrics.json"
    history_path = run_dir / "results" / "metrics_history.csv"

    if not metrics_path.exists() and not history_path.exists():
        return None

    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    history = _read_last_history_row(history_path)

    return {
        "run_name": run_name,
        "branch": run_cfg.get("branch"),
        "filter_mode": run_cfg.get("filter_mode"),
        "filter_ratio": run_cfg.get("filter_ratio"),
        "quantile_low": run_cfg.get("quantile_low"),
        "quantile_high": run_cfg.get("quantile_high"),
        "num_train_samples": _num_train_samples(run_cfg, dataset_name),
        "PSNR": history.get("psnr", metrics.get("psnr")),
        "SSIM": history.get("ssim", metrics.get("ssim")),
        "LPIPS": history.get("lpips", metrics.get("lpips")),
        "FID": history.get("fid", metrics.get("fid")),
        "validation_loss": history.get("val_loss", metrics.get("validation_loss")),
        "metrics_path": str(metrics_path) if metrics_path.exists() else "",
    }


def collect_rows(output_root: Path, train_config: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    dataset_name = str(train_config["dataset_name"])
    for run_cfg in train_config["runs"]:
        row = _row_for_run(run_cfg, output_root, dataset_name)
        if row is not None:
            rows.append(row)
    return rows


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "run_name",
        "branch",
        "filter_mode",
        "filter_ratio",
        "quantile_low",
        "quantile_high",
        "num_train_samples",
        *METRIC_COLUMNS,
        "metrics_path",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_markdown_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(rows: list[dict[str, object]], output_path: Path) -> None:
    lines = ["# TDnCNN Filter Size Summary", ""]
    for branch, filter_mode, title in GROUP_ORDER:
        group_rows = [
            row
            for row in rows
            if row["branch"] == branch and row["filter_mode"] == filter_mode
        ]
        if not group_rows:
            continue

        group_rows = sorted(group_rows, key=lambda row: float(row["filter_ratio"] or 0.0))
        lines.extend([f"## {title}", ""])
        lines.append("| run_name | filter_ratio | num_train_samples | PSNR | SSIM | LPIPS | FID | validation_loss |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in group_rows:
            lines.append(
                "| "
                + " | ".join(
                    _format_markdown_value(row[column])
                    for column in (
                        "run_name",
                        "filter_ratio",
                        "num_train_samples",
                        "PSNR",
                        "SSIM",
                        "LPIPS",
                        "FID",
                        "validation_loss",
                    )
                )
                + " |"
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_plots(rows: list[dict[str, object]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        print(f"Skipping filter-size plots: {error}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_rows = [
        row
        for row in rows
        if row["branch"] in {"baseline", "induced"} and row["filter_mode"] in {"top_k", "quantile"}
    ]

    for metric in METRIC_COLUMNS:
        plt.figure(figsize=(8, 5))
        plotted = False
        for branch in ("baseline", "induced"):
            for filter_mode in ("top_k", "quantile"):
                series = [
                    row
                    for row in filtered_rows
                    if row["branch"] == branch and row["filter_mode"] == filter_mode and row[metric] is not None
                ]
                if not series:
                    continue
                series = sorted(series, key=lambda row: float(row["filter_ratio"]))
                plt.plot(
                    [float(row["filter_ratio"]) for row in series],
                    [float(row[metric]) for row in series],
                    marker="o",
                    label=f"{branch} {filter_mode}",
                )
                plotted = True

        if not plotted:
            plt.close()
            continue

        plt.xlabel("filter_ratio")
        plt.ylabel(metric)
        plt.title(f"TDnCNN {metric} by filter size")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric.lower()}_by_filter_ratio.png")
        plt.close()


def main() -> None:
    args = parse_args()
    train_config = build_train_config(args.dataset)
    output_root = args.output_root or Path(train_config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(output_root, train_config)
    write_csv(rows, output_root / "filter_size_summary.csv")
    write_markdown(rows, output_root / "filter_size_summary.md")
    write_plots(rows, output_root / "filter_size_plots")
    print(f"Aggregated {len(rows)} TDnCNN runs into {output_root}")


if __name__ == "__main__":
    main()
