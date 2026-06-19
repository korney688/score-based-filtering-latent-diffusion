from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.dataset_registry import DATASET_SPECS
from src.evaluation.research_filtering_plots import write_filtering_research_plots
from src.evaluation.research_qualitative_plots import write_filtering_qualitative_plots, write_tdncnn_qualitative_plots
from src.evaluation.research_tdncnn_plots import write_tdncnn_research_plots
from src.evaluation.research_training_dynamics_plots import write_training_dynamics_plots

log = logging.getLogger(__name__)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate research-style report plots from existing experiment artifacts.")
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), required=True)
    parser.add_argument("--stage", choices=["filtering", "ddpm", "tdncnn", "all"], default="all")
    parser.add_argument("--input-root", type=Path, default=Path("experiments"))
    parser.add_argument("--output-subdir", default="research_style")
    parser.add_argument("--strict", type=parse_bool, default=False)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()
    generated = []

    if args.stage in {"filtering", "all"}:
        generated.extend(
            write_filtering_research_plots(
                dataset=args.dataset,
                input_root=args.input_root,
                output_subdir=args.output_subdir,
                strict=args.strict,
            )
        )
        generated.extend(
            write_filtering_qualitative_plots(
                dataset=args.dataset,
                input_root=args.input_root,
                output_subdir=args.output_subdir,
                strict=False if args.stage == "all" else args.strict,
            )
        )

    if args.stage in {"ddpm", "all"}:
        generated.extend(
            write_training_dynamics_plots(
                dataset=args.dataset,
                input_root=args.input_root,
                output_subdir=args.output_subdir,
                strict=args.strict,
            )
        )

    if args.stage in {"tdncnn", "all"}:
        generated.extend(
            write_tdncnn_research_plots(
                dataset=args.dataset,
                input_root=args.input_root,
                output_subdir=args.output_subdir,
                strict=args.strict,
            )
        )
        generated.extend(
            write_tdncnn_qualitative_plots(
                dataset=args.dataset,
                input_root=args.input_root,
                output_subdir=args.output_subdir,
                strict=False if args.stage == "all" else args.strict,
            )
        )

    if generated:
        print("generated_files:")
        for path in generated:
            print(path)
    else:
        print("generated_files: none")


if __name__ == "__main__":
    main()
