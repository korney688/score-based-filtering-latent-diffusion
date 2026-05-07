import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        nargs="?",
        default="filtering-analysis",
        choices=["filtering-analysis", "downstream-validation"],
    )
    args = parser.parse_args()

    if args.mode == "filtering-analysis":
        from src.evaluation import filtering_evaluation

        filtering_evaluation.main()
    elif args.mode == "downstream-validation":
        from scripts.internal import run_TDnCNN_image_suite

        run_TDnCNN_image_suite.main()


if __name__ == "__main__":
    main()
