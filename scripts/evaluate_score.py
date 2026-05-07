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
        default="score-validation",
        choices=["score-validation", "baseline-check", "calibration", "all"],
    )
    args = parser.parse_args()

    if args.mode in {"score-validation", "all"}:
        from src.evaluation import score_validation

        score_validation.main()
    if args.mode in {"baseline-check", "all"}:
        from src.evaluation import score_validation

        score_validation.baseline_check_main()
    if args.mode in {"calibration", "all"}:
        from src.evaluation import score_calibration

        score_calibration.main()


if __name__ == "__main__":
    main()
