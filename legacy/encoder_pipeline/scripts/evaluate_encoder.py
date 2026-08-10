import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def run_with_forwarded_args(func, forwarded_args: list[str]) -> None:
    # Temporarily forward CLI arguments to the selected evaluation module
    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0], *forwarded_args]
        func()
    finally:
        sys.argv = original_argv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        nargs="?",
        default="compare-encoders",
        choices=["compare-encoders", "noise-geometry", "all"],
    )
    args, forwarded_args = parser.parse_known_args()

    from src.evaluation import encoder_validation

    # Stage 1: compare reconstruction quality and latent noise geometry across encoders
    run_with_forwarded_args(encoder_validation.main, ["--mode", args.mode, *forwarded_args])


if __name__ == "__main__":
    main()
