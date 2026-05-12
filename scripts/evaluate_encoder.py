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
        choices=["compare-encoders", "noise-geometry", "score-validation", "all"],
    )
    args, forwarded_args = parser.parse_known_args()

    # Stage 1: compare reconstruction quality and latent noise geometry across encoders
    if args.mode in {"compare-encoders", "all"}:
        from src.evaluation import encoder_validation

        run_with_forwarded_args(encoder_validation.main, forwarded_args)
    if args.mode in {"noise-geometry", "all"}:
        from src.evaluation import encoder_validation

        run_with_forwarded_args(encoder_validation.noise_geometry_main, forwarded_args)

    # Stage 1: validate whether DDPM score behavior is useful for encoder selection
    if args.mode in {"score-validation", "all"}:
        from src.evaluation import encoder_score_validation

        run_with_forwarded_args(encoder_score_validation.main, forwarded_args)


if __name__ == "__main__":
    main()
