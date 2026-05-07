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
        default="compare-encoders",
        choices=["compare-encoders", "noise-geometry", "score-validation", "latent-consistency", "all"],
    )
    args = parser.parse_args()

    if args.mode in {"compare-encoders", "all"}:
        from src.evaluation import encoder_validation

        encoder_validation.main()
    if args.mode in {"noise-geometry", "all"}:
        from src.evaluation import encoder_validation

        encoder_validation.noise_geometry_main()
    if args.mode in {"latent-consistency", "all"}:
        from src.evaluation import encoder_score_validation

        encoder_score_validation.latent_consistency_main()
    if args.mode in {"score-validation", "all"}:
        from src.evaluation import encoder_score_validation

        encoder_score_validation.main()


if __name__ == "__main__":
    main()
