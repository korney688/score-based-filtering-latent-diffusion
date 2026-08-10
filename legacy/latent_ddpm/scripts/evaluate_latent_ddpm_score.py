import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Make project imports work when the script is called directly
    sys.path.append(str(PROJECT_ROOT))

from src.evaluation import latent_ddpm_score_validation


def main() -> None:
    # Delegate all validation logic to the evaluation module
    latent_ddpm_score_validation.main()


if __name__ == "__main__":
    main()
