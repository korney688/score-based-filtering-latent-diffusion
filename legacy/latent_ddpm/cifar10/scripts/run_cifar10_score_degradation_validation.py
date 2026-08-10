from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.evaluation import cifar10_score_degradation_validation


def main() -> None:
    cifar10_score_degradation_validation.main()


if __name__ == "__main__":
    main()
