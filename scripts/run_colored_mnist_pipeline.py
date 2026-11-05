from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

if "--config" not in sys.argv:
    sys.argv[1:1] = ["--config", "configs/experiments/colored_mnist_seed42_canonical.yaml"]

from src.mnist_pipeline import main


if __name__ == "__main__":
    main()
