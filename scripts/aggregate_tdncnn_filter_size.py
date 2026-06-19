import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def main() -> None:
    from scripts.internal import aggregate_tdncnn_filter_size

    aggregate_tdncnn_filter_size.main()


if __name__ == "__main__":
    main()
