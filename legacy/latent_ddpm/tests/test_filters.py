from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.filters import select_indices, select_lowest_top_k, select_quantile_spread


def _monotonic_score_table(n_items: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_index": np.arange(1000, 1000 + n_items, dtype=np.int64),
            "score": np.arange(n_items, dtype=np.float32),
        }
    )


def test_top_k_selects_lowest_scores() -> None:
    score_table = _monotonic_score_table()

    selected = select_lowest_top_k(score_table, keep_ratio=0.10)

    np.testing.assert_array_equal(selected, np.arange(1000, 1010, dtype=np.int64))


def test_quantile_spread_does_not_match_top_k() -> None:
    score_table = _monotonic_score_table()

    top_k = select_lowest_top_k(score_table, keep_ratio=0.10)
    quantile = select_quantile_spread(
        score_table,
        keep_ratio=0.10,
        min_points_per_bin=10,
        seed=42,
    )

    assert not np.array_equal(quantile, top_k)


def test_quantile_spread_selects_from_multiple_score_ranges() -> None:
    score_table = _monotonic_score_table()

    selected = select_quantile_spread(
        score_table,
        keep_ratio=0.10,
        min_points_per_bin=10,
        seed=42,
    )
    selected_scores = score_table.set_index("dataset_index").loc[selected, "score"]

    assert selected_scores.min() < 10
    assert selected_scores.max() >= 90
    assert selected_scores.nunique() > 1


def test_quantile_spread_is_deterministic_for_same_seed() -> None:
    score_table = _monotonic_score_table()

    first = select_quantile_spread(score_table, keep_ratio=0.10, min_points_per_bin=10, seed=7)
    second = select_quantile_spread(score_table, keep_ratio=0.10, min_points_per_bin=10, seed=7)

    np.testing.assert_array_equal(first, second)


def test_active_quantile_mode_uses_quantile_spread() -> None:
    score_table = _monotonic_score_table()

    selected = select_indices(
        score_table=score_table,
        filter_mode="quantile",
        keep_ratio=0.10,
        quantile_low=0.0,
        quantile_high=0.10,
        quantile_min_points_per_bin=10,
        quantile_seed=42,
    )

    assert selected.max() >= 1090


def test_quantile_spread_empty_score_table_error() -> None:
    score_table = pd.DataFrame({"dataset_index": [], "score": []})

    try:
        select_quantile_spread(score_table, keep_ratio=0.10)
    except ValueError as error:
        assert "score_table must not be empty" in str(error)
    else:
        raise AssertionError("Expected ValueError for an empty score_table")


if __name__ == "__main__":
    class FilterSelectionSmokeTests(unittest.TestCase):
        def test_top_k_selects_lowest_scores(self) -> None:
            test_top_k_selects_lowest_scores()

        def test_quantile_spread_does_not_match_top_k(self) -> None:
            test_quantile_spread_does_not_match_top_k()

        def test_quantile_spread_selects_from_multiple_score_ranges(self) -> None:
            test_quantile_spread_selects_from_multiple_score_ranges()

        def test_quantile_spread_is_deterministic_for_same_seed(self) -> None:
            test_quantile_spread_is_deterministic_for_same_seed()

        def test_active_quantile_mode_uses_quantile_spread(self) -> None:
            test_active_quantile_mode_uses_quantile_spread()

        def test_quantile_spread_empty_score_table_error(self) -> None:
            test_quantile_spread_empty_score_table_error()

    unittest.main()
