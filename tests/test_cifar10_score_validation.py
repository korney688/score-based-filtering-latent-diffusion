from __future__ import annotations

import numpy as np

from src.evaluation.cifar10_score_validation import (
    report_timestep_grid,
    stratified_three_way_split,
)


def test_stratified_three_way_split_exact_cifar_sizes() -> None:
    labels = np.repeat(np.arange(10, dtype=np.int64), 5000)
    splits = stratified_three_way_split(
        labels,
        {"ddpm_filter_train": 45000, "score_tuning": 2500, "score_confirmation": 2500},
        seed=42,
    )
    assert len(splits["ddpm_filter_train"]) == 45000
    assert len(splits["score_tuning"]) == 2500
    assert len(splits["score_confirmation"]) == 2500
    assert len(set(splits["ddpm_filter_train"]) & set(splits["score_tuning"])) == 0
    assert len(set(splits["ddpm_filter_train"]) & set(splits["score_confirmation"])) == 0
    assert len(set(splits["score_tuning"]) & set(splits["score_confirmation"])) == 0
    for split in splits.values():
        counts = np.bincount(labels[split], minlength=10)
        assert counts.min() == counts.max()


def test_report_timestep_mapping_for_t1000_is_one_based() -> None:
    grid = report_timestep_grid(1000, [1, 2, 5, 10, 500])
    assert [item["code_index"] for item in grid] == [0, 1, 4, 9, 499]


def test_report_timestep_mapping_uses_relative_grid_when_t_differs() -> None:
    grid = report_timestep_grid(10000, [1, 500])
    assert grid[0]["code_index"] == 0
    assert 4989 <= grid[1]["code_index"] <= 4999
