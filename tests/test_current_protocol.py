from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluation.image_lab_replication_analysis import (
    select_quantile_spread_exact,
    select_random_exact,
    select_topk_exact,
)
from src.utils.current_protocol import FILTER_FRACTION, validate_current_protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(name: str) -> dict:
    with (PROJECT_ROOT / "configs" / "experiments" / f"{name}.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_current_protocol_configs_are_consistent() -> None:
    validate_current_protocol("MNIST", load_config("mnist"), downstream_models=["tdncnn"])
    validate_current_protocol("CIFAR-10", load_config("cifar10"), downstream_models=["tdncnn", "drunet"])
    validate_current_protocol("ImageNet-100", load_config("imagenet100"), downstream_models=["tdncnn", "drunet"])


def test_exact_10_percent_subset_contract_and_random_determinism() -> None:
    scores = pd.DataFrame(
        {
            "original_index": np.arange(101, dtype=np.int64),
            "score_value": np.linspace(0.0, 1.0, 101),
        }
    )
    k = max(1, min(int(round(FILTER_FRACTION * len(scores))), len(scores)))
    topk = select_topk_exact(scores, k)
    quantile = select_quantile_spread_exact(scores, k, seed=42, min_points_per_bin=4)
    random_a = select_random_exact(scores, k, seed=42)
    random_b = select_random_exact(scores, k, seed=42)
    random_c = select_random_exact(scores, k, seed=43)
    assert len(topk) == len(quantile) == len(random_a) == k
    assert len(np.unique(random_a)) == len(random_a)
    assert np.array_equal(random_a, random_b)
    assert not np.array_equal(random_a, random_c)


def test_current_imports_do_not_load_legacy_autoencoder_modules() -> None:
    for name in ["src.DDPM_model", "src.autoencoder", "src.autoencoder_noise_consistency"]:
        sys.modules.pop(name, None)
    import src.mnist_pipeline  # noqa: F401
    import src.cifar10_pipeline  # noqa: F401
    import src.imagenet100_pipeline  # noqa: F401

    assert "src.DDPM_model" not in sys.modules
    assert "src.autoencoder" not in sys.modules
    assert "src.autoencoder_noise_consistency" not in sys.modules
