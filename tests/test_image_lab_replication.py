from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.run_image_lab_replication import make_initial_tdncnn_state, resolve_config
from src.evaluation.image_lab_replication_analysis import (
    build_scores_and_subsets,
    select_quantile_spread_exact,
    select_random_exact,
    select_topk_exact,
)
from src.image_lab_replication_datasets import (
    EvalNoiseDataset,
    MaterializedCorpusDataset,
    materialize_eval_noise_cache,
    materialize_variable_awgn_corpus,
)


class TinyImageDataset(Dataset):
    def __init__(self, n_items: int = 40, channels: int = 1, image_size: int = 8) -> None:
        self.images = []
        self.labels = []
        for idx in range(n_items):
            value = (idx + 1) / (n_items + 1)
            image = torch.full((channels, image_size, image_size), value, dtype=torch.float32)
            image[:, idx % image_size, :] = min(1.0, value + 0.1)
            self.images.append(image.clamp(0.0, 1.0))
            self.labels.append(idx % 10)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        return self.images[int(index)].clone(), int(self.labels[int(index)])


def _materialize(tmp_path: Path, n_items: int = 40):
    dataset = TinyImageDataset(n_items=n_items)
    tensor_path, manifest_path = materialize_variable_awgn_corpus(
        dataset=dataset,
        dataset_name="tiny",
        split="train",
        original_indices=np.arange(n_items),
        output_dir=tmp_path,
        target_psnr_min_db=10.0,
        target_psnr_max_db=40.0,
        corpus_seed=42,
        corruption_version="vtest",
        overwrite=True,
    )
    return dataset, tensor_path, manifest_path


def test_same_original_index_has_identical_corpus(tmp_path: Path) -> None:
    _, tensor_path, manifest_path = _materialize(tmp_path)
    first = MaterializedCorpusDataset(tensor_path, manifest_path)
    second = MaterializedCorpusDataset(tensor_path, manifest_path)
    torch.testing.assert_close(first.tensor_for_original_index(7), second.tensor_for_original_index(7))


def test_corpus_independent_of_access_order(tmp_path: Path) -> None:
    _, tensor_path, manifest_path = _materialize(tmp_path)
    ordered = MaterializedCorpusDataset(tensor_path, manifest_path, original_indices=[1, 2, 3])
    shuffled = MaterializedCorpusDataset(tensor_path, manifest_path, original_indices=[3, 1, 2])
    values = {1: ordered[0], 2: ordered[1], 3: ordered[2]}
    torch.testing.assert_close(values[3], shuffled[0])
    torch.testing.assert_close(values[1], shuffled[1])
    torch.testing.assert_close(values[2], shuffled[2])


def test_corpus_independent_of_num_workers(tmp_path: Path) -> None:
    _, tensor_path, manifest_path = _materialize(tmp_path)
    ds = MaterializedCorpusDataset(tensor_path, manifest_path)
    loader0 = DataLoader(ds, batch_size=5, shuffle=False, num_workers=0)
    loader2 = DataLoader(ds, batch_size=5, shuffle=False, num_workers=2)
    batch0 = torch.cat([batch for batch in loader0], dim=0)
    batch2 = torch.cat([batch for batch in loader2], dim=0)
    torch.testing.assert_close(batch0, batch2)


def test_manifest_matches_saved_corpus(tmp_path: Path) -> None:
    clean_dataset, tensor_path, manifest_path = _materialize(tmp_path)
    manifest = pd.read_csv(manifest_path)
    corpus = torch.load(tensor_path, map_location="cpu")
    row = manifest[manifest["original_index"] == 5].iloc[0]
    clean, _ = clean_dataset[5]
    x_corpus = corpus[int(row["cache_pos"])]
    mse = float(((x_corpus - clean) ** 2).mean().item())
    psnr = 10.0 * np.log10(1.0 / max(mse, 1e-12))
    assert abs(psnr - float(row["realized_psnr_db_postclip"])) < 1e-5


def test_topk_selects_minimum_scores() -> None:
    df = pd.DataFrame({"original_index": np.arange(10), "score_value": np.arange(10, 0, -1)})
    selected = select_topk_exact(df, 3)
    np.testing.assert_array_equal(selected, np.asarray([7, 8, 9], dtype=np.int64))


def test_subsets_use_original_indices_and_equal_10_percent_sizes() -> None:
    df = pd.DataFrame(
        {
            "original_index": np.arange(1000, 1100),
            "class_id": np.arange(100) % 10,
            "score_value": np.linspace(0.0, 1.0, 100),
            "target_psnr_db": np.linspace(10.0, 40.0, 100),
            "realized_psnr_db": np.linspace(10.0, 40.0, 100),
            "realized_snr_db": np.linspace(0.0, 30.0, 100),
        }
    )
    score_df, subsets = build_scores_and_subsets(df, subset_seed=42, include_topk5=True, quantile_min_points_per_bin=10)
    assert len(subsets["topk_10"]) == len(subsets["quantile_spread_10"]) == len(subsets["random_10"]) == 10
    assert min(subsets["topk_10"]) >= 1000
    assert score_df["score_rank"].min() == 1
    assert int(score_df.sort_values("score_value").iloc[0]["score_rank"]) == 1


def test_random_selector_is_deterministic() -> None:
    df = pd.DataFrame({"original_index": np.arange(1000, 1100), "score_value": np.arange(100)})
    first = select_random_exact(df, 10, seed=42)
    second = select_random_exact(df, 10, seed=42)
    np.testing.assert_array_equal(first, second)


def test_quantile_selector_returns_exact_size() -> None:
    df = pd.DataFrame({"original_index": np.arange(1000, 1103), "score_value": np.linspace(0.0, 1.0, 103)})
    selected = select_quantile_spread_exact(df, 10, seed=42, min_points_per_bin=9)
    assert len(selected) == 10
    assert len(np.unique(selected)) == 10


def test_eval_noise_cache_reused(tmp_path: Path) -> None:
    dataset = TinyImageDataset(n_items=16)
    paths = materialize_eval_noise_cache(
        dataset=dataset,
        dataset_name="tiny",
        original_indices=np.arange(8),
        output_dir=tmp_path,
        sigma_min=0.1,
        sigma_max=0.8,
        eval_seed=4242,
        eval_noise_version="vtest",
        overwrite=True,
    )
    first = EvalNoiseDataset(paths[0], paths[1])
    second = EvalNoiseDataset(paths[0], paths[1])
    torch.testing.assert_close(first[3][0], second[3][0])
    torch.testing.assert_close(first[3][1], second[3][1])


def test_train_and_eval_manifests_keep_split_labels(tmp_path: Path) -> None:
    dataset = TinyImageDataset(n_items=20)
    _, train_manifest = materialize_variable_awgn_corpus(
        dataset=dataset,
        dataset_name="tiny",
        split="train",
        original_indices=np.arange(10),
        output_dir=tmp_path / "train",
        target_psnr_min_db=10.0,
        target_psnr_max_db=40.0,
        corpus_seed=42,
        corruption_version="vtest",
        overwrite=True,
    )
    _, _, test_manifest = materialize_eval_noise_cache(
        dataset=dataset,
        dataset_name="tiny",
        original_indices=np.arange(10),
        output_dir=tmp_path / "test",
        sigma_min=0.1,
        sigma_max=0.8,
        eval_seed=4242,
        eval_noise_version="vtest",
        overwrite=True,
    )
    assert set(pd.read_csv(train_manifest)["split"]) == {"train"}
    assert set(pd.read_csv(test_manifest)["split"]) == {"test"}


def test_downstream_initial_state_reused(tmp_path: Path) -> None:
    args = argparse.Namespace(
        dataset="mnist",
        output_root=tmp_path,
        smoke=True,
        max_train_samples=None,
        max_test_samples=None,
        ddpm_train_steps=None,
        ddpm_batch_size=None,
        tdncnn_batch_size=None,
        full_reference_epochs=None,
        downstream_total_steps=None,
        target_psnr_min_db=10.0,
        target_psnr_max_db=40.0,
        corpus_seed=42,
        split_seed=42,
        subset_seed=42,
        model_seed=42,
        eval_seed=4242,
        task_seed=42,
        corruption_version="vtest",
        eval_noise_version="vtest",
        task_noise_version="vtest",
        task_sigma_min=0.1,
        task_sigma_max=0.8,
        eval_sigma_min=0.1,
        eval_sigma_max=0.8,
        lr_ddpm=1e-4,
        lr_tdncnn=1e-3,
        quantile_min_points_per_bin=30,
        include_topk5=False,
        num_workers=0,
        download=False,
    )
    config = resolve_config(args)
    root = tmp_path / "mnist_psnr10-40_s42" / "smoke"
    (root / "04_downstream").mkdir(parents=True)
    first = make_initial_tdncnn_state(root, config, in_channels=1)
    second = make_initial_tdncnn_state(root, config, in_channels=1)
    assert first == second
    state_a = torch.load(first, map_location="cpu")
    state_b = torch.load(second, map_location="cpu")
    for key in state_a:
        torch.testing.assert_close(state_a[key], state_b[key])


def test_ddpm_and_tdncnn_share_same_materialized_corpus(tmp_path: Path) -> None:
    _, tensor_path, manifest_path = _materialize(tmp_path)
    ddpm_ds = MaterializedCorpusDataset(tensor_path, manifest_path, normalized=True)
    tdncnn_ds = MaterializedCorpusDataset(tensor_path, manifest_path, normalized=False)
    denormalized = (ddpm_ds[4] + 1.0) / 2.0
    torch.testing.assert_close(denormalized, tdncnn_ds[4])


def test_first_timestep_mapping_in_resolved_config(tmp_path: Path) -> None:
    args = argparse.Namespace(
        dataset="mnist",
        output_root=tmp_path,
        smoke=True,
        max_train_samples=None,
        max_test_samples=None,
        ddpm_train_steps=None,
        ddpm_batch_size=None,
        tdncnn_batch_size=None,
        full_reference_epochs=None,
        downstream_total_steps=None,
        target_psnr_min_db=10.0,
        target_psnr_max_db=40.0,
        corpus_seed=42,
        split_seed=42,
        subset_seed=42,
        model_seed=42,
        eval_seed=4242,
        task_seed=42,
        corruption_version="vtest",
        eval_noise_version="vtest",
        task_noise_version="vtest",
        task_sigma_min=0.1,
        task_sigma_max=0.8,
        eval_sigma_min=0.1,
        eval_sigma_max=0.8,
        lr_ddpm=1e-4,
        lr_tdncnn=1e-3,
        quantile_min_points_per_bin=30,
        include_topk5=False,
        num_workers=0,
        download=False,
    )
    config = resolve_config(args)
    assert config["ddpm"]["score_timestep_report"] == 1
    assert config["ddpm"]["score_timestep_index"] == 0
    assert "eps_pred" in config["ddpm"]["score_formula"]


class ImageLabReplicationInvariantTests(unittest.TestCase):
    def _tmp(self):
        return tempfile.TemporaryDirectory()

    def test_same_original_index_has_identical_corpus(self) -> None:
        with self._tmp() as tmp:
            test_same_original_index_has_identical_corpus(Path(tmp))

    def test_corpus_independent_of_access_order(self) -> None:
        with self._tmp() as tmp:
            test_corpus_independent_of_access_order(Path(tmp))

    def test_corpus_independent_of_num_workers(self) -> None:
        with self._tmp() as tmp:
            test_corpus_independent_of_num_workers(Path(tmp))

    def test_manifest_matches_saved_corpus(self) -> None:
        with self._tmp() as tmp:
            test_manifest_matches_saved_corpus(Path(tmp))

    def test_topk_selects_minimum_scores(self) -> None:
        test_topk_selects_minimum_scores()

    def test_subsets_use_original_indices_and_equal_10_percent_sizes(self) -> None:
        test_subsets_use_original_indices_and_equal_10_percent_sizes()

    def test_random_selector_is_deterministic(self) -> None:
        test_random_selector_is_deterministic()

    def test_quantile_selector_returns_exact_size(self) -> None:
        test_quantile_selector_returns_exact_size()

    def test_eval_noise_cache_reused(self) -> None:
        with self._tmp() as tmp:
            test_eval_noise_cache_reused(Path(tmp))

    def test_train_and_eval_manifests_keep_split_labels(self) -> None:
        with self._tmp() as tmp:
            test_train_and_eval_manifests_keep_split_labels(Path(tmp))

    def test_downstream_initial_state_reused(self) -> None:
        with self._tmp() as tmp:
            test_downstream_initial_state_reused(Path(tmp))

    def test_ddpm_and_tdncnn_share_same_materialized_corpus(self) -> None:
        with self._tmp() as tmp:
            test_ddpm_and_tdncnn_share_same_materialized_corpus(Path(tmp))

    def test_first_timestep_mapping_in_resolved_config(self) -> None:
        with self._tmp() as tmp:
            test_first_timestep_mapping_in_resolved_config(Path(tmp))


if __name__ == "__main__":
    unittest.main()
