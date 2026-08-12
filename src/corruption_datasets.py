from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


EPS = 1e-12


def stable_int_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16) & ((1 << 63) - 1)


def psnr_from_mse(mse: float) -> float:
    return float(10.0 * math.log10(1.0 / max(float(mse), EPS)))


def snr_from_powers(signal_power: float, noise_power: float) -> float:
    return float(10.0 * math.log10(max(float(signal_power), EPS) / max(float(noise_power), EPS)))


def tensor_class_id(sample: Any) -> tuple[torch.Tensor, int]:
    if isinstance(sample, (tuple, list)):
        x = sample[0]
        class_id = int(sample[1]) if len(sample) > 1 and sample[1] is not None else -1
    else:
        x = sample
        class_id = -1
    if not isinstance(x, torch.Tensor):
        raise TypeError("Expected dataset sample image to be a torch.Tensor")
    return x.to(torch.float32).clamp(0.0, 1.0), class_id


def _metric_row(
    *,
    x_clean: torch.Tensor,
    x_preclip: torch.Tensor,
    x_corpus: torch.Tensor,
) -> dict[str, float]:
    pre_noise = x_preclip - x_clean
    post_noise = x_corpus - x_clean
    signal_power = float((x_clean**2).mean().item())
    noise_power_pre = float((pre_noise**2).mean().item())
    noise_power_post = float((post_noise**2).mean().item())
    clipped_fraction = float(((x_preclip < 0.0) | (x_preclip > 1.0)).to(torch.float32).mean().item())
    return {
        "realized_psnr_db_preclip": psnr_from_mse(noise_power_pre),
        "realized_psnr_db_postclip": psnr_from_mse(noise_power_post),
        "realized_snr_db_preclip": snr_from_powers(signal_power, noise_power_pre),
        "realized_snr_db_postclip": snr_from_powers(signal_power, noise_power_post),
        "signal_power": signal_power,
        "noise_power": noise_power_pre,
        "noise_power_postclip": noise_power_post,
        "clipped_fraction": clipped_fraction,
    }


def materialize_variable_awgn_corpus(
    *,
    dataset: Dataset,
    dataset_name: str,
    split: str,
    original_indices: Iterable[int],
    output_dir: str | Path,
    target_psnr_min_db: float,
    target_psnr_max_db: float,
    corpus_seed: int,
    corruption_version: str,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / f"{split}_corpus.pt"
    manifest_path = output_dir / f"{split}_manifest.csv"
    if tensor_path.exists() and manifest_path.exists() and not overwrite:
        return tensor_path, manifest_path

    rows: list[dict[str, Any]] = []
    corpus_tensors: list[torch.Tensor] = []
    for cache_pos, original_index in enumerate(list(original_indices)):
        x_clean, class_id = tensor_class_id(dataset[int(original_index)])
        sample_seed = stable_int_seed(dataset_name, split, int(original_index), corpus_seed, corruption_version)
        generator = torch.Generator().manual_seed(sample_seed)
        u = torch.rand((), generator=generator).item()
        target_psnr_db = float(target_psnr_min_db + (target_psnr_max_db - target_psnr_min_db) * u)
        sigma = float(10.0 ** (-target_psnr_db / 20.0))
        noise = torch.randn(x_clean.shape, generator=generator, dtype=x_clean.dtype) * sigma
        x_preclip = x_clean + noise
        x_corpus = x_preclip.clamp(0.0, 1.0).to(torch.float32)
        cache_key = f"{dataset_name}:{split}:{int(original_index)}:{corpus_seed}:{corruption_version}"
        metric_values = _metric_row(x_clean=x_clean, x_preclip=x_preclip, x_corpus=x_corpus)

        rows.append(
            {
                "dataset": dataset_name,
                "split": split,
                "original_index": int(original_index),
                "cache_pos": int(cache_pos),
                "class_id": int(class_id),
                "corpus_seed": int(corpus_seed),
                "sample_seed": int(sample_seed),
                "target_psnr_db": target_psnr_db,
                "sigma": sigma,
                **metric_values,
                "corruption_type": "variable_awgn_quality",
                "corruption_version": corruption_version,
                "cache_key": cache_key,
            }
        )
        corpus_tensors.append(x_corpus)

    torch.save(torch.stack(corpus_tensors, dim=0), tensor_path)
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return tensor_path, manifest_path


def materialize_eval_noise_cache(
    *,
    dataset: Dataset,
    dataset_name: str,
    original_indices: Iterable[int],
    output_dir: str | Path,
    sigma_min: float,
    sigma_max: float,
    eval_seed: int,
    eval_noise_version: str,
    overwrite: bool = False,
) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "test_eval_input.pt"
    target_path = output_dir / "test_eval_target.pt"
    manifest_path = output_dir / "test_noise_manifest.csv"
    if input_path.exists() and target_path.exists() and manifest_path.exists() and not overwrite:
        return input_path, target_path, manifest_path

    inputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    for cache_pos, original_index in enumerate(list(original_indices)):
        x_clean, class_id = tensor_class_id(dataset[int(original_index)])
        sample_seed = stable_int_seed(dataset_name, "test_eval_noise", int(original_index), eval_seed, eval_noise_version)
        generator = torch.Generator().manual_seed(sample_seed)
        u = torch.rand((), generator=generator).item()
        sigma = float(sigma_min + (sigma_max - sigma_min) * u)
        noise = torch.randn(x_clean.shape, generator=generator, dtype=x_clean.dtype) * sigma
        x_preclip = x_clean + noise
        x_input = x_preclip.clamp(0.0, 1.0).to(torch.float32)
        metric_values = _metric_row(x_clean=x_clean, x_preclip=x_preclip, x_corpus=x_input)
        rows.append(
            {
                "dataset": dataset_name,
                "split": "test",
                "original_index": int(original_index),
                "cache_pos": int(cache_pos),
                "class_id": int(class_id),
                "eval_seed": int(eval_seed),
                "sample_seed": int(sample_seed),
                "sigma": sigma,
                **metric_values,
                "eval_noise_version": eval_noise_version,
            }
        )
        inputs.append(x_input)
        targets.append(x_clean.to(torch.float32))

    torch.save(torch.stack(inputs, dim=0), input_path)
    torch.save(torch.stack(targets, dim=0), target_path)
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return input_path, target_path, manifest_path


class MaterializedCorpusDataset(Dataset):
    def __init__(
        self,
        tensor_path: str | Path,
        manifest_path: str | Path,
        *,
        original_indices: Iterable[int] | None = None,
        normalized: bool = False,
        return_original_index: bool = False,
    ) -> None:
        self.tensor_path = Path(tensor_path)
        self.manifest_path = Path(manifest_path)
        self.corpus = torch.load(self.tensor_path, map_location="cpu").to(torch.float32)
        self.manifest = pd.read_csv(self.manifest_path)
        self.normalized = bool(normalized)
        self.return_original_index = bool(return_original_index)
        self.pos_by_original = {
            int(row.original_index): int(row.cache_pos)
            for row in self.manifest[["original_index", "cache_pos"]].itertuples(index=False)
        }
        if original_indices is None:
            self.original_indices = self.manifest["original_index"].astype(int).to_numpy()
        else:
            self.original_indices = np.asarray(list(original_indices), dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.original_indices))

    def __getitem__(self, index: int):
        original_index = int(self.original_indices[int(index)])
        cache_pos = self.pos_by_original[original_index]
        x = self.corpus[cache_pos]
        if self.normalized:
            x = x * 2.0 - 1.0
        if self.return_original_index:
            return x.to(torch.float32), original_index
        return x.to(torch.float32)

    def tensor_for_original_index(self, original_index: int) -> torch.Tensor:
        return self.corpus[self.pos_by_original[int(original_index)]].to(torch.float32)


class CorpusTaskNoiseDataset(Dataset):
    def __init__(
        self,
        corpus_dataset: MaterializedCorpusDataset,
        *,
        dataset_name: str,
        task_seed: int,
        task_noise_version: str,
        sigma_min: float,
        sigma_max: float,
    ) -> None:
        self.corpus_dataset = corpus_dataset
        self.dataset_name = dataset_name
        self.task_seed = int(task_seed)
        self.task_noise_version = task_noise_version
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

    def __len__(self) -> int:
        return len(self.corpus_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_target, original_index = self.corpus_dataset[index]
        sample_seed = stable_int_seed(
            self.dataset_name,
            "train_task_noise",
            int(original_index),
            self.task_seed,
            self.task_noise_version,
        )
        generator = torch.Generator().manual_seed(sample_seed)
        u = torch.rand((), generator=generator).item()
        sigma = float(self.sigma_min + (self.sigma_max - self.sigma_min) * u)
        noise = torch.randn(x_target.shape, generator=generator, dtype=x_target.dtype) * sigma
        x_input = (x_target + noise).clamp(0.0, 1.0)
        return x_input.to(torch.float32), x_target.to(torch.float32)


class EvalNoiseDataset(Dataset):
    def __init__(self, input_path: str | Path, target_path: str | Path) -> None:
        self.inputs = torch.load(Path(input_path), map_location="cpu").to(torch.float32)
        self.targets = torch.load(Path(target_path), map_location="cpu").to(torch.float32)
        if self.inputs.shape != self.targets.shape:
            raise ValueError(f"Eval input/target shape mismatch: {self.inputs.shape} vs {self.targets.shape}")

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[int(index)], self.targets[int(index)]


def stratified_train_val_indices(
    manifest: pd.DataFrame,
    *,
    val_fraction: float,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    rng = np.random.default_rng(split_seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for _, group in manifest.groupby("class_id", sort=True):
        indices = group["original_index"].to_numpy(dtype=np.int64)
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 0
        val_parts.append(shuffled[:n_val])
        train_parts.append(shuffled[n_val:])
    train_idx = np.sort(np.concatenate(train_parts).astype(np.int64, copy=False))
    val_idx = np.sort(np.concatenate(val_parts).astype(np.int64, copy=False))
    return train_idx, val_idx
