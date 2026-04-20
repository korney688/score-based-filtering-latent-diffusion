from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class DenoisingDataset(Dataset):
    def __init__(self, clean_path, noisy_path, indices=None):
        self.clean_path = Path(clean_path)
        self.noisy_path = Path(noisy_path)

        with h5py.File(self.clean_path, "r") as clean_file:
            self.clean_data = clean_file["dataset"][:].astype(np.float32)

        with h5py.File(self.noisy_path, "r") as noisy_file:
            self.noisy_data = noisy_file["dataset"][:].astype(np.float32)

        if self.clean_data.shape != self.noisy_data.shape:
            raise ValueError(
                f"Clean/noisy shapes do not match: "
                f"{self.clean_data.shape} vs {self.noisy_data.shape}"
            )

        self.total_count = self.clean_data.shape[0]

        if indices is None:
            self.indices = np.arange(self.total_count, dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

        print(f"Total objects: {self.total_count}")
        print(f"Objects after filtering: {len(self.indices)}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_idx = int(self.indices[index])

        x_noisy = torch.from_numpy(self.noisy_data[sample_idx]).unsqueeze(0).to(torch.float32)
        x_clean = torch.from_numpy(self.clean_data[sample_idx]).unsqueeze(0).to(torch.float32)

        return x_noisy, x_clean


def load_filtered_indices(filtered_indices):
    if filtered_indices is None:
        return None, None

    if isinstance(filtered_indices, np.ndarray):
        return filtered_indices.astype(np.int64), None

    if isinstance(filtered_indices, (list, tuple, set)):
        return np.asarray(list(filtered_indices), dtype=np.int64), None

    path = Path(filtered_indices)

    if path.suffix == ".npy":
        return np.load(path).astype(np.int64), path

    if path.suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as filtered_file:
            if "selected_indices" in filtered_file:
                indices = filtered_file["selected_indices"][:].astype(np.int64)
            elif "indices" in filtered_file:
                indices = filtered_file["indices"][:].astype(np.int64)
            else:
                raise KeyError("Expected 'selected_indices' or 'indices' in filtered file.")
        return indices, path

    raise ValueError(f"Unsupported filtered_indices type: {type(filtered_indices)}")


def maybe_save_indices_npy(indices: np.ndarray, source_path: Path | None) -> Path | None:
    if source_path is None or source_path.suffix == ".npy":
        return source_path

    npy_path = source_path.with_suffix(".npy")
    if not npy_path.exists():
        np.save(npy_path, indices)
        print(f"Saved filtered indices to: {npy_path}")
    return npy_path


def split_indices(num_samples: int, split: float = 0.8, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < split < 1.0:
        raise ValueError(f"Split must be in (0, 1), got {split}")

    rng = np.random.default_rng(seed)
    all_indices = np.arange(num_samples, dtype=np.int64)
    rng.shuffle(all_indices)

    train_size = int(num_samples * split)
    train_indices = np.sort(all_indices[:train_size])
    test_indices = np.sort(all_indices[train_size:])

    return train_indices, test_indices


def create_dataloaders(
    clean_path,
    noisy_path,
    filtered_indices=None,
    batch_size: int = 32,
    split: float = 0.8,
    mode: str = "baseline",
    seed: int = 42,
):
    mode = mode.lower()
    if mode not in {"baseline", "filtered"}:
        raise ValueError(f"Unsupported mode: {mode}")

    with h5py.File(clean_path, "r") as clean_file:
        num_samples = clean_file["dataset"].shape[0]

    train_indices, test_indices = split_indices(num_samples=num_samples, split=split, seed=seed)

    loaded_filtered_indices, source_path = load_filtered_indices(filtered_indices)
    maybe_save_indices_npy(loaded_filtered_indices, source_path) if loaded_filtered_indices is not None else None

    if mode == "baseline":
        train_subset_indices = train_indices
    else:
        if loaded_filtered_indices is None:
            raise ValueError("filtered_indices must be provided when mode='filtered'")
        train_subset_indices = np.intersect1d(train_indices, loaded_filtered_indices, assume_unique=False)

    train_dataset = DenoisingDataset(
        clean_path=clean_path,
        noisy_path=noisy_path,
        indices=train_subset_indices,
    )
    test_dataset = DenoisingDataset(
        clean_path=clean_path,
        noisy_path=noisy_path,
        indices=test_indices,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Mode: {mode}")
    print(f"Train objects: {len(train_dataset)}")
    print(f"Test objects: {len(test_dataset)}")

    return train_loader, test_loader
