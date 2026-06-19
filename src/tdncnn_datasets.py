from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.dataset_registry import build_torchvision_split


class OnlineNoisyDataset(Dataset):
    def __init__(self, dataset: Dataset, sigma_min: float = 0.1, sigma_max: float = 0.8):
        self.dataset = dataset
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.dataset[index]
        x_clean = sample[0] if isinstance(sample, (tuple, list)) else sample

        sigma = torch.empty(1).uniform_(self.sigma_min, self.sigma_max)
        epsilon = torch.randn_like(x_clean)
        x_noisy = x_clean + sigma.view(1, 1, 1) * epsilon

        return x_noisy.to(torch.float32), x_clean.to(torch.float32)


OnlineNoisyMNIST = OnlineNoisyDataset


def limit_dataset(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None:
        return dataset
    if max_samples <= 0:
        raise ValueError(f"max_samples must be positive, got {max_samples}")
    return Subset(dataset, list(range(min(max_samples, len(dataset)))))


def build_dataset_split(
    train: bool,
    data_root: str | Path,
    dataset_name: str = "mnist",
    download: bool = False,
) -> Dataset:
    # TDnCNN works directly on clean images in [0, 1].
    return build_torchvision_split(
        dataset_cfg=dataset_name,
        train=train,
        data_root=Path(data_root),
        transform_profile="tensor",
        download=download,
    )


def load_filtered_indices(filtered_indices) -> np.ndarray | None:
    if filtered_indices is None:
        return None

    if isinstance(filtered_indices, np.ndarray):
        return filtered_indices.astype(np.int64)

    if isinstance(filtered_indices, (list, tuple, set)):
        return np.asarray(list(filtered_indices), dtype=np.int64)

    path = Path(filtered_indices)
    if path.suffix != ".npy":
        raise ValueError(f"Stage 3 filtered indices must be a .npy file, got: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Missing selected indices file: {path}")

    return np.load(path).astype(np.int64)


def create_dataloaders(
    filtered_indices=None,
    batch_size: int = 32,
    mode: str = "full",
    seed: int = 42,
    data_root: str | Path = "data",
    dataset_name: str = "mnist",
    download: bool = False,
    sigma_min: float = 0.1,
    sigma_max: float = 0.8,
    num_workers: int = 0,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
):
    mode = mode.lower()
    if mode not in {"full", "filtered"}:
        raise ValueError(f"Unsupported mode: {mode}")

    train_base = build_dataset_split(train=True, data_root=data_root, dataset_name=dataset_name, download=download)
    test_base = build_dataset_split(train=False, data_root=data_root, dataset_name=dataset_name, download=download)

    loaded_filtered_indices = load_filtered_indices(filtered_indices)
    if mode == "filtered":
        if loaded_filtered_indices is None:
            raise ValueError("filtered_indices must be provided when mode='filtered'")
        train_base = Subset(train_base, np.sort(loaded_filtered_indices).tolist())

    train_base = limit_dataset(train_base, max_train_samples)
    test_base = limit_dataset(test_base, max_test_samples)

    train_dataset = OnlineNoisyDataset(train_base, sigma_min=sigma_min, sigma_max=sigma_max)
    test_dataset = OnlineNoisyDataset(test_base, sigma_min=sigma_min, sigma_max=sigma_max)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    print(f"Mode: {mode}")
    print(f"Train objects: {len(train_dataset)}")
    print(f"Test objects: {len(test_dataset)}")

    return train_loader, test_loader
