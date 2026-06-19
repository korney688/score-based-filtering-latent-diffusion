from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset
from torchvision import datasets, transforms


DATASET_SPECS: dict[str, dict[str, Any]] = {
    "mnist": {
        "torchvision_class": datasets.MNIST,
        "channels": 1,
        "image_size": 28,
        "train_size": 60000,
        "mean": (0.5,),
        "std": (0.5,),
        "display_name": "MNIST",
    },
    "cifar10": {
        "torchvision_class": datasets.CIFAR10,
        "channels": 3,
        "image_size": 32,
        "train_size": 50000,
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "display_name": "CIFAR-10",
    },
    "imagenet100": {
        "torchvision_class": None,
        "channels": 3,
        "image_size": 64,
        "train_size": None,
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "display_name": "ImageNet-100",
        "folder_name": "imagenet100",
        "requires_local_data": True,
        "dataset_type": "imagefolder",
    },
}


def dataset_name(dataset_cfg: Any | None = None, default: str = "mnist") -> str:
    if dataset_cfg is None:
        return default
    if isinstance(dataset_cfg, str):
        return dataset_cfg.lower()
    value = None
    if isinstance(dataset_cfg, dict):
        value = dataset_cfg.get("slug") or dataset_cfg.get("name")
    else:
        value = getattr(dataset_cfg, "slug", None) or getattr(dataset_cfg, "name", None)
    return str(value or default).lower()


def dataset_spec(dataset_cfg: Any | None = None) -> dict[str, Any]:
    name = dataset_name(dataset_cfg)
    if name not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset: {name}. Supported datasets: {sorted(DATASET_SPECS)}")
    return DATASET_SPECS[name]


def dataset_display_name(dataset_cfg: Any | None = None) -> str:
    return str(dataset_spec(dataset_cfg)["display_name"])


def dataset_channels(dataset_cfg: Any | None = None) -> int:
    cfg_channels = None
    if isinstance(dataset_cfg, dict):
        cfg_channels = dataset_cfg.get("channels")
    elif dataset_cfg is not None:
        cfg_channels = getattr(dataset_cfg, "channels", None)
    return int(cfg_channels or dataset_spec(dataset_cfg)["channels"])


def _cfg_get(dataset_cfg: Any | None, key: str, default: Any) -> Any:
    if dataset_cfg is None:
        return default
    if isinstance(dataset_cfg, dict):
        return dataset_cfg.get(key, default)
    return getattr(dataset_cfg, key, default)


def _cfg_sequence(dataset_cfg: Any | None, key: str, default: tuple[float, ...]) -> tuple[float, ...]:
    value = _cfg_get(dataset_cfg, key, default)
    return tuple(float(item) for item in value)


def _normalized_transform(dataset_cfg: Any | None, spec: dict[str, Any]):
    mean = _cfg_sequence(dataset_cfg, "mean", spec["mean"])
    std = _cfg_sequence(dataset_cfg, "std", spec["std"])
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def _imagefolder_transform(dataset_cfg: Any | None, spec: dict[str, Any], *, normalized: bool):
    image_size = int(_cfg_get(dataset_cfg, "image_size", spec["image_size"]))
    transform_steps = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
    if normalized:
        mean = _cfg_sequence(dataset_cfg, "mean", spec["mean"])
        std = _cfg_sequence(dataset_cfg, "std", spec["std"])
        transform_steps.append(transforms.Normalize(mean, std))
    return transforms.Compose(transform_steps)


def _imagefolder_root(data_root: str | Path, dataset_cfg: Any | None, spec: dict[str, Any]) -> Path:
    slug = dataset_name(dataset_cfg)
    env_root = os.environ.get(f"{slug.upper()}_ROOT")
    if env_root:
        return Path(env_root)
    cfg_root = _cfg_get(dataset_cfg, "local_root", None)
    if cfg_root:
        return Path(str(cfg_root))
    root = Path(data_root)
    if (root / "train").exists() or (root / "val").exists():
        return root
    folder_name = str(_cfg_get(dataset_cfg, "folder_name", spec.get("folder_name", slug)))
    if root.name == folder_name:
        return root
    return root / folder_name


def _imagefolder_split_root(data_root: str | Path, train: bool, dataset_cfg: Any | None, spec: dict[str, Any]) -> Path:
    dataset_root = _imagefolder_root(data_root, dataset_cfg, spec)
    split_name = "train" if train else "val"
    split_root = dataset_root / split_name
    if split_root.exists():
        return split_root

    display_name = str(spec.get("display_name", dataset_name(dataset_cfg)))
    env_name = f"{dataset_name(dataset_cfg).upper()}_ROOT"
    expected = (
        f"Expected local {display_name} data under {dataset_root} with prepared "
        "ImageFolder-compatible splits: train/<class>/*.JPEG and val/<class>/*.JPEG. "
        "If your data is elsewhere, set local_root in the dataset config, pass a data_root "
        f"override, or set {env_name} to the directory that "
        "contains train/ and val/."
    )
    raise FileNotFoundError(f"Missing {display_name} split directory: {split_root}. {expected}")


def build_torchvision_split(
    dataset_cfg: Any | None = None,
    *,
    train: bool,
    data_root: str | Path,
    transform_profile: str = "normalized",
    download: bool = False,
) -> Dataset:
    spec = dataset_spec(dataset_cfg)
    if transform_profile == "normalized":
        transform = _normalized_transform(dataset_cfg, spec)
    elif transform_profile == "tensor":
        transform = transforms.ToTensor()
    else:
        raise ValueError(f"Unsupported transform_profile: {transform_profile}")

    if spec.get("dataset_type") == "imagefolder" or _cfg_get(dataset_cfg, "dataset_type", None) == "imagefolder":
        if download:
            raise ValueError(
                f"{spec['display_name']} download is not supported. "
                f"Place local data under data/{spec.get('folder_name', dataset_name(dataset_cfg))}/."
            )
        transform = _imagefolder_transform(dataset_cfg, spec, normalized=transform_profile == "normalized")
        split_root = _imagefolder_split_root(data_root, train=train, dataset_cfg=dataset_cfg, spec=spec)
        return datasets.ImageFolder(root=split_root, transform=transform)

    dataset_cls = spec["torchvision_class"]
    return dataset_cls(
        root=Path(data_root),
        train=train,
        download=download,
        transform=transform,
    )
