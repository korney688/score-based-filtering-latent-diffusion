from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF


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
    "mnist_noisy_gaussian": {
        "torchvision_class": datasets.MNIST,
        "channels": 1,
        "image_size": 28,
        "train_size": 60000,
        "mean": (0.5,),
        "std": (0.5,),
        "display_name": "MNIST noisy Gaussian",
    },
    "mnist_noisy_poisson_gaussian": {
        "torchvision_class": datasets.MNIST,
        "channels": 1,
        "image_size": 28,
        "train_size": 60000,
        "mean": (0.5,),
        "std": (0.5,),
        "display_name": "MNIST noisy Poisson-Gaussian",
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


def _cfg_mapping(dataset_cfg: Any | None, key: str) -> dict[str, Any]:
    value = _cfg_get(dataset_cfg, key, {})
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    return dict(value)


def load_dataset_config(slug: str, project_root: str | Path | None = None) -> dict[str, Any]:
    from omegaconf import OmegaConf

    root = Path(project_root or Path(__file__).resolve().parents[1])

    if not OmegaConf.has_resolver("hydra"):
        OmegaConf.register_new_resolver(
            "hydra",
            lambda key: str(root) if key == "runtime.cwd" else "",
            use_cache=False,
        )

    config_path = root / "configs" / "dataset" / f"{slug}.yaml"
    cfg = OmegaConf.load(config_path)
    wrapped = OmegaConf.create({"dataset": cfg})
    dataset_cfg = OmegaConf.to_container(wrapped.dataset, resolve=True)
    if not isinstance(dataset_cfg, dict):
        raise TypeError(f"Dataset config must be a mapping: {config_path}")
    return dataset_cfg


def apply_dataset_runtime_overrides(
    dataset_cfg: dict[str, Any],
    *,
    mode: str | None = None,
    corruption: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if mode is not None:
        dataset_cfg["dataset_mode"] = mode
    if corruption:
        dataset_cfg.setdefault("corruption", {})
        dataset_cfg["corruption"].update(corruption)
    return dataset_cfg


def _normalize_tensor(x: torch.Tensor, mean: tuple[float, ...], std: tuple[float, ...]) -> torch.Tensor:
    mean_tensor = torch.tensor(mean, dtype=x.dtype).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=x.dtype).view(-1, 1, 1)
    return (x - mean_tensor) / std_tensor


def dataset_mode(dataset_cfg: Any | None = None) -> str:
    return str(_cfg_get(dataset_cfg, "dataset_mode", "clean")).strip().lower().replace("-", "_")


def _canonical_corruption_cfg(dataset_cfg: Any | None, train: bool) -> dict[str, Any] | None:
    mode = dataset_mode(dataset_cfg)
    if mode not in {"clean", "noisy"}:
        raise ValueError(f"Unsupported dataset_mode: {mode}. Supported: clean, noisy.")

    if mode == "noisy":
        corruption_cfg = _cfg_mapping(dataset_cfg, "corruption")
        if not bool(corruption_cfg.get("enabled", False)):
            return None
        apply_to = str(corruption_cfg.get("apply_to", "all")).lower()
        if apply_to != "all" and not (train and apply_to == "train") and not ((not train) and apply_to in {"val", "test"}):
            return None
        return corruption_cfg

    # Backward-compatible support for older MNIST noisy configs without creating
    # a separate ImageNet-100 dataset mode.
    if bool(_cfg_get(dataset_cfg, "noisy_dataset", False)):
        noise_cfg = _cfg_mapping(dataset_cfg, "noise")
        apply_to = str(noise_cfg.get("apply_to", "train")).lower()
        if apply_to == "all" or (train and apply_to == "train") or ((not train) and apply_to in {"val", "test"}):
            return noise_cfg
    return None


def corruption_enabled(dataset_cfg: Any | None, train: bool) -> bool:
    return _canonical_corruption_cfg(dataset_cfg, train) is not None


class PoissonGaussianCorruption:
    def __init__(
        self,
        poisson_scale: float = 30.0,
        gaussian_sigma: float = 0.01,
        *,
        clip: bool = True,
    ) -> None:
        if poisson_scale <= 0:
            raise ValueError(f"poisson_scale must be positive, got {poisson_scale}")
        if gaussian_sigma < 0:
            raise ValueError(f"gaussian_sigma must be non-negative, got {gaussian_sigma}")
        self.poisson_scale = float(poisson_scale)
        self.gaussian_sigma = float(gaussian_sigma)
        self.clip = bool(clip)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "PoissonGaussianCorruption":
        kind = str(cfg.get("type", cfg.get("kind", "poisson_gaussian"))).lower().replace("-", "_")
        if kind != "poisson_gaussian":
            raise ValueError(f"Unsupported corruption type: {kind}. Supported: poisson_gaussian.")
        poisson_scale = float(cfg.get("poisson_scale", cfg.get("poisson_peak", cfg.get("peak", 30.0))))
        gaussian_sigma = float(cfg.get("gaussian_sigma", 0.01))
        return cls(
            poisson_scale=poisson_scale,
            gaussian_sigma=gaussian_sigma,
            clip=bool(cfg.get("clip", True)),
        )

    def __call__(self, image: torch.Tensor | Any, generator: torch.Generator | None = None) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            x = image.to(torch.float32)
            if x.numel() > 0 and float(x.max()) > 1.0:
                x = x / 255.0
        else:
            x = TF.to_tensor(image)
        x = x.clamp(0.0, 1.0)

        poisson = torch.poisson(x * self.poisson_scale, generator=generator) / self.poisson_scale
        gaussian = torch.randn(poisson.shape, generator=generator, dtype=poisson.dtype) * self.gaussian_sigma
        corrupted = poisson + gaussian
        if self.clip:
            corrupted = corrupted.clamp(0.0, 1.0)
        return corrupted.to(torch.float32)


class CorruptedDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        *,
        dataset_cfg: Any | None,
        corruption_cfg: dict[str, Any],
        spec: dict[str, Any],
        transform_profile: str,
        return_clean_target: bool = False,
    ) -> None:
        self.dataset = dataset
        self.corruption_cfg = corruption_cfg
        self.corruption = PoissonGaussianCorruption.from_config(corruption_cfg)
        self.transform_profile = transform_profile
        self.mean = _cfg_sequence(dataset_cfg, "mean", spec["mean"])
        self.std = _cfg_sequence(dataset_cfg, "std", spec["std"])
        self.return_clean_target = return_clean_target
        self.seed = int(corruption_cfg.get("seed", 42))

        for attr in ("classes", "class_to_idx", "targets"):
            if hasattr(dataset, attr):
                setattr(self, attr, getattr(dataset, attr))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        if isinstance(sample, (tuple, list)):
            x_clean, label = sample[0], sample[1]
        else:
            x_clean, label = sample, None

        if not isinstance(x_clean, torch.Tensor):
            x_clean = TF.to_tensor(x_clean)
        x_clean = x_clean.to(torch.float32).clamp(0.0, 1.0)

        generator = torch.Generator().manual_seed(self.seed + int(index))
        x_noisy = self.corruption(x_clean, generator=generator)
        if self.transform_profile == "normalized":
            x_out = _normalize_tensor(x_noisy, self.mean, self.std)
        elif self.transform_profile == "tensor":
            x_out = x_noisy
        else:
            raise ValueError(f"Unsupported transform_profile: {self.transform_profile}")

        if self.return_clean_target:
            return x_out.to(torch.float32), x_clean.to(torch.float32)
        if label is None:
            return x_out.to(torch.float32)
        return x_out.to(torch.float32), label


PhysicallyNoisyDataset = CorruptedDataset


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
    return_clean_for_corruption: bool = False,
) -> Dataset:
    spec = dataset_spec(dataset_cfg)
    corruption_cfg = _canonical_corruption_cfg(dataset_cfg, train)
    use_corruption = corruption_cfg is not None
    if use_corruption:
        transform = transforms.ToTensor()
    elif transform_profile == "normalized":
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
        transform = _imagefolder_transform(
            dataset_cfg,
            spec,
            normalized=False if use_corruption else transform_profile == "normalized",
        )
        split_root = _imagefolder_split_root(data_root, train=train, dataset_cfg=dataset_cfg, spec=spec)
        dataset = datasets.ImageFolder(root=split_root, transform=transform)
        if use_corruption:
            return CorruptedDataset(
                dataset,
                dataset_cfg=dataset_cfg,
                corruption_cfg=corruption_cfg,
                spec=spec,
                transform_profile=transform_profile,
                return_clean_target=return_clean_for_corruption,
            )
        return dataset

    dataset_cls = spec["torchvision_class"]
    dataset = dataset_cls(
        root=Path(data_root),
        train=train,
        download=download,
        transform=transform,
    )
    if use_corruption:
        return CorruptedDataset(
            dataset,
            dataset_cfg=dataset_cfg,
            corruption_cfg=corruption_cfg,
            spec=spec,
            transform_profile=transform_profile,
            return_clean_target=return_clean_for_corruption,
        )
    return dataset
