from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF


def deterministic_mnist_color(split: str, original_index: int, color_seed: int = 42) -> torch.Tensor:
    payload = f"{split}|{int(original_index)}|{int(color_seed)}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    seed = int(digest[:16], 16) & ((1 << 63) - 1)
    generator = torch.Generator().manual_seed(seed)
    return 0.25 + 0.75 * torch.rand((3,), generator=generator, dtype=torch.float32)


def deterministic_abstract_background(split: str, original_index: int, background_seed: int = 42, image_size: int = 28) -> torch.Tensor:
    payload = f"background|{split}|{int(original_index)}|{int(background_seed)}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    seed = int(digest[:16], 16) & ((1 << 63) - 1)
    generator = torch.Generator().manual_seed(seed)
    low = torch.rand((1, 3, 7, 7), generator=generator, dtype=torch.float32)
    smooth = F.interpolate(low, size=(int(image_size), int(image_size)), mode="bilinear", align_corners=False).squeeze(0)
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, int(image_size)),
        torch.linspace(0.0, 1.0, int(image_size)),
        indexing="ij",
    )
    phase = torch.rand((3, 1, 1), generator=generator, dtype=torch.float32) * 6.283185307179586
    freq = 1.5 + torch.rand((3, 1, 1), generator=generator, dtype=torch.float32) * 3.0
    waves = 0.5 + 0.5 * torch.sin(freq * (xx + yy).unsqueeze(0) * 6.283185307179586 + phase)
    palette = 0.25 + 0.75 * torch.rand((3, 1, 1), generator=generator, dtype=torch.float32)
    background = (0.55 * smooth + 0.30 * waves + 0.15 * palette).clamp(0.0, 1.0)
    return background.to(torch.float32)


class ColoredMNISTWrapper(Dataset):
    """Deterministic label-independent RGB MNIST colorization."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        split: str,
        color_seed: int = 42,
        background_seed: int = 42,
        mode: str = "abstract_background",
    ) -> None:
        self.dataset = dataset
        self.split = str(split)
        self.color_seed = int(color_seed)
        self.background_seed = int(background_seed)
        self.mode = str(mode).strip().lower()
        for attr in ("classes", "class_to_idx", "targets"):
            if hasattr(dataset, attr):
                setattr(self, attr, getattr(dataset, attr))

    def __len__(self) -> int:
        return len(self.dataset)

    def color_for_original_index(self, original_index: int) -> torch.Tensor:
        return deterministic_mnist_color(self.split, int(original_index), self.color_seed)

    def background_for_original_index(self, original_index: int, image_size: int = 28) -> torch.Tensor:
        return deterministic_abstract_background(self.split, int(original_index), self.background_seed, image_size)

    def metadata_for_original_index(self, original_index: int) -> dict[str, int | float]:
        color = self.color_for_original_index(int(original_index))
        return {
            "color_seed": int(self.color_seed),
            "background_seed": int(self.background_seed),
            "colorization_mode": self.mode,
            "color_r": float(color[0].item()),
            "color_g": float(color[1].item()),
            "color_b": float(color[2].item()),
        }

    def __getitem__(self, index: int) -> Any:
        sample = self.dataset[int(index)]
        if isinstance(sample, (tuple, list)):
            x_gray, label = sample[0], sample[1]
        else:
            x_gray, label = sample, None
        if not isinstance(x_gray, torch.Tensor):
            x_gray = TF.to_tensor(x_gray)
        x_gray = x_gray.to(torch.float32).clamp(0.0, 1.0)
        if x_gray.ndim == 2:
            x_gray = x_gray.unsqueeze(0)
        if x_gray.shape[0] != 1:
            raise ValueError(f"Colored MNIST expects a single-channel image, got shape {tuple(x_gray.shape)}")
        color = self.color_for_original_index(int(index)).to(dtype=x_gray.dtype)
        alpha = x_gray.clamp(0.0, 1.0)
        digit = color.view(3, 1, 1).expand(3, int(x_gray.shape[-2]), int(x_gray.shape[-1]))
        if self.mode in {"digit_black_background", "black_background"}:
            x_rgb = digit * alpha
        elif self.mode in {"abstract_background", "mnist_m_like"}:
            background = self.background_for_original_index(int(index), image_size=int(x_gray.shape[-1])).to(dtype=x_gray.dtype)
            x_rgb = background * (1.0 - alpha) + digit * alpha
        else:
            raise ValueError(f"Unsupported Colored MNIST mode: {self.mode}")
        x_rgb = x_rgb.clamp(0.0, 1.0)
        if label is None:
            return x_rgb
        return x_rgb, label


class ColoredMNISTDataset(ColoredMNISTWrapper):
    def __init__(
        self,
        *,
        root: str | Path,
        train: bool,
        download: bool = False,
        color_seed: int = 42,
        background_seed: int = 42,
        mode: str = "abstract_background",
    ) -> None:
        base = datasets.MNIST(root=Path(root), train=bool(train), download=bool(download), transform=transforms.ToTensor())
        super().__init__(
            base,
            split="train" if train else "test",
            color_seed=color_seed,
            background_seed=background_seed,
            mode=mode,
        )
