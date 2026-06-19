"""Prepare a tiny ImageFolder smoke dataset from CIFAR-10.

This script is separate from the training pipeline. It creates:

    data/smoke_imagenet/train/<class>/*.jpg
    data/smoke_imagenet/val/<class>/*.jpg

The image size is read from configs/dataset/imagenet100.yaml so the resulting
dataset matches the spatial size expected by the ImageNet-100 config.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from omegaconf import OmegaConf
from PIL import Image
from torchvision.datasets import CIFAR10


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "smoke_imagenet"
DEFAULT_CIFAR_ROOT = PROJECT_ROOT / "data" / ".cache" / "cifar10"
DEFAULT_IMAGENET_CONFIG = PROJECT_ROOT / "configs" / "dataset" / "imagenet100.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a small ImageFolder-compatible smoke dataset from CIFAR-10.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cifar-root", type=Path, default=DEFAULT_CIFAR_ROOT)
    parser.add_argument("--imagenet-config", type=Path, default=DEFAULT_IMAGENET_CONFIG)
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_image_size(config_path: Path) -> int:
    cfg = OmegaConf.load(config_path)
    image_size = int(cfg.get("image_size", 64))
    if image_size <= 0:
        raise ValueError(f"Expected positive image_size in {config_path}, got {image_size}")
    return image_size


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Pass --overwrite to recreate the smoke dataset."
            )
        shutil.rmtree(output_dir)
    (output_dir / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "val").mkdir(parents=True, exist_ok=True)


def safe_class_name(name: str, class_idx: int) -> str:
    normalized = "".join(char if char.isalnum() else "_" for char in name.lower()).strip("_")
    return f"{class_idx:03d}_{normalized}"


def save_split(
    dataset: CIFAR10,
    split_dir: Path,
    total_items: int,
    image_size: int,
) -> tuple[int, int]:
    class_names = [safe_class_name(name, idx) for idx, name in enumerate(dataset.classes)]
    per_class_limit = total_items // len(class_names)
    remainder = total_items % len(class_names)
    target_counts = {
        class_idx: per_class_limit + (1 if class_idx < remainder else 0)
        for class_idx in range(len(class_names))
    }
    saved_counts = {class_idx: 0 for class_idx in range(len(class_names))}

    for class_name in class_names:
        (split_dir / class_name).mkdir(parents=True, exist_ok=True)

    for image, label in dataset:
        if saved_counts[label] >= target_counts[label]:
            continue

        class_name = class_names[label]
        item_idx = saved_counts[label]
        output_path = split_dir / class_name / f"{item_idx:05d}.jpg"
        resized = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
        resized.save(output_path, format="JPEG", quality=95)
        saved_counts[label] += 1

        if sum(saved_counts.values()) >= total_items:
            break

    saved_total = sum(saved_counts.values())
    if saved_total != total_items:
        raise RuntimeError(f"Requested {total_items} images, saved {saved_total}")

    nonempty_classes = sum(1 for count in saved_counts.values() if count > 0)
    return nonempty_classes, saved_total


def main() -> None:
    args = parse_args()
    if args.train_size <= 0 or args.val_size <= 0:
        raise ValueError("--train-size and --val-size must be positive")

    image_size = load_image_size(args.imagenet_config)
    output_dir = args.output_dir.resolve()
    cifar_root = args.cifar_root.resolve()

    prepare_output_dir(output_dir, overwrite=args.overwrite)

    train_dataset = CIFAR10(root=cifar_root, train=True, download=True)
    val_dataset = CIFAR10(root=cifar_root, train=False, download=True)

    train_classes, train_images = save_split(
        dataset=train_dataset,
        split_dir=output_dir / "train",
        total_items=args.train_size,
        image_size=image_size,
    )
    val_classes, val_images = save_split(
        dataset=val_dataset,
        split_dir=output_dir / "val",
        total_items=args.val_size,
        image_size=image_size,
    )

    print("Smoke ImageFolder dataset created:")
    print(f"  output_dir: {output_dir}")
    print(f"  image_size: {image_size}x{image_size}")
    print(f"  train classes: {train_classes}")
    print(f"  val classes: {val_classes}")
    print(f"  train images: {train_images}")
    print(f"  val images: {val_images}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
