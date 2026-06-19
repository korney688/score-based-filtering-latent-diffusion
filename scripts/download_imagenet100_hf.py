"""Download ImageNet-100 from Hugging Face into ImageFolder layout.

This utility is intentionally separate from the training pipeline. It prepares:

    data/imagenet100/train/<class>/*
    data/imagenet100/val/<class>/*

Source dataset:
    https://huggingface.co/datasets/asafaa/imagent100
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile

from huggingface_hub import hf_hub_download, list_repo_files


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "asafaa/imagent100"
PART_RE = re.compile(r"^imagenet100_(train|val)_part\d+\.zip$")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract Hugging Face ImageNet-100 zipped ImageFolder chunks.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repo id.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "imagenet100",
        help="Target ImageFolder root. Defaults to data/imagenet100.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / ".cache" / "imagenet100_hf",
        help="Where downloaded ZIP chunks are cached.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded ZIP files after successful extraction.",
    )
    return parser.parse_args()


def discover_zip_parts(repo_id: str) -> list[str]:
    files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    zip_parts = sorted(path for path in files if PART_RE.match(Path(path).name))
    train_parts = [path for path in zip_parts if "_train_" in Path(path).name]
    val_parts = [path for path in zip_parts if "_val_" in Path(path).name]
    if not train_parts or not val_parts:
        raise RuntimeError(
            f"Could not find train/val ZIP chunks in {repo_id}. "
            "Expected files like imagenet100_train_part1.zip and imagenet100_val_part1.zip."
        )
    return zip_parts


def safe_extract_zip(zip_path: Path, output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"Unsafe path in archive {zip_path}: {member.filename}")
        archive.extractall(output_dir)


def count_classes(split_dir: Path) -> int:
    if not split_dir.exists():
        return 0
    return sum(1 for path in split_dir.iterdir() if path.is_dir())


def count_images(split_dir: Path) -> int:
    if not split_dir.exists():
        return 0
    return sum(
        1
        for path in split_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _merge_directory_contents(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if target.exists():
            if item.is_dir():
                _merge_directory_contents(item, target)
            elif item.is_file() and item.stat().st_size != target.stat().st_size:
                raise RuntimeError(f"Refusing to overwrite different existing file: {target}")
        else:
            shutil.move(str(item), str(target))


def normalize_imagefolder_layout(output_dir: Path) -> None:
    # Some prepared archives include an extra top-level directory. Normalize the
    # result to output_dir/train and output_dir/val without changing class names.
    for split in ("train", "val"):
        expected = output_dir / split
        if expected.exists():
            continue
        candidates = [
            path
            for path in output_dir.rglob(split)
            if path.is_dir() and path.parent != output_dir
        ]
        if not candidates:
            continue
        source = candidates[0]
        print(f"Normalizing {source} -> {expected}")
        _merge_directory_contents(source, expected)


def verify_imagefolder(output_dir: Path) -> dict[str, int]:
    stats = {
        "train_classes": count_classes(output_dir / "train"),
        "val_classes": count_classes(output_dir / "val"),
        "train_images": count_images(output_dir / "train"),
        "val_images": count_images(output_dir / "val"),
    }
    if stats["train_classes"] == 0 or stats["val_classes"] == 0:
        raise RuntimeError(
            "Extraction finished, but ImageFolder splits were not found. "
            f"Expected {output_dir / 'train'} and {output_dir / 'val'} with class subdirectories."
        )
    if stats["train_images"] == 0 or stats["val_images"] == 0:
        raise RuntimeError("Extraction finished, but no image files were found in train or val split.")
    return stats


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repository: {args.repo_id}")
    print(f"Output directory: {output_dir}")
    print(f"Archive cache: {cache_dir}")

    zip_parts = discover_zip_parts(args.repo_id)
    print(f"Found {len(zip_parts)} ZIP chunks:")
    for filename in zip_parts:
        print(f"  - {filename}")

    downloaded_paths: list[Path] = []
    for filename in zip_parts:
        print(f"Downloading {filename}")
        downloaded = hf_hub_download(
            repo_id=args.repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=cache_dir,
        )
        zip_path = Path(downloaded)
        downloaded_paths.append(zip_path)
        print(f"Extracting {zip_path.name}")
        safe_extract_zip(zip_path, output_dir)

    normalize_imagefolder_layout(output_dir)
    stats = verify_imagefolder(output_dir)
    print("ImageNet-100 ImageFolder verification:")
    print(f"  train classes: {stats['train_classes']}")
    print(f"  val classes: {stats['val_classes']}")
    print(f"  train images: {stats['train_images']}")
    print(f"  val images: {stats['val_images']}")
    print("Done.")

    if not args.keep_archives:
        print(f"Removing archive cache: {cache_dir}")
        shutil.rmtree(cache_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
