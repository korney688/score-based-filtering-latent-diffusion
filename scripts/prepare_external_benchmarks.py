from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


BENCHMARKS = ("Kodak24", "CBSD68", "Urban100")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate external denoising benchmark folders.")
    parser.add_argument("--root", type=Path, default=Path("data/external_benchmarks"))
    parser.add_argument("--strict", action="store_true", help="Fail if any benchmark folder or RGB image check is missing.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest output path. Defaults to <root>/dataset_manifest.json.",
    )
    return parser.parse_args()


def list_images(dataset_dir: Path) -> list[Path]:
    if not dataset_dir.exists():
        return []
    return sorted(path for path in dataset_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def inspect_image(path: Path) -> dict[str, object]:
    try:
        with Image.open(path) as image:
            return {
                "path": str(path),
                "valid": True,
                "mode": image.mode,
                "width": int(image.width),
                "height": int(image.height),
                "is_rgb": image.mode == "RGB",
            }
    except Exception as error:  # noqa: BLE001 - manifest should record validation errors.
        return {
            "path": str(path),
            "valid": False,
            "error": str(error),
            "is_rgb": False,
        }


def build_manifest(root: Path) -> dict[str, object]:
    datasets = {}
    for name in BENCHMARKS:
        dataset_dir = root / name
        images = list_images(dataset_dir)
        inspected = [inspect_image(path) for path in images]
        invalid = [item for item in inspected if not item.get("valid") or not item.get("is_rgb")]
        datasets[name] = {
            "path": str(dataset_dir),
            "exists": dataset_dir.is_dir(),
            "image_count": len(images),
            "valid_rgb_count": len(inspected) - len(invalid),
            "invalid_count": len(invalid),
            "invalid_images": invalid,
        }
    return {
        "root": str(root),
        "benchmarks": list(BENCHMARKS),
        "datasets": datasets,
    }


def main() -> None:
    args = parse_args()
    root = args.root
    manifest_path = args.manifest or root / "dataset_manifest.json"

    manifest = build_manifest(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"external_benchmark_root={root}")
    print(f"manifest={manifest_path}")
    has_error = False
    for name in BENCHMARKS:
        info = manifest["datasets"][name]
        print(
            f"{name}: exists={info['exists']} images={info['image_count']} "
            f"valid_rgb={info['valid_rgb_count']} invalid={info['invalid_count']}"
        )
        if not info["exists"] or info["image_count"] == 0 or info["invalid_count"] > 0:
            has_error = True

    if args.strict and has_error:
        raise SystemExit("External benchmark validation failed. See dataset_manifest.json.")


if __name__ == "__main__":
    main()
