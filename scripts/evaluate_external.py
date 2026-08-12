from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as skimage_ssim
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.DRUNet_image import build_drunet
from src.TDnCNN_image import DnCNN_2D, align_lpips_scaling_layer, prepare_for_lpips


BENCHMARKS = ("Kodak24", "CBSD68", "Urban100")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SUPPORTED_SIGMAS = {15, 25, 50}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate denoising checkpoints on external benchmarks.")
    parser.add_argument("--model-type", choices=["drunet", "tdncnn"], default="drunet")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to a trained DRUNet .pth checkpoint.")
    parser.add_argument("--full-checkpoint", type=Path, default=None)
    parser.add_argument("--topk-checkpoint", type=Path, default=None)
    parser.add_argument("--quantile-checkpoint", type=Path, default=None)
    parser.add_argument("--dataset", choices=BENCHMARKS, default=None, help="Single benchmark to evaluate.")
    parser.add_argument("--all-benchmarks", action="store_true", help="Evaluate Kodak24, CBSD68, and Urban100.")
    parser.add_argument("--root", type=Path, default=Path("data/external_benchmarks"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/external_benchmarks"))
    parser.add_argument("--sigma", type=int, default=25, choices=sorted(SUPPORTED_SIGMAS), help="Gaussian noise sigma in /255 units.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qualitative-count", type=int, default=5)
    parser.add_argument("--lpips", action="store_true", help="Compute LPIPS if pretrained weights are available.")
    parser.add_argument("--audit-only", action="store_true", help="Only inspect dataset structure and write audit files.")
    parser.add_argument("--max-images", type=int, default=None, help="Limit images per benchmark; use 1 for dry-run.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate evaluation plumbing on the first image only. Does not require a checkpoint.",
    )
    return parser.parse_args()


class IdentityDenoiser(torch.nn.Module):
    def forward(self, x: torch.Tensor, sigma=None) -> torch.Tensor:
        return x


def list_images(dataset_dir: Path) -> list[Path]:
    return sorted(path for path in dataset_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return None


def has_token(path: Path, tokens: tuple[str, ...]) -> bool:
    text = str(path).lower()
    return any(token in text for token in tokens)


def strip_pair_tokens(path: Path) -> str:
    stem = path.stem.lower()
    for token in ["clean", "gt", "groundtruth", "ground_truth", "reference", "ref", "target", "noisy", "noise", "sigma15", "sigma25", "sigma50"]:
        stem = stem.replace(token, "")
    return "".join(ch for ch in stem if ch.isalnum())


def audit_benchmark(root: Path, benchmark: str) -> dict[str, Any]:
    dataset_dir = root / benchmark
    files = list_images(dataset_dir) if dataset_dir.exists() else []
    extensions = sorted({path.suffix.lower() for path in files})
    clean_tokens = ("clean", "gt", "groundtruth", "ground_truth", "reference", "ref", "target")
    noisy_tokens = ("noisy", "noise", "sigma15", "sigma25", "sigma50")
    clean_files = [path for path in files if has_token(path.relative_to(dataset_dir), clean_tokens)]
    noisy_files = [path for path in files if has_token(path.relative_to(dataset_dir), noisy_tokens)]
    sigma_files = [path for path in files if has_token(path.relative_to(dataset_dir), ("sigma15", "sigma25", "sigma50"))]
    top_level_only = all(path.parent == dataset_dir for path in files)
    pair_rows = []
    clean_by_key = {strip_pair_tokens(path): path for path in clean_files}
    noisy_by_key = {strip_pair_tokens(path): path for path in noisy_files}
    for key in sorted(set(clean_by_key) & set(noisy_by_key)):
        clean_size = image_size(clean_by_key[key])
        noisy_size = image_size(noisy_by_key[key])
        pair_rows.append(
            {
                "key": key,
                "clean": str(clean_by_key[key].relative_to(dataset_dir)),
                "noisy": str(noisy_by_key[key].relative_to(dataset_dir)),
                "clean_size": clean_size,
                "noisy_size": noisy_size,
                "size_match": clean_size == noisy_size,
            }
        )
    if files and not clean_files and not noisy_files and top_level_only:
        status = "clean_only"
    elif clean_files and noisy_files and pair_rows and all(row["size_match"] for row in pair_rows):
        status = "paired_clean_noisy"
    elif files and noisy_files and not clean_files:
        status = "noisy_only"
    else:
        status = "unknown_structure" if files else "unknown_structure"
    sample_names = [str(path.relative_to(dataset_dir)) for path in files[:10]]
    sample_sizes = {str(path.relative_to(dataset_dir)): image_size(path) for path in files[:10]}
    special_dirs = sorted(
        {
            str(path.parent.relative_to(dataset_dir))
            for path in files
            if str(path.parent.relative_to(dataset_dir)) != "."
            and has_token(path.parent.relative_to(dataset_dir), clean_tokens + noisy_tokens)
        }
    )
    return {
        "benchmark": benchmark,
        "path": str(dataset_dir),
        "exists": dataset_dir.exists(),
        "file_count": len(files),
        "extensions": extensions,
        "sample_names": sample_names,
        "sample_sizes": sample_sizes,
        "clean_file_count": len(clean_files),
        "noisy_file_count": len(noisy_files),
        "sigma_file_count": len(sigma_files),
        "gt_reference_available": bool(clean_files) or status == "clean_only",
        "paired_size_checks": pair_rows,
        "special_dirs": special_dirs,
        "status": status,
    }


def audit_datasets(root: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "root": str(root.resolve()),
        "benchmarks": {benchmark: audit_benchmark(root, benchmark) for benchmark in BENCHMARKS},
    }
    (output_root / "dataset_audit.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# External Benchmarks Dataset Audit", ""]
    for benchmark in BENCHMARKS:
        item = payload["benchmarks"][benchmark]
        lines.extend(
            [
                f"## {benchmark}",
                f"- Status: `{item['status']}`",
                f"- Path: `{item['path']}`",
                f"- File count: `{item['file_count']}`",
                f"- Extensions: `{', '.join(item['extensions'])}`",
                f"- Clean/GT/reference files detected: `{item['clean_file_count']}`",
                f"- Noisy/sigma files detected: `{item['noisy_file_count']}`",
                f"- Sigma-tagged files detected: `{item['sigma_file_count']}`",
                f"- Special dirs: `{', '.join(item['special_dirs']) if item['special_dirs'] else 'none'}`",
                f"- First files: `{', '.join(item['sample_names'][:5])}`",
                f"- First image sizes: `{item['sample_sizes']}`",
                f"- Paired size checks: `{item['paired_size_checks'] if item['paired_size_checks'] else 'none'}`",
                "",
            ]
        )
    text = "\n".join(lines)
    (output_root / "dataset_audit.md").write_text(text + "\n", encoding="utf-8")
    (output_root / "external_benchmarks_dataset_audit.md").write_text(text + "\n", encoding="utf-8")
    return payload


def pad_to_multiple_of_8(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    _, _, height, width = x.shape
    pad_h = (8 - height % 8) % 8
    pad_w = (8 - width % 8) % 8
    if pad_h == 0 and pad_w == 0:
        return x, (height, width)
    mode = "reflect" if height > 1 and width > 1 else "replicate"
    return F.pad(x, (0, pad_w, 0, pad_h), mode=mode), (height, width)


def crop_to_original(x: torch.Tensor, original_hw: tuple[int, int]) -> torch.Tensor:
    height, width = original_hw
    return x[..., :height, :width]


def psnr_from_mse(mse: float) -> float:
    return 20.0 * math.log10(1.0 / math.sqrt(mse + 1e-8))


def load_rgb_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    return to_tensor(rgb).unsqueeze(0)


def load_drunet_checkpoint(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    model = build_drunet(official=True).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(payload)!r}")
    if any(str(key).startswith("module.") for key in state_dict):
        state_dict = {str(key).removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def load_tdncnn_checkpoint(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    model = DnCNN_2D(in_channels=3).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(payload)!r}")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def create_lpips_model(device: torch.device):
    if device.type == "cpu":
        lpips_device = torch.device("cpu")
    else:
        lpips_device = device
    try:
        import lpips

        model = lpips.LPIPS(net="alex").to(lpips_device)
        model.eval()
        align_lpips_scaling_layer(model, lpips_device)
        return model
    except Exception as error:  # noqa: BLE001 - LPIPS is optional for this stage.
        print(f"LPIPS disabled: {error}")
        return None


def compute_lpips_optional(x_pred: torch.Tensor, x_clean: torch.Tensor, lpips_model) -> float | None:
    if lpips_model is None:
        return None
    device = next(lpips_model.parameters()).device
    with torch.inference_mode():
        value = lpips_model(prepare_for_lpips(x_pred.to(device)), prepare_for_lpips(x_clean.to(device)))
    return float(value.mean().item())


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, lpips_model) -> dict[str, float | None]:
    mse = float(((pred.clamp(0, 1) - target.clamp(0, 1)) ** 2).mean().item())
    psnr = psnr_from_mse(mse)
    pred_np = pred.squeeze(0).detach().cpu().clamp(0, 1).numpy()
    target_np = target.squeeze(0).detach().cpu().clamp(0, 1).numpy()
    ssim = float(skimage_ssim(target_np, pred_np, data_range=1.0, channel_axis=0))
    lpips_value = compute_lpips_optional(pred.detach().cpu(), target.detach().cpu(), lpips_model)
    return {"MSE": mse, "PSNR": psnr, "SSIM": ssim, "LPIPS": lpips_value}


def save_qualitative(
    qualitative_dir: Path,
    stem: str,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    denoised: torch.Tensor,
) -> None:
    qualitative_dir.mkdir(parents=True, exist_ok=True)
    clean_cpu = clean.squeeze(0).detach().cpu().clamp(0.0, 1.0)
    noisy_cpu = noisy.squeeze(0).detach().cpu().clamp(0.0, 1.0)
    denoised_cpu = denoised.squeeze(0).detach().cpu().clamp(0.0, 1.0)
    error_map = (denoised_cpu - clean_cpu).abs().mean(dim=0, keepdim=True)

    save_image(clean_cpu, qualitative_dir / f"{stem}_gt.png")
    save_image(noisy_cpu, qualitative_dir / f"{stem}_noisy.png")
    save_image(denoised_cpu, qualitative_dir / f"{stem}_denoised.png")
    save_image(error_map, qualitative_dir / f"{stem}_error_map.png")


def load_pairs_for_status(root: Path, benchmark: str, audit_item: dict[str, Any], sigma: int, seed: int, max_images: int | None) -> list[dict[str, Any]]:
    dataset_dir = root / benchmark
    files = list_images(dataset_dir)
    if max_images is not None:
        files = files[:max_images]
    if audit_item["status"] != "clean_only":
        raise NotImplementedError("Only clean_only is implemented for the current benchmark structure.")
    pairs = []
    sigma_value = sigma / 255.0
    for image_idx, path in enumerate(files):
        clean = load_rgb_tensor(path)
        seed_payload = f"{benchmark}|{path.name}|{seed}|{sigma}".encode("utf-8")
        sample_seed = int(hashlib.sha256(seed_payload).hexdigest()[:16], 16) & ((1 << 63) - 1)
        gen = torch.Generator(device="cpu").manual_seed(sample_seed)
        noise = torch.randn(clean.shape, generator=gen, dtype=clean.dtype) * sigma_value
        noisy = (clean + noise).clamp(0.0, 1.0)
        pairs.append(
            {
                "benchmark": benchmark,
                "image": str(path.relative_to(dataset_dir)),
                "clean": clean,
                "noisy": noisy,
                "has_reference": True,
                "protocol_status": "clean_only",
                "sigma": sigma,
            }
        )
    return pairs


def predict_model(model: torch.nn.Module, x: torch.Tensor, model_type: str, sigma_value: float, device: torch.device) -> torch.Tensor:
    x = x.to(device)
    if model_type == "drunet":
        padded, original_hw = pad_to_multiple_of_8(x)
        with torch.inference_mode():
            pred = model(padded, sigma=sigma_value).clamp(0.0, 1.0)
        return crop_to_original(pred, original_hw).detach().cpu()
    with torch.inference_mode():
        return model(x).clamp(0.0, 1.0).detach().cpu()


def summarize_metrics(per_image: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (benchmark, variant), group in per_image.groupby(["Benchmark", "Variant"], sort=False):
        row = {"Benchmark": benchmark, "Variant": variant}
        for metric in ["PSNR", "SSIM", "LPIPS", "MSE"]:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[metric] = float(values.mean()) if values.notna().any() else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    out_rows = []
    for benchmark, group in summary.groupby("Benchmark", sort=False):
        full = group[group["Variant"] == "Full"].iloc[0]
        for _, row in group.iterrows():
            item = row.to_dict()
            item["Delta PSNR vs Full"] = float(row["PSNR"] - full["PSNR"]) if pd.notna(row["PSNR"]) and pd.notna(full["PSNR"]) else np.nan
            item["Delta SSIM vs Full"] = float(row["SSIM"] - full["SSIM"]) if pd.notna(row["SSIM"]) and pd.notna(full["SSIM"]) else np.nan
            item["Delta LPIPS vs Full"] = float(row["LPIPS"] - full["LPIPS"]) if pd.notna(row["LPIPS"]) and pd.notna(full["LPIPS"]) else np.nan
            out_rows.append(item)
    return pd.DataFrame(out_rows)


def markdown_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def tensor_image(x: torch.Tensor, width: int, height: int) -> Image.Image:
    arr = x.squeeze(0).detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    img = Image.fromarray((arr * 255).round().astype(np.uint8))
    img.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(img, ((width - img.width) // 2, (height - img.height) // 2))
    return canvas


def write_panel(output: Path, representatives: dict[str, dict[str, torch.Tensor]], metrics: pd.DataFrame, has_reference: bool = True) -> None:
    cols = ["Clean target", "Noisy input", "Full", "Top-K", "Quantile"] if has_reference else ["Input", "Full", "Top-K", "Quantile"]
    img_w, img_h = 420, 260
    label_h, row_label_w = 86, 160
    width = row_label_w + len(cols) * img_w + 60
    height = label_h + len(BENCHMARKS) * (img_h + 92) + 30
    canvas = Image.new("RGB", (max(width, 2400), height), "white")
    draw = ImageDraw.Draw(canvas)
    header_font = font(26, True)
    text_font = font(19)
    small_font = font(16)
    for c, col in enumerate(cols):
        draw.text((row_label_w + c * img_w + 20, 22), col, font=header_font, fill=(0, 0, 0))
    for r, benchmark in enumerate(BENCHMARKS):
        y = label_h + r * (img_h + 92)
        draw.text((20, y + 95), benchmark, font=header_font, fill=(0, 0, 0))
        item = representatives.get(benchmark)
        if item is None:
            continue
        images = {
            "Clean target": item["clean"],
            "Noisy input": item["noisy"],
            "Input": item["noisy"],
            "Full": item["Full"],
            "Top-K": item["Top-K"],
            "Quantile": item["Quantile"],
        }
        for c, col in enumerate(cols):
            x = row_label_w + c * img_w + 20
            canvas.paste(tensor_image(images[col], img_w - 28, img_h), (x, y))
            if col in {"Full", "Top-K", "Quantile"}:
                row = metrics[(metrics["Benchmark"] == benchmark) & (metrics["Variant"] == col)]
                if not row.empty and pd.notna(row.iloc[0]["PSNR"]):
                    text = f"PSNR {row.iloc[0]['PSNR']:.2f}  SSIM {row.iloc[0]['SSIM']:.3f}\nLPIPS {row.iloc[0]['LPIPS']:.3f}"
                else:
                    text = "metrics: N/A"
                draw.multiline_text((x, y + img_h + 8), text, font=small_font, fill=(0, 0, 0), spacing=3)
            elif col == "Noisy input":
                row = metrics[(metrics["Benchmark"] == benchmark) & (metrics["Variant"] == "Noisy input")]
                if not row.empty:
                    draw.text((x, y + img_h + 8), f"PSNR {row.iloc[0]['PSNR']:.2f}", font=text_font, fill=(0, 0, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def evaluate_tdncnn_variants(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    audit = audit_datasets(args.root, args.output_root)
    missing_roots = [
        name
        for name in BENCHMARKS
        if not bool(audit["benchmarks"][name]["exists"]) or int(audit["benchmarks"][name]["file_count"]) == 0
    ]
    if missing_roots:
        raise FileNotFoundError(
            "External benchmark data root is missing or empty for "
            f"{missing_roots}. Passed --root={args.root}. "
            "Expected the benchmark directories Kodak24, CBSD68, Urban100 under this root."
        )
    unsupported = [
        name
        for name in BENCHMARKS
        if audit["benchmarks"][name]["status"] not in {"clean_only", "paired_clean_noisy", "noisy_only"}
    ]
    if unsupported:
        raise RuntimeError(f"Unsupported benchmark structure after audit: {unsupported}")
    checkpoints = {
        "Full": args.full_checkpoint,
        "Top-K": args.topk_checkpoint,
        "Quantile": args.quantile_checkpoint,
    }
    missing = [name for name, path in checkpoints.items() if path is None or not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint(s): {missing}")
    protocol = {
        "model_type": "tdncnn",
        "benchmarks": BENCHMARKS,
        "sigma": int(args.sigma),
        "sigma_normalized": float(args.sigma / 255.0),
        "seed": int(args.seed),
        "status_by_benchmark": {name: audit["benchmarks"][name]["status"] for name in BENCHMARKS},
        "checkpoints": {name: str(Path(path).resolve()) for name, path in checkpoints.items()},
        "dry_run": bool(args.dry_run or args.max_images == 1),
        "max_images": args.max_images,
    }
    try:
        import yaml

        (args.output_root / "protocol.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True), encoding="utf-8")
    except Exception:
        (args.output_root / "protocol.yaml").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    device = torch.device(args.device)
    lpips_model = create_lpips_model(device) if args.lpips else None
    models = {name: load_tdncnn_checkpoint(Path(path), device) for name, path in checkpoints.items()}
    per_rows = []
    representatives: dict[str, dict[str, torch.Tensor]] = {}
    for benchmark in BENCHMARKS if args.all_benchmarks else (args.dataset,):
        audit_item = audit["benchmarks"][benchmark]
        pairs = load_pairs_for_status(args.root, benchmark, audit_item, args.sigma, args.seed, 1 if args.dry_run else args.max_images)
        for image_idx, pair in enumerate(tqdm(pairs, desc=f"External {benchmark}")):
            clean = pair["clean"]
            noisy = pair["noisy"]
            noisy_metrics = compute_metrics(noisy, clean, lpips_model)
            per_rows.append({"Benchmark": benchmark, "Variant": "Noisy input", "Image": pair["image"], **noisy_metrics})
            rep_payload = {"clean": clean, "noisy": noisy}
            for variant, model in models.items():
                pred = predict_model(model, noisy, "tdncnn", args.sigma / 255.0, device)
                metric_row = compute_metrics(pred, clean, lpips_model)
                per_rows.append({"Benchmark": benchmark, "Variant": variant, "Image": pair["image"], **metric_row})
                if image_idx == 0:
                    rep_payload[variant] = pred
                    save_qualitative(args.output_root / "qualitative" / benchmark / variant, Path(pair["image"]).stem, clean, noisy, pred)
            if image_idx == 0:
                representatives[benchmark] = rep_payload
    per_image = pd.DataFrame(per_rows)
    per_image.to_csv(args.output_root / "per_image_metrics.csv", index=False)
    summary = summarize_metrics(per_image[per_image["Variant"] != "Noisy input"])
    summary.to_csv(args.output_root / "summary_metrics.csv", index=False)
    (args.output_root / "summary_metrics.md").write_text(markdown_table(summary) + "\n", encoding="utf-8")
    write_panel(args.output_root / "all_benchmarks_panel.png", representatives, summary, has_reference=True)
    print(f"dataset_audit={args.output_root / 'dataset_audit.md'}")
    print(f"protocol={args.output_root / 'protocol.yaml'}")
    print(f"per_image_metrics={args.output_root / 'per_image_metrics.csv'}")
    print(f"summary_metrics={args.output_root / 'summary_metrics.csv'}")
    print(f"panel={args.output_root / 'all_benchmarks_panel.png'}")


def evaluate_benchmark(
    model: torch.nn.Module,
    benchmark_name: str,
    root: Path,
    output_dir: Path,
    sigma: int,
    device: torch.device,
    seed: int,
    qualitative_count: int,
    lpips_model,
    max_images: int | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    dataset_dir = root / benchmark_name
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Missing benchmark directory: {dataset_dir}")
    image_paths = list_images(dataset_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in benchmark directory: {dataset_dir}")
    if max_images is not None:
        image_paths = image_paths[:max_images]

    from src.TDnCNN_image import compute_ssim_item

    output_dir.mkdir(parents=True, exist_ok=True)
    qualitative_dir = output_dir / "qualitative"
    rows = []
    sigma_value = sigma / 255.0
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    for image_idx, image_path in enumerate(image_paths):
        clean = load_rgb_tensor(image_path).to(device)
        noise = torch.randn_like(clean) * sigma_value
        noisy = (clean + noise).clamp(0.0, 1.0)

        padded_noisy, original_hw = pad_to_multiple_of_8(noisy)
        with torch.no_grad():
            denoised_padded = model(padded_noisy, sigma=sigma_value).clamp(0.0, 1.0)
        denoised = crop_to_original(denoised_padded, original_hw)

        mse = float(((denoised - clean) ** 2).mean().item())
        psnr = psnr_from_mse(mse)
        ssim = compute_ssim_item(denoised.squeeze(0).detach().cpu().numpy(), clean.squeeze(0).detach().cpu().numpy())
        lpips_value = compute_lpips_optional(denoised.detach().cpu(), clean.detach().cpu(), lpips_model)

        rows.append(
            {
                "image": str(image_path.relative_to(dataset_dir)),
                "width": int(clean.shape[-1]),
                "height": int(clean.shape[-2]),
                "sigma": sigma,
                "mse": mse,
                "psnr": psnr,
                "ssim": ssim,
                "lpips": "" if lpips_value is None else lpips_value,
            }
        )

        if image_idx < qualitative_count:
            save_qualitative(qualitative_dir, image_path.stem, clean, noisy, denoised)

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    psnr_values = [float(row["psnr"]) for row in rows]
    ssim_values = [float(row["ssim"]) for row in rows]
    lpips_values = [float(row["lpips"]) for row in rows if row["lpips"] != ""]
    summary = {
        "dataset": benchmark_name,
        "image_count": len(rows),
        "sigma": sigma,
        "sigma_normalized": sigma_value,
        "mean_psnr": sum(psnr_values) / len(psnr_values),
        "mean_ssim": sum(ssim_values) / len(ssim_values),
        "mean_lpips": (sum(lpips_values) / len(lpips_values)) if lpips_values else None,
        "metrics_csv": str(metrics_path),
        "qualitative_dir": str(qualitative_dir),
        "dry_run": dry_run,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(output_dir / "report.md", summary)
    return summary


def write_report(path: Path, summary: dict[str, object]) -> None:
    lpips_text = "N/A" if summary["mean_lpips"] is None else f"{float(summary['mean_lpips']):.6f}"
    content = "\n".join(
        [
            f"# {summary['dataset']} External Denoising Evaluation",
            "",
            f"- Dataset: `{summary['dataset']}`",
            f"- Image count: `{summary['image_count']}`",
            f"- Sigma: `{summary['sigma']}` (`{summary['sigma_normalized']:.8f}`)",
            f"- Mean PSNR: `{float(summary['mean_psnr']):.4f}`",
            f"- Mean SSIM: `{float(summary['mean_ssim']):.6f}`",
            f"- Mean LPIPS: `{lpips_text}`",
            "",
            f"Per-image metrics: `{summary['metrics_csv']}`",
            f"Qualitative examples: `{summary['qualitative_dir']}`",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.all_benchmarks and args.dataset is not None:
        raise SystemExit("Use either --dataset or --all-benchmarks, not both.")
    if not args.all_benchmarks and args.dataset is None:
        raise SystemExit("Provide --dataset or --all-benchmarks.")
    if args.audit_only:
        audit_datasets(args.root, args.output_root)
        print(f"dataset_audit={args.output_root / 'dataset_audit.md'}")
        print(f"external_benchmarks_dataset_audit={args.output_root / 'external_benchmarks_dataset_audit.md'}")
        return
    if args.model_type == "tdncnn":
        evaluate_tdncnn_variants(args)
        return

    device = torch.device(args.device)
    if args.dry_run:
        model = IdentityDenoiser().to(device).eval()
    else:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required unless --dry-run is used.")
        model = load_drunet_checkpoint(args.checkpoint, device)
    lpips_model = create_lpips_model(device) if args.lpips else None

    run_name = "dry_run" if args.dry_run or args.checkpoint is None else args.checkpoint.stem
    benchmarks = BENCHMARKS if args.all_benchmarks else (args.dataset,)
    summaries = []
    for benchmark in benchmarks:
        output_dir = args.output_root / run_name / benchmark
        summary = evaluate_benchmark(
            model=model,
            benchmark_name=benchmark,
            root=args.root,
            output_dir=output_dir,
            sigma=args.sigma,
            device=device,
            seed=args.seed,
            qualitative_count=1 if args.dry_run else args.qualitative_count,
            lpips_model=lpips_model,
            max_images=1 if args.dry_run else args.max_images,
            dry_run=args.dry_run,
        )
        summaries.append(summary)
        print(
            f"{benchmark}: images={summary['image_count']} sigma={summary['sigma']} "
            f"psnr={summary['mean_psnr']:.4f} ssim={summary['mean_ssim']:.6f}"
        )

    aggregate_path = args.output_root / run_name / "summary.json"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(
        json.dumps({"checkpoint": None if args.checkpoint is None else str(args.checkpoint), "dry_run": args.dry_run, "summaries": summaries}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
