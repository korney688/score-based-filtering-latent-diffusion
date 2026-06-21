from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.DRUNet_image import build_drunet


BENCHMARKS = ("Kodak24", "CBSD68", "Urban100")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SUPPORTED_SIGMAS = {15, 25, 50}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DRUNet checkpoint on external denoising benchmarks.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to a trained DRUNet .pth checkpoint.")
    parser.add_argument("--dataset", choices=BENCHMARKS, default=None, help="Single benchmark to evaluate.")
    parser.add_argument("--all-benchmarks", action="store_true", help="Evaluate Kodak24, CBSD68, and Urban100.")
    parser.add_argument("--root", type=Path, default=Path("data/external_benchmarks"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/external_benchmarks"))
    parser.add_argument("--sigma", type=int, default=25, choices=sorted(SUPPORTED_SIGMAS), help="Gaussian noise sigma in /255 units.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qualitative-count", type=int, default=5)
    parser.add_argument("--lpips", action="store_true", help="Compute LPIPS if pretrained weights are available.")
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


def create_lpips_model(device: torch.device):
    if device.type == "cpu":
        lpips_device = torch.device("cpu")
    else:
        lpips_device = device
    try:
        import lpips

        model = lpips.LPIPS(net="alex").to(lpips_device)
        model.eval()
        return model
    except Exception as error:  # noqa: BLE001 - LPIPS is optional for this stage.
        print(f"LPIPS disabled: {error}")
        return None


def compute_lpips_optional(x_pred: torch.Tensor, x_clean: torch.Tensor, lpips_model) -> float | None:
    if lpips_model is None:
        return None
    from src.TDnCNN_image import compute_lpips

    return compute_lpips(x_pred.cpu(), x_clean.cpu(), lpips_model)


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
            max_images=1 if args.dry_run else None,
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
