from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.evaluation.selection_panels import tensor_to_pil


def save_denoiser_comparison_panel(
    samples: Sequence[dict],
    output_path: str | Path,
    *,
    columns: Sequence[str],
    title: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tile = 84
    label_h = 54
    header_h = 34
    row_h = tile + label_h
    left_w = 92
    canvas = Image.new("RGB", (left_w + len(columns) * tile, header_h + len(samples) * row_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill=(0, 0, 0))
    for c, name in enumerate(columns):
        draw.text((left_w + c * tile + 4, 8), name, fill=(0, 0, 0))
    for r, sample in enumerate(samples):
        y0 = header_h + r * row_h
        draw.text((8, y0 + 8), str(sample.get("name", r)), fill=(0, 0, 0))
        tensors: dict[str, torch.Tensor] = sample["tensors"]
        metrics: dict[str, dict[str, float]] = sample.get("metrics", {})
        for c, col in enumerate(columns):
            x0 = left_w + c * tile
            canvas.paste(tensor_to_pil(tensors[col], size=tile - 8), (x0 + 4, y0 + 4))
            metric = metrics.get(col)
            if metric:
                text = (
                    f"PSNR {metric.get('psnr', np.nan):.2f}\n"
                    f"SSIM {metric.get('ssim', np.nan):.3f}\n"
                    f"LPIPS {metric.get('lpips', np.nan):.3f}"
                )
                draw.text((x0 + 4, y0 + tile), text, fill=(0, 0, 0))
    canvas.save(output_path)
