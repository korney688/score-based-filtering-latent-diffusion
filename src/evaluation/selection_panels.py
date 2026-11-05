from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw


def tensor_to_pil(x: torch.Tensor, size: int = 72) -> Image.Image:
    x = x.detach().cpu().clamp(0.0, 1.0)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.shape[0] == 1:
        arr = x[0].numpy()
        image = Image.fromarray((arr * 255).round().astype(np.uint8), mode="L").convert("RGB")
    else:
        arr = x[:3].permute(1, 2, 0).numpy()
        image = Image.fromarray((arr * 255).round().astype(np.uint8)).convert("RGB")
    return image.resize((size, size), resample=Image.Resampling.NEAREST)


def _score_col(scores: pd.DataFrame) -> str:
    return "score_value" if "score_value" in scores.columns else "score"


def _label(row: pd.Series) -> str:
    score = float(row["score_value"] if "score_value" in row.index else row["score"])
    rank = int(row.get("score_rank", -1))
    target = float(row.get("target_psnr_db", np.nan))
    realized = float(row.get("realized_psnr_db_postclip", row.get("realized_psnr", np.nan)))
    sigma = float(row.get("sigma", np.nan))
    cls = row.get("class_id", row.get("label", "NA"))
    return f"class {cls}\nscore {score:.4g}\nrank {rank}\ntarget {target:.2f}\nrealized {realized:.2f}\nsigma {sigma:.4f}"


def _draw_pairs(
    rows: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    clean_for_index: Callable[[int], torch.Tensor],
    corrupt_for_index: Callable[[int], torch.Tensor],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = rows.head(8).copy()
    cell_w, cell_h = 180, 190
    header_h = 28
    canvas = Image.new("RGB", (cell_w * max(1, len(rows)), header_h + cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(0, 0, 0))
    for i, (_, row) in enumerate(rows.iterrows()):
        x0 = i * cell_w
        original = int(row["original_index"])
        canvas.paste(tensor_to_pil(clean_for_index(original)), (x0 + 8, header_h + 4))
        canvas.paste(tensor_to_pil(corrupt_for_index(original)), (x0 + 88, header_h + 4))
        draw.text((x0 + 8, header_h + 82), _label(row), fill=(0, 0, 0))
    canvas.save(output_path)


def save_topk_selection_examples(
    scores: pd.DataFrame,
    topk_indices: np.ndarray,
    output_path: str | Path,
    *,
    clean_for_index: Callable[[int], torch.Tensor],
    corrupt_for_index: Callable[[int], torch.Tensor],
    fraction_label: str,
) -> None:
    output_path = Path(output_path)
    score_col = _score_col(scores)
    selected = set(np.asarray(topk_indices, dtype=np.int64).tolist())
    ranked = scores.sort_values([score_col, "original_index"], ascending=[True, True]).reset_index(drop=True).copy()
    ranked["score_rank"] = np.arange(1, len(ranked) + 1)
    sel = ranked[ranked["original_index"].isin(selected)].head(8)
    rej = ranked[~ranked["original_index"].isin(selected)].tail(8).iloc[::-1]
    tmp_a = output_path.with_name(output_path.stem + "_selected_tmp.png")
    tmp_b = output_path.with_name(output_path.stem + "_rejected_tmp.png")
    _draw_pairs(sel, tmp_a, title=f"Lowest score / selected {fraction_label}", clean_for_index=clean_for_index, corrupt_for_index=corrupt_for_index)
    _draw_pairs(rej, tmp_b, title=f"Highest score / rejected {fraction_label}", clean_for_index=clean_for_index, corrupt_for_index=corrupt_for_index)
    a = Image.open(tmp_a)
    b = Image.open(tmp_b)
    canvas = Image.new("RGB", (max(a.width, b.width), a.height + b.height), "white")
    canvas.paste(a, (0, 0))
    canvas.paste(b, (0, a.height))
    canvas.save(output_path)
    tmp_a.unlink(missing_ok=True)
    tmp_b.unlink(missing_ok=True)


def save_topk_cleanest_noisiest(
    scores: pd.DataFrame,
    topk_indices: np.ndarray,
    output_path: str | Path,
    *,
    clean_for_index: Callable[[int], torch.Tensor],
    corrupt_for_index: Callable[[int], torch.Tensor],
    fraction_label: str,
) -> None:
    output_path = Path(output_path)
    selected = set(np.asarray(topk_indices, dtype=np.int64).tolist())
    ranked = scores.sort_values([_score_col(scores), "original_index"], ascending=[True, True]).reset_index(drop=True).copy()
    ranked["score_rank"] = np.arange(1, len(ranked) + 1)
    sub = ranked[ranked["original_index"].isin(selected)]
    psnr_col = "realized_psnr_db_postclip" if "realized_psnr_db_postclip" in sub.columns else "realized_psnr"
    cleanest = sub.sort_values(psnr_col, ascending=False).head(8)
    noisiest = sub.sort_values(psnr_col, ascending=True).head(8)
    tmp_a = output_path.with_name(output_path.stem + "_cleanest_tmp.png")
    tmp_b = output_path.with_name(output_path.stem + "_noisiest_tmp.png")
    _draw_pairs(cleanest, tmp_a, title=f"Cleanest inside selected Top-K {fraction_label}", clean_for_index=clean_for_index, corrupt_for_index=corrupt_for_index)
    _draw_pairs(noisiest, tmp_b, title=f"Noisiest inside selected Top-K {fraction_label}", clean_for_index=clean_for_index, corrupt_for_index=corrupt_for_index)
    a = Image.open(tmp_a)
    b = Image.open(tmp_b)
    canvas = Image.new("RGB", (max(a.width, b.width), a.height + b.height), "white")
    canvas.paste(a, (0, 0))
    canvas.paste(b, (0, a.height))
    canvas.save(output_path)
    tmp_a.unlink(missing_ok=True)
    tmp_b.unlink(missing_ok=True)
