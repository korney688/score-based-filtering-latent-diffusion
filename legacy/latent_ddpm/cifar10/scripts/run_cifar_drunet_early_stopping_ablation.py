from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import scripts.run_cifar_selected_drunet as base
from scripts.evaluate_external import (
    BENCHMARKS,
    audit_datasets,
    compute_metrics,
    load_pairs_for_status,
    predict_model,
)


FILTER_VARIANTS = {
    "topk": {"base_key": "topk_10", "subset": "Top-K", "source_dir": "02_topk_10"},
    "quantile": {"base_key": "quantile_spread_10", "subset": "Quantile", "source_dir": "03_quantile_10"},
}
PATIENCE_VALUES = (5, 10, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DRUNet early-stopping ablation for the existing CIFAR selected-score filters.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "experiments" / "cifar10_drunet_early_stopping_ablation.yaml")
    parser.add_argument("--stage", default="audit")
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-eval-images", type=int, default=None)
    parser.add_argument("--max-external-images", type=int, default=None)
    parser.add_argument("--lpips", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if not isinstance(cfg, dict):
        raise TypeError(f"Config is not a mapping: {args.config}")
    cfg["config_path"] = str(args.config.resolve())
    for key in [
        "current_cifar_root",
        "filtering_dir",
        "tdncnn_root",
        "source_drunet_root",
        "output_dir",
        "external_benchmark_root",
    ]:
        cfg[key] = str(resolve_path(cfg[key]))
    cfg["training"]["total_steps"] = int(cfg["training"]["max_optimizer_steps"])
    if args.smoke or args.stage == "smoke":
        cfg["output_dir"] = str(resolve_path(cfg["smoke"]["output_dir"]))
        cfg["model"]["official"] = bool(cfg["smoke"].get("official", False))
        cfg["training"]["total_steps"] = int(cfg["smoke"]["train_steps"])
        cfg["training"]["max_optimizer_steps"] = int(cfg["smoke"]["train_steps"])
        cfg["training"]["validation_interval_steps"] = int(cfg["smoke"]["validation_interval_steps"])
        cfg["early_stopping"]["warmup_steps"] = int(cfg["smoke"]["warmup_steps"])
        cfg["training"]["batch_size"] = min(int(cfg["training"]["batch_size"]), 4)
        cfg["evaluation"]["lpips_enabled"] = False
        cfg["external"]["lpips_enabled"] = False
    return cfg


def paths(cfg: dict[str, Any]) -> dict[str, Path]:
    root = Path(cfg["output_dir"])
    return {
        "root": root,
        "audit": root / "00_audit",
        "protocol": root / "01_protocol",
        "runs": root / "02_training",
        "internal": root / "03_internal_evaluation",
        "summary": root / "04_summary",
        "figures": root / "04_summary" / "figures",
        "external": root / "05_external_evaluation",
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def run_key(subset: str, patience: int) -> str:
    return f"{subset}_p{int(patience)}"


def run_dir(cfg: dict[str, Any], subset: str, patience: int) -> Path:
    return paths(cfg)["runs"] / run_key(subset, patience)


def patience_values(cfg: dict[str, Any]) -> list[int]:
    values = cfg["early_stopping"].get("patience_values", PATIENCE_VALUES)
    return [int(v) for v in values]


def resolve_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_initial_state(cfg: dict[str, Any], out: dict[str, Path]) -> Path:
    out_path = out["protocol"] / "initial_state_dict.pt"
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if bool(cfg["model"].get("official", True)):
        src = resolve_path(cfg["training"]["initial_state_path"])
        if src is None or not src.exists():
            raise FileNotFoundError(f"Missing source initial state: {src}")
        shutil.copy2(src, out_path)
    else:
        torch.manual_seed(int(cfg["training"]["model_seed"]))
        torch.save(base.build_model(cfg).state_dict(), out_path)
    return out_path


def make_loader(cfg: dict[str, Any], subset: str, *, validation: bool, smoke: bool) -> DataLoader:
    dataset = base.make_train_dataset(cfg, FILTER_VARIANTS[subset]["base_key"], validation=validation, smoke=smoke)
    return DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=not validation,
        num_workers=int(cfg["training"].get("num_workers", 0)),
        generator=torch.Generator().manual_seed(int(cfg["training"]["model_seed"])),
    )


@torch.no_grad()
def validation_loss(model: nn.Module, cfg: dict[str, Any], subset: str, device: torch.device, *, smoke: bool) -> float:
    model.eval()
    loader = make_loader(cfg, subset, validation=True, smoke=smoke)
    criterion = nn.MSELoss(reduction="sum")
    total = 0.0
    count = 0
    for x_input, x_target, sigma, _ in loader:
        x_input = x_input.to(device)
        x_target = x_target.to(device)
        sigma = sigma.to(device)
        total += float(criterion(model(x_input, sigma=sigma), x_target).item())
        count += int(np.prod(x_target.shape))
    model.train()
    return total / max(count, 1)


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), **payload}, path)


def train_one(cfg: dict[str, Any], subset: str, patience: int, *, device: torch.device, resume: bool, dry_run: bool, smoke: bool) -> None:
    out = paths(cfg)
    branch = run_dir(cfg, subset, patience)
    if dry_run:
        print(f"[dry-run] train {run_key(subset, patience)} -> {branch}")
        return
    branch.mkdir(parents=True, exist_ok=True)
    status_path = branch / "status.json"
    best_path = branch / "best.pt"
    last_path = branch / "last.pt"
    history_path = branch / "training_history.csv"
    if resume and status_path.exists() and best_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "complete":
            print(f"resume: skip completed {run_key(subset, patience)}")
            return

    initial_path = ensure_initial_state(cfg, out)
    model = base.build_model(cfg).to(device)
    model.load_state_dict(torch.load(initial_path, map_location=device), strict=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )
    start_step = 0
    best_val = float("inf")
    best_step: int | None = None
    checks_without_improvement = 0
    rows: list[dict[str, Any]] = []
    if resume and last_path.exists():
        payload = torch.load(last_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_step = int(payload.get("step", 0))
        best_val = float(payload.get("best_val_loss", best_val))
        best_step = payload.get("best_step")
        checks_without_improvement = int(payload.get("checks_without_improvement", 0))
        if history_path.exists():
            rows = pd.read_csv(history_path).to_dict("records")

    loader = make_loader(cfg, subset, validation=False, smoke=smoke)
    iterator = base.cycle(loader)
    criterion = nn.MSELoss()
    total_steps = int(cfg["training"]["total_steps"])
    val_interval = int(cfg["training"]["validation_interval_steps"])
    ckpt_interval = int(cfg["training"].get("checkpoint_interval_steps", 100))
    min_delta = float(cfg["early_stopping"].get("min_delta", 0.0))
    warmup_steps = int(cfg["early_stopping"].get("warmup_steps", 0))
    subset_size = int(len(base.load_indices(cfg, FILTER_VARIANTS[subset]["base_key"], smoke=smoke)))
    batch_size = int(cfg["training"]["batch_size"])
    write_json(status_path, {"status": "running", "variant": subset, "patience": patience, "step": start_step, "total_steps": total_steps})
    start_time = time.perf_counter()
    stopped = False
    stop_reason = "max_optimizer_steps"

    for step in tqdm(range(start_step + 1, total_steps + 1), desc=f"DRUNet {run_key(subset, patience)}", mininterval=5.0):
        model.train()
        x_input, x_target, sigma, _ = next(iterator)
        x_input = x_input.to(device)
        x_target = x_target.to(device)
        sigma = sigma.to(device)
        pred = model(x_input, sigma=sigma)
        loss = criterion(pred, x_target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        row = {
            "step": int(step),
            "train_loss": float(loss.item()),
            "val_loss": np.nan,
            "is_validation": False,
            "best_val_loss": best_val if math.isfinite(best_val) else np.nan,
            "best_step": best_step if best_step is not None else np.nan,
            "checks_without_improvement": checks_without_improvement,
            "effective_epochs": float(step * batch_size / max(subset_size, 1)),
        }

        should_validate = step % val_interval == 0 or step == total_steps
        if should_validate:
            val = validation_loss(model, cfg, subset, device, smoke=smoke)
            improved = val < (best_val - min_delta)
            if improved:
                best_val = val
                best_step = int(step)
                checks_without_improvement = 0
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    {
                        "step": int(step),
                        "best_step": int(best_step),
                        "best_val_loss": float(best_val),
                        "patience": int(patience),
                        "subset": subset,
                        "config_hash": base.short_hash(cfg),
                    },
                )
            elif step >= warmup_steps:
                checks_without_improvement += 1
            row.update(
                {
                    "val_loss": float(val),
                    "is_validation": True,
                    "best_val_loss": float(best_val),
                    "best_step": best_step if best_step is not None else np.nan,
                    "checks_without_improvement": int(checks_without_improvement),
                }
            )
            print(
                f"[early-stop] {run_key(subset, patience)} step={step}/{total_steps} "
                f"train={float(loss.item()):.6g} val={val:.6g} best={best_val:.6g} "
                f"best_step={best_step} stale_checks={checks_without_improvement}/{patience}",
                flush=True,
            )
            if step >= warmup_steps and checks_without_improvement >= patience:
                stopped = True
                stop_reason = "early_stopping"
        rows.append(row)

        if step % ckpt_interval == 0 or should_validate or stopped:
            pd.DataFrame(rows).to_csv(history_path, index=False)
            save_checkpoint(
                last_path,
                model,
                optimizer,
                {
                    "step": int(step),
                    "best_step": best_step,
                    "best_val_loss": float(best_val),
                    "checks_without_improvement": int(checks_without_improvement),
                    "patience": int(patience),
                    "subset": subset,
                    "config_hash": base.short_hash(cfg),
                },
            )
            write_json(status_path, {"status": "running", "variant": subset, "patience": patience, "step": int(step), "total_steps": total_steps})
        if stopped:
            break

    if not best_path.exists():
        val = validation_loss(model, cfg, subset, device, smoke=smoke)
        best_val = val
        best_step = total_steps
        save_checkpoint(best_path, model, optimizer, {"step": int(total_steps), "best_step": int(total_steps), "best_val_loss": float(best_val), "patience": int(patience), "subset": subset, "config_hash": base.short_hash(cfg)})
    elapsed = time.perf_counter() - start_time
    stopped_step = int(rows[-1]["step"]) if rows else start_step
    meta = {
        "status": "complete",
        "variant": run_key(subset, patience),
        "subset": subset,
        "subset_label": FILTER_VARIANTS[subset]["subset"],
        "patience": int(patience),
        "train_size": subset_size,
        "batch_size": batch_size,
        "max_optimizer_steps": total_steps,
        "stopped_step": stopped_step,
        "stop_reason": stop_reason,
        "best_step": best_step,
        "best_val_loss": float(best_val),
        "effective_epochs_at_best": None if best_step is None else float(best_step * batch_size / max(subset_size, 1)),
        "effective_epochs_at_stop": float(stopped_step * batch_size / max(subset_size, 1)),
        "wall_clock_seconds": elapsed,
        "initial_state": str(initial_path.resolve()),
        "initial_state_sha256": base.sha256_file(initial_path),
        "best_checkpoint": str(best_path.resolve()),
        "last_checkpoint": str(last_path.resolve()),
        "config_hash": base.short_hash(cfg),
        "early_stopping": cfg["early_stopping"],
        "task_noise": cfg["training"]["task_noise"],
        **base.device_info(device),
    }
    OmegaConf.save(config=OmegaConf.create(meta), f=branch / "resolved_config.yaml")
    write_json(branch / "run_meta.json", meta)
    write_json(status_path, meta)
    print(f"complete {run_key(subset, patience)}: best_step={best_step} best_val={best_val:.6g} stop={stop_reason}@{stopped_step}", flush=True)


def existing_internal_rows(source_root: Path) -> pd.DataFrame:
    per = source_root / "04_internal_evaluation" / "drunet_per_image_metrics.csv"
    df = pd.read_csv(per)
    keep = df[df["Variant"].isin(["Full", "Top-K", "Quantile"])].copy()
    keep["variant"] = keep["Variant"].map({"Full": "full_existing", "Top-K": "topk_equal_steps_existing", "Quantile": "quantile_equal_steps_existing"})
    keep["subset"] = keep["Variant"].map({"Full": "full", "Top-K": "topk", "Quantile": "quantile"})
    keep["patience"] = np.nan
    return keep


@torch.no_grad()
def evaluate_internal(cfg: dict[str, Any], *, device: torch.device, dry_run: bool, smoke: bool, max_images: int | None, lpips_enabled: bool) -> None:
    out = paths(cfg)
    if dry_run:
        print(f"[dry-run] internal evaluation -> {out['internal']}")
        return
    artifacts = base.canonical_artifact_paths(cfg)
    dataset = base.EvalNoiseWithSigmaDataset(artifacts["eval_input"], artifacts["eval_target"], artifacts["eval_manifest"], max_images=max_images)
    loader = DataLoader(dataset, batch_size=int(cfg["evaluation"]["batch_size"]), shuffle=False, num_workers=0)
    lpips_model = base.create_lpips_model(device, lpips_enabled)
    rows = []
    source_root = Path(cfg["source_drunet_root"])
    if not smoke and (source_root / "04_internal_evaluation" / "drunet_per_image_metrics.csv").exists():
        rows.extend(existing_internal_rows(source_root).to_dict("records"))
    elif not smoke:
        existing = [
            ("full_existing", "Full existing", "full", np.nan, source_root / "01_full" / "best.pt"),
            ("topk_equal_steps_existing", "Original Top-K equal steps", "topk", np.nan, source_root / "02_topk_10" / "best.pt"),
            ("quantile_equal_steps_existing", "Original Quantile equal steps", "quantile", np.nan, source_root / "03_quantile_10" / "best.pt"),
        ]
        for variant, label, subset, patience, ckpt in existing:
            if not ckpt.exists():
                continue
            rows.extend(evaluate_model_on_internal(cfg, ckpt, label, variant, subset, patience, loader, lpips_model, device))
    for subset in FILTER_VARIANTS:
        for patience in patience_values(cfg):
            branch = run_dir(cfg, subset, patience)
            ckpt = branch / "best.pt"
            if not ckpt.exists():
                if smoke:
                    continue
                raise FileNotFoundError(f"Missing early-stop checkpoint: {ckpt}")
            label = f"{FILTER_VARIANTS[subset]['subset']} p{patience}"
            rows.extend(evaluate_model_on_internal(cfg, ckpt, label, run_key(subset, patience), subset, patience, loader, lpips_model, device))
    out["internal"].mkdir(parents=True, exist_ok=True)
    out["summary"].mkdir(parents=True, exist_ok=True)
    per = pd.DataFrame(rows)
    per.to_csv(out["internal"] / "internal_per_image_metrics.csv", index=False)
    summary = summarize_internal(cfg, per)
    summary.to_csv(out["summary"] / "internal_early_stopping_comparison.csv", index=False)
    (out["summary"] / "internal_early_stopping_comparison.md").write_text(markdown_table(summary) + "\n", encoding="utf-8")
    write_training_curve_plot(cfg)
    write_psnr_effective_epoch_plot(cfg, summary)


def evaluate_model_on_internal(
    cfg: dict[str, Any],
    checkpoint: Path,
    label: str,
    variant: str,
    subset: str,
    patience: int | float,
    loader: DataLoader,
    lpips_model,
    device: torch.device,
) -> list[dict[str, Any]]:
    model = base.load_model(checkpoint, cfg, device)
    rows = []
    offset = 0
    for x_input, x_target, sigma, original_index in tqdm(loader, desc=f"Internal {label}", mininterval=10.0):
        x_input = x_input.to(device)
        x_target = x_target.to(device)
        sigma = sigma.to(device)
        pred = model(x_input, sigma=sigma)
        mse, psnr, ssim, lpips_values = base.batch_metrics(pred, x_target, lpips_model)
        for i in range(int(x_input.shape[0])):
            rows.append(
                {
                    "variant": variant,
                    "Variant": label,
                    "subset": subset,
                    "patience": patience,
                    "sample_index": int(offset + i),
                    "original_index": int(original_index[i].item()),
                    "MSE": float(mse[i]),
                    "PSNR": float(psnr[i]),
                    "SSIM": float(ssim[i]),
                    "LPIPS": float(lpips_values[i]),
                }
            )
        offset += int(x_input.shape[0])
    return rows


def run_meta_for(cfg: dict[str, Any], subset: str, patience: int) -> dict[str, Any]:
    return json.loads((run_dir(cfg, subset, patience) / "run_meta.json").read_text(encoding="utf-8"))


def summarize_internal(cfg: dict[str, Any], per: pd.DataFrame) -> pd.DataFrame:
    rows = []
    source_root = Path(cfg["source_drunet_root"])
    existing_meta = {
        "full_existing": json.loads((source_root / "01_full" / "run_meta.json").read_text(encoding="utf-8")),
        "topk_equal_steps_existing": json.loads((source_root / "02_topk_10" / "run_meta.json").read_text(encoding="utf-8")),
        "quantile_equal_steps_existing": json.loads((source_root / "03_quantile_10" / "run_meta.json").read_text(encoding="utf-8")),
    }
    for variant, group in per.groupby("variant", sort=False):
        row: dict[str, Any] = {"variant": variant}
        first = group.iloc[0]
        row["subset"] = first["subset"]
        row["patience"] = first["patience"]
        for metric in ["PSNR", "SSIM", "LPIPS", "MSE"]:
            row[metric] = float(pd.to_numeric(group[metric], errors="coerce").mean())
        if variant in existing_meta:
            meta = existing_meta[variant]
            row.update(
                {
                    "train_size": int(meta["subset_size"]),
                    "best_step": np.nan,
                    "stopped_step": int(meta["optimizer_steps"]),
                    "effective_epochs_at_best": np.nan,
                    "effective_epochs_at_stop": float(meta["effective_epochs"]),
                    "best_val_loss": float(meta["validation_loss"]),
                    "train_time": float(meta["wall_clock_seconds"]),
                }
            )
        else:
            meta = run_meta_for(cfg, str(first["subset"]), int(first["patience"]))
            row.update(
                {
                    "train_size": int(meta["train_size"]),
                    "best_step": int(meta["best_step"]),
                    "stopped_step": int(meta["stopped_step"]),
                    "effective_epochs_at_best": float(meta["effective_epochs_at_best"]),
                    "effective_epochs_at_stop": float(meta["effective_epochs_at_stop"]),
                    "best_val_loss": float(meta["best_val_loss"]),
                    "train_time": float(meta["wall_clock_seconds"]),
                }
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    full_rows = summary[summary["variant"] == "full_existing"]
    if full_rows.empty:
        summary["delta_PSNR_vs_Full"] = np.nan
    else:
        full_psnr = float(full_rows.iloc[0]["PSNR"])
        summary["delta_PSNR_vs_Full"] = summary["PSNR"] - full_psnr
    original_by_subset = {
        key: float(rows.iloc[0]["PSNR"])
        for key, rows in {
            "topk": summary[summary["variant"] == "topk_equal_steps_existing"],
            "quantile": summary[summary["variant"] == "quantile_equal_steps_existing"],
        }.items()
        if not rows.empty
    }
    summary["delta_PSNR_vs_original_same_subset"] = summary.apply(
        lambda row: float(row["PSNR"] - original_by_subset[str(row["subset"])]) if str(row["subset"]) in original_by_subset else np.nan,
        axis=1,
    )
    cols = [
        "variant",
        "subset",
        "patience",
        "train_size",
        "best_step",
        "stopped_step",
        "effective_epochs_at_best",
        "effective_epochs_at_stop",
        "best_val_loss",
        "PSNR",
        "SSIM",
        "LPIPS",
        "MSE",
        "delta_PSNR_vs_Full",
        "delta_PSNR_vs_original_same_subset",
        "train_time",
    ]
    return summary[[c for c in cols if c in summary.columns]]


def best_by_validation(cfg: dict[str, Any], subset: str) -> tuple[int, Path, dict[str, Any]]:
    candidates = []
    for patience in patience_values(cfg):
        meta_path = run_dir(cfg, subset, patience) / "run_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            candidates.append((float(meta["best_val_loss"]), int(patience), meta_path.parent / "best.pt", meta))
    if not candidates:
        raise FileNotFoundError(f"No completed early-stop runs for subset={subset}")
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, patience, ckpt, meta = candidates[0]
    return patience, ckpt, meta


@torch.no_grad()
def evaluate_external(cfg: dict[str, Any], *, device: torch.device, dry_run: bool, smoke: bool, max_images: int | None, lpips_enabled: bool) -> None:
    out = paths(cfg)
    if dry_run:
        print(f"[dry-run] external evaluation -> {out['external']}")
        return
    root = Path(cfg["external_benchmark_root"])
    audit = audit_datasets(root, out["external"])
    missing = [name for name in BENCHMARKS if not audit["benchmarks"][name]["exists"] or int(audit["benchmarks"][name]["file_count"]) == 0]
    if missing:
        raise FileNotFoundError(f"Missing external benchmark data: {missing} under {root}")
    topk_patience, topk_ckpt, topk_meta = best_by_validation(cfg, "topk")
    quantile_patience, quantile_ckpt, quantile_meta = best_by_validation(cfg, "quantile")
    source_root = Path(cfg["source_drunet_root"])
    models = {
        "Full existing": base.load_model(source_root / "01_full" / "best.pt", cfg, device),
        f"Early-stop Top-K p{topk_patience}": base.load_model(topk_ckpt, cfg, device),
        f"Early-stop Quantile p{quantile_patience}": base.load_model(quantile_ckpt, cfg, device),
    }
    original_topk_model = base.load_model(source_root / "02_topk_10" / "best.pt", cfg, device)
    lpips_model = base.create_lpips_model(device, lpips_enabled)
    sigma = int(cfg["external"]["sigma"])
    sigma_value = sigma / 255.0
    seed = int(cfg["external"]["seed"])
    per_rows = []
    representatives: dict[str, dict[str, Any]] = {}
    max_images_resolved = max_images if max_images is not None else (int(cfg["smoke"]["max_external_images"]) if smoke else None)
    for benchmark in BENCHMARKS:
        pairs = load_pairs_for_status(root, benchmark, audit["benchmarks"][benchmark], sigma, seed, max_images_resolved)
        for image_idx, pair in enumerate(tqdm(pairs, desc=f"External {benchmark}", mininterval=10.0)):
            clean = pair["clean"]
            noisy = pair["noisy"]
            rep = {"Clean": clean, "Noisy": noisy}
            for label, model in models.items():
                pred = predict_model(model, noisy, "drunet", sigma_value, device)
                metrics = compute_metrics(pred, clean, lpips_model)
                per_rows.append({"Benchmark": benchmark, "Variant": label, "Image": pair["image"], **metrics})
                if image_idx == 0:
                    rep[label] = pred
            if image_idx == 0:
                pred = predict_model(original_topk_model, noisy, "drunet", sigma_value, device)
                rep["Original Top-K"] = pred
                representatives[benchmark] = rep
    out["external"].mkdir(parents=True, exist_ok=True)
    per = pd.DataFrame(per_rows)
    per.to_csv(out["external"] / "external_early_stopping_per_image_metrics.csv", index=False)
    summary_rows = []
    for (benchmark, variant), group in per.groupby(["Benchmark", "Variant"], sort=False):
        row = {"Benchmark": benchmark, "Variant": variant}
        for metric in ["PSNR", "SSIM", "LPIPS", "MSE"]:
            row[metric] = float(pd.to_numeric(group[metric], errors="coerce").mean())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    old_external = source_root / "05_external_evaluation" / "drunet_external_summary.csv"
    old = pd.read_csv(old_external) if old_external.exists() else pd.DataFrame()
    for benchmark, group in summary.groupby("Benchmark", sort=False):
        full = group[group["Variant"] == "Full existing"]
        if full.empty:
            continue
        idx = group.index
        summary.loc[idx, "Delta PSNR vs Full"] = summary.loc[idx, "PSNR"] - float(full.iloc[0]["PSNR"])
        for i in idx:
            variant = str(summary.loc[i, "Variant"])
            if "Top-K" in variant and not old.empty:
                original = old[(old["Benchmark"] == benchmark) & (old["Variant"] == "Top-K")]
            elif "Quantile" in variant and not old.empty:
                original = old[(old["Benchmark"] == benchmark) & (old["Variant"] == "Quantile")]
            else:
                original = pd.DataFrame()
            summary.loc[i, "Delta PSNR vs original filtered model"] = (
                float(summary.loc[i, "PSNR"] - original.iloc[0]["PSNR"]) if not original.empty else np.nan
            )
    summary.to_csv(out["external"] / "external_early_stopping_comparison.csv", index=False)
    (out["external"] / "external_early_stopping_comparison.md").write_text(markdown_table(summary) + "\n", encoding="utf-8")
    write_external_panel(out["external"] / "external_early_stopping_examples.png", representatives, summary)
    write_json(out["external"] / "selected_by_validation.json", {"topk": topk_meta, "quantile": quantile_meta})


@torch.no_grad()
def evaluate_external_topk_only(
    cfg: dict[str, Any],
    *,
    device: torch.device,
    dry_run: bool,
    smoke: bool,
    max_images: int | None,
    lpips_enabled: bool,
) -> None:
    out = paths(cfg)["external"] / "topk_p10_only"
    if dry_run:
        print(f"[dry-run] external Top-K p10 only -> {out}")
        return
    root = Path(cfg["external_benchmark_root"])
    audit = audit_datasets(root, out)
    missing = [name for name in BENCHMARKS if not audit["benchmarks"][name]["exists"] or int(audit["benchmarks"][name]["file_count"]) == 0]
    if missing:
        raise FileNotFoundError(f"Missing external benchmark data: {missing} under {root}")

    source_root = Path(cfg["source_drunet_root"])
    topk_ckpt = paths(cfg)["runs"] / "topk_p10" / "best.pt"
    if not topk_ckpt.exists():
        raise FileNotFoundError(f"Missing Top-K p10 checkpoint: {topk_ckpt}")

    models = {
        "Full existing": base.load_model(source_root / "01_full" / "best.pt", cfg, device),
        "Original Top-K": base.load_model(source_root / "02_topk_10" / "best.pt", cfg, device),
        "Top-K p10 early stop": base.load_model(topk_ckpt, cfg, device),
    }
    lpips_model = base.create_lpips_model(device, lpips_enabled)
    sigma = int(cfg["external"]["sigma"])
    sigma_value = sigma / 255.0
    seed = int(cfg["external"]["seed"])
    max_images_resolved = max_images if max_images is not None else (int(cfg["smoke"]["max_external_images"]) if smoke else None)
    per_rows = []
    representatives: dict[str, dict[str, torch.Tensor]] = {}
    for benchmark in BENCHMARKS:
        pairs = load_pairs_for_status(root, benchmark, audit["benchmarks"][benchmark], sigma, seed, max_images_resolved)
        for image_idx, pair in enumerate(tqdm(pairs, desc=f"External Top-K p10 {benchmark}", mininterval=10.0)):
            clean = pair["clean"]
            noisy = pair["noisy"]
            if image_idx == 0:
                representatives[benchmark] = {"Clean": clean, "Noisy": noisy}
            for label, model in models.items():
                pred = predict_model(model, noisy, "drunet", sigma_value, device)
                metrics = compute_metrics(pred, clean, lpips_model)
                per_rows.append({"Benchmark": benchmark, "Variant": label, "Image": pair["image"], **metrics})
                if image_idx == 0:
                    representatives[benchmark][label] = pred

    out.mkdir(parents=True, exist_ok=True)
    per = pd.DataFrame(per_rows)
    per.to_csv(out / "external_topk_p10_per_image_metrics.csv", index=False)
    summary_rows = []
    for (benchmark, variant), group in per.groupby(["Benchmark", "Variant"], sort=False):
        row = {"Benchmark": benchmark, "Variant": variant, "Images": int(len(group))}
        for metric in ["PSNR", "SSIM", "LPIPS", "MSE"]:
            row[metric] = float(pd.to_numeric(group[metric], errors="coerce").mean())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    for benchmark, group in summary.groupby("Benchmark", sort=False):
        full = group[group["Variant"] == "Full existing"]
        original = group[group["Variant"] == "Original Top-K"]
        if full.empty:
            continue
        idx = group.index
        summary.loc[idx, "Delta PSNR vs Full"] = summary.loc[idx, "PSNR"] - float(full.iloc[0]["PSNR"])
        if not original.empty:
            summary.loc[idx, "Delta PSNR vs Original Top-K"] = summary.loc[idx, "PSNR"] - float(original.iloc[0]["PSNR"])
    summary.to_csv(out / "external_topk_p10_comparison.csv", index=False)
    (out / "external_topk_p10_comparison.md").write_text(markdown_table(summary) + "\n", encoding="utf-8")
    write_external_topk_panel(out / "external_topk_p10_examples.png", representatives, summary)


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


def write_external_panel(output: Path, reps: dict[str, dict[str, torch.Tensor]], metrics: pd.DataFrame) -> None:
    cols = ["Clean", "Noisy", "Full existing", "Original Top-K"]
    extra = [c for c in metrics["Variant"].dropna().unique().tolist() if c.startswith("Early-stop")]
    cols.extend(extra)
    img_w, img_h = 330, 220
    left_w, header_h = 135, 76
    row_h = img_h + 72
    canvas = Image.new("RGB", (left_w + len(cols) * img_w + 40, header_h + len(BENCHMARKS) * row_h + 20), "white")
    draw = ImageDraw.Draw(canvas)
    header_font = font(20, True)
    small_font = font(14)
    for c, col in enumerate(cols):
        draw.text((left_w + c * img_w + 8, 22), col, font=header_font, fill=(0, 0, 0))
    for r, benchmark in enumerate(BENCHMARKS):
        y = header_h + r * row_h
        draw.text((14, y + 82), benchmark, font=header_font, fill=(0, 0, 0))
        item = reps.get(benchmark)
        if not item:
            continue
        for c, col in enumerate(cols):
            x = left_w + c * img_w + 8
            canvas.paste(tensor_image(item[col], img_w - 16, img_h), (x, y))
            if col not in {"Clean", "Noisy", "Original Top-K"}:
                row = metrics[(metrics["Benchmark"] == benchmark) & (metrics["Variant"] == col)]
                if not row.empty:
                    text = f"PSNR {row.iloc[0]['PSNR']:.2f} SSIM {row.iloc[0]['SSIM']:.3f}\nLPIPS {row.iloc[0]['LPIPS']:.3f}"
                    draw.multiline_text((x, y + img_h + 6), text, font=small_font, fill=(0, 0, 0), spacing=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def write_external_topk_panel(output: Path, reps: dict[str, dict[str, torch.Tensor]], metrics: pd.DataFrame) -> None:
    cols = ["Clean", "Noisy", "Full existing", "Original Top-K", "Top-K p10 early stop"]
    img_w, img_h = 330, 220
    left_w, header_h = 135, 76
    row_h = img_h + 72
    canvas = Image.new("RGB", (left_w + len(cols) * img_w + 40, header_h + len(BENCHMARKS) * row_h + 20), "white")
    draw = ImageDraw.Draw(canvas)
    header_font = font(20, True)
    small_font = font(14)
    for c, col in enumerate(cols):
        draw.text((left_w + c * img_w + 8, 22), col, font=header_font, fill=(0, 0, 0))
    for r, benchmark in enumerate(BENCHMARKS):
        y = header_h + r * row_h
        draw.text((14, y + 82), benchmark, font=header_font, fill=(0, 0, 0))
        item = reps.get(benchmark)
        if not item:
            continue
        for c, col in enumerate(cols):
            x = left_w + c * img_w + 8
            canvas.paste(tensor_image(item[col], img_w - 16, img_h), (x, y))
            if col not in {"Clean", "Noisy"}:
                row = metrics[(metrics["Benchmark"] == benchmark) & (metrics["Variant"] == col)]
                if not row.empty:
                    text = f"PSNR {row.iloc[0]['PSNR']:.2f} SSIM {row.iloc[0]['SSIM']:.3f}\nLPIPS {row.iloc[0]['LPIPS']:.3f}"
                    draw.multiline_text((x, y + img_h + 6), text, font=small_font, fill=(0, 0, 0), spacing=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def write_training_curve_plot(cfg: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = paths(cfg)
    out["figures"].mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    source = Path(cfg["source_drunet_root"])
    for label, path in [
        ("Full existing", source / "01_full" / "training_history.csv"),
        ("Top-K original", source / "02_topk_10" / "training_history.csv"),
        ("Quantile original", source / "03_quantile_10" / "training_history.csv"),
    ]:
        if path.exists():
            df = pd.read_csv(path)
            ax.plot(df["step"], df["train_loss"], label=f"{label} train", linewidth=0.8, alpha=0.5)
    for subset in FILTER_VARIANTS:
        for patience in patience_values(cfg):
            hist = run_dir(cfg, subset, patience) / "training_history.csv"
            if not hist.exists():
                continue
            df = pd.read_csv(hist)
            val = df[df["is_validation"].astype(str).str.lower().isin(["true", "1"])]
            if not val.empty:
                ax.plot(val["step"], val["val_loss"], marker="o", label=f"{run_key(subset, patience)} val")
                meta = run_meta_for(cfg, subset, patience)
                ax.axvline(float(meta["best_step"]), linestyle="--", linewidth=0.8, alpha=0.6)
                ax.axvline(float(meta["stopped_step"]), linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out["summary"] / "05_training_curves_best_variants.png")
    plt.close(fig)


def write_psnr_effective_epoch_plot(cfg: dict[str, Any], summary: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = paths(cfg)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    x = pd.to_numeric(summary["effective_epochs_at_best"], errors="coerce").fillna(pd.to_numeric(summary["effective_epochs_at_stop"], errors="coerce"))
    ax.scatter(x, summary["PSNR"])
    for _, row in summary.iterrows():
        xpos = row["effective_epochs_at_best"] if pd.notna(row["effective_epochs_at_best"]) else row["effective_epochs_at_stop"]
        ax.annotate(str(row["variant"]), (float(xpos), float(row["PSNR"])), fontsize=8)
    ax.set_xlabel("effective epochs at best checkpoint")
    ax.set_ylabel("CIFAR test PSNR")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out["summary"] / "06_psnr_vs_effective_epochs.png")
    plt.close(fig)


def audit_stage(cfg: dict[str, Any], *, dry_run: bool) -> None:
    out = paths(cfg)
    if dry_run:
        print(f"[dry-run] audit -> {out['audit']}")
        return
    out["audit"].mkdir(parents=True, exist_ok=True)
    out["protocol"].mkdir(parents=True, exist_ok=True)
    existing = out["root"] / "00_audit" / "training_budget_audit.md"
    lines = [
        "# Early-Stopping Ablation Protocol",
        "",
        "- Source DRUNet result is preserved in place.",
        "- DDPM scores and subset indices are reused without recomputation.",
        "- Full DRUNet is not retrained.",
        "- Top-K and Quantile use validation-loss early stopping with patience values "
        + ", ".join(str(v) for v in patience_values(cfg))
        + ".",
        f"- Validation interval: {int(cfg['training']['validation_interval_steps'])} optimizer steps.",
        f"- Warmup: {int(cfg['early_stopping']['warmup_steps'])} optimizer steps.",
        f"- Minimum delta: {float(cfg['early_stopping']['min_delta'])}.",
        "",
    ]
    if existing.exists():
        lines.extend(["## Existing Budget Audit", f"See `{existing}`.", ""])
    (out["audit"] / "early_stopping_protocol.md").write_text("\n".join(lines), encoding="utf-8")
    OmegaConf.save(config=OmegaConf.create(cfg), f=out["protocol"] / "resolved_config.yaml")
    ensure_initial_state(cfg, out)


def conclusion_stage(cfg: dict[str, Any], *, dry_run: bool) -> None:
    out = paths(cfg)
    if dry_run:
        print(f"[dry-run] summary/conclusion -> {out['summary']}")
        return
    internal_path = out["summary"] / "internal_early_stopping_comparison.csv"
    external_path = out["external"] / "external_early_stopping_comparison.csv"
    if not internal_path.exists():
        raise FileNotFoundError(f"Run internal evaluation first: {internal_path}")
    internal = pd.read_csv(internal_path)
    lines = ["# DRUNet Early-Stopping Ablation Conclusion", ""]
    lines.extend(["## Internal Comparison", markdown_table(internal), ""])
    if external_path.exists():
        external = pd.read_csv(external_path)
        lines.extend(["## External Comparison", markdown_table(external), ""])
    for subset in ["topk", "quantile"]:
        candidates = internal[(internal["subset"] == subset) & internal["patience"].notna()].sort_values(["best_val_loss", "patience"])
        if not candidates.empty:
            best = candidates.iloc[0]
            lines.append(f"- Best {subset} by validation loss: `{best['variant']}`; PSNR={best['PSNR']:.4f}; delta vs Full={best['delta_PSNR_vs_Full']:.4f}.")
    lines.append("")
    lines.append("Interpretation must be based on validation-selected checkpoints; test PSNR was not used to choose patience.")
    (out["summary"] / "early_stopping_conclusion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def expanded_stages(stage: str, cfg: dict[str, Any]) -> list[str]:
    if stage == "all":
        return ["audit", *expanded_stages("train_filters", cfg), "evaluate_internal", "evaluate_external", "summary"]
    if stage == "smoke":
        return ["audit", "train_topk_p5", "train_quantile_p5", "evaluate_internal", "summary"]
    if stage == "train_filters":
        return [f"train_{subset}_p{p}" for subset in FILTER_VARIANTS for p in patience_values(cfg)]
    if stage == "train_topk_all":
        return [f"train_topk_p{p}" for p in patience_values(cfg)]
    if stage == "train_quantile_all":
        return [f"train_quantile_p{p}" for p in patience_values(cfg)]
    return [stage]


def run_stage(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    device = resolve_device(args.device)
    smoke = bool(args.smoke or args.stage == "smoke")
    for stage in expanded_stages(args.stage, cfg):
        print(f"\n===== {stage} =====", flush=True)
        if stage == "audit":
            audit_stage(cfg, dry_run=args.dry_run)
        elif stage.startswith("train_"):
            parts = stage.split("_")
            if len(parts) != 3 or not parts[2].startswith("p"):
                raise ValueError(f"Unsupported train stage: {stage}")
            subset = parts[1]
            patience = int(parts[2][1:])
            train_one(cfg, subset, patience, device=device, resume=args.resume, dry_run=args.dry_run, smoke=smoke)
        elif stage == "evaluate_internal":
            max_images = args.max_eval_images if args.max_eval_images is not None else (int(cfg["smoke"]["max_eval_images"]) if smoke else None)
            evaluate_internal(cfg, device=device, dry_run=args.dry_run, smoke=smoke, max_images=max_images, lpips_enabled=bool(args.lpips or cfg["evaluation"].get("lpips_enabled", True)))
        elif stage == "evaluate_external":
            max_images = args.max_external_images if args.max_external_images is not None else (int(cfg["smoke"]["max_external_images"]) if smoke else None)
            evaluate_external(cfg, device=device, dry_run=args.dry_run, smoke=smoke, max_images=max_images, lpips_enabled=bool(args.lpips or cfg["external"].get("lpips_enabled", True)))
        elif stage == "evaluate_external_topk_only":
            max_images = args.max_external_images if args.max_external_images is not None else (int(cfg["smoke"]["max_external_images"]) if smoke else None)
            evaluate_external_topk_only(
                cfg,
                device=device,
                dry_run=args.dry_run,
                smoke=smoke,
                max_images=max_images,
                lpips_enabled=bool(args.lpips or cfg["external"].get("lpips_enabled", True)),
            )
        elif stage == "summary":
            conclusion_stage(cfg, dry_run=args.dry_run)
        else:
            raise ValueError(f"Unsupported stage: {stage}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    if args.patience is not None:
        cfg["early_stopping"]["patience_values"] = [int(args.patience)]
    run_stage(args, cfg)


if __name__ == "__main__":
    main()
