from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as skimage_ssim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.TDnCNN_image import DnCNN_2D, align_lpips_scaling_layer, prepare_for_lpips
from src.image_lab_replication_datasets import (
    CorpusTaskNoiseDataset,
    EvalNoiseDataset,
    MaterializedCorpusDataset,
)


VARIANT_LABELS = {
    "full_45000": "full_45000_e18_t100",
    "topk_10": "topk_10_e18_t100",
    "quantile_spread_10": "quantile_spread_10_e18_t100",
}
HISTORICAL_ROWS = [
    {"variant": "old_topk_t1", "filter_timestep": 1, "PSNR": 20.001954, "SSIM": 0.577071, "LPIPS": 0.318484},
    {"variant": "old_quantile_t1", "filter_timestep": 1, "PSNR": 19.413274, "SSIM": 0.584568, "LPIPS": 0.298144},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train selected CIFAR-10 TDnCNN downstream branches.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--filtering-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--old-full-root", type=Path, default=PROJECT_ROOT / "experiments" / "image_lab_replication_v1" / "cifar10_psnr10-40_s42" / "main")
    parser.add_argument("--variants", nargs="+", default=["topk_10", "quantile_spread_10"], choices=["full_45000", "topk_10", "quantile_spread_10"])
    parser.add_argument("--comparison-mode", default="equal_steps", choices=["equal_steps"])
    parser.add_argument("--reuse-full-checkpoint", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=["audit", "train_full", "train_topk", "train_quantile", "evaluate", "figures", "summary", "all"], default="all")
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "protocol": root / "00_protocol",
        "subsets": root / "01_subsets",
        "training": root / "02_training",
        "evaluation": root / "03_evaluation",
        "summary": root / "04_summary",
        "figures": root / "05_figures",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_info(device: torch.device) -> dict[str, Any]:
    payload = {"device": str(device), "gpu_name": None, "peak_cuda_memory": None}
    if device.type == "cuda":
        payload["gpu_name"] = torch.cuda.get_device_name(0)
        payload["peak_cuda_memory"] = int(torch.cuda.max_memory_allocated(device))
    return payload


def load_old_config(old_root: Path) -> dict[str, Any]:
    config = OmegaConf.to_container(OmegaConf.load(old_root / "00_protocol" / "resolved_config.yaml"), resolve=True)
    if not isinstance(config, dict):
        raise TypeError("Old Full config is not a mapping")
    return config


def old_total_steps(old_root: Path, config: dict[str, Any]) -> int:
    summary_path = old_root / "04_downstream" / "downstream_training_summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        row = df[df["training_subset"] == "full"]
        if not row.empty and "total_steps" in row:
            return int(row.iloc[0]["total_steps"])
    n_full = len(pd.read_csv(old_root / "01_corpus" / "train_manifest.csv"))
    return int(math.ceil(n_full / int(config["downstream"]["batch_size"])) * int(config["downstream"]["full_reference_epochs"]))


def load_subsets(filtering_dir: Path) -> dict[str, np.ndarray]:
    subset_json = filtering_dir / "02_subsets" / "subset_indices.json"
    payload = read_json(subset_json)
    return {name: np.asarray(payload[name], dtype=np.int64) for name in ["full", "topk_10", "quantile_spread_10"]}


def save_subset_copies(args: argparse.Namespace, dirs: dict[str, Path], subsets: dict[str, np.ndarray]) -> dict[str, Any]:
    source_paths = {
        "full_45000": args.filtering_dir / "02_subsets" / "full_indices.npy",
        "topk_10": args.filtering_dir / "02_subsets" / "topk_10_indices.npy",
        "quantile_spread_10": args.filtering_dir / "02_subsets" / "quantile_spread_10_indices.npy",
    }
    payload: dict[str, Any] = {}
    for name in ["full_45000", "topk_10", "quantile_spread_10"]:
        out = dirs["subsets"] / f"{name}_indices.json"
        subset_key = "full" if name == "full_45000" else name
        write_json(out, [int(x) for x in np.sort(subsets[subset_key])])
        payload[name] = {
            "source_npy": str(source_paths[name].resolve()),
            "source_sha256": sha256_file(source_paths[name]) if source_paths[name].exists() else None,
            "copied_json": str(out.resolve()),
            "size": int(len(subsets[subset_key])),
            "unique": bool(len(np.unique(subsets[subset_key])) == len(subsets[subset_key])),
        }
    return payload


def config_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def audit(args: argparse.Namespace, dirs: dict[str, Path]) -> dict[str, Any]:
    old_root = args.old_full_root
    old_config = load_old_config(old_root)
    subsets = load_subsets(args.filtering_dir)
    subset_meta = save_subset_copies(args, dirs, subsets)

    experiment_root = args.experiment_root
    filtering_meta = read_json(args.filtering_dir / "00_protocol" / "run_meta.json")
    train_manifest = pd.read_csv(experiment_root / "01_corpus" / "train_manifest.csv")
    old_train_manifest = pd.read_csv(old_root / "01_corpus" / "train_manifest.csv")
    ddpm_filter_train = np.load(experiment_root / "00_protocol" / "ddpm_filter_train_original_indices.npy").astype(np.int64)
    tuning = set(np.load(experiment_root / "00_protocol" / "score_tuning_original_indices.npy").astype(np.int64).tolist())
    confirmation = set(np.load(experiment_root / "00_protocol" / "score_confirmation_original_indices.npy").astype(np.int64).tolist())
    full_new = set(subsets["full"].astype(int).tolist())

    full_checkpoint = old_root / "04_downstream" / "full" / "best.pt"
    initial_state = old_root / "04_downstream" / "initial_state_dict.pt"
    eval_input = old_root / "01_corpus" / "test_eval_input.pt"
    eval_target = old_root / "01_corpus" / "test_eval_target.pt"
    eval_manifest = old_root / "01_corpus" / "test_noise_manifest.csv"
    old_training = pd.read_csv(old_root / "04_downstream" / "downstream_training_summary.csv")
    old_full_training = old_training[old_training["training_subset"] == "full"].iloc[0].to_dict()
    total_steps = old_total_steps(old_root, old_config)
    batch_size = int(old_config["downstream"]["batch_size"])

    new_seed_col = "corpus_seed" if "corpus_seed" in train_manifest.columns else "corruption_seed"
    old_seed_col = "corpus_seed" if "corpus_seed" in old_train_manifest.columns else "corruption_seed"
    new_versions = set(train_manifest["corruption_version"].astype(str).unique()) if "corruption_version" in train_manifest.columns else set()
    old_versions = set(old_train_manifest["corruption_version"].astype(str).unique()) if "corruption_version" in old_train_manifest.columns else set()
    checks = {
        "new_full_matches_ddpm_filter_train": bool(np.array_equal(np.sort(subsets["full"]), np.sort(ddpm_filter_train))),
        "topk_size": int(len(subsets["topk_10"])),
        "quantile_size": int(len(subsets["quantile_spread_10"])),
        "subset_sizes_match": bool(len(subsets["topk_10"]) == len(subsets["quantile_spread_10"])),
        "topk_unique": bool(len(np.unique(subsets["topk_10"])) == len(subsets["topk_10"])),
        "quantile_unique": bool(len(np.unique(subsets["quantile_spread_10"])) == len(subsets["quantile_spread_10"])),
        "topk_in_working_split": bool(set(subsets["topk_10"].astype(int).tolist()).issubset(full_new)),
        "quantile_in_working_split": bool(set(subsets["quantile_spread_10"].astype(int).tolist()).issubset(full_new)),
        "validation_indices_absent_topk": bool(not (set(subsets["topk_10"].astype(int).tolist()) & (tuning | confirmation))),
        "validation_indices_absent_quantile": bool(not (set(subsets["quantile_spread_10"].astype(int).tolist()) & (tuning | confirmation))),
        "corpus_corruption_type_matches_old": bool(set(train_manifest["corruption_type"].astype(str).unique()) == set(old_train_manifest["corruption_type"].astype(str).unique())),
        "corpus_corruption_version_matches_old": bool(new_versions == old_versions) if new_versions and old_versions else None,
        "corpus_seed_matches_old": bool(set(train_manifest[new_seed_col].astype(int).unique()) == set(old_train_manifest[old_seed_col].astype(int).unique())),
        "corpus_psnr_range_matches_old": bool(
            np.isclose(train_manifest["target_psnr_db"].min(), old_train_manifest["target_psnr_db"].min())
            and np.isclose(train_manifest["target_psnr_db"].max(), old_train_manifest["target_psnr_db"].max())
        ),
        "downstream_target": "x_target = x_corpus",
        "downstream_input": "x_input = clip(x_corpus + task_noise, 0, 1)",
        "architecture": "DnCNN_2D(in_channels=3, num_layers=8, features=64)",
        "optimizer": "Adam",
        "scheduler": "none",
        "batch_size": batch_size,
        "total_optimizer_steps": total_steps,
        "old_full_train_size": int(old_full_training["sample_count"]),
        "new_working_train_size": int(len(subsets["full"])),
        "train_split_strict_match_with_old_full": bool(int(old_full_training["sample_count"]) == int(len(subsets["full"]))),
        "initial_state_available": bool(initial_state.exists()),
        "full_checkpoint_available": bool(full_checkpoint.exists()),
        "eval_noise_cache_available": bool(eval_input.exists() and eval_target.exists() and eval_manifest.exists()),
        "test_split": "CIFAR-10 torchvision test, cached in old Full run",
    }
    strict_full = bool(
        checks["new_full_matches_ddpm_filter_train"]
        and checks["corpus_corruption_type_matches_old"]
        and checks["corpus_seed_matches_old"]
        and checks["corpus_psnr_range_matches_old"]
        and checks["initial_state_available"]
        and checks["full_checkpoint_available"]
        and checks["eval_noise_cache_available"]
        and checks["train_split_strict_match_with_old_full"]
    )
    comparison_status = "strict" if strict_full else "preliminary_non_strict"
    if not strict_full and int(old_full_training["sample_count"]) == 50000 and len(subsets["full"]) == 45000:
        comparison_status = "preliminary_non_strict_train_split_50000_vs_45000"

    train_config = {
        "dataset": "cifar10",
        "variants": list(args.variants),
        "comparison_mode": "equal_optimizer_steps",
        "filter_epoch": 18,
        "filter_report_timestep": 100,
        "filter_code_timestep_index": 99,
        "filter_mc_k": 16,
        "subset_fraction": 0.10,
        "status": "tuning-selected, preliminary, confirmation pending",
        "corpus_cache": str((experiment_root / "01_corpus" / "train_corpus.pt").resolve()),
        "corpus_manifest": str((experiment_root / "01_corpus" / "train_manifest.csv").resolve()),
        "validation_indices": str((experiment_root / "00_protocol" / "score_tuning_original_indices.npy").resolve()),
        "initial_state": str(initial_state.resolve()) if initial_state.exists() else str((dirs["training"] / "initial_state_dict.pt").resolve()),
        "full_checkpoint": str(full_checkpoint.resolve()) if full_checkpoint.exists() else None,
        "eval_input": str(eval_input.resolve()),
        "eval_target": str(eval_target.resolve()),
        "eval_manifest": str(eval_manifest.resolve()),
        "model_seed": int(old_config["seeds"]["model_seed"]),
        "task_seed": int(old_config["seeds"]["task_seed"]),
        "task_noise": old_config["task_noise"],
        "eval_noise": old_config["eval_noise"],
        "batch_size": batch_size,
        "total_steps": total_steps,
        "lr": float(old_config["downstream"]["lr"]),
        "lpips_enabled": bool(old_config["downstream"]["lpips_enabled"]),
        "old_full_root": str(old_root.resolve()),
        "old_full_train_size": int(old_full_training["sample_count"]),
        "new_working_train_size": int(len(subsets["full"])),
        "comparison_status_vs_full": comparison_status,
        "filtering_run_meta": filtering_meta,
    }
    train_config["config_hash"] = config_hash(train_config)

    payload = {
        "paths": {
            "experiment_root": str(experiment_root.resolve()),
            "filtering_dir": str(args.filtering_dir.resolve()),
            "old_full_root": str(old_root.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "full_checkpoint": str(full_checkpoint.resolve()),
            "initial_state": str(initial_state.resolve()),
            "eval_noise_cache": str(eval_input.resolve()),
            "eval_target_cache": str(eval_target.resolve()),
            "eval_manifest": str(eval_manifest.resolve()),
        },
        "subset_files": subset_meta,
        "checks": checks,
        "comparison_status_vs_full": comparison_status,
        "resolved_downstream_config": train_config,
    }
    write_json(dirs["protocol"] / "compatibility_audit.json", payload)
    lines = [
        "# Compatibility Audit",
        "",
        f"- Top-K indices: {subset_meta['topk_10']['source_npy']}",
        f"- Quantile indices: {subset_meta['quantile_spread_10']['source_npy']}",
        f"- Corpus cache: {train_config['corpus_cache']}",
        f"- Corpus manifest: {train_config['corpus_manifest']}",
        f"- Old Full checkpoint: {train_config['full_checkpoint']}",
        f"- Initial state: {train_config['initial_state']}",
        f"- Eval-noise cache: {train_config['eval_input']}",
        f"- Batch size: {batch_size}",
        f"- Total optimizer steps: {total_steps}",
        f"- TDnCNN architecture: {checks['architecture']}",
        f"- Optimizer/scheduler: Adam lr={train_config['lr']}; scheduler=none",
        f"- Task-noise protocol: {old_config['task_noise']}",
        f"- Eval-noise protocol: {old_config['eval_noise']}",
        "",
        "## Checks",
    ]
    for key, value in checks.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            f"Full comparison status: {comparison_status}.",
            "New Top-K vs new Quantile comparison is strict because both use the same initial_state, optimizer steps, task-noise protocol, and eval-noise cache.",
        ]
    )
    (dirs["protocol"] / "compatibility_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    OmegaConf.save(config=OmegaConf.create(train_config), f=dirs["protocol"] / "resolved_downstream_config.yaml")
    return payload


def ensure_initial_state(config: dict[str, Any], dirs: dict[str, Path], device: torch.device) -> Path:
    source = Path(str(config["initial_state"]))
    if source.exists():
        return source
    path = dirs["training"] / "initial_state_dict.pt"
    if path.exists():
        return path
    torch.manual_seed(int(config["model_seed"]))
    model = DnCNN_2D(in_channels=3)
    torch.save(model.state_dict(), path)
    return path


def _cycle_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def load_model_state(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def branch_config(config: dict[str, Any], variant: str, subset_size: int) -> dict[str, Any]:
    payload = {
        "variant": variant,
        "subset_size": int(subset_size),
        "batch_size": int(config["batch_size"]),
        "total_steps": int(config["total_steps"]),
        "lr": float(config["lr"]),
        "model_seed": int(config["model_seed"]),
        "task_seed": int(config["task_seed"]),
        "task_noise": config["task_noise"],
        "architecture": "DnCNN_2D(in_channels=3, num_layers=8, features=64)",
        "initial_state": str(config["initial_state"]),
        "comparison_mode": "equal_optimizer_steps",
        "filter_epoch": 18,
        "filter_timestep": 100,
    }
    payload["config_hash"] = config_hash(payload)
    return payload


def train_variant(args: argparse.Namespace, dirs: dict[str, Path], variant: str) -> dict[str, Any]:
    config = OmegaConf.to_container(OmegaConf.load(dirs["protocol"] / "resolved_downstream_config.yaml"), resolve=True)
    subsets = load_subsets(args.filtering_dir)
    subset_key = "full" if variant == "full_45000" else variant
    indices = np.sort(subsets[subset_key])
    bcfg = branch_config(config, variant, len(indices))
    branch_dir = dirs["training"] / variant
    branch_dir.mkdir(parents=True, exist_ok=True)
    hash_path = branch_dir / "config_hash"
    if hash_path.exists() and hash_path.read_text(encoding="utf-8").strip() != bcfg["config_hash"]:
        raise RuntimeError(f"Config hash mismatch in {branch_dir}")
    if not hash_path.exists():
        hash_path.write_text(bcfg["config_hash"], encoding="utf-8")
    OmegaConf.save(config=OmegaConf.create(bcfg), f=branch_dir / "resolved_config.yaml")

    history_path = branch_dir / "training_history.csv"
    best_path = branch_dir / "best.pt"
    last_path = branch_dir / "last.pt"
    status_path = branch_dir / "status.json"
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    initial_state = ensure_initial_state(config, dirs, device)
    model = DnCNN_2D(in_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    start_step = 0
    best_loss = float("inf")
    rows: list[dict[str, Any]] = []
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_step = int(checkpoint.get("step", 0))
            best_loss = float(checkpoint.get("best_train_loss", best_loss))
        else:
            model.load_state_dict(checkpoint)
        if history_path.exists():
            rows = pd.read_csv(history_path).to_dict("records")
    else:
        model.load_state_dict(torch.load(initial_state, map_location=device))

    if start_step >= int(config["total_steps"]) and best_path.exists():
        meta = read_json(branch_dir / "run_meta.json")
        print_training_done(meta, best_path)
        return meta

    torch.manual_seed(int(config["model_seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config["model_seed"]))
    corpus = MaterializedCorpusDataset(
        Path(str(config["corpus_cache"])),
        Path(str(config["corpus_manifest"])),
        original_indices=indices,
        normalized=False,
        return_original_index=True,
    )
    train_dataset = CorpusTaskNoiseDataset(
        corpus,
        dataset_name=str(config["dataset"]),
        task_seed=int(config["task_seed"]),
        task_noise_version=str(config["task_noise"]["version"]),
        sigma_min=float(config["task_noise"]["sigma_min"]),
        sigma_max=float(config["task_noise"]["sigma_max"]),
    )
    generator = torch.Generator().manual_seed(int(config["model_seed"]))
    loader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=0, generator=generator)
    iterator = _cycle_loader(loader)
    criterion = nn.MSELoss()
    start_time = time.perf_counter()
    write_json(status_path, {"status": "running", "variant": variant, "step": start_step, "total_steps": int(config["total_steps"])})
    progress = tqdm(range(start_step + 1, int(config["total_steps"]) + 1), desc=f"Training {variant}", mininterval=10.0)
    for step in progress:
        x_input, x_target = next(iterator)
        x_input = x_input.to(device)
        x_target = x_target.to(device)
        pred = model(x_input)
        loss = criterion(pred, x_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = float(loss.item())
        if loss_value < best_loss:
            best_loss = loss_value
            torch.save(model.state_dict(), best_path)
        rows.append({"step": int(step), "train_loss": loss_value})
        if step % 100 == 0 or step == int(config["total_steps"]):
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": int(step),
                    "best_train_loss": float(best_loss),
                    "config_hash": bcfg["config_hash"],
                },
                last_path,
            )
            pd.DataFrame(rows).to_csv(history_path, index=False)
            write_json(status_path, {"status": "running", "variant": variant, "step": int(step), "total_steps": int(config["total_steps"])})
        if step % 50 == 0 or step == int(config["total_steps"]):
            progress.set_postfix(loss=f"{loss_value:.5f}", best=f"{best_loss:.5f}")

    validation_loss = validate_variant(config, indices, best_path, device)
    elapsed = time.perf_counter() - start_time
    meta = {
        "variant": variant,
        "subset_size": int(len(indices)),
        "batch_size": int(config["batch_size"]),
        "total_steps": int(config["total_steps"]),
        "total_seen_samples": int(config["total_steps"]) * int(config["batch_size"]),
        "effective_epochs": float(int(config["total_steps"]) * int(config["batch_size"]) / max(len(indices), 1)),
        "wall_clock_seconds": float(elapsed),
        "steps_per_second": float(int(config["total_steps"]) / max(elapsed, 1e-9)),
        "best_train_loss": float(best_loss),
        "validation_loss": float(validation_loss),
        "initial_state": str(initial_state.resolve()),
        "best_checkpoint": str(best_path.resolve()),
        "last_checkpoint": str(last_path.resolve()),
        "config_hash": bcfg["config_hash"],
        **device_info(device),
    }
    write_json(branch_dir / "run_meta.json", meta)
    write_json(status_path, {"status": "complete", **meta})
    print_training_done(meta, best_path)
    return meta


def validate_variant(config: dict[str, Any], train_indices: np.ndarray, best_path: Path, device: torch.device) -> float:
    tuning_path = Path(str(config["validation_indices"]))
    validation_indices = np.load(tuning_path).astype(np.int64)
    corpus = MaterializedCorpusDataset(
        Path(str(config["corpus_cache"])),
        Path(str(config["corpus_manifest"])),
        original_indices=validation_indices,
        normalized=False,
        return_original_index=True,
    )
    dataset = CorpusTaskNoiseDataset(
        corpus,
        dataset_name=str(config["dataset"]),
        task_seed=int(config["task_seed"]),
        task_noise_version=str(config["task_noise"]["version"]) + "_validation",
        sigma_min=float(config["task_noise"]["sigma_min"]),
        sigma_max=float(config["task_noise"]["sigma_max"]),
    )
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=0)
    model = DnCNN_2D(in_channels=3).to(device)
    model.load_state_dict(load_model_state(best_path, device))
    model.eval()
    criterion = nn.MSELoss(reduction="sum")
    total = 0.0
    items = 0
    with torch.inference_mode():
        for x_input, x_target in loader:
            x_input = x_input.to(device)
            x_target = x_target.to(device)
            total += float(criterion(model(x_input), x_target).item())
            items += int(np.prod(x_target.shape))
    return total / max(items, 1)


def print_training_done(meta: dict[str, Any], best_path: Path) -> None:
    print("MODEL_DONE")
    print(f"variant={meta['variant']}")
    print(f"current_step={meta['total_steps']}/{meta['total_steps']}")
    print(f"train_time={float(meta['wall_clock_seconds']):.3f}")
    print(f"best_validation_loss={float(meta['validation_loss']):.8f}")
    print(f"checkpoint={best_path}")


def create_lpips_model(device: torch.device, enabled: bool):
    if not enabled:
        return None
    try:
        import lpips

        model = lpips.LPIPS(net="alex").to(device)
        model.eval()
        align_lpips_scaling_layer(model, device)
        return model
    except Exception as error:
        print(f"LPIPS disabled: {error}")
        return None


def compute_ssim_item(pred: np.ndarray, target: np.ndarray) -> float:
    return float(skimage_ssim(target, pred, data_range=1.0, channel_axis=0))


def batch_metrics(pred: torch.Tensor, target: torch.Tensor, lpips_model) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    mse = ((pred - target) ** 2).flatten(start_dim=1).mean(dim=1)
    psnr = 10.0 * torch.log10(1.0 / (mse + 1e-12))
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    ssim_vals = np.asarray([compute_ssim_item(pred_np[i], target_np[i]) for i in range(pred_np.shape[0])], dtype=np.float64)
    if lpips_model is None:
        lpips_vals = np.full(pred_np.shape[0], np.nan, dtype=np.float64)
    else:
        with torch.inference_mode():
            values = lpips_model(prepare_for_lpips(pred), prepare_for_lpips(target))
        lpips_vals = values.detach().flatten().cpu().numpy().astype(np.float64)
    return mse.detach().cpu().numpy(), psnr.detach().cpu().numpy(), ssim_vals, lpips_vals


def evaluate_model_rows(
    *,
    variant: str,
    model_path: Path | None,
    loader: DataLoader,
    device: torch.device,
    lpips_model,
) -> list[dict[str, Any]]:
    model = None
    if model_path is not None:
        model = DnCNN_2D(in_channels=3).to(device)
        model.load_state_dict(load_model_state(model_path, device))
        model.eval()
    rows: list[dict[str, Any]] = []
    offset = 0
    with torch.inference_mode():
        for x_input, x_target in tqdm(loader, desc=f"Evaluating {variant}", mininterval=10.0):
            x_input = x_input.to(device)
            x_target = x_target.to(device)
            pred = x_input if model is None else model(x_input)
            mse, psnr, ssim_vals, lpips_vals = batch_metrics(pred, x_target, lpips_model)
            for idx in range(int(x_input.shape[0])):
                rows.append(
                    {
                        "variant": variant,
                        "sample_index": int(offset + idx),
                        "MSE": float(mse[idx]),
                        "PSNR": float(psnr[idx]),
                        "SSIM": float(ssim_vals[idx]),
                        "LPIPS": float(lpips_vals[idx]),
                    }
                )
            offset += int(x_input.shape[0])
    return rows


def aggregate_metrics(per_image: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in per_image.groupby("variant", sort=False):
        row: dict[str, Any] = {"variant": variant, "num_images": int(len(group))}
        for metric in ["MSE", "PSNR", "SSIM", "LPIPS"]:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
            row[f"{metric}_median"] = float(group[metric].median())
        rows.append(row)
    return pd.DataFrame(rows)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def stage_evaluate(args: argparse.Namespace, dirs: dict[str, Path]) -> None:
    config = OmegaConf.to_container(OmegaConf.load(dirs["protocol"] / "resolved_downstream_config.yaml"), resolve=True)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    dataset = EvalNoiseDataset(Path(str(config["eval_input"])), Path(str(config["eval_target"])))
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=0)
    lpips_model = create_lpips_model(device, bool(config["lpips_enabled"]))
    rows: list[dict[str, Any]] = []
    rows.extend(evaluate_model_rows(variant="full_existing", model_path=Path(str(config["full_checkpoint"])), loader=loader, device=device, lpips_model=lpips_model))
    eval_variants = []
    if (dirs["training"] / "full_45000" / "best.pt").exists():
        eval_variants.append("full_45000")
    eval_variants.extend(["topk_10", "quantile_spread_10"])
    for variant in eval_variants:
        rows.extend(
            evaluate_model_rows(
                variant=VARIANT_LABELS[variant],
                model_path=dirs["training"] / variant / "best.pt",
                loader=loader,
                device=device,
                lpips_model=lpips_model,
            )
        )
    per_image = pd.DataFrame(rows)
    per_image.to_csv(dirs["evaluation"] / "per_image_metrics.csv", index=False)
    summary = aggregate_metrics(per_image)
    summary.to_csv(dirs["evaluation"] / "summary_metrics.csv", index=False)
    (dirs["evaluation"] / "summary_metrics.md").write_text(dataframe_to_markdown(summary) + "\n", encoding="utf-8")
    write_json(
        dirs["evaluation"] / "evaluation_meta.json",
        {
            "eval_input": str(Path(str(config["eval_input"])).resolve()),
            "eval_target": str(Path(str(config["eval_target"])).resolve()),
            "eval_manifest": str(Path(str(config["eval_manifest"])).resolve()),
            "lpips_enabled": bool(lpips_model is not None),
            **device_info(device),
        },
    )


def bootstrap_ci(values: np.ndarray, seed: int, n: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=np.float64)
    boots = np.empty(n, dtype=np.float64)
    for i in range(n):
        sample = vals[rng.integers(0, len(vals), size=len(vals))]
        boots[i] = float(np.mean(sample))
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def stage_summary(args: argparse.Namespace, dirs: dict[str, Path]) -> None:
    config = OmegaConf.to_container(OmegaConf.load(dirs["protocol"] / "resolved_downstream_config.yaml"), resolve=True)
    eval_summary = pd.read_csv(dirs["evaluation"] / "summary_metrics.csv")
    old_eval = pd.read_csv(args.old_full_root / "05_evaluation" / "evaluation_aggregates.csv")
    train_rows = []
    for variant in ["full_45000", "topk_10", "quantile_spread_10"]:
        meta_path = dirs["training"] / variant / "run_meta.json"
        if meta_path.exists():
            train_rows.append(read_json(meta_path))
    train = {row["variant"]: row for row in train_rows}
    full_existing = eval_summary[eval_summary["variant"] == "full_existing"].iloc[0]
    if "full_45000_e18_t100" in set(eval_summary["variant"]):
        full = eval_summary[eval_summary["variant"] == "full_45000_e18_t100"].iloc[0]
        full_baseline_name = "full_45000_e18_t100"
        full_baseline_status = "strict_45000_baseline"
    else:
        full = full_existing
        full_baseline_name = "full_existing"
        full_baseline_status = str(config["comparison_status_vs_full"])
    rows = []
    rows.append(
        {
            "variant": "full_existing",
            "filter_epoch": np.nan,
            "filter_timestep": np.nan,
            "subset_size": int(config["old_full_train_size"]),
            "optimizer_steps": int(config["total_steps"]),
            "effective_epochs": float(pd.read_csv(args.old_full_root / "04_downstream" / "downstream_training_summary.csv").query("training_subset == 'full'").iloc[0]["effective_epochs"]),
            "train_time_seconds": float(pd.read_csv(args.old_full_root / "04_downstream" / "downstream_training_summary.csv").query("training_subset == 'full'").iloc[0]["wall_clock_train_seconds"]),
            "MSE": float(full_existing["MSE_mean"]),
            "PSNR": float(full_existing["PSNR_mean"]),
            "SSIM": float(full_existing["SSIM_mean"]),
            "LPIPS": float(full_existing["LPIPS_mean"]),
            "delta_PSNR_vs_full": float(full_existing["PSNR_mean"] - full["PSNR_mean"]),
            "delta_SSIM_vs_full": float(full_existing["SSIM_mean"] - full["SSIM_mean"]),
            "delta_LPIPS_vs_full": float(full_existing["LPIPS_mean"] - full["LPIPS_mean"]),
            "comparison_status": "historical_full_50000_non_strict" if full_baseline_name != "full_existing" else str(config["comparison_status_vs_full"]),
        }
    )
    if "full_45000" in train and full_baseline_name == "full_45000_e18_t100":
        meta = train["full_45000"]
        rows.append(
            {
                "variant": "full_45000_e18_t100",
                "filter_epoch": 18,
                "filter_timestep": 100,
                "subset_size": int(meta["subset_size"]),
                "optimizer_steps": int(meta["total_steps"]),
                "effective_epochs": float(meta["effective_epochs"]),
                "train_time_seconds": float(meta["wall_clock_seconds"]),
                "MSE": float(full["MSE_mean"]),
                "PSNR": float(full["PSNR_mean"]),
                "SSIM": float(full["SSIM_mean"]),
                "LPIPS": float(full["LPIPS_mean"]),
                "delta_PSNR_vs_full": 0.0,
                "delta_SSIM_vs_full": 0.0,
                "delta_LPIPS_vs_full": 0.0,
                "comparison_status": full_baseline_status,
            }
        )
    for hist in HISTORICAL_ROWS:
        old_name = "topk_10" if "topk" in hist["variant"] else "quantile_spread_10"
        old_row = old_eval[old_eval["training_subset"] == old_name].iloc[0]
        rows.append(
            {
                "variant": hist["variant"],
                "filter_epoch": np.nan,
                "filter_timestep": 1,
                "subset_size": 5000,
                "optimizer_steps": int(config["total_steps"]),
                "effective_epochs": float(pd.read_csv(args.old_full_root / "04_downstream" / "downstream_training_summary.csv").query("training_subset == @old_name").iloc[0]["effective_epochs"]),
                "train_time_seconds": float(pd.read_csv(args.old_full_root / "04_downstream" / "downstream_training_summary.csv").query("training_subset == @old_name").iloc[0]["wall_clock_train_seconds"]),
                "MSE": float(old_row["mse_mean"]),
                "PSNR": float(hist["PSNR"]),
                "SSIM": float(hist["SSIM"]),
                "LPIPS": float(hist["LPIPS"]),
                "delta_PSNR_vs_full": float(hist["PSNR"] - full["PSNR_mean"]),
                "delta_SSIM_vs_full": float(hist["SSIM"] - full["SSIM_mean"]),
                "delta_LPIPS_vs_full": float(hist["LPIPS"] - full["LPIPS_mean"]),
                "comparison_status": "historical_baseline_non_strict_vs_45000" if full_baseline_name != "full_existing" else "historical_baseline",
            }
        )
    for variant in ["topk_10", "quantile_spread_10"]:
        label = VARIANT_LABELS[variant]
        row = eval_summary[eval_summary["variant"] == label].iloc[0]
        meta = train[variant]
        rows.append(
            {
                "variant": label,
                "filter_epoch": 18,
                "filter_timestep": 100,
                "subset_size": int(meta["subset_size"]),
                "optimizer_steps": int(meta["total_steps"]),
                "effective_epochs": float(meta["effective_epochs"]),
                "train_time_seconds": float(meta["wall_clock_seconds"]),
                "MSE": float(row["MSE_mean"]),
                "PSNR": float(row["PSNR_mean"]),
                "SSIM": float(row["SSIM_mean"]),
                "LPIPS": float(row["LPIPS_mean"]),
                "delta_PSNR_vs_full": float(row["PSNR_mean"] - full["PSNR_mean"]),
                "delta_SSIM_vs_full": float(row["SSIM_mean"] - full["SSIM_mean"]),
                "delta_LPIPS_vs_full": float(row["LPIPS_mean"] - full["LPIPS_mean"]),
                "comparison_status": "strict_vs_full_45000" if full_baseline_name == "full_45000_e18_t100" else str(config["comparison_status_vs_full"]),
            }
        )
    comp = pd.DataFrame(rows)
    comp.to_csv(dirs["summary"] / "downstream_comparison.csv", index=False)
    (dirs["summary"] / "downstream_comparison.md").write_text(dataframe_to_markdown(comp) + "\n", encoding="utf-8")

    per_image = pd.read_csv(dirs["evaluation"] / "per_image_metrics.csv")
    topk = per_image[per_image["variant"] == "topk_10_e18_t100"].sort_values("sample_index")
    quant = per_image[per_image["variant"] == "quantile_spread_10_e18_t100"].sort_values("sample_index")
    delta = topk[["sample_index", "MSE", "PSNR", "SSIM", "LPIPS"]].merge(
        quant[["sample_index", "MSE", "PSNR", "SSIM", "LPIPS"]],
        on="sample_index",
        suffixes=("_topk", "_quantile"),
    )
    for metric in ["MSE", "PSNR", "SSIM", "LPIPS"]:
        delta[f"delta_{metric}_topk_minus_quantile"] = delta[f"{metric}_topk"] - delta[f"{metric}_quantile"]
    delta.to_csv(dirs["summary"] / "per_image_deltas.csv", index=False)
    stats = []
    for metric in ["PSNR", "SSIM", "LPIPS", "MSE"]:
        vals = delta[f"delta_{metric}_topk_minus_quantile"].to_numpy(dtype=np.float64)
        lo, hi = bootstrap_ci(vals, seed=42 + len(stats) * 17, n=int(args.bootstrap_resamples))
        if metric in {"PSNR", "SSIM"}:
            topk_better = float(np.mean(vals > 0))
            quant_better = float(np.mean(vals < 0))
        else:
            topk_better = float(np.mean(vals < 0))
            quant_better = float(np.mean(vals > 0))
        stats.append(
            {
                "metric": metric,
                "mean_delta_topk_minus_quantile": float(np.mean(vals)),
                "median_delta_topk_minus_quantile": float(np.median(vals)),
                "bootstrap95_ci_low": lo,
                "bootstrap95_ci_high": hi,
                "fraction_topk_better": topk_better,
                "fraction_quantile_better": quant_better,
            }
        )
    pd.DataFrame(stats).to_csv(dirs["summary"] / "per_image_delta_summary.csv", index=False)

    t1_rows = []
    for old_name, new_label in [("topk_10", "topk_10_e18_t100"), ("quantile_spread_10", "quantile_spread_10_e18_t100")]:
        old_row = old_eval[old_eval["training_subset"] == old_name].iloc[0]
        new_row = eval_summary[eval_summary["variant"] == new_label].iloc[0]
        t1_rows.append({"variant": old_name, "filter_timestep": 1, "PSNR": float(old_row["psnr_mean"]), "SSIM": float(old_row["ssim_mean"]), "LPIPS": float(old_row["lpips_mean"]), "delta_vs_t1": 0.0})
        t1_rows.append({"variant": new_label, "filter_timestep": 100, "PSNR": float(new_row["PSNR_mean"]), "SSIM": float(new_row["SSIM_mean"]), "LPIPS": float(new_row["LPIPS_mean"]), "delta_vs_t1": float(new_row["PSNR_mean"] - old_row["psnr_mean"])})
    pd.DataFrame(t1_rows).to_csv(dirs["summary"] / "t1_vs_t100.csv", index=False)

    strict_full_row = comp[comp["variant"] == full_baseline_name].iloc[0]
    topk_new = comp[comp["variant"] == "topk_10_e18_t100"].iloc[0]
    quant_new = comp[comp["variant"] == "quantile_spread_10_e18_t100"].iloc[0]
    lines = [
        "# Meeting Summary",
        "",
        "1. Used filter subsets: `topk_10` and `quantile_spread_10` from `06_filtering_e18_t100_mc16`.",
        "2. DDPM filtering configuration: epoch 18, report timestep 100, code index 99, MC K=16.",
        f"3. Subset sizes: Top-K={int(topk_new['subset_size'])}, Quantile={int(quant_new['subset_size'])}.",
        f"4. Optimizer steps: {int(config['total_steps'])}; batch size: {int(config['batch_size'])}.",
        f"5. Main Full baseline: {full_baseline_name}; status: {full_baseline_status}. Old Full 50k remains historical/non-strict.",
        f"   Full baseline metrics: PSNR={float(strict_full_row['PSNR']):.6f}, SSIM={float(strict_full_row['SSIM']):.6f}, LPIPS={float(strict_full_row['LPIPS']):.6f}.",
        f"6. New Top-K: PSNR={float(topk_new['PSNR']):.6f}, SSIM={float(topk_new['SSIM']):.6f}, LPIPS={float(topk_new['LPIPS']):.6f}.",
        f"7. New Quantile: PSNR={float(quant_new['PSNR']):.6f}, SSIM={float(quant_new['SSIM']):.6f}, LPIPS={float(quant_new['LPIPS']):.6f}.",
        f"8. Top-K minus Quantile: PSNR={float(topk_new['PSNR'] - quant_new['PSNR']):+.6f}, SSIM={float(topk_new['SSIM'] - quant_new['SSIM']):+.6f}, LPIPS={float(topk_new['LPIPS'] - quant_new['LPIPS']):+.6f}.",
        "9. Visual panels:",
        "- 05_figures/01_downstream_representative_examples.png",
        "- 05_figures/02_topk_vs_quantile_extremes.png",
        "- 05_figures/03_t1_vs_t100_examples.png",
        "- 05_figures/04_per_image_metric_deltas.png",
        "10. Limitations: old Full was trained on 50,000 CIFAR train images, while the new filtered working split has 45,000 images; old Full comparison is historical/non-strict. New Full 45k, Top-K and Quantile use the same initial state, optimizer steps, task-noise protocol and eval cache.",
    ]
    (dirs["summary"] / "meeting_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def tensor_img(x: torch.Tensor, size: int = 148) -> Image.Image:
    arr = x.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255).round().astype(np.uint8)).resize((size, size), Image.Resampling.NEAREST)


def abs_error_img(pred: torch.Tensor, target: torch.Tensor, size: int = 148) -> Image.Image:
    err = (pred.detach().cpu().clamp(0, 1) - target.detach().cpu().clamp(0, 1)).abs()
    mx = float(err.max().item())
    if mx > 0:
        err = err / mx
    return tensor_img(err, size=size)


def predict_selected(config: dict[str, Any], dirs: dict[str, Path], indices: list[int], include_old: bool = False) -> dict[str, torch.Tensor]:
    device = resolve_device(None)
    inputs = torch.load(Path(str(config["eval_input"])), map_location="cpu").to(torch.float32)[indices]
    targets = torch.load(Path(str(config["eval_target"])), map_location="cpu").to(torch.float32)[indices]
    payload = {"Clean target": targets, "Noisy input": inputs}
    full_45000_path = dirs["training"] / "full_45000" / "best.pt"
    model_paths = {
        "Full 45k" if full_45000_path.exists() else "Full": full_45000_path if full_45000_path.exists() else Path(str(config["full_checkpoint"])),
        "New Top-K": dirs["training"] / "topk_10" / "best.pt",
        "New Quantile": dirs["training"] / "quantile_spread_10" / "best.pt",
    }
    if include_old:
        old_root = Path(str(config["old_full_root"]))
        model_paths = {
            "Old Top-K t=1": old_root / "04_downstream" / "topk_10" / "best.pt",
            "New Top-K t=100": dirs["training"] / "topk_10" / "best.pt",
            "Old Quantile t=1": old_root / "04_downstream" / "quantile_spread_10" / "best.pt",
            "New Quantile t=100": dirs["training"] / "quantile_spread_10" / "best.pt",
        }
    with torch.inference_mode():
        for label, path in model_paths.items():
            model = DnCNN_2D(in_channels=3).to(device)
            model.load_state_dict(load_model_state(path, device))
            model.eval()
            preds = []
            for start in range(0, len(inputs), 32):
                preds.append(model(inputs[start : start + 32].to(device)).detach().cpu().clamp(0, 1))
            payload[label] = torch.cat(preds, dim=0)
    return payload


def metric_text(per_image: pd.DataFrame, variant: str, sample_index: int) -> str:
    row = per_image[(per_image["variant"] == variant) & (per_image["sample_index"] == sample_index)].iloc[0]
    return f"PSNR {row.PSNR:.2f}\nSSIM {row.SSIM:.3f}\nLPIPS {row.LPIPS:.3f}"


def save_grid_panel(output: Path, title: str, rows: list[dict[str, Any]], columns: list[str], images: dict[str, torch.Tensor], captions: dict[tuple[int, str], str]) -> None:
    img_size = 188
    col_w = 480
    row_h = 285
    margin_x = 28
    row_label_w = 120
    title_h = 52
    header_h = 28
    canvas = Image.new("RGB", (margin_x * 2 + row_label_w + len(columns) * col_w, title_h + header_h + len(rows) * row_h + 20), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(24, True)
    header_font = font(16, True)
    text_font = font(14)
    bbox = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((canvas.width - bbox[2]) // 2, 12), title, font=title_font, fill=(0, 0, 0))
    for c, col in enumerate(columns):
        draw.text((margin_x + row_label_w + c * col_w, title_h), col, font=header_font, fill=(0, 0, 0))
    for r, row in enumerate(rows):
        y = title_h + header_h + r * row_h
        draw.text((margin_x, y + 70), str(row["label"]), font=header_font, fill=(0, 0, 0))
        sample_pos = int(row["pos"])
        for c, col in enumerate(columns):
            x = margin_x + row_label_w + c * col_w
            canvas.paste(tensor_img(images[col][sample_pos], img_size), (x, y))
            draw.rectangle([x, y, x + img_size - 1, y + img_size - 1], outline=(90, 90, 90))
            text = captions.get((sample_pos, col), "")
            if text:
                draw.multiline_text((x, y + img_size + 8), text, font=text_font, fill=(0, 0, 0), spacing=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def stage_figures(args: argparse.Namespace, dirs: dict[str, Path]) -> None:
    config = OmegaConf.to_container(OmegaConf.load(dirs["protocol"] / "resolved_downstream_config.yaml"), resolve=True)
    per_image = pd.read_csv(dirs["evaluation"] / "per_image_metrics.csv")
    eval_manifest = pd.read_csv(Path(str(config["eval_manifest"])))
    heavy = int(eval_manifest.sort_values("realized_psnr_db_postclip").iloc[0]["cache_pos"])
    light = int(eval_manifest.sort_values("realized_psnr_db_postclip").iloc[-1]["cache_pos"])
    med_val = float(eval_manifest["realized_psnr_db_postclip"].median())
    median = int((eval_manifest["realized_psnr_db_postclip"] - med_val).abs().sort_values().index[0])
    reps = [heavy, median, light]
    rep_rows = [{"label": "heavy", "pos": 0}, {"label": "median", "pos": 1}, {"label": "light", "pos": 2}]
    images = predict_selected(config, dirs, reps)
    captions = {}
    full_col = "Full 45k" if "Full 45k" in images else "Full"
    full_variant = "full_45000_e18_t100" if full_col == "Full 45k" else "full_existing"
    for pos, sample_idx in enumerate(reps):
        for col, variant in [(full_col, full_variant), ("New Top-K", "topk_10_e18_t100"), ("New Quantile", "quantile_spread_10_e18_t100")]:
            captions[(pos, col)] = metric_text(per_image, variant, sample_idx)
    save_grid_panel(
        dirs["figures"] / "01_downstream_representative_examples.png",
        "Downstream representative examples",
        rep_rows,
        ["Clean target", "Noisy input", full_col, "New Top-K", "New Quantile"],
        images,
        captions,
    )

    delta = pd.read_csv(dirs["summary"] / "per_image_deltas.csv")
    max_idx = int(delta.loc[delta["delta_PSNR_topk_minus_quantile"].idxmax(), "sample_index"])
    zero_idx = int((delta["delta_PSNR_topk_minus_quantile"].abs()).idxmin())
    zero_idx = int(delta.loc[zero_idx, "sample_index"])
    min_idx = int(delta.loc[delta["delta_PSNR_topk_minus_quantile"].idxmin(), "sample_index"])
    ex_indices = [max_idx, zero_idx, min_idx]
    ex_rows = [{"label": "max Top-K - Quantile", "pos": 0}, {"label": "near zero", "pos": 1}, {"label": "min Top-K - Quantile", "pos": 2}]
    ex_images = predict_selected(config, dirs, ex_indices)
    ex_images["Absolute error Top-K"] = (ex_images["New Top-K"] - ex_images["Clean target"]).abs()
    ex_images["Absolute error Quantile"] = (ex_images["New Quantile"] - ex_images["Clean target"]).abs()
    ex_captions = {}
    for pos, sample_idx in enumerate(ex_indices):
        drow = delta[delta["sample_index"] == sample_idx].iloc[0]
        ex_captions[(pos, "Top-K")] = metric_text(per_image, "topk_10_e18_t100", sample_idx) + f"\ndPSNR {drow.delta_PSNR_topk_minus_quantile:+.2f}"
        ex_captions[(pos, "Quantile")] = metric_text(per_image, "quantile_spread_10_e18_t100", sample_idx) + f"\ndPSNR {drow.delta_PSNR_topk_minus_quantile:+.2f}"
    # Rename keys for requested column titles.
    ex_images["Top-K"] = ex_images.pop("New Top-K")
    ex_images["Quantile"] = ex_images.pop("New Quantile")
    save_grid_panel(
        dirs["figures"] / "02_topk_vs_quantile_extremes.png",
        "Top-K vs Quantile extremes",
        ex_rows,
        ["Clean target", "Noisy input", "Top-K", "Quantile", "Absolute error Top-K", "Absolute error Quantile"],
        ex_images,
        ex_captions,
    )

    if (Path(str(config["old_full_root"])) / "04_downstream" / "topk_10" / "best.pt").exists():
        old_images = predict_selected(config, dirs, reps, include_old=True)
        old_images["Clean"] = images["Clean target"]
        old_images["Noisy input"] = images["Noisy input"]
        old_captions = {}
        old_per = pd.read_csv(Path(str(config["old_full_root"])) / "05_evaluation" / "per_image_metrics.csv")
        for pos, sample_idx in enumerate(reps):
            for col, variant in [
                ("Old Top-K t=1", "topk_10"),
                ("New Top-K t=100", "topk_10_e18_t100"),
                ("Old Quantile t=1", "quantile_spread_10"),
                ("New Quantile t=100", "quantile_spread_10_e18_t100"),
            ]:
                source = old_per.rename(columns={"training_subset": "variant", "mse": "MSE", "psnr": "PSNR", "ssim": "SSIM", "lpips": "LPIPS"}) if variant in {"topk_10", "quantile_spread_10"} else per_image
                old_captions[(pos, col)] = metric_text(source, variant, sample_idx)
        save_grid_panel(
            dirs["figures"] / "03_t1_vs_t100_examples.png",
            "Old t=1 vs new t=100",
            rep_rows,
            ["Clean", "Noisy input", "Old Top-K t=1", "New Top-K t=100", "Old Quantile t=1", "New Quantile t=100"],
            old_images,
            old_captions,
        )
    else:
        (dirs["figures"] / "03_t1_vs_t100_examples.SKIPPED.txt").write_text("Old t=1 checkpoints not found.\n", encoding="utf-8")

    delta_summary = pd.read_csv(dirs["summary"] / "per_image_delta_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8), constrained_layout=True)
    for ax, metric in zip(axes.ravel(), ["PSNR", "SSIM", "LPIPS", "MSE"]):
        vals = delta[f"delta_{metric}_topk_minus_quantile"].to_numpy(dtype=np.float64)
        row = delta_summary[delta_summary["metric"] == metric].iloc[0]
        ax.hist(vals, bins=60, density=True, color="#4c78a8", alpha=0.75)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.axvline(float(row["mean_delta_topk_minus_quantile"]), color="#f58518", linewidth=2)
        ax.set_title(f"{metric}: mean {row['mean_delta_topk_minus_quantile']:+.4f} CI [{row['bootstrap95_ci_low']:+.4f}, {row['bootstrap95_ci_high']:+.4f}]")
        ax.set_xlabel(f"Top-K minus Quantile {metric}")
        ax.set_ylabel("Density")
    fig.savefig(dirs["figures"] / "04_per_image_metric_deltas.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    dirs = ensure_dirs(args.output_dir)
    stages = ["audit", "train_full", "train_topk", "train_quantile", "evaluate", "summary", "figures"] if args.stage == "all" else [args.stage]
    if "audit" in stages:
        audit(args, dirs)
    if "train_full" in stages:
        train_variant(args, dirs, "full_45000")
    if "train_topk" in stages:
        train_variant(args, dirs, "topk_10")
    if "train_quantile" in stages:
        train_variant(args, dirs, "quantile_spread_10")
    if "evaluate" in stages:
        stage_evaluate(args, dirs)
    if "summary" in stages:
        stage_summary(args, dirs)
    if "figures" in stages:
        stage_figures(args, dirs)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
