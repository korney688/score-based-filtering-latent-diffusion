from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
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

from scripts.run_cifar_selected_filtering import build_model, forward_noise_seeds, score_batch
from src.TDnCNN_image import DnCNN_2D, align_lpips_scaling_layer, prepare_for_lpips
from src.dataset_registry import build_torchvision_split, load_dataset_config
from src.evaluation.image_lab_replication_analysis import select_quantile_spread_exact
from src.image_lab_replication_datasets import (
    CorpusTaskNoiseDataset,
    EvalNoiseDataset,
    MaterializedCorpusDataset,
    materialize_variable_awgn_corpus,
    tensor_class_id,
)


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
EXPECTED_FULL_METRICS = {"PSNR": 20.070554, "SSIM": 0.620386, "LPIPS": 0.269916}
PREVIOUS_45K_ROWS = [
    {"variant": "Full 45k", "train_size": 45000, "PSNR": 19.7610, "SSIM": 0.5904, "LPIPS": 0.2943},
    {"variant": "Top-K 45k", "train_size": 4500, "PSNR": 19.9386, "SSIM": 0.6109, "LPIPS": 0.2821},
    {"variant": "Quantile 45k", "train_size": 4500, "PSNR": 19.3978, "SSIM": 0.5758, "LPIPS": 0.3288},
]
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-10 full 50k e18/t100 MC16 filtering and TDnCNN pipeline.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--stage",
        choices=[
            "audit",
            "prepare",
            "score",
            "filter",
            "train_topk",
            "train_quantile",
            "evaluate",
            "summary",
            "drunet_full",
            "drunet_topk",
            "drunet_quantile",
            "drunet_evaluate",
            "drunet_external",
            "drunet_summary",
            "drunet_all",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def resolve_path(path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    path = Path(str(path_like))
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path: Path, output_root: Path | None) -> dict[str, Any]:
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(cfg, dict):
        raise TypeError(f"Config is not a mapping: {config_path}")
    cfg["config_path"] = str(config_path.resolve())
    paths = cfg.setdefault("paths", {})
    if output_root is not None:
        paths["output_root"] = str(output_root)
    cfg["resolved_paths"] = {key: str(resolve_path(value)) for key, value in paths.items()}
    return cfg


def dirs(output_root: Path) -> dict[str, Path]:
    payload = {
        "audit": output_root / "00_audit",
        "corpus": output_root / "01_corpus",
        "corpus_cache": output_root / "01_corpus" / "corpus_cache",
        "scores": output_root / "02_scores",
        "score_chunks": output_root / "02_scores" / "chunks",
        "filtering": output_root / "03_filtering",
        "figures": output_root / "04_figures",
        "downstream": output_root / "05_downstream",
        "evaluation": output_root / "06_evaluation",
        "summary": output_root / "07_summary",
    }
    return payload


def ensure_dirs(paths: dict[str, Path], dry_run: bool) -> None:
    if dry_run:
        return
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] would write {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any, dry_run: bool) -> None:
    write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n", dry_run)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def read_old_full(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg["resolved_paths"]["existing_full_root"])
    resolved_config = OmegaConf.to_container(OmegaConf.load(root / "00_protocol" / "resolved_config.yaml"), resolve=True)
    train_summary = pd.read_csv(root / "04_downstream" / "downstream_training_summary.csv")
    train_full = train_summary[train_summary["training_subset"] == "full"].iloc[0].to_dict()
    metrics = pd.read_csv(root / "05_evaluation" / "evaluation_aggregates.csv")
    metric_full = metrics[metrics["training_subset"] == "full"].iloc[0].to_dict()
    return {
        "root": root,
        "resolved_config": resolved_config,
        "train_summary": train_full,
        "metrics": metric_full,
        "checkpoint": root / "04_downstream" / "full" / "best.pt",
        "initial_state": root / "04_downstream" / "initial_state_dict.pt",
        "eval_input": root / "01_corpus" / "test_eval_input.pt",
        "eval_target": root / "01_corpus" / "test_eval_target.pt",
        "eval_manifest": root / "01_corpus" / "test_noise_manifest.csv",
        "corpus": root / "01_corpus" / "train_corpus.pt",
        "manifest": root / "01_corpus" / "train_manifest.csv",
    }


def ddpm_split_info(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg["resolved_paths"]["score_validation_root"])
    config_path = root / "00_protocol" / "resolved_score_validation_config.yaml"
    sv_cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    ddpm_indices = np.load(root / "00_protocol" / "ddpm_filter_train_original_indices.npy").astype(np.int64)
    tune = np.load(root / "00_protocol" / "score_tuning_original_indices.npy").astype(np.int64)
    confirm = np.load(root / "00_protocol" / "score_confirmation_original_indices.npy").astype(np.int64)
    ckpt = Path(cfg["resolved_paths"]["ddpm_checkpoint"])
    return {
        "root": root,
        "config_path": config_path,
        "resolved_config": sv_cfg,
        "checkpoint": ckpt,
        "checkpoint_sha256": sha256_file(ckpt) if ckpt.exists() else None,
        "ddpm_filter_train_size": int(len(ddpm_indices)),
        "score_tuning_size": int(len(tune)),
        "score_confirmation_size": int(len(confirm)),
        "ddpm_trained_on": "ddpm_filter_train_45000" if len(ddpm_indices) == 45000 else f"train_{len(ddpm_indices)}",
    }


def audit(cfg: dict[str, Any], paths: dict[str, Path], dry_run: bool) -> dict[str, Any]:
    old = read_old_full(cfg)
    ddpm = ddpm_split_info(cfg)
    old_cfg = old["resolved_config"]
    metrics = old["metrics"]
    train = old["train_summary"]
    manifest = pd.read_csv(old["manifest"], usecols=["original_index"])
    eval_cache_ok = old["eval_input"].exists() and old["eval_target"].exists() and old["eval_manifest"].exists()
    metric_match = (
        abs(float(metrics["psnr_mean"]) - EXPECTED_FULL_METRICS["PSNR"]) < 1e-5
        and abs(float(metrics["ssim_mean"]) - EXPECTED_FULL_METRICS["SSIM"]) < 1e-5
        and abs(float(metrics["lpips_mean"]) - EXPECTED_FULL_METRICS["LPIPS"]) < 1e-5
    )
    checks = {
        "train_split_size": int(train["sample_count"]),
        "manifest_rows": int(len(manifest)),
        "uses_full_cifar_train": int(train["sample_count"]) == 50000 and len(manifest) == 50000,
        "batch_size": int(train["batch_size"]),
        "optimizer_steps": int(train["total_steps"]),
        "initial_state_available": old["initial_state"].exists(),
        "full_checkpoint_available": old["checkpoint"].exists(),
        "eval_noise_cache_available": eval_cache_ok,
        "historical_metrics_match_expected": metric_match,
        "task_noise_matches_config": old_cfg.get("task_noise") == cfg.get("task_noise"),
        "eval_noise_matches_config": old_cfg.get("eval_noise") == cfg.get("eval_noise"),
        "model_seed": int(old_cfg["seeds"]["model_seed"]),
        "eval_seed": int(old_cfg["seeds"]["eval_seed"]),
        "checkpoint_selection_rule": str(old_cfg["downstream"].get("checkpoint_selection", "best_train_loss")),
    }
    strict = bool(
        checks["uses_full_cifar_train"]
        and checks["initial_state_available"]
        and checks["full_checkpoint_available"]
        and checks["eval_noise_cache_available"]
        and checks["historical_metrics_match_expected"]
        and checks["batch_size"] == int(cfg["downstream"]["batch_size"])
        and checks["optimizer_steps"] == int(cfg["downstream"]["total_steps"])
        and checks["task_noise_matches_config"]
        and checks["eval_noise_matches_config"]
    )
    payload = {
        "existing_full50k": {
            "root": str(old["root"].resolve()),
            "checkpoint": str(old["checkpoint"].resolve()),
            "initial_state_dict": str(old["initial_state"].resolve()),
            "resolved_config": str((old["root"] / "00_protocol" / "resolved_config.yaml").resolve()),
            "training_history": str((old["root"] / "04_downstream" / "full" / "training_history.csv").resolve()),
            "training_summary": str((old["root"] / "04_downstream" / "downstream_training_summary.csv").resolve()),
            "summary_metrics": str((old["root"] / "05_evaluation" / "evaluation_aggregates.csv").resolve()),
            "run_summary": str((old["root"] / "06_summary" / "run_summary.md").resolve()),
            "corpus_cache": str(old["corpus"].resolve()),
            "corpus_manifest": str(old["manifest"].resolve()),
            "eval_input": str(old["eval_input"].resolve()),
            "eval_target": str(old["eval_target"].resolve()),
            "eval_manifest": str(old["eval_manifest"].resolve()),
            "train_size": int(train["sample_count"]),
            "optimizer_steps": int(train["total_steps"]),
            "batch_size": int(train["batch_size"]),
            "task_noise": old_cfg.get("task_noise"),
            "eval_noise": old_cfg.get("eval_noise"),
            "model_seed": int(old_cfg["seeds"]["model_seed"]),
            "eval_seed": int(old_cfg["seeds"]["eval_seed"]),
            "initial_state_sha256": sha256_file(old["initial_state"]) if old["initial_state"].exists() else None,
            "checkpoint_sha256": sha256_file(old["checkpoint"]) if old["checkpoint"].exists() else None,
            "metrics": {
                "MSE": float(metrics["mse_mean"]),
                "PSNR": float(metrics["psnr_mean"]),
                "SSIM": float(metrics["ssim_mean"]),
                "LPIPS": float(metrics["lpips_mean"]),
            },
            "strict_baseline_usable": strict,
        },
        "ddpm_epoch18": {
            "checkpoint": str(ddpm["checkpoint"].resolve()),
            "checkpoint_sha256": ddpm["checkpoint_sha256"],
            "trained_split": ddpm["ddpm_trained_on"],
            "ddpm_filter_train_size": ddpm["ddpm_filter_train_size"],
            "score_tuning_size": ddpm["score_tuning_size"],
            "score_confirmation_size": ddpm["score_confirmation_size"],
            "transfer_note": "DDPM epoch 18 was trained on 45k and will be applied to full CIFAR-10 train 50k for score inference.",
        },
        "checks": checks,
        "comparison_status": "strict_baseline" if strict else "partially_compatible",
    }
    write_json(paths["audit"] / "full50k_audit.json", payload, dry_run)
    lines = [
        "# Full CIFAR-10 50k Audit",
        "",
        f"- Full root: `{payload['existing_full50k']['root']}`",
        f"- Full checkpoint: `{payload['existing_full50k']['checkpoint']}`",
        f"- Train split size: {payload['existing_full50k']['train_size']}",
        f"- Batch size: {payload['existing_full50k']['batch_size']}",
        f"- Optimizer steps: {payload['existing_full50k']['optimizer_steps']}",
        f"- Initial-state hash: `{payload['existing_full50k']['initial_state_sha256']}`",
        f"- Eval cache: `{payload['existing_full50k']['eval_input']}`",
        f"- PSNR/SSIM/LPIPS: {payload['existing_full50k']['metrics']['PSNR']:.6f} / {payload['existing_full50k']['metrics']['SSIM']:.6f} / {payload['existing_full50k']['metrics']['LPIPS']:.6f}",
        f"- Strict baseline usable: {payload['existing_full50k']['strict_baseline_usable']}",
        "",
        "## Noise Protocols",
        f"- Task noise: `{payload['existing_full50k']['task_noise']}`",
        f"- Eval noise: `{payload['existing_full50k']['eval_noise']}`",
        "",
        "## DDPM Epoch 18",
        f"- Checkpoint: `{payload['ddpm_epoch18']['checkpoint']}`",
        f"- Trained split: {payload['ddpm_epoch18']['trained_split']}",
        f"- Transfer note: {payload['ddpm_epoch18']['transfer_note']}",
    ]
    write_text(paths["audit"] / "full50k_audit.md", "\n".join(lines) + "\n", dry_run)
    print(f"audit_full_root={payload['existing_full50k']['root']}")
    print(f"audit_full_checkpoint={payload['existing_full50k']['checkpoint']}")
    print(f"audit_strict_baseline={payload['existing_full50k']['strict_baseline_usable']}")
    print(f"audit_ddpm_checkpoint={payload['ddpm_epoch18']['checkpoint']}")
    print(f"audit_ddpm_trained_split={payload['ddpm_epoch18']['trained_split']}")
    return payload


def prepare(cfg: dict[str, Any], paths: dict[str, Path], dry_run: bool) -> None:
    old = read_old_full(cfg)
    source_manifest = old["manifest"]
    source_corpus = old["corpus"]
    out_manifest = paths["corpus"] / "corpus_manifest.csv"
    out_config = paths["corpus"] / "resolved_config.yaml"
    if not source_manifest.exists() or not source_corpus.exists():
        if dry_run:
            print("[dry-run] would materialize full CIFAR-10 variable AWGN corpus")
            return
        dataset_cfg = load_dataset_config("cifar10")
        train_ds = build_torchvision_split(
            dataset_cfg=dataset_cfg,
            train=True,
            data_root=Path(str(dataset_cfg["paths"]["data_root"])),
            transform_profile="tensor",
            download=bool(cfg.get("download", False)),
        )
        materialize_variable_awgn_corpus(
            dataset=train_ds,
            dataset_name="cifar10",
            split="train",
            original_indices=np.arange(50000, dtype=np.int64),
            output_dir=paths["corpus_cache"],
            target_psnr_min_db=10.0,
            target_psnr_max_db=40.0,
            corpus_seed=int(cfg["corpus_seed"]),
            corruption_version="variable_awgn_quality_v1",
            overwrite=False,
        )
        source_corpus = paths["corpus_cache"] / "train_corpus.pt"
        source_manifest = paths["corpus_cache"] / "train_manifest.csv"
    manifest = pd.read_csv(source_manifest)
    if len(manifest) != 50000:
        raise RuntimeError(f"Full corpus manifest is not 50000 rows: {source_manifest}")
    requested = [
        "original_index",
        "class_id",
        "target_psnr_db",
        "realized_psnr_db_postclip",
        "realized_snr_db_postclip",
        "sigma",
        "corruption_mse",
        "clipped_fraction",
        "signal_power",
        "sample_seed",
    ]
    if "corruption_mse" not in manifest and "noise_power_postclip" in manifest:
        manifest["corruption_mse"] = manifest["noise_power_postclip"]
    missing = [col for col in requested if col not in manifest.columns]
    if missing:
        raise RuntimeError(f"Corpus manifest missing columns {missing}: {source_manifest}")
    if not dry_run:
        paths["corpus"].mkdir(parents=True, exist_ok=True)
        manifest[requested].to_csv(out_manifest, index=False)
        (paths["corpus_cache"] / "REUSED_FROM.txt").write_text(str(source_corpus.resolve()) + "\n", encoding="utf-8")
        resolved = {
            "n_full": 50000,
            "use_full_cifar_train": True,
            "exclude_score_validation_split": False,
            "corpus_seed": int(cfg["corpus_seed"]),
            "target_psnr_db": "Uniform(10, 40) dB",
            "corpus_cache_path": str(source_corpus.resolve()),
            "source_manifest_path": str(source_manifest.resolve()),
            "local_manifest_path": str(out_manifest.resolve()),
            "reuse_existing_full50k_corpus": True,
        }
        OmegaConf.save(config=OmegaConf.create(resolved), f=out_config)
    else:
        print(f"[dry-run] would reuse full corpus {source_corpus}")
    print(f"prepare_n_full={len(manifest)}")
    print(f"prepare_corpus_cache={source_corpus.resolve()}")
    print(f"prepare_manifest={source_manifest.resolve()}")


def full_original_indices() -> np.ndarray:
    return np.arange(50000, dtype=np.int64)


def score(cfg: dict[str, Any], paths: dict[str, Path], dry_run: bool, resume: bool, device_arg: str | None) -> None:
    out = paths["scores"] / "scores_full50k.csv"
    if dry_run:
        print(f"[dry-run] would score 50000 images into {out}")
        return
    if out.exists() and resume:
        existing = pd.read_csv(out, usecols=["original_index"])
        print(f"score_resumed_existing=true")
        print(f"score_n_images={len(existing)}")
        return
    old = read_old_full(cfg)
    sv_cfg = OmegaConf.to_container(
        OmegaConf.load(Path(cfg["resolved_paths"]["score_validation_root"]) / "00_protocol" / "resolved_score_validation_config.yaml"),
        resolve=True,
    )
    ckpt = Path(cfg["resolved_paths"]["ddpm_checkpoint"])
    device = resolve_device(device_arg)
    model = build_model(sv_cfg, ckpt, device)
    manifest = pd.read_csv(old["manifest"])
    dataset = MaterializedCorpusDataset(old["corpus"], old["manifest"], original_indices=full_original_indices(), normalized=False, return_original_index=True)
    loader = DataLoader(dataset, batch_size=int(cfg.get("score_batch_size", 256)), shuffle=False, num_workers=int(cfg.get("num_workers", 0)), pin_memory=device.type == "cuda")
    seeds = forward_noise_seeds(int(cfg.get("forward_seed", cfg.get("subset_seed", 42))), int(cfg["mc_k"]))
    done = {path.stem for path in paths["score_chunks"].glob("chunk_*.csv")} if resume else set()
    start = time.perf_counter()
    paths["score_chunks"].mkdir(parents=True, exist_ok=True)
    for chunk_id, (x, original) in enumerate(tqdm(loader, desc="Scoring full50k")):
        name = f"chunk_{chunk_id:05d}"
        path = paths["score_chunks"] / f"{name}.csv"
        if name in done:
            continue
        values = score_batch(model, x, int(cfg["code_timestep"]), seeds, chunk_id, device, use_amp=True).detach().cpu().numpy().astype(np.float64)
        pd.DataFrame({"original_index": original.numpy().astype(np.int64), "score_value": values}).to_csv(path, index=False)
    parts = [pd.read_csv(path) for path in sorted(paths["score_chunks"].glob("chunk_*.csv"))]
    score_df = pd.concat(parts, ignore_index=True)
    columns = [
        "original_index",
        "class_id",
        "target_psnr_db",
        "realized_psnr_db_postclip",
        "realized_snr_db_postclip",
        "sigma",
        "corruption_mse",
        "clipped_fraction",
        "signal_power",
    ]
    if "corruption_mse" not in manifest and "noise_power_postclip" in manifest:
        manifest["corruption_mse"] = manifest["noise_power_postclip"]
    score_df = score_df.merge(manifest[columns], on="original_index", how="left")
    score_df["class_name"] = score_df["class_id"].map(lambda item: CIFAR10_CLASSES[int(item)])
    score_df = score_df.sort_values(["score_value", "original_index"], ascending=[True, True]).reset_index(drop=True)
    score_df["score_rank"] = np.arange(1, len(score_df) + 1, dtype=np.int64)
    score_df["score_percentile"] = (score_df["score_rank"] - 1) / max(len(score_df) - 1, 1)
    if len(score_df) != 50000 or score_df["original_index"].nunique() != 50000:
        raise RuntimeError("scores_full50k.csv must contain 50000 unique original_index rows")
    score_df.to_csv(out, index=False)
    meta = {
        "n_images": int(len(score_df)),
        "checkpoint": str(ckpt.resolve()),
        "checkpoint_sha256": sha256_file(ckpt),
        "checkpoint_epoch": int(cfg["checkpoint_epoch"]),
        "report_timestep": int(cfg["report_timestep"]),
        "code_timestep": int(cfg["code_timestep"]),
        "mc_k": int(cfg["mc_k"]),
        "forward_noise_seeds": seeds,
        "score_seconds": time.perf_counter() - start,
    }
    write_json(paths["scores"] / "score_meta.json", meta, dry_run=False)
    print(f"score_n_images={len(score_df)}")
    print(f"score_output={out.resolve()}")


def subset_frame(scores: pd.DataFrame, indices: np.ndarray) -> pd.DataFrame:
    return scores[scores["original_index"].isin(set(indices.astype(int).tolist()))].copy()


def filter_subsets(cfg: dict[str, Any], paths: dict[str, Path], dry_run: bool) -> None:
    scores_path = paths["scores"] / "scores_full50k.csv"
    if dry_run:
        print(f"[dry-run] would form Top-K and Quantile 5000 from {scores_path}")
        return
    scores = pd.read_csv(scores_path)
    n_full = len(scores)
    n_subset = int(round(float(cfg["subset_fraction"]) * n_full))
    if n_full != 50000 or n_subset != 5000:
        raise RuntimeError(f"Expected full=50000 and subset=5000, got full={n_full}, subset={n_subset}")
    topk = (
        scores.sort_values(["score_value", "original_index"], ascending=[True, True])
        .head(n_subset)["original_index"]
        .to_numpy(dtype=np.int64)
    )
    quantile = select_quantile_spread_exact(scores, n_subset, seed=int(cfg["subset_seed"]), min_points_per_bin=30)
    subsets = {"full": np.sort(scores["original_index"].to_numpy(dtype=np.int64)), "topk_10": np.sort(topk), "quantile_spread_10": np.sort(quantile)}
    if len(np.unique(subsets["topk_10"])) != n_subset or len(np.unique(subsets["quantile_spread_10"])) != n_subset:
        raise RuntimeError("Subset indices are not unique")
    expected_topk = set(scores.sort_values(["score_value", "original_index"], ascending=[True, True]).head(n_subset)["original_index"].astype(int))
    if set(subsets["topk_10"].astype(int).tolist()) != expected_topk:
        raise RuntimeError("Top-K is not the exact minimum-score subset")
    paths["filtering"].mkdir(parents=True, exist_ok=True)
    for name, arr in subsets.items():
        np.save(paths["filtering"] / f"{name}_indices.npy", arr)
    write_json(paths["filtering"] / "subset_indices.json", {k: [int(x) for x in v.tolist()] for k, v in subsets.items()}, dry_run=False)
    membership = scores[["original_index", "score_value", "score_rank"]].copy()
    membership["is_topk_10"] = membership["original_index"].isin(set(subsets["topk_10"].tolist()))
    membership["is_quantile_spread_10"] = membership["original_index"].isin(set(subsets["quantile_spread_10"].tolist()))
    membership.to_csv(paths["filtering"] / "subset_membership.csv", index=False)
    full_df = subset_frame(scores, subsets["full"])
    rows = []
    for name in ["full", "topk_10", "quantile_spread_10"]:
        df = subset_frame(scores, subsets[name])
        row = {"subset": name, "size": int(len(df))}
        for col in ["score_value", "target_psnr_db", "realized_psnr_db_postclip", "realized_snr_db_postclip", "sigma", "corruption_mse", "clipped_fraction"]:
            row[f"{col}_mean"] = float(df[col].mean())
            row[f"{col}_median"] = float(df[col].median())
        rows.append(row)
    summary = pd.DataFrame(rows)
    full = summary[summary["subset"] == "full"].iloc[0]
    for col in ["target_psnr_db_median", "realized_psnr_db_postclip_median", "realized_snr_db_postclip_median", "sigma_median"]:
        summary[f"delta_{col}_vs_full"] = summary[col] - float(full[col])
    summary.to_csv(paths["filtering"] / "filtering_summary.csv", index=False)
    lines = [
        "# Full50k Filtering Summary",
        "",
        "DDPM: epoch 18, report timestep 100, code index 99, MC K=16.",
        "Full CIFAR-10 train split: 50000 images. Top-K and Quantile-spread: 5000 images each.",
        "",
        dataframe_to_markdown(summary),
    ]
    write_text(paths["filtering"] / "filtering_summary.md", "\n".join(lines) + "\n", dry_run=False)
    print(f"filter_full={n_full}")
    print(f"filter_topk={len(subsets['topk_10'])}")
    print(f"filter_quantile={len(subsets['quantile_spread_10'])}")


def plot_full_subset(ax, full: pd.Series, subset: pd.Series, bins: np.ndarray, label: str, metric_label: str) -> None:
    full_arr = full.to_numpy(dtype=np.float64)
    sub_arr = subset.to_numpy(dtype=np.float64)
    ax.hist(full_arr, bins=bins, density=True, alpha=0.38, label=f"Full 50k n={len(full_arr)}")
    ax.hist(sub_arr, bins=bins, density=True, alpha=0.45, label=f"{label} n={len(sub_arr)}")
    fm = float(np.median(full_arr))
    sm = float(np.median(sub_arr))
    ax.axvline(fm, linestyle="--", linewidth=1.6)
    ax.axvline(sm, linestyle="-", linewidth=1.6)
    ax.set_title(f"{label}: {metric_label}\nFull med {fm:.2f}; subset med {sm:.2f}; delta {sm - fm:+.2f}")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)


def make_figures(cfg: dict[str, Any], paths: dict[str, Path], dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] would create filtering figures in {paths['figures']}")
        return
    scores = pd.read_csv(paths["scores"] / "scores_full50k.csv")
    subsets = {name: np.load(paths["filtering"] / f"{name}_indices.npy").astype(np.int64) for name in ["full", "topk_10", "quantile_spread_10"]}
    full = subset_frame(scores, subsets["full"])
    metrics = [("target_psnr_db", "target PSNR"), ("realized_psnr_db_postclip", "realized PSNR"), ("realized_snr_db_postclip", "realized SNR")]
    methods = [("topk_10", "Top-K 10%"), ("quantile_spread_10", "Quantile-spread 10%")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
    for r, (name, label) in enumerate(methods):
        sub = subset_frame(scores, subsets[name])
        for c, (metric, metric_label) in enumerate(metrics):
            bins = np.histogram_bin_edges(full[metric], bins=45)
            plot_full_subset(axes[r, c], full[metric], sub[metric], bins, label, metric_label)
            axes[r, c].set_xlabel(metric_label)
    paths["figures"].mkdir(parents=True, exist_ok=True)
    fig.savefig(paths["figures"] / "01_quality_shift_all_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for name, label in [("full", "Full 50k"), *methods]:
        df = subset_frame(scores, subsets[name])
        vals = np.sort(df["realized_psnr_db_postclip"].to_numpy(dtype=np.float64))
        ax.plot(vals, np.arange(1, len(vals) + 1) / len(vals), label=f"{label} n={len(vals)} med={np.median(vals):.2f}")
    ax.set_xlabel("realized PSNR, dB")
    ax.set_ylabel("ECDF")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(paths["figures"] / "02_realized_psnr_ecdf.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    selected_examples(cfg, paths, scores, subsets)
    print(f"figures_dir={paths['figures'].resolve()}")


def get_font(size: int, bold: bool = False):
    candidates = ["C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def tensor_to_image(tensor: torch.Tensor, size: int = 128) -> Image.Image:
    arr = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255).round().astype(np.uint8)).resize((size, size), Image.Resampling.NEAREST)


def selected_examples(cfg: dict[str, Any], paths: dict[str, Path], scores: pd.DataFrame, subsets: dict[str, np.ndarray]) -> None:
    old = read_old_full(cfg)
    dataset_cfg = load_dataset_config("cifar10")
    train_ds = build_torchvision_split(
        dataset_cfg=dataset_cfg,
        train=True,
        data_root=Path(str(dataset_cfg["paths"]["data_root"])),
        transform_profile="tensor",
        download=False,
    )
    corpus = torch.load(old["corpus"], map_location="cpu").to(torch.float32)
    manifest = pd.read_csv(old["manifest"])
    pos = {int(row.original_index): int(row.cache_pos) for row in manifest[["original_index", "cache_pos"]].itertuples(index=False)}
    topk_df = subset_frame(scores, subsets["topk_10"])
    rows = [
        ("lowest score", scores.nsmallest(1, "score_value").iloc[0]),
        ("highest score", scores.nlargest(1, "score_value").iloc[0]),
        ("cleanest inside Top-K", topk_df.nlargest(1, "realized_psnr_db_postclip").iloc[0]),
        ("noisiest inside Top-K", topk_df.nsmallest(1, "realized_psnr_db_postclip").iloc[0]),
    ]
    tile_w = 300
    img_size = 128
    canvas = Image.new("RGB", (80 + len(rows) * tile_w, 270), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = get_font(18, True)
    text_font = get_font(14)
    for i, (label, row) in enumerate(rows):
        x = 40 + i * tile_w
        original = int(row["original_index"])
        clean, _ = tensor_class_id(train_ds[original])
        corrupt = corpus[pos[original]]
        canvas.paste(tensor_to_image(clean, img_size), (x, 42))
        canvas.paste(tensor_to_image(corrupt, img_size), (x + img_size + 10, 42))
        draw.text((x, 12), label, fill=(0, 0, 0), font=title_font)
        draw.multiline_text(
            (x, 180),
            f"idx {original}\nscore {float(row['score_value']):.4f}\nPSNR {float(row['realized_psnr_db_postclip']):.2f}",
            fill=(0, 0, 0),
            font=text_font,
        )
    canvas.save(paths["figures"] / "03_selected_examples.png")


def load_model_state(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def train_variant(cfg: dict[str, Any], paths: dict[str, Path], variant: str, dry_run: bool, resume: bool, device_arg: str | None) -> None:
    label = "topk_10_e18_t100_full50k" if variant == "topk_10" else "quantile_10_e18_t100_full50k"
    branch = paths["downstream"] / label
    if dry_run:
        print(f"[dry-run] would train {label} for {cfg['downstream']['total_steps']} optimizer steps")
        return
    old = read_old_full(cfg)
    indices = np.load(paths["filtering"] / f"{variant}_indices.npy").astype(np.int64)
    branch.mkdir(parents=True, exist_ok=True)
    best_path = branch / "best.pt"
    last_path = branch / "last.pt"
    history_path = branch / "training_history.csv"
    if resume and best_path.exists() and last_path.exists():
        print(f"train_resumed_existing={label}")
        return
    device = resolve_device(device_arg)
    model = DnCNN_2D(in_channels=3).to(device)
    model.load_state_dict(torch.load(old["initial_state"], map_location=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["downstream"]["lr"]))
    criterion = nn.MSELoss()
    corpus = MaterializedCorpusDataset(old["corpus"], old["manifest"], original_indices=indices, normalized=False, return_original_index=True)
    train_ds = CorpusTaskNoiseDataset(
        corpus,
        dataset_name="cifar10",
        task_seed=int(cfg["task_seed"]),
        task_noise_version=str(cfg["task_noise"]["version"]),
        sigma_min=float(cfg["task_noise"]["sigma_min"]),
        sigma_max=float(cfg["task_noise"]["sigma_max"]),
    )
    loader = DataLoader(
        train_ds,
        batch_size=int(cfg["downstream"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        generator=torch.Generator().manual_seed(int(cfg["model_seed"])),
    )
    iterator = cycle_loader(loader)
    rows = []
    best_loss = float("inf")
    start = time.perf_counter()
    for step in tqdm(range(1, int(cfg["downstream"]["total_steps"]) + 1), desc=f"Training {label}", mininterval=10.0):
        x_in, x_tgt = next(iterator)
        x_in = x_in.to(device)
        x_tgt = x_tgt.to(device)
        pred = model(x_in)
        loss = criterion(pred, x_tgt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = float(loss.item())
        rows.append({"step": step, "train_loss": loss_value})
        if loss_value < best_loss:
            best_loss = loss_value
            torch.save(model.state_dict(), best_path)
        if step % 100 == 0 or step == int(cfg["downstream"]["total_steps"]):
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "step": step, "best_train_loss": best_loss}, last_path)
            pd.DataFrame(rows).to_csv(history_path, index=False)
    elapsed = time.perf_counter() - start
    meta = {
        "variant": label,
        "subset_key": variant,
        "subset_size": int(len(indices)),
        "batch_size": int(cfg["downstream"]["batch_size"]),
        "total_steps": int(cfg["downstream"]["total_steps"]),
        "effective_epochs": float(int(cfg["downstream"]["total_steps"]) * int(cfg["downstream"]["batch_size"]) / len(indices)),
        "wall_clock_seconds": float(elapsed),
        "best_train_loss": float(best_loss),
        "initial_state": str(old["initial_state"].resolve()),
        "best_checkpoint": str(best_path.resolve()),
        "comparison_mode": "equal_optimizer_steps",
    }
    write_json(branch / "run_meta.json", meta, dry_run=False)
    OmegaConf.save(config=OmegaConf.create({**meta, "task_noise": cfg["task_noise"], "eval_noise": cfg["eval_noise"]}), f=branch / "resolved_config.yaml")
    print(f"train_done={label}")
    print(f"checkpoint={best_path.resolve()}")


def cycle_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


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


def batch_metrics(pred: torch.Tensor, target: torch.Tensor, lpips_model) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    mse = ((pred - target) ** 2).flatten(start_dim=1).mean(dim=1)
    psnr = 10.0 * torch.log10(1.0 / (mse + EPS))
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    ssim_vals = np.asarray([skimage_ssim(target_np[i], pred_np[i], data_range=1.0, channel_axis=0) for i in range(pred_np.shape[0])], dtype=np.float64)
    if lpips_model is None:
        lpips_vals = np.full(pred_np.shape[0], np.nan, dtype=np.float64)
    else:
        with torch.inference_mode():
            lpips_vals = lpips_model(prepare_for_lpips(pred), prepare_for_lpips(target)).detach().flatten().cpu().numpy().astype(np.float64)
    return mse.detach().cpu().numpy(), psnr.detach().cpu().numpy(), ssim_vals, lpips_vals


def evaluate_model(variant: str, model_path: Path | None, loader: DataLoader, device: torch.device, lpips_model) -> list[dict[str, Any]]:
    model = None
    if model_path is not None:
        model = DnCNN_2D(in_channels=3).to(device)
        model.load_state_dict(load_model_state(model_path, device))
        model.eval()
    rows = []
    offset = 0
    with torch.inference_mode():
        for x_input, x_target in tqdm(loader, desc=f"Evaluating {variant}", mininterval=10.0):
            x_input = x_input.to(device)
            x_target = x_target.to(device)
            pred = x_input if model is None else model(x_input)
            mse, psnr, ssim_vals, lpips_vals = batch_metrics(pred, x_target, lpips_model)
            for idx in range(int(x_input.shape[0])):
                rows.append({"variant": variant, "sample_index": offset + idx, "MSE": float(mse[idx]), "PSNR": float(psnr[idx]), "SSIM": float(ssim_vals[idx]), "LPIPS": float(lpips_vals[idx])})
            offset += int(x_input.shape[0])
    return rows


def evaluate(cfg: dict[str, Any], paths: dict[str, Path], dry_run: bool, device_arg: str | None) -> None:
    if dry_run:
        print(f"[dry-run] would evaluate Full existing, Top-K, Quantile into {paths['evaluation']}")
        return
    old = read_old_full(cfg)
    dataset = EvalNoiseDataset(old["eval_input"], old["eval_target"])
    loader = DataLoader(dataset, batch_size=int(cfg["downstream"]["batch_size"]), shuffle=False, num_workers=0)
    device = resolve_device(device_arg)
    lpips_model = create_lpips_model(device, bool(cfg["downstream"].get("lpips_enabled", True)))
    rows = []
    rows.extend(evaluate_model("Full 50k", old["checkpoint"], loader, device, lpips_model))
    for label in ["topk_10_e18_t100_full50k", "quantile_10_e18_t100_full50k"]:
        rows.extend(evaluate_model(label, paths["downstream"] / label / "best.pt", loader, device, lpips_model))
    per = pd.DataFrame(rows)
    paths["evaluation"].mkdir(parents=True, exist_ok=True)
    per.to_csv(paths["evaluation"] / "per_image_metrics.csv", index=False)
    rows2 = []
    for variant, group in per.groupby("variant", sort=False):
        row = {"variant": variant, "num_images": int(len(group))}
        for metric in ["MSE", "PSNR", "SSIM", "LPIPS"]:
            row[metric] = float(group[metric].mean())
        rows2.append(row)
    comp = pd.DataFrame(rows2)
    full = comp[comp["variant"] == "Full 50k"].iloc[0]
    comp["delta_PSNR_vs_full"] = comp["PSNR"] - float(full["PSNR"])
    comp["delta_SSIM_vs_full"] = comp["SSIM"] - float(full["SSIM"])
    comp["delta_LPIPS_vs_full"] = comp["LPIPS"] - float(full["LPIPS"])
    comp.to_csv(paths["evaluation"] / "downstream_comparison.csv", index=False)
    write_text(paths["evaluation"] / "downstream_comparison.md", dataframe_to_markdown(comp) + "\n", dry_run=False)
    print(f"evaluation_output={paths['evaluation'].resolve()}")


def summary(cfg: dict[str, Any], paths: dict[str, Path], dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] would summarize into {paths['summary']}")
        return
    old = read_old_full(cfg)
    comp_path = paths["evaluation"] / "downstream_comparison.csv"
    eval_comp = pd.read_csv(comp_path) if comp_path.exists() else pd.DataFrame()
    rows = []
    old_train = old["train_summary"]
    old_metrics = old["metrics"]
    full_metrics = {
        "PSNR": float(old_metrics["psnr_mean"]),
        "SSIM": float(old_metrics["ssim_mean"]),
        "LPIPS": float(old_metrics["lpips_mean"]),
    }
    rows.append({
        "variant": "Full 50k",
        "train_size": 50000,
        "filter_epoch": np.nan,
        "filter_timestep": np.nan,
        "optimizer_steps": int(old_train["total_steps"]),
        "effective_epochs": float(old_train["effective_epochs"]),
        "train_time": float(old_train["wall_clock_train_seconds"]),
        **full_metrics,
        "delta_PSNR_vs_full": 0.0,
        "delta_SSIM_vs_full": 0.0,
        "delta_LPIPS_vs_full": 0.0,
        "comparison_status": "strict_baseline_reused",
    })
    if not eval_comp.empty:
        for label, train_size in [("topk_10_e18_t100_full50k", 5000), ("quantile_10_e18_t100_full50k", 5000)]:
            if label not in set(eval_comp["variant"]):
                continue
            erow = eval_comp[eval_comp["variant"] == label].iloc[0]
            meta = json.loads((paths["downstream"] / label / "run_meta.json").read_text(encoding="utf-8"))
            rows.append({
                "variant": label,
                "train_size": train_size,
                "filter_epoch": int(cfg["checkpoint_epoch"]),
                "filter_timestep": int(cfg["report_timestep"]),
                "optimizer_steps": int(meta["total_steps"]),
                "effective_epochs": float(meta["effective_epochs"]),
                "train_time": float(meta["wall_clock_seconds"]),
                "PSNR": float(erow["PSNR"]),
                "SSIM": float(erow["SSIM"]),
                "LPIPS": float(erow["LPIPS"]),
                "delta_PSNR_vs_full": float(erow["delta_PSNR_vs_full"]),
                "delta_SSIM_vs_full": float(erow["delta_SSIM_vs_full"]),
                "delta_LPIPS_vs_full": float(erow["delta_LPIPS_vs_full"]),
                "comparison_status": "strict_vs_reused_full50k_eval_cache",
            })
    for row in PREVIOUS_45K_ROWS:
        rows.append({
            "variant": row["variant"],
            "train_size": row["train_size"],
            "filter_epoch": 18 if row["variant"] != "Full 45k" else np.nan,
            "filter_timestep": 100 if row["variant"] != "Full 45k" else np.nan,
            "optimizer_steps": 7815,
            "effective_epochs": np.nan,
            "train_time": np.nan,
            "PSNR": row["PSNR"],
            "SSIM": row["SSIM"],
            "LPIPS": row["LPIPS"],
            "delta_PSNR_vs_full": row["PSNR"] - full_metrics["PSNR"],
            "delta_SSIM_vs_full": row["SSIM"] - full_metrics["SSIM"],
            "delta_LPIPS_vs_full": row["LPIPS"] - full_metrics["LPIPS"],
            "comparison_status": "previous_45k_experiment",
        })
    table = pd.DataFrame(rows)
    paths["summary"].mkdir(parents=True, exist_ok=True)
    table.to_csv(paths["summary"] / "final_downstream_table.csv", index=False)
    write_text(paths["summary"] / "final_downstream_table.md", dataframe_to_markdown(table) + "\n", dry_run=False)
    print(f"summary_table={paths['summary'] / 'final_downstream_table.csv'}")


def run(args: argparse.Namespace) -> None:
    if str(args.stage).startswith("drunet_"):
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_cifar_selected_drunet.py"),
            "--stage",
            str(args.stage),
        ]
        if args.resume:
            cmd.append("--resume")
        if args.dry_run:
            cmd.append("--dry-run")
        if args.device:
            cmd.extend(["--device", str(args.device)])
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        return

    cfg = load_config(args.config, args.output_root)
    output_root = Path(cfg["resolved_paths"]["output_root"])
    paths = dirs(output_root)
    ensure_dirs(paths, args.dry_run)
    stage_list = ["audit", "prepare", "score", "filter", "train_topk", "train_quantile", "evaluate", "summary"] if args.stage == "all" else [args.stage]
    if "audit" in stage_list:
        audit(cfg, paths, args.dry_run)
    if "prepare" in stage_list:
        prepare(cfg, paths, args.dry_run)
    if "score" in stage_list:
        score(cfg, paths, args.dry_run, args.resume, args.device)
    if "filter" in stage_list:
        filter_subsets(cfg, paths, args.dry_run)
        make_figures(cfg, paths, args.dry_run)
    if "train_topk" in stage_list:
        train_variant(cfg, paths, "topk_10", args.dry_run, args.resume, args.device)
    if "train_quantile" in stage_list:
        train_variant(cfg, paths, "quantile_spread_10", args.dry_run, args.resume, args.device)
    if "evaluate" in stage_list:
        evaluate(cfg, paths, args.dry_run, args.device)
    if "summary" in stage_list:
        summary(cfg, paths, args.dry_run)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
