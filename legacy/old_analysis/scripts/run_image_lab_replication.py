from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.DDPM_model import build_DDPM_model
from src.TDnCNN_image import DnCNN_2D, compute_lpips
from src.dataset_registry import build_torchvision_split, load_dataset_config
from src.evaluation.image_lab_replication_analysis import (
    aggregate_metric_rows,
    build_scores_and_subsets,
    mse_psnr_ssim,
    save_subset_indices,
    write_class_balance,
    write_quality_diagnostics,
    write_quality_examples,
    write_score_vs_quality,
)
from src.image_lab_replication_datasets import (
    CorpusTaskNoiseDataset,
    EvalNoiseDataset,
    MaterializedCorpusDataset,
    materialize_eval_noise_cache,
    materialize_variable_awgn_corpus,
    stable_int_seed,
    stratified_train_val_indices,
)


DATASET_DEFAULTS = {
    "mnist": {
        "run_name": "mnist_psnr10-40_s42",
        "autoencoder_checkpoint": "checkpoints/mnist/autoencoders/ae_noise_consistency_mnist/autoencoder_checkpoint.pt",
        "autoencoder_kind": "noise_consistency",
        "latent_noise_mode": "induced",
        "ddpm_base_dim": 16,
        "ddpm_deep": 3,
        "tdncnn_batch_size": 32,
        "ddpm_batch_size": 64,
        "ddpm_train_steps": 2000,
        "full_reference_epochs": 5,
        "lpips": False,
    },
    "cifar10": {
        "run_name": "cifar10_psnr10-40_s42",
        "autoencoder_checkpoint": "checkpoints/cifar10/autoencoders/noise_consistency_large_latent128/autoencoder_checkpoint.pt",
        "autoencoder_kind": "noise_consistency",
        "latent_noise_mode": "induced",
        "ddpm_base_dim": 16,
        "ddpm_deep": 3,
        "tdncnn_batch_size": 32,
        "ddpm_batch_size": 64,
        "ddpm_train_steps": 3000,
        "full_reference_epochs": 5,
        "lpips": True,
    },
}

STAGES = ["prepare", "ddpm", "filter", "downstream", "evaluate", "summary"]
SCORE_FORMULA = "score_value = sum(flatten(eps_pred) ** 2)"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    rows = []
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float):
                values.append("NA" if math.isnan(value) else f"{value:.6f}")
            else:
                values.append(str(value))
        rows.append(values)
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    header_line = "| " + " | ".join(header.ljust(width) for header, width in zip(headers, widths)) + " |"
    sep_line = "| " + " | ".join("-" * width for width in widths) + " |"
    body = ["| " + " | ".join(value.ljust(width) for value, width in zip(row, widths)) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *body])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled MNIST/CIFAR-10 image lab replication.")
    parser.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), required=True)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "experiments" / "image_lab_replication_v1")
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--include-topk5", action="store_true")
    parser.add_argument("--target-psnr-min-db", type=float, default=10.0)
    parser.add_argument("--target-psnr-max-db", type=float, default=40.0)
    parser.add_argument("--corpus-seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=4242)
    parser.add_argument("--task-seed", type=int, default=42)
    parser.add_argument("--corruption-version", default="variable_awgn_quality_v1")
    parser.add_argument("--eval-noise-version", default="eval_awgn_v1")
    parser.add_argument("--task-noise-version", default="tdncnn_task_awgn_v1")
    parser.add_argument("--task-sigma-min", type=float, default=0.1)
    parser.add_argument("--task-sigma-max", type=float, default=0.8)
    parser.add_argument("--eval-sigma-min", type=float, default=0.1)
    parser.add_argument("--eval-sigma-max", type=float, default=0.8)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--ddpm-train-steps", type=int, default=None)
    parser.add_argument("--ddpm-batch-size", type=int, default=None)
    parser.add_argument("--tdncnn-batch-size", type=int, default=None)
    parser.add_argument("--full-reference-epochs", type=int, default=None)
    parser.add_argument("--downstream-total-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr-ddpm", type=float, default=1e-4)
    parser.add_argument("--lr-tdncnn", type=float, default=1e-3)
    parser.add_argument("--quantile-min-points-per-bin", type=int, default=30)
    return parser.parse_args()


def selected_stage_names(stage: str, prepare_only: bool) -> list[str]:
    if prepare_only:
        return ["prepare"]
    if stage == "all":
        return list(STAGES)
    start = STAGES.index(stage)
    return STAGES[start:]


def device_info(device: torch.device) -> dict[str, Any]:
    payload = {"device": str(device), "gpu_name": None, "peak_cuda_memory": None}
    if device.type == "cuda":
        payload["gpu_name"] = torch.cuda.get_device_name(0)
        payload["peak_cuda_memory"] = int(torch.cuda.max_memory_allocated(device))
    return payload


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    defaults = dict(DATASET_DEFAULTS[args.dataset])
    if args.smoke:
        defaults.update(
            {
                "run_mode": "smoke",
                "max_train_samples": args.max_train_samples or 96,
                "max_test_samples": args.max_test_samples or 32,
                "ddpm_train_steps": args.ddpm_train_steps or 4,
                "ddpm_batch_size": args.ddpm_batch_size or 16,
                "tdncnn_batch_size": args.tdncnn_batch_size or 16,
                "full_reference_epochs": args.full_reference_epochs or 1,
                "downstream_total_steps": args.downstream_total_steps or 4,
            }
        )
    else:
        defaults.update(
            {
                "run_mode": "main",
                "max_train_samples": args.max_train_samples,
                "max_test_samples": args.max_test_samples,
                "ddpm_train_steps": args.ddpm_train_steps or defaults["ddpm_train_steps"],
                "ddpm_batch_size": args.ddpm_batch_size or defaults["ddpm_batch_size"],
                "tdncnn_batch_size": args.tdncnn_batch_size or defaults["tdncnn_batch_size"],
                "full_reference_epochs": args.full_reference_epochs or defaults["full_reference_epochs"],
                "downstream_total_steps": args.downstream_total_steps,
            }
        )

    return {
        "dataset": args.dataset,
        "run_name": defaults["run_name"],
        "run_mode": defaults["run_mode"],
        "output_root": str(args.output_root),
        "target_psnr_min_db": float(args.target_psnr_min_db),
        "target_psnr_max_db": float(args.target_psnr_max_db),
        "corruption_type": "variable_awgn_quality",
        "corruption_version": args.corruption_version,
        "seeds": {
            "split_seed": int(args.split_seed),
            "corpus_seed": int(args.corpus_seed),
            "subset_seed": int(args.subset_seed),
            "model_seed": int(args.model_seed),
            "eval_seed": int(args.eval_seed),
            "task_seed": int(args.task_seed),
        },
        "sample_limits": {
            "max_train_samples": defaults["max_train_samples"],
            "max_test_samples": defaults["max_test_samples"],
        },
        "task_noise": {
            "kind": "awgn_uniform_sigma",
            "sigma_min": float(args.task_sigma_min),
            "sigma_max": float(args.task_sigma_max),
            "version": args.task_noise_version,
        },
        "eval_noise": {
            "kind": "awgn_uniform_sigma",
            "sigma_min": float(args.eval_sigma_min),
            "sigma_max": float(args.eval_sigma_max),
            "version": args.eval_noise_version,
        },
        "ddpm": {
            "autoencoder_checkpoint": defaults["autoencoder_checkpoint"],
            "autoencoder_kind": defaults["autoencoder_kind"],
            "latent_noise_mode": defaults["latent_noise_mode"],
            "base_dim": int(defaults["ddpm_base_dim"]),
            "deep": int(defaults["ddpm_deep"]),
            "batch_size": int(defaults["ddpm_batch_size"]),
            "train_steps": int(defaults["ddpm_train_steps"]),
            "lr": float(args.lr_ddpm),
            "checkpoint_selection": "best_val_loss",
            "score_formula": SCORE_FORMULA,
            "score_timestep_report": 1,
            "score_timestep_index": 0,
        },
        "filtering": {
            "subset_seed": int(args.subset_seed),
            "keep_ratio": 0.10,
            "include_topk5": bool(args.include_topk5),
            "quantile_min_points_per_bin": int(args.quantile_min_points_per_bin),
        },
        "downstream": {
            "model": "TDnCNN",
            "comparison_mode": "equal_optimizer_steps",
            "batch_size": int(defaults["tdncnn_batch_size"]),
            "full_reference_epochs": int(defaults["full_reference_epochs"]),
            "total_steps_override": defaults["downstream_total_steps"],
            "lr": float(args.lr_tdncnn),
            "initial_state_seed": int(args.model_seed),
            "lpips_enabled": bool(defaults["lpips"]),
        },
        "num_workers": int(args.num_workers),
        "download": bool(args.download),
    }


def config_hash(config: dict[str, Any]) -> str:
    stable_config = dict(config)
    stable_config.pop("output_root", None)
    payload = json.dumps(stable_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def run_root(config: dict[str, Any]) -> Path:
    return Path(config["output_root"]) / str(config["run_name"]) / str(config["run_mode"])


def init_run(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    root = run_root(config)
    expected_hash = config_hash(config)
    if root.exists():
        hash_path = root / "config_hash"
        if not args.resume:
            raise FileExistsError(f"Run root exists: {root}. Pass --resume to continue.")
        if hash_path.exists():
            existing_hash = hash_path.read_text(encoding="utf-8").strip()
            if existing_hash != expected_hash:
                raise RuntimeError(
                    f"Run root config hash mismatch: existing={existing_hash}, requested={expected_hash}. "
                    f"Use a different output root or run name."
                )
    root.mkdir(parents=True, exist_ok=True)
    for dirname in ["00_protocol", "01_corpus", "02_ddpm", "03_filtering", "04_downstream", "05_evaluation", "06_summary"]:
        (root / dirname).mkdir(parents=True, exist_ok=True)
    Path(config["output_root"]).mkdir(parents=True, exist_ok=True)
    (Path(config["output_root"]) / "combined_summary").mkdir(parents=True, exist_ok=True)
    (Path(config["output_root"]) / "README.md").write_text(
        "# Image Lab Replication v1\n\nControlled MNIST/CIFAR-10 score-based filtering replication artifacts.\n",
        encoding="utf-8",
    )
    OmegaConf.save(config=OmegaConf.create(config), f=root / "00_protocol" / "resolved_config.yaml")
    OmegaConf.save(config=OmegaConf.create(config), f=Path(config["output_root"]) / "campaign_config.yaml")
    (root / "config_hash").write_text(expected_hash + "\n", encoding="utf-8")
    meta = {
        "project_root": str(PROJECT_ROOT),
        "run_root": str(root),
        "created_or_resumed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config_hash": expected_hash,
    }
    (root / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_status(root, "initialized", "ok")
    return root


def write_status(root: Path, stage: str, status: str, extra: dict[str, Any] | None = None) -> None:
    payload = {
        "stage": stage,
        "status": status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra:
        payload.update(extra)
    (root / "status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_clean_splits(config: dict[str, Any]):
    dataset_cfg = load_dataset_config(str(config["dataset"]), PROJECT_ROOT)
    data_root = Path(str(dataset_cfg.get("paths", {}).get("data_root", PROJECT_ROOT / "data")))
    train = build_torchvision_split(
        dataset_cfg=dataset_cfg,
        train=True,
        data_root=data_root,
        transform_profile="tensor",
        download=bool(config["download"]),
    )
    test = build_torchvision_split(
        dataset_cfg=dataset_cfg,
        train=False,
        data_root=data_root,
        transform_profile="tensor",
        download=bool(config["download"]),
    )
    return dataset_cfg, train, test


def stage_prepare(root: Path, config: dict[str, Any]) -> None:
    dataset_cfg, train_dataset, test_dataset = load_clean_splits(config)
    train_limit = config["sample_limits"]["max_train_samples"]
    test_limit = config["sample_limits"]["max_test_samples"]
    train_indices = np.arange(len(train_dataset), dtype=np.int64)
    test_indices = np.arange(len(test_dataset), dtype=np.int64)
    if train_limit is not None:
        train_indices = train_indices[: int(train_limit)]
    if test_limit is not None:
        test_indices = test_indices[: int(test_limit)]

    corpus_dir = root / "01_corpus"
    tensor_path, manifest_path = materialize_variable_awgn_corpus(
        dataset=train_dataset,
        dataset_name=str(config["dataset"]),
        split="train",
        original_indices=train_indices,
        output_dir=corpus_dir,
        target_psnr_min_db=float(config["target_psnr_min_db"]),
        target_psnr_max_db=float(config["target_psnr_max_db"]),
        corpus_seed=int(config["seeds"]["corpus_seed"]),
        corruption_version=str(config["corruption_version"]),
        overwrite=False,
    )
    eval_input_path, eval_target_path, test_manifest_path = materialize_eval_noise_cache(
        dataset=test_dataset,
        dataset_name=str(config["dataset"]),
        original_indices=test_indices,
        output_dir=corpus_dir,
        sigma_min=float(config["eval_noise"]["sigma_min"]),
        sigma_max=float(config["eval_noise"]["sigma_max"]),
        eval_seed=int(config["seeds"]["eval_seed"]),
        eval_noise_version=str(config["eval_noise"]["version"]),
        overwrite=False,
    )
    manifest = pd.read_csv(manifest_path)
    train_split, val_split = stratified_train_val_indices(
        manifest,
        val_fraction=0.10,
        split_seed=int(config["seeds"]["split_seed"]),
    )
    np.save(root / "00_protocol" / "ddpm_train_original_indices.npy", train_split)
    np.save(root / "00_protocol" / "ddpm_val_original_indices.npy", val_split)
    quality_summary = manifest[
        [
            "target_psnr_db",
            "realized_psnr_db_postclip",
            "realized_snr_db_postclip",
            "clipped_fraction",
        ]
    ].describe()
    quality_summary.to_csv(corpus_dir / "quality_summary.csv")
    write_quality_examples(
        clean_dataset=train_dataset,
        corpus_tensor_path=tensor_path,
        manifest_path=manifest_path,
        score_subset_df=None,
        output_dir=corpus_dir,
    )
    write_status(
        root,
        "prepare",
        "ok",
        {
            "train_corpus": str(tensor_path),
            "train_manifest": str(manifest_path),
            "test_eval_input": str(eval_input_path),
            "test_eval_target": str(eval_target_path),
            "test_noise_manifest": str(test_manifest_path),
            "train_samples": int(len(train_indices)),
            "test_samples": int(len(test_indices)),
        },
    )


def _cycle_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def stage_ddpm(root: Path, config: dict[str, Any]) -> None:
    output_dir = root / "02_ddpm"
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    metrics_path = output_dir / "DDPM_metrics.csv"
    if best_path.exists() and last_path.exists() and metrics_path.exists():
        write_status(root, "ddpm", "ok", {"skipped_existing": True})
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(config["seeds"]["model_seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config["seeds"]["model_seed"]))
        torch.cuda.reset_peak_memory_stats(device)

    dataset_cfg, _, _ = load_clean_splits(config)
    corpus = MaterializedCorpusDataset(
        root / "01_corpus" / "train_corpus.pt",
        root / "01_corpus" / "train_manifest.csv",
        normalized=True,
    )
    train_idx = np.load(root / "00_protocol" / "ddpm_train_original_indices.npy")
    val_idx = np.load(root / "00_protocol" / "ddpm_val_original_indices.npy")
    train_ds = MaterializedCorpusDataset(
        root / "01_corpus" / "train_corpus.pt",
        root / "01_corpus" / "train_manifest.csv",
        original_indices=train_idx,
        normalized=True,
    )
    val_ds = MaterializedCorpusDataset(
        root / "01_corpus" / "train_corpus.pt",
        root / "01_corpus" / "train_manifest.csv",
        original_indices=val_idx,
        normalized=True,
    )
    generator = torch.Generator().manual_seed(int(config["seeds"]["model_seed"]))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["ddpm"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        generator=generator,
    )
    val_loader = DataLoader(val_ds, batch_size=int(config["ddpm"]["batch_size"]), shuffle=False, num_workers=0)
    model = build_DDPM_model(
        base_dim=int(config["ddpm"]["base_dim"]),
        deep=int(config["ddpm"]["deep"]),
        device=str(device),
        latent_noise_mode=str(config["ddpm"]["latent_noise_mode"]),
        autoencoder_kind=str(config["ddpm"]["autoencoder_kind"]),
        autoencoder_checkpoint_path=PROJECT_ROOT / str(config["ddpm"]["autoencoder_checkpoint"]),
        dataset_cfg=dataset_cfg,
    )
    optimizer = torch.optim.Adam(model.model.parameters(), lr=float(config["ddpm"]["lr"]))
    criterion = nn.MSELoss()
    train_iter = _cycle_loader(train_loader)
    best_val_loss = float("inf")
    rows: list[dict[str, Any]] = []
    train_steps = int(config["ddpm"]["train_steps"])
    eval_interval = max(1, train_steps // 4)
    start = time.perf_counter()
    for step in range(1, train_steps + 1):
        batch = next(train_iter).to(device)
        loss = model.train_step(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        row = {"step": step, "train_loss": float(loss.item()), "val_loss": float("nan")}
        if step == train_steps or step % eval_interval == 0:
            model.model.eval()
            val_total = 0.0
            val_items = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = val_batch.to(device)
                    val_loss = model.train_step(val_batch)
                    val_total += float(val_loss.item()) * int(val_batch.shape[0])
                    val_items += int(val_batch.shape[0])
            val_loss_mean = val_total / max(val_items, 1)
            row["val_loss"] = val_loss_mean
            if val_loss_mean < best_val_loss:
                best_val_loss = val_loss_mean
                torch.save(
                    {
                        "model_state_dict": model.model.state_dict(),
                        "DDPM_params": {"base_dim": int(config["ddpm"]["base_dim"]), "deep": int(config["ddpm"]["deep"])},
                        "latent_noise_mode": str(config["ddpm"]["latent_noise_mode"]),
                        "autoencoder_kind": str(config["ddpm"]["autoencoder_kind"]),
                        "autoencoder_checkpoint_path": str(PROJECT_ROOT / str(config["ddpm"]["autoencoder_checkpoint"])),
                        "val_loss": best_val_loss,
                        "config": config,
                    },
                    best_path,
                )
            model.model.train()
        rows.append(row)
    torch.save(
        {
            "model_state_dict": model.model.state_dict(),
            "DDPM_params": {"base_dim": int(config["ddpm"]["base_dim"]), "deep": int(config["ddpm"]["deep"])},
            "latent_noise_mode": str(config["ddpm"]["latent_noise_mode"]),
            "autoencoder_kind": str(config["ddpm"]["autoencoder_kind"]),
            "autoencoder_checkpoint_path": str(PROJECT_ROOT / str(config["ddpm"]["autoencoder_checkpoint"])),
            "val_loss": best_val_loss,
            "config": config,
        },
        last_path,
    )
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    elapsed = time.perf_counter() - start
    (output_dir / "training_meta.json").write_text(
        json.dumps(
            {
                "train_steps": train_steps,
                "wall_clock_train_seconds": elapsed,
                "steps_per_second": train_steps / max(elapsed, 1e-9),
                **device_info(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _ = corpus
    write_status(root, "ddpm", "ok", {"best_path": str(best_path), "last_path": str(last_path)})


def sigma_to_t0_score(model, x: torch.Tensor, original_indices: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    t = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    z_0 = model._encode_to_latent(x)
    sigma = model._latent_noise_std(t)
    z_items = []
    for item, original_index in enumerate(original_indices.tolist()):
        seed = stable_int_seed(config["dataset"], "score_diffusion_noise", int(original_index), config["seeds"]["model_seed"], "t0")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        if model.latent_noise_mode == "baseline":
            eps_z = torch.randn(z_0[item : item + 1].shape, generator=generator, dtype=z_0.dtype).to(x.device)
            z_t = z_0[item : item + 1] + sigma[item : item + 1] * eps_z
        else:
            eps_x = torch.randn(x[item : item + 1].shape, generator=generator, dtype=x.dtype).to(x.device)
            x_noisy = x[item : item + 1] + sigma[item : item + 1] * eps_x
            z_t = model._encode_to_latent(x_noisy)
        z_items.append(z_t)
    z_batch = torch.cat(z_items, dim=0)
    eps_pred = model.model(z_batch, t)
    return eps_pred.flatten(start_dim=1).pow(2).sum(dim=1)


def stage_filter(root: Path, config: dict[str, Any]) -> None:
    filtering_dir = root / "03_filtering"
    scores_path = filtering_dir / "scores_and_subsets.csv"
    if scores_path.exists() and (filtering_dir / "filtering_meta.json").exists():
        write_status(root, "filter", "ok", {"skipped_existing": True})
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    dataset_cfg, train_clean, _ = load_clean_splits(config)
    checkpoint = torch.load(root / "02_ddpm" / "best.pt", map_location=device)
    model = build_DDPM_model(
        base_dim=int(config["ddpm"]["base_dim"]),
        deep=int(config["ddpm"]["deep"]),
        device=str(device),
        latent_noise_mode=str(checkpoint.get("latent_noise_mode", config["ddpm"]["latent_noise_mode"])),
        autoencoder_kind=str(checkpoint.get("autoencoder_kind", config["ddpm"]["autoencoder_kind"])),
        autoencoder_checkpoint_path=checkpoint.get(
            "autoencoder_checkpoint_path",
            str(PROJECT_ROOT / str(config["ddpm"]["autoencoder_checkpoint"])),
        ),
        dataset_cfg=dataset_cfg,
    )
    model.model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    corpus_ds = MaterializedCorpusDataset(
        root / "01_corpus" / "train_corpus.pt",
        root / "01_corpus" / "train_manifest.csv",
        normalized=True,
        return_original_index=True,
    )
    loader = DataLoader(corpus_ds, batch_size=int(config["ddpm"]["batch_size"]), shuffle=False, num_workers=0)
    rows: list[pd.DataFrame] = []
    start = time.perf_counter()
    with torch.no_grad():
        for batch, original_indices in loader:
            batch = batch.to(device)
            original_indices = original_indices.to(device)
            score = sigma_to_t0_score(model, batch, original_indices, config)
            rows.append(
                pd.DataFrame(
                    {
                        "original_index": original_indices.detach().cpu().numpy().astype(np.int64),
                        "score_value": score.detach().cpu().numpy().astype(np.float64),
                    }
                )
            )
    score_df = pd.concat(rows, ignore_index=True)
    manifest = pd.read_csv(root / "01_corpus" / "train_manifest.csv")
    merged = score_df.merge(
        manifest[
            [
                "original_index",
                "class_id",
                "target_psnr_db",
                "realized_psnr_db_postclip",
                "realized_snr_db_postclip",
            ]
        ],
        on="original_index",
        how="left",
    ).rename(
        columns={
            "realized_psnr_db_postclip": "realized_psnr_db",
            "realized_snr_db_postclip": "realized_snr_db",
        }
    )
    score_subset_df, subsets = build_scores_and_subsets(
        merged,
        subset_seed=int(config["seeds"]["subset_seed"]),
        include_topk5=bool(config["filtering"]["include_topk5"]),
        quantile_min_points_per_bin=int(config["filtering"]["quantile_min_points_per_bin"]),
    )
    score_subset_df.to_csv(scores_path, index=False)
    save_subset_indices(subsets, filtering_dir)
    write_quality_diagnostics(score_subset_df, subsets, filtering_dir)
    correlations = write_score_vs_quality(score_subset_df, filtering_dir)
    write_class_balance(score_subset_df, subsets, filtering_dir)
    write_quality_examples(
        clean_dataset=train_clean,
        corpus_tensor_path=root / "01_corpus" / "train_corpus.pt",
        manifest_path=root / "01_corpus" / "train_manifest.csv",
        score_subset_df=score_subset_df,
        output_dir=filtering_dir,
    )
    elapsed = time.perf_counter() - start
    (filtering_dir / "filtering_meta.json").write_text(
        json.dumps(
            {
                "score_formula": SCORE_FORMULA,
                "score_timestep_report": 1,
                "score_timestep_index": 0,
                "score_seconds": elapsed,
                "scores_per_second": len(score_subset_df) / max(elapsed, 1e-9),
                "subsets": {name: int(len(indices)) for name, indices in subsets.items()},
                "correlations": correlations,
                **device_info(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_status(root, "filter", "ok", {"scores_and_subsets": str(scores_path)})


def load_subsets(root: Path, include_topk5: bool) -> dict[str, np.ndarray]:
    df = pd.read_csv(root / "03_filtering" / "scores_and_subsets.csv")
    subsets = {"full": np.sort(df["original_index"].to_numpy(dtype=np.int64))}
    for name, col in (
        ("topk_10", "is_topk_10"),
        ("quantile_spread_10", "is_quantile_spread_10"),
        ("random_10", "is_random_10"),
        ("topk_5", "is_topk_5"),
    ):
        if name == "topk_5" and not include_topk5:
            continue
        if col in df:
            subsets[name] = np.sort(df[df[col].astype(bool)]["original_index"].to_numpy(dtype=np.int64))
    return subsets


def make_initial_tdncnn_state(root: Path, config: dict[str, Any], in_channels: int) -> Path:
    path = root / "04_downstream" / "initial_state_dict.pt"
    if path.exists():
        return path
    torch.manual_seed(int(config["seeds"]["model_seed"]))
    model = DnCNN_2D(in_channels=in_channels)
    torch.save(model.state_dict(), path)
    return path


def train_tdncnn_branch(
    *,
    root: Path,
    config: dict[str, Any],
    subset_name: str,
    original_indices: np.ndarray,
    total_steps: int,
    in_channels: int,
    initial_state_path: Path,
) -> dict[str, Any]:
    branch_dir = root / "04_downstream" / subset_name
    branch_dir.mkdir(parents=True, exist_ok=True)
    history_path = branch_dir / "training_history.csv"
    best_path = branch_dir / "best.pt"
    if history_path.exists() and best_path.exists():
        meta_path = branch_dir / "training_meta.json"
        if meta_path.exists():
            existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if "validation_loss" in existing_meta:
                return existing_meta

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(int(config["seeds"]["model_seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config["seeds"]["model_seed"]))
    corpus = MaterializedCorpusDataset(
        root / "01_corpus" / "train_corpus.pt",
        root / "01_corpus" / "train_manifest.csv",
        original_indices=original_indices,
        normalized=False,
        return_original_index=True,
    )
    train_dataset = CorpusTaskNoiseDataset(
        corpus,
        dataset_name=str(config["dataset"]),
        task_seed=int(config["seeds"]["task_seed"]),
        task_noise_version=str(config["task_noise"]["version"]),
        sigma_min=float(config["task_noise"]["sigma_min"]),
        sigma_max=float(config["task_noise"]["sigma_max"]),
    )
    generator = torch.Generator().manual_seed(int(config["seeds"]["model_seed"]))
    loader = DataLoader(
        train_dataset,
        batch_size=int(config["downstream"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        generator=generator,
    )
    iterator = _cycle_loader(loader)
    model = DnCNN_2D(in_channels=in_channels).to(device)
    model.load_state_dict(torch.load(initial_state_path, map_location=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["downstream"]["lr"]))
    criterion = nn.MSELoss()
    best_loss = float("inf")
    rows = []
    start = time.perf_counter()
    for step in range(1, total_steps + 1):
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
        rows.append({"step": step, "train_loss": loss_value})
    torch.save(model.state_dict(), branch_dir / "last.pt")
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    val_indices = np.load(root / "00_protocol" / "ddpm_val_original_indices.npy")
    val_corpus = MaterializedCorpusDataset(
        root / "01_corpus" / "train_corpus.pt",
        root / "01_corpus" / "train_manifest.csv",
        original_indices=val_indices,
        normalized=False,
        return_original_index=True,
    )
    val_dataset = CorpusTaskNoiseDataset(
        val_corpus,
        dataset_name=str(config["dataset"]),
        task_seed=int(config["seeds"]["task_seed"]),
        task_noise_version=str(config["task_noise"]["version"]) + "_validation",
        sigma_min=float(config["task_noise"]["sigma_min"]),
        sigma_max=float(config["task_noise"]["sigma_max"]),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["downstream"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    validation_loss_sum = 0.0
    validation_items = 0
    with torch.no_grad():
        for x_input, x_target in val_loader:
            x_input = x_input.to(device)
            x_target = x_target.to(device)
            val_loss = criterion(model(x_input), x_target)
            validation_loss_sum += float(val_loss.item()) * int(x_input.shape[0])
            validation_items += int(x_input.shape[0])
    validation_loss = validation_loss_sum / max(validation_items, 1)
    elapsed = time.perf_counter() - start
    history = pd.DataFrame(rows)
    history.to_csv(history_path, index=False)
    meta = {
        "training_subset": subset_name,
        "sample_count": int(len(original_indices)),
        "batch_size": int(config["downstream"]["batch_size"]),
        "total_steps": int(total_steps),
        "total_seen_samples": int(total_steps * int(config["downstream"]["batch_size"])),
        "effective_epochs": float(total_steps * int(config["downstream"]["batch_size"]) / max(len(original_indices), 1)),
        "wall_clock_train_seconds": elapsed,
        "steps_per_second": total_steps / max(elapsed, 1e-9),
        "best_train_loss": best_loss,
        "validation_loss": validation_loss,
        **device_info(device),
    }
    (branch_dir / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def stage_downstream(root: Path, config: dict[str, Any]) -> None:
    dataset_cfg, _, _ = load_clean_splits(config)
    in_channels = int(dataset_cfg.get("channels", 1))
    subsets = load_subsets(root, include_topk5=bool(config["filtering"]["include_topk5"]))
    n_full = len(subsets["full"])
    batch_size = int(config["downstream"]["batch_size"])
    if config["downstream"]["total_steps_override"] is None:
        total_steps = int(math.ceil(n_full / batch_size) * int(config["downstream"]["full_reference_epochs"]))
    else:
        total_steps = int(config["downstream"]["total_steps_override"])
    initial_state_path = make_initial_tdncnn_state(root, config, in_channels)
    metas = []
    for subset_name in subsets:
        metas.append(
            train_tdncnn_branch(
                root=root,
                config=config,
                subset_name=subset_name,
                original_indices=subsets[subset_name],
                total_steps=total_steps,
                in_channels=in_channels,
                initial_state_path=initial_state_path,
            )
        )
    pd.DataFrame(metas).to_csv(root / "04_downstream" / "downstream_training_summary.csv", index=False)
    write_status(root, "downstream", "ok", {"total_steps": total_steps})


def create_lpips_model_or_none(device: torch.device, enabled: bool):
    if not enabled:
        return None
    try:
        import lpips

        model = lpips.LPIPS(net="alex").to(device)
        model.eval()
        return model
    except Exception as error:
        print(f"LPIPS disabled: {error}")
        return None


def evaluate_predictions(
    *,
    model: nn.Module | None,
    loader: DataLoader,
    device: torch.device,
    subset_name: str,
    lpips_model,
) -> tuple[list[dict[str, Any]], torch.Tensor | None]:
    rows: list[dict[str, Any]] = []
    example_triplet = None
    sample_offset = 0
    for x_input, x_target in loader:
        x_input = x_input.to(device)
        x_target = x_target.to(device)
        if model is None:
            pred = x_input.clamp(0.0, 1.0)
        else:
            pred = model(x_input).clamp(0.0, 1.0)
        mse, psnr, ssim = mse_psnr_ssim(pred, x_target)
        lpips_values: list[float] = [float("nan")] * int(x_input.shape[0])
        if lpips_model is not None:
            for idx in range(int(x_input.shape[0])):
                lpips_values[idx] = compute_lpips(pred[idx : idx + 1], x_target[idx : idx + 1], lpips_model)
        for idx in range(int(x_input.shape[0])):
            rows.append(
                {
                    "training_subset": subset_name,
                    "sample_index": sample_offset + idx,
                    "mse": float(mse[idx]),
                    "psnr": float(psnr[idx]),
                    "ssim": float(ssim[idx]),
                    "lpips": float(lpips_values[idx]),
                }
            )
        if example_triplet is None:
            example_triplet = torch.stack([x_target[0].detach().cpu(), x_input[0].detach().cpu(), pred[0].detach().cpu()], dim=0)
        sample_offset += int(x_input.shape[0])
    return rows, example_triplet


def stage_evaluate(root: Path, config: dict[str, Any]) -> None:
    eval_dir = root / "05_evaluation"
    per_image_path = eval_dir / "per_image_metrics.csv"
    if per_image_path.exists() and (eval_dir / "evaluation_meta.json").exists():
        write_status(root, "evaluate", "ok", {"skipped_existing": True})
        return
    dataset_cfg, _, _ = load_clean_splits(config)
    in_channels = int(dataset_cfg.get("channels", 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    eval_dataset = EvalNoiseDataset(root / "01_corpus" / "test_eval_input.pt", root / "01_corpus" / "test_eval_target.pt")
    loader = DataLoader(eval_dataset, batch_size=int(config["downstream"]["batch_size"]), shuffle=False, num_workers=0)
    lpips_model = create_lpips_model_or_none(device, bool(config["downstream"]["lpips_enabled"]))
    rows: list[dict[str, Any]] = []
    examples: list[torch.Tensor] = []
    noisy_rows, noisy_example = evaluate_predictions(model=None, loader=loader, device=device, subset_name="noisy_input", lpips_model=lpips_model)
    rows.extend(noisy_rows)
    if noisy_example is not None:
        examples.append(noisy_example)

    subsets = load_subsets(root, include_topk5=bool(config["filtering"]["include_topk5"]))
    for subset_name in subsets:
        branch_dir = root / "04_downstream" / subset_name
        model = DnCNN_2D(in_channels=in_channels).to(device)
        model.load_state_dict(torch.load(branch_dir / "best.pt", map_location=device))
        model.eval()
        branch_rows, example = evaluate_predictions(
            model=model,
            loader=loader,
            device=device,
            subset_name=subset_name,
            lpips_model=lpips_model,
        )
        rows.extend(branch_rows)
        if example is not None:
            examples.append(example)
    per_image_df = pd.DataFrame(rows)
    per_image_df.to_csv(per_image_path, index=False)
    aggregate = aggregate_metric_rows(per_image_df)
    aggregate.to_csv(eval_dir / "evaluation_aggregates.csv", index=False)
    if examples:
        from torchvision.utils import make_grid, save_image

        save_image(make_grid(torch.cat(examples, dim=0).clamp(0.0, 1.0), nrow=3, padding=2), eval_dir / "evaluation_examples.png")
    test_manifest = pd.read_csv(root / "01_corpus" / "test_noise_manifest.csv")
    bottom_count = max(1, int(round(0.25 * len(test_manifest))))
    bottom_positions = set(test_manifest.sort_values("realized_snr_db_postclip").head(bottom_count)["cache_pos"].astype(int).tolist())
    hard_df = per_image_df[per_image_df["sample_index"].isin(bottom_positions)].copy()
    if not hard_df.empty:
        aggregate_metric_rows(hard_df).to_csv(eval_dir / "evaluation_aggregates_lowest_snr25.csv", index=False)
    (eval_dir / "evaluation_meta.json").write_text(
        json.dumps(
            {
                "eval_noise_cache": str(root / "01_corpus" / "test_eval_input.pt"),
                "eval_target_cache": str(root / "01_corpus" / "test_eval_target.pt"),
                "lpips_enabled": bool(lpips_model is not None),
                **device_info(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_status(root, "evaluate", "ok", {"per_image_metrics": str(per_image_path), **device_info(device)})


def stage_summary(root: Path, config: dict[str, Any]) -> None:
    summary_dir = root / "06_summary"
    eval_path = root / "05_evaluation" / "evaluation_aggregates.csv"
    score_path = root / "03_filtering" / "scores_and_subsets.csv"
    downstream_path = root / "04_downstream" / "downstream_training_summary.csv"
    eval_df = pd.read_csv(eval_path)
    score_df = pd.read_csv(score_path)
    downstream_df = pd.read_csv(downstream_path)
    rows = []
    for subset in ["noisy_input", "full", "topk_10", "quantile_spread_10", "random_10", "topk_5"]:
        if subset not in set(eval_df["training_subset"]):
            continue
        eval_row = eval_df[eval_df["training_subset"] == subset].iloc[0].to_dict()
        if subset == "noisy_input":
            rows.append(
                {
                    "Training subset": subset,
                    "Size": 0,
                    "SNR mean": float("nan"),
                    "SNR median": float("nan"),
                    "MSE": eval_row.get("mse_mean"),
                    "PSNR": eval_row.get("psnr_mean"),
                    "SSIM": eval_row.get("ssim_mean"),
                    "LPIPS": eval_row.get("lpips_mean"),
                    "Validation loss": float("nan"),
                    "Train time": 0.0,
                }
            )
            continue
        selected_col = None if subset == "full" else f"is_{subset}"
        selected_df = score_df if selected_col is None else score_df[score_df[selected_col].astype(bool)]
        train_meta = downstream_df[downstream_df["training_subset"] == subset].iloc[0].to_dict()
        rows.append(
            {
                "Training subset": subset,
                "Size": int(len(selected_df)),
                "SNR mean": float(selected_df["realized_snr_db"].mean()),
                "SNR median": float(selected_df["realized_snr_db"].median()),
                "MSE": eval_row.get("mse_mean"),
                "PSNR": eval_row.get("psnr_mean"),
                "SSIM": eval_row.get("ssim_mean"),
                "LPIPS": eval_row.get("lpips_mean"),
                "Validation loss": train_meta.get("validation_loss"),
                "Train time": train_meta.get("wall_clock_train_seconds"),
            }
        )
    table = pd.DataFrame(rows)
    full = table[table["Training subset"] == "full"].iloc[0]
    table["delta_PSNR_vs_full"] = table["PSNR"] - float(full["PSNR"])
    table["delta_SSIM_vs_full"] = table["SSIM"] - float(full["SSIM"])
    table["delta_MSE_vs_full"] = table["MSE"] - float(full["MSE"])
    table["delta_LPIPS_vs_full"] = table["LPIPS"] - float(full["LPIPS"])
    table.to_csv(summary_dir / "final_table.csv", index=False)
    best_row = table[table["Training subset"] != "noisy_input"].sort_values("PSNR", ascending=False).iloc[0]
    topk = table[table["Training subset"] == "topk_10"].iloc[0]
    quantile = table[table["Training subset"] == "quantile_spread_10"].iloc[0]
    random = table[table["Training subset"] == "random_10"].iloc[0]

    lines = [
        f"# Image Lab Replication Summary: {config['dataset']}",
        "",
        f"- Corpus corruption: variable_awgn_quality, target PSNR [{config['target_psnr_min_db']}, {config['target_psnr_max_db']}] dB.",
        f"- Score formula: `{SCORE_FORMULA}`.",
        "- Score timestep: report t=1, code index=0.",
        "- Top-K selects minimum score values.",
        "- Downstream protocol: x_input = clip(x_corpus + task_noise, 0, 1), x_target = x_corpus.",
        "- Evaluation protocol: x_test_input = clip(x_test_clean + eval_noise, 0, 1), x_test_target = x_test_clean.",
        "",
        dataframe_to_markdown(table),
        "",
        f"Best by PSNR: {best_row['Training subset']} ({float(best_row['PSNR']):.4f}).",
        f"Top-K vs Full: delta PSNR {float(topk['delta_PSNR_vs_full']):+.4f}, delta SSIM {float(topk['delta_SSIM_vs_full']):+.4f}.",
        f"Top-K vs Quantile-spread: delta PSNR {float(topk['PSNR'] - quantile['PSNR']):+.4f}, delta SSIM {float(topk['SSIM'] - quantile['SSIM']):+.4f}.",
        f"Top-K vs Random: delta PSNR {float(topk['PSNR'] - random['PSNR']):+.4f}, delta SSIM {float(topk['SSIM'] - random['SSIM']):+.4f}.",
    ]
    (summary_dir / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_status(root, "summary", "ok", {"summary": str(summary_dir / "run_summary.md")})


def main() -> None:
    args = parse_args()
    config = resolve_config(args)
    root = init_run(args, config)
    stages = selected_stage_names(args.stage, args.prepare_only)
    stage_fns = {
        "prepare": stage_prepare,
        "ddpm": stage_ddpm,
        "filter": stage_filter,
        "downstream": stage_downstream,
        "evaluate": stage_evaluate,
        "summary": stage_summary,
    }
    for stage in stages:
        print(f"stage={stage} root={root}")
        stage_fns[stage](root, config)
    print(f"run_root={root}")


if __name__ == "__main__":
    main()
