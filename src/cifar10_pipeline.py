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
import torch.nn.functional as F
from skimage.metrics import structural_similarity as skimage_ssim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import yaml

from src.DRUNet_image import build_drunet
from src.Restormer_image import build_restormer
from src.TDnCNN_image import align_lpips_scaling_layer, prepare_for_lpips
from src.corruption_datasets import EvalNoiseDataset, MaterializedCorpusDataset, stable_int_seed
from src.utils.current_protocol import (
    print_protocol_summary,
    should_log_progress,
    stage_logging,
    training_progress,
    validate_current_protocol,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# CIFAR-10 score validation implementation. Kept in this domain pipeline so
# MNIST, CIFAR-10, and ImageNet-100 expose the same top-level structure.
import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image

from src.ddpm.image_space import build_image_space_ddpm_model
from src.TDnCNN_image import DnCNN_2D
from src.dataset_registry import build_torchvision_split, load_dataset_config
from src.corruption_datasets import (
    CorpusTaskNoiseDataset,
    EvalNoiseDataset,
    MaterializedCorpusDataset,
    materialize_eval_noise_cache,
    stable_int_seed,
    tensor_class_id,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "cifar10" / "cifar10_001"
SCORE_FORMULA = "q_sum = sum(flatten(eps_pred(x_t,t)) ** 2); q_mean = mean(flatten(eps_pred(x_t,t)) ** 2)"
PSNR_LEVELS = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
REPORT_TIMESTEPS_T1000 = [1, 2, 5, 10, 20, 50, 100, 200, 500]
EPS = 1e-12


@dataclass(frozen=True)
class ValidationPaths:
    root: Path
    protocol: Path
    corpus: Path
    paired: Path
    ddpm: Path
    scan: Path
    selected: Path
    confirmation: Path
    preview: Path
    commands: Path


def parse_score_validation_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-10 current DDPM score validation.")
    parser.add_argument(
        "--stage",
        choices=["audit", "prepare", "train-ddpm", "scan", "confirm", "preview", "commands", "downstream", "all-validation"],
        default="all-validation",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--corpus-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--forward-seed", type=int, default=777)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--score-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--ddpm-steps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--coarse-mc-k", type=int, default=4)
    parser.add_argument("--fine-mc-k", type=int, default=16)
    parser.add_argument("--stability-groups", type=int, default=4)
    parser.add_argument("--stability-k", type=int, default=4)
    parser.add_argument("--top-candidates", type=int, default=3)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--target-psnr-min-db", type=float, default=10.0)
    parser.add_argument("--target-psnr-max-db", type=float, default=40.0)
    parser.add_argument("--base-dim", type=int, default=16)
    parser.add_argument("--deep", type=int, default=3)
    parser.add_argument("--topk-stability-jaccard10-min", type=float, default=0.5)
    parser.add_argument("--rank-stability-spearman-min", type=float, default=0.5)
    parser.add_argument(
        "--downstream-subset",
        choices=["full_45000", "topk_10", "quantile_spread_10", "random_10"],
        default=None,
    )
    parser.add_argument("--downstream-total-steps", type=int, default=None)
    parser.add_argument("--tdncnn-batch-size", type=int, default=32)
    parser.add_argument("--downstream-validation-interval-steps", type=int, default=500)
    parser.add_argument("--downstream-early-stopping-patience", type=int, default=10)
    parser.add_argument("--downstream-early-stopping-min-delta", type=float, default=1.0e-4)
    parser.add_argument("--downstream-early-stopping-warmup-steps", type=int, default=1000)
    return parser.parse_args()


def validation_paths(root: Path) -> ValidationPaths:
    return ValidationPaths(
        root=root,
        protocol=root / "00_protocol",
        corpus=root / "01_corpus",
        paired=root / "02_paired_banks",
        ddpm=root / "03_ddpm",
        scan=root / "04_timestep_scan",
        selected=root / "05_selected_configuration",
        confirmation=root / "06_confirmation",
        preview=root / "07_filtering_preview",
        commands=root / "08_downstream_commands",
    )


def ensure_dirs(paths: ValidationPaths) -> None:
    for path in paths.__dict__.values():
        path.mkdir(parents=True, exist_ok=True)


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int, device: torch.device | None = None) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is not None and device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def cfg_to_dict(args: argparse.Namespace, dataset_cfg: dict[str, Any]) -> dict[str, Any]:
    if args.smoke:
        split_sizes = {"ddpm_filter_train": 80, "score_tuning": 20, "score_confirmation": 20}
        epochs = min(int(args.epochs), 2)
        scan_timesteps = [1, 10]
    else:
        split_sizes = {"ddpm_filter_train": 45000, "score_tuning": 2500, "score_confirmation": 2500}
        epochs = int(args.epochs)
        scan_timesteps = REPORT_TIMESTEPS_T1000
    return {
        "dataset": "cifar10",
        "output_root": str(args.output_root),
        "smoke": bool(args.smoke),
        "split_sizes": split_sizes,
        "target_psnr_db": {
            "min": float(args.target_psnr_min_db),
            "max": float(args.target_psnr_max_db),
            "distribution": "Uniform(10, 40) dB",
        },
        "psnr_levels": PSNR_LEVELS,
        "seeds": {
            "split_seed": int(args.split_seed),
            "corpus_seed": int(args.corpus_seed),
            "model_seed": int(args.model_seed),
            "forward_seed": int(args.forward_seed),
            "bootstrap_seed": int(args.bootstrap_seed),
        },
        "ddpm": {
            "data_domain": "image_space",
            "prediction_type": "epsilon",
            "base_dim": int(args.base_dim),
            "deep": int(args.deep),
            "epochs": epochs,
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "grad_clip": float(args.grad_clip),
            "n_steps": int(args.ddpm_steps),
            "beta_schedule": "linear",
            "beta_start": float(args.beta_start),
            "beta_end": float(args.beta_end),
            "score_formula": SCORE_FORMULA,
        },
        "scan": {
            "report_timesteps_requested_for_T1000": scan_timesteps,
            "coarse_mc_k": int(args.coarse_mc_k),
            "fine_mc_k": int(args.fine_mc_k),
            "top_candidates": int(args.top_candidates),
            "stability_groups": int(args.stability_groups),
            "stability_k": int(args.stability_k),
            "bootstrap_iters": int(args.bootstrap_iters),
            "selection_direction": "lower score is cleaner; do not auto-invert",
            "constraints": {
                "bootstrap_ci_low_above_zero": True,
                "global_score_sigma_positive": True,
                "global_score_corruption_mse_positive": True,
                "global_score_realized_psnr_negative": True,
                "topk_median_realized_psnr_delta_positive": True,
                "all_classes_represented": True,
                "topk_stability_jaccard10_min": float(args.topk_stability_jaccard10_min),
                "rank_stability_spearman_min": float(args.rank_stability_spearman_min),
            },
        },
        "downstream_gate": {
            "tdncnn_runs_are_not_executed_by_validation": True,
            "requires_pass_marker": str(args.output_root / "06_confirmation" / "PASS.marker"),
            "exclude_validation_original_indices": True,
        },
        "dataset_cfg": dataset_cfg,
        "download": bool(args.download),
        "num_workers": int(args.num_workers),
    }


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    headers = [str(col) for col in df.columns]
    rows = []
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float):
                values.append("nan" if math.isnan(value) else f"{value:.6f}")
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


def load_cifar10_train(dataset_cfg: dict[str, Any], download: bool) -> Dataset:
    data_root = Path(str(dataset_cfg.get("paths", {}).get("data_root", PROJECT_ROOT / "data")))
    return build_torchvision_split(
        dataset_cfg=dataset_cfg,
        train=True,
        data_root=data_root,
        transform_profile="tensor",
        download=download,
    )


def class_labels(dataset: Dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        return np.asarray(getattr(dataset, "targets"), dtype=np.int64)
    labels = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        labels.append(int(sample[1]) if isinstance(sample, (tuple, list)) and len(sample) > 1 else -1)
    return np.asarray(labels, dtype=np.int64)


def stratified_three_way_split(labels: np.ndarray, split_sizes: dict[str, int], seed: int) -> dict[str, np.ndarray]:
    total_required = int(sum(split_sizes.values()))
    if len(labels) < total_required:
        scale = len(labels) / total_required
        scaled = {key: max(1, int(round(value * scale))) for key, value in split_sizes.items()}
        diff = len(labels) - sum(scaled.values())
        scaled["ddpm_filter_train"] += diff
        split_sizes = scaled
    classes = sorted(int(item) for item in np.unique(labels))
    rng = np.random.default_rng(seed)
    per_class_parts = {name: [] for name in split_sizes}
    remaining_by_class: dict[int, np.ndarray] = {}
    for cls in classes:
        idx = np.flatnonzero(labels == cls).astype(np.int64)
        rng.shuffle(idx)
        remaining_by_class[cls] = idx

    consumed = {cls: 0 for cls in classes}
    for split_name, split_size in split_sizes.items():
        quotas = []
        for cls in classes:
            cls_total = len(remaining_by_class[cls])
            quotas.append(split_size * cls_total / max(len(labels), 1))
        floors = np.floor(quotas).astype(np.int64)
        remainder = int(split_size - floors.sum())
        if remainder > 0:
            frac = np.asarray(quotas) - floors
            order = np.lexsort((np.asarray(classes), -frac))
            floors[order[:remainder]] += 1
        for cls, take in zip(classes, floors):
            start = consumed[cls]
            stop = min(start + int(take), len(remaining_by_class[cls]))
            per_class_parts[split_name].append(remaining_by_class[cls][start:stop])
            consumed[cls] = stop
    return {
        name: np.sort(np.concatenate(parts).astype(np.int64, copy=False))
        for name, parts in per_class_parts.items()
    }


def metric_row(x_clean: torch.Tensor, x_preclip: torch.Tensor, x_corrupted: torch.Tensor) -> dict[str, float]:
    pre_noise = x_preclip - x_clean
    post_noise = x_corrupted - x_clean
    signal_power = float((x_clean**2).mean().item())
    pre_mse = float((pre_noise**2).mean().item())
    post_mse = float((post_noise**2).mean().item())
    clipped_fraction = float(((x_preclip < 0.0) | (x_preclip > 1.0)).to(torch.float32).mean().item())
    return {
        "realized_psnr_db_preclip": 10.0 * math.log10(1.0 / max(pre_mse, EPS)),
        "realized_psnr_db_postclip": 10.0 * math.log10(1.0 / max(post_mse, EPS)),
        "realized_snr_db_preclip": 10.0 * math.log10(max(signal_power, EPS) / max(pre_mse, EPS)),
        "realized_snr_db_postclip": 10.0 * math.log10(max(signal_power, EPS) / max(post_mse, EPS)),
        "signal_power": signal_power,
        "corruption_mse": post_mse,
        "noise_power": pre_mse,
        "noise_power_postclip": post_mse,
        "clipped_fraction": clipped_fraction,
    }


def materialize_corpus(dataset: Dataset, original_indices: np.ndarray, output_dir: Path, config: dict[str, Any]) -> None:
    tensor_path = output_dir / "train_corpus.pt"
    manifest_path = output_dir / "train_manifest.csv"
    if tensor_path.exists() and manifest_path.exists():
        return
    rows = []
    tensors = []
    for cache_pos, original_index in enumerate(original_indices.tolist()):
        x_clean, class_id = tensor_class_id(dataset[int(original_index)])
        seed = stable_int_seed("cifar10_score_validation", "corpus", int(original_index), config["seeds"]["corpus_seed"])
        generator = torch.Generator().manual_seed(seed)
        u = torch.rand((), generator=generator).item()
        target_psnr = float(
            config["target_psnr_db"]["min"]
            + (config["target_psnr_db"]["max"] - config["target_psnr_db"]["min"]) * u
        )
        sigma = float(10.0 ** (-target_psnr / 20.0))
        noise = torch.randn(x_clean.shape, generator=generator, dtype=x_clean.dtype) * sigma
        x_preclip = x_clean + noise
        x_corrupted = x_preclip.clamp(0.0, 1.0).to(torch.float32)
        rows.append(
            {
                "dataset": "cifar10",
                "split": "train",
                "original_index": int(original_index),
                "cache_pos": int(cache_pos),
                "class_id": int(class_id),
                "target_psnr_db": target_psnr,
                "realized_psnr_db_postclip": 0.0,
                "realized_snr_db_postclip": 0.0,
                "sigma": sigma,
                "corruption_seed": int(seed),
                "corruption_type": "variable_awgn_quality",
                **metric_row(x_clean=x_clean, x_preclip=x_preclip, x_corrupted=x_corrupted),
            }
        )
        tensors.append(x_corrupted)
    torch.save(torch.stack(tensors, dim=0), tensor_path)
    pd.DataFrame(rows).to_csv(manifest_path, index=False)


def materialize_paired_bank(dataset: Dataset, split_name: str, original_indices: np.ndarray, output_dir: Path) -> None:
    tensor_path = output_dir / f"{split_name}_paired_bank.pt"
    manifest_path = output_dir / f"{split_name}_paired_manifest.csv"
    if tensor_path.exists() and manifest_path.exists():
        return
    rows = []
    bank_items = []
    for cache_pos, original_index in enumerate(original_indices.tolist()):
        x_clean, class_id = tensor_class_id(dataset[int(original_index)])
        seed = stable_int_seed("cifar10_score_validation", "paired_bank", split_name, int(original_index), 42)
        generator = torch.Generator().manual_seed(seed)
        base_noise = torch.randn(x_clean.shape, generator=generator, dtype=x_clean.dtype)
        levels = []
        for level_pos, psnr_db in enumerate(PSNR_LEVELS):
            sigma = float(10.0 ** (-float(psnr_db) / 20.0))
            x_preclip = x_clean + sigma * base_noise
            x_corrupted = x_preclip.clamp(0.0, 1.0).to(torch.float32)
            levels.append(x_corrupted)
            rows.append(
                {
                    "split": split_name,
                    "original_index": int(original_index),
                    "cache_pos": int(cache_pos),
                    "level_pos": int(level_pos),
                    "class_id": int(class_id),
                    "target_psnr_db": float(psnr_db),
                    "sigma": sigma,
                    "base_noise_seed": int(seed),
                    **metric_row(x_clean=x_clean, x_preclip=x_preclip, x_corrupted=x_corrupted),
                }
            )
        bank_items.append(torch.stack(levels, dim=0))
    torch.save(torch.stack(bank_items, dim=0), tensor_path)
    pd.DataFrame(rows).to_csv(manifest_path, index=False)


class PairedBankDataset(Dataset):
    def __init__(self, tensor_path: Path, manifest_path: Path) -> None:
        self.bank = torch.load(tensor_path, map_location="cpu").to(torch.float32)
        self.manifest = pd.read_csv(manifest_path)
        image_rows = self.manifest[["original_index", "cache_pos", "class_id"]].drop_duplicates("cache_pos")
        self.original_indices = image_rows.sort_values("cache_pos")["original_index"].to_numpy(dtype=np.int64)
        self.class_ids = image_rows.sort_values("cache_pos")["class_id"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return int(self.bank.shape[0])

    def __getitem__(self, index: int):
        idx = int(index)
        return self.bank[idx], int(self.original_indices[idx]), int(self.class_ids[idx])


def normalize_for_ddpm(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32) * 2.0 - 1.0


def build_validation_model(config: dict[str, Any], dataset_cfg: dict[str, Any], device: torch.device):
    model = build_image_space_ddpm_model(
        base_dim=int(config["ddpm"]["base_dim"]),
        deep=int(config["ddpm"]["deep"]),
        device=str(device),
        dataset_cfg=dataset_cfg,
        n_steps=int(config["ddpm"]["n_steps"]),
        beta_start=float(config["ddpm"]["beta_start"]),
        beta_end=float(config["ddpm"]["beta_end"]),
    )
    return model


def checkpoint_paths(ddpm_dir: Path) -> list[Path]:
    return sorted((ddpm_dir / "checkpoints").glob("epoch_*.pt"))


def assert_image_space_checkpoint(checkpoint: dict[str, Any], checkpoint_path: Path) -> None:
    legacy_keys = {
        "autoencoder_checkpoint_path",
        "autoencoder_kind",
        "decoder_checkpoint",
        "encoder_checkpoint",
        "latent_dim",
        "latent_noise_mode",
    }
    present = sorted(key for key in legacy_keys if key in checkpoint)
    if present:
        raise RuntimeError(
            f"Refusing to load latent/autoencoder DDPM checkpoint into image-space CIFAR DDPM: {checkpoint_path}. "
            f"Legacy metadata keys present: {present}."
        )
    config = checkpoint.get("config", {})
    if isinstance(config, dict):
        ddpm_cfg = config.get("ddpm", {})
        config_present = sorted(key for key in legacy_keys if isinstance(ddpm_cfg, dict) and key in ddpm_cfg)
        if config_present:
            raise RuntimeError(
                f"Refusing to load latent/autoencoder DDPM checkpoint into image-space CIFAR DDPM: {checkpoint_path}. "
                f"Legacy config keys present: {config_present}."
            )
    state = checkpoint.get("model_state_dict", {})
    first_conv_shape = None
    if isinstance(state, dict):
        for tensor in state.values():
            if hasattr(tensor, "shape") and len(tensor.shape) == 4:
                first_conv_shape = tuple(int(dim) for dim in tensor.shape)
                break
    if first_conv_shape is None:
        raise RuntimeError(f"Could not find first convolution tensor in DDPM checkpoint: {checkpoint_path}")
    if first_conv_shape[1] != 3:
        raise RuntimeError(
            f"Refusing to load non-image-space CIFAR DDPM checkpoint: {checkpoint_path}. "
            f"First convolution input channels must be 3, got shape {first_conv_shape}."
        )


def load_checkpoint_into_model(model, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=model.device)
    assert_image_space_checkpoint(checkpoint, checkpoint_path)
    model.model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint


def report_timestep_grid(n_steps: int, requested: list[int]) -> list[dict[str, int]]:
    rows = []
    for report_t in requested:
        if n_steps == 1000:
            code_index = int(report_t) - 1
        else:
            rel = (int(report_t) - 1) / 999.0
            code_index = int(round(rel * (n_steps - 1)))
        code_index = int(np.clip(code_index, 0, n_steps - 1))
        rows.append({"report_timestep": int(report_t), "code_index": code_index})
    dedup = {(row["report_timestep"], row["code_index"]): row for row in rows}
    return list(dedup.values())


@torch.no_grad()
def score_batch(
    model,
    x_norm: torch.Tensor,
    original_indices: np.ndarray,
    code_index: int,
    mc_k: int,
    forward_seed: int,
    seed_group: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device(model.device)
    x_norm = x_norm.to(device)
    batch_size = int(x_norm.shape[0])
    t = torch.full((batch_size,), int(code_index), dtype=torch.long, device=device)
    q_sum_acc = torch.zeros(batch_size, device=device)
    q_mean_acc = torch.zeros(batch_size, device=device)
    for k_idx in range(int(mc_k)):
        x_t_items = []
        for item, original_index in enumerate(original_indices.tolist()):
            seed = stable_int_seed(
                "cifar10_score_validation",
                "forward_noise",
                int(original_index),
                int(code_index),
                int(k_idx),
                int(seed_group),
                int(forward_seed),
            )
            generator = torch.Generator(device="cpu").manual_seed(seed)
            eps = torch.randn(x_norm[item : item + 1].shape, generator=generator, dtype=x_norm.dtype).to(device)
            x_t = model.q_sample(x_norm[item : item + 1], t[item : item + 1], eps)
            x_t_items.append(x_t)
        x_t_batch = torch.cat(x_t_items, dim=0)
        eps_pred = model.model(x_t_batch, t)
        flat = eps_pred.flatten(start_dim=1).pow(2)
        q_sum_acc += flat.sum(dim=1)
        q_mean_acc += flat.mean(dim=1)
    q_sum = (q_sum_acc / float(mc_k)).detach().cpu().numpy().astype(np.float64)
    q_mean = (q_mean_acc / float(mc_k)).detach().cpu().numpy().astype(np.float64)
    return q_sum, q_mean


def score_paired_bank(
    model,
    bank_path: Path,
    manifest_path: Path,
    code_index: int,
    report_timestep: int,
    mc_k: int,
    batch_size: int,
    forward_seed: int,
    seed_group: int = 0,
) -> pd.DataFrame:
    ds = PairedBankDataset(bank_path, manifest_path)
    loader = DataLoader(ds, batch_size=max(1, batch_size // len(PSNR_LEVELS)), shuffle=False, num_workers=0)
    rows = []
    level_count = len(PSNR_LEVELS)
    manifest = pd.read_csv(manifest_path)
    for bank_batch, original_indices, class_ids in loader:
        n_images = int(bank_batch.shape[0])
        flat = bank_batch.view(n_images * level_count, *bank_batch.shape[2:])
        repeated_original = np.repeat(original_indices.numpy().astype(np.int64), level_count)
        x_norm = normalize_for_ddpm(flat)
        q_sum, q_mean = score_batch(
            model,
            x_norm,
            repeated_original + np.tile(np.arange(level_count, dtype=np.int64) * 10_000_000, n_images),
            code_index=code_index,
            mc_k=mc_k,
            forward_seed=forward_seed,
            seed_group=seed_group,
        )
        rows.append(
            pd.DataFrame(
                {
                    "original_index": repeated_original,
                    "class_id": np.repeat(class_ids.numpy().astype(np.int64), level_count),
                    "level_pos": np.tile(np.arange(level_count, dtype=np.int64), n_images),
                    "score_q_sum": q_sum,
                    "score_q_mean": q_mean,
                    "report_timestep": int(report_timestep),
                    "code_index": int(code_index),
                    "mc_k": int(mc_k),
                }
            )
        )
    score_df = pd.concat(rows, ignore_index=True)
    return score_df.merge(
        manifest[
            [
                "original_index",
                "level_pos",
                "target_psnr_db",
                "sigma",
                "realized_psnr_db_postclip",
                "realized_snr_db_postclip",
                "corruption_mse",
                "clipped_fraction",
                "signal_power",
            ]
        ],
        on=["original_index", "level_pos"],
        how="left",
    )


def score_global_split(
    model,
    corpus_dir: Path,
    original_indices: np.ndarray,
    code_index: int,
    report_timestep: int,
    mc_k: int,
    batch_size: int,
    forward_seed: int,
    seed_group: int = 0,
) -> pd.DataFrame:
    ds = MaterializedCorpusDataset(
        corpus_dir / "train_corpus.pt",
        corpus_dir / "train_manifest.csv",
        original_indices=original_indices,
        normalized=True,
        return_original_index=True,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    rows = []
    for x_norm, batch_original in loader:
        original_np = batch_original.numpy().astype(np.int64)
        q_sum, q_mean = score_batch(
            model,
            x_norm,
            original_np,
            code_index=code_index,
            mc_k=mc_k,
            forward_seed=forward_seed,
            seed_group=seed_group,
        )
        rows.append(
            pd.DataFrame(
                {
                    "original_index": original_np,
                    "score_q_sum": q_sum,
                    "score_q_mean": q_mean,
                    "report_timestep": int(report_timestep),
                    "code_index": int(code_index),
                    "mc_k": int(mc_k),
                }
            )
        )
    score_df = pd.concat(rows, ignore_index=True)
    manifest = pd.read_csv(corpus_dir / "train_manifest.csv")
    return score_df.merge(manifest, on="original_index", how="left")


def corr(a: pd.Series, b: pd.Series, method: str) -> float:
    value = a.corr(b, method=method)
    return float(value) if pd.notna(value) else float("nan")


def bootstrap_ci(values: np.ndarray, fn: Callable[[np.ndarray], float], seed: int, iters: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(int(iters)):
        sample = values[rng.integers(0, len(values), size=len(values))]
        stats.append(float(fn(sample)))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def paired_metrics(score_df: pd.DataFrame, bootstrap_seed: int, bootstrap_iters: int) -> tuple[dict[str, float], pd.DataFrame]:
    rho_rows = []
    order_correct = 0
    order_total = 0
    for original_index, group in score_df.groupby("original_index", sort=False):
        group = group.sort_values("sigma")
        rho = corr(group["score_q_sum"], group["sigma"], "spearman")
        rho_rows.append(
            {
                "original_index": int(original_index),
                "class_id": int(group["class_id"].iloc[0]),
                "within_image_spearman": rho,
            }
        )
        values = group[["sigma", "score_q_sum"]].to_numpy(dtype=np.float64)
        for i in range(values.shape[0]):
            for j in range(values.shape[0]):
                if values[i, 0] > values[j, 0]:
                    order_correct += int(values[i, 1] > values[j, 1])
                    order_total += 1
    rho_df = pd.DataFrame(rho_rows)
    rho_values = rho_df["within_image_spearman"].to_numpy(dtype=np.float64)
    ci_low, ci_high = bootstrap_ci(rho_values, np.median, bootstrap_seed, bootstrap_iters)
    metrics = {
        "median_within_image_spearman": float(np.nanmedian(rho_values)),
        "mean_within_image_spearman": float(np.nanmean(rho_values)),
        "iqr_within_image_spearman": float(np.nanpercentile(rho_values, 75) - np.nanpercentile(rho_values, 25)),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "fraction_images_rho_gt_0": float(np.nanmean(rho_values > 0.0)),
        "pairwise_ordering_accuracy": float(order_correct / max(order_total, 1)),
    }
    class_rows = []
    for class_id, group in rho_df.groupby("class_id", sort=True):
        class_values = group["within_image_spearman"].to_numpy(dtype=np.float64)
        c_low, c_high = bootstrap_ci(class_values, np.median, bootstrap_seed + int(class_id), bootstrap_iters)
        class_rows.append(
            {
                "class_id": int(class_id),
                "median_within_image_spearman": float(np.nanmedian(class_values)),
                "mean_within_image_spearman": float(np.nanmean(class_values)),
                "iqr_within_image_spearman": float(
                    np.nanpercentile(class_values, 75) - np.nanpercentile(class_values, 25)
                ),
                "bootstrap_ci_low": c_low,
                "bootstrap_ci_high": c_high,
                "fraction_images_rho_gt_0": float(np.nanmean(class_values > 0.0)),
            }
        )
    return metrics, pd.DataFrame(class_rows)


def js_divergence_counts(full_counts: np.ndarray, selected_counts: np.ndarray) -> float:
    p = full_counts.astype(np.float64) / max(float(full_counts.sum()), 1.0)
    q = selected_counts.astype(np.float64) / max(float(selected_counts.sum()), 1.0)
    m = 0.5 * (p + q)
    return float(
        0.5 * np.sum(p * np.log((p + EPS) / (m + EPS)))
        + 0.5 * np.sum(q * np.log((q + EPS) / (m + EPS)))
    )


def global_metrics(score_df: pd.DataFrame) -> dict[str, Any]:
    df = score_df.copy()
    n_top = max(1, int(round(0.10 * len(df))))
    topk = df.sort_values(["score_q_sum", "original_index"], ascending=[True, True]).head(n_top)
    oracle = df.sort_values(["realized_psnr_db_postclip", "original_index"], ascending=[False, True]).head(n_top)
    classes = sorted(int(item) for item in df["class_id"].unique())
    full_counts = df["class_id"].value_counts().reindex(classes, fill_value=0).to_numpy()
    top_counts = topk["class_id"].value_counts().reindex(classes, fill_value=0).to_numpy()
    full_median_psnr = float(df["realized_psnr_db_postclip"].median())
    top_median_psnr = float(topk["realized_psnr_db_postclip"].median())
    metrics: dict[str, Any] = {
        "pearson_score_sigma": corr(df["score_q_sum"], df["sigma"], "pearson"),
        "spearman_score_sigma": corr(df["score_q_sum"], df["sigma"], "spearman"),
        "pearson_score_corruption_mse": corr(df["score_q_sum"], df["corruption_mse"], "pearson"),
        "spearman_score_corruption_mse": corr(df["score_q_sum"], df["corruption_mse"], "spearman"),
        "pearson_score_realized_psnr": corr(df["score_q_sum"], df["realized_psnr_db_postclip"], "pearson"),
        "spearman_score_realized_psnr": corr(df["score_q_sum"], df["realized_psnr_db_postclip"], "spearman"),
        "pearson_score_clipped_fraction": corr(df["score_q_sum"], df["clipped_fraction"], "pearson"),
        "spearman_score_clipped_fraction": corr(df["score_q_sum"], df["clipped_fraction"], "spearman"),
        "pearson_score_signal_power": corr(df["score_q_sum"], df["signal_power"], "pearson"),
        "spearman_score_signal_power": corr(df["score_q_sum"], df["signal_power"], "spearman"),
        "topk_10_count": int(n_top),
        "median_psnr_full": full_median_psnr,
        "median_psnr_topk": top_median_psnr,
        "delta_median_psnr_topk_minus_full": float(top_median_psnr - full_median_psnr),
        "mean_psnr_delta_topk_minus_full": float(
            topk["realized_psnr_db_postclip"].mean() - df["realized_psnr_db_postclip"].mean()
        ),
        "psnr_quantile_full_10": float(df["realized_psnr_db_postclip"].quantile(0.10)),
        "psnr_quantile_full_50": full_median_psnr,
        "psnr_quantile_full_90": float(df["realized_psnr_db_postclip"].quantile(0.90)),
        "psnr_quantile_topk_10": float(topk["realized_psnr_db_postclip"].quantile(0.10)),
        "psnr_quantile_topk_50": top_median_psnr,
        "psnr_quantile_topk_90": float(topk["realized_psnr_db_postclip"].quantile(0.90)),
        "median_sigma_delta_topk_minus_full": float(topk["sigma"].median() - df["sigma"].median()),
        "cleanest_decile_recall": float(
            len(set(topk["original_index"]).intersection(set(oracle["original_index"]))) / max(len(oracle), 1)
        ),
        "class_js_divergence": js_divergence_counts(full_counts, top_counts),
        "all_classes_represented": bool(np.all(top_counts > 0)),
        "topk_indices": topk["original_index"].astype(int).tolist(),
    }
    return metrics


def rank_positions(df: pd.DataFrame) -> pd.Series:
    ranked = df.sort_values(["score_q_sum", "original_index"], ascending=[True, True]).reset_index(drop=True)
    return pd.Series(np.arange(1, len(ranked) + 1, dtype=np.int64), index=ranked["original_index"].astype(int))


def stability_metrics(score_tables: list[pd.DataFrame]) -> dict[str, float]:
    if len(score_tables) < 2:
        return {
            "median_spearman_rank_correlation": float("nan"),
            "median_jaccard_topk10": float("nan"),
            "mean_rank_displacement": float("nan"),
        }
    base = score_tables[0]
    base_rank = rank_positions(base)
    n = len(base)
    n10 = max(1, int(round(0.10 * n)))
    base_top10 = set(base.nsmallest(n10, "score_q_sum")["original_index"].astype(int))
    spearman = []
    j10 = []
    disp = []
    for table in score_tables[1:]:
        rank = rank_positions(table)
        common = base_rank.index.intersection(rank.index)
        spearman.append(float(base_rank.loc[common].corr(rank.loc[common], method="spearman")))
        top10 = set(table.nsmallest(n10, "score_q_sum")["original_index"].astype(int))
        j10.append(len(base_top10 & top10) / max(len(base_top10 | top10), 1))
        disp.append(float(np.mean(np.abs(base_rank.loc[common].to_numpy() - rank.loc[common].to_numpy()))))
    return {
        "median_spearman_rank_correlation": float(np.nanmedian(spearman)),
        "median_jaccard_topk10": float(np.nanmedian(j10)),
        "mean_rank_displacement": float(np.nanmean(disp)),
    }


def write_protocol_audit(paths: ValidationPaths, config: dict[str, Any]) -> None:
    current_notes = [
        "# Protocol Audit",
        "",
        "## Current CIFAR image-space run",
        "- current branch does not use encoder, decoder, latent representation, latent DDPM, or autoencoder checkpoints.",
        "- DDPM implementation: `src/ddpm/image_space.py::ImageSpaceDDPM`, train target is `eps_pred` MSE against image-space forward noise.",
        "- filtering uses the checkpoint selected inside this run root only.",
        "- report timestep is one-based; code index is zero-based.",
        "- current `build_image_space_ddpm_model` uses explicit `n_steps`, linear beta schedule `1e-4 -> 0.02` by default.",
        "- current filtering score uses one deterministic forward-noise realization per original index at code index 0.",
        "- corruption normalization: materialized corpus is saved in `[0, 1]`; DDPM receives `x * 2 - 1`.",
        "- corruption clipping: AWGN corpus and paired bank use `clip(x_clean + sigma * noise, 0, 1)`.",
        "",
        "## New validation protocol",
        f"- output root: `{paths.root}`",
        f"- DDPM scheme: `{config['ddpm']['data_domain']}` + `{config['ddpm']['prediction_type']}` prediction.",
        f"- DDPM diffusion steps: `{config['ddpm']['n_steps']}`.",
        f"- beta schedule: linear `{config['ddpm']['beta_start']}` -> `{config['ddpm']['beta_end']}`.",
        f"- score formula: `{config['ddpm']['score_formula']}`.",
        "- CIFAR-10 test split is not used during DDPM validation.",
        "- score_tuning is used for selection; score_confirmation is used once after selection.",
        "",
        "If a future run changes image-space DDPM architecture or schedule, its metrics must remain separate from this validation.",
    ]
    (paths.protocol / "protocol_audit.md").write_text("\n".join(current_notes) + "\n", encoding="utf-8")


def stage_prepare(paths: ValidationPaths, config: dict[str, Any]) -> None:
    dataset_cfg = config["dataset_cfg"]
    dataset = load_cifar10_train(dataset_cfg, download=bool(config["download"]))
    labels = class_labels(dataset)
    splits = stratified_three_way_split(labels, config["split_sizes"], seed=int(config["seeds"]["split_seed"]))
    for name, indices in splits.items():
        np.save(paths.protocol / f"{name}_original_indices.npy", indices)
    split_rows = []
    for name, indices in splits.items():
        counts = pd.Series(labels[indices]).value_counts().sort_index()
        for class_id, count in counts.items():
            split_rows.append({"split": name, "class_id": int(class_id), "count": int(count)})
    pd.DataFrame(split_rows).to_csv(paths.protocol / "split_class_balance.csv", index=False)
    all_indices = np.sort(np.concatenate(list(splits.values())).astype(np.int64, copy=False))
    materialize_corpus(dataset, all_indices, paths.corpus, config)
    materialize_paired_bank(dataset, "score_tuning", splits["score_tuning"], paths.paired)
    materialize_paired_bank(dataset, "score_confirmation", splits["score_confirmation"], paths.paired)
    write_protocol_audit(paths, config)


def train_epoch(model, loader: DataLoader, optimizer, grad_clip: float, device: torch.device) -> float:
    model.model.train()
    total = 0.0
    count = 0
    for batch in loader:
        x = batch.to(device)
        loss = model.train_step(x)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.item()) * int(x.shape[0])
        count += int(x.shape[0])
    return total / max(count, 1)


@torch.no_grad()
def validate_loss(model, loader: DataLoader, device: torch.device) -> float:
    model.model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        x = batch.to(device)
        loss = model.train_step(x)
        total += float(loss.item()) * int(x.shape[0])
        count += int(x.shape[0])
    return total / max(count, 1)


@torch.no_grad()
def score_distribution_stats(model, loader: DataLoader, device: torch.device, max_batches: int = 8) -> dict[str, float]:
    model.model.eval()
    values = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x = batch.to(device)
        original = np.arange(batch_idx * x.shape[0], batch_idx * x.shape[0] + x.shape[0], dtype=np.int64)
        q_sum, _ = score_batch(
            model,
            x,
            original,
            code_index=0,
            mc_k=1,
            forward_seed=12345,
        )
        values.append(q_sum)
    if not values:
        return {"score_skewness": float("nan"), "score_excess_kurtosis": float("nan")}
    series = pd.Series(np.concatenate(values))
    return {"score_skewness": float(series.skew()), "score_excess_kurtosis": float(series.kurt())}


def stage_train_ddpm(paths: ValidationPaths, config: dict[str, Any], args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    set_seed(int(config["seeds"]["model_seed"]), device)
    dataset_cfg = config["dataset_cfg"]
    train_idx = np.load(paths.protocol / "ddpm_filter_train_original_indices.npy")
    tune_idx = np.load(paths.protocol / "score_tuning_original_indices.npy")
    train_ds = MaterializedCorpusDataset(paths.corpus / "train_corpus.pt", paths.corpus / "train_manifest.csv", original_indices=train_idx, normalized=True)
    val_ds = MaterializedCorpusDataset(paths.corpus / "train_corpus.pt", paths.corpus / "train_manifest.csv", original_indices=tune_idx, normalized=True)
    generator = torch.Generator().manual_seed(int(config["seeds"]["model_seed"]))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["ddpm"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        generator=generator,
    )
    val_loader = DataLoader(val_ds, batch_size=int(config["ddpm"]["batch_size"]), shuffle=False, num_workers=0)
    model = build_validation_model(config, dataset_cfg, device)
    optimizer = torch.optim.Adam(model.model.parameters(), lr=float(config["ddpm"]["lr"]))
    ckpt_dir = paths.ddpm / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = paths.ddpm / "ddpm_epoch_metrics.csv"
    rows = pd.read_csv(metrics_path).to_dict("records") if metrics_path.exists() else []
    existing = checkpoint_paths(paths.ddpm)
    start_epoch = 1
    if existing:
        last = existing[-1]
        checkpoint = torch.load(last, map_location=device)
        assert_image_space_checkpoint(checkpoint, last)
        model.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", len(existing))) + 1
    start = time.perf_counter()
    train_start = time.time()
    total_epochs = int(config["ddpm"]["epochs"])
    for epoch in range(start_epoch, int(config["ddpm"]["epochs"]) + 1):
        train_loss = train_epoch(model, train_loader, optimizer, float(config["ddpm"]["grad_clip"]), device)
        val_loss = validate_loss(model, val_loader, device)
        stats = score_distribution_stats(model, val_loader, device)
        corruption_config_hash = hashlib.sha256(
            json.dumps(config.get("target_psnr_db", {}), sort_keys=True).encode("utf-8")
        ).hexdigest()
        checkpoint = {
            "epoch": int(epoch),
            "domain": "image",
            "dataset": "cifar10",
            "input_channels": 3,
            "run_id": Path(str(config["output_root"])).name,
            "corruption_config_hash": corruption_config_hash,
            "model_state_dict": model.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "DDPM_params": {
                "data_domain": str(config["ddpm"]["data_domain"]),
                "prediction_type": str(config["ddpm"]["prediction_type"]),
                "base_dim": int(config["ddpm"]["base_dim"]),
                "deep": int(config["ddpm"]["deep"]),
                "n_steps": int(config["ddpm"]["n_steps"]),
                "beta_start": float(config["ddpm"]["beta_start"]),
                "beta_end": float(config["ddpm"]["beta_end"]),
            },
            "data_domain": str(config["ddpm"]["data_domain"]),
            "prediction_type": str(config["ddpm"]["prediction_type"]),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": config,
        }
        ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
        torch.save(checkpoint, ckpt_path)
        rows.append(
            {
                "epoch": int(epoch),
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "checkpoint_sha256": sha256_file(ckpt_path),
                "corruption_config_hash": corruption_config_hash,
                **stats,
            }
        )
        pd.DataFrame(rows).to_csv(metrics_path, index=False)
        training_progress(
            "CIFAR-10",
            "ddpm_train",
            unit="epoch",
            current=epoch,
            total=total_epochs,
            train_loss=train_loss,
            val_loss=val_loss,
            lr=optimizer.param_groups[0]["lr"],
            start_time=train_start,
            extra={"checkpoint": ckpt_path.name},
        )
    (paths.ddpm / "training_meta.json").write_text(
        json.dumps(
            {
                "epochs_requested": int(config["ddpm"]["epochs"]),
                "epochs_available": len(checkpoint_paths(paths.ddpm)),
                "seconds_this_invocation": time.perf_counter() - start,
                "device": str(device),
                "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def timestep_metadata(model, grid: list[dict[str, int]]) -> pd.DataFrame:
    rows = []
    for item in grid:
        idx = int(item["code_index"])
        alpha_bar = float(model.alphas_cumprod[idx].detach().cpu().item())
        rows.append(
            {
                "report_timestep": int(item["report_timestep"]),
                "code_index": idx,
                "beta_t": float(model.betas[idx].detach().cpu().item()),
                "alpha_bar_t": alpha_bar,
                "sqrt_one_minus_alpha_bar_t": float(math.sqrt(max(0.0, 1.0 - alpha_bar))),
            }
        )
    return pd.DataFrame(rows)


def evaluate_candidate(
    *,
    model,
    checkpoint_epoch: int,
    code_index: int,
    report_timestep: int,
    mc_k: int,
    split_name: str,
    paths: ValidationPaths,
    config: dict[str, Any],
    args: argparse.Namespace,
    seed_group: int = 0,
    write_scores: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    paired = score_paired_bank(
        model,
        paths.paired / f"{split_name}_paired_bank.pt",
        paths.paired / f"{split_name}_paired_manifest.csv",
        code_index=code_index,
        report_timestep=report_timestep,
        mc_k=mc_k,
        batch_size=int(args.score_batch_size),
        forward_seed=int(config["seeds"]["forward_seed"]),
        seed_group=seed_group,
    )
    split_idx = np.load(paths.protocol / f"{split_name}_original_indices.npy")
    global_df = score_global_split(
        model,
        paths.corpus,
        split_idx,
        code_index=code_index,
        report_timestep=report_timestep,
        mc_k=mc_k,
        batch_size=int(args.score_batch_size),
        forward_seed=int(config["seeds"]["forward_seed"]),
        seed_group=seed_group,
    )
    paired_summary, class_df = paired_metrics(
        paired,
        bootstrap_seed=int(config["seeds"]["bootstrap_seed"]),
        bootstrap_iters=int(config["scan"]["bootstrap_iters"]),
    )
    global_summary = global_metrics(global_df)
    metric_row_out = {
        "split": split_name,
        "epoch": int(checkpoint_epoch),
        "report_timestep": int(report_timestep),
        "code_index": int(code_index),
        "mc_k": int(mc_k),
        **{f"paired_{key}": value for key, value in paired_summary.items()},
        **{f"global_{key}": value for key, value in global_summary.items() if key != "topk_indices"},
    }
    if write_scores:
        out_dir = paths.scan if split_name == "score_tuning" else paths.confirmation
        stem = f"epoch_{checkpoint_epoch:03d}_t{report_timestep}_idx{code_index}_k{mc_k}"
        paired.to_csv(out_dir / f"{stem}_{split_name}_paired_scores.csv", index=False)
        global_df.to_csv(out_dir / f"{stem}_{split_name}_global_scores.csv", index=False)
        class_df.to_csv(out_dir / f"{stem}_{split_name}_paired_class_metrics.csv", index=False)
        np.save(out_dir / f"{stem}_{split_name}_topk_indices.npy", np.asarray(global_summary["topk_indices"], dtype=np.int64))
    return metric_row_out, global_df


def select_candidates(metrics_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    df = metrics_df.copy()
    constraints = config["scan"]["constraints"]
    df["passes_basic_constraints"] = (
        (df["paired_bootstrap_ci_low"] > 0.0)
        & (df["global_spearman_score_sigma"] > 0.0)
        & (df["global_spearman_score_corruption_mse"] > 0.0)
        & (df["global_spearman_score_realized_psnr"] < 0.0)
        & (df["global_delta_median_psnr_topk_minus_full"] > 0.0)
        & (df["global_all_classes_represented"].astype(bool))
    )
    sort_cols = [
        "passes_basic_constraints",
        "paired_median_within_image_spearman",
        "global_delta_median_psnr_topk_minus_full",
        "global_cleanest_decile_recall",
    ]
    return df.sort_values(sort_cols, ascending=[False, False, False, False]).head(int(config["scan"]["top_candidates"]))


def stage_scan(paths: ValidationPaths, config: dict[str, Any], args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    dataset_cfg = config["dataset_cfg"]
    model = build_validation_model(config, dataset_cfg, device)
    grid = report_timestep_grid(int(config["ddpm"]["n_steps"]), config["scan"]["report_timesteps_requested_for_T1000"])
    timestep_metadata(model, grid).to_csv(paths.scan / "timestep_metadata.csv", index=False)
    ckpts = checkpoint_paths(paths.ddpm)
    rows = []
    coarse_path = paths.scan / "coarse_score_validation_metrics.csv"
    if coarse_path.exists() and args.resume:
        rows = pd.read_csv(coarse_path).to_dict("records")
    done = {(int(r["epoch"]), int(r["code_index"]), int(r["mc_k"])) for r in rows}
    for ckpt_path in ckpts:
        checkpoint = load_checkpoint_into_model(model, ckpt_path)
        epoch = int(checkpoint["epoch"])
        for item in grid:
            key = (epoch, int(item["code_index"]), int(config["scan"]["coarse_mc_k"]))
            if key in done:
                continue
            metric_row_out, _ = evaluate_candidate(
                model=model,
                checkpoint_epoch=epoch,
                code_index=int(item["code_index"]),
                report_timestep=int(item["report_timestep"]),
                mc_k=int(config["scan"]["coarse_mc_k"]),
                split_name="score_tuning",
                paths=paths,
                config=config,
                args=args,
                write_scores=False,
            )
            rows.append(metric_row_out)
            pd.DataFrame(rows).to_csv(coarse_path, index=False)
    coarse_df = pd.DataFrame(rows)
    top = select_candidates(coarse_df, config)
    top.to_csv(paths.scan / "top_candidates_coarse.csv", index=False)
    fine_rows = []
    stability_rows = []
    for row in top.itertuples(index=False):
        ckpt_path = paths.ddpm / "checkpoints" / f"epoch_{int(row.epoch):03d}.pt"
        load_checkpoint_into_model(model, ckpt_path)
        metric_row_out, _ = evaluate_candidate(
            model=model,
            checkpoint_epoch=int(row.epoch),
            code_index=int(row.code_index),
            report_timestep=int(row.report_timestep),
            mc_k=int(config["scan"]["fine_mc_k"]),
            split_name="score_tuning",
            paths=paths,
            config=config,
            args=args,
            write_scores=True,
        )
        fine_rows.append(metric_row_out)
        group_tables = []
        tune_idx = np.load(paths.protocol / "score_tuning_original_indices.npy")
        for group_id in range(int(config["scan"]["stability_groups"])):
            group_tables.append(
                score_global_split(
                    model,
                    paths.corpus,
                    tune_idx,
                    code_index=int(row.code_index),
                    report_timestep=int(row.report_timestep),
                    mc_k=int(config["scan"]["stability_k"]),
                    batch_size=int(args.score_batch_size),
                    forward_seed=int(config["seeds"]["forward_seed"]),
                    seed_group=100 + group_id,
                )
            )
        stability = stability_metrics(group_tables)
        stability_rows.append(
            {
                "epoch": int(row.epoch),
                "report_timestep": int(row.report_timestep),
                "code_index": int(row.code_index),
                "mc_k": int(config["scan"]["stability_k"]),
                **stability,
            }
        )
    fine_df = pd.DataFrame(fine_rows)
    stability_df = pd.DataFrame(stability_rows)
    fine_df.to_csv(paths.scan / "fine_score_validation_metrics.csv", index=False)
    stability_df.to_csv(paths.scan / "score_stability_metrics.csv", index=False)
    merged = fine_df.merge(stability_df, on=["epoch", "report_timestep", "code_index"], how="left", suffixes=("", "_stability"))
    merged.to_csv(paths.scan / "score_validation_metrics.csv", index=False)
    write_scan_plots(paths, coarse_df, merged)
    write_tuning_summary(paths, merged, config)


def write_heatmap(df: pd.DataFrame, value_col: str, output_path: Path, title: str) -> None:
    if df.empty or value_col not in df:
        return
    pivot = df.pivot_table(index="epoch", columns="report_timestep", values=value_col, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.to_numpy(dtype=np.float64), aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(c) for c in pivot.index])
    ax.set_xlabel("report timestep")
    ax.set_ylabel("epoch")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_scan_plots(paths: ValidationPaths, coarse_df: pd.DataFrame, fine_df: pd.DataFrame) -> None:
    write_heatmap(
        coarse_df,
        "paired_median_within_image_spearman",
        paths.scan / "epoch_timestep_within_image_spearman_heatmap.png",
        "Median within-image Spearman",
    )
    write_heatmap(
        coarse_df,
        "global_spearman_score_sigma",
        paths.scan / "epoch_timestep_global_spearman_heatmap.png",
        "Global Spearman score vs sigma",
    )
    write_heatmap(
        coarse_df,
        "global_delta_median_psnr_topk_minus_full",
        paths.scan / "epoch_timestep_topk_psnr_shift_heatmap.png",
        "Top-K median PSNR shift",
    )
    write_heatmap(
        fine_df,
        "median_jaccard_topk10",
        paths.scan / "score_stability_heatmap.png",
        "Top-K 10 stability",
    )
    if not fine_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(fine_df["paired_median_within_image_spearman"], bins=10)
        axes[0].set_title("Best candidates Spearman")
        axes[1].hist(fine_df["global_delta_median_psnr_topk_minus_full"], bins=10)
        axes[1].set_title("Best candidates Top-K PSNR shift")
        fig.tight_layout()
        fig.savefig(paths.scan / "best_candidates_distributions.png", dpi=180)
        plt.close(fig)
    for required in ["paired_score_curves.png", "best_candidates_examples.png"]:
        placeholder = paths.scan / required
        if not placeholder.exists():
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, required.replace("_", " "), ha="center", va="center")
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(placeholder, dpi=120)
            plt.close(fig)


def write_tuning_summary(paths: ValidationPaths, metrics_df: pd.DataFrame, config: dict[str, Any]) -> None:
    candidates = select_candidates(metrics_df, config)
    if candidates.empty:
        return
    best = candidates.iloc[0].to_dict()
    ckpt_path = paths.ddpm / "checkpoints" / f"epoch_{int(best['epoch']):03d}.pt"
    require_path_under(ckpt_path, paths.root, "selected DDPM checkpoint")
    selected = {
        "selection_split": "score_tuning",
        "epoch": int(best["epoch"]),
        "report_timestep": int(best["report_timestep"]),
        "code_index": int(best["code_index"]),
        "mc_k": int(config["scan"]["fine_mc_k"]),
        "selected_ddpm_checkpoint_path": str(ckpt_path),
        "selected_ddpm_checkpoint_sha256": sha256_file(ckpt_path) if ckpt_path.exists() else None,
        "score_formula": SCORE_FORMULA,
        "direction": "lower q_sum is selected for Top-K; direction was not inverted",
        "tuning_metrics": {key: value for key, value in best.items() if key not in {"topk_indices"}},
    }
    OmegaConf.save(config=OmegaConf.create(selected), f=paths.selected / "selected_score_config.yaml")
    lines = [
        "# Score Validation Tuning Summary",
        "",
        f"Selected epoch: {selected['epoch']}",
        f"Selected report timestep/code index: {selected['report_timestep']} / {selected['code_index']}",
        f"Selected MC K: {selected['mc_k']}",
        "",
        dataframe_to_markdown(candidates.drop(columns=[col for col in ["global_topk_indices"] if col in candidates]).reset_index(drop=True)),
    ]
    (paths.scan / "score_validation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_selected(paths: ValidationPaths) -> dict[str, Any]:
    selected_path = paths.selected / "selected_score_config.yaml"
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing selected config: {selected_path}. Run --stage scan first.")
    return OmegaConf.to_container(OmegaConf.load(selected_path), resolve=True)


def stage_confirm(paths: ValidationPaths, config: dict[str, Any], args: argparse.Namespace) -> None:
    selected = load_selected(paths)
    device = resolve_device(args.device)
    model = build_validation_model(config, config["dataset_cfg"], device)
    ckpt_path = paths.ddpm / "checkpoints" / f"epoch_{int(selected['epoch']):03d}.pt"
    load_checkpoint_into_model(model, ckpt_path)
    confirm_metrics, confirm_scores = evaluate_candidate(
        model=model,
        checkpoint_epoch=int(selected["epoch"]),
        code_index=int(selected["code_index"]),
        report_timestep=int(selected["report_timestep"]),
        mc_k=int(selected["mc_k"]),
        split_name="score_confirmation",
        paths=paths,
        config=config,
        args=args,
        write_scores=True,
    )
    pd.DataFrame([confirm_metrics]).to_csv(paths.confirmation / "confirmation_metrics.csv", index=False)
    stability_tables = []
    confirm_idx = np.load(paths.protocol / "score_confirmation_original_indices.npy")
    for group_id in range(int(config["scan"]["stability_groups"])):
        stability_tables.append(
            score_global_split(
                model,
                paths.corpus,
                confirm_idx,
                code_index=int(selected["code_index"]),
                report_timestep=int(selected["report_timestep"]),
                mc_k=int(config["scan"]["stability_k"]),
                batch_size=int(args.score_batch_size),
                forward_seed=int(config["seeds"]["forward_seed"]),
                seed_group=500 + group_id,
            )
        )
    stability = stability_metrics(stability_tables)
    (paths.confirmation / "confirmation_stability.json").write_text(json.dumps(stability, indent=2), encoding="utf-8")
    constraints = config["scan"]["constraints"]
    passed = (
        confirm_metrics["paired_bootstrap_ci_low"] > 0.0
        and confirm_metrics["global_spearman_score_sigma"] > 0.0
        and confirm_metrics["global_spearman_score_corruption_mse"] > 0.0
        and confirm_metrics["global_spearman_score_realized_psnr"] < 0.0
        and confirm_metrics["global_delta_median_psnr_topk_minus_full"] > 0.0
        and bool(confirm_metrics["global_all_classes_represented"])
        and stability["median_jaccard_topk10"] >= float(constraints["topk_stability_jaccard10_min"])
        and stability["median_spearman_rank_correlation"] >= float(constraints["rank_stability_spearman_min"])
    )
    marker = paths.confirmation / ("PASS.marker" if passed else "FAIL.marker")
    marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n", encoding="utf-8")
    other = paths.confirmation / ("FAIL.marker" if passed else "PASS.marker")
    if other.exists():
        other.unlink()
    selected["confirmation_metrics"] = confirm_metrics
    selected["confirmation_stability"] = stability
    selected["status"] = "PASS" if passed else "FAIL"
    OmegaConf.save(config=OmegaConf.create(selected), f=paths.selected / "selected_score_config.yaml")


def summarize_subset(values: pd.Series, seed: int, iters: int) -> dict[str, float]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=np.float64)
    ci_low, ci_high = bootstrap_ci(arr, np.median, seed, iters)
    return {"median": float(np.median(arr)) if len(arr) else float("nan"), "ci_low": ci_low, "ci_high": ci_high}


def stage_preview(paths: ValidationPaths, config: dict[str, Any]) -> None:
    selected = load_selected(paths)
    stem = f"epoch_{int(selected['epoch']):03d}_t{int(selected['report_timestep'])}_idx{int(selected['code_index'])}_k{int(selected['mc_k'])}"
    candidates = sorted(paths.confirmation.glob(f"{stem}_score_confirmation_global_scores.csv"))
    if not candidates:
        candidates = sorted(paths.confirmation.glob("*_score_confirmation_global_scores.csv"))
    if not candidates:
        return
    df = pd.read_csv(candidates[-1])
    n = max(1, int(round(0.10 * len(df))))
    rng = np.random.default_rng(int(config["seeds"]["bootstrap_seed"]))
    subsets = {
        "full": df,
        "topk_10": df.nsmallest(n, "score_q_sum"),
        "quantile_spread_10": df.iloc[
            np.sort(np.concatenate([rng.choice(part, size=max(1, int(round(0.10 * len(part)))), replace=False) for part in np.array_split(np.arange(len(df)), 10)]))
        ],
        "random_10": df.iloc[np.sort(rng.choice(np.arange(len(df)), size=n, replace=False))],
    }
    rows = []
    for subset_name, subset_df in subsets.items():
        for metric in ["target_psnr_db", "realized_psnr_db_postclip", "sigma", "score_q_sum"]:
            stats = summarize_subset(subset_df[metric], int(config["seeds"]["bootstrap_seed"]), int(config["scan"]["bootstrap_iters"]))
            rows.append({"subset": subset_name, "metric": metric, **stats})
    pd.DataFrame(rows).to_csv(paths.preview / "filtering_preview_summary.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ["target_psnr_db", "realized_psnr_db_postclip", "sigma"]):
        for subset_name, subset_df in subsets.items():
            ax.hist(subset_df[metric], bins=30, alpha=0.35, density=True, label=subset_name)
        ax.set_title(metric)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(paths.preview / "full_vs_topk_quantile_random_distributions.png", dpi=180)
    plt.close(fig)
    write_preview_examples(paths, config, df)


def write_preview_examples(paths: ValidationPaths, config: dict[str, Any], df: pd.DataFrame) -> None:
    dataset = load_cifar10_train(config["dataset_cfg"], download=bool(config["download"]))
    manifest = pd.read_csv(paths.corpus / "train_manifest.csv")
    corpus = torch.load(paths.corpus / "train_corpus.pt", map_location="cpu").to(torch.float32)
    pos = {int(row.original_index): int(row.cache_pos) for row in manifest[["original_index", "cache_pos"]].itertuples(index=False)}
    groups = [
        ("lowest_score", df.nsmallest(16, "score_q_sum")),
        ("highest_score", df.nlargest(16, "score_q_sum")),
        ("oracle_cleanest", df.nlargest(16, "realized_psnr_db_postclip")),
        ("oracle_noisiest", df.nsmallest(16, "realized_psnr_db_postclip")),
    ]
    for name, group in groups:
        images = []
        for original_index in group["original_index"].astype(int).tolist():
            clean, _ = tensor_class_id(dataset[original_index])
            images.append(clean)
            images.append(corpus[pos[original_index]])
        if images:
            save_image(make_grid(torch.stack(images).clamp(0.0, 1.0), nrow=8, padding=2), paths.preview / f"{name}_panel.png")


def select_quantile_spread_from_scores(score_df: pd.DataFrame, n_subset: int, seed: int) -> np.ndarray:
    sorted_df = score_df.sort_values(["score_q_sum", "original_index"], ascending=[True, True]).reset_index(drop=True)
    num_bins = min(max(1, n_subset), max(1, len(sorted_df) // 30))
    bins = np.array_split(np.arange(len(sorted_df), dtype=np.int64), num_bins)
    quotas_float = np.asarray([n_subset * len(item) / max(len(sorted_df), 1) for item in bins])
    quotas = np.floor(quotas_float).astype(np.int64)
    remainder = int(n_subset - quotas.sum())
    if remainder > 0:
        order = np.lexsort((np.arange(len(quotas)), -(quotas_float - quotas)))
        quotas[order[:remainder]] += 1
    rng = np.random.default_rng(seed)
    selected_positions = []
    for bin_positions, quota in zip(bins, quotas):
        if quota > 0 and len(bin_positions) > 0:
            selected_positions.append(rng.choice(bin_positions, size=min(int(quota), len(bin_positions)), replace=False))
    if not selected_positions:
        return np.asarray([], dtype=np.int64)
    return np.sort(sorted_df.iloc[np.concatenate(selected_positions)]["original_index"].to_numpy(dtype=np.int64))


def make_downstream_initial_state(paths: ValidationPaths, config: dict[str, Any]) -> Path:
    output_path = paths.commands / "initial_tdncnn_state_dict.pt"
    if output_path.exists():
        return output_path
    torch.manual_seed(int(config["seeds"]["model_seed"]))
    model = DnCNN_2D(in_channels=3)
    torch.save(model.state_dict(), output_path)
    return output_path


def materialize_downstream_eval_cache(paths: ValidationPaths, config: dict[str, Any]) -> tuple[Path, Path, Path]:
    dataset_cfg = config["dataset_cfg"]
    data_root = Path(str(dataset_cfg.get("paths", {}).get("data_root", PROJECT_ROOT / "data")))
    test_dataset = build_torchvision_split(
        dataset_cfg=dataset_cfg,
        train=False,
        data_root=data_root,
        transform_profile="tensor",
        download=bool(config["download"]),
    )
    return materialize_eval_noise_cache(
        dataset=test_dataset,
        dataset_name="cifar10",
        original_indices=np.arange(len(test_dataset), dtype=np.int64),
        output_dir=paths.commands / "eval_noise_cache",
        sigma_min=0.1,
        sigma_max=0.8,
        eval_seed=4242,
        eval_noise_version="score_validation_downstream_eval_awgn_v1",
        overwrite=False,
    )


def stage_commands(paths: ValidationPaths, config: dict[str, Any], args: argparse.Namespace) -> None:
    selected = load_selected(paths) if (paths.selected / "selected_score_config.yaml").exists() else {}
    pass_marker = paths.confirmation / "PASS.marker"
    if not pass_marker.exists():
        fail_marker = paths.confirmation / "FAIL.marker"
        reason = f"Filtering is blocked until PASS marker exists: {pass_marker}"
        if fail_marker.exists():
            reason += f" (FAIL marker present: {fail_marker})"
        raise RuntimeError(reason)
    status = "PASS"
    train_idx = (
        np.load(paths.protocol / "ddpm_filter_train_original_indices.npy")
        if (paths.protocol / "ddpm_filter_train_original_indices.npy").exists()
        else np.asarray([], dtype=np.int64)
    )
    train_scores_path = paths.commands / "ddpm_filter_train_scores.csv"
    n = max(1, int(round(0.10 * len(train_idx))))
    ckpt_path = paths.ddpm / "checkpoints" / f"epoch_{int(selected['epoch']):03d}.pt"
    require_path_under(ckpt_path, paths.root, "filtering DDPM checkpoint")
    if "selected_ddpm_checkpoint_path" in selected:
        selected_path = Path(str(selected["selected_ddpm_checkpoint_path"]))
        if not selected_path.is_absolute():
            selected_path = PROJECT_ROOT / selected_path
        if selected_path.resolve() != ckpt_path.resolve():
            raise RuntimeError(f"Selected checkpoint path mismatch: {selected_path} != {ckpt_path}")
    if train_scores_path.exists():
        score_df = pd.read_csv(train_scores_path)
    else:
        device = resolve_device(args.device)
        model = build_validation_model(config, config["dataset_cfg"], device)
        load_checkpoint_into_model(model, ckpt_path)
        score_df = score_global_split(
            model,
            paths.corpus,
            train_idx,
            code_index=int(selected["code_index"]),
            report_timestep=int(selected["report_timestep"]),
            mc_k=int(selected["mc_k"]),
            batch_size=int(args.score_batch_size),
            forward_seed=int(config["seeds"]["forward_seed"]),
            seed_group=900,
        )
        score_df.to_csv(train_scores_path, index=False)
    topk = score_df.nsmallest(n, "score_q_sum")["original_index"].to_numpy(dtype=np.int64)
    quantile = select_quantile_spread_from_scores(score_df, n_subset=n, seed=42)
    initial_state_path = make_downstream_initial_state(paths, config)
    eval_input_path, eval_target_path, eval_manifest_path = materialize_downstream_eval_cache(paths, config)
    rng = np.random.default_rng(42)
    random_10 = np.sort(rng.choice(train_idx, size=min(n, len(train_idx)), replace=False)) if len(train_idx) else np.asarray([], dtype=np.int64)
    subset_paths = {
        "full_45000": paths.commands / "full_45000_indices.npy",
        "topk_10": paths.commands / "topk_10_indices.npy",
        "random_10": paths.commands / "random_10_indices.npy",
        "quantile_spread_10": paths.commands / "quantile_spread_10_indices.npy",
    }
    np.save(subset_paths["full_45000"], train_idx)
    np.save(subset_paths["topk_10"], np.sort(topk))
    np.save(subset_paths["random_10"], random_10)
    np.save(subset_paths["quantile_spread_10"], np.sort(quantile))
    subset_manifest = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path),
            "size": int(len(np.load(path))),
            "source_run_root": str(paths.root),
        }
        for name, path in subset_paths.items()
    }
    filtering_provenance = {
        "status": status,
        "selected_ddpm_checkpoint_path": str(ckpt_path),
        "selected_ddpm_checkpoint_sha256": sha256_file(ckpt_path),
        "checkpoint_epoch": int(selected["epoch"]),
        "report_timestep": int(selected["report_timestep"]),
        "code_timestep": int(selected["code_index"]),
        "mc_k": int(selected["mc_k"]),
        "score_formula": SCORE_FORMULA,
        "selected_score_config": str(paths.selected / "selected_score_config.yaml"),
        "selected_score_config_sha256": sha256_file(paths.selected / "selected_score_config.yaml"),
        "subsets": subset_manifest,
    }
    (paths.commands / "subset_manifest.json").write_text(json.dumps(subset_manifest, indent=2), encoding="utf-8")
    (paths.commands / "filtering_provenance.json").write_text(json.dumps(filtering_provenance, indent=2), encoding="utf-8")
    commands = [
        "# Downstream commands",
        "",
        f"Gate status: {status}",
        "",
        "These commands are intentionally separate from DDPM validation and should be run only after PASS.marker exists.",
        "",
        "```powershell",
        f"Test-Path \"{pass_marker}\"",
        f".\\.venv\\Scripts\\python.exe scripts/run_cifar10_pipeline.py --stage filter --resume --experiment-root \"{paths.root}\"",
        "# Then run downstream branches with the same trainer entrypoint, one branch per command:",
        f".\\.venv\\Scripts\\python.exe scripts/run_cifar10_pipeline.py --stage tdncnn_full --resume --experiment-root \"{paths.root}\"",
        f".\\.venv\\Scripts\\python.exe scripts/run_cifar10_pipeline.py --stage tdncnn_topk --resume --experiment-root \"{paths.root}\"",
        f".\\.venv\\Scripts\\python.exe scripts/run_cifar10_pipeline.py --stage tdncnn_quantile --resume --experiment-root \"{paths.root}\"",
        f".\\.venv\\Scripts\\python.exe scripts/run_cifar10_pipeline.py --stage tdncnn_random --resume --experiment-root \"{paths.root}\"",
        f"# full_45000 indices: {paths.commands / 'full_45000_indices.npy'}",
        f"# topk_10 indices: {paths.commands / 'topk_10_indices.npy'}",
        f"# quantile_spread_10 indices: {paths.commands / 'quantile_spread_10_indices.npy'}",
        f"# random_10 indices: {paths.commands / 'random_10_indices.npy'}",
        f"# shared initial state: {initial_state_path}",
        f"# shared eval input: {eval_input_path}",
        f"# shared eval target: {eval_target_path}",
        f"# shared eval manifest: {eval_manifest_path}",
        "```",
        "",
        "Required downstream invariants: equal optimizer steps, one initial state, one eval-noise cache, validation indices excluded.",
        f"Selected config: `{selected}`",
    ]
    (paths.commands / "downstream_commands.md").write_text("\n".join(commands) + "\n", encoding="utf-8")


def cycle_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def downstream_total_steps(args: argparse.Namespace, n_full: int) -> int:
    if args.downstream_total_steps is not None:
        return int(args.downstream_total_steps)
    return int(math.ceil(n_full / int(args.tdncnn_batch_size)) * 5)


def make_tdncnn_validation_dataset(paths: ValidationPaths, config: dict[str, Any]) -> CorpusTaskNoiseDataset:
    indices_path = paths.protocol / "score_tuning_original_indices.npy"
    if not indices_path.exists():
        raise FileNotFoundError(f"Missing downstream validation indices: {indices_path}")
    corpus = MaterializedCorpusDataset(
        paths.corpus / "train_corpus.pt",
        paths.corpus / "train_manifest.csv",
        original_indices=np.load(indices_path).astype(np.int64),
        normalized=False,
        return_original_index=True,
    )
    return CorpusTaskNoiseDataset(
        corpus,
        dataset_name="cifar10",
        task_seed=42,
        task_noise_version="score_validation_tdncnn_task_awgn_v1_validation",
        sigma_min=0.1,
        sigma_max=0.8,
    )


@torch.no_grad()
def validate_tdncnn_downstream_loss(model: nn.Module, paths: ValidationPaths, config: dict[str, Any], device: torch.device, batch_size: int) -> float:
    dataset = make_tdncnn_validation_dataset(paths, config)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.MSELoss(reduction="sum")
    model.eval()
    total = 0.0
    count = 0
    for x_input, x_target in loader:
        x_input = x_input.to(device)
        x_target = x_target.to(device)
        total += float(criterion(model(x_input), x_target).item())
        count += int(np.prod(x_target.shape))
    model.train()
    return total / max(count, 1)


@torch.no_grad()
def evaluate_tdncnn(model: nn.Module, paths: ValidationPaths, device: torch.device, batch_size: int) -> dict[str, float]:
    dataset = EvalNoiseDataset(paths.commands / "eval_noise_cache" / "test_eval_input.pt", paths.commands / "eval_noise_cache" / "test_eval_target.pt")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    mse_sum = 0.0
    count = 0
    for x_input, x_target in loader:
        x_input = x_input.to(device)
        x_target = x_target.to(device)
        pred = model(x_input).clamp(0.0, 1.0)
        mse = (pred - x_target).pow(2).flatten(start_dim=1).mean(dim=1)
        mse_sum += float(mse.sum().item())
        count += int(mse.numel())
    mse_mean = mse_sum / max(count, 1)
    return {"eval_mse": mse_mean, "eval_psnr": float(10.0 * math.log10(1.0 / max(mse_mean, EPS)))}


def topk_ablation_root(root: Path, model_name: str, variant: str) -> Path:
    return root / "10_topk_ablations" / model_name / variant


def make_augmented_tdncnn_dataset(paths: ValidationPaths, config: dict[str, Any], task_noise_version: str) -> CorpusTaskNoiseDataset:
    # TDnCNN's current dataset wrapper does not return sigma, so use the DRUNet
    # wrapper and drop sigma in the training loop.
    indices_path = paths.commands / "topk_10_indices.npy"
    require_path_under(indices_path, paths.root, "TDnCNN Top-K ablation indices")
    corpus = MaterializedCorpusDataset(
        paths.corpus / "train_corpus.pt",
        paths.corpus / "train_manifest.csv",
        original_indices=np.load(indices_path).astype(np.int64),
        normalized=False,
        return_original_index=True,
    )
    noise_cfg = deep_get(config, "downstream.task_noise", {})
    return CorpusTaskNoiseWithSigmaDataset(
        corpus,
        task_seed=int(deep_get(config, "seeds.task_noise_seed", 4242)),
        task_noise_version=task_noise_version,
        sigma_min=float(noise_cfg.get("sigma_min", 0.1)),
        sigma_max=float(noise_cfg.get("sigma_max", 0.8)),
        augmentation_seed=42,
        augment=True,
    )


def train_tdncnn_topk_ablation(cfg: dict[str, Any], args: argparse.Namespace, root: Path, *, variant: str, lr_multiplier: float) -> None:
    paths = validation_paths(root)
    pass_marker = paths.confirmation / "PASS.marker"
    if not pass_marker.exists():
        raise RuntimeError(f"TDnCNN Top-K ablation is blocked until PASS marker exists: {pass_marker}")
    output_dir = topk_ablation_root(root, "tdncnn", variant)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    metrics_path = output_dir / "metrics.json"
    if args.resume and metrics_path.exists() and best_path.exists():
        print(f"[SKIP] TDnCNN {variant} | completed metrics found")
        return
    full_indices = np.load(paths.commands / "full_45000_indices.npy").astype(np.int64)
    topk_indices = np.load(paths.commands / "topk_10_indices.npy").astype(np.int64)
    batch_size = int(deep_get(cfg, "downstream.tdncnn.batch_size", 32))
    total_steps = int(math.ceil(len(full_indices) / batch_size) * 5)
    device = resolve_device(args.device, args.smoke)
    set_seed(int(cfg["seeds"]["model_seed"]), device)
    task_noise_version = f"cifar10_topk_aug_{variant}_task_awgn_v1"
    dataset = make_augmented_tdncnn_dataset(paths, cfg, task_noise_version)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(deep_get(cfg, "runtime.num_workers", 0)),
        generator=torch.Generator().manual_seed(int(deep_get(cfg, "seeds.model_seed", 42))),
    )
    iterator = cycle_loader(loader)
    model = DnCNN_2D(in_channels=3).to(device)
    model.load_state_dict(torch.load(paths.commands / "initial_tdncnn_state_dict.pt", map_location=device), strict=True)
    lr = float(deep_get(cfg, "downstream.tdncnn.lr", 1e-3)) * float(lr_multiplier)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    early = deep_get(cfg, "downstream.tdncnn.early_stopping", {})
    val_interval = int(deep_get(cfg, "downstream.tdncnn.validation_interval_steps", 500))
    patience = int(early.get("patience", 10))
    min_delta = float(early.get("min_delta", 1e-4))
    warmup_steps = int(early.get("warmup_steps", 1000))
    best_val_loss = float("inf")
    best_step = 0
    checks_without_improvement = 0
    stop_reason = "max_steps"
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    log_start = time.time()
    for step in range(1, total_steps + 1):
        x_input, x_target, _sigma, _original_index = next(iterator)
        x_input = x_input.to(device)
        x_target = x_target.to(device)
        loss = criterion(model(x_input), x_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = float(loss.detach().cpu().item())
        val_loss = float("nan")
        if step == 1 or step == total_steps or step % val_interval == 0:
            val_loss = validate_tdncnn_downstream_loss(model, paths, cfg, device, batch_size)
            if step >= warmup_steps and val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_step = int(step)
                checks_without_improvement = 0
                torch.save(model.state_dict(), best_path)
            elif step >= warmup_steps:
                checks_without_improvement += 1
                if checks_without_improvement >= patience:
                    stop_reason = "early_stopping"
        rows.append({"step": step, "train_loss": train_loss, "val_loss": val_loss, "best_val_loss": best_val_loss, "best_step": best_step, "patience_counter": checks_without_improvement})
        if should_log_progress(step, total_steps, val_interval) or np.isfinite(val_loss):
            training_progress("CIFAR-10", f"tdncnn_{variant}", unit="step", current=step, total=total_steps, train_loss=train_loss, val_loss=val_loss, best_val_loss=best_val_loss, best_step=best_step, patience_counter=checks_without_improvement, patience=patience, lr=lr, start_time=log_start)
            pd.DataFrame(rows).to_csv(output_dir / "training_history.csv", index=False)
        if stop_reason == "early_stopping":
            break
    if not best_path.exists():
        torch.save(model.state_dict(), best_path)
    torch.save(model.state_dict(), output_dir / "last.pt")
    pd.DataFrame(rows).to_csv(output_dir / "training_history.csv", index=False)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    eval_metrics = evaluate_tdncnn(model, paths, device, batch_size)
    meta = {
        "model": "tdncnn",
        "variant": variant,
        "subset": "topk_10",
        "subset_size": int(len(topk_indices)),
        "augmentation": "RandomCrop32ReflectPad4+HorizontalFlip0.5",
        "lr_multiplier": float(lr_multiplier),
        "lr": float(lr),
        "optimizer_steps": int(step),
        "batch_size": int(batch_size),
        "best_step": int(best_step),
        "best_val_loss": float(best_val_loss),
        "stop_reason": stop_reason,
        "wall_clock_train_seconds": time.perf_counter() - start,
        "initial_state_sha256": sha256_file(paths.commands / "initial_tdncnn_state_dict.pt"),
        "subset_indices_sha256": sha256_file(paths.commands / "topk_10_indices.npy"),
        "best_checkpoint": str(best_path),
        **eval_metrics,
    }
    metrics_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def stage_downstream(paths: ValidationPaths, config: dict[str, Any], args: argparse.Namespace) -> None:
    pass_marker = paths.confirmation / "PASS.marker"
    if not pass_marker.exists():
        raise RuntimeError(f"TDnCNN downstream is blocked until PASS marker exists: {pass_marker}")
    if args.downstream_subset is None:
        raise ValueError("--downstream-subset is required for --stage downstream")
    stage_commands(paths, config, args)
    subset_path = paths.commands / f"{args.downstream_subset}_indices.npy"
    require_path_under(subset_path, paths.root, "TDnCNN subset indices")
    selected_config_path = paths.selected / "selected_score_config.yaml"
    corpus_manifest_path = paths.corpus / "train_manifest.csv"
    original_indices = np.load(subset_path).astype(np.int64)
    full_indices = np.load(paths.commands / "full_45000_indices.npy").astype(np.int64)
    total_steps = downstream_total_steps(args, n_full=len(full_indices))
    output_dir = paths.root / "09_downstream" / str(args.downstream_subset)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "training_history.csv"
    best_path = output_dir / "best.pt"
    if history_path.exists() and best_path.exists():
        return
    device = resolve_device(args.device)
    set_seed(int(config["seeds"]["model_seed"]), device)
    corpus = MaterializedCorpusDataset(
        paths.corpus / "train_corpus.pt",
        paths.corpus / "train_manifest.csv",
        original_indices=original_indices,
        normalized=False,
        return_original_index=True,
    )
    train_dataset = CorpusTaskNoiseDataset(
        corpus,
        dataset_name="cifar10",
        task_seed=42,
        task_noise_version="score_validation_tdncnn_task_awgn_v1",
        sigma_min=0.1,
        sigma_max=0.8,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=int(args.tdncnn_batch_size),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        generator=torch.Generator().manual_seed(int(config["seeds"]["model_seed"])),
    )
    iterator = cycle_loader(loader)
    model = DnCNN_2D(in_channels=3).to(device)
    model.load_state_dict(torch.load(paths.commands / "initial_tdncnn_state_dict.pt", map_location=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_step = 0
    checks_without_improvement = 0
    stop_reason = "max_steps"
    val_interval = max(1, int(args.downstream_validation_interval_steps))
    patience = max(1, int(args.downstream_early_stopping_patience))
    min_delta = max(0.0, float(args.downstream_early_stopping_min_delta))
    warmup_steps = max(0, int(args.downstream_early_stopping_warmup_steps))
    rows = []
    start = time.perf_counter()
    log_start = time.time()
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
        val_loss = float("nan")
        if step == 1 or step == total_steps or step % val_interval == 0:
            val_loss = validate_tdncnn_downstream_loss(model, paths, config, device, int(args.tdncnn_batch_size))
            if step >= warmup_steps and val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_step = int(step)
                checks_without_improvement = 0
                torch.save(model.state_dict(), best_path)
            elif step >= warmup_steps:
                checks_without_improvement += 1
                if checks_without_improvement >= patience:
                    stop_reason = "early_stopping"
        rows.append(
            {
                "step": step,
                "train_loss": loss_value,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "best_step": best_step,
                "patience_counter": checks_without_improvement,
            }
        )
        if should_log_progress(step, total_steps, val_interval) or np.isfinite(val_loss):
            training_progress(
                "CIFAR-10",
                f"tdncnn_{args.downstream_subset}",
                unit="step",
                current=step,
                total=total_steps,
                train_loss=loss_value,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                best_step=best_step,
                patience_counter=checks_without_improvement,
                patience=patience,
                lr=optimizer.param_groups[0]["lr"],
                start_time=log_start,
            )
            pd.DataFrame(rows).to_csv(history_path, index=False)
        if stop_reason == "early_stopping":
            break
    if not best_path.exists():
        torch.save(model.state_dict(), best_path)
    torch.save(model.state_dict(), output_dir / "last.pt")
    pd.DataFrame(rows).to_csv(history_path, index=False)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    eval_metrics = evaluate_tdncnn(model, paths, device, int(args.tdncnn_batch_size))
    meta = {
        "subset": str(args.downstream_subset),
        "subset_size": int(len(original_indices)),
        "total_steps": int(total_steps),
        "optimizer_steps": int(step),
        "batch_size": int(args.tdncnn_batch_size),
        "early_stopping_enabled": True,
        "early_stopping_monitor": "val_loss",
        "early_stopping_mode": "min",
        "early_stopping_patience": int(patience),
        "early_stopping_min_delta": float(min_delta),
        "early_stopping_warmup_steps": int(warmup_steps),
        "validation_interval_steps": int(val_interval),
        "best_step": int(best_step),
        "best_val_loss": float(best_val_loss),
        "stop_reason": stop_reason,
        "wall_clock_train_seconds": time.perf_counter() - start,
        "initial_state": str(paths.commands / "initial_tdncnn_state_dict.pt"),
        "initial_state_sha256": sha256_file(paths.commands / "initial_tdncnn_state_dict.pt"),
        "subset_indices": str(subset_path),
        "subset_indices_sha256": sha256_file(subset_path),
        "corpus_manifest": str(corpus_manifest_path),
        "corpus_manifest_sha256": sha256_file(corpus_manifest_path),
        "selected_score_config": str(selected_config_path),
        "selected_score_config_sha256": sha256_file(selected_config_path),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(output_dir / "last.pt"),
        "eval_noise_cache": str(paths.commands / "eval_noise_cache"),
        **eval_metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def write_run_config(paths: ValidationPaths, config: dict[str, Any]) -> None:
    OmegaConf.save(config=OmegaConf.create(config), f=paths.protocol / "resolved_score_validation_config.yaml")
    (paths.root / "README.md").write_text(
        "# CIFAR-10 Current Score Validation\n\n"
        "This root is separate from historical experiments. TDnCNN is gated by 06_confirmation/PASS.marker.\n",
        encoding="utf-8",
    )


def run_score_validation_stage(args: argparse.Namespace) -> None:
    dataset_cfg = load_dataset_config("cifar10", PROJECT_ROOT)
    paths = validation_paths(args.output_root)
    if paths.root.exists() and not args.resume and args.stage != "audit":
        raise FileExistsError(f"Output root exists: {paths.root}. Pass --resume to continue.")
    ensure_dirs(paths)
    config = cfg_to_dict(args, dataset_cfg)
    write_run_config(paths, config)
    stages = {
        "audit": [lambda: write_protocol_audit(paths, config)],
        "prepare": [lambda: stage_prepare(paths, config)],
        "train-ddpm": [lambda: stage_train_ddpm(paths, config, args)],
        "scan": [lambda: stage_scan(paths, config, args)],
        "confirm": [lambda: stage_confirm(paths, config, args)],
        "preview": [lambda: stage_preview(paths, config)],
        "commands": [lambda: stage_commands(paths, config, args)],
        "downstream": [lambda: stage_downstream(paths, config, args)],
        "all-validation": [
            lambda: stage_prepare(paths, config),
            lambda: stage_train_ddpm(paths, config, args),
            lambda: stage_scan(paths, config, args),
            lambda: stage_confirm(paths, config, args),
            lambda: stage_preview(paths, config),
            lambda: stage_commands(paths, config, args),
        ],
    }
    for fn in stages[args.stage]:
        fn()
    print(f"score_validation_root={paths.root}")


def score_validation_main() -> None:
    run_score_validation_stage(parse_score_validation_args())


STAGES = [
    "prepare",
    "ddpm_train",
    "score_validate",
    "score_select",
    "filter",
    "tdncnn_full",
    "tdncnn_topk",
    "tdncnn_quantile",
    "tdncnn_random",
    "tdncnn_evaluate",
    "drunet_full",
    "drunet_topk",
    "drunet_quantile",
    "drunet_random",
    "drunet_evaluate",
    "restormer_full",
    "restormer_topk",
    "restormer_quantile",
    "restormer_random",
    "restormer_evaluate",
    "tdncnn_topk_aug",
    "tdncnn_topk_aug_lr05",
    "drunet_topk_aug",
    "drunet_topk_aug_lr05",
    "topk_ablation_evaluate",
    "external_evaluate",
    "summary",
]

VALIDATION_STAGE_MAP = {
    "prepare": "prepare",
    "ddpm_train": "train-ddpm",
    "score_validate": "scan",
    "score_select": "confirm",
    "filter": "commands",
    "tdncnn_full": "downstream",
    "tdncnn_topk": "downstream",
    "tdncnn_quantile": "downstream",
    "tdncnn_random": "downstream",
}

TDNCNN_SUBSETS = {
    "tdncnn_full": "full_45000",
    "tdncnn_topk": "topk_10",
    "tdncnn_quantile": "quantile_spread_10",
    "tdncnn_random": "random_10",
}

DRUNET_SUBSETS = {
    "drunet_full": "full_45000",
    "drunet_topk": "topk_10",
    "drunet_quantile": "quantile_spread_10",
    "drunet_random": "random_10",
}
RESTORMER_SUBSETS = {
    "restormer_full": "full_45000",
    "restormer_topk": "topk_10",
    "restormer_quantile": "quantile_spread_10",
    "restormer_random": "random_10",
}

TOPK_ABLATION_STAGES = {
    "tdncnn_topk_aug": ("tdncnn", "topk_aug", 1.0),
    "tdncnn_topk_aug_lr05": ("tdncnn", "topk_aug_lr05", 0.5),
    "drunet_topk_aug": ("drunet", "topk_aug", 1.0),
    "drunet_topk_aug_lr05": ("drunet", "topk_aug_lr05", 0.5),
}

SUBSET_LABELS = {
    "full_45000": "Full",
    "topk_10": "Top-K",
    "quantile_spread_10": "Quantile",
    "random_10": "Random",
}

def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config must be a mapping: {path}")
    return data


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_path_under(path: Path, parent: Path, label: str) -> Path:
    resolved = path.resolve()
    root = parent.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{label} must be inside current run root: {resolved} not under {root}") from exc
    return resolved


def deep_get(data: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical CIFAR-10 current image-space pipeline.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "experiments" / "cifar10.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--experiment-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--stage", choices=[*STAGES, "all", "validation_all", "tdncnn_all", "drunet_all", "restormer_all", "topk_ablation_all"], default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def resolve_experiment_root(cfg: dict[str, Any], args: argparse.Namespace) -> Path:
    run_id = args.run_id or str(deep_get(cfg, "experiment.run_id", "cifar10_001"))
    if args.smoke and not run_id.endswith("_synthetic_smoke"):
        run_id = f"{run_id}_synthetic_smoke"
    if args.experiment_root is not None:
        return args.experiment_root
    root = Path(str(deep_get(cfg, "experiment.root", "experiments/cifar10")))
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root / run_id


def expanded_stages(stage: str) -> list[str]:
    if stage == "validation_all":
        return ["prepare", "ddpm_train", "score_validate", "score_select", "filter"]
    if stage == "tdncnn_all":
        return ["tdncnn_full", "tdncnn_topk", "tdncnn_quantile", "tdncnn_random", "tdncnn_evaluate"]
    if stage == "drunet_all":
        return ["drunet_full", "drunet_topk", "drunet_quantile", "drunet_random", "drunet_evaluate"]
    if stage == "restormer_all":
        return ["restormer_full", "restormer_topk", "restormer_quantile", "restormer_random", "restormer_evaluate"]
    if stage == "topk_ablation_all":
        return ["tdncnn_topk_aug", "tdncnn_topk_aug_lr05", "drunet_topk_aug", "drunet_topk_aug_lr05", "topk_ablation_evaluate"]
    if stage == "all":
        return STAGES.copy()
    return [stage]


def validation_args(cfg: dict[str, Any], args: argparse.Namespace, root: Path, stage: str) -> argparse.Namespace:
    argv = [
        "--stage",
        VALIDATION_STAGE_MAP[stage],
        "--output-root",
        str(root),
        "--epochs",
        str(int(deep_get(cfg, "ddpm.epochs", 40))),
        "--ddpm-steps",
        str(int(deep_get(cfg, "ddpm.T", 1000))),
        "--beta-start",
        str(float(deep_get(cfg, "ddpm.beta_start", 1e-4))),
        "--beta-end",
        str(float(deep_get(cfg, "ddpm.beta_end", 0.02))),
        "--base-dim",
        str(int(deep_get(cfg, "ddpm.base_dim", 16))),
        "--deep",
        str(int(deep_get(cfg, "ddpm.depth", 3))),
        "--target-psnr-min-db",
        str(float(deep_get(cfg, "corruption.target_psnr_min_db", 10.0))),
        "--target-psnr-max-db",
        str(float(deep_get(cfg, "corruption.target_psnr_max_db", 40.0))),
        "--split-seed",
        str(int(deep_get(cfg, "seeds.split_seed", 42))),
        "--corpus-seed",
        str(int(deep_get(cfg, "seeds.corpus_seed", 42))),
        "--model-seed",
        str(int(deep_get(cfg, "seeds.model_seed", 42))),
        "--forward-seed",
        str(int(deep_get(cfg, "seeds.forward_noise_seed", 777))),
        "--bootstrap-seed",
        str(int(deep_get(cfg, "seeds.bootstrap_seed", 2026))),
        "--batch-size",
        str(int(deep_get(cfg, "ddpm.batch_size", 128))),
        "--score-batch-size",
        str(int(deep_get(cfg, "score_validation.batch_size", 256))),
        "--num-workers",
        str(int(deep_get(cfg, "runtime.num_workers", 0))),
        "--coarse-mc-k",
        str(int(deep_get(cfg, "score_validation.coarse_mc_k", 4))),
        "--fine-mc-k",
        str(int(deep_get(cfg, "score_validation.final_mc_k", 16))),
        "--top-candidates",
        str(int(deep_get(cfg, "score_validation.final_candidate_count", 3))),
        "--bootstrap-iters",
        str(int(deep_get(cfg, "score_validation.bootstrap_iterations", 1000))),
        "--tdncnn-batch-size",
        str(int(deep_get(cfg, "smoke.downstream_batch_size", 4) if args.smoke else deep_get(cfg, "downstream.tdncnn.batch_size", 32))),
        "--downstream-validation-interval-steps",
        str(int(deep_get(cfg, "smoke.downstream_validation_interval_steps", 1) if args.smoke else deep_get(cfg, "downstream.tdncnn.validation_interval_steps", 500))),
        "--downstream-early-stopping-patience",
        str(int(deep_get(cfg, "smoke.downstream_early_stopping_patience", 2) if args.smoke else deep_get(cfg, "downstream.tdncnn.early_stopping.patience", 10))),
        "--downstream-early-stopping-min-delta",
        str(float(deep_get(cfg, "downstream.tdncnn.early_stopping.min_delta", 1e-4))),
        "--downstream-early-stopping-warmup-steps",
        str(int(deep_get(cfg, "smoke.downstream_early_stopping_warmup_steps", 0) if args.smoke else deep_get(cfg, "downstream.tdncnn.early_stopping.warmup_steps", 1000))),
    ]
    if args.resume:
        argv.append("--resume")
    if args.smoke:
        argv.append("--smoke")
    if args.device:
        argv.extend(["--device", args.device])
    if stage in TDNCNN_SUBSETS:
        argv.extend(["--downstream-subset", TDNCNN_SUBSETS[stage]])
        max_steps = deep_get(cfg, "smoke.downstream_max_optimizer_steps", None) if args.smoke else deep_get(cfg, "downstream.tdncnn.max_optimizer_steps", None)
        if max_steps is not None:
            argv.extend(["--downstream-total-steps", str(int(max_steps))])
    return parse_score_validation_args_from(argv)


def parse_score_validation_args_from(argv: list[str]) -> argparse.Namespace:
    old_argv = sys.argv
    try:
        sys.argv = ["cifar10_score_validation.py", *argv]
        return parse_score_validation_args()
    finally:
        sys.argv = old_argv


def write_tdncnn_summary(root: Path) -> None:
    out_dir = root / "07_evaluation" / "tdncnn"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    mapping = [
        ("Full", "full_45000"),
        ("Top-K", "topk_10"),
        ("Quantile", "quantile_spread_10"),
        ("Random", "random_10"),
    ]
    for label, subset in mapping:
        metrics_path = root / "09_downstream" / subset / "metrics.json"
        if not metrics_path.exists():
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "Architecture": "TDnCNN",
                "Subset": label,
                "Train size": int(payload.get("subset_size", 0)),
                "Optimizer steps": int(payload.get("total_steps", 0)),
                "PSNR": float(payload.get("eval_psnr", np.nan)),
                "MSE": float(payload.get("eval_mse", np.nan)),
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        full = table[table["Subset"] == "Full"]
        full_psnr = float(full.iloc[0]["PSNR"]) if not full.empty else float("nan")
        table["Delta PSNR vs Full"] = table["PSNR"] - full_psnr
    table.to_csv(out_dir / "summary_table.csv", index=False)
    (out_dir / "summary_table.md").write_text(table.to_csv(index=False) + "\n", encoding="utf-8")


def resolve_device(device_arg: str | None, smoke: bool = False) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cpu" if smoke else ("cuda" if torch.cuda.is_available() else "cpu"))


def cycle_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


class CorpusTaskNoiseWithSigmaDataset(Dataset):
    def __init__(
        self,
        corpus_dataset: MaterializedCorpusDataset,
        *,
        task_seed: int,
        task_noise_version: str,
        sigma_min: float,
        sigma_max: float,
        augmentation_seed: int | None = None,
        augment: bool = False,
    ) -> None:
        self.corpus_dataset = corpus_dataset
        self.task_seed = int(task_seed)
        self.task_noise_version = str(task_noise_version)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.augmentation_seed = None if augmentation_seed is None else int(augmentation_seed)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.corpus_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_target, original_index = self.corpus_dataset[index]
        if self.augment:
            x_target = apply_cifar_spatial_augmentation(
                x_target,
                original_index=int(original_index),
                augmentation_seed=int(self.augmentation_seed or 42),
                task_noise_version=self.task_noise_version,
            )
        sample_seed = stable_int_seed("cifar10", "train_task_noise", int(original_index), self.task_seed, self.task_noise_version)
        generator = torch.Generator().manual_seed(sample_seed)
        u = torch.rand((), generator=generator).item()
        sigma = float(self.sigma_min + (self.sigma_max - self.sigma_min) * u)
        noise = torch.randn(x_target.shape, generator=generator, dtype=x_target.dtype) * sigma
        x_input = (x_target + noise).clamp(0.0, 1.0)
        return x_input.to(torch.float32), x_target.to(torch.float32), torch.tensor(sigma, dtype=torch.float32), torch.tensor(int(original_index), dtype=torch.long)


def apply_cifar_spatial_augmentation(
    x: torch.Tensor,
    *,
    original_index: int,
    augmentation_seed: int,
    task_noise_version: str,
) -> torch.Tensor:
    seed = stable_int_seed("cifar10", "topk_spatial_augmentation", int(original_index), int(augmentation_seed), task_noise_version)
    generator = torch.Generator().manual_seed(seed)
    padded = F.pad(x.unsqueeze(0), (4, 4, 4, 4), mode="reflect").squeeze(0)
    top = int(torch.randint(0, 9, (), generator=generator).item())
    left = int(torch.randint(0, 9, (), generator=generator).item())
    cropped = padded[:, top : top + 32, left : left + 32]
    if float(torch.rand((), generator=generator).item()) < 0.5:
        cropped = torch.flip(cropped, dims=[2])
    return cropped.clamp(0.0, 1.0).to(torch.float32)


class EvalNoiseWithSigmaDataset(Dataset):
    def __init__(self, input_path: Path, target_path: Path, manifest_path: Path) -> None:
        base = EvalNoiseDataset(input_path, target_path)
        self.inputs = base.inputs
        self.targets = base.targets
        self.manifest = pd.read_csv(manifest_path)
        if "sigma" not in self.manifest.columns:
            raise ValueError(f"Eval manifest must contain sigma: {manifest_path}")

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.manifest.iloc[int(index)]
        return self.inputs[int(index)], self.targets[int(index)], torch.tensor(float(row["sigma"]), dtype=torch.float32), torch.tensor(int(row.get("original_index", index)), dtype=torch.long)


def drunet_stage_root(root: Path, subset: str) -> Path:
    return root / "06_downstream" / "drunet" / subset.replace("full_45000", "full").replace("quantile_spread_10", "quantile").replace("random_10", "random")


def build_current_drunet(cfg: dict[str, Any], *, smoke: bool) -> nn.Module:
    model_cfg = cfg.get("downstream", {}).get("drunet", {})
    official = bool(deep_get(cfg, "smoke.drunet_official", False) if smoke else model_cfg.get("official", True))
    return build_drunet(
        in_channels=3,
        official=official,
        nc=model_cfg.get("nc"),
        nb=int(model_cfg.get("nb", 4)),
        act_mode=str(model_cfg.get("act_mode", "R")),
        downsample_mode=str(model_cfg.get("downsample_mode", "strideconv")),
        upsample_mode=str(model_cfg.get("upsample_mode", "convtranspose")),
    )


def drunet_initial_state(root: Path, cfg: dict[str, Any], *, smoke: bool) -> Path:
    path = root / "06_downstream" / "drunet" / "initial_state.pt"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(deep_get(cfg, "seeds.model_seed", 42)))
    torch.save(build_current_drunet(cfg, smoke=smoke).state_dict(), path)
    return path


def restormer_stage_root(root: Path, subset: str) -> Path:
    return (
        root
        / "06_downstream"
        / "restormer"
        / subset.replace("full_45000", "full")
        .replace("quantile_spread_10", "quantile")
        .replace("random_10", "random")
    )
def build_current_restormer(
    cfg: dict[str, Any],
    *,
    smoke: bool,
) -> nn.Module:
    model_cfg = cfg.get("downstream", {}).get("restormer", {})

    if smoke:
        dim = int(deep_get(cfg, "smoke.restormer_dim", 16))
        num_blocks = deep_get(
            cfg,
            "smoke.restormer_num_blocks",
            [1, 1, 1, 1],
        )
        num_refinement_blocks = int(
            deep_get(cfg, "smoke.restormer_num_refinement_blocks", 1)
        )
        heads = deep_get(
            cfg,
            "smoke.restormer_heads",
            [1, 2, 4, 8],
        )
    else:
        dim = int(model_cfg.get("dim", 48))
        num_blocks = model_cfg.get(
            "num_blocks",
            [4, 6, 6, 8],
        )
        num_refinement_blocks = int(
            model_cfg.get("num_refinement_blocks", 4)
        )
        heads = model_cfg.get(
            "heads",
            [1, 2, 4, 8],
        )

    return build_restormer(
        in_channels=3,
        out_channels=3,
        dim=dim,
        num_blocks=num_blocks,
        num_refinement_blocks=num_refinement_blocks,
        heads=heads,
        ffn_expansion_factor=float(
            model_cfg.get("ffn_expansion_factor", 2.66)
        ),
        bias=bool(model_cfg.get("bias", False)),
        layer_norm_type=str(
            model_cfg.get("layer_norm_type", "WithBias")
        ),
    )
def restormer_initial_state(
    root: Path,
    cfg: dict[str, Any],
    *,
    smoke: bool,
) -> Path:
    path = (
        root
        / "06_downstream"
        / "restormer"
        / "initial_state.pt"
    )

    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(
        int(deep_get(cfg, "seeds.model_seed", 42))
    )

    model = build_current_restormer(
        cfg,
        smoke=smoke,
    )

    torch.save(
        model.state_dict(),
        path,
    )

    return path

def drunet_train_dataset(root: Path, cfg: dict[str, Any], subset: str, *, validation: bool = False) -> CorpusTaskNoiseWithSigmaDataset:
    return drunet_train_dataset_with_options(root, cfg, subset, validation=validation, augment=False, task_noise_version_suffix="")


def drunet_train_dataset_with_options(
    root: Path,
    cfg: dict[str, Any],
    subset: str,
    *,
    validation: bool = False,
    augment: bool = False,
    task_noise_version_suffix: str = "",
) -> CorpusTaskNoiseWithSigmaDataset:
    if validation:
        indices_path = root / "00_protocol" / "score_tuning_original_indices.npy"
        version_suffix = "_validation"
    else:
        indices_path = root / "08_downstream_commands" / f"{subset}_indices.npy"
        version_suffix = ""
    if not indices_path.exists():
        raise FileNotFoundError(f"Missing CIFAR DRUNet indices: {indices_path}")
    require_path_under(indices_path, root, "DRUNet subset indices")
    corpus = MaterializedCorpusDataset(
        root / "01_corpus" / "train_corpus.pt",
        root / "01_corpus" / "train_manifest.csv",
        original_indices=np.load(indices_path).astype(np.int64),
        normalized=False,
        return_original_index=True,
    )
    noise_cfg = deep_get(cfg, "downstream.task_noise", {})
    return CorpusTaskNoiseWithSigmaDataset(
        corpus,
        task_seed=int(deep_get(cfg, "seeds.task_noise_seed", 4242)),
        task_noise_version=str(noise_cfg.get("version", "cifar10_current_task_awgn_v1")) + task_noise_version_suffix + version_suffix,
        sigma_min=float(noise_cfg.get("sigma_min", 0.1)),
        sigma_max=float(noise_cfg.get("sigma_max", 0.8)),
        augmentation_seed=42,
        augment=bool(augment and not validation),
    )
def restormer_train_dataset(
    root: Path,
    cfg: dict[str, Any],
    subset: str,
    *,
    validation: bool = False,
) -> CorpusTaskNoiseWithSigmaDataset:
    return drunet_train_dataset(
        root,
        cfg,
        subset,
        validation=validation,
    )

@torch.no_grad()

def restormer_validation_loss(
    root: Path,
    cfg: dict[str, Any],
    model: nn.Module,
    device: torch.device,
    *,
    batch_size: int,
) -> float:
    loader = DataLoader(
        restormer_train_dataset(
            root,
            cfg,
            "full_45000",
            validation=True,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    criterion = nn.MSELoss(reduction="sum")

    model.eval()

    total = 0.0
    count = 0

    for x_input, x_target, _sigma, _ in loader:
        x_input = x_input.to(device)
        x_target = x_target.to(device)

        pred = model(x_input)

        total += float(
            criterion(pred, x_target).item()
        )

        count += int(np.prod(x_target.shape))

    model.train()

    return total / max(count, 1)

@torch.no_grad()
def drunet_validation_loss(root: Path, cfg: dict[str, Any], model: nn.Module, device: torch.device, *, batch_size: int) -> float:
    loader = DataLoader(drunet_train_dataset(root, cfg, "full_45000", validation=True), batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.MSELoss(reduction="sum")
    model.eval()
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


def run_drunet_train(cfg: dict[str, Any], args: argparse.Namespace, root: Path, subset: str) -> None:
    pass_marker = root / "06_confirmation" / "PASS.marker"
    if not pass_marker.exists():
        raise RuntimeError(f"DRUNet downstream is blocked until PASS marker exists: {pass_marker}")
    stage_cfg = cfg["downstream"]["drunet"]
    out_dir = drunet_stage_root(root, subset)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    status_path = out_dir / "status.json"
    if args.resume and status_path.exists() and best_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "complete":
            print(f"[SKIP] DRUNet {subset} | completed best.pt found")
            return
    device = resolve_device(args.device, args.smoke)
    batch_size = int(deep_get(cfg, "smoke.downstream_batch_size", 4) if args.smoke else stage_cfg.get("batch_size", 64))
    max_steps = deep_get(cfg, "smoke.downstream_max_optimizer_steps", None) if args.smoke else stage_cfg.get("max_optimizer_steps")
    if max_steps is None:
        full_indices = np.load(root / "08_downstream_commands" / "full_45000_indices.npy")
        max_steps = int(math.ceil(len(full_indices) / batch_size) * 5)
    model = build_current_drunet(cfg, smoke=args.smoke).to(device)
    initial = drunet_initial_state(root, cfg, smoke=args.smoke)
    model.load_state_dict(torch.load(initial, map_location=device), strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(stage_cfg.get("lr", 1e-4)), weight_decay=float(stage_cfg.get("weight_decay", 0.0)))
    loader = DataLoader(
        drunet_train_dataset(root, cfg, subset),
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(deep_get(cfg, "runtime.num_workers", 0)),
        generator=torch.Generator().manual_seed(int(deep_get(cfg, "seeds.model_seed", 42))),
    )
    iterator = cycle_loader(loader)
    criterion = nn.MSELoss()
    early = stage_cfg["early_stopping"]
    val_interval = max(1, int(deep_get(cfg, "smoke.downstream_validation_interval_steps", 1) if args.smoke else stage_cfg.get("validation_interval_steps", 500)))
    patience = max(1, int(deep_get(cfg, "smoke.downstream_early_stopping_patience", 2) if args.smoke else early.get("patience", 10)))
    min_delta = max(0.0, float(early.get("min_delta", 1e-4)))
    warmup_steps = max(0, int(deep_get(cfg, "smoke.downstream_early_stopping_warmup_steps", 0) if args.smoke else early.get("warmup_steps", 1000)))
    best_val_loss = float("inf")
    best_step = 0
    checks_without_improvement = 0
    stop_reason = "max_steps"
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    log_start = time.time()
    for step in range(1, int(max_steps) + 1):
        x_input, x_target, sigma, _ = next(iterator)
        x_input = x_input.to(device)
        x_target = x_target.to(device)
        sigma = sigma.to(device)
        loss = criterion(model(x_input, sigma=sigma), x_target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        train_loss = float(loss.detach().cpu().item())
        val_loss = float("nan")
        if step == 1 or step == int(max_steps) or step % val_interval == 0:
            val_loss = drunet_validation_loss(root, cfg, model, device, batch_size=batch_size)
            if step >= warmup_steps and val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_step = int(step)
                checks_without_improvement = 0
                torch.save(model.state_dict(), best_path)
            elif step >= warmup_steps:
                checks_without_improvement += 1
                if checks_without_improvement >= patience:
                    stop_reason = "early_stopping"
        rows.append({"step": step, "train_loss": train_loss, "val_loss": val_loss, "best_val_loss": best_val_loss, "best_step": best_step, "patience_counter": checks_without_improvement})
        if should_log_progress(step, int(max_steps), val_interval) or np.isfinite(val_loss):
            training_progress(
                "CIFAR-10",
                f"drunet_{subset}",
                unit="step",
                current=step,
                total=int(max_steps),
                train_loss=train_loss,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                best_step=best_step,
                patience_counter=checks_without_improvement,
                patience=patience,
                lr=optimizer.param_groups[0]["lr"],
                start_time=log_start,
            )
            pd.DataFrame(rows).to_csv(out_dir / "training_history.csv", index=False)
        if stop_reason == "early_stopping":
            break
    if not best_path.exists():
        torch.save(model.state_dict(), best_path)
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "step": int(step)}, out_dir / "last.pt")
    pd.DataFrame(rows).to_csv(out_dir / "training_history.csv", index=False)
    subset_indices = np.load(root / "08_downstream_commands" / f"{subset}_indices.npy")
    subset_path = root / "08_downstream_commands" / f"{subset}_indices.npy"
    selected_config_path = root / "05_selected_configuration" / "selected_score_config.yaml"
    corpus_manifest_path = root / "01_corpus" / "train_manifest.csv"
    meta = {
        "model": "drunet",
        "subset": subset,
        "variant": SUBSET_LABELS[subset],
        "subset_size": int(len(subset_indices)),
        "optimizer_steps": int(step),
        "batch_size": int(batch_size),
        "initial_state": str(initial),
        "initial_state_sha256": sha256_file(initial),
        "subset_indices": str(subset_path),
        "subset_indices_sha256": sha256_file(subset_path),
        "corpus_manifest": str(corpus_manifest_path),
        "corpus_manifest_sha256": sha256_file(corpus_manifest_path),
        "selected_score_config": str(selected_config_path),
        "selected_score_config_sha256": sha256_file(selected_config_path),
        "early_stopping_enabled": True,
        "early_stopping_monitor": "val_loss",
        "early_stopping_mode": "min",
        "early_stopping_patience": int(patience),
        "early_stopping_min_delta": float(min_delta),
        "early_stopping_warmup_steps": int(warmup_steps),
        "validation_interval_steps": int(val_interval),
        "best_step": int(best_step),
        "best_val_loss": float(best_val_loss),
        "stop_reason": stop_reason,
        "wall_clock_train_seconds": time.perf_counter() - start,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(out_dir / "last.pt"),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "status.json").write_text(json.dumps({"status": "complete", **meta}, indent=2), encoding="utf-8")

def run_restormer_train(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    root: Path,
    subset: str,
) -> None:
    pass_marker = root / "06_confirmation" / "PASS.marker"

    if not pass_marker.exists():
        raise RuntimeError(
            f"Restormer downstream is blocked until PASS marker exists: {pass_marker}"
        )

    stage_cfg = cfg["downstream"]["restormer"]

    out_dir = restormer_stage_root(root, subset)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_path = out_dir / "best.pt"
    status_path = out_dir / "status.json"

    if args.resume and status_path.exists() and best_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))

        if status.get("status") == "complete":
            print(
                f"[SKIP] Restormer {subset} | completed best.pt found"
            )
            return

    device = resolve_device(args.device, args.smoke)

    batch_size = int(
        deep_get(cfg, "smoke.downstream_batch_size", 4)
        if args.smoke
        else stage_cfg.get("batch_size", 16)
    )

    max_steps = (
        deep_get(
            cfg,
            "smoke.downstream_max_optimizer_steps",
            None,
        )
        if args.smoke
        else stage_cfg.get("max_optimizer_steps")
    )

    if max_steps is None:
        full_indices = np.load(
            root
            / "08_downstream_commands"
            / "full_45000_indices.npy"
        )

        max_steps = int(
            math.ceil(len(full_indices) / batch_size) * 5
        )

    model = build_current_restormer(
        cfg,
        smoke=args.smoke,
    ).to(device)

    initial = restormer_initial_state(
        root,
        cfg,
        smoke=args.smoke,
    )

    model.load_state_dict(
        torch.load(initial, map_location=device),
        strict=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(stage_cfg.get("lr", 1e-4)),
        weight_decay=float(
            stage_cfg.get("weight_decay", 0.0)
        ),
    )

    loader = DataLoader(
        restormer_train_dataset(
            root,
            cfg,
            subset,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(
            deep_get(cfg, "runtime.num_workers", 0)
        ),
        generator=torch.Generator().manual_seed(
            int(deep_get(cfg, "seeds.model_seed", 42))
        ),
    )

    iterator = cycle_loader(loader)

    criterion = nn.MSELoss()

    early = stage_cfg["early_stopping"]

    val_interval = max(
        1,
        int(
            deep_get(
                cfg,
                "smoke.downstream_validation_interval_steps",
                1,
            )
            if args.smoke
            else stage_cfg.get(
                "validation_interval_steps",
                500,
            )
        ),
    )

    patience = max(
        1,
        int(
            deep_get(
                cfg,
                "smoke.downstream_early_stopping_patience",
                2,
            )
            if args.smoke
            else early.get("patience", 10)
        ),
    )

    min_delta = max(
        0.0,
        float(
            early.get(
                "min_delta",
                1e-4,
            )
        ),
    )

    warmup_steps = max(
        0,
        int(
            deep_get(
                cfg,
                "smoke.downstream_early_stopping_warmup_steps",
                0,
            )
            if args.smoke
            else early.get(
                "warmup_steps",
                1000,
            )
        ),
    )

    best_val_loss = float("inf")
    best_step = 0
    checks_without_improvement = 0
    stop_reason = "max_steps"

    rows: list[dict[str, Any]] = []

    start = time.perf_counter()
    log_start = time.time()

    for step in range(1, int(max_steps) + 1):
        x_input, x_target, _sigma, _ = next(iterator)

        x_input = x_input.to(device)
        x_target = x_target.to(device)

        pred = model(x_input)

        loss = criterion(
            pred,
            x_target,
        )

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        optimizer.step()

        train_loss = float(
            loss.detach().cpu().item()
        )

        val_loss = float("nan")

        if (
            step == 1
            or step == int(max_steps)
            or step % val_interval == 0
        ):
            val_loss = restormer_validation_loss(
                root,
                cfg,
                model,
                device,
                batch_size=batch_size,
            )

            if (
                step >= warmup_steps
                and val_loss < best_val_loss - min_delta
            ):
                best_val_loss = val_loss
                best_step = int(step)
                checks_without_improvement = 0

                torch.save(
                    model.state_dict(),
                    best_path,
                )

            elif step >= warmup_steps:
                checks_without_improvement += 1

                if checks_without_improvement >= patience:
                    stop_reason = "early_stopping"

        rows.append(
            {
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "best_step": best_step,
                "patience_counter": checks_without_improvement,
            }
        )

        if (
            should_log_progress(
                step,
                int(max_steps),
                val_interval,
            )
            or np.isfinite(val_loss)
        ):
            training_progress(
                "CIFAR-10",
                f"restormer_{subset}",
                unit="step",
                current=step,
                total=int(max_steps),
                train_loss=train_loss,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                best_step=best_step,
                patience_counter=checks_without_improvement,
                patience=patience,
                lr=optimizer.param_groups[0]["lr"],
                start_time=log_start,
            )

            pd.DataFrame(rows).to_csv(
                out_dir / "training_history.csv",
                index=False,
            )

        if stop_reason == "early_stopping":
            break

    if not best_path.exists():
        torch.save(
            model.state_dict(),
            best_path,
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": int(step),
        },
        out_dir / "last.pt",
    )

    pd.DataFrame(rows).to_csv(
        out_dir / "training_history.csv",
        index=False,
    )

    subset_path = (
        root
        / "08_downstream_commands"
        / f"{subset}_indices.npy"
    )

    subset_indices = np.load(subset_path)

    selected_config_path = (
        root
        / "05_selected_configuration"
        / "selected_score_config.yaml"
    )

    corpus_manifest_path = (
        root
        / "01_corpus"
        / "train_manifest.csv"
    )

    meta = {
        "model": "restormer",
        "subset": subset,
        "variant": SUBSET_LABELS[subset],
        "subset_size": int(len(subset_indices)),
        "optimizer_steps": int(step),
        "batch_size": int(batch_size),

        "initial_state": str(initial),
        "initial_state_sha256": sha256_file(initial),

        "subset_indices": str(subset_path),
        "subset_indices_sha256": sha256_file(subset_path),

        "corpus_manifest": str(corpus_manifest_path),
        "corpus_manifest_sha256": sha256_file(
            corpus_manifest_path
        ),

        "selected_score_config": str(
            selected_config_path
        ),
        "selected_score_config_sha256": sha256_file(
            selected_config_path
        ),

        "early_stopping_enabled": True,
        "early_stopping_monitor": "val_loss",
        "early_stopping_mode": "min",
        "early_stopping_patience": int(patience),
        "early_stopping_min_delta": float(min_delta),
        "early_stopping_warmup_steps": int(warmup_steps),
        "validation_interval_steps": int(val_interval),

        "best_step": int(best_step),
        "best_val_loss": float(best_val_loss),
        "stop_reason": stop_reason,

        "wall_clock_train_seconds": (
            time.perf_counter() - start
        ),

        "best_checkpoint": str(best_path),
        "last_checkpoint": str(
            out_dir / "last.pt"
        ),
    }

    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    (out_dir / "status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                **meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

def train_drunet_topk_ablation(cfg: dict[str, Any], args: argparse.Namespace, root: Path, *, variant: str, lr_multiplier: float) -> None:
    pass_marker = root / "06_confirmation" / "PASS.marker"
    if not pass_marker.exists():
        raise RuntimeError(f"DRUNet Top-K ablation is blocked until PASS marker exists: {pass_marker}")
    stage_cfg = cfg["downstream"]["drunet"]
    out_dir = topk_ablation_root(root, "drunet", variant)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    status_path = out_dir / "status.json"
    if args.resume and status_path.exists() and best_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "complete":
            print(f"[SKIP] DRUNet {variant} | completed best.pt found")
            return
    device = resolve_device(args.device, args.smoke)
    batch_size = int(deep_get(cfg, "smoke.downstream_batch_size", 4) if args.smoke else stage_cfg.get("batch_size", 64))
    max_steps = deep_get(cfg, "smoke.downstream_max_optimizer_steps", None) if args.smoke else stage_cfg.get("max_optimizer_steps")
    if max_steps is None:
        full_indices = np.load(root / "08_downstream_commands" / "full_45000_indices.npy")
        max_steps = int(math.ceil(len(full_indices) / batch_size) * 5)
    model = build_current_drunet(cfg, smoke=args.smoke).to(device)
    initial = drunet_initial_state(root, cfg, smoke=args.smoke)
    model.load_state_dict(torch.load(initial, map_location=device), strict=True)
    lr = float(stage_cfg.get("lr", 1e-4)) * float(lr_multiplier)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=float(stage_cfg.get("weight_decay", 0.0)))
    loader = DataLoader(
        drunet_train_dataset_with_options(root, cfg, "topk_10", augment=True, task_noise_version_suffix=f"_{variant}"),
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(deep_get(cfg, "runtime.num_workers", 0)),
        generator=torch.Generator().manual_seed(int(deep_get(cfg, "seeds.model_seed", 42))),
    )
    iterator = cycle_loader(loader)
    criterion = nn.MSELoss()
    early = stage_cfg["early_stopping"]
    val_interval = max(1, int(deep_get(cfg, "smoke.downstream_validation_interval_steps", 1) if args.smoke else stage_cfg.get("validation_interval_steps", 500)))
    patience = max(1, int(deep_get(cfg, "smoke.downstream_early_stopping_patience", 2) if args.smoke else early.get("patience", 10)))
    min_delta = max(0.0, float(early.get("min_delta", 1e-4)))
    warmup_steps = max(0, int(deep_get(cfg, "smoke.downstream_early_stopping_warmup_steps", 0) if args.smoke else early.get("warmup_steps", 1000)))
    best_val_loss = float("inf")
    best_step = 0
    checks_without_improvement = 0
    stop_reason = "max_steps"
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    log_start = time.time()
    for step in range(1, int(max_steps) + 1):
        x_input, x_target, sigma, _ = next(iterator)
        x_input = x_input.to(device)
        x_target = x_target.to(device)
        sigma = sigma.to(device)
        loss = criterion(model(x_input, sigma=sigma), x_target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        train_loss = float(loss.detach().cpu().item())
        val_loss = float("nan")
        if step == 1 or step == int(max_steps) or step % val_interval == 0:
            val_loss = drunet_validation_loss(root, cfg, model, device, batch_size=batch_size)
            if step >= warmup_steps and val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_step = int(step)
                checks_without_improvement = 0
                torch.save(model.state_dict(), best_path)
            elif step >= warmup_steps:
                checks_without_improvement += 1
                if checks_without_improvement >= patience:
                    stop_reason = "early_stopping"
        rows.append({"step": step, "train_loss": train_loss, "val_loss": val_loss, "best_val_loss": best_val_loss, "best_step": best_step, "patience_counter": checks_without_improvement})
        if should_log_progress(step, int(max_steps), val_interval) or np.isfinite(val_loss):
            training_progress("CIFAR-10", f"drunet_{variant}", unit="step", current=step, total=int(max_steps), train_loss=train_loss, val_loss=val_loss, best_val_loss=best_val_loss, best_step=best_step, patience_counter=checks_without_improvement, patience=patience, lr=lr, start_time=log_start)
            pd.DataFrame(rows).to_csv(out_dir / "training_history.csv", index=False)
        if stop_reason == "early_stopping":
            break
    if not best_path.exists():
        torch.save(model.state_dict(), best_path)
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "step": int(step)}, out_dir / "last.pt")
    pd.DataFrame(rows).to_csv(out_dir / "training_history.csv", index=False)
    topk_indices = np.load(root / "08_downstream_commands" / "topk_10_indices.npy")
    meta = {
        "status": "complete",
        "model": "drunet",
        "variant": variant,
        "subset": "topk_10",
        "subset_size": int(len(topk_indices)),
        "augmentation": "RandomCrop32ReflectPad4+HorizontalFlip0.5",
        "lr_multiplier": float(lr_multiplier),
        "lr": float(lr),
        "optimizer_steps": int(step),
        "batch_size": int(batch_size),
        "initial_state_sha256": sha256_file(initial),
        "subset_indices_sha256": sha256_file(root / "08_downstream_commands" / "topk_10_indices.npy"),
        "early_stopping_enabled": True,
        "early_stopping_monitor": "val_loss",
        "early_stopping_mode": "min",
        "early_stopping_patience": int(patience),
        "early_stopping_min_delta": float(min_delta),
        "early_stopping_warmup_steps": int(warmup_steps),
        "validation_interval_steps": int(val_interval),
        "best_step": int(best_step),
        "best_val_loss": float(best_val_loss),
        "stop_reason": stop_reason,
        "wall_clock_train_seconds": time.perf_counter() - start,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(out_dir / "last.pt"),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    status_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


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
    psnr = 10.0 * torch.log10(1.0 / (mse + 1e-12))
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    ssim = np.asarray([skimage_ssim(target_np[i], pred_np[i], data_range=1.0, channel_axis=0) for i in range(pred_np.shape[0])], dtype=np.float64)
    if lpips_model is None:
        lpips_values = np.full(pred_np.shape[0], np.nan, dtype=np.float64)
    else:
        with torch.inference_mode():
            lpips_values = lpips_model(prepare_for_lpips(pred), prepare_for_lpips(target)).detach().flatten().cpu().numpy().astype(np.float64)
    return mse.detach().cpu().numpy(), psnr.detach().cpu().numpy(), ssim, lpips_values


@torch.no_grad()
def run_drunet_evaluate(cfg: dict[str, Any], args: argparse.Namespace, root: Path) -> None:
    device = resolve_device(args.device, args.smoke)
    eval_dir = root / "07_evaluation" / "drunet"
    eval_dir.mkdir(parents=True, exist_ok=True)
    dataset = EvalNoiseWithSigmaDataset(
        root / "08_downstream_commands" / "eval_noise_cache" / "test_eval_input.pt",
        root / "08_downstream_commands" / "eval_noise_cache" / "test_eval_target.pt",
        root / "08_downstream_commands" / "eval_noise_cache" / "test_noise_manifest.csv",
    )
    loader = DataLoader(dataset, batch_size=int(deep_get(cfg, "evaluation.batch_size", 128)), shuffle=False, num_workers=0)
    lpips_model = create_lpips_model(device, enabled=bool(deep_get(cfg, "evaluation.lpips_enabled", True)))
    rows: list[dict[str, Any]] = []
    for subset in ["full_45000", "topk_10", "quantile_spread_10", "random_10"]:
        model = build_current_drunet(cfg, smoke=args.smoke).to(device)
        model.load_state_dict(torch.load(drunet_stage_root(root, subset) / "best.pt", map_location=device), strict=True)
        model.eval()
        offset = 0
        for x_input, x_target, sigma, original_index in loader:
            x_input = x_input.to(device)
            x_target = x_target.to(device)
            sigma = sigma.to(device)
            mse, psnr, ssim, lpips_values = batch_metrics(model(x_input, sigma=sigma), x_target, lpips_model)
            for i in range(int(x_input.shape[0])):
                rows.append({"Architecture": "DRUNet", "Variant": SUBSET_LABELS[subset], "sample_index": int(offset + i), "original_index": int(original_index[i].item()), "MSE": float(mse[i]), "PSNR": float(psnr[i]), "SSIM": float(ssim[i]), "LPIPS": float(lpips_values[i])})
            offset += int(x_input.shape[0])
    per = pd.DataFrame(rows)
    per.to_csv(eval_dir / "per_image_metrics.csv", index=False)
    summary_rows = []
    for variant, group in per.groupby("Variant", sort=False):
        row: dict[str, Any] = {"Architecture": "DRUNet", "Variant": variant, "Images": int(len(group))}
        for metric in ["PSNR", "SSIM", "LPIPS", "MSE"]:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[metric] = float(values.mean()) if values.notna().any() else np.nan
        meta_path = drunet_stage_root(root, next(key for key, label in SUBSET_LABELS.items() if label == variant)) / "run_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            row["Train size"] = int(meta.get("subset_size", 0))
            row["Optimizer steps"] = int(meta.get("optimizer_steps", 0))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    full = summary[summary["Variant"] == "Full"]
    if not full.empty:
        full_psnr = float(full.iloc[0]["PSNR"])
        summary["Delta PSNR vs DRUNet Full"] = summary["PSNR"] - full_psnr
    summary.to_csv(eval_dir / "summary_table.csv", index=False)
    (eval_dir / "summary_table.md").write_text(summary.to_csv(index=False) + "\n", encoding="utf-8")

@torch.no_grad()
def run_restormer_evaluate(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    root: Path,
) -> None:
    device = resolve_device(args.device, args.smoke)

    eval_dir = root / "07_evaluation" / "restormer"
    eval_dir.mkdir(parents=True, exist_ok=True)

    dataset = EvalNoiseWithSigmaDataset(
        root
        / "08_downstream_commands"
        / "eval_noise_cache"
        / "test_eval_input.pt",
        root
        / "08_downstream_commands"
        / "eval_noise_cache"
        / "test_eval_target.pt",
        root
        / "08_downstream_commands"
        / "eval_noise_cache"
        / "test_noise_manifest.csv",
    )

    loader = DataLoader(
        dataset,
        batch_size=int(
            deep_get(cfg, "evaluation.batch_size", 128)
        ),
        shuffle=False,
        num_workers=0,
    )

    lpips_model = create_lpips_model(
        device,
        enabled=bool(
            deep_get(cfg, "evaluation.lpips_enabled", True)
        ),
    )

    rows: list[dict[str, Any]] = []

    for subset in [
        "full_45000",
        "topk_10",
        "quantile_spread_10",
        "random_10",
    ]:
        model = build_current_restormer(
            cfg,
            smoke=args.smoke,
        ).to(device)

        checkpoint = (
            restormer_stage_root(root, subset)
            / "best.pt"
        )

        model.load_state_dict(
            torch.load(
                checkpoint,
                map_location=device,
            ),
            strict=True,
        )

        model.eval()

        offset = 0

        for x_input, x_target, _sigma, original_index in loader:
            x_input = x_input.to(device)
            x_target = x_target.to(device)

            pred = model(x_input)

            mse, psnr, ssim, lpips_values = batch_metrics(
                pred,
                x_target,
                lpips_model,
            )

            for i in range(int(x_input.shape[0])):
                rows.append(
                    {
                        "Architecture": "Restormer",
                        "Variant": SUBSET_LABELS[subset],
                        "sample_index": int(offset + i),
                        "original_index": int(
                            original_index[i].item()
                        ),
                        "MSE": float(mse[i]),
                        "PSNR": float(psnr[i]),
                        "SSIM": float(ssim[i]),
                        "LPIPS": float(lpips_values[i]),
                    }
                )

            offset += int(x_input.shape[0])

    per = pd.DataFrame(rows)

    per.to_csv(
        eval_dir / "per_image_metrics.csv",
        index=False,
    )

    summary_rows = []

    for variant, group in per.groupby(
        "Variant",
        sort=False,
    ):
        row: dict[str, Any] = {
            "Architecture": "Restormer",
            "Variant": variant,
            "Images": int(len(group)),
        }

        for metric in [
            "PSNR",
            "SSIM",
            "LPIPS",
            "MSE",
        ]:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            )

            row[metric] = (
                float(values.mean())
                if values.notna().any()
                else np.nan
            )

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    if not summary.empty:
        full_rows = summary[
            summary["Variant"] == "Full"
        ]

        if not full_rows.empty:
            full_psnr = float(
                full_rows.iloc[0]["PSNR"]
            )

            summary["Delta_PSNR_vs_Full"] = (
                summary["PSNR"] - full_psnr
            )

    summary.to_csv(
        eval_dir / "summary_table.csv",
        index=False,
    )

@torch.no_grad()
def run_topk_ablation_evaluate(cfg: dict[str, Any], args: argparse.Namespace, root: Path) -> None:
    device = resolve_device(args.device, args.smoke)
    out_dir = root / "10_topk_ablations" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    lpips_model = create_lpips_model(device, enabled=bool(deep_get(cfg, "evaluation.lpips_enabled", True)))
    eval_dataset = EvalNoiseWithSigmaDataset(
        root / "08_downstream_commands" / "eval_noise_cache" / "test_eval_input.pt",
        root / "08_downstream_commands" / "eval_noise_cache" / "test_eval_target.pt",
        root / "08_downstream_commands" / "eval_noise_cache" / "test_noise_manifest.csv",
    )
    loader = DataLoader(eval_dataset, batch_size=int(deep_get(cfg, "evaluation.batch_size", 128)), shuffle=False, num_workers=0)
    jobs = [
        ("TDnCNN", "Top-K baseline", root / "09_downstream" / "topk_10" / "best.pt", None),
        ("TDnCNN", "Top-K aug", topk_ablation_root(root, "tdncnn", "topk_aug") / "best.pt", None),
        ("TDnCNN", "Top-K aug LR x0.5", topk_ablation_root(root, "tdncnn", "topk_aug_lr05") / "best.pt", None),
        ("DRUNet", "Top-K baseline", drunet_stage_root(root, "topk_10") / "best.pt", "drunet"),
        ("DRUNet", "Top-K aug", topk_ablation_root(root, "drunet", "topk_aug") / "best.pt", "drunet"),
        ("DRUNet", "Top-K aug LR x0.5", topk_ablation_root(root, "drunet", "topk_aug_lr05") / "best.pt", "drunet"),
    ]
    rows: list[dict[str, Any]] = []
    for architecture, variant, checkpoint, model_kind in jobs:
        if not checkpoint.exists():
            print(f"[SKIP] missing ablation checkpoint: {checkpoint}")
            continue
        if architecture == "TDnCNN":
            model = DnCNN_2D(in_channels=3).to(device)
            model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
        else:
            model = build_current_drunet(cfg, smoke=args.smoke).to(device)
            model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
        model.eval()
        offset = 0
        for x_input, x_target, sigma, original_index in loader:
            x_input = x_input.to(device)
            x_target = x_target.to(device)
            sigma = sigma.to(device)
            pred = model(x_input, sigma=sigma) if model_kind == "drunet" else model(x_input)
            mse, psnr, ssim, lpips_values = batch_metrics(pred, x_target, lpips_model)
            for i in range(int(x_input.shape[0])):
                rows.append({
                    "Architecture": architecture,
                    "Variant": variant,
                    "sample_index": int(offset + i),
                    "original_index": int(original_index[i].item()),
                    "MSE": float(mse[i]),
                    "PSNR": float(psnr[i]),
                    "SSIM": float(ssim[i]),
                    "LPIPS": float(lpips_values[i]),
                    "checkpoint": str(checkpoint),
                })
            offset += int(x_input.shape[0])
    if not rows:
        raise FileNotFoundError("No Top-K ablation checkpoints were available for evaluation.")
    per = pd.DataFrame(rows)
    per.to_csv(out_dir / "per_image_metrics.csv", index=False)
    summary_rows = []
    for (architecture, variant), group in per.groupby(["Architecture", "Variant"], sort=False):
        row: dict[str, Any] = {"Architecture": architecture, "Variant": variant, "Images": int(len(group))}
        for metric in ["PSNR", "SSIM", "LPIPS", "MSE"]:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[metric] = float(values.mean()) if values.notna().any() else np.nan
        baseline = per[(per["Architecture"] == architecture) & (per["Variant"] == "Top-K baseline")]
        if not baseline.empty:
            row["Delta PSNR vs Top-K baseline"] = row["PSNR"] - float(pd.to_numeric(baseline["PSNR"]).mean())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary_table.csv", index=False)
    (out_dir / "summary_table.md").write_text(summary.to_csv(index=False) + "\n", encoding="utf-8")



def write_summary(root: Path) -> None:
    out_dir = root / "08_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    tdncnn = root / "07_evaluation" / "tdncnn" / "summary_table.csv"
    drunet = root / "07_evaluation" / "drunet" / "summary_table.csv"
    selected = root / "05_selected_configuration" / "selected_score_config.yaml"
    pass_marker = root / "06_confirmation" / "PASS.marker"
    random_indices = root / "08_downstream_commands" / "random_10_indices.npy"
    lines = [
        "# CIFAR-10 Current Pipeline Summary",
        "",
        "- domain: CIFAR-10",
        "- DDPM domain: image space",
        "- filtering gate: `06_confirmation/PASS.marker`",
        f"- PASS marker exists: `{pass_marker.exists()}`",
        f"- selected score config: `{selected}`",
        f"- Random baseline indices: `{random_indices}`",
        "",
    ]
    if tdncnn.exists():
        lines.extend(["## TDnCNN", pd.read_csv(tdncnn).to_csv(index=False), ""])
    if drunet.exists():
        lines.extend(["## DRUNet", pd.read_csv(drunet).to_csv(index=False), ""])
    (out_dir / "final_summary.md").write_text("\n".join(lines), encoding="utf-8")


def verify_random(root: Path) -> dict[str, Any]:
    subset_dir = root / "08_downstream_commands"
    paths = {
        "topk_10": subset_dir / "topk_10_indices.npy",
        "quantile_spread_10": subset_dir / "quantile_spread_10_indices.npy",
        "random_10": subset_dir / "random_10_indices.npy",
    }
    result: dict[str, Any] = {"available": all(path.exists() for path in paths.values())}
    if result["available"]:
        arrays = {name: np.load(path).astype(np.int64) for name, path in paths.items()}
        sizes = {name: int(len(arr)) for name, arr in arrays.items()}
        result.update(
            {
                "sizes": sizes,
                "same_k": len(set(sizes.values())) == 1,
                "unique": {name: bool(len(np.unique(arr)) == len(arr)) for name, arr in arrays.items()},
            }
        )
    return result


def run_external_evaluate(cfg: dict[str, Any], args: argparse.Namespace, root: Path) -> None:
    from scripts import evaluate_external

    external_root = Path(str(deep_get(cfg, "external.root", "data/external_benchmarks")))
    if not external_root.is_absolute():
        external_root = PROJECT_ROOT / external_root
    sigma = str(int(deep_get(cfg, "external.sigma", 25)))
    seed = str(int(deep_get(cfg, "external.seed", 42)))
    jobs = [
        (
            "tdncnn",
            root / "08_external_evaluation" / "tdncnn",
            {
                "full": root / "09_downstream" / "full_45000" / "best.pt",
                "topk": root / "09_downstream" / "topk_10" / "best.pt",
                "quantile": root / "09_downstream" / "quantile_spread_10" / "best.pt",
            },
        ),
        (
            "drunet",
            root / "08_external_evaluation" / "drunet",
            {
                "full": drunet_stage_root(root, "full_45000") / "best.pt",
                "topk": drunet_stage_root(root, "topk_10") / "best.pt",
                "quantile": drunet_stage_root(root, "quantile_spread_10") / "best.pt",
            },
        ),
    ]
    for model_type, output_root, checkpoints in jobs:
        missing = [str(path) for path in checkpoints.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Cannot run CIFAR external evaluation for {model_type}; missing checkpoints: {missing}")
        old_argv = sys.argv
        try:
            sys.argv = [
                "evaluate_external.py",
                "--model-type",
                model_type,
                "--all-benchmarks",
                "--root",
                str(external_root),
                "--output-root",
                str(output_root),
                "--sigma",
                sigma,
                "--seed",
                seed,
                "--lpips",
                "--full-checkpoint",
                str(checkpoints["full"]),
                "--topk-checkpoint",
                str(checkpoints["topk"]),
                "--quantile-checkpoint",
                str(checkpoints["quantile"]),
            ]
            evaluate_external.main()
        finally:
            sys.argv = old_argv


def run_stage(cfg: dict[str, Any], args: argparse.Namespace, root: Path, stage: str) -> None:
    if args.dry_run:
        print(f"[dry-run] CIFAR-10 stage={stage} root={root}")
        return
    if stage in VALIDATION_STAGE_MAP:
        run_score_validation_stage(validation_args(cfg, args, root, stage))
    elif stage == "tdncnn_evaluate":
        write_tdncnn_summary(root)
        print(f"tdncnn_evaluate: wrote {root / '07_evaluation' / 'tdncnn' / 'summary_table.csv'}")
    elif stage in DRUNET_SUBSETS:
        run_drunet_train(cfg, args, root, DRUNET_SUBSETS[stage])
    elif stage == "drunet_evaluate":
        run_drunet_evaluate(cfg, args, root)
        print(f"drunet_evaluate: wrote {root / '07_evaluation' / 'drunet' / 'summary_table.csv'}")
    elif stage in RESTORMER_SUBSETS:
        run_restormer_train(cfg,args,root,RESTORMER_SUBSETS[stage],)
    elif stage == "restormer_evaluate":
        run_restormer_evaluate(cfg,args,root,)
        print("restormer_evaluate: wrote "f"{root / '07_evaluation' / 'restormer' / 'summary_table.csv'}")
    elif stage in TOPK_ABLATION_STAGES:
        model_name, variant, lr_multiplier = TOPK_ABLATION_STAGES[stage]
        if model_name == "tdncnn":
            train_tdncnn_topk_ablation(cfg, args, root, variant=variant, lr_multiplier=lr_multiplier)
        else:
            train_drunet_topk_ablation(cfg, args, root, variant=variant, lr_multiplier=lr_multiplier)
    elif stage == "topk_ablation_evaluate":
        run_topk_ablation_evaluate(cfg, args, root)
        print(f"topk_ablation_evaluate: wrote {root / '10_topk_ablations' / 'evaluation' / 'summary_table.csv'}")
    elif stage == "external_evaluate":
        run_external_evaluate(cfg, args, root)
        print(f"external_evaluate: wrote {root / '08_external_evaluation'}")
    elif stage == "summary":
        random_status = verify_random(root)
        write_summary(root)
        print(f"summary: random_status={json.dumps(random_status, sort_keys=True)}")
    else:
        raise ValueError(f"Unsupported stage: {stage}")


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    root = resolve_experiment_root(cfg, args)
    validate_current_protocol("CIFAR-10", cfg, downstream_models=["tdncnn", "drunet","restormer"])
    print(
        json.dumps(
            {
                "pipeline": "cifar10",
                "root": str(root),
                "stage": args.stage,
                "stages": expanded_stages(args.stage),
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=2,
        )
    )
    print_protocol_summary(
        "CIFAR-10",
        cfg,
        run_id=args.run_id or str(deep_get(cfg, "experiment.run_id", "cifar10_001")),
        device=args.device or "auto",
        downstream_models=["tdncnn", "drunet", "restormer"],
    )
    for stage in expanded_stages(args.stage):
        with stage_logging("CIFAR-10", stage, run_id=args.run_id or str(deep_get(cfg, "experiment.run_id", "cifar10_001"))):
            run_stage(cfg, args, root, stage)


if __name__ == "__main__":
    main()
