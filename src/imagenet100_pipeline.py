from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as skimage_ssim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

from src.DRUNet_image import build_drunet
from src.TDnCNN_image import DnCNN_2D, align_lpips_scaling_layer, prepare_for_lpips
from src.Unet_model import UNet
from src.filtering.selection import select_quantile_spread_exact
from src.corruption_datasets import stable_int_seed
from src.evaluation.denoiser_panels import save_denoiser_comparison_panel
from src.evaluation.quality_shift_plots import plot_ecdf_realized_psnr, plot_quality_shift_3x3, plot_subset_fraction_metric
from src.evaluation.selection_panels import save_topk_cleanest_noisiest, save_topk_selection_examples
from src.utils.current_protocol import (
    print_protocol_summary,
    should_log_progress,
    stage_logging,
    training_progress,
    validate_current_protocol,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-12
STAGE_DIRS = {
    "preflight": "00_protocol/preflight",
    "prepare": "01_corpus",
    "ddpm_train": "02_ddpm",
    "score_validate": "03_score_validation",
    "score_select": "04_selected_score_config",
    "filter": "05_filtering",
    "filter_grid": "05_filtering",
    "tdncnn_full": "06_downstream/tdncnn/full",
    "tdncnn_topk": "06_downstream/tdncnn/topk_p10",
    "tdncnn_quantile": "06_downstream/tdncnn/quantile_p10",
    "tdncnn_random": "06_downstream/tdncnn/random_p10",
    "tdncnn_topk_p10": "06_downstream/tdncnn/topk_p10",
    "tdncnn_quantile_p10": "06_downstream/tdncnn/quantile_p10",
    "tdncnn_random_p10": "06_downstream/tdncnn/random_p10",
    "tdncnn_topk_p20": "06_downstream/tdncnn/topk_p20",
    "tdncnn_topk_p30": "06_downstream/tdncnn/topk_p30",
    "tdncnn_quantile_p20": "06_downstream/tdncnn/quantile_p20",
    "tdncnn_quantile_p30": "06_downstream/tdncnn/quantile_p30",
    "tdncnn_random_p20": "06_downstream/tdncnn/random_p20",
    "tdncnn_random_p30": "06_downstream/tdncnn/random_p30",
    "tdncnn_grid": "06_downstream/tdncnn",
    "tdncnn_evaluate": "07_evaluation/tdncnn",
    "tdncnn_external_eval": "08_external_benchmarks/tdncnn",
    "drunet_full": "06_downstream/drunet/full",
    "drunet_topk": "06_downstream/drunet/topk_p10",
    "drunet_quantile": "06_downstream/drunet/quantile_p10",
    "drunet_random": "06_downstream/drunet/random_p10",
    "drunet_topk_p10": "06_downstream/drunet/topk_p10",
    "drunet_quantile_p10": "06_downstream/drunet/quantile_p10",
    "drunet_random_p10": "06_downstream/drunet/random_p10",
    "drunet_topk_p20": "06_downstream/drunet/topk_p20",
    "drunet_topk_p30": "06_downstream/drunet/topk_p30",
    "drunet_quantile_p20": "06_downstream/drunet/quantile_p20",
    "drunet_quantile_p30": "06_downstream/drunet/quantile_p30",
    "drunet_random_p20": "06_downstream/drunet/random_p20",
    "drunet_random_p30": "06_downstream/drunet/random_p30",
    "drunet_grid": "06_downstream/drunet",
    "drunet_evaluate": "07_evaluation/drunet",
    "drunet_external_eval": "08_external_benchmarks/drunet",
    "evaluate_grid": "07_evaluation",
    "figures_grid": "08_figures",
    "external_grid": "08_external_benchmarks",
    "summary_grid": "09_summary",
    "grid_all": "09_summary",
    "downstream_early_stopping_audit": "00_protocol/downstream_early_stopping_audit",
    # Accept the older TDnCNN stage names.
    "downstream_full": "06_downstream/tdncnn/full",
    "downstream_topk": "06_downstream/tdncnn/topk_p10",
    "downstream_quantile": "06_downstream/tdncnn/quantile_p10",
    "downstream_random": "06_downstream/tdncnn/random_p10",
    "evaluate": "07_evaluation/tdncnn",
    "external_eval": "08_external_benchmarks/tdncnn",
    "summary": "09_summary",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def score_validate_log(stage_root: Path, message: str, **fields: Any) -> None:
    payload = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "message": message,
        **fields,
    }
    print(f"[score_validate] {message} {json.dumps(fields, sort_keys=True)}", flush=True)
    with (stage_root / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def downstream_log(stage_root: Path, message: str, **fields: Any) -> None:
    payload = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "message": message,
        **fields,
    }
    print(f"[downstream] {message} {json.dumps(fields, sort_keys=True)}", flush=True)
    with (stage_root / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_yaml(path: Path) -> dict[str, Any]:
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(cfg, dict):
        raise TypeError(f"Config must be a mapping: {path}")
    return cfg


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(payload), f=path)


def config_hash(cfg: dict[str, Any]) -> str:
    stable = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return sha256_bytes(stable)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def git_status_short() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=PROJECT_ROOT, text=True)
    except Exception as error:
        return f"git status unavailable: {error}"


def psnr_from_mse(mse: float) -> float:
    return float(10.0 * math.log10(1.0 / max(float(mse), EPS)))


def snr_from_powers(signal_power: float, noise_power: float) -> float:
    return float(10.0 * math.log10(max(float(signal_power), EPS) / max(float(noise_power), EPS)))


def metric_row(x_clean: torch.Tensor, x_preclip: torch.Tensor, x_corpus: torch.Tensor) -> dict[str, float]:
    noise_post = x_corpus - x_clean
    corruption_mse = float((noise_post**2).mean().item())
    signal_power = float((x_clean**2).mean().item())
    clipped_fraction = float(((x_preclip < 0.0) | (x_preclip > 1.0)).float().mean().item())
    return {
        "realized_psnr_db_postclip": psnr_from_mse(corruption_mse),
        "realized_snr_db_postclip": snr_from_powers(signal_power, corruption_mse),
        "corruption_mse": corruption_mse,
        "clipped_fraction": clipped_fraction,
        "signal_power": signal_power,
    }


def class_name_from_dataset(dataset: Dataset, class_id: int) -> str:
    classes = getattr(dataset, "classes", None)
    if classes is not None and 0 <= int(class_id) < len(classes):
        return str(classes[int(class_id)])
    return str(class_id)


def sample_identifier(dataset: Dataset, index: int, split: str) -> str:
    samples = getattr(dataset, "samples", None)
    if samples:
        path = Path(samples[int(index)][0])
        try:
            split_root = Path(getattr(dataset, "root"))
            return str(path.relative_to(split_root)).replace("\\", "/")
        except Exception:
            return path.name
    return f"{split}/synthetic/{int(index):08d}.png"


class SyntheticImageNet100(Dataset):
    def __init__(self, *, split: str, image_size: int, classes: int, samples_per_class: int, seed: int) -> None:
        self.split = split
        self.image_size = int(image_size)
        self.classes = [f"synset_{idx:03d}" for idx in range(int(classes))]
        self.targets = [cls for cls in range(int(classes)) for _ in range(int(samples_per_class))]
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        index = int(index)
        cls = int(self.targets[index])
        seed = stable_int_seed("imagenet100_synthetic", self.split, index, cls, self.seed)
        gen = torch.Generator().manual_seed(seed)
        base = torch.rand((3, self.image_size, self.image_size), generator=gen)
        tint = torch.tensor(
            [((cls * 37) % 255) / 255.0, ((cls * 73) % 255) / 255.0, ((cls * 109) % 255) / 255.0],
            dtype=torch.float32,
        ).view(3, 1, 1)
        x = (0.72 * base + 0.28 * tint).clamp(0.0, 1.0)
        return x, cls


def build_dataset(cfg: dict[str, Any], *, split: str, smoke: bool) -> Dataset:
    ds_cfg = cfg["dataset"]
    image_size = int(ds_cfg["image_size"])
    if smoke or str(ds_cfg.get("type", "imagefolder")) == "synthetic":
        samples_per_class = int(cfg.get("smoke", {}).get("samples_per_class", 3))
        classes = int(cfg.get("smoke", {}).get("classes", min(5, int(ds_cfg.get("num_classes", 100)))))
        return SyntheticImageNet100(
            split=split,
            image_size=image_size,
            classes=classes,
            samples_per_class=samples_per_class,
            seed=int(cfg["seeds"]["dataset_seed"]),
        )
    root = Path(str(ds_cfg["root"]))
    split_root = root / ("train" if split == "train" else "val")
    if not split_root.is_dir():
        raise FileNotFoundError(f"Missing ImageNet-100 {split} split: {split_root}")
    transform = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size), transforms.ToTensor()])
    return datasets.ImageFolder(split_root, transform=transform)


def split_class_counts(dataset: Dataset) -> dict[int, int]:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = [int(dataset[idx][1]) for idx in range(len(dataset))]
    values, counts = np.unique(np.asarray(targets, dtype=np.int64), return_counts=True)
    return {int(k): int(v) for k, v in zip(values, counts)}


def deterministic_corruption(
    x_clean: torch.Tensor,
    *,
    dataset_name: str,
    split: str,
    original_index: int,
    corpus_seed: int,
    corruption_version: str,
    psnr_min: float,
    psnr_max: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    # Seed corpus noise by image identity so loader order does not affect it.
    sample_seed = stable_int_seed(dataset_name, split, int(original_index), int(corpus_seed), corruption_version)
    generator = torch.Generator().manual_seed(sample_seed)
    psnr_uniform = torch.rand((), generator=generator).item()
    target_psnr_db = float(psnr_min + (psnr_max - psnr_min) * psnr_uniform)
    # Set sigma from target PSNR; clipping can change the measured PSNR.
    sigma = float(10.0 ** (-target_psnr_db / 20.0))
    base_noise = torch.randn(x_clean.shape, generator=generator, dtype=x_clean.dtype)
    x_preclip = x_clean + sigma * base_noise
    x_corpus = x_preclip.clamp(0.0, 1.0).to(torch.float32)
    return x_corpus, {
        "target_psnr_db": target_psnr_db,
        "sigma": sigma,
        "sample_seed": int(sample_seed),
        **metric_row(x_clean=x_clean, x_preclip=x_preclip, x_corpus=x_corpus),
    }


def ladder_corruption(
    x_clean: torch.Tensor,
    *,
    dataset_name: str,
    split: str,
    original_index: int,
    ladder_seed: int,
    psnr_levels: Iterable[float],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    seed = stable_int_seed(dataset_name, "ladder", split, int(original_index), int(ladder_seed))
    gen = torch.Generator().manual_seed(seed)
    base_noise = torch.randn(x_clean.shape, generator=gen, dtype=x_clean.dtype)
    images = []
    rows = []
    for level_pos, psnr_db in enumerate(psnr_levels):
        sigma = float(10.0 ** (-float(psnr_db) / 20.0))
        x_preclip = x_clean + sigma * base_noise
        x_corpus = x_preclip.clamp(0.0, 1.0).to(torch.float32)
        images.append(x_corpus)
        rows.append(
            {
                "level_pos": int(level_pos),
                "target_psnr_db": float(psnr_db),
                "sigma": sigma,
                "base_noise_seed": int(seed),
                **metric_row(x_clean=x_clean, x_preclip=x_preclip, x_corpus=x_corpus),
            }
        )
    return torch.stack(images, dim=0), rows


class OnDemandCorpusDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        manifest: pd.DataFrame,
        *,
        original_indices: Iterable[int] | None = None,
        normalize: bool = False,
        return_index: bool = False,
    ) -> None:
        self.dataset = dataset
        self.manifest = manifest.copy()
        self.rows = {
            int(row["original_index"]): row.to_dict()
            for _, row in self.manifest.iterrows()
        }
        self.original_indices = (
            self.manifest["original_index"].astype(int).to_numpy()
            if original_indices is None
            else np.asarray(list(original_indices), dtype=np.int64)
        )
        self.normalize = bool(normalize)
        self.return_index = bool(return_index)

    def __len__(self) -> int:
        return int(len(self.original_indices))

    def __getitem__(self, index: int):
        original_index = int(self.original_indices[int(index)])
        row = self.rows[original_index]
        x_clean, class_id = self.dataset[original_index]
        x_clean = x_clean.to(torch.float32).clamp(0.0, 1.0)
        psnr_min = float(row["target_psnr_db"])
        sigma = float(row["sigma"])
        sample_seed = int(row["sample_seed"])
        gen = torch.Generator().manual_seed(sample_seed)
        _ = torch.rand((), generator=gen).item()
        base_noise = torch.randn(x_clean.shape, generator=gen, dtype=x_clean.dtype)
        x = (x_clean + sigma * base_noise).clamp(0.0, 1.0).to(torch.float32)
        # Recompute sigma from target PSNR to check the noise manifest.
        expected_sigma = 10.0 ** (-psnr_min / 20.0)
        if abs(expected_sigma - sigma) > 1e-10:
            raise RuntimeError(f"Manifest sigma mismatch for original_index={original_index}")
        if self.normalize:
            x = x * 2.0 - 1.0
        if self.return_index:
            return x, original_index, int(class_id)
        return x


class CorpusTaskNoiseDataset(Dataset):
    def __init__(self, corpus: OnDemandCorpusDataset, *, cfg: dict[str, Any], variant: str) -> None:
        self.corpus = corpus
        self.cfg = cfg
        self.variant = variant
        self.noise_cfg = cfg["downstream"]["task_noise"]

    def __len__(self) -> int:
        return len(self.corpus)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        x_target, original_index, _class_id = self.corpus[index]
        seed = stable_int_seed(
            self.cfg["dataset"]["name"],
            "downstream_task_noise",
            int(original_index),
            int(self.cfg["seeds"]["task_noise_seed"]),
            str(self.noise_cfg["version"]),
        )
        gen = torch.Generator().manual_seed(seed)
        sigma_min = float(self.noise_cfg["sigma_min"])
        sigma_max = float(self.noise_cfg["sigma_max"])
        u = torch.rand((), generator=gen).item()
        sigma = float(sigma_min + (sigma_max - sigma_min) * u)
        noise = torch.randn(x_target.shape, generator=gen, dtype=x_target.dtype) * sigma
        x_input = (x_target + noise).clamp(0.0, 1.0).to(torch.float32)
        return x_input, x_target.to(torch.float32), torch.tensor(sigma, dtype=torch.float32), int(original_index)


class DownstreamValidationDataset(Dataset):
    def __init__(self, clean_dataset: Dataset, original_indices: np.ndarray, cfg: dict[str, Any], *, split: str) -> None:
        self.clean_dataset = clean_dataset
        self.original_indices = np.asarray(original_indices, dtype=np.int64)
        self.cfg = cfg
        self.split = split
        self.noise_cfg = cfg["downstream"]["task_noise"]

    def __len__(self) -> int:
        return int(len(self.original_indices))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        original_index = int(self.original_indices[int(index)])
        x_clean, _class_id = self.clean_dataset[original_index]
        x_clean = x_clean.to(torch.float32).clamp(0.0, 1.0)
        corpus_cfg = self.cfg["corpus"]
        x_target, _metrics = deterministic_corruption(
            x_clean,
            dataset_name=str(self.cfg["dataset"]["name"]),
            split=f"{self.split}_corpus",
            original_index=original_index,
            corpus_seed=int(self.cfg["seeds"]["corpus_seed"]),
            corruption_version=str(corpus_cfg["version"]),
            psnr_min=float(corpus_cfg["target_psnr_min_db"]),
            psnr_max=float(corpus_cfg["target_psnr_max_db"]),
        )
        seed = stable_int_seed(
            self.cfg["dataset"]["name"],
            f"{self.split}_task_noise",
            int(original_index),
            int(self.cfg["seeds"]["task_noise_seed"]),
            str(self.noise_cfg["version"]),
        )
        gen = torch.Generator().manual_seed(seed)
        sigma_min = float(self.noise_cfg["sigma_min"])
        sigma_max = float(self.noise_cfg["sigma_max"])
        sigma = float(sigma_min + (sigma_max - sigma_min) * torch.rand((), generator=gen).item())
        noise = torch.randn(x_target.shape, generator=gen, dtype=x_target.dtype) * sigma
        x_input = (x_target + noise).clamp(0.0, 1.0).to(torch.float32)
        return x_input, x_target.to(torch.float32), torch.tensor(sigma, dtype=torch.float32), int(original_index)


class EvalNoiseDataset(Dataset):
    def __init__(self, input_path: Path, target_path: Path) -> None:
        self.inputs = torch.load(input_path, map_location="cpu").to(torch.float32)
        self.targets = torch.load(target_path, map_location="cpu").to(torch.float32)

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[int(index)], self.targets[int(index)]


class ImageSpaceDDPM(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        image_size: int,
        base_dim: int,
        depth: int,
        n_steps: int,
        beta_start: float,
        beta_end: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        if image_size % (2**depth) != 0:
            raise ValueError(f"image_size={image_size} must be divisible by 2**depth={2**depth}")
        self.n_steps = int(n_steps)
        self.prediction_type = "epsilon"
        self.model = UNet(
            in_channels=in_channels,
            out_channels=in_channels,
            base_dim=base_dim,
            time_dim=128,
            kernel_sizes=[3] * int(depth),
            strides=[2] * int(depth),
        ).to(device)
        self.register_buffer("betas", torch.linspace(float(beta_start), float(beta_end), int(n_steps), device=device))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))

    @property
    def device(self) -> torch.device:
        return self.betas.device

    def q_sample(self, x0_norm: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        return torch.sqrt(alpha_bar) * x0_norm + torch.sqrt(1.0 - alpha_bar) * noise

    def training_loss(self, x0_norm: torch.Tensor) -> torch.Tensor:
        batch = int(x0_norm.shape[0])
        t = torch.randint(0, self.n_steps, (batch,), dtype=torch.long, device=x0_norm.device)
        noise = torch.randn_like(x0_norm)
        xt = self.q_sample(x0_norm, t, noise)
        pred = self.model(xt, t)
        return F.mse_loss(pred, noise)


def build_ddpm(cfg: dict[str, Any], device: torch.device) -> ImageSpaceDDPM:
    ddpm = cfg["ddpm"]
    ds = cfg["dataset"]
    if str(ddpm.get("prediction_type", "epsilon")) != "epsilon":
        raise ValueError("ImageNet image-space branch requires DDPM prediction_type=epsilon")
    use_smoke_arch = bool(cfg.get("_smoke_runtime", False))
    return ImageSpaceDDPM(
        in_channels=int(ds["channels"]),
        image_size=int(ds["image_size"]),
        base_dim=int(ddpm.get("base_dim_smoke" if use_smoke_arch else "base_dim", ddpm["base_dim"])),
        depth=int(ddpm.get("depth_smoke" if use_smoke_arch else "depth", ddpm["depth"])),
        n_steps=int(ddpm["T"]),
        beta_start=float(ddpm["beta_start"]),
        beta_end=float(ddpm["beta_end"]),
        device=device,
    )


def normalize_for_ddpm(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32) * 2.0 - 1.0


def denormalize_from_ddpm(x: torch.Tensor) -> torch.Tensor:
    return (x.to(torch.float32) + 1.0) * 0.5


def timestep_grid(cfg: dict[str, Any], model: ImageSpaceDDPM) -> list[dict[str, Any]]:
    T = int(model.n_steps)
    requested = [int(x) for x in cfg["score_validation"]["timestep_scan"]["report_timesteps"]]
    rows = []
    for report_t in requested:
        if T == 1000:
            code_index = int(report_t) - 1
        else:
            rel = (int(report_t) - 1) / 999.0
            code_index = int(round(rel * (T - 1)))
        if code_index < 0 or code_index >= T:
            raise ValueError(f"Invalid timestep: report={report_t}, code_index={code_index}, T={T}")
        beta_t = float(model.betas[code_index].detach().cpu())
        alpha_t = float(model.alphas[code_index].detach().cpu())
        alpha_bar_t = float(model.alphas_cumprod[code_index].detach().cpu())
        rows.append(
            {
                "report_timestep": int(report_t),
                "code_timestep_index": int(code_index),
                "beta_t": beta_t,
                "alpha_t": alpha_t,
                "alpha_bar_t": alpha_bar_t,
                "sqrt_alpha_bar_t": float(math.sqrt(max(alpha_bar_t, 0.0))),
                "sqrt_one_minus_alpha_bar_t": float(math.sqrt(max(1.0 - alpha_bar_t, 0.0))),
                "effective_diffusion_noise": float(math.sqrt(max(1.0 - alpha_bar_t, 0.0))),
            }
        )
    return list({(row["report_timestep"], row["code_timestep_index"]): row for row in rows}.values())


def mc_seed_bank(cfg: dict[str, Any], *, k: int, group: int = 0) -> list[int]:
    return [
        stable_int_seed("imagenet100_image_space", "mc", int(cfg["seeds"]["forward_noise_seed"]), int(group), int(i))
        for i in range(int(k))
    ]


@torch.inference_mode()
def score_batch(
    model: ImageSpaceDDPM,
    x_01: torch.Tensor,
    *,
    original_indices: np.ndarray,
    code_timestep_index: int,
    seed_bank: list[int],
    paired_level_count: int | None = None,
    seed_group: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    device = model.device
    normalized_images = normalize_for_ddpm(x_01).to(device)
    batch = int(normalized_images.shape[0])
    timesteps = torch.full((batch,), int(code_timestep_index), dtype=torch.long, device=device)
    score_sum = torch.zeros(batch, device=device)
    squared_score_sum = torch.zeros(batch, device=device)
    for mc_pos, mc_seed in enumerate(seed_bank):
        noises = []
        for row_pos, original_index in enumerate(original_indices.tolist()):
            # Use the same image index to seed noise across paired levels.
            seed_index = int(original_index)
            seed = stable_int_seed(
                "imagenet100_image_space",
                "forward_noise",
                int(seed_index),
                int(code_timestep_index),
                int(mc_seed),
                int(mc_pos),
                int(seed_group),
            )
            generator = torch.Generator().manual_seed(seed)
            noises.append(torch.randn(normalized_images[row_pos].shape, generator=generator, dtype=normalized_images.dtype))
        noise = torch.stack(noises, dim=0).to(device)
        noisy_images = model.q_sample(normalized_images, timesteps, noise)
        predicted_noise = model.model(noisy_images, timesteps)
        score = predicted_noise.flatten(start_dim=1).pow(2).sum(dim=1).float()
        score_sum += score
        squared_score_sum += score.pow(2)
    # Average squared epsilon norms over MC draws and estimate spread from moments.
    mean = score_sum / float(len(seed_bank))
    variance = (squared_score_sum / float(len(seed_bank)) - mean.pow(2)).clamp_min(0.0)
    return mean.detach().cpu().numpy().astype(np.float64), torch.sqrt(variance).detach().cpu().numpy().astype(np.float64)


def score_batch_chunked(
    model: ImageSpaceDDPM,
    images: list[torch.Tensor],
    original_keys: list[int],
    *,
    code_timestep_index: int,
    seed_bank: list[int],
    batch_size: int,
    paired_level_count: int | None = None,
    seed_group: int = 0,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    scores: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    safe_batch = max(1, int(batch_size))
    total_chunks = int(math.ceil(len(images) / safe_batch)) if images else 0
    for chunk_idx, start in enumerate(range(0, len(images), safe_batch), start=1):
        end = min(start + safe_batch, len(images))
        chunk_scores, chunk_stds = score_batch(
            model,
            torch.stack(images[start:end], dim=0),
            original_indices=np.asarray(original_keys[start:end], dtype=np.int64),
            code_timestep_index=code_timestep_index,
            seed_bank=seed_bank,
            paired_level_count=paired_level_count,
            seed_group=seed_group,
        )
        scores.append(chunk_scores)
        stds.append(chunk_stds)
        if progress_callback is not None:
            progress_callback(chunk_idx, total_chunks, end)
    return np.concatenate(scores).astype(np.float64), np.concatenate(stds).astype(np.float64)


def oracle_score_from_quality(df: pd.DataFrame) -> np.ndarray:
    return df["sigma"].to_numpy(dtype=np.float64)


def class_stratified_indices(dataset: Dataset, *, per_class: int, seed: int) -> np.ndarray:
    targets = np.asarray(getattr(dataset, "targets", [int(dataset[idx][1]) for idx in range(len(dataset))]), dtype=np.int64)
    rng = np.random.default_rng(seed)
    parts = []
    for cls in sorted(np.unique(targets).astype(int).tolist()):
        idx = np.flatnonzero(targets == cls).astype(np.int64, copy=True)
        rng.shuffle(idx)
        parts.append(np.sort(idx[: min(int(per_class), len(idx))]))
    return np.sort(np.concatenate(parts).astype(np.int64, copy=False))


def class_stratified_indices_excluding(dataset: Dataset, *, per_class: int, seed: int, exclude: Iterable[int]) -> np.ndarray:
    targets = np.asarray(getattr(dataset, "targets", [int(dataset[idx][1]) for idx in range(len(dataset))]), dtype=np.int64)
    excluded = np.asarray(sorted(set(int(x) for x in exclude)), dtype=np.int64)
    rng = np.random.default_rng(seed)
    parts = []
    for cls in sorted(np.unique(targets).astype(int).tolist()):
        idx = np.flatnonzero(targets == cls).astype(np.int64, copy=True)
        if len(excluded):
            idx = idx[~np.isin(idx, excluded)]
        rng.shuffle(idx)
        parts.append(np.sort(idx[: min(int(per_class), len(idx))]))
    return np.sort(np.concatenate(parts).astype(np.int64, copy=False)) if parts else np.asarray([], dtype=np.int64)


def class_stratified_total_indices(dataset: Dataset, *, total: int, seed: int) -> np.ndarray:
    targets = np.asarray(getattr(dataset, "targets", [int(dataset[idx][1]) for idx in range(len(dataset))]), dtype=np.int64)
    classes = sorted(np.unique(targets).astype(int).tolist())
    if not classes:
        return np.asarray([], dtype=np.int64)
    per_class = max(1, int(math.ceil(int(total) / len(classes))))
    selected = class_stratified_indices(dataset, per_class=per_class, seed=seed)
    return np.sort(selected[: min(int(total), len(selected))].astype(np.int64, copy=False))


def resolved_workload(cfg: dict[str, Any], *, actual_train_samples: int | None = None, actual_val_samples: int | None = None) -> dict[str, Any]:
    num_classes = int(cfg["dataset"]["num_classes"])
    train_samples = int(actual_train_samples or cfg["dataset"].get("expected_train_samples", 0))
    val_samples = int(actual_val_samples or cfg["dataset"].get("expected_val_samples", 0))
    ladder_per_class = int(cfg["score_validation"]["samples_per_class"])
    ladder_images = ladder_per_class * num_classes
    val_corpus_per_class = int(cfg["score_validation"].get("validation_corpus_samples_per_class", 0))
    val_corpus_total = int(cfg["score_validation"].get("validation_corpus_samples", val_corpus_per_class * num_classes))
    epochs = cfg["score_validation"]["checkpoint_scan"]["epochs"]
    if epochs == "all":
        coarse_epochs = list(range(1, int(cfg["ddpm"]["epochs"]) + 1))
    else:
        coarse_epochs = [int(x) for x in epochs]
    timesteps = [int(x) for x in cfg["score_validation"]["timestep_scan"]["report_timesteps"]]
    coarse_k = int(cfg["score_validation"]["mc"]["coarse_k"])
    final_k = int(cfg["score_validation"]["mc"]["final_k"])
    final_candidates = int(cfg["score_validation"]["checkpoint_scan"].get("top_candidates", 3))
    ladder_levels = len(cfg["score_validation"]["psnr_ladder"])
    coarse = ladder_images * ladder_levels * len(timesteps) * coarse_k * len(coarse_epochs)
    final_ladder = ladder_images * ladder_levels * final_k * final_candidates
    final_validation_corpus = val_corpus_total * final_k * final_candidates
    additional_coarse_validation_corpus = val_corpus_total * coarse_k * len(timesteps) * len(coarse_epochs)
    return {
        "dataset": {
            "train_samples": train_samples,
            "val_samples": val_samples,
            "num_classes": num_classes,
        },
        "score_validation": {
            "ladder_samples_total": ladder_images,
            "ladder_samples_per_class": ladder_per_class,
            "validation_corpus_samples_total": val_corpus_total,
            "validation_corpus_samples_per_class": val_corpus_per_class,
            "psnr_ladder_levels": ladder_levels,
        },
        "ddpm": {
            "epochs": int(cfg["ddpm"]["epochs"]),
            "checkpoint_interval": int(cfg["ddpm"]["checkpoint_interval"]),
            "coarse_checkpoint_epochs": coarse_epochs,
            "num_coarse_checkpoints": len(coarse_epochs),
            "prediction_type": cfg["ddpm"]["prediction_type"],
            "T": int(cfg["ddpm"]["T"]),
            "beta_schedule": cfg["ddpm"]["beta_schedule"],
            "batch_size": int(cfg["ddpm"]["batch_size"]),
        },
        "scan": {
            "final_candidate_count": final_candidates,
            "timestep_grid": timesteps,
            "coarse_mc_k": coarse_k,
            "final_mc_k": final_k,
        },
        "downstream": {
            "tdncnn_max_optimizer_steps": int(cfg["downstream"]["max_steps"]),
            "tdncnn_batch_size": int(cfg["downstream"]["batch_size"]),
            "drunet_max_optimizer_steps": int(cfg.get("drunet", cfg["downstream"])["max_steps"]),
            "drunet_batch_size": int(cfg.get("drunet", cfg["downstream"])["batch_size"]),
        },
        "batch_sizes": {
            "ddpm": int(cfg["ddpm"]["batch_size"]),
            "score_validation": int(cfg["score_validation"]["batch_size"]),
            "filtering": int(cfg["filtering"]["batch_size"]),
            "tdncnn": int(cfg["downstream"]["batch_size"]),
            "drunet": int(cfg.get("drunet", cfg["downstream"])["batch_size"]),
            "evaluation": int(cfg["evaluation"]["batch_size"]),
        },
        "forward_evaluations": {
            "coarse_formula": "N_ladder_images * 7 * N_timesteps * K_coarse * N_coarse_checkpoints",
            "coarse": int(coarse),
            "final_ladder_formula": "N_final_ladder_images * 7 * K_final * N_final_candidates",
            "final_ladder": int(final_ladder),
            "final_validation_corpus_formula": "N_validation_corpus * K_final * N_final_candidates",
            "final_validation_corpus": int(final_validation_corpus),
            "requested_total": int(coarse + final_ladder + final_validation_corpus),
            "additional_coarse_validation_corpus_current_implementation": int(additional_coarse_validation_corpus),
            "implementation_total_including_coarse_validation_corpus": int(
                coarse + final_ladder + final_validation_corpus + additional_coarse_validation_corpus
            ),
        },
    }


def write_workload_audit(ctx: PipelineContext, *, actual_train_samples: int | None = None, actual_val_samples: int | None = None) -> dict[str, Any]:
    workload = resolved_workload(ctx.cfg, actual_train_samples=actual_train_samples, actual_val_samples=actual_val_samples)
    out_json = ctx.experiment_root / "00_protocol" / "resolved_production_workload.json"
    write_json(out_json, workload)
    lines = [
        "# Resolved Production Workload",
        "",
        f"- train samples: {workload['dataset']['train_samples']}",
        f"- val samples: {workload['dataset']['val_samples']}",
        f"- ladder images: {workload['score_validation']['ladder_samples_total']} ({workload['score_validation']['ladder_samples_per_class']} per class)",
        f"- validation corpus: {workload['score_validation']['validation_corpus_samples_total']} ({workload['score_validation']['validation_corpus_samples_per_class']} per class)",
        f"- DDPM epochs: {workload['ddpm']['epochs']}",
        f"- checkpoint interval: {workload['ddpm']['checkpoint_interval']}",
        f"- coarse checkpoint epochs: {workload['ddpm']['coarse_checkpoint_epochs']}",
        f"- coarse checkpoints: {workload['ddpm']['num_coarse_checkpoints']}",
        f"- final candidates: {workload['scan']['final_candidate_count']}",
        f"- timestep grid: {workload['scan']['timestep_grid']}",
        f"- coarse MC K: {workload['scan']['coarse_mc_k']}",
        f"- final MC K: {workload['scan']['final_mc_k']}",
        f"- TDnCNN downstream max optimizer steps: {workload['downstream']['tdncnn_max_optimizer_steps']}",
        f"- DRUNet downstream max optimizer steps: {workload['downstream']['drunet_max_optimizer_steps']}",
        f"- batch sizes: {workload['batch_sizes']}",
        "",
        "## DDPM Forward Evaluations",
        f"- coarse: {workload['forward_evaluations']['coarse']}",
        f"- final ladder: {workload['forward_evaluations']['final_ladder']}",
        f"- final validation corpus: {workload['forward_evaluations']['final_validation_corpus']}",
        f"- requested total: {workload['forward_evaluations']['requested_total']}",
        f"- additional coarse validation corpus in current implementation: {workload['forward_evaluations']['additional_coarse_validation_corpus_current_implementation']}",
        f"- implementation total including coarse validation corpus: {workload['forward_evaluations']['implementation_total_including_coarse_validation_corpus']}",
    ]
    (ctx.experiment_root / "00_protocol" / "resolved_production_workload.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return workload


def paired_metrics(score_df: pd.DataFrame, *, bootstrap_seed: int, bootstrap_iters: int) -> tuple[dict[str, float], pd.DataFrame]:
    rho_rows = []
    order_total = 0
    order_correct = 0
    adjacent_total = 0
    adjacent_correct = 0
    for original_index, group in score_df.groupby("original_index", sort=False):
        group = group.sort_values("sigma")
        rho = group["score"].corr(group["sigma"], method="spearman")
        rho_rows.append(
            {
                "original_index": int(original_index),
                "class_id": int(group["class_id"].iloc[0]),
                "within_image_spearman": float(rho) if pd.notna(rho) else float("nan"),
            }
        )
        arr = group[["sigma", "score"]].to_numpy(dtype=np.float64)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[0]):
                if arr[i, 0] > arr[j, 0]:
                    order_total += 1
                    order_correct += int(arr[i, 1] > arr[j, 1])
        for i in range(1, arr.shape[0]):
            adjacent_total += 1
            adjacent_correct += int(arr[i, 1] > arr[i - 1, 1])
    rho_df = pd.DataFrame(rho_rows)
    values = rho_df["within_image_spearman"].dropna().to_numpy(dtype=np.float64)
    rng = np.random.default_rng(int(bootstrap_seed))
    boot = []
    if len(values):
        for _ in range(int(bootstrap_iters)):
            boot.append(float(np.median(values[rng.integers(0, len(values), size=len(values))])))
    ci_low, ci_high = (np.percentile(boot, [2.5, 97.5]).tolist() if boot else [float("nan"), float("nan")])
    metrics = {
        "median_within_image_spearman": float(np.nanmedian(values)) if len(values) else float("nan"),
        "mean_within_image_spearman": float(np.nanmean(values)) if len(values) else float("nan"),
        "iqr_within_image_spearman": float(np.nanpercentile(values, 75) - np.nanpercentile(values, 25)) if len(values) else float("nan"),
        "fraction_images_rho_gt_0": float(np.nanmean(values > 0.0)) if len(values) else float("nan"),
        "bootstrap_ci_low": float(ci_low),
        "bootstrap_ci_high": float(ci_high),
        "pairwise_ordering_accuracy": float(order_correct / max(order_total, 1)),
        "adjacent_ordering_accuracy": float(adjacent_correct / max(adjacent_total, 1)),
    }
    return metrics, rho_df


def rank_positions(df: pd.DataFrame) -> pd.Series:
    ranked = df.sort_values(["score", "original_index"], ascending=[True, True]).reset_index(drop=True)
    return pd.Series(np.arange(1, len(ranked) + 1, dtype=np.int64), index=ranked["original_index"].astype(int))


def global_metrics(score_df: pd.DataFrame, stable_df: pd.DataFrame | None, *, topk_ratio: float) -> dict[str, Any]:
    df = score_df.copy()
    n_top = max(1, min(int(round(float(topk_ratio) * len(df))), len(df)))
    topk = df.sort_values(["score", "original_index"], ascending=[True, True]).head(n_top)
    classes = sorted(int(item) for item in df["class_id"].unique())
    selected_counts = topk["class_id"].value_counts().reindex(classes, fill_value=0)
    full_counts = df["class_id"].value_counts().reindex(classes, fill_value=0)
    rng = np.random.default_rng(0)
    random_indices = rng.choice(np.sort(df["original_index"].to_numpy(dtype=np.int64)), size=n_top, replace=False)
    random_df = df[df["original_index"].isin(set(random_indices.tolist()))]
    random_counts = random_df["class_id"].value_counts().reindex(classes, fill_value=0)
    p = full_counts.to_numpy(dtype=np.float64) / max(len(df), 1)
    q = selected_counts.to_numpy(dtype=np.float64) / max(len(topk), 1)
    r = random_counts.to_numpy(dtype=np.float64) / max(len(random_df), 1)
    class_bias = float(np.max(np.abs(q - p))) if len(p) else 0.0
    m_top = 0.5 * (p + q)
    m_random = 0.5 * (p + r)
    top_js = float(0.5 * np.sum(p * np.log((p + EPS) / (m_top + EPS))) + 0.5 * np.sum(q * np.log((q + EPS) / (m_top + EPS))))
    random_js = float(0.5 * np.sum(p * np.log((p + EPS) / (m_random + EPS))) + 0.5 * np.sum(r * np.log((r + EPS) / (m_random + EPS))))
    metrics: dict[str, Any] = {
        "global_spearman_score_target_psnr": float(df["score"].corr(df["target_psnr_db"], method="spearman")),
        "global_spearman_score_realized_psnr": float(df["score"].corr(df["realized_psnr_db_postclip"], method="spearman")),
        "global_spearman_score_realized_snr": float(df["score"].corr(df["realized_snr_db_postclip"], method="spearman")),
        "global_spearman_score_sigma": float(df["score"].corr(df["sigma"], method="spearman")),
        "global_spearman_score_corruption_mse": float(df["score"].corr(df["corruption_mse"], method="spearman")),
        "topk_median_realized_psnr_shift": float(topk["realized_psnr_db_postclip"].median() - df["realized_psnr_db_postclip"].median()),
        "topk_target_psnr_shift": float(topk["target_psnr_db"].median() - df["target_psnr_db"].median()),
        "topk_realized_snr_shift": float(topk["realized_snr_db_postclip"].median() - df["realized_snr_db_postclip"].median()),
        "class_coverage": float((selected_counts > 0).mean()) if len(selected_counts) else 0.0,
        "random_10_class_coverage": float((random_counts > 0).mean()) if len(random_counts) else 0.0,
        "class_distribution_max_abs_deviation": class_bias,
        "class_distribution_js_divergence": top_js,
        "random_10_class_distribution_js_divergence": random_js,
        "per_class_selected_count": {str(int(k)): int(v) for k, v in selected_counts.items()},
    }
    if stable_df is not None and len(stable_df):
        n = len(df)
        n_top = max(1, min(int(round(float(topk_ratio) * n)), n))
        a = set(df.sort_values(["score", "original_index"], ascending=[True, True]).head(n_top)["original_index"].astype(int))
        b = set(stable_df.sort_values(["score", "original_index"], ascending=[True, True]).head(n_top)["original_index"].astype(int))
        ranks_a = rank_positions(df)
        ranks_b = rank_positions(stable_df)
        common = ranks_a.index.intersection(ranks_b.index)
        metrics["topk_jaccard"] = float(len(a & b) / max(len(a | b), 1))
        metrics["score_rank_stability_spearman"] = float(ranks_a.loc[common].corr(ranks_b.loc[common], method="spearman"))
    else:
        metrics["topk_jaccard"] = float("nan")
        metrics["score_rank_stability_spearman"] = float("nan")
    return metrics


@dataclass
class PipelineContext:
    cfg: dict[str, Any]
    config_path: Path
    run_id: str
    experiment_root: Path
    smoke: bool
    real_smoke: bool
    dry_run: bool
    resume: bool

    @property
    def cfg_hash(self) -> str:
        return config_hash({**self.cfg, "run_id": self.run_id})

    def stage_root(self, stage: str) -> Path:
        return self.experiment_root / STAGE_DIRS[stage]

    def marker_path(self, stage: str) -> Path:
        return self.stage_root(stage) / "_SUCCESS.json"

    def marker_valid(self, stage: str, required: list[Path]) -> bool:
        marker = self.marker_path(stage)
        if not marker.exists():
            return False
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            return False
        if payload.get("run_id") != self.run_id or payload.get("config_hash") != self.cfg_hash:
            return False
        return all(path.exists() for path in required)

    def write_marker(self, stage: str, start_time: float, outputs: list[Path], inputs: dict[str, str] | None = None) -> None:
        payload = {
            "stage": stage,
            "run_id": self.run_id,
            "start_time": start_time,
            "end_time": time.time(),
            "command": " ".join(sys.argv),
            "config_hash": self.cfg_hash,
            "input_artifact_hashes": inputs or {},
            "output_artifact_list": [str(path) for path in outputs],
        }
        write_json(self.marker_path(stage), payload)


def protocol_dirs(ctx: PipelineContext) -> None:
    for rel in [
        "00_protocol",
        "01_corpus",
        "02_ddpm/checkpoints",
        "03_score_validation",
        "04_selected_score_config",
        "05_filtering/subset_indices",
        "05_filtering/subset_manifests",
        "05_filtering/plots",
        "05_filtering/panels",
        "05_filtering/p10",
        "05_filtering/p20",
        "05_filtering/p30",
        "06_downstream/tdncnn",
        "06_downstream/drunet",
        "07_evaluation/tdncnn",
        "07_evaluation/drunet",
        "07_evaluation/combined",
        "08_figures",
        "08_external_benchmarks/tdncnn",
        "08_external_benchmarks/drunet",
        "08_external_benchmarks/combined",
        "09_summary",
    ]:
        (ctx.experiment_root / rel).mkdir(parents=True, exist_ok=True)


def stage_prepare(ctx: PipelineContext) -> None:
    stage = "prepare"
    start = time.time()
    manifest_path = ctx.stage_root(stage) / "corpus_manifest.csv"
    if ctx.resume and ctx.marker_valid(stage, [manifest_path]):
        print(f"[resume] prepare already complete: {manifest_path}")
        return
    if ctx.dry_run:
        print(f"[dry-run] would prepare deterministic variable-AWGN corpus manifest at {manifest_path}")
        return
    protocol_dirs(ctx)
    train_ds = build_dataset(ctx.cfg, split="train", smoke=ctx.smoke)
    val_ds = build_dataset(ctx.cfg, split="val", smoke=ctx.smoke)
    expected_classes = int(ctx.cfg["dataset"]["num_classes"])
    actual_classes = len(getattr(train_ds, "classes", split_class_counts(train_ds)))
    if not ctx.smoke and actual_classes != expected_classes and not bool(ctx.cfg["dataset"].get("allow_count_mismatch", False)):
        raise RuntimeError(f"Expected {expected_classes} ImageNet-100 classes, found {actual_classes}")
    if ctx.real_smoke:
        original_indices = class_stratified_indices(
            train_ds,
            per_class=int(ctx.cfg["real_smoke"]["train_samples_per_class"]),
            seed=int(ctx.cfg["seeds"]["split_seed"]),
        )
    else:
        original_indices = np.arange(len(train_ds), dtype=np.int64)
    rows = []
    noise_cfg = ctx.cfg["corpus"]
    for original_index in tqdm(original_indices.tolist(), desc="Preparing corpus manifest"):
        x_clean, class_id = train_ds[int(original_index)]
        x_clean = x_clean.to(torch.float32).clamp(0.0, 1.0)
        _x_corpus, metrics = deterministic_corruption(
            x_clean,
            dataset_name=str(ctx.cfg["dataset"]["name"]),
            split="train",
            original_index=original_index,
            corpus_seed=int(ctx.cfg["seeds"]["corpus_seed"]),
            corruption_version=str(noise_cfg["version"]),
            psnr_min=float(noise_cfg["target_psnr_min_db"]),
            psnr_max=float(noise_cfg["target_psnr_max_db"]),
        )
        rows.append(
            {
                "dataset": ctx.cfg["dataset"]["name"],
                "split": "train",
                "original_index": int(original_index),
                "relative_path": sample_identifier(train_ds, original_index, "train"),
                "class_id": int(class_id),
                "class_name": class_name_from_dataset(train_ds, int(class_id)),
                **metrics,
                "corruption_type": "variable_awgn_quality",
                "corruption_version": str(noise_cfg["version"]),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path, index=False)
    quality = manifest[
        ["target_psnr_db", "realized_psnr_db_postclip", "realized_snr_db_postclip", "sigma", "corruption_mse", "clipped_fraction"]
    ].describe(percentiles=[0.1, 0.5, 0.9])
    quality.to_csv(ctx.stage_root(stage) / "quality_summary.csv")
    checks = deterministic_example_checks(ctx.cfg, train_ds, manifest, ctx.smoke)
    write_json(ctx.stage_root(stage) / "deterministic_example_checks.json", checks)
    protocol = {
        "run_id": ctx.run_id,
        "config_path": str(ctx.config_path),
        "config_hash": ctx.cfg_hash,
        "git_commit": git_commit(),
        "git_status_short": git_status_short(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
        "dataset_paths": ctx.cfg["dataset"],
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "manifest_samples": int(len(manifest)),
        "train_class_counts": split_class_counts(train_ds),
        "val_class_counts": split_class_counts(val_ds),
        "image_range": "[0,1] corpus, DDPM normalization to [-1,1]",
        "transforms": "Resize + CenterCrop + ToTensor for ImageFolder; deterministic synthetic fixture in smoke",
    }
    save_yaml(ctx.experiment_root / "00_protocol" / "resolved_config.yaml", ctx.cfg)
    write_workload_audit(ctx, actual_train_samples=len(train_ds), actual_val_samples=len(val_ds))
    write_json(ctx.experiment_root / "00_protocol" / "protocol_summary.json", protocol)
    (ctx.experiment_root / "00_protocol" / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (ctx.experiment_root / "00_protocol" / "git_status.txt").write_text(protocol["git_status_short"], encoding="utf-8")
    ctx.write_marker(stage, start, [manifest_path, ctx.stage_root(stage) / "quality_summary.csv"])


def deterministic_example_checks(cfg: dict[str, Any], train_ds: Dataset, manifest: pd.DataFrame, smoke: bool) -> dict[str, Any]:
    indices = manifest["original_index"].head(min(3, len(manifest))).astype(int).tolist()
    rows = {}
    for original_index in indices:
        x_clean, _ = train_ds[original_index]
        a, ma = deterministic_corruption(
            x_clean,
            dataset_name=str(cfg["dataset"]["name"]),
            split="train",
            original_index=original_index,
            corpus_seed=int(cfg["seeds"]["corpus_seed"]),
            corruption_version=str(cfg["corpus"]["version"]),
            psnr_min=float(cfg["corpus"]["target_psnr_min_db"]),
            psnr_max=float(cfg["corpus"]["target_psnr_max_db"]),
        )
        b, mb = deterministic_corruption(
            x_clean,
            dataset_name=str(cfg["dataset"]["name"]),
            split="train",
            original_index=original_index,
            corpus_seed=int(cfg["seeds"]["corpus_seed"]),
            corruption_version=str(cfg["corpus"]["version"]),
            psnr_min=float(cfg["corpus"]["target_psnr_min_db"]),
            psnr_max=float(cfg["corpus"]["target_psnr_max_db"]),
        )
        rows[str(original_index)] = {
            "tensor_equal": bool(torch.equal(a, b)),
            "target_psnr_equal": bool(ma["target_psnr_db"] == mb["target_psnr_db"]),
            "sample_seed": int(ma["sample_seed"]),
        }
    return {"smoke": bool(smoke), "checks": rows}


def stage_ddpm_train(ctx: PipelineContext) -> None:
    stage = "ddpm_train"
    start = time.time()
    ckpt_dir = ctx.stage_root(stage) / "checkpoints"
    final_epoch = int(ctx.cfg["ddpm"]["epochs_smoke" if ctx.smoke else "epochs"])
    last_ckpt = ckpt_dir / f"epoch_{final_epoch:03d}.pt"
    if ctx.resume and ctx.marker_valid(stage, [last_ckpt]):
        print(f"[resume] ddpm_train already complete: {last_ckpt}")
        return
    if ctx.dry_run:
        print(f"[dry-run] would train image-space DDPM for {final_epoch} epochs into {ckpt_dir}")
        return
    manifest = pd.read_csv(ctx.stage_root("prepare") / "corpus_manifest.csv")
    train_ds = build_dataset(ctx.cfg, split="train", smoke=ctx.smoke)
    if ctx.smoke:
        max_samples = int(ctx.cfg.get("smoke", {}).get("ddpm_samples", len(manifest)))
    elif ctx.real_smoke:
        max_samples = int(ctx.cfg["real_smoke"]["ddpm_train_samples"])
    else:
        max_samples = len(manifest)
    train_indices = manifest["original_index"].head(max_samples).astype(int).to_numpy()
    corpus = OnDemandCorpusDataset(train_ds, manifest, original_indices=train_indices, normalize=True, return_index=False)
    loader = DataLoader(
        corpus,
        batch_size=int(ctx.cfg["ddpm"]["batch_size_smoke" if ctx.smoke else "batch_size"]),
        shuffle=True,
        num_workers=0 if ctx.smoke else int(ctx.cfg["runtime"]["num_workers"]),
        generator=torch.Generator().manual_seed(int(ctx.cfg["seeds"]["model_seed"])),
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not ctx.smoke else "cpu")
    model = build_ddpm(ctx.cfg, device)
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=float(ctx.cfg["ddpm"]["lr"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(ctx.cfg["ddpm"].get("amp", True)) and device.type == "cuda")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    global_step = 0
    max_optimizer_steps = int(ctx.cfg["real_smoke"]["ddpm_optimizer_steps"]) if ctx.real_smoke else None
    stop_training = False
    log_start = time.time()
    log_interval = int(ctx.cfg.get("ddpm", {}).get("log_interval_epochs", 1))
    for epoch in range(1, final_epoch + 1):
        model.train()
        losses = []
        for x_norm in loader:
            x_norm = x_norm.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
                loss = model.training_loss(x_norm)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            losses.append(float(loss.detach().cpu().item()))
            if max_optimizer_steps is not None and global_step >= max_optimizer_steps:
                stop_training = True
                break
        scheduler.step()
        row = {"epoch": epoch, "global_step": global_step, "train_loss": float(np.mean(losses))}
        rows.append(row)
        checkpoint = {
            "model_state_dict": model.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "config_hash": ctx.cfg_hash,
            "prediction_type": model.prediction_type,
        }
        torch.save(checkpoint, ckpt_dir / f"epoch_{epoch:03d}.pt")
        pd.DataFrame(rows).to_csv(ctx.stage_root(stage) / "training_history.csv", index=False)
        if should_log_progress(epoch, final_epoch, log_interval):
            training_progress(
                "ImageNet-100",
                "ddpm_train",
                unit="epoch",
                current=epoch,
                total=final_epoch,
                train_loss=row["train_loss"],
                lr=optimizer.param_groups[0]["lr"],
                start_time=log_start,
                extra={"global_step": global_step, "checkpoint": f"epoch_{epoch:03d}.pt"},
            )
        if stop_training:
            break
    reload_payload = torch.load(last_ckpt, map_location=device)
    reload_model = build_ddpm(ctx.cfg, device)
    reload_model.model.load_state_dict(reload_payload["model_state_dict"])
    write_json(
        ctx.stage_root(stage) / "checkpoint_reload_check.json",
        {"checkpoint": str(last_ckpt), "epoch": int(reload_payload["epoch"]), "global_step": int(reload_payload["global_step"]), "status": "ok"},
    )
    ctx.write_marker(stage, start, [last_ckpt, ctx.stage_root(stage) / "training_history.csv", ctx.stage_root(stage) / "checkpoint_reload_check.json"])


def available_checkpoint_epochs(ctx: PipelineContext) -> list[int]:
    ckpt_dir = ctx.stage_root("ddpm_train") / "checkpoints"
    found = []
    for path in sorted(ckpt_dir.glob("epoch_*.pt")):
        try:
            found.append(int(path.stem.split("_")[1]))
        except Exception:
            pass
    requested = ctx.cfg["score_validation"]["checkpoint_scan"]["epochs"]
    if requested == "all":
        return found
    return [int(e) for e in requested if int(e) in set(found)]


def load_ddpm_checkpoint(ctx: PipelineContext, epoch: int, device: torch.device) -> ImageSpaceDDPM:
    model = build_ddpm(ctx.cfg, device)
    path = ctx.stage_root("ddpm_train") / "checkpoints" / f"epoch_{int(epoch):03d}.pt"
    payload = torch.load(path, map_location=device)
    model.model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def stage_score_validate(ctx: PipelineContext) -> None:
    # Compare candidates using the same validation noise levels.
    stage = "score_validate"
    start = time.time()
    candidate_path = ctx.stage_root(stage) / "candidate_metrics.csv"
    if ctx.resume and ctx.marker_valid(stage, [candidate_path]):
        print(f"[resume] score_validate already complete: {candidate_path}")
        return
    if ctx.dry_run:
        print(f"[dry-run] would scan checkpoints/timesteps and write {candidate_path}")
        return
    stage_root = ctx.stage_root(stage)
    stage_root.mkdir(parents=True, exist_ok=True)
    progress_path = stage_root / "progress.jsonl"
    partial_candidate_path = stage_root / "candidate_metrics_partial.csv"
    raw_ladder_path = stage_root / "raw_ladder_scores.csv"
    raw_global_path = stage_root / "validation_corpus_scores.csv"
    for partial_path in [progress_path, partial_candidate_path, raw_ladder_path, raw_global_path]:
        partial_path.unlink(missing_ok=True)
    val_ds = build_dataset(ctx.cfg, split="val", smoke=ctx.smoke)
    train_ds = build_dataset(ctx.cfg, split="train", smoke=ctx.smoke)
    write_workload_audit(ctx, actual_train_samples=len(train_ds), actual_val_samples=len(val_ds))
    manifest = pd.read_csv(ctx.stage_root("prepare") / "corpus_manifest.csv")
    per_class = int(ctx.cfg["score_validation"]["samples_per_class_smoke" if ctx.smoke else "samples_per_class"])
    val_indices = class_stratified_indices(val_ds, per_class=per_class, seed=int(ctx.cfg["seeds"]["split_seed"]))
    eval_indices = class_stratified_indices(
        val_ds,
        per_class=per_class,
        seed=int(ctx.cfg["seeds"]["split_seed"]) + 1009,
    )
    downstream_val_per_class = int(ctx.cfg["downstream"].get("validation_samples_per_class", per_class))
    downstream_validation_indices = class_stratified_indices_excluding(
        val_ds,
        per_class=downstream_val_per_class,
        seed=int(ctx.cfg["seeds"]["split_seed"]) + 2003,
        exclude=[*val_indices.tolist(), *eval_indices.tolist()],
    )
    np.save(ctx.experiment_root / "00_protocol" / "score_validation_indices.npy", val_indices)
    np.save(ctx.experiment_root / "00_protocol" / "downstream_evaluation_indices.npy", eval_indices)
    np.save(ctx.experiment_root / "00_protocol" / "downstream_validation_indices.npy", downstream_validation_indices)
    write_json(
        stage_root / "subset_split_summary.json",
        {
            "score_validation_count": int(len(val_indices)),
            "downstream_validation_count": int(len(downstream_validation_indices)),
            "downstream_evaluation_count": int(len(eval_indices)),
            "overlap_count": int(len(set(val_indices.tolist()) & set(eval_indices.tolist()))),
            "score_validation_downstream_validation_overlap_count": int(len(set(val_indices.tolist()) & set(downstream_validation_indices.tolist()))),
            "downstream_validation_evaluation_overlap_count": int(len(set(downstream_validation_indices.tolist()) & set(eval_indices.tolist()))),
            "split_seed": int(ctx.cfg["seeds"]["split_seed"]),
        },
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not ctx.smoke else "cpu")
    probe_model = build_ddpm(ctx.cfg, device)
    grid = timestep_grid(ctx.cfg, probe_model)
    pd.DataFrame(grid).to_csv(stage_root / "resolved_timestep_grid.csv", index=False)
    epochs = available_checkpoint_epochs(ctx)
    if not epochs:
        raise RuntimeError("No DDPM checkpoints available for score validation")
    write_json(stage_root / "resolved_checkpoint_scan.json", {"epochs": epochs})
    total_coarse = len(epochs) * len(grid)
    score_validate_log(
        stage_root,
        "start",
        epochs=epochs,
        timesteps=[int(item["report_timestep"]) for item in grid],
        coarse_candidates=int(total_coarse),
        score_validation_images=int(len(val_indices)),
        validation_corpus_target=int(ctx.cfg["score_validation"]["validation_corpus_samples_smoke" if ctx.smoke else "validation_corpus_samples"]),
        coarse_mc_k=int(ctx.cfg["score_validation"]["mc"]["coarse_k"]),
        final_mc_k=int(ctx.cfg["score_validation"]["mc"]["final_k"]),
        device=str(device),
        cuda_allocated_mb=float(torch.cuda.memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        cuda_reserved_mb=float(torch.cuda.memory_reserved(device) / 1024**2) if device.type == "cuda" else 0.0,
    )
    rows = []
    use_oracle = bool(ctx.smoke and ctx.cfg.get("smoke", {}).get("oracle_scores", True))
    completed_coarse = 0
    for epoch_idx, epoch in enumerate(epochs, start=1):
        epoch_start = time.time()
        score_validate_log(stage_root, "load_checkpoint", phase="coarse", epoch=int(epoch), epoch_index=epoch_idx, epoch_count=len(epochs))
        model = None if use_oracle else load_ddpm_checkpoint(ctx, epoch, device)
        for timestep_idx, trow in enumerate(grid, start=1):
            candidate_start = time.time()
            coarse_k = int(ctx.cfg["score_validation"]["mc"]["coarse_k"])
            final_k = int(ctx.cfg["score_validation"]["mc"]["final_k"])
            score_validate_log(
                stage_root,
                "candidate_start",
                phase="coarse",
                epoch=int(epoch),
                report_timestep=int(trow["report_timestep"]),
                code_timestep_index=int(trow["code_timestep_index"]),
                mc_k=coarse_k,
                timestep_index=timestep_idx,
                timestep_count=len(grid),
                completed_candidates=int(completed_coarse),
                total_candidates=int(total_coarse),
            )
            ladder_df = compute_ladder_scores(
                ctx,
                val_ds,
                val_indices,
                model,
                trow,
                coarse_k,
                use_oracle=use_oracle,
                stage_root=stage_root,
                phase="coarse",
                epoch=int(epoch),
            )
            score_validate_log(
                stage_root,
                "ladder_done",
                phase="coarse",
                epoch=int(epoch),
                report_timestep=int(trow["report_timestep"]),
                rows=int(len(ladder_df)),
                elapsed_sec=round(time.time() - candidate_start, 3),
            )
            pm, per_image = paired_metrics(
                ladder_df,
                bootstrap_seed=int(ctx.cfg["score_validation"]["bootstrap_seed"]),
                bootstrap_iters=int(ctx.cfg["score_validation"]["bootstrap_iters_smoke" if ctx.smoke else "bootstrap_iters"]),
            )
            global_df = compute_validation_corpus_scores(
                ctx,
                train_ds,
                manifest,
                model,
                trow,
                coarse_k,
                use_oracle=use_oracle,
                stage_root=stage_root,
                phase="coarse",
                epoch=int(epoch),
            )
            score_validate_log(
                stage_root,
                "validation_corpus_done",
                phase="coarse",
                epoch=int(epoch),
                report_timestep=int(trow["report_timestep"]),
                rows=int(len(global_df)),
                elapsed_sec=round(time.time() - candidate_start, 3),
            )
            stable_df = compute_validation_corpus_scores(
                ctx,
                train_ds,
                manifest,
                model,
                trow,
                coarse_k,
                use_oracle=use_oracle,
                seed_group=1,
                stage_root=stage_root,
                phase="coarse_stability",
                epoch=int(epoch),
            )
            gm = global_metrics(global_df, stable_df, topk_ratio=float(ctx.cfg["filtering"]["topk_ratio"]))
            row = {"checkpoint_epoch": int(epoch), **trow, "mc_k": coarse_k, "scan_phase": "coarse", **pm, **gm}
            rows.append(row)
            ladder_df["checkpoint_epoch"] = int(epoch)
            global_df["checkpoint_epoch"] = int(epoch)
            append_csv(raw_ladder_path, ladder_df)
            append_csv(raw_global_path, global_df)
            append_csv(partial_candidate_path, pd.DataFrame([row]))
            completed_coarse += 1
            score_validate_log(
                stage_root,
                "candidate_done",
                phase="coarse",
                epoch=int(epoch),
                report_timestep=int(trow["report_timestep"]),
                mc_k=coarse_k,
                median_within_image_spearman=float(row["median_within_image_spearman"]),
                topk_median_realized_psnr_shift=float(row["topk_median_realized_psnr_shift"]),
                elapsed_sec=round(time.time() - candidate_start, 3),
                completed_candidates=int(completed_coarse),
                total_candidates=int(total_coarse),
                cuda_allocated_mb=float(torch.cuda.memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
                cuda_reserved_mb=float(torch.cuda.memory_reserved(device) / 1024**2) if device.type == "cuda" else 0.0,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        score_validate_log(
            stage_root,
            "epoch_done",
            phase="coarse",
            epoch=int(epoch),
            elapsed_sec=round(time.time() - epoch_start, 3),
            completed_candidates=int(completed_coarse),
            total_candidates=int(total_coarse),
            cuda_reserved_mb=float(torch.cuda.memory_reserved(device) / 1024**2) if device.type == "cuda" else 0.0,
        )
    candidates = pd.DataFrame(rows)
    top_n = int(ctx.cfg["score_validation"]["checkpoint_scan"].get("top_candidates", 3))
    final_k = int(ctx.cfg["score_validation"]["mc"]["final_k"])
    if final_k != int(ctx.cfg["score_validation"]["mc"]["coarse_k"]):
        final_rows = []
        finalist_keys = (
            candidates.sort_values(
                ["median_within_image_spearman", "topk_median_realized_psnr_shift", "topk_jaccard"],
                ascending=[False, False, False],
            )[["checkpoint_epoch", "report_timestep"]]
            .drop_duplicates()
            .head(top_n)
        )
        score_validate_log(
            stage_root,
            "final_refinement_start",
            finalists=[{"epoch": int(row.checkpoint_epoch), "report_timestep": int(row.report_timestep)} for row in finalist_keys.itertuples(index=False)],
            final_mc_k=final_k,
        )
        for finalist_idx, (_row_idx, key) in enumerate(finalist_keys.iterrows(), start=1):
            epoch = int(key["checkpoint_epoch"])
            report_timestep = int(key["report_timestep"])
            trow = next(item for item in grid if int(item["report_timestep"]) == report_timestep)
            candidate_start = time.time()
            score_validate_log(
                stage_root,
                "candidate_start",
                phase="final",
                epoch=epoch,
                report_timestep=report_timestep,
                code_timestep_index=int(trow["code_timestep_index"]),
                mc_k=final_k,
                finalist_index=finalist_idx,
                finalist_count=len(finalist_keys),
            )
            model = None if use_oracle else load_ddpm_checkpoint(ctx, epoch, device)
            ladder_df = compute_ladder_scores(
                ctx,
                val_ds,
                val_indices,
                model,
                trow,
                final_k,
                use_oracle=use_oracle,
                stage_root=stage_root,
                phase="final",
                epoch=epoch,
            )
            score_validate_log(
                stage_root,
                "ladder_done",
                phase="final",
                epoch=epoch,
                report_timestep=report_timestep,
                rows=int(len(ladder_df)),
                elapsed_sec=round(time.time() - candidate_start, 3),
            )
            pm, _per_image = paired_metrics(
                ladder_df,
                bootstrap_seed=int(ctx.cfg["score_validation"]["bootstrap_seed"]),
                bootstrap_iters=int(ctx.cfg["score_validation"]["bootstrap_iters_smoke" if ctx.smoke else "bootstrap_iters"]),
            )
            global_df = compute_validation_corpus_scores(
                ctx,
                train_ds,
                manifest,
                model,
                trow,
                final_k,
                use_oracle=use_oracle,
                stage_root=stage_root,
                phase="final",
                epoch=epoch,
            )
            score_validate_log(
                stage_root,
                "validation_corpus_done",
                phase="final",
                epoch=epoch,
                report_timestep=report_timestep,
                rows=int(len(global_df)),
                elapsed_sec=round(time.time() - candidate_start, 3),
            )
            stable_df = compute_validation_corpus_scores(
                ctx,
                train_ds,
                manifest,
                model,
                trow,
                final_k,
                use_oracle=use_oracle,
                seed_group=1,
                stage_root=stage_root,
                phase="final_stability",
                epoch=epoch,
            )
            gm = global_metrics(global_df, stable_df, topk_ratio=float(ctx.cfg["filtering"]["topk_ratio"]))
            row = {"checkpoint_epoch": epoch, **trow, "mc_k": final_k, "scan_phase": "final", **pm, **gm}
            final_rows.append(row)
            ladder_df["checkpoint_epoch"] = epoch
            global_df["checkpoint_epoch"] = epoch
            append_csv(raw_ladder_path, ladder_df)
            append_csv(raw_global_path, global_df)
            append_csv(partial_candidate_path, pd.DataFrame([row]))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            score_validate_log(
                stage_root,
                "candidate_done",
                phase="final",
                epoch=epoch,
                report_timestep=report_timestep,
                mc_k=final_k,
                median_within_image_spearman=float(row["median_within_image_spearman"]),
                topk_median_realized_psnr_shift=float(row["topk_median_realized_psnr_shift"]),
                elapsed_sec=round(time.time() - candidate_start, 3),
                finalist_index=finalist_idx,
                finalist_count=len(finalist_keys),
                cuda_allocated_mb=float(torch.cuda.memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
                cuda_reserved_mb=float(torch.cuda.memory_reserved(device) / 1024**2) if device.type == "cuda" else 0.0,
            )
        if final_rows:
            candidates = pd.concat([candidates, pd.DataFrame(final_rows)], ignore_index=True)
    candidates.to_csv(candidate_path, index=False)
    write_heatmaps(stage_root, candidates)
    candidates.sort_values(
        ["median_within_image_spearman", "topk_median_realized_psnr_shift", "topk_jaccard"],
        ascending=[False, False, False],
    ).head(20).to_csv(stage_root / "best_candidate_table.csv", index=False)
    write_json(ctx.experiment_root / "00_protocol" / "mc_seed_banks.json", {"coarse": mc_seed_bank(ctx.cfg, k=4), "final": mc_seed_bank(ctx.cfg, k=16)})
    score_validate_log(
        stage_root,
        "complete",
        candidate_metrics=str(candidate_path),
        raw_ladder_scores=str(raw_ladder_path),
        validation_corpus_scores=str(raw_global_path),
        elapsed_sec=round(time.time() - start, 3),
    )
    ctx.write_marker(stage, start, [candidate_path, stage_root / "raw_ladder_scores.csv"])


def compute_ladder_scores(
    ctx: PipelineContext,
    val_ds: Dataset,
    val_indices: np.ndarray,
    model: ImageSpaceDDPM | None,
    trow: dict[str, Any],
    mc_k: int,
    *,
    use_oracle: bool,
    stage_root: Path | None = None,
    phase: str = "",
    epoch: int | None = None,
) -> pd.DataFrame:
    rows = []
    images = []
    original_keys = []
    levels = ctx.cfg["score_validation"]["psnr_ladder"]
    for original_index in val_indices.tolist():
        x_clean, class_id = val_ds[int(original_index)]
        bank, level_rows = ladder_corruption(
            x_clean.to(torch.float32).clamp(0.0, 1.0),
            dataset_name=str(ctx.cfg["dataset"]["name"]),
            split="val",
            original_index=int(original_index),
            ladder_seed=int(ctx.cfg["seeds"]["ladder_seed"]),
            psnr_levels=levels,
        )
        for level_pos, item in enumerate(level_rows):
            rows.append({"original_index": int(original_index), "class_id": int(class_id), **item})
            images.append(bank[level_pos])
            original_keys.append(int(original_index))
    df = pd.DataFrame(rows)
    if use_oracle:
        df["score"] = oracle_score_from_quality(df)
        df["score_std"] = 0.0
    else:
        def log_chunk(chunk_idx: int, total_chunks: int, rows_done: int) -> None:
            if stage_root is None:
                return
            if chunk_idx == 1 or chunk_idx == total_chunks or chunk_idx % 10 == 0:
                score_validate_log(
                    stage_root,
                    "ladder_score_batch",
                    phase=phase,
                    epoch=epoch,
                    report_timestep=int(trow["report_timestep"]),
                    mc_k=int(mc_k),
                    chunk=int(chunk_idx),
                    chunks=int(total_chunks),
                    rows_done=int(rows_done),
                    rows_total=int(len(images)),
                )

        scores, stds = score_batch_chunked(
            model,
            images,
            original_keys,
            code_timestep_index=int(trow["code_timestep_index"]),
            seed_bank=mc_seed_bank(ctx.cfg, k=mc_k),
            batch_size=int(ctx.cfg["score_validation"].get("batch_size_smoke", ctx.cfg["score_validation"]["batch_size"]) if ctx.smoke else ctx.cfg["score_validation"]["batch_size"]),
            paired_level_count=len(levels),
            progress_callback=log_chunk,
        )
        df["score"] = scores
        df["score_std"] = stds
    df["report_timestep"] = int(trow["report_timestep"])
    df["code_timestep_index"] = int(trow["code_timestep_index"])
    df["mc_k"] = int(mc_k)
    return df


def compute_validation_corpus_scores(
    ctx: PipelineContext,
    train_ds: Dataset,
    manifest: pd.DataFrame,
    model: ImageSpaceDDPM | None,
    trow: dict[str, Any],
    mc_k: int,
    *,
    use_oracle: bool,
    seed_group: int = 0,
    stage_root: Path | None = None,
    phase: str = "",
    epoch: int | None = None,
) -> pd.DataFrame:
    if ctx.smoke:
        max_count = int(ctx.cfg["score_validation"]["validation_corpus_samples_smoke"])
        df = manifest.head(min(max_count, len(manifest))).copy()
    else:
        per_class = int(ctx.cfg["score_validation"].get("validation_corpus_samples_per_class", 1))
        parts = []
        rng = np.random.default_rng(int(ctx.cfg["seeds"]["split_seed"]) + 202)
        for _cls, group in manifest.groupby("class_id", sort=True):
            idx = group.index.to_numpy(dtype=np.int64, copy=True)
            rng.shuffle(idx)
            parts.append(manifest.loc[np.sort(idx[: min(per_class, len(idx))])])
        df = pd.concat(parts, ignore_index=True) if parts else manifest.head(0).copy()
        max_count = int(ctx.cfg["score_validation"]["validation_corpus_samples"])
        df = df.head(min(max_count, len(df))).copy()
    if use_oracle:
        df["score"] = oracle_score_from_quality(df)
        df["score_std"] = 0.0
    else:
        corpus = OnDemandCorpusDataset(train_ds, manifest, original_indices=df["original_index"].astype(int).to_numpy(), return_index=True)
        loader = DataLoader(
            corpus,
            batch_size=int(ctx.cfg["score_validation"]["batch_size"]),
            shuffle=False,
            num_workers=0 if ctx.smoke else int(ctx.cfg["runtime"]["num_workers"]),
            pin_memory=bool(not ctx.smoke and torch.cuda.is_available()),
        )
        parts = []
        total_batches = len(loader)
        for batch_idx, (x, original, _class_id) in enumerate(loader, start=1):
            score, std = score_batch(
                model,
                x,
                original_indices=original.numpy().astype(np.int64),
                code_timestep_index=int(trow["code_timestep_index"]),
                seed_bank=mc_seed_bank(ctx.cfg, k=mc_k, group=seed_group),
                seed_group=seed_group,
            )
            parts.append(pd.DataFrame({"original_index": original.numpy().astype(np.int64), "score": score, "score_std": std}))
            if stage_root is not None and (batch_idx == 1 or batch_idx == total_batches or batch_idx % 10 == 0):
                score_validate_log(
                    stage_root,
                    "validation_corpus_batch",
                    phase=phase,
                    epoch=epoch,
                    report_timestep=int(trow["report_timestep"]),
                    mc_k=int(mc_k),
                    seed_group=int(seed_group),
                    batch=int(batch_idx),
                    batches=int(total_batches),
                    rows_done=int(min(batch_idx * int(ctx.cfg["score_validation"]["batch_size"]), len(corpus))),
                    rows_total=int(len(corpus)),
                )
        df = df.merge(pd.concat(parts, ignore_index=True), on="original_index", how="left")
    df["report_timestep"] = int(trow["report_timestep"])
    df["code_timestep_index"] = int(trow["code_timestep_index"])
    df["mc_k"] = int(mc_k)
    return df


def write_heatmaps(root: Path, candidates: pd.DataFrame) -> None:
    specs = [
        ("median_within_image_spearman", "heatmap_median_within_image_spearman.png"),
        ("topk_median_realized_psnr_shift", "heatmap_topk_median_realized_psnr_shift.png"),
        ("global_spearman_score_realized_psnr", "heatmap_global_spearman_score_realized_psnr.png"),
        ("pairwise_ordering_accuracy", "heatmap_pairwise_ordering_accuracy.png"),
        ("fraction_images_rho_gt_0", "heatmap_fraction_positive_rho.png"),
        ("topk_jaccard", "heatmap_topk_jaccard.png"),
    ]
    for metric, filename in specs:
        pivot = candidates.pivot_table(index="checkpoint_epoch", columns="report_timestep", values=metric, aggfunc="mean")
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        im = ax.imshow(pivot.to_numpy(dtype=np.float64), aspect="auto", interpolation="nearest")
        ax.set_xticks(np.arange(len(pivot.columns)), [str(int(c)) for c in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)), [str(int(i)) for i in pivot.index])
        ax.set_xlabel("report timestep")
        ax.set_ylabel("checkpoint epoch")
        ax.set_title(metric)
        fig.colorbar(im, ax=ax)
        fig.savefig(root / filename, dpi=150)
        plt.close(fig)
    top = candidates.sort_values("median_within_image_spearman", ascending=False).head(5)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for epoch, group in top.groupby("checkpoint_epoch"):
        ax.plot(group["report_timestep"], group["median_within_image_spearman"], marker="o", label=f"epoch {epoch}")
    ax.set_xlabel("report timestep")
    ax.set_ylabel("median within-image Spearman")
    ax.legend()
    fig.savefig(root / "best_checkpoints_by_timestep.png", dpi=150)
    plt.close(fig)


def stage_score_select(ctx: PipelineContext) -> None:
    stage = "score_select"
    start = time.time()
    selected_path = ctx.stage_root(stage) / "selected_score_config.yaml"
    if ctx.resume and ctx.marker_valid(stage, [selected_path]):
        print(f"[resume] score_select already complete: {selected_path}")
        return
    if ctx.dry_run:
        print(f"[dry-run] would select score config into {selected_path}")
        return
    # Apply coverage and sign gates, then use the fixed order to break ties.
    candidates = pd.read_csv(ctx.stage_root("score_validate") / "candidate_metrics.csv")
    if "scan_phase" in candidates.columns and (candidates["scan_phase"] == "final").any():
        candidates = candidates[candidates["scan_phase"] == "final"].copy()
    gates = ctx.cfg["score_selection"]["gates"]
    validation_corpus_n = int(ctx.cfg["score_validation"]["validation_corpus_samples"])
    abs_coverage_threshold = float(gates["validation_min_class_coverage_large_corpus"])
    relative_coverage_tolerance = float(gates["validation_random_relative_coverage_tolerance"])
    max_js_divergence = float(gates["validation_max_distribution_js_divergence"])
    coverage_absolute_mask = candidates["class_coverage"] >= abs_coverage_threshold
    coverage_random_relative_mask = (
        candidates["class_coverage"] >= candidates["random_10_class_coverage"] - relative_coverage_tolerance
    ) & (candidates["class_distribution_js_divergence"] <= max_js_divergence)
    if ctx.smoke:
        coverage_mask = candidates["class_coverage"] >= float(gates.get("min_class_coverage_smoke", 0.0))
        coverage_rule = "synthetic smoke: relaxed coverage"
    elif validation_corpus_n >= int(gates["validation_coverage_large_corpus_min_samples"]):
        large_rule = str(
            gates.get(
                "validation_large_corpus_rule",
                "absolute_or_random_relative_coverage_and_distribution_divergence",
            )
        )
        if large_rule == "absolute_or_random_relative_coverage_and_distribution_divergence":
            coverage_mask = coverage_absolute_mask | coverage_random_relative_mask
            coverage_rule = (
                "large validation corpus: "
                f"class_coverage >= {abs_coverage_threshold} OR "
                f"(class_coverage >= random_10_class_coverage - {relative_coverage_tolerance} "
                f"and JS <= {max_js_divergence})"
            )
        else:
            coverage_mask = coverage_absolute_mask
            coverage_rule = f"large validation corpus: class_coverage >= {abs_coverage_threshold}"
    else:
        coverage_mask = coverage_random_relative_mask
        coverage_rule = (
            "small validation corpus: class_coverage >= random_10_class_coverage - "
            f"{relative_coverage_tolerance} and JS <= {max_js_divergence}"
        )
    gate_masks = {
        "bootstrap_ci_low_gt_0": candidates["bootstrap_ci_low"] > 0.0,
        "topk_median_realized_psnr_shift_gt_0": candidates["topk_median_realized_psnr_shift"] > 0.0,
        "validation_class_coverage": coverage_mask,
        "validation_class_coverage_absolute": coverage_absolute_mask,
        "validation_class_coverage_random_relative": coverage_random_relative_mask,
        "global_spearman_score_sigma_gt_0": candidates["global_spearman_score_sigma"] > 0.0,
        "global_spearman_score_corruption_mse_gt_0": candidates["global_spearman_score_corruption_mse"] > 0.0,
        "global_spearman_score_realized_psnr_lt_0": candidates["global_spearman_score_realized_psnr"] < 0.0,
    }
    gate_counts = {name: int(mask.sum()) for name, mask in gate_masks.items()}
    gate_counts["rows"] = int(len(candidates))
    print(f"score_select gates: {json.dumps({'coverage_rule': coverage_rule, 'counts': gate_counts}, sort_keys=True)}", flush=True)
    scientific_mask = (
        gate_masks["bootstrap_ci_low_gt_0"]
        & gate_masks["topk_median_realized_psnr_shift_gt_0"]
        & coverage_mask
        & gate_masks["global_spearman_score_sigma_gt_0"]
        & gate_masks["global_spearman_score_corruption_mse_gt_0"]
        & gate_masks["global_spearman_score_realized_psnr_lt_0"]
    )
    eligible = candidates[scientific_mask].copy()
    relaxed_real_smoke = False
    if eligible.empty and ctx.real_smoke and bool(ctx.cfg["real_smoke"].get("relaxed_score_selection", False)):
        eligible = candidates.sort_values(
            ["median_within_image_spearman", "topk_median_realized_psnr_shift", "topk_jaccard"],
            ascending=[False, False, False],
        ).head(1).copy()
        relaxed_real_smoke = True
    if eligible.empty:
        write_json(
            ctx.stage_root(stage) / "diagnostic_summary.json",
            {
                "status": "no_eligible_candidate",
                "rows": len(candidates),
                "coverage_rule": coverage_rule,
                "gate_counts": gate_counts,
                "top_candidates": candidates.sort_values(
                    ["median_within_image_spearman", "topk_median_realized_psnr_shift", "topk_jaccard"],
                    ascending=[False, False, False],
                )
                .head(10)
                .to_dict(orient="records"),
            },
        )
        raise SystemExit("score_select gate failed: no eligible candidate; filtering/downstream remain blocked")
    close = float(ctx.cfg["score_selection"]["close_metric_tolerance"])
    best_primary = float(eligible["median_within_image_spearman"].max())
    eligible["primary_close"] = (best_primary - eligible["median_within_image_spearman"]).abs() <= close
    eligible = eligible.sort_values(
        [
            "primary_close",
            "median_within_image_spearman",
            "topk_median_realized_psnr_shift",
            "topk_jaccard",
            "class_distribution_max_abs_deviation",
            "checkpoint_epoch",
            "report_timestep",
        ],
        ascending=[False, False, False, False, True, True, True],
    )
    row = eligible.iloc[0].to_dict()
    ckpt_path = ctx.stage_root("ddpm_train") / "checkpoints" / f"epoch_{int(row['checkpoint_epoch']):03d}.pt"
    if not ckpt_path.exists():
        raise RuntimeError(f"Selected checkpoint does not exist: {ckpt_path}")
    manifest_path = ctx.stage_root("prepare") / "corpus_manifest.csv"
    grid = pd.read_csv(ctx.stage_root("score_validate") / "resolved_timestep_grid.csv")
    trow = grid[grid["report_timestep"] == int(row["report_timestep"])].iloc[0].to_dict()
    payload = {
        "dataset": ctx.cfg["dataset"]["name"],
        "run_id": ctx.run_id,
        "corpus_manifest_path": str(manifest_path),
        "corpus_manifest_hash": sha256_file(manifest_path),
        "checkpoint_epoch": int(row["checkpoint_epoch"]),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "ddpm_prediction_type": "epsilon",
        "score_formula": "mean_K sum(flatten(eps_pred(x_t,t)) ** 2)",
        "score_reduction": "sum",
        "score_direction": "lower score is selected by Top-K",
        "report_timestep": int(row["report_timestep"]),
        "code_timestep_index": int(row["code_timestep_index"]),
        "beta_t": float(trow["beta_t"]),
        "alpha_t": float(trow["alpha_t"]),
        "alpha_bar_t": float(trow["alpha_bar_t"]),
        "sqrt_alpha_bar_t": float(trow["sqrt_alpha_bar_t"]),
        "sqrt_one_minus_alpha_bar_t": float(trow["sqrt_one_minus_alpha_bar_t"]),
        "effective_diffusion_noise": float(trow["effective_diffusion_noise"]),
        "mc_k": int(ctx.cfg["score_validation"]["mc"]["final_k"]),
        "mc_seed_bank": mc_seed_bank(ctx.cfg, k=int(ctx.cfg["score_validation"]["mc"]["final_k"])),
        "validation_subset_path": str(ctx.experiment_root / "00_protocol" / "score_validation_indices.npy"),
        "primary_metric_value": float(row["median_within_image_spearman"]),
        "bootstrap_ci": [float(row["bootstrap_ci_low"]), float(row["bootstrap_ci_high"])],
        "topk_psnr_shift": float(row["topk_median_realized_psnr_shift"]),
        "class_coverage": float(row["class_coverage"]),
        "validation_class_coverage_rule": coverage_rule,
        "validation_class_coverage_absolute_threshold": abs_coverage_threshold,
        "validation_class_coverage_absolute_passed": bool(row["class_coverage"] >= abs_coverage_threshold),
        "validation_class_coverage_random_relative_passed": bool(
            (float(row["class_coverage"]) >= float(row["random_10_class_coverage"]) - relative_coverage_tolerance)
            and (float(row["class_distribution_js_divergence"]) <= max_js_divergence)
        ),
        "validation_random_10_class_coverage": float(row.get("random_10_class_coverage", float("nan"))),
        "validation_class_distribution_js_divergence": float(row.get("class_distribution_js_divergence", float("nan"))),
        "topk_jaccard": float(row["topk_jaccard"]),
        "real_smoke_relaxed_selection": bool(relaxed_real_smoke),
        "config_hash": ctx.cfg_hash,
        "git_commit": git_commit(),
    }
    save_yaml(selected_path, payload)
    write_json(
        ctx.stage_root(stage) / "selection_report.json",
        {"selected": payload, "eligible_count": int(len(eligible)), "coverage_rule": coverage_rule, "gate_counts": gate_counts},
    )
    ctx.write_marker(stage, start, [selected_path, ctx.stage_root(stage) / "selection_report.json"])


def require_selected(ctx: PipelineContext) -> dict[str, Any]:
    marker = ctx.marker_path("score_select")
    selected = ctx.stage_root("score_select") / "selected_score_config.yaml"
    if not marker.exists() or not selected.exists():
        raise RuntimeError("Filtering/downstream blocked: missing successful score_select marker or selected_score_config.yaml")
    cfg = read_yaml(selected)
    if cfg.get("config_hash") != ctx.cfg_hash:
        provenance_cfg = ctx.cfg.get("provenance", {})
        if not isinstance(provenance_cfg, dict):
            provenance_cfg = {}
        if not bool(provenance_cfg.get("allow_downstream_reuse_with_mismatched_config_hash", False)):
            raise RuntimeError("selected_score_config config_hash does not match current run")
        print(
            "[provenance] selected_score_config config_hash differs from current run; "
            "continuing because downstream-only artifact reuse is explicitly enabled."
        )
    return cfg


def stage_filter(ctx: PipelineContext) -> None:
    stage = "filter"
    start = time.time()
    scores_path = ctx.stage_root(stage) / "scores_full.csv"
    if ctx.resume and ctx.marker_valid(stage, [scores_path, ctx.stage_root(stage) / "subset_indices" / "topk_p10_indices.npy"]):
        print(f"[resume] filter already complete: {scores_path}")
        return
    if ctx.dry_run:
        print(f"[dry-run] would compute full-corpus scores and filtering artifacts at {ctx.stage_root(stage)}")
        return
    selected = require_selected(ctx)
    train_ds = build_dataset(ctx.cfg, split="train", smoke=ctx.smoke)
    manifest = pd.read_csv(ctx.stage_root("prepare") / "corpus_manifest.csv")
    if sha256_file(ctx.stage_root("prepare") / "corpus_manifest.csv") != selected["corpus_manifest_hash"]:
        raise RuntimeError("Corpus manifest hash mismatch")
    use_oracle = bool(ctx.smoke and ctx.cfg.get("smoke", {}).get("oracle_scores", True))
    if use_oracle:
        score_df = manifest.copy()
        score_df["score"] = oracle_score_from_quality(score_df)
        score_df["score_std"] = 0.0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_ddpm_checkpoint(ctx, int(selected["checkpoint_epoch"]), device)
        corpus = OnDemandCorpusDataset(train_ds, manifest, return_index=True)
        loader = DataLoader(corpus, batch_size=int(ctx.cfg["filtering"]["batch_size"]), shuffle=False, num_workers=0 if ctx.smoke else int(ctx.cfg["runtime"]["num_workers"]))
        chunks = ctx.stage_root(stage) / "score_shards"
        chunks.mkdir(parents=True, exist_ok=True)
        parts = []
        for chunk_id, (x, original, _class_id) in enumerate(tqdm(loader, desc="Scoring full corpus")):
            chunk_path = chunks / f"chunk_{chunk_id:06d}.csv"
            if ctx.resume and chunk_path.exists():
                parts.append(pd.read_csv(chunk_path))
                continue
            score, std = score_batch(
                model,
                x,
                original_indices=original.numpy().astype(np.int64),
                code_timestep_index=int(selected["code_timestep_index"]),
                seed_bank=[int(x) for x in selected["mc_seed_bank"]],
            )
            part = pd.DataFrame({"original_index": original.numpy().astype(np.int64), "score": score, "score_std": std})
            part.to_csv(chunk_path, index=False)
            parts.append(part)
        score_df = manifest.merge(pd.concat(parts, ignore_index=True), on="original_index", how="left")
    score_df["checkpoint_epoch"] = int(selected["checkpoint_epoch"])
    score_df["report_timestep"] = int(selected["report_timestep"])
    score_df["mc_k"] = int(selected["mc_k"])
    if score_df["original_index"].nunique() != len(score_df) or not np.isfinite(score_df["score"]).all():
        raise RuntimeError("Invalid full-corpus scores")
    score_df = score_df.sort_values(["score", "original_index"], ascending=[True, True]).reset_index(drop=True)
    score_df["score_rank"] = np.arange(1, len(score_df) + 1, dtype=np.int64)
    score_df.to_csv(scores_path, index=False)
    write_filtering_subsets(ctx, score_df)
    write_filtering_figures(ctx, score_df)
    ctx.write_marker(stage, start, [scores_path, ctx.stage_root(stage) / "subset_indices" / "topk_p10_indices.npy"])


def stage_filter_grid(ctx: PipelineContext) -> None:
    stage = "filter_grid"
    start = time.time()
    root = ctx.stage_root("filter")
    scores_path = root / "scores_full.csv"
    required = [
        scores_path,
        root / "subset_indices" / "topk_p10_indices.npy",
        root / "subset_indices" / "quantile_p10_indices.npy",
        root / "subset_indices" / "random_p10_indices.npy",
        root / "subset_indices" / "topk_p20_indices.npy",
        root / "subset_indices" / "quantile_p20_indices.npy",
        root / "subset_indices" / "random_p20_indices.npy",
        root / "subset_indices" / "topk_p30_indices.npy",
        root / "subset_indices" / "quantile_p30_indices.npy",
        root / "subset_indices" / "random_p30_indices.npy",
    ]
    if ctx.resume and ctx.marker_valid(stage, required):
        print(f"[resume] filter_grid already complete from existing scores: {scores_path}")
        return
    if ctx.dry_run:
        print(f"[dry-run] would build p10/p20/p30 filtering grid from existing {scores_path}")
        return
    require_selected(ctx)
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing existing full score artifact; refusing to rerun DDPM scoring in filter_grid: {scores_path}")
    scores = pd.read_csv(scores_path)
    if scores["original_index"].nunique() != len(scores) or not np.isfinite(scores["score"]).all():
        raise RuntimeError(f"Invalid scores_full.csv: {scores_path}")
    if "score_rank" not in scores.columns:
        scores = scores.sort_values(["score", "original_index"], ascending=[True, True]).reset_index(drop=True)
        scores["score_rank"] = np.arange(1, len(scores) + 1, dtype=np.int64)
    write_filtering_subsets(ctx, scores)
    write_filtering_figures(ctx, scores)
    ctx.write_marker(stage, start, required)


def filtering_fractions(ctx: PipelineContext) -> list[float]:
    raw = ctx.cfg.get("filtering", {}).get("grid_fractions")
    if raw is None:
        raw = [
            float(ctx.cfg["filtering"].get("topk_ratio", 0.10)),
            float(ctx.cfg["filtering"].get("quantile_ratio", ctx.cfg["filtering"].get("topk_ratio", 0.10))),
            float(ctx.cfg["filtering"].get("random_ratio", ctx.cfg["filtering"].get("topk_ratio", 0.10))),
        ]
    values = sorted({float(x) for x in raw})
    if not values:
        raise ValueError("filtering.grid_fractions must not be empty")
    return values


def fraction_tag(fraction: float) -> str:
    return f"p{int(round(float(fraction) * 100.0))}"


def subset_size_for_fraction(n_full: int, fraction: float) -> int:
    return max(1, min(int(round(float(fraction) * int(n_full))), int(n_full)))


def variant_strategy_fraction(variant: str) -> tuple[str, float]:
    if variant == "full":
        return "full", 1.0
    aliases = {
        "topk_10": "topk_p10",
        "quantile_10": "quantile_p10",
        "random_10": "random_p10",
    }
    canonical = aliases.get(variant, variant)
    if canonical.startswith("topk_p"):
        return "topk", int(canonical.rsplit("p", 1)[1]) / 100.0
    if canonical.startswith("quantile_p"):
        return "quantile", int(canonical.rsplit("p", 1)[1]) / 100.0
    if canonical.startswith("random_p"):
        return "random", int(canonical.rsplit("p", 1)[1]) / 100.0
    raise ValueError(f"Unknown downstream/filtering variant: {variant}")


def grid_variants(ctx: PipelineContext) -> list[str]:
    variants = ["full"]
    for fraction in filtering_fractions(ctx):
        tag = fraction_tag(fraction)
        variants.extend([f"topk_{tag}", f"quantile_{tag}", f"random_{tag}"])
    return variants


def display_variant(variant: str) -> tuple[str, float, str]:
    strategy, fraction = variant_strategy_fraction(variant)
    if strategy == "full":
        return "Full", 1.0, "Full 100%"
    labels = {"topk": "Top-K", "quantile": "Quantile", "random": "Random"}
    return labels[strategy], fraction, f"{labels[strategy]} {int(round(fraction * 100.0))}%"


def save_subset_array(path: Path, indices: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    indices = np.sort(np.asarray(indices, dtype=np.int64))
    if path.exists():
        existing = np.load(path).astype(np.int64)
        if not np.array_equal(existing, indices):
            raise FileExistsError(f"Existing subset differs; refusing to overwrite {path}")
        return
    np.save(path, indices)


def write_filtering_subsets(ctx: PipelineContext, scores: pd.DataFrame) -> None:
    n_full = len(scores)
    root = ctx.stage_root("filter")
    subsets: dict[str, np.ndarray] = {"full": np.sort(scores["original_index"].to_numpy(dtype=np.int64))}
    q_input = scores.rename(columns={"score": "score_value"})
    source = np.sort(scores["original_index"].to_numpy(dtype=np.int64))
    for fraction in filtering_fractions(ctx):
        tag = fraction_tag(fraction)
        n_subset = subset_size_for_fraction(n_full, fraction)
        topk = scores.sort_values(["score", "original_index"], ascending=[True, True]).head(n_subset)["original_index"].to_numpy(dtype=np.int64)
        quantile = select_quantile_spread_exact(
            q_input,
            n_subset,
            seed=int(ctx.cfg["seeds"]["subset_seed"]),
            min_points_per_bin=int(ctx.cfg["filtering"]["quantile_min_points_per_bin"]),
        )
        rng = np.random.default_rng(int(ctx.cfg["seeds"]["subset_seed"]))
        random = rng.choice(source, size=n_subset, replace=False)
        names = (f"topk_{tag}", f"quantile_{tag}", f"random_{tag}")
        subsets[names[0]] = np.sort(topk)
        subsets[names[1]] = np.sort(quantile)
        subsets[names[2]] = np.sort(random)
    for name, indices in subsets.items():
        strategy, fraction = variant_strategy_fraction(name)
        expected = n_full if strategy == "full" else subset_size_for_fraction(n_full, fraction)
        if len(indices) != expected:
            raise AssertionError(f"{name}: got {len(indices)}, expected {expected}")
        if len(np.unique(indices)) != len(indices):
            raise AssertionError(f"{name}: duplicate original indices")
        path = root / "subset_indices" / f"{name}_indices.npy"
        save_subset_array(path, indices)
        scores[scores["original_index"].isin(set(indices.tolist()))].to_csv(root / "subset_manifests" / f"{name}_manifest.csv", index=False)
        if strategy != "full":
            pct_dir = root / fraction_tag(fraction)
            pct_dir.mkdir(parents=True, exist_ok=True)
            save_subset_array(pct_dir / f"{name}_indices.npy", indices)
            scores[scores["original_index"].isin(set(indices.tolist()))].to_csv(pct_dir / f"{name}_manifest.csv", index=False)
    legacy_aliases = {"topk_10": "topk_p10", "quantile_10": "quantile_p10", "random_10": "random_p10"}
    for alias, canonical in legacy_aliases.items():
        if canonical not in subsets:
            continue
        indices = subsets[canonical]
        save_subset_array(root / "subset_indices" / f"{alias}_indices.npy", indices)
        scores[scores["original_index"].isin(set(indices.tolist()))].to_csv(root / "subset_manifests" / f"{alias}_manifest.csv", index=False)
    membership = scores[["original_index", "score", "score_rank"]].copy()
    for name, indices in subsets.items():
        membership[f"is_{name}"] = membership["original_index"].isin(set(indices.tolist()))
    membership.to_csv(root / "subset_membership.csv", index=False)
    overlap_rows = []
    names = list(subsets)
    for a in names:
        for b in names:
            overlap_rows.append({"a": a, "b": b, "overlap": int(len(set(subsets[a]) & set(subsets[b])))})
    pd.DataFrame(overlap_rows).to_csv(root / "overlap_matrix.csv", index=False)
    summary_rows = []
    full = scores
    for name, indices in subsets.items():
        df = scores[scores["original_index"].isin(set(indices.tolist()))]
        row = {"subset": name, "size": int(len(df))}
        for col in ["target_psnr_db", "realized_psnr_db_postclip", "realized_snr_db_postclip", "sigma", "score"]:
            row[f"{col}_mean"] = float(df[col].mean())
            row[f"{col}_median"] = float(df[col].median())
            row[f"{col}_std"] = float(df[col].std())
            row[f"{col}_q10"] = float(df[col].quantile(0.10))
            row[f"{col}_q90"] = float(df[col].quantile(0.90))
            row[f"delta_{col}_median_vs_full"] = float(df[col].median() - full[col].median())
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(root / "selection_summary.csv", index=False)
    class_counts = []
    class_summary = []
    all_classes = list(range(int(ctx.cfg["dataset"]["num_classes"])))
    full_total_by_class = scores["class_id"].value_counts().reindex(all_classes, fill_value=0).sort_index()
    full_dist = full_total_by_class.to_numpy(dtype=np.float64) / max(float(full_total_by_class.sum()), 1.0)
    for name, indices in subsets.items():
        df = scores[scores["original_index"].isin(set(indices.tolist()))]
        counts = df["class_id"].value_counts().reindex(all_classes, fill_value=0).sort_index()
        rates = counts / full_total_by_class.replace(0, np.nan)
        rates = rates.fillna(0.0)
        dist = counts.to_numpy(dtype=np.float64) / max(float(counts.sum()), 1.0)
        mid = 0.5 * (full_dist + dist)
        js = float(
            0.5 * np.sum(full_dist * np.log((full_dist + EPS) / (mid + EPS)))
            + 0.5 * np.sum(dist * np.log((dist + EPS) / (mid + EPS)))
        )
        for cls, rate in rates.items():
            class_counts.append({"subset": name, "class_id": int(cls), "selection_rate": float(rate), "count": int(counts.loc[cls])})
        selected_counts = counts.to_numpy(dtype=np.int64)
        class_summary.append(
            {
                "subset": name,
                "class_coverage": float((selected_counts > 0).mean()),
                "selected_samples_per_class_min": int(selected_counts.min()) if len(selected_counts) else 0,
                "selected_samples_per_class_median": float(np.median(selected_counts)) if len(selected_counts) else 0.0,
                "selected_samples_per_class_max": int(selected_counts.max()) if len(selected_counts) else 0,
                "selection_rate_min": float(rates.min()) if len(rates) else 0.0,
                "selection_rate_median": float(rates.median()) if len(rates) else 0.0,
                "selection_rate_max": float(rates.max()) if len(rates) else 0.0,
                "class_distribution_js_divergence": js,
            }
        )
    pd.DataFrame(class_counts).to_csv(root / "class_counts.csv", index=False)
    class_summary_df = pd.DataFrame(class_summary)
    topk_cov = float(class_summary_df[class_summary_df["subset"] == "topk_p10"]["class_coverage"].iloc[0])
    random_cov = float(class_summary_df[class_summary_df["subset"] == "random_p10"]["class_coverage"].iloc[0])
    class_summary_df["topk_coverage_minus_random_10_coverage"] = topk_cov - random_cov
    class_summary_df.to_csv(root / "class_coverage_summary.csv", index=False)
    if (
        bool(ctx.cfg["filtering"].get("full_train_topk_requires_all_classes", True))
        and not ctx.smoke
        and not ctx.real_smoke
        and topk_cov < 1.0
    ):
        raise RuntimeError(f"Full-train Top-K class coverage check failed: coverage={topk_cov:.4f}, expected all classes")
    write_json(
        root / "filtering_metadata.json",
        {
            "subset_size_rule": "n_subset = int(round(ratio * n_full)), clamped to [1,n_full], matching CIFAR implementation",
            "fractions": filtering_fractions(ctx),
            "n_full": n_full,
            "subset_sizes": {name: int(len(indices)) for name, indices in subsets.items()},
            "subset_sha256": {name: sha256_file(root / "subset_indices" / f"{name}_indices.npy") for name in subsets},
            "topk_rule": "minimum score, original_index tie-breaker",
            "quantile_spread_reference": "src.filtering.selection.select_quantile_spread_exact",
            "random_seed": int(ctx.cfg["seeds"]["subset_seed"]),
            "full_train_topk_requires_all_classes": bool(ctx.cfg["filtering"].get("full_train_topk_requires_all_classes", True)),
            "topk_class_coverage": topk_cov,
            "random_10_class_coverage": random_cov,
            "topk_coverage_minus_random_10_coverage": topk_cov - random_cov,
        },
    )


def write_filtering_figures(ctx: PipelineContext, scores: pd.DataFrame) -> None:
    root = ctx.stage_root("filter")
    clean_for_index, corrupt_for_index = filtering_image_lookup(ctx)
    for fraction in filtering_fractions(ctx):
        tag = fraction_tag(fraction)
        topk_name = f"topk_{tag}"
        quantile_name = f"quantile_{tag}"
        random_name = f"random_{tag}"
        subsets = {
            "Top-K": np.load(root / "subset_indices" / f"{topk_name}_indices.npy"),
            "Quantile": np.load(root / "subset_indices" / f"{quantile_name}_indices.npy"),
            "Random": np.load(root / "subset_indices" / f"{random_name}_indices.npy"),
        }
        plot_quality_shift_3x3(
            scores,
            {"topk": subsets["Top-K"], "quantile": subsets["Quantile"], "random": subsets["Random"]},
            root / "plots" / f"quality_shift_{tag}_3x3.png",
            fraction_label=tag,
        )
        plot_ecdf_realized_psnr(scores, subsets, root / "plots" / f"ecdf_realized_psnr_{tag}.png", fraction_label=tag)
        save_topk_selection_examples(
            scores,
            subsets["Top-K"],
            root / "panels" / f"topk_selection_examples_{tag}.png",
            clean_for_index=clean_for_index,
            corrupt_for_index=corrupt_for_index,
            fraction_label=tag,
        )
        save_topk_cleanest_noisiest(
            scores,
            subsets["Top-K"],
            root / "panels" / f"topk_cleanest_noisiest_{tag}.png",
            clean_for_index=clean_for_index,
            corrupt_for_index=corrupt_for_index,
            fraction_label=tag,
        )
    if (root / "plots" / "quality_shift_p10_3x3.png").exists():
        shutil.copyfile(root / "plots" / "quality_shift_p10_3x3.png", root / "plots" / "quality_shift_3x3.png")
    if (root / "plots" / "ecdf_realized_psnr_p10.png").exists():
        shutil.copyfile(root / "plots" / "ecdf_realized_psnr_p10.png", root / "plots" / "ecdf_realized_psnr.png")
    write_visual_panel(ctx, scores)


def tensor_to_pil(x: torch.Tensor, size: int = 96) -> Image.Image:
    arr = x.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255).round().astype(np.uint8)).resize((size, size))


def filtering_image_lookup(ctx: PipelineContext) -> tuple[Callable[[int], torch.Tensor], Callable[[int], torch.Tensor]]:
    train_ds = build_dataset(ctx.cfg, split="train", smoke=ctx.smoke)
    manifest = pd.read_csv(ctx.stage_root("prepare") / "corpus_manifest.csv")
    corpus = OnDemandCorpusDataset(train_ds, manifest, return_index=True)

    def clean_for_index(original_index: int) -> torch.Tensor:
        x, _ = train_ds[int(original_index)]
        return x.to(torch.float32).clamp(0.0, 1.0)

    def corrupt_for_index(original_index: int) -> torch.Tensor:
        pos = int(np.where(corpus.original_indices == int(original_index))[0][0])
        x_corrupt, _, _ = corpus[pos]
        return x_corrupt.to(torch.float32).clamp(0.0, 1.0)

    return clean_for_index, corrupt_for_index


def write_visual_panel(ctx: PipelineContext, scores: pd.DataFrame) -> None:
    root = ctx.stage_root("filter")
    train_ds = build_dataset(ctx.cfg, split="train", smoke=ctx.smoke)
    _clean_for_index, corrupt_for_index = filtering_image_lookup(ctx)
    choices = [
        ("lowest score", scores.nsmallest(1, "score").iloc[0]),
        ("highest score", scores.nlargest(1, "score").iloc[0]),
        ("oracle cleanest", scores.nlargest(1, "realized_psnr_db_postclip").iloc[0]),
        ("oracle noisiest", scores.nsmallest(1, "realized_psnr_db_postclip").iloc[0]),
    ]
    w, h = 270, 245
    canvas = Image.new("RGB", (w * len(choices), h), "white")
    draw = ImageDraw.Draw(canvas)
    for pos, (title, row) in enumerate(choices):
        original = int(row["original_index"])
        clean, _ = train_ds[original]
        x_corrupt = corrupt_for_index(original)
        x0 = pos * w
        canvas.paste(tensor_to_pil(clean), (x0 + 10, 30))
        canvas.paste(tensor_to_pil(x_corrupt), (x0 + 112, 30))
        draw.text((x0 + 10, 8), title, fill=(0, 0, 0))
        draw.multiline_text(
            (x0 + 10, 135),
            f"idx {original}\nclass {row['class_name']}\nscore {float(row['score']):.4g}\ntarget {float(row['target_psnr_db']):.2f}\nrealized {float(row['realized_psnr_db_postclip']):.2f}\nsigma {float(row['sigma']):.4f}",
            fill=(0, 0, 0),
        )
    canvas.save(root / "panels" / "selection_examples.png")


def cycle_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


VARIANT_STAGE_ALIASES = {
    "downstream_full": ("tdncnn", "full"),
    "downstream_topk": ("tdncnn", "topk_10"),
    "downstream_quantile": ("tdncnn", "quantile_10"),
    "downstream_random": ("tdncnn", "random_10"),
    "tdncnn_full": ("tdncnn", "full"),
    "tdncnn_topk": ("tdncnn", "topk_p10"),
    "tdncnn_quantile": ("tdncnn", "quantile_p10"),
    "tdncnn_random": ("tdncnn", "random_p10"),
    "tdncnn_topk_p10": ("tdncnn", "topk_p10"),
    "tdncnn_quantile_p10": ("tdncnn", "quantile_p10"),
    "tdncnn_random_p10": ("tdncnn", "random_p10"),
    "tdncnn_topk_p20": ("tdncnn", "topk_p20"),
    "tdncnn_topk_p30": ("tdncnn", "topk_p30"),
    "tdncnn_quantile_p20": ("tdncnn", "quantile_p20"),
    "tdncnn_quantile_p30": ("tdncnn", "quantile_p30"),
    "tdncnn_random_p20": ("tdncnn", "random_p20"),
    "tdncnn_random_p30": ("tdncnn", "random_p30"),
    "drunet_full": ("drunet", "full"),
    "drunet_topk": ("drunet", "topk_p10"),
    "drunet_quantile": ("drunet", "quantile_p10"),
    "drunet_random": ("drunet", "random_p10"),
    "drunet_topk_p10": ("drunet", "topk_p10"),
    "drunet_quantile_p10": ("drunet", "quantile_p10"),
    "drunet_random_p10": ("drunet", "random_p10"),
    "drunet_topk_p20": ("drunet", "topk_p20"),
    "drunet_topk_p30": ("drunet", "topk_p30"),
    "drunet_quantile_p20": ("drunet", "quantile_p20"),
    "drunet_quantile_p30": ("drunet", "quantile_p30"),
    "drunet_random_p20": ("drunet", "random_p20"),
    "drunet_random_p30": ("drunet", "random_p30"),
}


def downstream_variant_name(stage: str) -> str:
    if stage in VARIANT_STAGE_ALIASES:
        return VARIANT_STAGE_ALIASES[stage][1]
    return {
        "downstream_full": "full",
        "downstream_topk": "topk_10",
        "downstream_quantile": "quantile_10",
        "downstream_random": "random_10",
    }[stage]


def downstream_model_name(stage: str) -> str:
    if stage in VARIANT_STAGE_ALIASES:
        return VARIANT_STAGE_ALIASES[stage][0]
    raise KeyError(stage)


def downstream_model_cfg(ctx: PipelineContext, model_name: str) -> dict[str, Any]:
    if model_name == "drunet":
        merged = dict(ctx.cfg["downstream"])
        merged.update(dict(ctx.cfg.get("drunet", {})))
        return merged
    return dict(ctx.cfg["downstream"])


def downstream_num_workers(ctx: PipelineContext, model_name: str, phase: str) -> int:
    if ctx.smoke:
        return 0
    model_cfg = downstream_model_cfg(ctx, model_name)
    value = model_cfg.get(f"{phase}_num_workers", model_cfg.get("num_workers", ctx.cfg["runtime"]["num_workers"]))
    return max(0, int(value))


def downstream_validation_batch_size(ctx: PipelineContext, model_name: str, train_batch_size: int) -> int:
    model_cfg = downstream_model_cfg(ctx, model_name)
    key = "validation_batch_size_smoke" if ctx.smoke else "validation_batch_size"
    return max(1, int(model_cfg.get(key, train_batch_size)))


def build_downstream_model(ctx: PipelineContext, model_name: str, device: torch.device) -> nn.Module:
    channels = int(ctx.cfg["dataset"]["channels"])
    if model_name == "tdncnn":
        return DnCNN_2D(in_channels=channels).to(device)
    if model_name == "drunet":
        cfg = downstream_model_cfg(ctx, model_name)
        official = bool(cfg.get("official", True))
        if ctx.smoke and not ctx.real_smoke:
            official = bool(cfg.get("official_smoke", False))
        model = build_drunet(
            in_channels=channels,
            features=int(cfg.get("features", 64)),
            num_layers=int(cfg.get("num_layers", 5)),
            official=official,
            nc=cfg.get("nc"),
            nb=int(cfg.get("nb", 4)),
            act_mode=str(cfg.get("act_mode", "R")),
            downsample_mode=str(cfg.get("downsample_mode", "strideconv")),
            upsample_mode=str(cfg.get("upsample_mode", "convtranspose")),
        )
        return model.to(device)
    raise ValueError(f"Unsupported downstream model: {model_name}")


def downstream_forward(model_name: str, model: nn.Module, x: torch.Tensor, sigma: torch.Tensor | float | None) -> torch.Tensor:
    if model_name == "drunet":
        if sigma is None:
            raise ValueError("DRUNet downstream forward requires sigma")
        return model(x, sigma=sigma)
    return model(x)


def load_downstream_state(model: nn.Module, checkpoint: Path, device: torch.device) -> None:
    payload = torch.load(checkpoint, map_location=device)
    state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(payload)!r}")
    if any(str(key).startswith("module.") for key in state_dict):
        state_dict = {str(key).removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)


def downstream_training_protocol(ctx: PipelineContext, model_name: str, variant: str) -> dict[str, Any]:
    model_cfg = downstream_model_cfg(ctx, model_name)
    early_cfg = dict(model_cfg.get("early_stopping", {}))
    subset_path = ctx.stage_root("filter") / "subset_indices" / f"{variant}_indices.npy"
    init_path = ctx.experiment_root / "06_downstream" / model_name / "initial_state.pt"
    return {
        "protocol_version": "imagenet100_downstream_v2_common_task_noise",
        "dataset": {
            "name": str(ctx.cfg["dataset"]["name"]),
            "channels": int(ctx.cfg["dataset"]["channels"]),
            "image_size": int(ctx.cfg["dataset"]["image_size"]),
        },
        "model": model_name,
        "variant": variant,
        "subset_indices_sha256": sha256_file(subset_path) if subset_path.exists() else None,
        "initial_state_sha256": sha256_file(init_path) if init_path.exists() else None,
        "task_noise": dict(ctx.cfg["downstream"]["task_noise"]),
        "task_noise_seed_contract": "stable_int_seed(dataset, downstream_task_noise, original_index, task_noise_seed, version)",
        "max_steps": int(model_cfg["max_steps_smoke" if ctx.smoke else "max_steps"]),
        "batch_size": int(model_cfg["batch_size_smoke" if ctx.smoke else "batch_size"]),
        "lr": float(model_cfg["lr"]),
        "weight_decay": float(model_cfg.get("weight_decay", 0.0)),
        "validation_interval_steps": int(
            model_cfg.get("validation_interval_steps_smoke" if ctx.smoke else "validation_interval_steps", 500)
        ),
        "early_stopping": {
            "enabled": bool(early_cfg.get("enabled", False)),
            "monitor": str(early_cfg.get("monitor", "")),
            "mode": str(early_cfg.get("mode", "")),
            "patience": int(early_cfg.get("patience_smoke" if ctx.smoke else "patience", early_cfg.get("patience", 0))),
            "min_delta": float(early_cfg.get("min_delta", 1e-4)),
            "warmup_steps": int(
                early_cfg.get("warmup_steps_smoke" if ctx.smoke else "warmup_steps", early_cfg.get("warmup_steps", 0))
            ),
        },
    }


def downstream_training_protocol_hash(ctx: PipelineContext, model_name: str, variant: str) -> str:
    return config_hash(downstream_training_protocol(ctx, model_name, variant))


def downstream_best_matches_history(root: Path) -> bool:
    history_path = root / "training_history.csv"
    best_path = root / "best.pt"
    if not history_path.exists() or not best_path.exists():
        return False
    try:
        history = pd.read_csv(history_path)
    except Exception:
        return False
    if "val_loss" not in history.columns or "step" not in history.columns:
        return False
    val = pd.to_numeric(history["val_loss"], errors="coerce")
    finite = history[np.isfinite(val)].copy()
    if finite.empty:
        return False
    finite["_val_loss_numeric"] = pd.to_numeric(finite["val_loss"], errors="coerce")
    best_row = finite.loc[finite["_val_loss_numeric"].idxmin()]
    best_loss = float(best_row["_val_loss_numeric"])
    best_step = int(best_row["step"])
    try:
        payload = torch.load(best_path, map_location="cpu")
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    ckpt_loss = payload.get("best_val_loss")
    ckpt_step = payload.get("step")
    if ckpt_loss is None or ckpt_step is None:
        return False
    return int(ckpt_step) == best_step and math.isclose(float(ckpt_loss), best_loss, rel_tol=1e-8, abs_tol=1e-10)


@torch.no_grad()
def downstream_validation_loss(
    ctx: PipelineContext,
    model_name: str,
    model: nn.Module,
    device: torch.device,
    *,
    batch_size: int,
) -> float:
    val_ds = build_dataset(ctx.cfg, split="val", smoke=ctx.smoke)
    indices_path = ctx.experiment_root / "00_protocol" / "downstream_validation_indices.npy"
    if indices_path.exists():
        indices = np.load(indices_path).astype(np.int64)
    else:
        eval_path = ctx.experiment_root / "00_protocol" / "downstream_evaluation_indices.npy"
        exclude = np.load(eval_path).astype(np.int64).tolist() if eval_path.exists() else []
        indices = class_stratified_indices_excluding(
            val_ds,
            per_class=int(ctx.cfg["downstream"].get("validation_samples_per_class", 1)),
            seed=int(ctx.cfg["seeds"]["split_seed"]) + 2003,
            exclude=exclude,
        )
        np.save(indices_path, indices)
    if ctx.real_smoke:
        indices = indices[: min(int(ctx.cfg["real_smoke"]["evaluation_images"]), len(indices))]
    dataset = DownstreamValidationDataset(val_ds, indices, ctx.cfg, split="downstream_validation")
    loader = DataLoader(
        dataset,
        batch_size=downstream_validation_batch_size(ctx, model_name, batch_size),
        shuffle=False,
        num_workers=downstream_num_workers(ctx, model_name, "validation"),
    )
    criterion = nn.MSELoss(reduction="sum")
    model.eval()
    total = 0.0
    count = 0
    for x_in, x_tgt, sigma, _original in loader:
        x_in = x_in.to(device)
        x_tgt = x_tgt.to(device)
        sigma = sigma.to(device)
        pred = downstream_forward(model_name, model, x_in, sigma)
        total += float(criterion(pred, x_tgt).item())
        count += int(np.prod(x_tgt.shape))
    model.train()
    return total / max(count, 1)


def downstream_artifacts_complete(ctx: PipelineContext, stage: str, model_name: str, variant: str) -> bool:
    root = ctx.stage_root(stage)
    best = root / "best.pt"
    history = root / "training_history.csv"
    meta_path = root / "run_meta.json"
    last = root / "last.pt"
    if not (best.exists() and history.exists() and meta_path.exists() and last.exists()):
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if str(meta.get("model")) != model_name or str(meta.get("variant")) != variant:
        return False
    if bool(meta.get("early_stopping_enabled", False)) is not True:
        return False
    if str(meta.get("early_stopping_monitor", "val_loss")) != "val_loss":
        return False
    if str(meta.get("early_stopping_mode", "min")) != "min":
        return False
    model_cfg = downstream_model_cfg(ctx, model_name)
    early_cfg = dict(model_cfg.get("early_stopping", {}))
    if int(meta.get("max_optimizer_steps", -1)) != int(model_cfg["max_steps_smoke" if ctx.smoke else "max_steps"]):
        return False
    if int(meta.get("validation_interval_steps", -1)) != int(
        model_cfg.get("validation_interval_steps_smoke" if ctx.smoke else "validation_interval_steps", 500)
    ):
        return False
    if int(meta.get("early_stopping_patience", -1)) != int(
        early_cfg.get("patience_smoke" if ctx.smoke else "patience", early_cfg.get("patience", 10))
    ):
        return False
    if not math.isclose(float(meta.get("early_stopping_min_delta", float("nan"))), float(early_cfg.get("min_delta", 1e-4)), rel_tol=0.0, abs_tol=1e-12):
        return False
    if int(meta.get("early_stopping_warmup_steps", -1)) != int(
        early_cfg.get("warmup_steps_smoke" if ctx.smoke else "warmup_steps", early_cfg.get("warmup_steps", 0))
    ):
        return False
    expected_protocol_hash = downstream_training_protocol_hash(ctx, model_name, variant)
    if str(meta.get("training_protocol_hash", "")) != expected_protocol_hash:
        return False
    try:
        payload = torch.load(best, map_location="cpu")
    except Exception:
        return False
    if not isinstance(payload, dict) or str(payload.get("training_protocol_hash", "")) != expected_protocol_hash:
        return False
    if not downstream_best_matches_history(root):
        return False
    return True


def stage_downstream(ctx: PipelineContext, stage: str) -> None:
    start = time.time()
    model_name = downstream_model_name(stage)
    variant = downstream_variant_name(stage)
    ckpt = ctx.stage_root(stage) / "best.pt"
    if ctx.resume and downstream_artifacts_complete(ctx, stage, model_name, variant):
        print(f"[resume] {stage} compatible downstream artifacts found: {ckpt}")
        return
    if ctx.dry_run:
        print(f"[dry-run] would train downstream {model_name} {variant} into {ctx.stage_root(stage)}")
        return
    require_selected(ctx)
    ctx.stage_root(stage).mkdir(parents=True, exist_ok=True)
    train_ds = build_dataset(ctx.cfg, split="train", smoke=ctx.smoke)
    manifest = pd.read_csv(ctx.stage_root("prepare") / "corpus_manifest.csv")
    indices_path = ctx.stage_root("filter") / "subset_indices" / f"{variant}_indices.npy"
    indices = np.load(indices_path).astype(np.int64)
    corpus = OnDemandCorpusDataset(train_ds, manifest, original_indices=indices, return_index=True)
    train_data = CorpusTaskNoiseDataset(corpus, cfg=ctx.cfg, variant=variant)
    model_cfg = downstream_model_cfg(ctx, model_name)
    loader = DataLoader(
        train_data,
        batch_size=int(model_cfg["batch_size_smoke" if ctx.smoke else "batch_size"]),
        shuffle=True,
        num_workers=downstream_num_workers(ctx, model_name, "train"),
        generator=torch.Generator().manual_seed(int(ctx.cfg["seeds"]["model_seed"])),
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not ctx.smoke else "cpu")
    init_path = ctx.experiment_root / "06_downstream" / model_name / "initial_state.pt"
    torch.manual_seed(int(ctx.cfg["seeds"]["model_seed"]))
    model = build_downstream_model(ctx, model_name, device)
    if not init_path.exists():
        init_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), init_path)
    load_downstream_state(model, init_path, device)
    training_protocol = downstream_training_protocol(ctx, model_name, variant)
    training_protocol_hash = config_hash(training_protocol)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(model_cfg["lr"]),
        weight_decay=float(model_cfg.get("weight_decay", 0.0)),
    )
    criterion = nn.MSELoss()
    max_steps = int(model_cfg["max_steps_smoke" if ctx.smoke else "max_steps"])
    batch_size = int(model_cfg["batch_size_smoke" if ctx.smoke else "batch_size"])
    log_interval = max(1, int(model_cfg.get("log_interval", 100)))
    val_interval = max(1, int(model_cfg.get("validation_interval_steps_smoke" if ctx.smoke else "validation_interval_steps", 500)))
    early_cfg = dict(model_cfg.get("early_stopping", {}))
    patience = int(early_cfg.get("patience_smoke" if ctx.smoke else "patience", early_cfg.get("patience", 10)))
    min_delta = float(early_cfg.get("min_delta", 1e-4))
    warmup_steps = int(early_cfg.get("warmup_steps_smoke" if ctx.smoke else "warmup_steps", early_cfg.get("warmup_steps", 0)))
    iterator = cycle_loader(loader)
    rows = []
    best_val_loss = float("inf")
    patience_best = float("inf")
    best_step = 0
    checks_without_improvement = 0
    stop_reason = "max_steps"
    history_path = ctx.stage_root(stage) / "training_history.csv"
    progress_path = ctx.stage_root(stage) / "progress.jsonl"
    progress_path.unlink(missing_ok=True)
    downstream_log(
        ctx.stage_root(stage),
        "start",
        stage=stage,
        model=model_name,
        variant=variant,
        subset_size=int(len(indices)),
        max_steps=int(max_steps),
        batch_size=int(batch_size),
        train_num_workers=int(downstream_num_workers(ctx, model_name, "train")),
        validation_batch_size=int(downstream_validation_batch_size(ctx, model_name, batch_size)),
        validation_num_workers=int(downstream_num_workers(ctx, model_name, "validation")),
        early_stopping_enabled=True,
        early_stopping_monitor="val_loss",
        early_stopping_patience=int(patience),
        validation_interval_steps=int(val_interval),
        log_interval=int(log_interval),
        device=str(device),
    )
    progress = tqdm(range(1, max_steps + 1), desc=f"{stage} {model_name} {variant}", total=max_steps)
    for step in progress:
        x_in, x_tgt, sigma, _original = next(iterator)
        x_in = x_in.to(device)
        x_tgt = x_tgt.to(device)
        sigma = sigma.to(device)
        pred = downstream_forward(model_name, model, x_in, sigma)
        loss = criterion(pred, x_tgt)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu().item())
        val_loss = float("nan")
        if step == 1 or step == max_steps or step % val_interval == 0:
            val_loss = downstream_validation_loss(ctx, model_name, model, device, batch_size=batch_size)
            # Save every validation minimum, including improvements below min_delta.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = int(step)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "step": step,
                        "variant": variant,
                        "best_val_loss": best_val_loss,
                        "training_protocol_hash": training_protocol_hash,
                    },
                    ckpt,
                )
            # Warmup/min_delta affect stopping patience, not which checkpoint is best.
            if step >= warmup_steps and val_loss < patience_best - min_delta:
                patience_best = val_loss
                checks_without_improvement = 0
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
        progress.set_postfix(loss=f"{loss_value:.6f}", val=f"{val_loss:.6f}" if np.isfinite(val_loss) else "nan", best=f"{best_val_loss:.6f}")
        if step == 1 or step == max_steps or step % log_interval == 0 or np.isfinite(val_loss):
            pd.DataFrame(rows).to_csv(history_path, index=False)
            elapsed = time.time() - start
            downstream_log(
                ctx.stage_root(stage),
                "step",
                stage=stage,
                model=model_name,
                variant=variant,
                step=int(step),
                max_steps=int(max_steps),
                train_loss=loss_value,
                val_loss=val_loss,
                best_val_loss=float(best_val_loss),
                best_step=int(best_step),
                patience_counter=int(checks_without_improvement),
                elapsed_sec=round(elapsed, 3),
                steps_per_sec=round(float(step) / max(elapsed, EPS), 4),
                cuda_allocated_mb=float(torch.cuda.memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
                cuda_reserved_mb=float(torch.cuda.memory_reserved(device) / 1024**2) if device.type == "cuda" else 0.0,
            )
            training_progress(
                "ImageNet-100",
                f"{model_name}_{variant}",
                unit="step",
                current=step,
                total=max_steps,
                train_loss=loss_value,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                best_step=best_step,
                patience_counter=checks_without_improvement,
                patience=patience,
                lr=optimizer.param_groups[0]["lr"],
                start_time=start,
            )
        if stop_reason == "early_stopping":
            break
    if not ckpt.exists():
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "step": step,
                "variant": variant,
                "best_val_loss": best_val_loss,
                "training_protocol_hash": training_protocol_hash,
            },
            ckpt,
        )
    pd.DataFrame(rows).to_csv(history_path, index=False)
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "step": step}, ctx.stage_root(stage) / "last.pt")
    finite_val = pd.DataFrame(rows)["val_loss"].dropna()
    final_val_loss = float(finite_val.iloc[-1]) if len(finite_val) else float("nan")
    train_time_seconds = float(time.time() - start)
    write_json(
        ctx.stage_root(stage) / "run_meta.json",
        {
            "model": model_name,
            "variant": variant,
            "subset_size": int(len(indices)),
            "max_optimizer_steps": int(max_steps),
            "optimizer_steps": int(step),
            "stopped_step": int(step),
            "initial_state": str(init_path),
            "initial_state_sha256": sha256_file(init_path),
            "sample_exposures": int(step * int(model_cfg["batch_size_smoke" if ctx.smoke else "batch_size"])),
            "task_noise": ctx.cfg["downstream"]["task_noise"],
            "early_stopping_enabled": True,
            "early_stopping_monitor": "val_loss",
            "early_stopping_mode": "min",
            "early_stopping_patience": int(patience),
            "early_stopping_min_delta": float(min_delta),
            "early_stopping_warmup_steps": int(warmup_steps),
            "validation_interval_steps": int(val_interval),
            "best_step": int(best_step),
            "best_val_loss": float(best_val_loss),
            "final_val_loss": final_val_loss,
            "stop_reason": stop_reason,
            "effective_epochs_at_best": float(best_step * batch_size / max(1, len(indices))),
            "effective_epochs_at_stop": float(step * batch_size / max(1, len(indices))),
            "total_seen_samples_at_best": int(best_step * batch_size),
            "train_time_seconds": train_time_seconds,
            "optimizer": "Adam",
            "lr": float(model_cfg["lr"]),
            "scheduler": "none",
            "batch_size": int(batch_size),
            "train_num_workers": int(downstream_num_workers(ctx, model_name, "train")),
            "validation_batch_size": int(downstream_validation_batch_size(ctx, model_name, batch_size)),
            "validation_num_workers": int(downstream_num_workers(ctx, model_name, "validation")),
            "training_protocol": training_protocol,
            "training_protocol_hash": training_protocol_hash,
        },
    )
    downstream_log(
        ctx.stage_root(stage),
        "complete",
        stage=stage,
        model=model_name,
        variant=variant,
        final_step=int(step),
        max_steps=int(max_steps),
        best_step=int(best_step),
        best_val_loss=float(best_val_loss),
        stop_reason=stop_reason,
        elapsed_sec=round(train_time_seconds, 3),
        best_checkpoint=str(ckpt),
        training_history=str(history_path),
    )
    ctx.write_marker(stage, start, [ckpt, history_path])


def create_eval_cache(ctx: PipelineContext) -> tuple[Path, Path, Path]:
    root = ctx.experiment_root / "07_evaluation" / "combined"
    root.mkdir(parents=True, exist_ok=True)
    input_path = root / "eval_input.pt"
    target_path = root / "eval_target.pt"
    manifest_path = root / "eval_noise_manifest.csv"
    if input_path.exists() and target_path.exists() and manifest_path.exists():
        return input_path, target_path, manifest_path
    val_ds = build_dataset(ctx.cfg, split="val", smoke=ctx.smoke)
    indices_path = ctx.experiment_root / "00_protocol" / "downstream_evaluation_indices.npy"
    indices = np.load(indices_path).astype(np.int64) if indices_path.exists() else np.arange(min(16, len(val_ds)), dtype=np.int64)
    if ctx.real_smoke:
        indices = indices[: min(int(ctx.cfg["real_smoke"]["evaluation_images"]), len(indices))]
    # Save one noisy input per image and use it for all models and subsets.
    inputs = []
    targets = []
    rows = []
    noise_cfg = ctx.cfg["evaluation"]["eval_noise"]
    for pos, original_index in enumerate(indices.tolist()):
        x_clean, class_id = val_ds[int(original_index)]
        x_clean = x_clean.to(torch.float32).clamp(0, 1)
        sigma = float(noise_cfg["sigma"])
        seed = stable_int_seed(ctx.cfg["dataset"]["name"], "eval_noise", int(original_index), int(noise_cfg["seed"]), str(noise_cfg["version"]))
        gen = torch.Generator().manual_seed(seed)
        noise = torch.randn(x_clean.shape, generator=gen, dtype=x_clean.dtype) * sigma
        x_input = (x_clean + noise).clamp(0, 1)
        inputs.append(x_input)
        targets.append(x_clean)
        rows.append({"sample_index": pos, "original_index": int(original_index), "class_id": int(class_id), "sigma": sigma, "sample_seed": int(seed)})
    torch.save(torch.stack(inputs, dim=0), input_path)
    torch.save(torch.stack(targets, dim=0), target_path)
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    write_json(root / "eval_noise_cache_meta.json", {"input_sha256": sha256_file(input_path), "target_sha256": sha256_file(target_path), "manifest_sha256": sha256_file(manifest_path)})
    return input_path, target_path, manifest_path


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


def eval_metrics(pred: torch.Tensor, target: torch.Tensor, lpips_model) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    mse = ((pred - target) ** 2).flatten(start_dim=1).mean(dim=1)
    psnr = 10.0 * torch.log10(1.0 / (mse + EPS))
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    ssim = np.asarray([skimage_ssim(target_np[i], pred_np[i], data_range=1.0, channel_axis=0) for i in range(pred_np.shape[0])])
    if lpips_model is None:
        lp = np.full(pred_np.shape[0], np.nan)
    else:
        with torch.inference_mode():
            lp = lpips_model(prepare_for_lpips(pred), prepare_for_lpips(target)).detach().flatten().cpu().numpy()
    return mse.detach().cpu().numpy(), psnr.detach().cpu().numpy(), ssim, lp


def model_variant_stages(model_name: str) -> list[tuple[str, str, str]]:
    prefix = "tdncnn" if model_name == "tdncnn" else "drunet"
    stages = [
        ("Full", "full", f"{prefix}_full"),
        ("Top-K 10%", "topk_p10", f"{prefix}_topk"),
        ("Top-K 20%", "topk_p20", f"{prefix}_topk_p20"),
        ("Top-K 30%", "topk_p30", f"{prefix}_topk_p30"),
        ("Quantile 10%", "quantile_p10", f"{prefix}_quantile"),
        ("Quantile 20%", "quantile_p20", f"{prefix}_quantile_p20"),
        ("Quantile 30%", "quantile_p30", f"{prefix}_quantile_p30"),
        ("Random 10%", "random_p10", f"{prefix}_random"),
        ("Random 20%", "random_p20", f"{prefix}_random_p20"),
        ("Random 30%", "random_p30", f"{prefix}_random_p30"),
    ]
    return stages


def variant_table_fields(label: str, variant: str) -> tuple[str, float]:
    strategy, fraction, _display = display_variant(variant)
    return strategy, fraction


def grid_downstream_stages(model_name: str) -> list[str]:
    prefix = "tdncnn" if model_name == "tdncnn" else "drunet"
    return [
        f"{prefix}_full",
        f"{prefix}_topk",
        f"{prefix}_topk_p20",
        f"{prefix}_topk_p30",
        f"{prefix}_quantile",
        f"{prefix}_quantile_p20",
        f"{prefix}_quantile_p30",
        f"{prefix}_random",
        f"{prefix}_random_p20",
        f"{prefix}_random_p30",
    ]


def stage_downstream_grid(ctx: PipelineContext, model_name: str) -> None:
    stage = f"{model_name}_grid"
    start = time.time()
    branch_stages = grid_downstream_stages(model_name)
    required = [ctx.stage_root(item) / "best.pt" for item in branch_stages]
    branch_specs = [(item, downstream_model_name(item), downstream_variant_name(item)) for item in branch_stages]
    if ctx.resume and ctx.marker_valid(stage, required) and all(
        downstream_artifacts_complete(ctx, item, branch_model, branch_variant)
        for item, branch_model, branch_variant in branch_specs
    ):
        print(f"[resume] {stage} already complete")
        return
    if ctx.dry_run:
        print(f"[dry-run] would run {stage}: {', '.join(branch_stages)}")
        return
    for item in branch_stages:
        stage_downstream(ctx, item)
    ctx.write_marker(stage, start, required)


def stage_downstream_early_stopping_audit(ctx: PipelineContext) -> None:
    stage = "downstream_early_stopping_audit"
    if ctx.dry_run:
        print("[dry-run] would audit downstream early-stopping protocol")
        return
    start = time.time()
    out_dir = ctx.stage_root(stage)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "downstream_early_stopping_audit.md"
    root_copy = ctx.experiment_root / "00_protocol" / "downstream_early_stopping_audit.md"
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for model_name, architecture in [("tdncnn", "TDnCNN"), ("drunet", "DRUNet")]:
        model_cfg = downstream_model_cfg(ctx, model_name)
        early_cfg = dict(model_cfg.get("early_stopping", {}))
        enabled = bool(early_cfg.get("enabled", False))
        monitor = str(early_cfg.get("monitor", ""))
        mode = str(early_cfg.get("mode", ""))
        patience = int(early_cfg.get("patience_smoke" if ctx.smoke else "patience", early_cfg.get("patience", 0)))
        max_steps = int(model_cfg["max_steps_smoke" if ctx.smoke else "max_steps"])
        val_interval = int(model_cfg.get("validation_interval_steps_smoke" if ctx.smoke else "validation_interval_steps", 0))
        warmup_steps = int(early_cfg.get("warmup_steps_smoke" if ctx.smoke else "warmup_steps", early_cfg.get("warmup_steps", 0)))
        min_delta = float(early_cfg.get("min_delta", -1.0))
        for _label, variant, _stage_name in model_variant_stages(model_name):
            if not enabled:
                failures.append(f"{architecture} {variant}: early_stopping.enabled is not true")
            if monitor != "val_loss":
                failures.append(f"{architecture} {variant}: monitor is {monitor!r}, expected 'val_loss'")
            if mode != "min":
                failures.append(f"{architecture} {variant}: mode is {mode!r}, expected 'min'")
            rows.append(
                {
                    "Architecture": architecture,
                    "Variant": variant,
                    "Early stopping": "ON" if enabled else "OFF",
                    "Monitor": monitor,
                    "Mode": mode,
                    "Patience": patience,
                    "Min delta": min_delta,
                    "Warmup steps": warmup_steps,
                    "Validation frequency": val_interval,
                    "Max steps": max_steps,
                    "Evaluated checkpoint": "best.pt",
                }
            )
    if failures:
        raise RuntimeError("Downstream early-stopping audit failed:\n" + "\n".join(failures))
    audit = pd.DataFrame(rows)
    audit.to_csv(out_dir / "downstream_early_stopping_audit.csv", index=False)
    headers = [
        "Architecture",
        "Variant",
        "Early stopping",
        "Monitor",
        "Mode",
        "Patience",
        "Min delta",
        "Warmup steps",
        "Validation frequency",
        "Max steps",
        "Evaluated checkpoint",
    ]
    lines = [
        "# Downstream Early Stopping Audit",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", "---", "---", "---", "---", "---:", "---:", "---:", "---:", "---:", "---"]) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[key]) for key in headers) + " |")
    lines.extend(
        [
            "",
            "All main ImageNet downstream branches use validation-controlled training.",
            "Internal and external evaluation stages load `best.pt`.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copyfile(md_path, root_copy)
    ctx.write_marker(stage, start, [md_path, root_copy, out_dir / "downstream_early_stopping_audit.csv"])


def stage_evaluate_grid(ctx: PipelineContext) -> None:
    stage = "evaluate_grid"
    start = time.time()
    combined = ctx.experiment_root / "07_evaluation" / "combined" / "subset_grid_results_all.csv"
    if (
        ctx.resume
        and ctx.marker_valid(stage, [combined])
        and evaluation_artifacts_complete(ctx, "tdncnn")
        and evaluation_artifacts_complete(ctx, "drunet")
    ):
        print(f"[resume] evaluate_grid already complete: {combined}")
        return
    if ctx.dry_run:
        print("[dry-run] would evaluate TDnCNN and DRUNet grid best.pt checkpoints")
        return
    stage_evaluate_model(ctx, "tdncnn")
    stage_evaluate_model(ctx, "drunet")
    if not combined.exists():
        write_combined_internal_if_ready(ctx)
    ctx.write_marker(stage, start, [combined])


def evaluation_protocol(ctx: PipelineContext, model_name: str) -> dict[str, Any]:
    checkpoints = {}
    for _label, variant, dstage in model_variant_stages(model_name):
        path = ctx.stage_root(dstage) / "best.pt"
        checkpoints[variant] = sha256_file(path) if path.exists() else None
    cache_meta_path = ctx.experiment_root / "07_evaluation" / "combined" / "eval_noise_cache_meta.json"
    cache_meta = json.loads(cache_meta_path.read_text(encoding="utf-8")) if cache_meta_path.exists() else {}
    return {
        "protocol_version": "imagenet100_eval_v1_best_checkpoints",
        "model": model_name,
        "evaluation": dict(ctx.cfg["evaluation"]),
        "best_checkpoint_sha256": checkpoints,
        "eval_cache": cache_meta,
    }


def evaluation_protocol_hash(ctx: PipelineContext, model_name: str) -> str:
    return config_hash(evaluation_protocol(ctx, model_name))


def evaluation_artifacts_complete(ctx: PipelineContext, model_name: str) -> bool:
    stage = f"{model_name}_evaluate"
    root = ctx.stage_root(stage)
    summary = root / "summary_table.csv"
    per_image = root / "per_image_metrics.csv"
    meta_path = root / "evaluation_meta.json"
    grid_path = ctx.experiment_root / "07_evaluation" / f"subset_grid_results_{model_name}.csv"
    if not (summary.exists() and per_image.exists() and meta_path.exists() and grid_path.exists()):
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if str(meta.get("evaluation_protocol_hash", "")) != evaluation_protocol_hash(ctx, model_name):
        return False
    return all(
        downstream_artifacts_complete(ctx, dstage, model_name, variant)
        for _label, variant, dstage in model_variant_stages(model_name)
    )


def write_combined_internal_if_ready(ctx: PipelineContext) -> None:
    out_root = ctx.experiment_root / "07_evaluation" / "combined"
    tdn = ctx.stage_root("tdncnn_evaluate") / "summary_table.csv"
    dru = ctx.stage_root("drunet_evaluate") / "summary_table.csv"
    if not (tdn.exists() and dru.exists()):
        return
    out_root.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([pd.read_csv(tdn), pd.read_csv(dru)], ignore_index=True)
    combined.to_csv(out_root / "internal_combined_table.csv", index=False)
    grid_paths = [ctx.experiment_root / "07_evaluation" / "subset_grid_results_tdncnn.csv", ctx.experiment_root / "07_evaluation" / "subset_grid_results_drunet.csv"]
    if all(path.exists() for path in grid_paths):
        grid = pd.concat([pd.read_csv(path) for path in grid_paths], ignore_index=True)
        grid.to_csv(out_root / "subset_grid_results_all.csv", index=False)
    for model_name in ["tdncnn", "drunet"]:
        src = ctx.stage_root(f"{model_name}_evaluate") / "summary_table.csv"
        shutil.copyfile(src, out_root / f"{model_name}_internal_results.csv")


def stage_evaluate_model(ctx: PipelineContext, model_name: str) -> None:
    stage = f"{model_name}_evaluate"
    start = time.time()
    ctx.stage_root(stage).mkdir(parents=True, exist_ok=True)
    (ctx.experiment_root / "07_evaluation").mkdir(parents=True, exist_ok=True)
    summary = ctx.stage_root(stage) / "summary_table.csv"
    if ctx.resume and ctx.marker_valid(stage, [summary]) and evaluation_artifacts_complete(ctx, model_name):
        print(f"[resume] {stage} already complete: {summary}")
        return
    if ctx.dry_run:
        print(f"[dry-run] would evaluate {model_name} downstream variants into {ctx.stage_root(stage)}")
        return
    input_path, target_path, manifest_path = create_eval_cache(ctx)
    loader = DataLoader(EvalNoiseDataset(input_path, target_path), batch_size=int(ctx.cfg["evaluation"]["batch_size"]), shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() and not ctx.smoke else "cpu")
    lpips_model = create_lpips_model(device, bool(ctx.cfg["evaluation"].get("lpips_enabled", True)))
    rows = []
    fixed_sigma = float(ctx.cfg["evaluation"]["eval_noise"]["sigma"])
    for label, variant, dstage in model_variant_stages(model_name):
        model = build_downstream_model(ctx, model_name, device)
        load_downstream_state(model, ctx.stage_root(dstage) / "best.pt", device)
        model.eval()
        offset = 0
        with torch.inference_mode():
            for x_input, x_target in loader:
                x_input = x_input.to(device)
                x_target = x_target.to(device)
                sigma = torch.full((x_input.shape[0],), fixed_sigma, dtype=x_input.dtype, device=device)
                pred = downstream_forward(model_name, model, x_input, sigma)
                mse, psnr, ssim, lp = eval_metrics(pred, x_target, lpips_model)
                for i in range(len(mse)):
                    rows.append(
                        {
                            "Architecture": model_name,
                            "Variant": label,
                            "variant_key": variant,
                            "Strategy": variant_table_fields(label, variant)[0],
                            "Fraction": variant_table_fields(label, variant)[1],
                            "sample_index": offset + i,
                            "MSE": float(mse[i]),
                            "PSNR": float(psnr[i]),
                            "SSIM": float(ssim[i]),
                            "LPIPS": float(lp[i]),
                        }
                    )
                offset += len(mse)
    per_image = pd.DataFrame(rows)
    per_image.to_csv(ctx.stage_root(stage) / "per_image_metrics.csv", index=False)
    agg = per_image.groupby(["Architecture", "Strategy", "Fraction", "Variant", "variant_key"])[["PSNR", "SSIM", "LPIPS", "MSE"]].agg(["mean", "std", "median"]).reset_index()
    flat_cols = ["Architecture", "Strategy", "Fraction", "Variant", "variant_key"] + [f"{a}_{b}" for a, b in agg.columns.tolist()[5:]]
    agg.columns = flat_cols
    full_psnr = float(agg[agg["Variant"] == "Full"]["PSNR_mean"].iloc[0])
    train_sizes = {}
    best_steps = {}
    effective_epochs = {}
    for label, _variant, dstage in model_variant_stages(model_name):
        meta = json.loads((ctx.stage_root(dstage) / "run_meta.json").read_text(encoding="utf-8"))
        train_sizes[label] = int(meta["subset_size"])
        best_steps[label] = int(meta.get("best_step", meta.get("optimizer_steps", 0)))
        effective_epochs[label] = float(meta.get("effective_epochs_at_best", 0.0))
    agg["Train size"] = agg["Variant"].map(train_sizes).astype(int)
    agg["Best step"] = agg["Variant"].map(best_steps).astype(int)
    agg["Effective epochs"] = agg["Variant"].map(effective_epochs).astype(float)
    agg["Delta PSNR vs model Full"] = agg["PSNR_mean"] - full_psnr
    grid = agg[
        [
            "Architecture",
            "Strategy",
            "Fraction",
            "Train size",
            "Best step",
            "Effective epochs",
            "PSNR_mean",
            "SSIM_mean",
            "LPIPS_mean",
            "MSE_mean",
            "Delta PSNR vs model Full",
            "Variant",
            "variant_key",
        ]
    ].rename(
        columns={
            "PSNR_mean": "PSNR",
            "SSIM_mean": "SSIM",
            "LPIPS_mean": "LPIPS",
            "MSE_mean": "MSE",
            "Delta PSNR vs model Full": "Delta PSNR vs Full",
        }
    )
    agg.to_csv(summary, index=False)
    grid_name = f"subset_grid_results_{model_name}.csv"
    grid.to_csv(ctx.experiment_root / "07_evaluation" / grid_name, index=False)
    grid.to_csv(ctx.stage_root(stage) / grid_name, index=False)
    write_json(
        ctx.stage_root(stage) / "evaluation_meta.json",
        {
            "model": model_name,
            "evaluation_protocol": evaluation_protocol(ctx, model_name),
            "evaluation_protocol_hash": evaluation_protocol_hash(ctx, model_name),
        },
    )
    write_combined_internal_if_ready(ctx)
    ctx.write_marker(stage, start, [summary, ctx.stage_root(stage) / "per_image_metrics.csv", ctx.stage_root(stage) / "evaluation_meta.json"])


def downstream_stage_for_variant(model_name: str, variant: str) -> str:
    prefix = "tdncnn" if model_name == "tdncnn" else "drunet"
    mapping = {
        "full": f"{prefix}_full",
        "topk_10": f"{prefix}_topk",
        "quantile_10": f"{prefix}_quantile",
        "random_10": f"{prefix}_random",
        "topk_p10": f"{prefix}_topk",
        "quantile_p10": f"{prefix}_quantile",
        "random_p10": f"{prefix}_random",
    }
    if variant in mapping:
        return mapping[variant]
    strategy, fraction = variant_strategy_fraction(variant)
    tag = fraction_tag(fraction)
    if strategy == "topk":
        return f"{prefix}_topk_{tag}"
    if strategy == "quantile":
        return f"{prefix}_quantile_{tag}"
    if strategy == "random":
        return f"{prefix}_random_{tag}"
    raise ValueError(f"No downstream stage for {model_name} {variant}")


@torch.no_grad()
def predict_downstream_variant(ctx: PipelineContext, model_name: str, variant: str, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    stage = downstream_stage_for_variant(model_name, variant)
    model = build_downstream_model(ctx, model_name, device)
    load_downstream_state(model, ctx.stage_root(stage) / "best.pt", device)
    model.eval()
    x_in = x.unsqueeze(0).to(device)
    fixed_sigma = float(ctx.cfg["evaluation"]["eval_noise"]["sigma"])
    sigma = torch.full((1,), fixed_sigma, dtype=x_in.dtype, device=device)
    pred = downstream_forward(model_name, model, x_in, sigma)
    return pred.squeeze(0).cpu().clamp(0.0, 1.0)


def metric_for_sample(per: pd.DataFrame, variant: str, sample_index: int) -> dict[str, float]:
    row = per[(per["variant_key"] == variant) & (per["sample_index"] == int(sample_index))]
    if row.empty:
        return {}
    item = row.iloc[0]
    return {"psnr": float(item["PSNR"]), "ssim": float(item["SSIM"]), "lpips": float(item["LPIPS"]), "mse": float(item["MSE"])}


def stage_figures_grid(ctx: PipelineContext) -> None:
    stage = "figures_grid"
    start = time.time()
    out = ctx.stage_root(stage)
    required = [out / "subset_fraction_vs_psnr_tdncnn.png", out / "subset_fraction_vs_psnr_drunet.png"]
    meta_path = out / "figures_meta.json"
    if ctx.resume and ctx.marker_valid(stage, required + [meta_path]) and figures_artifacts_complete(ctx):
        print(f"[resume] figures_grid already complete: {out}")
        return
    if ctx.dry_run:
        print(f"[dry-run] would write grid figures to {out}")
        return
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for model_name in ["tdncnn", "drunet"]:
        grid_path = ctx.experiment_root / "07_evaluation" / f"subset_grid_results_{model_name}.csv"
        if not grid_path.exists():
            raise FileNotFoundError(f"Missing grid evaluation table: {grid_path}")
        results = pd.read_csv(grid_path).rename(columns={"Architecture": "model", "Strategy": "strategy", "Fraction": "fraction"})
        results["strategy"] = results["strategy"].astype(str).str.lower().str.replace("-", "", regex=False)
        for metric, suffix in [("PSNR", "psnr"), ("SSIM", "ssim"), ("LPIPS", "lpips"), ("MSE", "mse")]:
            metric_path = out / f"subset_fraction_vs_{suffix}_{model_name}.png"
            plot_subset_fraction_metric(results, metric, metric_path)
            written.append(metric_path)

        per = pd.read_csv(ctx.stage_root(f"{model_name}_evaluate") / "per_image_metrics.csv")
        input_path, target_path, _manifest_path = create_eval_cache(ctx)
        inputs = torch.load(input_path, map_location="cpu").to(torch.float32)
        targets = torch.load(target_path, map_location="cpu").to(torch.float32)
        full_psnr = per[per["variant_key"] == "full"].sort_values("PSNR")
        image_indices = [int(full_psnr.iloc[0]["sample_index"]), int(full_psnr.iloc[len(full_psnr) // 2]["sample_index"]), int(full_psnr.iloc[-1]["sample_index"])] if len(full_psnr) >= 3 else [0]
        device = torch.device("cuda" if torch.cuda.is_available() and not ctx.smoke else "cpu")
        for fraction in filtering_fractions(ctx):
            tag = fraction_tag(fraction)
            percent = int(round(fraction * 100.0))
            topk = f"topk_{tag}"
            quantile = f"quantile_{tag}"
            random = f"random_{tag}"
            columns = ["Clean", "Noisy input", "Full", "Top-K", "Quantile", "Random"]
            samples = []
            for name, image_index in zip(["hard", "median", "easy"], image_indices):
                tensors = {
                    "Clean": targets[image_index],
                    "Noisy input": inputs[image_index],
                    "Full": predict_downstream_variant(ctx, model_name, "full", inputs[image_index], device),
                    "Top-K": predict_downstream_variant(ctx, model_name, topk, inputs[image_index], device),
                    "Quantile": predict_downstream_variant(ctx, model_name, quantile, inputs[image_index], device),
                    "Random": predict_downstream_variant(ctx, model_name, random, inputs[image_index], device),
                }
                metrics = {
                    "Full": metric_for_sample(per, "full", image_index),
                    "Top-K": metric_for_sample(per, topk, image_index),
                    "Quantile": metric_for_sample(per, quantile, image_index),
                    "Random": metric_for_sample(per, random, image_index),
                }
                samples.append({"name": name, "tensors": tensors, "metrics": metrics})
            panel_path = out / f"denoiser_panel_{model_name}_{tag}.png"
            save_denoiser_comparison_panel(samples, panel_path, columns=columns, title=f"{model_name} p{percent}")
            written.append(panel_path)

        topk_samples = []
        for name, image_index in zip(["hard", "median", "easy"], image_indices):
            tensors = {
                "Clean": targets[image_index],
                "Noisy": inputs[image_index],
                "Full": predict_downstream_variant(ctx, model_name, "full", inputs[image_index], device),
            }
            metrics = {"Full": metric_for_sample(per, "full", image_index)}
            for fraction in filtering_fractions(ctx):
                tag = fraction_tag(fraction)
                variant = f"topk_{tag}"
                label = f"Top-K {int(round(fraction * 100.0))}%"
                tensors[label] = predict_downstream_variant(ctx, model_name, variant, inputs[image_index], device)
                metrics[label] = metric_for_sample(per, variant, image_index)
            topk_samples.append({"name": name, "tensors": tensors, "metrics": metrics})
        comparison_path = out / f"topk_fraction_comparison_{model_name}.png"
        save_denoiser_comparison_panel(topk_samples, comparison_path, columns=list(topk_samples[0]["tensors"].keys()), title=f"{model_name} Top-K fraction comparison")
        written.append(comparison_path)
    write_json(
        meta_path,
        {
            "figures_protocol_hash": figures_protocol_hash(ctx),
            "evaluation_hashes": {
                model_name: evaluation_protocol_hash(ctx, model_name)
                for model_name in ["tdncnn", "drunet"]
            },
        },
    )
    ctx.write_marker(stage, start, written + [meta_path])


def figures_protocol_hash(ctx: PipelineContext) -> str:
    return config_hash(
        {
            "protocol_version": "imagenet100_figures_v1_grid_eval_outputs",
            "tdncnn_eval": evaluation_protocol_hash(ctx, "tdncnn"),
            "drunet_eval": evaluation_protocol_hash(ctx, "drunet"),
        }
    )


def figures_artifacts_complete(ctx: PipelineContext) -> bool:
    meta_path = ctx.stage_root("figures_grid") / "figures_meta.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return str(meta.get("figures_protocol_hash", "")) == figures_protocol_hash(ctx)


def write_combined_external_if_ready(ctx: PipelineContext) -> None:
    out_root = ctx.experiment_root / "08_external_benchmarks" / "combined"
    tdn = ctx.stage_root("tdncnn_external_eval") / "per_benchmark_summary.csv"
    dru = ctx.stage_root("drunet_external_eval") / "per_benchmark_summary.csv"
    if not (tdn.exists() and dru.exists()):
        return
    out_root.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([pd.read_csv(tdn), pd.read_csv(dru)], ignore_index=True)
    combined.to_csv(out_root / "external_combined_table.csv", index=False)
    pd.read_csv(tdn).to_csv(out_root / "tdncnn_external_results.csv", index=False)
    pd.read_csv(dru).to_csv(out_root / "drunet_external_results.csv", index=False)
    pivot = combined.pivot_table(index=["Benchmark", "Variant"], columns="Model", values="PSNR", aggfunc="mean").reset_index()
    if {"tdncnn", "drunet"}.issubset(set(pivot.columns)):
        pivot["DRUNet minus TDnCNN PSNR"] = pivot["drunet"] - pivot["tdncnn"]
    pivot.to_csv(out_root / "tdncnn_vs_drunet_comparison.csv", index=False)
    for variant in ["Top-K", "Quantile", "Random"]:
        rows = []
        for (benchmark, model_name), group in combined.groupby(["Benchmark", "Model"], sort=False):
            full = group[group["Variant"] == "Full"]
            selected = group[group["Variant"] == variant]
            if full.empty or selected.empty:
                continue
            full_row = full.iloc[0]
            row = selected.iloc[0].to_dict()
            row["Delta PSNR vs model Full"] = float(row["PSNR"] - full_row["PSNR"])
            row["Delta SSIM vs model Full"] = float(row["SSIM"] - full_row["SSIM"])
            row["Delta LPIPS vs model Full"] = float(row["LPIPS"] - full_row["LPIPS"])
            row["Delta MSE vs model Full"] = float(row["MSE"] - full_row["MSE"])
            rows.append(row)
        pd.DataFrame(rows).to_csv(out_root / f"delta_{variant.lower().replace('-', 'k')}_vs_full.csv", index=False)


def external_model_variant_stages(ctx: PipelineContext, model_name: str) -> list[tuple[str, str, str]]:
    all_stages = model_variant_stages(model_name)
    by_variant = {item[1]: item for item in all_stages}
    if bool(ctx.cfg.get("external_benchmarks", {}).get("include_all_variants", False)):
        return all_stages
    requested = ctx.cfg.get("external_benchmarks", {}).get("variants")
    if requested is None:
        requested = ["full", "topk_p10", "topk_p20", "topk_p30"]
    selected: list[tuple[str, str, str]] = []
    for raw_variant in requested:
        variant = {"topk_10": "topk_p10", "quantile_10": "quantile_p10", "random_10": "random_p10"}.get(str(raw_variant), str(raw_variant))
        if variant not in by_variant:
            raise ValueError(f"Unsupported external benchmark variant: {raw_variant}")
        selected.append(by_variant[variant])
    return selected


def stage_external_eval_model(ctx: PipelineContext, model_name: str) -> None:
    stage = f"{model_name}_external_eval"
    start = time.time()
    combined = ctx.stage_root(stage) / "combined_table.csv"
    if ctx.resume and ctx.marker_valid(stage, [combined]):
        print(f"[resume] {stage} already complete: {combined}")
        return
    if ctx.dry_run:
        print(f"[dry-run] would evaluate external benchmarks for {model_name} from {ctx.cfg['external_benchmarks']['root']} into {ctx.stage_root(stage)}")
        return
    root = Path(str(ctx.cfg["external_benchmarks"]["root"]))
    if ctx.smoke:
        write_json(combined.with_suffix(".json"), {"smoke": True, "skipped": "external evaluation is not scientific in smoke mode"})
        pd.DataFrame(columns=["Benchmark", "Model", "Variant", "PSNR", "SSIM", "LPIPS", "MSE", "Delta PSNR vs model Full"]).to_csv(combined, index=False)
        ctx.write_marker(stage, start, [combined])
        return
    if not root.exists():
        raise FileNotFoundError(f"Missing external benchmark root: {root}")
    from scripts.evaluate_external import (
        BENCHMARKS,
        audit_datasets,
        compute_metrics,
        create_lpips_model as create_external_lpips,
        load_drunet_checkpoint,
        load_pairs_for_status,
        load_tdncnn_checkpoint,
        predict_model,
        summarize_metrics,
    )

    out_root = ctx.stage_root(stage)
    audit = audit_datasets(root, out_root)
    missing = [name for name in BENCHMARKS if not audit["benchmarks"][name]["exists"] or int(audit["benchmarks"][name]["file_count"]) == 0]
    if missing:
        raise FileNotFoundError(f"Missing external benchmark data for {missing} under {root}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = create_external_lpips(device) if bool(ctx.cfg["evaluation"].get("lpips_enabled", True)) else None
    variant_paths = {label: ctx.stage_root(dstage) / "best.pt" for label, _variant, dstage in external_model_variant_stages(ctx, model_name)}
    per_rows = []
    sigma = int(ctx.cfg["external_benchmarks"]["sigma"])
    seed = int(ctx.cfg["external_benchmarks"]["seed"])
    loader_fn = load_drunet_checkpoint if model_name == "drunet" else load_tdncnn_checkpoint
    for benchmark in BENCHMARKS:
        max_images = int(ctx.cfg["real_smoke"]["external_images_per_benchmark"]) if ctx.real_smoke else None
        pairs = load_pairs_for_status(root, benchmark, audit["benchmarks"][benchmark], sigma, seed, max_images)
        for image_idx, pair in enumerate(tqdm(pairs, desc=f"External {benchmark}")):
            clean = pair["clean"]
            noisy = pair["noisy"]
            for variant, checkpoint in variant_paths.items():
                model = loader_fn(checkpoint, device)
                pred = predict_model(model, noisy, model_name, sigma / 255.0, device)
                metrics = compute_metrics(pred, clean, lpips_model)
                per_rows.append({"Benchmark": benchmark, "Model": model_name, "Variant": variant, "Image": pair["image"], **metrics})
                del model
    per_image = pd.DataFrame(per_rows)
    per_image.to_csv(out_root / "per_image_metrics.csv", index=False)
    summary = summarize_metrics(per_image.drop(columns=["Model"]))
    summary.insert(1, "Model", model_name)
    summary.to_csv(out_root / "per_benchmark_summary.csv", index=False)
    combined_table = summary.groupby(["Model", "Variant"])[["PSNR", "SSIM", "LPIPS", "MSE"]].mean(numeric_only=True).reset_index()
    full_psnr = float(combined_table[combined_table["Variant"] == "Full"]["PSNR"].iloc[0])
    combined_table["Delta PSNR vs model Full"] = combined_table["PSNR"] - full_psnr
    combined_table.to_csv(combined, index=False)
    write_json(out_root / "eval_noise_metadata.json", {"sigma": sigma, "sigma_normalized": sigma / 255.0, "seed": seed})
    write_combined_external_if_ready(ctx)
    ctx.write_marker(stage, start, [combined])


def stage_summary(ctx: PipelineContext) -> None:
    stage = "summary"
    start = time.time()
    out = ctx.stage_root(stage) / "final_summary.md"
    if ctx.dry_run:
        print(f"[dry-run] would write final summary to {out}")
        return
    inventory = []
    for path in ctx.experiment_root.rglob("*"):
        if path.is_file():
            inventory.append({"path": str(path.relative_to(ctx.experiment_root)), "bytes": path.stat().st_size})
    write_json(ctx.stage_root(stage) / "artifact_inventory.json", inventory)
    lines = [
        "# ImageNet-100 Image-Space Variable-AWGN Summary",
        "",
        f"- run_id: `{ctx.run_id}`",
        f"- experiment_root: `{ctx.experiment_root}`",
        f"- selected score config: `{ctx.stage_root('score_select') / 'selected_score_config.yaml'}`",
        f"- scores: `{ctx.stage_root('filter') / 'scores_full.csv'}`",
        f"- TDnCNN internal evaluation: `{ctx.stage_root('tdncnn_evaluate') / 'summary_table.csv'}`",
        f"- DRUNet internal evaluation: `{ctx.stage_root('drunet_evaluate') / 'summary_table.csv'}`",
        f"- combined internal evaluation: `{ctx.experiment_root / '07_evaluation' / 'combined' / 'internal_combined_table.csv'}`",
        "",
        "No scientific ImageNet result is implied until the full remote run finishes.",
    ]
    external_outputs = [
        ("TDnCNN external evaluation", ctx.stage_root("tdncnn_external_eval") / "combined_table.csv"),
        ("DRUNet external evaluation", ctx.stage_root("drunet_external_eval") / "combined_table.csv"),
        ("combined external evaluation", ctx.experiment_root / "08_external_benchmarks" / "combined" / "external_combined_table.csv"),
    ]
    existing_external = [(label, path) for label, path in external_outputs if path.exists()]
    if existing_external:
        lines.insert(-2, "")
        lines.insert(-2, "External benchmark outputs:")
        for label, path in existing_external:
            lines.insert(-2, f"- {label}: `{path}`")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ctx.write_marker(stage, start, [out, ctx.stage_root(stage) / "artifact_inventory.json"])


def stage_preflight(ctx: PipelineContext) -> None:
    stage = "preflight"
    start = time.time()
    out = ctx.stage_root(stage) / "preflight_report.json"
    if ctx.resume and ctx.marker_valid(stage, [out]):
        print(f"[resume] preflight already complete: {out}")
        return
    if ctx.dry_run:
        workload = resolved_workload(ctx.cfg)
        print(json.dumps({"dry_run": True, "resolved_workload": workload}, indent=2))
        return
    ctx.stage_root(stage).mkdir(parents=True, exist_ok=True)
    checks: dict[str, Any] = {}
    launcher = PROJECT_ROOT / "scripts" / "run_imagenet100_pipeline.sh"
    try:
        subprocess.check_output(["bash", "-n", str(launcher)], cwd=PROJECT_ROOT, stderr=subprocess.STDOUT, text=True)
        checks["bash_n_launcher"] = "ok"
    except Exception as error:
        checks["bash_n_launcher"] = f"failed: {error}"
        if not sys.platform.startswith("win"):
            raise

    train_root = Path(str(ctx.cfg["dataset"]["root"])) / "train"
    val_root = Path(str(ctx.cfg["dataset"]["root"])) / "val"
    checks["train_dir"] = str(train_root)
    checks["val_dir"] = str(val_root)
    if not train_root.is_dir() or not val_root.is_dir():
        raise FileNotFoundError(f"Missing ImageNet-100 train/val dirs under {ctx.cfg['dataset']['root']}")
    train_classes = sorted(p for p in train_root.iterdir() if p.is_dir())
    val_classes = sorted(p for p in val_root.iterdir() if p.is_dir())
    train_count = sum(1 for p in train_root.rglob("*") if p.is_file())
    val_count = sum(1 for p in val_root.rglob("*") if p.is_file())
    checks["train_classes"] = len(train_classes)
    checks["val_classes"] = len(val_classes)
    checks["train_samples"] = train_count
    checks["val_samples"] = val_count
    if len(train_classes) != int(ctx.cfg["dataset"]["num_classes"]) or len(val_classes) != int(ctx.cfg["dataset"]["num_classes"]):
        raise RuntimeError(f"Expected 100 classes, got train={len(train_classes)} val={len(val_classes)}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device("cuda")
    checks["cuda_device"] = torch.cuda.get_device_name(0)

    ds = build_dataset(ctx.cfg, split="train", smoke=False)
    x_items = []
    for idx in range(min(2, len(ds))):
        x, _cls = ds[idx]
        x_items.append(x.to(torch.float32).clamp(0, 1))
    x_batch = normalize_for_ddpm(torch.stack(x_items, dim=0)).to(device)
    model = build_ddpm(ctx.cfg, device)
    model.train()
    loss = model.training_loss(x_batch)
    loss.backward()
    checks["production_ddpm_forward_backward"] = {"loss": float(loss.detach().cpu().item()), "status": "ok"}
    ckpt = ctx.stage_root(stage) / "preflight_ddpm_checkpoint.pt"
    torch.save({"model_state_dict": model.model.state_dict(), "config_hash": ctx.cfg_hash}, ckpt)
    reload_model = build_ddpm(ctx.cfg, device)
    reload_model.model.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"])
    checks["checkpoint_save_load"] = {"path": str(ckpt), "sha256": sha256_file(ckpt), "status": "ok"}

    checks["external_benchmarks"] = "not required by remote production pipeline; run locally after downloading checkpoints"

    output_base = ctx.experiment_root.parents[1] if len(ctx.experiment_root.parents) > 1 else ctx.experiment_root.parent
    output_base.mkdir(parents=True, exist_ok=True)
    probe = output_base / ".write_probe_imagenet100_image_space_awgn"
    probe.write_text("ok\n", encoding="utf-8")
    checks["output_writable"] = probe.read_text(encoding="utf-8").strip() == "ok"
    probe.unlink(missing_ok=True)
    try:
        usage = shutil.disk_usage(output_base)
        checks["free_space_bytes"] = int(usage.free)
    except Exception as error:
        checks["free_space_bytes"] = f"unavailable: {error}"

    graph_stage_names = " ".join(STAGE_DIRS.keys())
    checks["stage_graph_uses_only_image_space_stages"] = not any(token in graph_stage_names for token in ["encoder", "latent", "autoencoder"])
    if not checks["stage_graph_uses_only_image_space_stages"]:
        raise RuntimeError("Forbidden encoder/latent stage found in new stage graph")
    workload = write_workload_audit(ctx, actual_train_samples=train_count, actual_val_samples=val_count)
    checks["resolved_workload"] = workload
    write_json(out, checks)
    print(json.dumps({"preflight": "ok", "resolved_workload": workload["forward_evaluations"]}, indent=2))
    ctx.write_marker(stage, start, [out, ctx.experiment_root / "00_protocol" / "resolved_production_workload.json"])


def run_stage(ctx: PipelineContext, stage: str) -> None:
    if stage == "preflight":
        stage_preflight(ctx)
    elif stage == "prepare":
        stage_prepare(ctx)
    elif stage == "ddpm_train":
        stage_ddpm_train(ctx)
    elif stage == "score_validate":
        stage_score_validate(ctx)
    elif stage == "score_select":
        stage_score_select(ctx)
    elif stage == "filter":
        stage_filter(ctx)
    elif stage == "filter_grid":
        stage_filter_grid(ctx)
    elif stage == "downstream_early_stopping_audit":
        stage_downstream_early_stopping_audit(ctx)
    elif stage.startswith("downstream_") or stage in VARIANT_STAGE_ALIASES:
        stage_downstream(ctx, stage)
    elif stage == "tdncnn_grid":
        stage_downstream_grid(ctx, "tdncnn")
    elif stage == "drunet_grid":
        stage_downstream_grid(ctx, "drunet")
    elif stage in {"evaluate", "tdncnn_evaluate"}:
        stage_evaluate_model(ctx, "tdncnn")
    elif stage == "drunet_evaluate":
        stage_evaluate_model(ctx, "drunet")
    elif stage == "evaluate_grid":
        stage_evaluate_grid(ctx)
    elif stage == "figures_grid":
        stage_figures_grid(ctx)
    elif stage in {"external_eval", "tdncnn_external_eval"}:
        stage_external_eval_model(ctx, "tdncnn")
    elif stage == "drunet_external_eval":
        stage_external_eval_model(ctx, "drunet")
    elif stage == "external_grid":
        stage_external_eval_model(ctx, "tdncnn")
        stage_external_eval_model(ctx, "drunet")
    elif stage in {"summary", "summary_grid"}:
        stage_summary(ctx)
    elif stage == "grid_all":
        stage_downstream_early_stopping_audit(ctx)
        stage_filter_grid(ctx)
        stage_downstream_grid(ctx, "tdncnn")
        stage_downstream_grid(ctx, "drunet")
        stage_evaluate_grid(ctx)
        stage_figures_grid(ctx)
        stage_summary(ctx)
    else:
        raise ValueError(f"Unknown stage: {stage}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ImageNet-100 current research pipeline.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--stage", choices=[*STAGE_DIRS, "internal_all"], required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--real-smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--external-benchmark-root", type=Path, default=None)
    return parser.parse_args()


def make_context(args: argparse.Namespace) -> PipelineContext:
    cfg = read_yaml(args.config)
    if args.smoke and args.real_smoke:
        raise ValueError("--smoke and --real-smoke are mutually exclusive")
    cfg["_smoke_runtime"] = bool(args.smoke)
    cfg["_real_smoke_runtime"] = bool(args.real_smoke)
    if args.real_smoke:
        rs = cfg["real_smoke"]
        cfg["dataset"]["allow_count_mismatch"] = True
        cfg["ddpm"]["epochs"] = 1
        cfg["ddpm"]["batch_size"] = int(rs["ddpm_batch_size"])
        cfg["score_validation"]["samples_per_class"] = int(rs["score_validation_samples_per_class"])
        cfg["score_validation"]["validation_corpus_samples_per_class"] = int(rs["validation_corpus_samples_per_class"])
        cfg["score_validation"]["validation_corpus_samples"] = int(rs["validation_corpus_samples_per_class"]) * int(cfg["dataset"]["num_classes"])
        cfg["score_validation"]["timestep_scan"]["report_timesteps"] = [int(x) for x in rs["timestep_grid"]]
        cfg["score_validation"]["mc"]["coarse_k"] = int(rs["mc_k"])
        cfg["score_validation"]["mc"]["final_k"] = int(rs["mc_k"])
        cfg["score_validation"]["checkpoint_scan"]["epochs"] = [1]
        cfg["score_validation"]["checkpoint_scan"]["top_candidates"] = int(rs["final_candidates"])
        cfg["downstream"]["batch_size"] = int(rs["downstream_batch_size"])
        cfg["downstream"]["max_steps"] = int(rs["downstream_optimizer_steps"])
        cfg.setdefault("drunet", {})
        cfg["drunet"]["batch_size"] = int(rs["downstream_batch_size"])
        cfg["drunet"]["max_steps"] = int(rs["downstream_optimizer_steps"])
        cfg["runtime"]["num_workers"] = 0
    if args.data_root is not None:
        cfg.setdefault("dataset", {})["root"] = str(args.data_root)
    if args.external_benchmark_root is not None:
        cfg.setdefault("external_benchmarks", {})["root"] = str(args.external_benchmark_root)
    return PipelineContext(
        cfg=cfg,
        config_path=args.config.resolve(),
        run_id=str(args.run_id),
        experiment_root=args.experiment_root.resolve(),
        smoke=bool(args.smoke),
        real_smoke=bool(args.real_smoke),
        dry_run=bool(args.dry_run),
        resume=bool(args.resume),
    )


def main() -> None:
    args = parse_args()
    ctx = make_context(args)
    if ctx.experiment_root.exists() and not ctx.resume and not ctx.dry_run and args.stage != 'preflight':
        raise FileExistsError('Output root exists; choose a new root or explicitly --resume an owned run.')
    validate_current_protocol("ImageNet-100", ctx.cfg, downstream_models=["tdncnn", "drunet"])
    print_protocol_summary(
        "ImageNet-100",
        ctx.cfg,
        run_id=ctx.run_id,
        device="cuda" if torch.cuda.is_available() and not ctx.smoke else "cpu",
        downstream_models=["tdncnn", "drunet"],
    )
    if ctx.cfg["dataset"].get("auto_download", False):
        raise RuntimeError("Dataset auto-download is disabled for ImageNet-100")
    stages = (["prepare", "ddpm_train", "score_validate", "score_select", "filter",
               "downstream_early_stopping_audit", "filter_grid", "tdncnn_grid", "drunet_grid",
               "evaluate_grid", "figures_grid", "summary_grid"] if args.stage == "internal_all" else [args.stage])
    for stage in stages:
        with stage_logging("ImageNet-100", stage, run_id=ctx.run_id):
            run_stage(ctx, stage)


if __name__ == "__main__":
    main()
