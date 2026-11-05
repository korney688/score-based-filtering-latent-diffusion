from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:  # pragma: no cover - optional dependency
    skimage_ssim = None

from src.ddpm.image_space import ImageSpaceDDPM, build_image_space_ddpm_model
from src.colored_mnist import ColoredMNISTDataset, ColoredMNISTWrapper
from src.DRUNet_image import build_drunet
from src.TDnCNN_image import DnCNN_2D
from src.filtering.selection import (
    fraction_tag,
    select_quantile_spread_exact,
    select_random_exact,
    select_topk_exact,
    subset_size_for_fraction,
)
from src.evaluation.denoiser_panels import save_denoiser_comparison_panel
from src.evaluation.quality_shift_plots import (
    plot_ecdf_realized_psnr,
    plot_quality_shift_3x3,
    plot_subset_fraction_metric,
)
from src.evaluation.selection_panels import (
    save_topk_cleanest_noisiest,
    save_topk_selection_examples,
)
from src.corruption_datasets import (
    CorpusTaskNoiseDataset,
    EvalNoiseDataset,
    MaterializedCorpusDataset,
    materialize_eval_noise_cache,
    materialize_variable_awgn_corpus,
    stable_int_seed,
)
from src.utils.current_protocol import (
    print_protocol_summary,
    should_log_progress,
    stage_logging,
    stage_skip,
    training_progress,
    validate_current_protocol,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
    "external_evaluate",
    "summary",
]
GRID_STAGE_NAMES = [
    "audit",
    "filter_grid",
    "downstream_full",
    "downstream_topk_p10",
    "downstream_topk_p20",
    "downstream_topk_p30",
    "downstream_quantile_p10",
    "downstream_quantile_p20",
    "downstream_quantile_p30",
    "downstream_random_p10",
    "downstream_random_p20",
    "downstream_random_p30",
    "drunet_topk_p20",
    "drunet_topk_p30",
    "drunet_quantile_p20",
    "drunet_quantile_p30",
    "drunet_random_p20",
    "drunet_random_p30",
    "evaluate_grid",
    "figures",
]
STAGE_ALIASES = {
    "tdncnn_all": ["tdncnn_full", "tdncnn_topk", "tdncnn_quantile", "tdncnn_random"],
    "drunet_all": ["drunet_full", "drunet_topk", "drunet_quantile", "drunet_random"],
    "drunet_grid": [
        "drunet_full",
        "drunet_topk",
        "drunet_topk_p20",
        "drunet_topk_p30",
        "drunet_quantile",
        "drunet_quantile_p20",
        "drunet_quantile_p30",
        "drunet_random",
        "drunet_random_p20",
        "drunet_random_p30",
    ],
    "evaluate": ["tdncnn_evaluate", "drunet_evaluate"],
    "external_eval": ["external_evaluate"],
    "downstream_grid": [
        "downstream_full",
        "downstream_topk_p10",
        "downstream_topk_p20",
        "downstream_topk_p30",
        "downstream_quantile_p10",
        "downstream_quantile_p20",
        "downstream_quantile_p30",
        "downstream_random_p10",
        "downstream_random_p20",
        "downstream_random_p30",
    ],
    "full_grid": [
        "prepare",
        "ddpm_train",
        "score_validate",
        "score_select",
        "filter_grid",
        "downstream_grid",
        "drunet_grid",
        "evaluate_grid",
        "figures",
        "summary",
    ],
    "tail_grid": [
        "filter_grid",
        "downstream_grid",
        "drunet_grid",
        "evaluate_grid",
        "figures",
        "summary",
    ],
}
VARIANTS = {
    "full": "full_indices.npy",
    "topk_10": "topk_10_indices.npy",
    "quantile_spread_10": "quantile_spread_10_indices.npy",
    "random_10": "random_10_indices.npy",
}
LEGACY_P10_ALIASES = {
    "topk_10": "topk_p10",
    "quantile_spread_10": "quantile_spread_p10",
    "random_10": "random_p10",
}
EPS = 1e-12


@dataclass
class Context:
    cfg: dict[str, Any]
    config_path: Path
    run_id: str
    experiment_root: Path
    data_root: Path
    device: torch.device
    resume: bool
    dry_run: bool
    smoke: bool
    force_fail_gate: bool
    downstream_budget_mode: str | None = None


class SyntheticMNISTDataset(Dataset):
    def __init__(self, samples_per_class: int, *, image_size: int = 28, seed: int = 42) -> None:
        self.images: list[torch.Tensor] = []
        self.labels: list[int] = []
        gen = torch.Generator().manual_seed(int(seed))
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, image_size),
            torch.linspace(-1.0, 1.0, image_size),
            indexing="ij",
        )
        for label in range(10):
            for rep in range(int(samples_per_class)):
                cx = -0.65 + 0.13 * label
                cy = -0.35 + 0.18 * (rep % 5)
                blob = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (0.08 + 0.01 * (label % 3)))
                stripe = ((xx * (label + 1) + yy * (rep + 1)) > 0).to(torch.float32) * 0.12
                jitter = torch.rand((image_size, image_size), generator=gen) * 0.03
                image = (blob + stripe + jitter).clamp(0.0, 1.0).unsqueeze(0).to(torch.float32)
                self.images.append(image)
                self.labels.append(label)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        idx = int(index)
        return self.images[idx].clone(), int(self.labels[idx])


class TinyTDnCNN(nn.Module):
    def __init__(self, in_channels: int = 1, features: int = 32, num_layers: int = 5) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, features, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(max(0, int(num_layers) - 2)):
            layers.extend([nn.Conv2d(features, features, 3, padding=1), nn.BatchNorm2d(features), nn.ReLU(inplace=True)])
        layers.append(nn.Conv2d(features, in_channels, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.net(x)).clamp(0.0, 1.0)


class CorpusTaskNoiseWithSigmaDataset(Dataset):
    def __init__(
        self,
        corpus_dataset: MaterializedCorpusDataset,
        *,
        dataset_name: str,
        task_seed: int,
        task_noise_version: str,
        sigma_min: float,
        sigma_max: float,
    ) -> None:
        self.corpus_dataset = corpus_dataset
        self.dataset_name = dataset_name
        self.task_seed = int(task_seed)
        self.task_noise_version = str(task_noise_version)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

    def __len__(self) -> int:
        return len(self.corpus_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_target, original_index = self.corpus_dataset[index]
        sample_seed = stable_int_seed(
            self.dataset_name,
            "train_task_noise",
            int(original_index),
            self.task_seed,
            self.task_noise_version,
        )
        generator = torch.Generator().manual_seed(sample_seed)
        u = torch.rand((), generator=generator).item()
        sigma = float(self.sigma_min + (self.sigma_max - self.sigma_min) * u)
        noise = torch.randn(x_target.shape, generator=generator, dtype=x_target.dtype) * sigma
        x_input = (x_target + noise).clamp(0.0, 1.0)
        return x_input.to(torch.float32), x_target.to(torch.float32), torch.tensor(sigma, dtype=torch.float32)


class EvalNoiseWithSigmaDataset(Dataset):
    def __init__(self, input_path: str | Path, target_path: str | Path, manifest_path: str | Path) -> None:
        self.inputs = torch.load(Path(input_path), map_location="cpu").to(torch.float32)
        self.targets = torch.load(Path(target_path), map_location="cpu").to(torch.float32)
        self.manifest = pd.read_csv(manifest_path)
        if self.inputs.shape != self.targets.shape:
            raise ValueError(f"Eval input/target shape mismatch: {self.inputs.shape} vs {self.targets.shape}")
        if len(self.manifest) != int(self.inputs.shape[0]):
            raise ValueError(f"Eval manifest length {len(self.manifest)} does not match tensors {self.inputs.shape[0]}")

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = int(index)
        return self.inputs[idx], self.targets[idx], torch.tensor(float(self.manifest.iloc[idx]["sigma"]), dtype=torch.float32)


def deep_get(d: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def subset_grid_fractions(ctx: Context) -> list[float]:
    raw = deep_get(ctx.cfg, "filtering.grid_fractions", deep_get(ctx.cfg, "filtering.fractions", [0.10, 0.20, 0.30]))
    values = [float(x) for x in raw]
    if not values:
        raise ValueError("filtering.grid_fractions must not be empty")
    return sorted(dict.fromkeys(values))


def subset_fraction_percent(fraction: float) -> int:
    return int(round(float(fraction) * 100.0))


def grid_variant_names(ctx: Context) -> list[str]:
    variants = ["full"]
    for fraction in subset_grid_fractions(ctx):
        tag = fraction_tag(fraction)
        variants.extend([f"topk_{tag}", f"quantile_spread_{tag}", f"random_{tag}"])
    return variants


def variant_strategy_fraction(variant: str) -> tuple[str, float]:
    if variant == "full":
        return "full", 1.0
    canonical = LEGACY_P10_ALIASES.get(variant, variant)
    if canonical.startswith("topk_"):
        return "topk", int(canonical.rsplit("p", 1)[1]) / 100.0
    if canonical.startswith("quantile_spread_"):
        return "quantile", int(canonical.rsplit("p", 1)[1]) / 100.0
    if canonical.startswith("random_"):
        return "random", int(canonical.rsplit("p", 1)[1]) / 100.0
    raise ValueError(f"Unknown subset variant: {variant}")


def variant_index_filename(variant: str) -> str:
    if variant in VARIANTS:
        return VARIANTS[variant]
    return f"{variant}_indices.npy"


def variant_indices_path(ctx: Context, variant: str) -> Path:
    root = ctx.experiment_root / "05_filtering" / "subset_indices"
    direct = root / variant_index_filename(variant)
    if direct.exists():
        return direct
    alias = LEGACY_P10_ALIASES.get(variant)
    if alias is not None:
        aliased = root / variant_index_filename(alias)
        if aliased.exists():
            return aliased
    reverse = {v: k for k, v in LEGACY_P10_ALIASES.items()}
    legacy = reverse.get(variant)
    if legacy is not None:
        legacy_path = root / variant_index_filename(legacy)
        if legacy_path.exists():
            return legacy_path
    return direct


def downstream_budget_mode(ctx: Context) -> str:
    return str(ctx.downstream_budget_mode or deep_get(ctx.cfg, "downstream.budget_mode", "main")).strip().lower()


def downstream_max_steps_for_variant(ctx: Context, variant: str, train_size: int, batch_size: int, configured_max_steps: int) -> int:
    mode = downstream_budget_mode(ctx)
    if mode not in {"equal_epochs", "diagnostic_equal_epochs"}:
        return int(configured_max_steps)
    full_indices = np.load(variant_indices_path(ctx, "full"))
    default_epochs = int(configured_max_steps) * int(batch_size) / max(1, int(len(full_indices)))
    target_epochs = deep_get(ctx.cfg, "downstream.diagnostic_equal_epochs.target_effective_epochs", None)
    target_epochs = default_epochs if target_epochs is None else float(target_epochs)
    return max(1, int(math.ceil(float(target_epochs) * int(train_size) / int(batch_size))))


def budget_fields(step: int, train_size: int, batch_size: int, prefix: str) -> dict[str, float | int]:
    total_seen = int(step) * int(batch_size)
    effective_epochs = total_seen / max(1, int(train_size))
    return {
        f"total_seen_samples_at_{prefix}": int(total_seen),
        f"effective_epochs_at_{prefix}": float(effective_epochs),
    }


def dataset_key(ctx: Context) -> str:
    return str(deep_get(ctx.cfg, "dataset.name", "mnist")).lower()


def dataset_title(ctx: Context) -> str:
    return "Colored MNIST" if dataset_key(ctx) == "colored_mnist" else "MNIST"


def dataset_channels(ctx: Context) -> int:
    return int(deep_get(ctx.cfg, "dataset.channels", 1))


def colorization_enabled(ctx: Context) -> bool:
    return bool(deep_get(ctx.cfg, "dataset.colorization.enabled", False))


def color_seed(ctx: Context) -> int:
    return int(deep_get(ctx.cfg, "dataset.colorization.color_seed", 42))


def background_seed(ctx: Context) -> int:
    return int(deep_get(ctx.cfg, "dataset.colorization.background_seed", color_seed(ctx)))


def colorization_mode(ctx: Context) -> str:
    return str(deep_get(ctx.cfg, "dataset.colorization.mode", "abstract_background"))


def seed_namespace(ctx: Context, suffix: str) -> str:
    key = dataset_key(ctx)
    if key == "mnist":
        return f"mnist_{suffix}"
    return f"{key}_{suffix}"


def require_path_under(path: Path, root: Path, label: str) -> None:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{label} must be inside current project root: {resolved_root}. Got: {resolved_path}") from error


def resolve_context(args: argparse.Namespace) -> Context:
    cfg = read_yaml(Path(args.config))
    smoke = bool(args.smoke)
    run_id = args.run_id or deep_get(cfg, "experiment.run_id", "mnist_image_space_awgn_v1")
    if smoke and not run_id.endswith("_synthetic_smoke"):
        run_id = f"{run_id}_synthetic_smoke"
    root = Path(args.experiment_root) if args.experiment_root else PROJECT_ROOT / deep_get(cfg, "experiment.root") / run_id
    if root.is_absolute():
        require_path_under(root, PROJECT_ROOT, "experiment_root")
    else:
        root = PROJECT_ROOT / root
    data_root = Path(args.data_root) if args.data_root else PROJECT_ROOT / deep_get(cfg, "dataset.data_root", "data")
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    return Context(
        cfg=cfg,
        config_path=Path(args.config),
        run_id=run_id,
        experiment_root=root,
        data_root=data_root,
        device=device,
        resume=bool(args.resume),
        dry_run=bool(args.dry_run),
        smoke=smoke,
        force_fail_gate=bool(args.force_fail_gate),
        downstream_budget_mode=getattr(args, "downstream_budget_mode", None),
    )


def stage_root(ctx: Context, stage: str) -> Path:
    mapping = {
        "audit": "00_protocol",
        "prepare": "01_corpus",
        "ddpm_train": "02_ddpm",
        "score_validate": "03_score_validation",
        "score_select": "04_selected_score" if dataset_key(ctx) == "colored_mnist" else "04_selected_configuration",
        "filter": "05_filtering",
        "filter_grid": "05_filtering",
        "evaluate_grid": "07_evaluation",
        "figures": "08_figures",
        "external_evaluate": "09_external_evaluation",
        "summary": "08_summary",
    }
    if stage.startswith("tdncnn_") or stage.startswith("drunet_") or stage.startswith("downstream_"):
        return ctx.experiment_root / "06_downstream"
    return ctx.experiment_root / mapping[stage]


def selected_score_root(ctx: Context) -> Path:
    return stage_root(ctx, "score_select")


def evaluation_root(ctx: Context) -> Path:
    name = "07_internal_evaluation" if dataset_key(ctx) == "colored_mnist" else "07_evaluation"
    return ctx.experiment_root / name


def marker_path(ctx: Context, stage: str) -> Path:
    return stage_root(ctx, stage) / f"_{stage}_SUCCESS.json"


def is_done(ctx: Context, stage: str) -> bool:
    return ctx.resume and marker_path(ctx, stage).exists()


def mark_done(ctx: Context, stage: str, outputs: Iterable[Path]) -> None:
    write_json(
        marker_path(ctx, stage),
        {
            "stage": stage,
            "run_id": ctx.run_id,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": str(ctx.config_path),
            "outputs": [str(p) for p in outputs],
        },
    )


def load_clean_dataset(ctx: Context, *, train: bool) -> Dataset:
    image_size = int(deep_get(ctx.cfg, "dataset.image_size", 28))
    split = "train" if train else "test"
    if ctx.smoke and bool(deep_get(ctx.cfg, "smoke.synthetic", True)):
        key = "smoke.train_samples_per_class" if train else "smoke.test_samples_per_class"
        base = SyntheticMNISTDataset(int(deep_get(ctx.cfg, key, 4)), image_size=image_size, seed=int(deep_get(ctx.cfg, "seeds.split_seed", 42)))
        if colorization_enabled(ctx):
            return ColoredMNISTWrapper(
                base,
                split=split,
                color_seed=color_seed(ctx),
                background_seed=background_seed(ctx),
                mode=colorization_mode(ctx),
            )
        return base
    if colorization_enabled(ctx):
        return ColoredMNISTDataset(
            root=str(ctx.data_root),
            train=train,
            download=bool(deep_get(ctx.cfg, "dataset.download", False)),
            color_seed=color_seed(ctx),
            background_seed=background_seed(ctx),
            mode=colorization_mode(ctx),
        )
    return datasets.MNIST(
        root=str(ctx.data_root),
        train=train,
        download=bool(deep_get(ctx.cfg, "dataset.download", False)),
        transform=transforms.ToTensor(),
    )


def class_stratified_indices(dataset: Dataset, per_class: int, *, seed: int) -> np.ndarray:
    buckets: dict[int, list[int]] = {}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        buckets.setdefault(int(label), []).append(idx)
    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    for label in sorted(buckets):
        values = np.asarray(buckets[label], dtype=np.int64)
        rng.shuffle(values)
        selected.extend(values[: min(int(per_class), len(values))].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def enrich_manifest(manifest_path: Path) -> None:
    df = pd.read_csv(manifest_path)
    if "label" not in df.columns:
        df["label"] = df["class_id"]
    if "realized_psnr" not in df.columns:
        df["realized_psnr"] = df["realized_psnr_db_postclip"]
    if "realized_snr" not in df.columns:
        df["realized_snr"] = df["realized_snr_db_postclip"]
    if "corruption_mse" not in df.columns:
        df["corruption_mse"] = df["noise_power_postclip"]
    df.to_csv(manifest_path, index=False)


def stage_prepare(ctx: Context) -> None:
    if is_done(ctx, "prepare"):
        stage_skip("prepare", "completed marker found")
        return
    if ctx.dry_run:
        print(f"prepare: would write {ctx.experiment_root}")
        return
    # Fix corpus noise before DDPM training so all subsets share the same targets.
    root = ctx.experiment_root
    image_size = int(deep_get(ctx.cfg, "dataset.image_size", 28))
    protocol = root / "00_protocol"
    protocol.mkdir(parents=True, exist_ok=True)
    write_json(
        protocol / "environment.json",
        {
            "run_id": ctx.run_id,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(ctx.device),
        },
    )
    (protocol / "resolved_config.yaml").write_text(yaml.safe_dump(ctx.cfg, sort_keys=False), encoding="utf-8")

    train_ds = load_clean_dataset(ctx, train=True)
    test_ds = load_clean_dataset(ctx, train=False)
    name = dataset_key(ctx)
    train_indices = np.arange(len(train_ds), dtype=np.int64)
    test_indices = np.arange(len(test_ds), dtype=np.int64)
    tensor_path, manifest_path = materialize_variable_awgn_corpus(
        dataset=train_ds,
        dataset_name=name,
        split="train",
        original_indices=train_indices,
        output_dir=root / "01_corpus",
        target_psnr_min_db=float(deep_get(ctx.cfg, "corpus.target_psnr_min_db", 10.0)),
        target_psnr_max_db=float(deep_get(ctx.cfg, "corpus.target_psnr_max_db", 40.0)),
        corpus_seed=int(deep_get(ctx.cfg, "seeds.corpus_seed", 42)),
        corruption_version=str(deep_get(ctx.cfg, "corpus.version", "variable_awgn_psnr10_40_v1")),
        overwrite=False,
    )
    enrich_manifest(manifest_path)
    score_per_class = int(deep_get(ctx.cfg, "smoke.score_samples_per_class" if ctx.smoke else "score_validation.samples_per_class", 20))
    corpus_per_class = int(
        deep_get(ctx.cfg, "smoke.validation_corpus_samples_per_class" if ctx.smoke else "score_validation.validation_corpus_samples_per_class", 50)
    )
    score_indices = class_stratified_indices(train_ds, score_per_class, seed=int(deep_get(ctx.cfg, "seeds.split_seed", 42)))
    validation_corpus_indices = class_stratified_indices(train_ds, corpus_per_class, seed=int(deep_get(ctx.cfg, "seeds.split_seed", 42)) + 1)
    validation_fraction = float(deep_get(ctx.cfg, "downstream.validation_fraction", 0.10))
    validation_count = max(1, int(round(validation_fraction * len(train_indices))))
    validation_rng = np.random.default_rng(int(deep_get(ctx.cfg, "downstream.validation_seed", 43)))
    downstream_validation_indices = np.sort(validation_rng.choice(train_indices, size=validation_count, replace=False).astype(np.int64))
    eval_indices = test_indices
    if ctx.smoke:
        eval_indices = test_indices[: min(len(test_indices), 20)]
    split_dir = root / "00_protocol"
    np.save(split_dir / "score_validation_indices.npy", score_indices)
    np.save(split_dir / "validation_corpus_indices.npy", validation_corpus_indices)
    np.save(split_dir / "downstream_validation_indices.npy", downstream_validation_indices)
    np.save(split_dir / "evaluation_indices.npy", eval_indices)
    write_json(
        split_dir / "dataset_summary.json",
        {
            "dataset": name,
            "emnist_present": False,
            "train_samples": int(len(train_ds)),
            "test_samples": int(len(test_ds)),
            "channels": dataset_channels(ctx),
            "image_size": image_size,
            "colorization_enabled": colorization_enabled(ctx),
            "color_seed": color_seed(ctx) if colorization_enabled(ctx) else None,
            "score_validation_samples": int(len(score_indices)),
            "validation_corpus_samples": int(len(validation_corpus_indices)),
            "downstream_validation_samples": int(len(downstream_validation_indices)),
            "downstream_train_pool_samples": int(len(train_indices) - len(downstream_validation_indices)),
        },
    )
    mark_done(ctx, "prepare", [tensor_path, manifest_path, split_dir / "dataset_summary.json"])


def build_ddpm(ctx: Context) -> ImageSpaceDDPM:
    smoke_prefix = "smoke." if ctx.smoke else ""
    base_dim = int(deep_get(ctx.cfg, "ddpm.base_dim", 16))
    depth = int(deep_get(ctx.cfg, "ddpm.depth", 2))
    in_channels = int(deep_get(ctx.cfg, "ddpm.input_channels", dataset_channels(ctx)))
    out_channels = int(deep_get(ctx.cfg, "ddpm.output_channels", in_channels))
    if in_channels != dataset_channels(ctx) or out_channels != dataset_channels(ctx):
        raise ValueError(
            f"DDPM channels must match dataset channels: dataset={dataset_channels(ctx)}, "
            f"input={in_channels}, output={out_channels}"
        )
    return build_image_space_ddpm_model(
        base_dim=base_dim,
        deep=depth,
        device=ctx.device,
        in_channels=in_channels,
        n_steps=int(deep_get(ctx.cfg, "ddpm.T", 1000)),
        beta_start=float(deep_get(ctx.cfg, "ddpm.beta_start", 1e-4)),
        beta_end=float(deep_get(ctx.cfg, "ddpm.beta_end", 0.02)),
    )


def checkpoint_path(ctx: Context, epoch: int) -> Path:
    return ctx.experiment_root / "02_ddpm" / "checkpoints" / f"epoch_{int(epoch):03d}.pt"


def stage_ddpm_train(ctx: Context) -> None:
    if is_done(ctx, "ddpm_train"):
        print("ddpm_train: resume skip")
        return
    if ctx.dry_run:
        print("ddpm_train: would train image-space MNIST DDPM")
        return
    torch.manual_seed(int(deep_get(ctx.cfg, "seeds.model_seed", 42)))
    model = build_ddpm(ctx)
    corpus = MaterializedCorpusDataset(ctx.experiment_root / "01_corpus" / "train_corpus.pt", ctx.experiment_root / "01_corpus" / "train_manifest.csv")
    batch_size = int(deep_get(ctx.cfg, "smoke.ddpm_batch_size" if ctx.smoke else "ddpm.batch_size", 128))
    epochs = int(deep_get(ctx.cfg, "smoke.ddpm_epochs" if ctx.smoke else "ddpm.epochs", 20))
    max_steps = deep_get(ctx.cfg, "smoke.ddpm_max_optimizer_steps" if ctx.smoke else "ddpm.max_optimizer_steps", None)
    max_steps = None if max_steps is None else int(max_steps)
    loader = DataLoader(corpus, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=float(deep_get(ctx.cfg, "ddpm.lr", 1e-4)))
    ckpt_dir = ctx.experiment_root / "02_ddpm" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    global_step = 0
    log_start = time.time()
    log_interval = int(deep_get(ctx.cfg, "ddpm.log_interval_epochs", 1))
    for epoch in range(1, epochs + 1):
        losses: list[float] = []
        model.train()
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            loss = model.train_step(batch)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            global_step += 1
            if max_steps is not None and global_step >= max_steps:
                break
        rows.append({"epoch": epoch, "global_step": global_step, "loss": float(np.mean(losses) if losses else math.nan)})
        if epoch % int(deep_get(ctx.cfg, "ddpm.checkpoint_interval", 1)) == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "prediction_type": "epsilon",
                    "data_domain": "image_space",
                    "channels": dataset_channels(ctx),
                    "input_channels": dataset_channels(ctx),
                    "output_channels": dataset_channels(ctx),
                    "config_hash": sha256_file(ctx.config_path),
                },
                checkpoint_path(ctx, epoch),
            )
        pd.DataFrame(rows).to_csv(ctx.experiment_root / "02_ddpm" / "training_history.csv", index=False)
        if should_log_progress(epoch, epochs, log_interval):
            training_progress(
                dataset_title(ctx),
                "ddpm_train",
                unit="epoch",
                current=epoch,
                total=epochs,
                train_loss=rows[-1]["loss"],
                lr=opt.param_groups[0]["lr"],
                start_time=log_start,
                extra={"global_step": global_step},
            )
        if max_steps is not None and global_step >= max_steps:
            break
    mark_done(ctx, "ddpm_train", [ctx.experiment_root / "02_ddpm" / "training_history.csv"])


@torch.no_grad()
def score_batch(model: ImageSpaceDDPM, x0: torch.Tensor, report_timestep: int, seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    x0 = x0.to(model.device)
    timesteps = torch.full((x0.shape[0],), int(report_timestep) - 1, dtype=torch.long, device=model.device)
    score_samples: list[torch.Tensor] = []
    # Average squared noise energy; report timesteps start at 1.
    for seed in seeds:
        generator = torch.Generator(device=model.device).manual_seed(int(seed))
        noise = torch.randn(x0.shape, generator=generator, device=model.device, dtype=x0.dtype)
        noisy_images = model.q_sample(x0, timesteps, noise)
        predicted_noise = model.model(noisy_images, timesteps)
        score_samples.append(predicted_noise.flatten(1).pow(2).sum(dim=1).detach().cpu())
    stacked = torch.stack(score_samples, dim=0)
    return stacked.mean(dim=0).numpy(), stacked.std(dim=0, unbiased=False).numpy()


def spearman(a: Iterable[float], b: Iterable[float]) -> float:
    x = pd.Series(list(a)).rank(method="average").to_numpy(dtype=np.float64)
    y = pd.Series(list(b)).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(x) < EPS or np.std(y) < EPS:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def mc_seeds(ctx: Context, k: int, *parts: object) -> list[int]:
    return [stable_int_seed(seed_namespace(ctx, "forward"), ctx.run_id, *parts, i, int(deep_get(ctx.cfg, "seeds.forward_noise_seed", 777))) for i in range(int(k))]


def assert_current_image_space_checkpoint(ctx: Context, ckpt: dict[str, Any], checkpoint: Path) -> None:
    legacy_keys = {
        "autoencoder_checkpoint_path",
        "autoencoder_kind",
        "encoder_checkpoint",
        "latent_dim",
        "latent_noise_mode",
    }
    found = sorted(key for key in legacy_keys if key in ckpt)
    if found:
        raise RuntimeError(f"Refusing to load legacy encoder/latent checkpoint {checkpoint}: {found}")
    domain = ckpt.get("data_domain", "image_space")
    if domain not in {"image", "image_space"}:
        raise RuntimeError(f"Refusing to load non-image checkpoint {checkpoint}: data_domain={domain!r}")
    expected = dataset_channels(ctx)
    for key in ("channels", "input_channels", "output_channels"):
        if key in ckpt and int(ckpt[key]) != expected:
            raise RuntimeError(f"Refusing to load checkpoint {checkpoint}: {key}={ckpt[key]} but expected {expected}")


def load_ddpm_checkpoint(ctx: Context, epoch: int) -> ImageSpaceDDPM:
    model = build_ddpm(ctx)
    path = checkpoint_path(ctx, epoch)
    ckpt = torch.load(path, map_location=ctx.device)
    assert_current_image_space_checkpoint(ctx, ckpt, path)
    model.model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model


def resolved_epochs(ctx: Context) -> list[int]:
    raw = deep_get(ctx.cfg, "smoke.checkpoint_epochs" if ctx.smoke else "score_validation.checkpoint_epochs", "all")
    if raw == "all":
        return sorted(int(p.stem.split("_")[-1]) for p in (ctx.experiment_root / "02_ddpm" / "checkpoints").glob("epoch_*.pt"))
    return [int(x) for x in raw]


def resolved_timesteps(ctx: Context) -> list[int]:
    return [int(x) for x in deep_get(ctx.cfg, "smoke.timestep_grid" if ctx.smoke else "score_validation.timestep_grid", [1, 100])]


def ladder_scores(ctx: Context, dataset: Dataset, indices: np.ndarray, model: ImageSpaceDDPM, epoch: int, timestep: int, k: int, use_oracle: bool) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    levels = [float(x) for x in deep_get(ctx.cfg, "score_validation.psnr_ladder", [10, 15, 20, 25, 30, 35, 40])]
    for original_index in indices.tolist():
        clean, label = dataset[int(original_index)]
        base_seed = stable_int_seed(seed_namespace(ctx, "ladder"), int(original_index), int(deep_get(ctx.cfg, "seeds.corpus_seed", 42)))
        base_gen = torch.Generator().manual_seed(base_seed)
        base_noise = torch.randn(clean.shape, generator=base_gen, dtype=clean.dtype)
        batch: list[torch.Tensor] = []
        metrics: list[dict[str, Any]] = []
        for psnr_db in levels:
            sigma = float(10.0 ** (-psnr_db / 20.0))
            pre = clean + sigma * base_noise
            x = pre.clamp(0.0, 1.0)
            mse = float(((x - clean) ** 2).mean().item())
            batch.append(x)
            metrics.append({"target_psnr_db": psnr_db, "sigma": sigma, "corruption_mse": mse, "realized_psnr": 10.0 * math.log10(1.0 / max(mse, EPS))})
        if use_oracle:
            scores = np.asarray([m["sigma"] ** 2 for m in metrics], dtype=np.float64)
            stds = np.zeros_like(scores)
        else:
            scores, stds = score_batch(model, torch.stack(batch), timestep, mc_seeds(ctx, k, "ladder", epoch, timestep, original_index))
        for m, score, std in zip(metrics, scores, stds):
            rows.append(
                {
                    "checkpoint_epoch": epoch,
                    "report_timestep": timestep,
                    "code_timestep_index": timestep - 1,
                    "original_index": int(original_index),
                    "class_id": int(label),
                    "score_value": float(score),
                    "score_std": float(std),
                    **m,
                }
            )
    return pd.DataFrame(rows)


def validation_corpus_scores(ctx: Context, model: ImageSpaceDDPM, epoch: int, timestep: int, k: int, use_oracle: bool) -> pd.DataFrame:
    manifest = pd.read_csv(ctx.experiment_root / "01_corpus" / "train_manifest.csv")
    indices = np.load(ctx.experiment_root / "00_protocol" / "validation_corpus_indices.npy")
    subset = manifest[manifest["original_index"].isin(set(indices.tolist()))].copy()
    corpus = MaterializedCorpusDataset(ctx.experiment_root / "01_corpus" / "train_corpus.pt", ctx.experiment_root / "01_corpus" / "train_manifest.csv", original_indices=indices)
    rows: list[dict[str, Any]] = []
    loader = DataLoader(corpus, batch_size=int(deep_get(ctx.cfg, "evaluation.batch_size", 256)), shuffle=False, num_workers=0)
    offset = 0
    for batch in loader:
        current = indices[offset : offset + batch.shape[0]]
        meta = subset.set_index("original_index").loc[current]
        if use_oracle:
            scores = meta["sigma"].to_numpy(dtype=np.float64) ** 2
            stds = np.zeros_like(scores)
        else:
            scores, stds = score_batch(model, batch, timestep, mc_seeds(ctx, k, "val_corpus", epoch, timestep, offset))
        for original_index, score, std in zip(current, scores, stds):
            row = subset[subset["original_index"] == int(original_index)].iloc[0].to_dict()
            rows.append(
                {
                    "checkpoint_epoch": epoch,
                    "report_timestep": timestep,
                    "code_timestep_index": timestep - 1,
                    "score_value": float(score),
                    "score_std": float(std),
                    **row,
                }
            )
        offset += batch.shape[0]
    return pd.DataFrame(rows)


def bootstrap_median_ci(values: list[float], *, seed: int, iterations: int) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(int(seed))
    medians = []
    for _ in range(max(1, int(iterations))):
        sample = rng.choice(arr, size=arr.size, replace=True)
        medians.append(float(np.median(sample)))
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def summarize_candidate(
    ladder: pd.DataFrame,
    val: pd.DataFrame,
    topk_ratio: float,
    *,
    bootstrap_seed: int,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    rhos = []
    ordering = []
    for _, group in ladder.groupby("original_index"):
        rhos.append(spearman(group["score_value"], group["sigma"]))
        ordered = group.sort_values("sigma")
        scores = ordered["score_value"].to_numpy()
        ok = 0
        total = 0
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                ok += int(scores[j] > scores[i])
                total += 1
        ordering.append(ok / total if total else 0.0)
    ci_lower, ci_upper = bootstrap_median_ci(rhos, seed=bootstrap_seed, iterations=bootstrap_iterations)
    n_top = max(1, int(round(float(topk_ratio) * len(val))))
    top_idx = select_topk_exact(val.rename(columns={"score_value": "score_value"}), n_top)
    top = val[val["original_index"].isin(set(top_idx.tolist()))]
    full_psnr = float(val["realized_psnr_db_postclip"].median())
    top_psnr = float(top["realized_psnr_db_postclip"].median())
    coverage = float(top["class_id"].nunique() / max(1, val["class_id"].nunique()))
    return {
        "checkpoint_epoch": int(ladder["checkpoint_epoch"].iloc[0]),
        "report_timestep": int(ladder["report_timestep"].iloc[0]),
        "code_timestep_index": int(ladder["code_timestep_index"].iloc[0]),
        "median_within_image_spearman": float(np.median(rhos)),
        "mean_within_image_spearman": float(np.mean(rhos)),
        "iqr_within_image_spearman": float(np.percentile(rhos, 75) - np.percentile(rhos, 25)),
        "fraction_rho_positive": float(np.mean(np.asarray(rhos) > 0.0)),
        "pairwise_ordering_accuracy": float(np.mean(ordering)),
        "bootstrap_ci_lower": ci_lower,
        "bootstrap_ci_upper": ci_upper,
        "global_spearman_score_realized_psnr": spearman(val["score_value"], val["realized_psnr_db_postclip"]),
        "global_spearman_score_sigma": spearman(val["score_value"], val["sigma"]),
        "topk_median_realized_psnr_shift": float(top_psnr - full_psnr),
        "topk_class_coverage": coverage,
        "topk_jaccard": 1.0,
        "class_bias": float(abs(coverage - 1.0)),
    }


def stage_score_validate(ctx: Context) -> None:
    if is_done(ctx, "score_validate"):
        print("score_validate: resume skip")
        return
    if ctx.dry_run:
        print("score_validate: would scan checkpoints/timesteps")
        return
    train_ds = load_clean_dataset(ctx, train=True)
    val_indices = np.load(ctx.experiment_root / "00_protocol" / "score_validation_indices.npy")
    # Use the same validation images and noise levels for all candidates.
    epochs = resolved_epochs(ctx)
    timesteps = resolved_timesteps(ctx)
    k = int(deep_get(ctx.cfg, "smoke.coarse_mc_k" if ctx.smoke else "score_validation.coarse_mc_k", 4))
    use_oracle = bool(ctx.smoke and deep_get(ctx.cfg, "score_validation.oracle_scores_for_smoke", False))
    bootstrap_seed = int(deep_get(ctx.cfg, "seeds.bootstrap_seed", 123))
    bootstrap_iterations = int(deep_get(ctx.cfg, "score_validation.bootstrap_iterations", 200))
    out = ctx.experiment_root / "03_score_validation"
    out.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict[str, Any]] = []
    raw_ladder: list[pd.DataFrame] = []
    raw_val: list[pd.DataFrame] = []
    for epoch in epochs:
        model = load_ddpm_checkpoint(ctx, epoch)
        for timestep in timesteps:
            if timestep < 1 or timestep > model.n_steps:
                raise ValueError(f"Invalid report timestep {timestep}; T={model.n_steps}")
            ladder = ladder_scores(ctx, train_ds, val_indices, model, epoch, timestep, k, use_oracle)
            val = validation_corpus_scores(ctx, model, epoch, timestep, k, use_oracle)
            all_metrics.append(
                summarize_candidate(
                    ladder,
                    val,
                    float(deep_get(ctx.cfg, "score_validation.topk_ratio", 0.10)),
                    bootstrap_seed=bootstrap_seed,
                    bootstrap_iterations=bootstrap_iterations,
                )
            )
            raw_ladder.append(ladder)
            raw_val.append(val)
    coarse_metrics = pd.DataFrame(all_metrics)
    pd.concat(raw_ladder, ignore_index=True).to_csv(out / "raw_ladder_scores_coarse.csv", index=False)
    pd.concat(raw_val, ignore_index=True).to_csv(out / "validation_corpus_scores_coarse.csv", index=False)
    coarse_metrics.to_csv(out / "candidate_metrics_coarse.csv", index=False)

    final_k = int(deep_get(ctx.cfg, "smoke.final_mc_k" if ctx.smoke else "score_validation.final_mc_k", k))
    n_final = int(deep_get(ctx.cfg, "smoke.final_candidate_count" if ctx.smoke else "score_validation.final_candidate_count", 3))
    ranked = coarse_metrics.sort_values(
        ["median_within_image_spearman", "topk_median_realized_psnr_shift", "topk_jaccard", "class_bias", "checkpoint_epoch", "report_timestep"],
        ascending=[False, False, False, True, True, True],
    ).head(max(1, n_final))

    if final_k != k:
        final_metrics: list[dict[str, Any]] = []
        final_ladder: list[pd.DataFrame] = []
        final_val: list[pd.DataFrame] = []
        for row in ranked.itertuples(index=False):
            epoch = int(row.checkpoint_epoch)
            timestep = int(row.report_timestep)
            model = load_ddpm_checkpoint(ctx, epoch)
            ladder = ladder_scores(ctx, train_ds, val_indices, model, epoch, timestep, final_k, use_oracle)
            val = validation_corpus_scores(ctx, model, epoch, timestep, final_k, use_oracle)
            final_metrics.append(
                summarize_candidate(
                    ladder,
                    val,
                    float(deep_get(ctx.cfg, "score_validation.topk_ratio", 0.10)),
                    bootstrap_seed=bootstrap_seed,
                    bootstrap_iterations=bootstrap_iterations,
                )
            )
            final_ladder.append(ladder)
            final_val.append(val)
    else:
        keep = set((int(r.checkpoint_epoch), int(r.report_timestep)) for r in ranked.itertuples(index=False))
        final_metrics = [m for m in all_metrics if (int(m["checkpoint_epoch"]), int(m["report_timestep"])) in keep]
        final_ladder = [df for df in raw_ladder if (int(df["checkpoint_epoch"].iloc[0]), int(df["report_timestep"].iloc[0])) in keep]
        final_val = [df for df in raw_val if (int(df["checkpoint_epoch"].iloc[0]), int(df["report_timestep"].iloc[0])) in keep]

    pd.concat(final_ladder, ignore_index=True).to_csv(out / "raw_ladder_scores.csv", index=False)
    pd.concat(final_val, ignore_index=True).to_csv(out / "validation_corpus_scores.csv", index=False)
    pd.DataFrame(final_metrics).to_csv(out / "candidate_metrics.csv", index=False)
    write_json(
        out / "resolved_scan.json",
        {
            "epochs": epochs,
            "timesteps": timesteps,
            "coarse_mc_k": k,
            "final_mc_k": final_k,
            "final_candidate_count": n_final,
            "oracle_scores": use_oracle,
        },
    )
    mark_done(ctx, "score_validate", [out / "candidate_metrics.csv"])


def stage_score_select(ctx: Context) -> None:
    if is_done(ctx, "score_select"):
        print("score_select: resume skip")
        return
    if ctx.dry_run:
        print("score_select: would apply PASS/FAIL gate")
        return
    out = selected_score_root(ctx)
    out.mkdir(parents=True, exist_ok=True)
    # Apply validation gates before selecting checkpoint and timestep; keep the score sign.
    metrics = pd.read_csv(ctx.experiment_root / "03_score_validation" / "candidate_metrics.csv")
    min_class_coverage = float(
        deep_get(
            ctx.cfg,
            "smoke.min_class_coverage" if ctx.smoke else "score_validation.min_class_coverage",
            0.95,
        )
    )
    eligible = metrics[
        (metrics["bootstrap_ci_lower"] > float(deep_get(ctx.cfg, "score_validation.min_ci_lower", 0.0)))
        & (metrics["topk_median_realized_psnr_shift"] > float(deep_get(ctx.cfg, "score_validation.min_topk_psnr_shift", 0.0)))
        & (metrics["topk_class_coverage"] >= min_class_coverage)
        & (metrics["global_spearman_score_sigma"] > 0.0)
        & (metrics["global_spearman_score_realized_psnr"] < 0.0)
    ].copy()
    if ctx.force_fail_gate:
        eligible = eligible.iloc[0:0]
    if eligible.empty:
        write_json(out / "selection_diagnostic.json", {"status": "FAIL", "reason": "no eligible score-validation candidate"})
        (out / "FAIL.marker").write_text("FAIL\n", encoding="utf-8")
        raise SystemExit("score_select failed: no eligible candidate")
    eligible = eligible.sort_values(
        ["median_within_image_spearman", "topk_median_realized_psnr_shift", "topk_jaccard", "class_bias", "checkpoint_epoch", "report_timestep"],
        ascending=[False, False, False, True, True, True],
    )
    best = eligible.iloc[0].to_dict()
    epoch = int(best["checkpoint_epoch"])
    timestep = int(best["report_timestep"])
    ckpt = checkpoint_path(ctx, epoch)
    final_k = int(deep_get(ctx.cfg, "smoke.final_mc_k" if ctx.smoke else "score_validation.final_mc_k", 16))
    selected = {
        "dataset": dataset_key(ctx),
        "run_id": ctx.run_id,
        "corpus_manifest_path": str(ctx.experiment_root / "01_corpus" / "train_manifest.csv"),
        "corpus_manifest_hash": sha256_file(ctx.experiment_root / "01_corpus" / "train_manifest.csv"),
        "checkpoint_epoch": epoch,
        "checkpoint_path": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
        "ddpm_prediction_type": "epsilon",
        "data_domain": "image_space",
        "input_channels": dataset_channels(ctx),
        "output_channels": dataset_channels(ctx),
        "score_formula": "sum(flatten(epsilon_theta(x_t, t)) ** 2)",
        "score_reduction": "mean_over_mc",
        "score_direction": "higher_score_means_stronger_degradation",
        "report_timestep": timestep,
        "code_timestep_index": timestep - 1,
        "mc_k": final_k,
        "primary_metric_value": float(best["median_within_image_spearman"]),
        "bootstrap_ci": [float(best["bootstrap_ci_lower"]), float(best["bootstrap_ci_upper"])],
        "topk_psnr_shift": float(best["topk_median_realized_psnr_shift"]),
        "class_coverage": float(best["topk_class_coverage"]),
        "topk_jaccard": float(best["topk_jaccard"]),
        "config_hash": sha256_file(ctx.config_path),
        "oracle_scores": bool(ctx.smoke and deep_get(ctx.cfg, "score_validation.oracle_scores_for_smoke", False)),
    }
    (out / "selected_score_config.yaml").write_text(yaml.safe_dump(selected, sort_keys=False), encoding="utf-8")
    (out / "PASS.marker").write_text("PASS\n", encoding="utf-8")
    pd.DataFrame([best]).to_csv(out / "selection_report.csv", index=False)
    mark_done(ctx, "score_select", [out / "selected_score_config.yaml", out / "PASS.marker"])


def require_pass(ctx: Context) -> dict[str, Any]:
    selected_path = selected_score_root(ctx) / "selected_score_config.yaml"
    pass_marker = selected_score_root(ctx) / "PASS.marker"
    if not selected_path.exists() or not pass_marker.exists() or not marker_path(ctx, "score_select").exists():
        raise SystemExit("Filtering/downstream blocked: missing successful score_select PASS artifacts")
    return read_yaml(selected_path)


def save_subset_indices(path: Path, arr: np.ndarray) -> None:
    arr = np.sort(arr.astype(np.int64, copy=False))
    if path.exists():
        existing = np.load(path).astype(np.int64)
        if not np.array_equal(existing, arr):
            raise FileExistsError(f"Existing subset indices differ, refusing to overwrite: {path}")
        return
    np.save(path, arr)


def write_filter_subsets(ctx: Context, scores: pd.DataFrame) -> list[Path]:
    out = ctx.experiment_root / "05_filtering"
    fractions = subset_grid_fractions(ctx)
    subset_seed = int(deep_get(ctx.cfg, "filtering.random_seed", deep_get(ctx.cfg, "seeds.subset_seed", 42)))
    validation_path = ctx.experiment_root / "00_protocol" / "downstream_validation_indices.npy"
    downstream_validation = np.load(validation_path).astype(np.int64) if validation_path.exists() else np.asarray([], dtype=np.int64)
    eligible_scores = scores[~scores["original_index"].isin(set(downstream_validation.tolist()))].copy()
    source_pool = np.sort(eligible_scores["original_index"].to_numpy(dtype=np.int64))
    subsets = {"full": source_pool}
    for fraction in fractions:
        tag = fraction_tag(fraction)
        n_subset = subset_size_for_fraction(len(source_pool), fraction)
        subsets[f"topk_{tag}"] = select_topk_exact(eligible_scores, n_subset)
        subsets[f"quantile_spread_{tag}"] = select_quantile_spread_exact(
            eligible_scores,
            n_subset,
            seed=subset_seed,
            min_points_per_bin=int(deep_get(ctx.cfg, "filtering.quantile_min_points_per_bin", 30)),
        )
        subsets[f"random_{tag}"] = select_random_exact(eligible_scores, n_subset, seed=subset_seed)
    if "topk_p10" in subsets:
        subsets["topk_10"] = subsets["topk_p10"]
        subsets["quantile_spread_10"] = subsets["quantile_spread_p10"]
        subsets["random_10"] = subsets["random_p10"]
    idx_dir = out / "subset_indices"
    idx_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = out / "subset_metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in subsets.items():
        arr = np.sort(arr.astype(np.int64, copy=False))
        expected = len(source_pool) if name == "full" else subset_size_for_fraction(len(source_pool), variant_strategy_fraction(name)[1])
        if len(arr) != expected:
            raise AssertionError(f"{name} selected {len(arr)} samples, expected {expected}")
        if len(np.unique(arr)) != len(arr):
            raise AssertionError(f"{name} contains duplicate original indices")
        index_path = idx_dir / variant_index_filename(name)
        save_subset_indices(index_path, arr)
        if name != "full" and name not in VARIANTS:
            pdir = out / fraction_tag(variant_strategy_fraction(name)[1])
            pdir.mkdir(parents=True, exist_ok=True)
            save_subset_indices(pdir / variant_index_filename(name), arr)
        sub = scores[scores["original_index"].isin(set(arr.tolist()))]
        strategy, fraction = variant_strategy_fraction(name)
        method = "quantile_spread" if strategy == "quantile" else strategy
        metadata = {
            "subset": name,
            "method": method,
            "fraction": float(fraction),
            "K": int(len(arr)),
            "source_pool_size": int(len(source_pool)),
            "excluded_downstream_validation_count": int(len(downstream_validation)),
            "seed": None if name == "full" else subset_seed,
            "without_replacement": True,
            "selection_independent_of_score": bool(strategy == "random"),
            "score_used_for_selection": bool(strategy in {"topk", "quantile"}),
            "class_coverage": float(sub["class_id"].nunique() / max(1, scores["class_id"].nunique())),
            "represented_classes": int(sub["class_id"].nunique()),
            "indices_sha256": sha256_file(index_path),
        }
        write_json(meta_dir / f"{name}.json", metadata)
        if name != "full" and name not in VARIANTS:
            write_json(out / fraction_tag(fraction) / f"{name}_metadata.json", metadata)
    summary = []
    for name, arr in subsets.items():
        sub = scores[scores["original_index"].isin(set(arr.tolist()))]
        strategy, fraction = variant_strategy_fraction(name)
        summary.append(
            {
                "subset": name,
                "strategy": strategy,
                "fraction": float(fraction),
                "n": int(len(arr)),
                "classes": int(sub["class_id"].nunique()),
                "class_coverage": float(sub["class_id"].nunique() / max(1, scores["class_id"].nunique())),
                "median_realized_psnr": float(sub["realized_psnr_db_postclip"].median()),
            }
        )
    pd.DataFrame(summary).to_csv(out / "selection_summary.csv", index=False)
    return [out / "selection_summary.csv", idx_dir]


def stage_filter(ctx: Context) -> None:
    if is_done(ctx, "filter"):
        stage_skip("filter", "completed marker found")
        return
    if ctx.dry_run:
        print("filter: would compute full-corpus scores and subsets")
        return
    selected = require_pass(ctx)
    manifest = pd.read_csv(ctx.experiment_root / "01_corpus" / "train_manifest.csv")
    corpus = MaterializedCorpusDataset(ctx.experiment_root / "01_corpus" / "train_corpus.pt", ctx.experiment_root / "01_corpus" / "train_manifest.csv")
    use_oracle = bool(selected.get("oracle_scores", False))
    model = load_ddpm_checkpoint(ctx, int(selected["checkpoint_epoch"]))
    timestep = int(selected["report_timestep"])
    k = int(selected["mc_k"])
    rows: list[dict[str, Any]] = []
    loader = DataLoader(corpus, batch_size=int(deep_get(ctx.cfg, "evaluation.batch_size", 256)), shuffle=False, num_workers=0)
    offset = 0
    indices = manifest["original_index"].to_numpy(dtype=np.int64)
    for batch in loader:
        current = indices[offset : offset + batch.shape[0]]
        meta = manifest.set_index("original_index").loc[current]
        if use_oracle:
            scores = meta["sigma"].to_numpy(dtype=np.float64) ** 2
            stds = np.zeros_like(scores)
        else:
            scores, stds = score_batch(model, batch, timestep, mc_seeds(ctx, k, "full", offset))
        for original_index, score, std in zip(current, scores, stds):
            row = manifest[manifest["original_index"] == int(original_index)].iloc[0].to_dict()
            rows.append({"score_value": float(score), "score_std": float(std), "checkpoint_epoch": selected["checkpoint_epoch"], "report_timestep": timestep, "mc_k": k, **row})
        offset += batch.shape[0]
    out = ctx.experiment_root / "05_filtering"
    out.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(rows)
    ranked = scores.sort_values(["score_value", "original_index"], ascending=[True, True]).reset_index()
    rank_by_index = pd.Series(np.arange(1, len(ranked) + 1, dtype=np.int64), index=ranked["index"])
    scores["score_rank"] = rank_by_index.loc[scores.index].to_numpy(dtype=np.int64)
    scores.to_csv(out / "scores_full.csv", index=False)
    outputs = write_filter_subsets(ctx, scores)
    mark_done(ctx, "filter", [out / "scores_full.csv", *outputs])


def stage_filter_grid(ctx: Context) -> None:
    if is_done(ctx, "filter_grid"):
        stage_skip("filter_grid", "completed marker found")
        return
    if ctx.dry_run:
        print("filter_grid: would reuse scores_full.csv when present and write subset grid")
        return
    require_pass(ctx)
    out = ctx.experiment_root / "05_filtering"
    scores_path = out / "scores_full.csv"
    if scores_path.exists():
        scores = pd.read_csv(scores_path)
        outputs = write_filter_subsets(ctx, scores)
    else:
        filter_ctx = Context(**{**ctx.__dict__, "resume": False})
        stage_filter(filter_ctx)
        outputs = [scores_path, out / "subset_indices"]
    if not ctx.dry_run:
        mark_done(ctx, "filter_grid", [scores_path, *outputs])


def ensure_initial_tdncnn(ctx: Context) -> Path:
    path = ctx.experiment_root / "06_downstream" / "tdncnn" / "initial_state.pt"
    if path.exists():
        return path
    torch.manual_seed(int(deep_get(ctx.cfg, "seeds.model_seed", 42)))
    model = DnCNN_2D(
        in_channels=dataset_channels(ctx),
        num_layers=int(deep_get(ctx.cfg, "downstream.tdncnn.num_layers", deep_get(ctx.cfg, "downstream.num_layers", 8))),
        features=int(deep_get(ctx.cfg, "downstream.tdncnn.features", deep_get(ctx.cfg, "downstream.features", 64))),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def build_tdncnn(ctx: Context) -> nn.Module:
    return DnCNN_2D(
        in_channels=dataset_channels(ctx),
        num_layers=int(deep_get(ctx.cfg, "downstream.tdncnn.num_layers", deep_get(ctx.cfg, "downstream.num_layers", 8))),
        features=int(deep_get(ctx.cfg, "downstream.tdncnn.features", deep_get(ctx.cfg, "downstream.features", 64))),
    )


def variant_from_stage(stage: str) -> str:
    legacy = {
        "tdncnn_full": "full",
        "tdncnn_topk": "topk_10",
        "tdncnn_quantile": "quantile_spread_10",
        "tdncnn_random": "random_10",
        "downstream_full": "full",
    }
    if stage in legacy:
        return legacy[stage]
    for fraction in (10, 20, 30):
        suffix = f"p{fraction}"
        if stage == f"downstream_topk_{suffix}":
            return f"topk_{suffix}"
        if stage == f"downstream_quantile_{suffix}":
            return f"quantile_spread_{suffix}"
        if stage == f"downstream_random_{suffix}":
            return f"random_{suffix}"
    raise KeyError(stage)


def stage_tdncnn_train(ctx: Context, stage: str) -> None:
    if is_done(ctx, stage):
        stage_skip(stage, "completed marker found")
        return
    if ctx.dry_run:
        print(f"{stage}: would train TDnCNN variant")
        return
    require_pass(ctx)
    variant = variant_from_stage(stage)
    out = ctx.experiment_root / "06_downstream" / "tdncnn" / variant
    if ctx.resume and (out / "best.pt").exists() and (out / "training_history.csv").exists() and (out / "run_meta.json").exists():
        stage_skip(stage, f"existing TDnCNN artifacts found for {variant}")
        return
    indices = np.load(variant_indices_path(ctx, variant))
    corpus = MaterializedCorpusDataset(
        ctx.experiment_root / "01_corpus" / "train_corpus.pt",
        ctx.experiment_root / "01_corpus" / "train_manifest.csv",
        original_indices=indices,
        return_original_index=True,
    )
    train_ds = CorpusTaskNoiseDataset(
        corpus,
        dataset_name=dataset_key(ctx),
        task_seed=int(deep_get(ctx.cfg, "seeds.task_noise_seed", 4242)),
        task_noise_version=str(deep_get(ctx.cfg, "downstream.task_noise.version", "mnist_task_awgn_v1")),
        sigma_min=float(deep_get(ctx.cfg, "downstream.task_noise.sigma_min", 0.1)),
        sigma_max=float(deep_get(ctx.cfg, "downstream.task_noise.sigma_max", 0.8)),
    )
    batch_size = int(deep_get(ctx.cfg, "smoke.downstream_batch_size" if ctx.smoke else "downstream.batch_size", 128))
    configured_max_steps = int(deep_get(ctx.cfg, "smoke.downstream_max_optimizer_steps" if ctx.smoke else "downstream.max_optimizer_steps", 2000))
    max_steps = downstream_max_steps_for_variant(ctx, variant, len(indices), batch_size, configured_max_steps)
    val_interval = int(
        deep_get(ctx.cfg, "smoke.downstream_validation_interval_steps" if ctx.smoke else "downstream.validation_interval_steps", 100)
    )
    patience = int(
        deep_get(ctx.cfg, "smoke.downstream_early_stopping_patience" if ctx.smoke else "downstream.early_stopping.patience", 10)
    )
    min_delta = float(deep_get(ctx.cfg, "downstream.early_stopping.min_delta", 1e-4))
    warmup_steps = int(
        deep_get(ctx.cfg, "smoke.downstream_early_stopping_warmup_steps" if ctx.smoke else "downstream.early_stopping.warmup_steps", 100)
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_indices = np.load(ctx.experiment_root / "00_protocol" / "downstream_validation_indices.npy").astype(np.int64)
    val_corpus = MaterializedCorpusDataset(
        ctx.experiment_root / "01_corpus" / "train_corpus.pt",
        ctx.experiment_root / "01_corpus" / "train_manifest.csv",
        original_indices=validation_indices,
        return_original_index=True,
    )
    val_ds = CorpusTaskNoiseDataset(
        val_corpus,
        dataset_name=dataset_key(ctx),
        task_seed=int(deep_get(ctx.cfg, "seeds.task_noise_seed", 4242)) + 100003,
        task_noise_version=str(deep_get(ctx.cfg, "downstream.task_noise.version", "mnist_task_awgn_v1")) + "_validation",
        sigma_min=float(deep_get(ctx.cfg, "downstream.task_noise.sigma_min", 0.1)),
        sigma_max=float(deep_get(ctx.cfg, "downstream.task_noise.sigma_max", 0.8)),
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model = build_tdncnn(ctx).to(ctx.device)
    model.load_state_dict(torch.load(ensure_initial_tdncnn(ctx), map_location=ctx.device), strict=True)
    opt = torch.optim.Adam(model.parameters(), lr=float(deep_get(ctx.cfg, "downstream.lr", 1e-3)))
    rows: list[dict[str, Any]] = []
    step = 0
    best = math.inf
    patience_best = math.inf
    best_step = 0
    checks_without_improvement = 0
    stop_reason = "max_steps"
    out.mkdir(parents=True, exist_ok=True)
    log_start = time.time()
    while step < max_steps:
        for x_in, x_target in loader:
            model.train()
            x_in = x_in.to(ctx.device)
            x_target = x_target.to(ctx.device)
            opt.zero_grad(set_to_none=True)
            pred = model(x_in)
            loss = F.mse_loss(pred, x_target)
            loss.backward()
            opt.step()
            step += 1
            value = float(loss.item())
            row = {"step": step, "train_loss": value, "val_loss": np.nan, "best_val_loss": best}
            if step == 1 or step % val_interval == 0 or step >= max_steps:
                model.eval()
                val_total = 0.0
                val_count = 0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx = vx.to(ctx.device)
                        vy = vy.to(ctx.device)
                        pred_val = model(vx)
                        val_total += float(F.mse_loss(pred_val, vy, reduction="sum").item())
                        val_count += int(np.prod(vy.shape))
                model.train()
                val_loss = val_total / max(val_count, 1)
                row["val_loss"] = val_loss
                # Save the lowest validation loss; min_delta and warmup control patience.
                if val_loss < best:
                    best = val_loss
                    best_step = step
                    torch.save({"model_state_dict": model.state_dict(), "step": step, "variant": variant, "best_val_loss": best}, out / "best.pt")
                # Count patience after warmup; allow best.pt to improve at any check.
                if step >= warmup_steps and val_loss < patience_best - min_delta:
                    patience_best = val_loss
                    checks_without_improvement = 0
                elif step >= warmup_steps:
                    checks_without_improvement += 1
                    if checks_without_improvement >= patience:
                        stop_reason = "early_stopping"
                row["best_val_loss"] = best
            rows.append(row)
            if should_log_progress(step, max_steps, val_interval) or np.isfinite(row["val_loss"]):
                training_progress(
                    dataset_title(ctx),
                    f"tdncnn_{variant}",
                    unit="step",
                    current=step,
                    total=max_steps,
                    train_loss=value,
                    val_loss=row["val_loss"],
                    best_val_loss=best,
                    best_step=best_step,
                    patience_counter=checks_without_improvement,
                    patience=patience,
                    lr=opt.param_groups[0]["lr"],
                    start_time=log_start,
                )
                pd.DataFrame(rows).to_csv(out / "training_history.csv", index=False)
            if step >= max_steps:
                break
            if stop_reason == "early_stopping":
                break
        if stop_reason == "early_stopping":
            break
    if not (out / "best.pt").exists():
        torch.save({"model_state_dict": model.state_dict(), "step": step, "variant": variant, "best_val_loss": best}, out / "best.pt")
    torch.save({"model_state_dict": model.state_dict(), "step": step, "variant": variant}, out / "last.pt")
    history_df = pd.DataFrame(rows)
    history_df.to_csv(out / "training_history.csv", index=False)
    finite_val = history_df["val_loss"].dropna() if "val_loss" in history_df else pd.Series(dtype=float)
    write_json(
        out / "run_meta.json",
        {
            "variant": variant,
            "train_size": int(len(indices)),
            "batch_size": int(batch_size),
            "max_steps": int(max_steps),
            "configured_max_steps": int(configured_max_steps),
            "budget_mode": downstream_budget_mode(ctx),
            "optimizer_steps": int(step),
            "initial_state": str(ensure_initial_tdncnn(ctx)),
            "early_stopping_enabled": True,
            "early_stopping_monitor": "val_loss",
            "early_stopping_patience": int(patience),
            "early_stopping_min_delta": float(min_delta),
            "validation_interval_steps": int(val_interval),
            "best_step": int(best_step),
            "best_val_loss": float(best),
            "final_val_loss": float(finite_val.iloc[-1]) if not finite_val.empty else float("nan"),
            **budget_fields(best_step, len(indices), batch_size, "best"),
            **budget_fields(step, len(indices), batch_size, "stop"),
            "stop_reason": stop_reason,
        },
    )
    mark_done(ctx, stage, [out / "best.pt", out / "training_history.csv"])


def build_current_drunet(ctx: Context) -> nn.Module:
    official = bool(deep_get(ctx.cfg, "smoke.drunet_official", False) if ctx.smoke else deep_get(ctx.cfg, "drunet.official", True))
    return build_drunet(
        in_channels=dataset_channels(ctx),
        official=official,
        features=int(deep_get(ctx.cfg, "drunet.features", 64)),
        num_layers=int(deep_get(ctx.cfg, "drunet.num_layers", 5)),
        nc=deep_get(ctx.cfg, "drunet.nc", None),
        nb=int(deep_get(ctx.cfg, "drunet.nb", 4)),
        act_mode=str(deep_get(ctx.cfg, "drunet.act_mode", "R")),
        downsample_mode=str(deep_get(ctx.cfg, "drunet.downsample_mode", "strideconv")),
        upsample_mode=str(deep_get(ctx.cfg, "drunet.upsample_mode", "convtranspose")),
    )


def ensure_initial_drunet(ctx: Context) -> Path:
    path = ctx.experiment_root / "06_downstream" / "drunet" / "initial_state.pt"
    if path.exists():
        return path
    torch.manual_seed(int(deep_get(ctx.cfg, "seeds.model_seed", 42)))
    model = build_current_drunet(ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def variant_from_drunet_stage(stage: str) -> str:
    legacy = {
        "drunet_full": "full",
        "drunet_topk": "topk_10",
        "drunet_quantile": "quantile_spread_10",
        "drunet_random": "random_10",
    }
    if stage in legacy:
        return legacy[stage]
    for fraction in (20, 30):
        suffix = f"p{fraction}"
        if stage == f"drunet_topk_{suffix}":
            return f"topk_{suffix}"
        if stage == f"drunet_quantile_{suffix}":
            return f"quantile_spread_{suffix}"
        if stage == f"drunet_random_{suffix}":
            return f"random_{suffix}"
    raise KeyError(stage)


def drunet_out_dir(ctx: Context, variant: str) -> Path:
    return ctx.experiment_root / "06_downstream" / "drunet" / variant


def drunet_dataset(ctx: Context, original_indices: np.ndarray, *, validation: bool) -> CorpusTaskNoiseWithSigmaDataset:
    corpus = MaterializedCorpusDataset(
        ctx.experiment_root / "01_corpus" / "train_corpus.pt",
        ctx.experiment_root / "01_corpus" / "train_manifest.csv",
        original_indices=original_indices,
        return_original_index=True,
    )
    suffix = "_validation" if validation else ""
    return CorpusTaskNoiseWithSigmaDataset(
        corpus,
        dataset_name=dataset_key(ctx),
        task_seed=int(deep_get(ctx.cfg, "seeds.task_noise_seed", 4242)) + (100003 if validation else 0),
        task_noise_version=str(deep_get(ctx.cfg, "downstream.task_noise.version", "mnist_task_awgn_v1")) + suffix,
        sigma_min=float(deep_get(ctx.cfg, "downstream.task_noise.sigma_min", 0.1)),
        sigma_max=float(deep_get(ctx.cfg, "downstream.task_noise.sigma_max", 0.8)),
    )


@torch.no_grad()
def drunet_validation_loss(ctx: Context, model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    total = 0.0
    count = 0
    for x_in, x_target, sigma in loader:
        x_in = x_in.to(ctx.device)
        x_target = x_target.to(ctx.device)
        sigma = sigma.to(ctx.device)
        pred = model(x_in, sigma=sigma)
        total += float(F.mse_loss(pred, x_target, reduction="sum").item())
        count += int(np.prod(x_target.shape))
    model.train()
    return total / max(count, 1)


def stage_drunet_train(ctx: Context, stage: str) -> None:
    if is_done(ctx, stage):
        stage_skip(stage, "completed marker found")
        return
    if ctx.dry_run:
        print(f"{stage}: would train DRUNet variant")
        return
    require_pass(ctx)
    variant = variant_from_drunet_stage(stage)
    out = drunet_out_dir(ctx, variant)
    if ctx.resume and (out / "best.pt").exists() and (out / "training_history.csv").exists() and (out / "run_meta.json").exists():
        stage_skip(stage, f"existing DRUNet artifacts found for {variant}")
        return
    indices = np.load(variant_indices_path(ctx, variant))
    train_ds = drunet_dataset(ctx, indices.astype(np.int64), validation=False)
    validation_indices = np.load(ctx.experiment_root / "00_protocol" / "downstream_validation_indices.npy").astype(np.int64)
    val_ds = drunet_dataset(ctx, validation_indices, validation=True)
    batch_size = int(deep_get(ctx.cfg, "smoke.drunet_batch_size" if ctx.smoke else "drunet.batch_size", deep_get(ctx.cfg, "smoke.downstream_batch_size" if ctx.smoke else "downstream.batch_size", 64)))
    configured_max_steps = int(deep_get(ctx.cfg, "smoke.drunet_max_optimizer_steps" if ctx.smoke else "drunet.max_steps", deep_get(ctx.cfg, "smoke.downstream_max_optimizer_steps" if ctx.smoke else "downstream.max_optimizer_steps", 2000)))
    max_steps = downstream_max_steps_for_variant(ctx, variant, len(indices), batch_size, configured_max_steps)
    val_interval = int(deep_get(ctx.cfg, "smoke.drunet_validation_interval_steps" if ctx.smoke else "drunet.validation_interval_steps", deep_get(ctx.cfg, "smoke.downstream_validation_interval_steps" if ctx.smoke else "downstream.validation_interval_steps", 100)))
    patience = int(deep_get(ctx.cfg, "smoke.drunet_early_stopping_patience" if ctx.smoke else "drunet.early_stopping.patience", deep_get(ctx.cfg, "smoke.downstream_early_stopping_patience" if ctx.smoke else "downstream.early_stopping.patience", 10)))
    min_delta = float(deep_get(ctx.cfg, "drunet.early_stopping.min_delta", deep_get(ctx.cfg, "downstream.early_stopping.min_delta", 1e-4)))
    warmup_steps = int(deep_get(ctx.cfg, "smoke.drunet_early_stopping_warmup_steps" if ctx.smoke else "drunet.early_stopping.warmup_steps", deep_get(ctx.cfg, "smoke.downstream_early_stopping_warmup_steps" if ctx.smoke else "downstream.early_stopping.warmup_steps", 100)))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model = build_current_drunet(ctx).to(ctx.device)
    model.load_state_dict(torch.load(ensure_initial_drunet(ctx), map_location=ctx.device), strict=True)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(deep_get(ctx.cfg, "drunet.lr", 1e-4)),
        weight_decay=float(deep_get(ctx.cfg, "drunet.weight_decay", 0.0)),
    )
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    step = 0
    best = math.inf
    patience_best = math.inf
    best_step = 0
    checks_without_improvement = 0
    stop_reason = "max_steps"
    log_start = time.time()
    while step < max_steps:
        for x_in, x_target, sigma in loader:
            model.train()
            x_in = x_in.to(ctx.device)
            x_target = x_target.to(ctx.device)
            sigma = sigma.to(ctx.device)
            opt.zero_grad(set_to_none=True)
            pred = model(x_in, sigma=sigma)
            loss = F.mse_loss(pred, x_target)
            loss.backward()
            opt.step()
            step += 1
            row = {"step": step, "train_loss": float(loss.item()), "val_loss": np.nan, "best_val_loss": best}
            if step == 1 or step % val_interval == 0 or step >= max_steps:
                val_loss = drunet_validation_loss(ctx, model, val_loader)
                row["val_loss"] = val_loss
                # Save the true validation minimum even when its improvement is below min_delta.
                if val_loss < best:
                    best = val_loss
                    best_step = step
                    torch.save({"model_state_dict": model.state_dict(), "step": step, "variant": variant, "best_val_loss": best}, out / "best.pt")
                if step >= warmup_steps and val_loss < patience_best - min_delta:
                    patience_best = val_loss
                    checks_without_improvement = 0
                elif step >= warmup_steps:
                    checks_without_improvement += 1
                    if checks_without_improvement >= patience:
                        stop_reason = "early_stopping"
                row["best_val_loss"] = best
            rows.append(row)
            if should_log_progress(step, max_steps, val_interval) or np.isfinite(row["val_loss"]):
                training_progress(
                    dataset_title(ctx),
                    f"drunet_{variant}",
                    unit="step",
                    current=step,
                    total=max_steps,
                    train_loss=row["train_loss"],
                    val_loss=row["val_loss"],
                    best_val_loss=best,
                    best_step=best_step,
                    patience_counter=checks_without_improvement,
                    patience=patience,
                    lr=opt.param_groups[0]["lr"],
                    start_time=log_start,
                )
                pd.DataFrame(rows).to_csv(out / "training_history.csv", index=False)
            if step >= max_steps or stop_reason == "early_stopping":
                break
        if step >= max_steps or stop_reason == "early_stopping":
            break
    if not (out / "best.pt").exists():
        torch.save({"model_state_dict": model.state_dict(), "step": step, "variant": variant, "best_val_loss": best}, out / "best.pt")
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(), "step": step, "variant": variant}, out / "last.pt")
    history_df = pd.DataFrame(rows)
    history_df.to_csv(out / "training_history.csv", index=False)
    finite_val = history_df["val_loss"].dropna() if "val_loss" in history_df else pd.Series(dtype=float)
    write_json(
        out / "run_meta.json",
        {
            "model": "drunet",
            "variant": variant,
            "train_size": int(len(indices)),
            "batch_size": int(batch_size),
            "max_steps": int(max_steps),
            "configured_max_steps": int(configured_max_steps),
            "budget_mode": downstream_budget_mode(ctx),
            "optimizer_steps": int(step),
            "initial_state": str(ensure_initial_drunet(ctx)),
            "early_stopping_enabled": True,
            "best_step": int(best_step),
            "best_val_loss": float(best),
            "final_val_loss": float(finite_val.iloc[-1]) if not finite_val.empty else float("nan"),
            **budget_fields(best_step, len(indices), batch_size, "best"),
            **budget_fields(step, len(indices), batch_size, "stop"),
            "stop_reason": stop_reason,
        },
    )
    mark_done(ctx, stage, [out / "best.pt", out / "training_history.csv"])


def psnr_from_mse(mse: float) -> float:
    return float(10.0 * math.log10(1.0 / max(float(mse), EPS)))


def ssim_image(pred: torch.Tensor, target: torch.Tensor) -> float:
    if skimage_ssim is None:
        return float("nan")
    p = pred.detach().cpu().numpy()
    t = target.detach().cpu().numpy()
    if p.shape[0] == 1:
        return float(skimage_ssim(t[0], p[0], data_range=1.0))
    return float(skimage_ssim(t, p, data_range=1.0, channel_axis=0))


def dataframe_markdown(df: pd.DataFrame) -> str:
    columns = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float):
                values.append("nan" if math.isnan(value) else f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def grid_evaluation_root(ctx: Context) -> Path:
    return ctx.experiment_root / "07_evaluation"


def grid_figures_root(ctx: Context) -> Path:
    return ctx.experiment_root / "08_figures"


def ensure_grid_eval_noise_cache(ctx: Context) -> tuple[Path, Path, Path]:
    test_ds = load_clean_dataset(ctx, train=False)
    eval_indices = np.load(ctx.experiment_root / "00_protocol" / "evaluation_indices.npy")
    return materialize_eval_noise_cache(
        dataset=test_ds,
        dataset_name=dataset_key(ctx),
        original_indices=eval_indices,
        output_dir=grid_evaluation_root(ctx) / "eval_noise_cache",
        sigma_min=float(deep_get(ctx.cfg, "evaluation.eval_noise.sigma_min", 0.1)),
        sigma_max=float(deep_get(ctx.cfg, "evaluation.eval_noise.sigma_max", 0.8)),
        eval_seed=int(deep_get(ctx.cfg, "seeds.eval_noise_seed", 4242)),
        eval_noise_version=str(deep_get(ctx.cfg, "evaluation.eval_noise.version", "mnist_eval_awgn_v1")),
        overwrite=False,
    )


def tdncnn_variant_dir(ctx: Context, variant: str) -> Path:
    return ctx.experiment_root / "06_downstream" / "tdncnn" / variant


def tdncnn_variant_checkpoint(ctx: Context, variant: str) -> Path:
    direct = tdncnn_variant_dir(ctx, variant) / "best.pt"
    if direct.exists():
        return direct
    reverse = {v: k for k, v in LEGACY_P10_ALIASES.items()}
    legacy = reverse.get(variant)
    if legacy is not None:
        legacy_path = tdncnn_variant_dir(ctx, legacy) / "best.pt"
        if legacy_path.exists():
            return legacy_path
    return direct


def drunet_variant_checkpoint(ctx: Context, variant: str) -> Path:
    direct = drunet_out_dir(ctx, variant) / "best.pt"
    if direct.exists():
        return direct
    reverse = {v: k for k, v in LEGACY_P10_ALIASES.items()}
    legacy = reverse.get(variant)
    if legacy is not None:
        legacy_path = drunet_out_dir(ctx, legacy) / "best.pt"
        if legacy_path.exists():
            return legacy_path
    return direct


@torch.no_grad()
def evaluate_tdncnn_variants(ctx: Context, variants: list[str], *, out: Path, grid: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = ensure_grid_eval_noise_cache(ctx) if grid else ensure_eval_noise_cache(ctx)
    eval_ds = EvalNoiseDataset(paths[0], paths[1])
    loader = DataLoader(eval_ds, batch_size=int(deep_get(ctx.cfg, "evaluation.batch_size", 256)), shuffle=False)
    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for variant in variants:
        ckpt = tdncnn_variant_checkpoint(ctx, variant)
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing TDnCNN checkpoint for {variant}: {ckpt}")
        model = build_tdncnn(ctx).to(ctx.device)
        model.load_state_dict(torch.load(ckpt, map_location=ctx.device)["model_state_dict"], strict=True)
        model.eval()
        idx = 0
        values = []
        strategy, fraction = variant_strategy_fraction(variant)
        for x_in, x_target in loader:
            x_in = x_in.to(ctx.device)
            pred = model(x_in).cpu()
            target = x_target.cpu()
            for p, t in zip(pred, target):
                mse = float(((p - t) ** 2).mean().item())
                row = {
                    "model": "TDnCNN",
                    "variant": variant,
                    "strategy": strategy,
                    "fraction": float(fraction),
                    "image_index": idx,
                    "mse": mse,
                    "psnr": psnr_from_mse(mse),
                    "ssim": ssim_image(p, t),
                    "lpips": float("nan"),
                }
                rows.append(row)
                values.append(row)
                idx += 1
        frame = pd.DataFrame(values)
        meta_path = ckpt.parent / "run_meta.json"
        meta = read_json(meta_path) if meta_path.exists() else {}
        train_size = int(len(np.load(variant_indices_path(ctx, variant))))
        batch_size = int(meta.get("batch_size", deep_get(ctx.cfg, "downstream.batch_size", 128)))
        best_step = int(meta.get("best_step", torch.load(ckpt, map_location="cpu").get("step", 0)))
        summary.append(
            {
                "model": "TDnCNN",
                "strategy": strategy,
                "fraction": float(fraction),
                "variant": variant,
                "train_size": train_size,
                "max_steps": int(meta.get("max_steps", meta.get("optimizer_steps", 0))),
                "best_step": best_step,
                "stopped_step": int(meta.get("optimizer_steps", 0)),
                "effective_epochs_at_best": float(meta.get("effective_epochs_at_best", best_step * batch_size / max(1, train_size))),
                "total_seen_samples_at_best": int(meta.get("total_seen_samples_at_best", best_step * batch_size)),
                "best_val_loss": float(meta.get("best_val_loss", float("nan"))),
                "PSNR": float(frame["psnr"].mean()),
                "SSIM": float(frame["ssim"].mean()),
                "MSE": float(frame["mse"].mean()),
                "random_seed": int(deep_get(ctx.cfg, "filtering.random_seed", deep_get(ctx.cfg, "seeds.subset_seed", 42))) if strategy == "random" else np.nan,
            }
        )
    per = pd.DataFrame(rows)
    summ = pd.DataFrame(summary)
    full = summ[summ["strategy"] == "full"].iloc[0]
    summ["delta_PSNR_vs_Full"] = summ["PSNR"] - float(full["PSNR"])
    summ["delta_SSIM_vs_Full"] = summ["SSIM"] - float(full["SSIM"])
    summ["delta_MSE_vs_Full"] = summ["MSE"] - float(full["MSE"])
    out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "per_image_metrics.csv", index=False)
    ordered_cols = [
        "model",
        "strategy",
        "fraction",
        "variant",
        "train_size",
        "max_steps",
        "best_step",
        "stopped_step",
        "effective_epochs_at_best",
        "total_seen_samples_at_best",
        "best_val_loss",
        "PSNR",
        "SSIM",
        "MSE",
        "delta_PSNR_vs_Full",
        "delta_SSIM_vs_Full",
        "delta_MSE_vs_Full",
        "random_seed",
    ]
    summ[ordered_cols].to_csv(out / "subset_grid_results.csv", index=False)
    (out / "subset_grid_results.md").write_text(dataframe_markdown(summ[ordered_cols]) + "\n", encoding="utf-8")
    return per, summ[ordered_cols]


@torch.no_grad()
def evaluate_drunet_variants(ctx: Context, variants: list[str], *, out: Path, grid: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = ensure_grid_eval_noise_cache(ctx) if grid else ensure_eval_noise_cache(ctx)
    eval_ds = EvalNoiseWithSigmaDataset(paths[0], paths[1], paths[2])
    loader = DataLoader(eval_ds, batch_size=int(deep_get(ctx.cfg, "evaluation.batch_size", 256)), shuffle=False)
    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for variant in variants:
        ckpt = drunet_variant_checkpoint(ctx, variant)
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing DRUNet checkpoint for {variant}: {ckpt}")
        model = build_current_drunet(ctx).to(ctx.device)
        payload = torch.load(ckpt, map_location=ctx.device)
        model.load_state_dict(payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload, strict=True)
        model.eval()
        idx = 0
        values = []
        strategy, fraction = variant_strategy_fraction(variant)
        for x_in, x_target, sigma in loader:
            x_in = x_in.to(ctx.device)
            sigma = sigma.to(ctx.device)
            pred = model(x_in, sigma=sigma).cpu()
            target = x_target.cpu()
            for p, t in zip(pred, target):
                mse = float(((p - t) ** 2).mean().item())
                row = {
                    "model": "DRUNet",
                    "variant": variant,
                    "strategy": strategy,
                    "fraction": float(fraction),
                    "image_index": idx,
                    "mse": mse,
                    "psnr": psnr_from_mse(mse),
                    "ssim": ssim_image(p, t),
                    "lpips": float("nan"),
                }
                rows.append(row)
                values.append(row)
                idx += 1
        frame = pd.DataFrame(values)
        meta_path = ckpt.parent / "run_meta.json"
        meta = read_json(meta_path) if meta_path.exists() else {}
        train_size = int(len(np.load(variant_indices_path(ctx, variant))))
        batch_size = int(meta.get("batch_size", deep_get(ctx.cfg, "drunet.batch_size", deep_get(ctx.cfg, "downstream.batch_size", 64))))
        best_step = int(meta.get("best_step", torch.load(ckpt, map_location="cpu").get("step", 0)))
        summary.append(
            {
                "model": "DRUNet",
                "strategy": strategy,
                "fraction": float(fraction),
                "variant": variant,
                "train_size": train_size,
                "max_steps": int(meta.get("max_steps", meta.get("optimizer_steps", 0))),
                "best_step": best_step,
                "stopped_step": int(meta.get("optimizer_steps", 0)),
                "effective_epochs_at_best": float(meta.get("effective_epochs_at_best", best_step * batch_size / max(1, train_size))),
                "total_seen_samples_at_best": int(meta.get("total_seen_samples_at_best", best_step * batch_size)),
                "best_val_loss": float(meta.get("best_val_loss", float("nan"))),
                "PSNR": float(frame["psnr"].mean()),
                "SSIM": float(frame["ssim"].mean()),
                "MSE": float(frame["mse"].mean()),
                "random_seed": int(deep_get(ctx.cfg, "filtering.random_seed", deep_get(ctx.cfg, "seeds.subset_seed", 42))) if strategy == "random" else np.nan,
            }
        )
    per = pd.DataFrame(rows)
    summ = pd.DataFrame(summary)
    full = summ[summ["strategy"] == "full"].iloc[0]
    summ["delta_PSNR_vs_Full"] = summ["PSNR"] - float(full["PSNR"])
    summ["delta_SSIM_vs_Full"] = summ["SSIM"] - float(full["SSIM"])
    summ["delta_MSE_vs_Full"] = summ["MSE"] - float(full["MSE"])
    out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "per_image_metrics.csv", index=False)
    ordered_cols = [
        "model",
        "strategy",
        "fraction",
        "variant",
        "train_size",
        "max_steps",
        "best_step",
        "stopped_step",
        "effective_epochs_at_best",
        "total_seen_samples_at_best",
        "best_val_loss",
        "PSNR",
        "SSIM",
        "MSE",
        "delta_PSNR_vs_Full",
        "delta_SSIM_vs_Full",
        "delta_MSE_vs_Full",
        "random_seed",
    ]
    summ[ordered_cols].to_csv(out / "subset_grid_results.csv", index=False)
    (out / "subset_grid_results.md").write_text(dataframe_markdown(summ[ordered_cols]) + "\n", encoding="utf-8")
    return per, summ[ordered_cols]


def stage_evaluate_grid(ctx: Context) -> None:
    if is_done(ctx, "evaluate_grid"):
        stage_skip("evaluate_grid", "completed marker found")
        return
    if ctx.dry_run:
        print("evaluate_grid: would evaluate TDnCNN and DRUNet full/topk/quantile/random p10/p20/p30")
        return
    out = grid_evaluation_root(ctx)
    # Evaluate every branch with the same test noise and clean targets.
    variants = grid_variant_names(ctx)
    tdncnn_per, tdncnn_summary = evaluate_tdncnn_variants(ctx, variants, out=out / "tdncnn", grid=True)
    drunet_per, drunet_summary = evaluate_drunet_variants(ctx, variants, out=out / "drunet", grid=True)
    per = pd.concat([tdncnn_per, drunet_per], ignore_index=True)
    summary = pd.concat([tdncnn_summary, drunet_summary], ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "per_image_metrics.csv", index=False)
    summary.to_csv(out / "subset_grid_results.csv", index=False)
    (out / "subset_grid_results.md").write_text(dataframe_markdown(summary) + "\n", encoding="utf-8")
    mark_done(ctx, "evaluate_grid", [out / "subset_grid_results.csv", out / "per_image_metrics.csv"])


def _clean_and_corpus_lookup(ctx: Context):
    clean_ds = load_clean_dataset(ctx, train=True)
    corpus = MaterializedCorpusDataset(ctx.experiment_root / "01_corpus" / "train_corpus.pt", ctx.experiment_root / "01_corpus" / "train_manifest.csv")

    def clean_for_index(original_index: int) -> torch.Tensor:
        x, _ = clean_ds[int(original_index)]
        return x.to(torch.float32).clamp(0.0, 1.0)

    def corrupt_for_index(original_index: int) -> torch.Tensor:
        return corpus.tensor_for_original_index(int(original_index)).to(torch.float32).clamp(0.0, 1.0)

    return clean_for_index, corrupt_for_index


@torch.no_grad()
def _predict_tdncnn(ctx: Context, variant: str, x: torch.Tensor) -> torch.Tensor:
    model = build_tdncnn(ctx).to(ctx.device)
    model.load_state_dict(torch.load(tdncnn_variant_checkpoint(ctx, variant), map_location=ctx.device)["model_state_dict"], strict=True)
    model.eval()
    return model(x.unsqueeze(0).to(ctx.device)).squeeze(0).cpu().clamp(0.0, 1.0)


def _metric_for(per: pd.DataFrame, variant: str, image_index: int) -> dict[str, float]:
    row = per[(per["variant"] == variant) & (per["image_index"] == int(image_index))]
    if row.empty:
        return {}
    item = row.iloc[0]
    return {"psnr": float(item["psnr"]), "ssim": float(item["ssim"]), "mse": float(item["mse"])}


@torch.no_grad()
def stage_figures(ctx: Context) -> None:
    if is_done(ctx, "figures"):
        stage_skip("figures", "completed marker found")
        return
    if ctx.dry_run:
        print("figures: would write publication plots and panels")
        return
    scores_all = pd.read_csv(ctx.experiment_root / "05_filtering" / "scores_full.csv")
    full_indices = np.load(variant_indices_path(ctx, "full"))
    scores = scores_all[scores_all["original_index"].isin(set(full_indices.tolist()))].copy()
    out = grid_figures_root(ctx)
    out.mkdir(parents=True, exist_ok=True)
    clean_for_index, corrupt_for_index = _clean_and_corpus_lookup(ctx)
    written: list[Path] = []
    for fraction in subset_grid_fractions(ctx):
        tag = fraction_tag(fraction)
        percent = subset_fraction_percent(fraction)
        subsets = {
            "topk": np.load(variant_indices_path(ctx, f"topk_{tag}")),
            "quantile": np.load(variant_indices_path(ctx, f"quantile_spread_{tag}")),
            "random": np.load(variant_indices_path(ctx, f"random_{tag}")),
        }
        quality_path = out / f"quality_shift_p{percent}_3x3.png"
        ecdf_path = out / f"ecdf_realized_psnr_p{percent}.png"
        plot_quality_shift_3x3(scores, subsets, quality_path, fraction_label=f"p{percent}")
        plot_ecdf_realized_psnr(scores, {"Top-K": subsets["topk"], "Quantile": subsets["quantile"], "Random": subsets["random"]}, ecdf_path, fraction_label=f"p{percent}")
        written.extend([quality_path, ecdf_path])
        selection_path = out / f"topk_selection_examples_p{percent}.png"
        cleanest_path = out / f"topk_cleanest_noisiest_p{percent}.png"
        save_topk_selection_examples(
            scores,
            subsets["topk"],
            selection_path,
            clean_for_index=clean_for_index,
            corrupt_for_index=corrupt_for_index,
            fraction_label=f"p{percent}",
        )
        save_topk_cleanest_noisiest(
            scores,
            subsets["topk"],
            cleanest_path,
            clean_for_index=clean_for_index,
            corrupt_for_index=corrupt_for_index,
            fraction_label=f"p{percent}",
        )
        written.extend([selection_path, cleanest_path])

    results_path = grid_evaluation_root(ctx) / "subset_grid_results.csv"
    per_path = grid_evaluation_root(ctx) / "per_image_metrics.csv"
    if results_path.exists():
        results = pd.read_csv(results_path)
        if "model" in results.columns:
            results = results[results["model"] == "TDnCNN"].copy()
        for metric, filename in [("PSNR", "subset_fraction_vs_psnr.png"), ("SSIM", "subset_fraction_vs_ssim.png"), ("MSE", "subset_fraction_vs_mse.png")]:
            metric_path = out / filename
            plot_subset_fraction_metric(results, metric, metric_path)
            written.append(metric_path)

    if per_path.exists():
        per = pd.read_csv(per_path)
        if "model" in per.columns:
            per = per[per["model"] == "TDnCNN"].copy()
        eval_input, eval_target, _ = ensure_grid_eval_noise_cache(ctx)
        inputs = torch.load(eval_input, map_location="cpu").to(torch.float32)
        targets = torch.load(eval_target, map_location="cpu").to(torch.float32)
        full_psnr = per[per["variant"] == "full"].sort_values("psnr")
        if len(full_psnr) >= 3:
            image_indices = [
                int(full_psnr.iloc[0]["image_index"]),
                int(full_psnr.iloc[len(full_psnr) // 2]["image_index"]),
                int(full_psnr.iloc[-1]["image_index"]),
            ]
        else:
            image_indices = [int(x) for x in full_psnr["image_index"].tolist()]
        for fraction in subset_grid_fractions(ctx):
            tag = fraction_tag(fraction)
            percent = subset_fraction_percent(fraction)
            variants = [("Full output", "full"), ("Top-K output", f"topk_{tag}"), ("Quantile output", f"quantile_spread_{tag}"), ("Random output", f"random_{tag}")]
            samples = []
            for name, image_index in zip(["hard", "median", "easy"], image_indices):
                tensors = {"Clean": targets[image_index], "Noisy input": inputs[image_index]}
                metrics = {}
                for col, variant in variants:
                    tensors[col] = _predict_tdncnn(ctx, variant, inputs[image_index])
                    metrics[col] = _metric_for(per, variant, image_index)
                samples.append({"name": name, "tensors": tensors, "metrics": metrics})
            panel_path = out / f"denoiser_topk_examples_p{percent}.png"
            save_denoiser_comparison_panel(samples, panel_path, columns=["Clean", "Noisy input", "Full output", "Top-K output", "Quantile output", "Random output"], title=f"TDnCNN p{percent}")
            written.append(panel_path)
        if all(tdncnn_variant_checkpoint(ctx, f"topk_{fraction_tag(f)}").exists() for f in subset_grid_fractions(ctx)):
            samples = []
            for name, image_index in zip(["hard", "median", "easy"], image_indices):
                tensors = {"Clean": targets[image_index], "Noisy": inputs[image_index], "Full": _predict_tdncnn(ctx, "full", inputs[image_index])}
                metrics = {"Full": _metric_for(per, "full", image_index)}
                for fraction in subset_grid_fractions(ctx):
                    tag = fraction_tag(fraction)
                    label = f"Top-K {subset_fraction_percent(fraction)}%"
                    variant = f"topk_{tag}"
                    tensors[label] = _predict_tdncnn(ctx, variant, inputs[image_index])
                    metrics[label] = _metric_for(per, variant, image_index)
                samples.append({"name": name, "tensors": tensors, "metrics": metrics})
            comparison_path = out / "topk_fraction_comparison.png"
            save_denoiser_comparison_panel(samples, comparison_path, columns=list(samples[0]["tensors"].keys()), title="Top-K fraction comparison")
            written.append(comparison_path)
    mark_done(ctx, "figures", written)


def _sha_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def _training_history_summary(path: Path) -> dict[str, Any]:
    history = pd.read_csv(path)
    finite = history[pd.to_numeric(history["val_loss"], errors="coerce").notna()].copy()
    finite["val_loss"] = finite["val_loss"].astype(float)
    best = finite.loc[finite["val_loss"].idxmin()]
    final = finite.iloc[-1]
    return {
        "history": history,
        "finite": finite,
        "best_step_hist": int(best["step"]),
        "best_val_loss_hist": float(best["val_loss"]),
        "final_val_loss": float(final["val_loss"]),
    }


def stage_protocol_audit(ctx: Context) -> None:
    if ctx.dry_run:
        print("audit: would inspect existing Colored MNIST artifacts and write protocol audit report")
        return
    run = ctx.experiment_root
    model_name = "tdncnn"
    variants = ["full", "topk_10", "quantile_spread_10", "random_10"]
    out = run / "00_protocol"
    tables = out / "audit_tables"
    tables.mkdir(parents=True, exist_ok=True)
    batch_size = int(deep_get(ctx.cfg, "downstream.batch_size", 128))
    metas = {v: read_json(run / "06_downstream" / model_name / v / "run_meta.json") for v in variants}
    indices = {v: np.load(variant_indices_path(ctx, v)).astype(np.int64) for v in variants}
    full_epochs = int(metas["full"]["optimizer_steps"]) * batch_size / max(1, len(indices["full"]))
    budget_rows = []
    for v in variants:
        steps = int(metas[v]["optimizer_steps"])
        seen = steps * batch_size
        epochs = seen / max(1, len(indices[v]))
        budget_rows.append(
            {
                "variant": v,
                "train_size": int(len(indices[v])),
                "batch_size": batch_size,
                "optimizer_steps": steps,
                "total_seen_samples": seen,
                "unique_train_samples": int(len(indices[v])),
                "effective_epochs": epochs,
                "relative_presentations_vs_full": epochs / full_epochs,
            }
        )
    budget = pd.DataFrame(budget_rows)
    budget.to_csv(tables / "budget.csv", index=False)

    checkpoint_rows = []
    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    for v in variants:
        history_path = run / "06_downstream" / model_name / v / "training_history.csv"
        info = _training_history_summary(history_path)
        finite = info["finite"]
        ax.plot(finite["step"], finite["val_loss"], marker="o", linewidth=1.4, markersize=3, label=v)
        best_payload = torch.load(run / "06_downstream" / model_name / v / "best.pt", map_location="cpu")
        last_payload = torch.load(run / "06_downstream" / model_name / v / "last.pt", map_location="cpu")
        checkpoint_rows.append(
            {
                "variant": v,
                "evaluated_checkpoint": "best.pt",
                "best_step_meta": int(metas[v].get("best_step", 0)),
                "best_step_hist": int(info["best_step_hist"]),
                "stopped_step": int(metas[v].get("optimizer_steps", 0)),
                "best_val_loss_meta": float(metas[v].get("best_val_loss", float("nan"))),
                "best_val_loss_hist": float(info["best_val_loss_hist"]),
                "final_val_loss": float(info["final_val_loss"]),
                "stop_reason": metas[v].get("stop_reason"),
                "best_pt_step": int(best_payload.get("step", -1)),
                "last_pt_step": int(last_payload.get("step", -1)),
            }
        )
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("validation loss")
    ax.set_title("Colored MNIST TDnCNN validation loss audit")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    curve_path = out / "audit_training_curves.png"
    fig.savefig(curve_path, dpi=180)
    plt.close(fig)
    checkpoints = pd.DataFrame(checkpoint_rows)
    checkpoints.to_csv(tables / "checkpoints.csv", index=False)

    scores = pd.read_csv(run / "05_filtering" / "scores_full.csv")
    quality_rows = []
    for v in variants:
        sub = scores[scores["original_index"].isin(set(indices[v].tolist()))]
        quality_rows.append(
            {
                "variant": v,
                "target_psnr_mean": float(sub["target_psnr_db"].mean()),
                "target_psnr_median": float(sub["target_psnr_db"].median()),
                "realized_psnr_mean": float(sub["realized_psnr_db_postclip"].mean()),
                "realized_psnr_median": float(sub["realized_psnr_db_postclip"].median()),
                "realized_snr_mean": float(sub["realized_snr_db_postclip"].mean()),
                "realized_snr_median": float(sub["realized_snr_db_postclip"].median()),
                "sigma_mean": float(sub["sigma"].mean()),
                "sigma_median": float(sub["sigma"].median()),
            }
        )
    quality = pd.DataFrame(quality_rows)
    quality.to_csv(tables / "quality_stats.csv", index=False)

    per = pd.read_csv(run / "07_internal_evaluation" / model_name / "per_image_metrics.csv")
    wide = per.pivot(index="image_index", columns="variant", values=["psnr", "mse", "ssim"])
    rng = np.random.default_rng(int(deep_get(ctx.cfg, "seeds.bootstrap_seed", 123)))
    paired_rows = []
    for v in variants[1:]:
        for metric in ["psnr", "mse", "ssim"]:
            delta = (wide[(metric, v)] - wide[(metric, "full")]).dropna().to_numpy(dtype=np.float64)
            means = [float(np.mean(rng.choice(delta, size=len(delta), replace=True))) for _ in range(1000)]
            better = delta < 0 if metric == "mse" else delta > 0
            paired_rows.append(
                {
                    "comparison": f"{v}-full",
                    "metric": metric,
                    "mean_delta": float(np.mean(delta)),
                    "median_delta": float(np.median(delta)),
                    "bootstrap_mean_ci_low": float(np.percentile(means, 2.5)),
                    "bootstrap_mean_ci_high": float(np.percentile(means, 97.5)),
                    "fraction_subset_better_than_full": float(np.mean(better)),
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(tables / "paired_bootstrap.csv", index=False)

    validation = np.load(run / "00_protocol" / "downstream_validation_indices.npy")
    eval_manifest = run / "07_internal_evaluation" / "eval_noise_cache" / "test_noise_manifest.csv"
    hashes = {
        "resolved_config": sha256_file(run / "00_protocol" / "resolved_config.yaml"),
        "corpus_manifest": sha256_file(run / "01_corpus" / "train_manifest.csv"),
        "eval_input": sha256_file(run / "07_internal_evaluation" / "eval_noise_cache" / "test_eval_input.pt"),
        "eval_target": sha256_file(run / "07_internal_evaluation" / "eval_noise_cache" / "test_eval_target.pt"),
        "eval_manifest": sha256_file(eval_manifest),
        "validation_indices_values": _sha_array(validation),
        "initial_state": sha256_file(run / "06_downstream" / model_name / "initial_state.pt"),
    }
    write_json(tables / "hashes.json", hashes)

    lines = [
        "# Colored MNIST Protocol Audit",
        "",
        f"Run: `{run}`",
        f"Model audited: `{model_name}`",
        "",
        "## Training Budget",
        "",
        dataframe_markdown(budget),
        "",
        f"Protocol used: equal optimizer steps. A 10% subset sees each unique sample {float(budget[budget.variant == 'topk_10']['relative_presentations_vs_full'].iloc[0]):.1f}x as often as Full.",
        "",
        "## Checkpoint Selection",
        "",
        dataframe_markdown(checkpoints),
        "",
        "## Quality Statistics",
        "",
        dataframe_markdown(quality),
        "",
        "## Paired Comparison",
        "",
        dataframe_markdown(paired),
        "",
        "## Hashes",
        "",
        *[f"- {k}: `{v}`" for k, v in hashes.items()],
        "",
    ]
    report_path = out / "colored_mnist_protocol_audit.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"audit: wrote {report_path}")


@torch.no_grad()
def stage_tdncnn_evaluate(ctx: Context) -> None:
    if is_done(ctx, "tdncnn_evaluate"):
        print("tdncnn_evaluate: resume skip")
        return
    if ctx.dry_run:
        print("tdncnn_evaluate: would evaluate TDnCNN variants")
        return
    paths = ensure_eval_noise_cache(ctx)
    eval_ds = EvalNoiseDataset(paths[0], paths[1])
    loader = DataLoader(eval_ds, batch_size=int(deep_get(ctx.cfg, "evaluation.batch_size", 256)), shuffle=False)
    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for variant in VARIANTS:
        ckpt = ctx.experiment_root / "06_downstream" / "tdncnn" / variant / "best.pt"
        model = build_tdncnn(ctx).to(ctx.device)
        model.load_state_dict(torch.load(ckpt, map_location=ctx.device)["model_state_dict"], strict=True)
        model.eval()
        idx = 0
        values = []
        for x_in, x_target in loader:
            x_in = x_in.to(ctx.device)
            pred = model(x_in).cpu()
            target = x_target.cpu()
            for p, t in zip(pred, target):
                mse = float(((p - t) ** 2).mean().item())
                row = {"variant": variant, "image_index": idx, "mse": mse, "psnr": psnr_from_mse(mse), "ssim": ssim_image(p, t), "lpips": float("nan")}
                rows.append(row)
                values.append(row)
                idx += 1
        frame = pd.DataFrame(values)
        train_meta = read_yaml(selected_score_root(ctx) / "selected_score_config.yaml")
        summary.append(
            {
                "variant": variant,
                "train_size": int(len(np.load(ctx.experiment_root / "05_filtering" / "subset_indices" / VARIANTS[variant]))),
                "optimizer_steps": int(deep_get(ctx.cfg, "smoke.downstream_max_optimizer_steps" if ctx.smoke else "downstream.max_optimizer_steps", 2000)),
                "psnr": float(frame["psnr"].mean()),
                "mse": float(frame["mse"].mean()),
                "ssim": float(frame["ssim"].mean()),
                "lpips": float("nan"),
                "selected_checkpoint_epoch": int(train_meta["checkpoint_epoch"]),
            }
        )
    out = evaluation_root(ctx) / "tdncnn"
    out.mkdir(parents=True, exist_ok=True)
    per = pd.DataFrame(rows)
    summ = pd.DataFrame(summary)
    full_psnr = float(summ[summ["variant"] == "full"]["psnr"].iloc[0])
    summ["delta_psnr_vs_full"] = summ["psnr"] - full_psnr
    per.to_csv(out / "per_image_metrics.csv", index=False)
    summ.to_csv(out / "summary_metrics.csv", index=False)
    mark_done(ctx, "tdncnn_evaluate", [out / "summary_metrics.csv"])


def ensure_eval_noise_cache(ctx: Context) -> tuple[Path, Path, Path]:
    test_ds = load_clean_dataset(ctx, train=False)
    eval_indices = np.load(ctx.experiment_root / "00_protocol" / "evaluation_indices.npy")
    return materialize_eval_noise_cache(
        dataset=test_ds,
        dataset_name=dataset_key(ctx),
        original_indices=eval_indices,
        output_dir=evaluation_root(ctx) / "eval_noise_cache",
        sigma_min=float(deep_get(ctx.cfg, "evaluation.eval_noise.sigma_min", 0.1)),
        sigma_max=float(deep_get(ctx.cfg, "evaluation.eval_noise.sigma_max", 0.8)),
        eval_seed=int(deep_get(ctx.cfg, "seeds.eval_noise_seed", 4242)),
        eval_noise_version=str(deep_get(ctx.cfg, "evaluation.eval_noise.version", "mnist_eval_awgn_v1")),
        overwrite=False,
    )


@torch.no_grad()
def stage_drunet_evaluate(ctx: Context) -> None:
    if is_done(ctx, "drunet_evaluate"):
        print("drunet_evaluate: resume skip")
        return
    if ctx.dry_run:
        print("drunet_evaluate: would evaluate DRUNet variants")
        return
    paths = ensure_eval_noise_cache(ctx)
    eval_ds = EvalNoiseWithSigmaDataset(paths[0], paths[1], paths[2])
    loader = DataLoader(eval_ds, batch_size=int(deep_get(ctx.cfg, "evaluation.batch_size", 256)), shuffle=False)
    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for variant in VARIANTS:
        ckpt = drunet_out_dir(ctx, variant) / "best.pt"
        model = build_current_drunet(ctx).to(ctx.device)
        payload = torch.load(ckpt, map_location=ctx.device)
        model.load_state_dict(payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload, strict=True)
        model.eval()
        idx = 0
        values = []
        for x_in, x_target, sigma in loader:
            x_in = x_in.to(ctx.device)
            sigma = sigma.to(ctx.device)
            pred = model(x_in, sigma=sigma).cpu()
            target = x_target.cpu()
            for p, t in zip(pred, target):
                mse = float(((p - t) ** 2).mean().item())
                row = {"variant": variant, "image_index": idx, "mse": mse, "psnr": psnr_from_mse(mse), "ssim": ssim_image(p, t), "lpips": float("nan")}
                rows.append(row)
                values.append(row)
                idx += 1
        frame = pd.DataFrame(values)
        summary.append(
            {
                "variant": variant,
                "train_size": int(len(np.load(ctx.experiment_root / "05_filtering" / "subset_indices" / VARIANTS[variant]))),
                "optimizer_steps": int(deep_get(ctx.cfg, "smoke.drunet_max_optimizer_steps" if ctx.smoke else "drunet.max_steps", 2000)),
                "psnr": float(frame["psnr"].mean()),
                "mse": float(frame["mse"].mean()),
                "ssim": float(frame["ssim"].mean()),
                "lpips": float("nan"),
            }
        )
    out = evaluation_root(ctx) / "drunet"
    out.mkdir(parents=True, exist_ok=True)
    per = pd.DataFrame(rows)
    summ = pd.DataFrame(summary)
    full_psnr = float(summ[summ["variant"] == "full"]["psnr"].iloc[0])
    summ["delta_psnr_vs_full"] = summ["psnr"] - full_psnr
    per.to_csv(out / "per_image_metrics.csv", index=False)
    summ.to_csv(out / "summary_metrics.csv", index=False)
    mark_done(ctx, "drunet_evaluate", [out / "summary_metrics.csv"])


def stage_external_evaluate(ctx: Context) -> None:
    if is_done(ctx, "external_evaluate"):
        print("external_evaluate: resume skip")
        return
    if ctx.dry_run:
        print("external_evaluate: would run TDnCNN and DRUNet on external benchmarks")
        return
    if ctx.smoke:
        out = ctx.experiment_root / "09_external_evaluation"
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "SKIPPED_SMOKE.json", {"reason": "External benchmark is skipped in synthetic smoke"})
        mark_done(ctx, "external_evaluate", [out / "SKIPPED_SMOKE.json"])
        print("external_evaluate: skipped in synthetic smoke")
        return
    if dataset_channels(ctx) != 3:
        out = ctx.experiment_root / "09_external_evaluation"
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "SKIPPED.json", {"reason": "External benchmark evaluation is RGB-only", "channels": dataset_channels(ctx)})
        mark_done(ctx, "external_evaluate", [out / "SKIPPED.json"])
        print("external_evaluate: skipped because dataset is not RGB")
        return
    from scripts import evaluate_external

    external_root = Path(str(deep_get(ctx.cfg, "external.root", "data/external_benchmarks")))
    if not external_root.is_absolute():
        external_root = PROJECT_ROOT / external_root
    if not external_root.exists():
        raise FileNotFoundError(f"Missing external benchmark root: {external_root}")
    sigma = str(int(deep_get(ctx.cfg, "external.sigma", 25)))
    seed = str(int(deep_get(ctx.cfg, "external.seed", 42)))
    max_images = deep_get(ctx.cfg, "external.max_images", None)
    jobs = [
        (
            "tdncnn",
            ctx.experiment_root / "09_external_evaluation" / "tdncnn",
            {
                "full": ctx.experiment_root / "06_downstream" / "tdncnn" / "full" / "best.pt",
                "topk": ctx.experiment_root / "06_downstream" / "tdncnn" / "topk_10" / "best.pt",
                "quantile": ctx.experiment_root / "06_downstream" / "tdncnn" / "quantile_spread_10" / "best.pt",
            },
        ),
        (
            "drunet",
            ctx.experiment_root / "09_external_evaluation" / "drunet",
            {
                "full": drunet_out_dir(ctx, "full") / "best.pt",
                "topk": drunet_out_dir(ctx, "topk_10") / "best.pt",
                "quantile": drunet_out_dir(ctx, "quantile_spread_10") / "best.pt",
            },
        ),
    ]
    written: list[Path] = []
    for model_type, output_root, checkpoints in jobs:
        missing = [str(path) for path in checkpoints.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Cannot run external evaluation for {model_type}; missing checkpoints: {missing}")
        argv = [
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
            "--device",
            str(ctx.device),
            "--full-checkpoint",
            str(checkpoints["full"]),
            "--topk-checkpoint",
            str(checkpoints["topk"]),
            "--quantile-checkpoint",
            str(checkpoints["quantile"]),
        ]
        if bool(deep_get(ctx.cfg, "external.lpips", False)):
            argv.append("--lpips")
        if max_images is not None:
            argv.extend(["--max-images", str(int(max_images))])
        old_argv = sys.argv
        try:
            sys.argv = argv
            evaluate_external.main()
        finally:
            sys.argv = old_argv
        written.append(output_root / "summary_metrics.csv")
    mark_done(ctx, "external_evaluate", written)


def stage_summary(ctx: Context) -> None:
    if is_done(ctx, "summary"):
        print("summary: resume skip")
        return
    if ctx.dry_run:
        print("summary: would write final summary")
        return
    out = ctx.experiment_root / "08_summary"
    out.mkdir(parents=True, exist_ok=True)
    selected = read_yaml(selected_score_root(ctx) / "selected_score_config.yaml")
    grid_metrics_path = grid_evaluation_root(ctx) / "subset_grid_results.csv"
    if grid_metrics_path.exists():
        grid_metrics = pd.read_csv(grid_metrics_path)
        tdncnn_metrics = grid_metrics[grid_metrics["model"] == "TDnCNN"].copy() if "model" in grid_metrics.columns else grid_metrics
        drunet_metrics = grid_metrics[grid_metrics["model"] == "DRUNet"].copy() if "model" in grid_metrics.columns else pd.DataFrame()
    else:
        grid_metrics = pd.DataFrame()
        tdncnn_metrics = pd.read_csv(evaluation_root(ctx) / "tdncnn" / "summary_metrics.csv")
        drunet_path = evaluation_root(ctx) / "drunet" / "summary_metrics.csv"
        drunet_metrics = pd.read_csv(drunet_path) if drunet_path.exists() else pd.DataFrame()
    lines = [
        f"# {dataset_title(ctx)} Image-Space Variable-AWGN Summary",
        "",
        f"run_id: `{ctx.run_id}`",
        f"dataset: {dataset_title(ctx)}, image `[B,{dataset_channels(ctx)},28,28]`",
        f"selected checkpoint epoch: `{selected['checkpoint_epoch']}`",
        f"selected report timestep: `{selected['report_timestep']}`",
        f"MC K: `{selected['mc_k']}`",
        "",
    ]
    if not grid_metrics.empty and "model" in grid_metrics.columns:
        lines.extend(["## Downstream Grid", "", dataframe_markdown(grid_metrics), ""])
    else:
        lines.extend(["## TDnCNN", "", dataframe_markdown(tdncnn_metrics), ""])
    if (grid_metrics.empty or "model" not in grid_metrics.columns) and not drunet_metrics.empty:
        lines.extend(["## DRUNet", "", dataframe_markdown(drunet_metrics), ""])
    external_root = ctx.experiment_root / "09_external_evaluation"
    external_outputs = sorted(external_root.glob("*/summary_metrics.csv")) if external_root.exists() else []
    if external_outputs:
        lines.extend(["## External Benchmarks", ""])
        for path in external_outputs:
            lines.append(f"- `{path.parent.name}`: `{path}`")
        lines.append("")
    elif (external_root / "SKIPPED.json").exists():
        lines.extend(["## External Benchmarks", "", "Skipped for non-RGB dataset.", ""])
    (out / "final_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        out / "summary.json",
        {
            "run_id": ctx.run_id,
            "selected": selected,
            "grid_rows": grid_metrics.to_dict(orient="records") if not grid_metrics.empty else [],
            "tdncnn_rows": tdncnn_metrics.to_dict(orient="records"),
            "drunet_rows": drunet_metrics.to_dict(orient="records"),
            "external_summary_paths": [str(path) for path in external_outputs],
        },
    )
    mark_done(ctx, "summary", [out / "final_summary.md"])


def run_stage(ctx: Context, stage: str) -> None:
    if stage == "audit":
        stage_protocol_audit(ctx)
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
    elif stage in {"tdncnn_full", "tdncnn_topk", "tdncnn_quantile", "tdncnn_random", "downstream_full", "downstream_topk_p10", "downstream_topk_p20", "downstream_topk_p30", "downstream_quantile_p10", "downstream_quantile_p20", "downstream_quantile_p30", "downstream_random_p10", "downstream_random_p20", "downstream_random_p30"}:
        stage_tdncnn_train(ctx, stage)
    elif stage == "tdncnn_evaluate":
        stage_tdncnn_evaluate(ctx)
    elif stage == "evaluate_grid":
        stage_evaluate_grid(ctx)
    elif stage == "figures":
        stage_figures(ctx)
    elif stage in {"drunet_full", "drunet_topk", "drunet_quantile", "drunet_random", "drunet_topk_p20", "drunet_topk_p30", "drunet_quantile_p20", "drunet_quantile_p30", "drunet_random_p20", "drunet_random_p30"}:
        stage_drunet_train(ctx, stage)
    elif stage == "drunet_evaluate":
        stage_drunet_evaluate(ctx)
    elif stage == "external_evaluate":
        stage_external_evaluate(ctx)
    elif stage == "summary":
        stage_summary(ctx)
    else:
        raise ValueError(f"Unknown stage: {stage}")


def run_smoke_checks(ctx: Context) -> None:
    if ctx.dry_run:
        print("smoke_checks: dry-run")
        return
    tmp = ctx.experiment_root
    stages = [
        "prepare",
        "ddpm_train",
        "score_validate",
        "score_select",
        "filter_grid",
        *STAGE_ALIASES["downstream_grid"],
        "evaluate_grid",
        "figures",
        "drunet_full",
        "drunet_topk",
        "drunet_quantile",
        "drunet_random",
        "drunet_evaluate",
        "external_evaluate",
        "summary",
    ]
    for st in stages:
        run_stage(ctx, st)
    sample = MaterializedCorpusDataset(tmp / "01_corpus" / "train_corpus.pt", tmp / "01_corpus" / "train_manifest.csv")
    x = sample[0].unsqueeze(0)
    model = build_ddpm(ctx)
    t = torch.zeros((1,), dtype=torch.long, device=ctx.device)
    y = model.model(x.to(ctx.device), t)
    assert tuple(y.shape) == tuple(x.shape)
    scores, _ = score_batch(model, x, 1, [1])
    assert scores.shape == (1,) and np.isfinite(scores).all()
    idx_root = tmp / "05_filtering" / "subset_indices"
    full = np.load(idx_root / "full_indices.npy")
    for fraction in subset_grid_fractions(ctx):
        tag = fraction_tag(fraction)
        expected = subset_size_for_fraction(len(full), fraction)
        topk = np.load(idx_root / f"topk_{tag}_indices.npy")
        quantile = np.load(idx_root / f"quantile_spread_{tag}_indices.npy")
        random = np.load(idx_root / f"random_{tag}_indices.npy")
        assert len(random) == len(topk) == len(quantile) == expected
        assert len(np.unique(random)) == len(random)
        assert set(random.tolist()).issubset(set(full.tolist()))
    score_df = pd.read_csv(tmp / "05_filtering" / "scores_full.csv")
    validation_indices = np.load(tmp / "00_protocol" / "downstream_validation_indices.npy").astype(np.int64)
    score_df = score_df[~score_df["original_index"].isin(set(validation_indices.tolist()))].copy()
    seed = int(deep_get(ctx.cfg, "filtering.random_seed", deep_get(ctx.cfg, "seeds.subset_seed", 42)))
    random = np.load(idx_root / "random_p10_indices.npy")
    same_random = select_random_exact(score_df, len(random), seed=seed)
    different_random = select_random_exact(score_df, len(random), seed=seed + 1)
    np.testing.assert_array_equal(random, same_random)
    assert not np.array_equal(random, different_random)
    random_meta = json.loads((tmp / "05_filtering" / "subset_metadata" / "random_10.json").read_text(encoding="utf-8"))
    assert random_meta["method"] == "random"
    assert random_meta["selection_independent_of_score"] is True
    import sys as _sys

    forbidden = ["src.DDPM_model", "src.autoencoder", "src.autoencoder_noise_consistency"]
    loaded = [name for name in forbidden if name in _sys.modules]
    if loaded:
        raise RuntimeError(f"Forbidden legacy modules loaded: {loaded}")
    fail_ctx = Context(**{**ctx.__dict__, "run_id": ctx.run_id + "_fail_gate", "experiment_root": ctx.experiment_root.parent / (ctx.experiment_root.name + "_fail_gate"), "force_fail_gate": True, "resume": False})
    for st in ["prepare", "ddpm_train", "score_validate"]:
        run_stage(fail_ctx, st)
    failed = False
    try:
        run_stage(fail_ctx, "score_select")
    except SystemExit:
        failed = True
    if not failed:
        raise RuntimeError("Synthetic FAIL gate did not stop score_select")
    print("smoke_checks: passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Colored MNIST research pipeline.")
    parser.add_argument("--config", default="configs/experiments/colored_mnist_seed42.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--stage", default="full_grid", choices=["all", "smoke_checks", *STAGES, *GRID_STAGE_NAMES, *STAGE_ALIASES.keys()])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Use synthetic tiny MNIST-like data and smoke budgets.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--downstream-budget-mode", choices=["main", "equal_epochs", "diagnostic_equal_epochs"], default=None)
    parser.add_argument("--force-fail-gate", action="store_true", help="Testing hook: force score_select to fail.")
    return parser.parse_args()


def expand_stage_selection(stage: str) -> list[str]:
    if stage == "all":
        return STAGES
    expanded: list[str] = []
    for item in STAGE_ALIASES.get(stage, [stage]):
        expanded.extend(expand_stage_selection(item) if item in STAGE_ALIASES else [item])
    return expanded


def main() -> None:
    args = parse_args()
    ctx = resolve_context(args)
    if ctx.experiment_root.exists() and not ctx.resume and not ctx.dry_run:
        raise FileExistsError('Output root exists; choose a new root or explicitly --resume an owned run.')
    domain = dataset_title(ctx)
    validate_current_protocol(domain, ctx.cfg, downstream_models=["tdncnn", "drunet"])
    print_protocol_summary(domain, ctx.cfg, run_id=ctx.run_id, device=str(ctx.device), downstream_models=["tdncnn", "drunet"])
    if args.stage == "smoke_checks":
        run_smoke_checks(ctx)
        return
    stages = expand_stage_selection(args.stage)
    for stage in stages:
        with stage_logging(domain, stage, run_id=ctx.run_id):
            run_stage(ctx, stage)


if __name__ == "__main__":
    main()
