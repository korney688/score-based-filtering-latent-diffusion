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
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:  # pragma: no cover - optional dependency
    skimage_ssim = None

from src.ddpm.image_space import ImageSpaceDDPM, build_image_space_ddpm_model
from src.evaluation.image_lab_replication_analysis import (
    select_quantile_spread_exact,
    select_random_exact,
    select_topk_exact,
)
from src.image_lab_replication_datasets import (
    CorpusTaskNoiseDataset,
    EvalNoiseDataset,
    MaterializedCorpusDataset,
    materialize_eval_noise_cache,
    materialize_variable_awgn_corpus,
    stable_int_seed,
)
from src.utils.current_protocol import (
    print_protocol_summary,
    stage_logging,
    stage_skip,
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
    "summary",
]
VARIANTS = {
    "full": "full_indices.npy",
    "topk_10": "topk_10_indices.npy",
    "quantile_spread_10": "quantile_spread_10_indices.npy",
    "random_10": "random_10_indices.npy",
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


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve_context(args: argparse.Namespace) -> Context:
    cfg = read_yaml(Path(args.config))
    smoke = bool(args.smoke)
    run_id = args.run_id or deep_get(cfg, "experiment.run_id", "mnist_image_space_awgn_v1")
    if smoke and not run_id.endswith("_synthetic_smoke"):
        run_id = f"{run_id}_synthetic_smoke"
    root = Path(args.experiment_root) if args.experiment_root else PROJECT_ROOT / deep_get(cfg, "experiment.root") / run_id
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
    )


def stage_root(ctx: Context, stage: str) -> Path:
    mapping = {
        "prepare": "01_corpus",
        "ddpm_train": "02_ddpm",
        "score_validate": "03_score_validation",
        "score_select": "04_selected_configuration",
        "filter": "05_filtering",
        "summary": "08_summary",
    }
    if stage.startswith("tdncnn_"):
        return ctx.experiment_root / "06_downstream"
    return ctx.experiment_root / mapping[stage]


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
    if ctx.smoke and bool(deep_get(ctx.cfg, "smoke.synthetic", True)):
        key = "smoke.train_samples_per_class" if train else "smoke.test_samples_per_class"
        return SyntheticMNISTDataset(int(deep_get(ctx.cfg, key, 4)), image_size=image_size, seed=int(deep_get(ctx.cfg, "seeds.split_seed", 42)))
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
    root = ctx.experiment_root
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
    train_indices = np.arange(len(train_ds), dtype=np.int64)
    test_indices = np.arange(len(test_ds), dtype=np.int64)
    tensor_path, manifest_path = materialize_variable_awgn_corpus(
        dataset=train_ds,
        dataset_name="mnist",
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
            "dataset": "mnist",
            "emnist_present": False,
            "train_samples": int(len(train_ds)),
            "test_samples": int(len(test_ds)),
            "channels": 1,
            "image_size": 28,
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
    return build_image_space_ddpm_model(
        base_dim=base_dim,
        deep=depth,
        device=ctx.device,
        in_channels=1,
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
                    "channels": 1,
                    "config_hash": sha256_file(ctx.config_path),
                },
                checkpoint_path(ctx, epoch),
            )
        if max_steps is not None and global_step >= max_steps:
            break
    pd.DataFrame(rows).to_csv(ctx.experiment_root / "02_ddpm" / "training_history.csv", index=False)
    mark_done(ctx, "ddpm_train", [ctx.experiment_root / "02_ddpm" / "training_history.csv"])


@torch.no_grad()
def score_batch(model: ImageSpaceDDPM, x0: torch.Tensor, report_timestep: int, seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    x0 = x0.to(model.device)
    t = torch.full((x0.shape[0],), int(report_timestep) - 1, dtype=torch.long, device=model.device)
    vals: list[torch.Tensor] = []
    for seed in seeds:
        gen = torch.Generator(device=model.device).manual_seed(int(seed))
        noise = torch.randn(x0.shape, generator=gen, device=model.device, dtype=x0.dtype)
        xt = model.q_sample(x0, t, noise)
        pred = model.model(xt, t)
        vals.append(pred.flatten(1).pow(2).sum(dim=1).detach().cpu())
    stacked = torch.stack(vals, dim=0)
    return stacked.mean(dim=0).numpy(), stacked.std(dim=0, unbiased=False).numpy()


def spearman(a: Iterable[float], b: Iterable[float]) -> float:
    x = pd.Series(list(a)).rank(method="average").to_numpy(dtype=np.float64)
    y = pd.Series(list(b)).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(x) < EPS or np.std(y) < EPS:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def mc_seeds(ctx: Context, k: int, *parts: object) -> list[int]:
    return [stable_int_seed("mnist_forward", ctx.run_id, *parts, i, int(deep_get(ctx.cfg, "seeds.forward_noise_seed", 777))) for i in range(int(k))]


def load_ddpm_checkpoint(ctx: Context, epoch: int) -> ImageSpaceDDPM:
    model = build_ddpm(ctx)
    ckpt = torch.load(checkpoint_path(ctx, epoch), map_location=ctx.device)
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
        base_seed = stable_int_seed("mnist_ladder", int(original_index), int(deep_get(ctx.cfg, "seeds.corpus_seed", 42)))
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
    out = ctx.experiment_root / "04_selected_configuration"
    out.mkdir(parents=True, exist_ok=True)
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
        "dataset": "mnist",
        "run_id": ctx.run_id,
        "corpus_manifest_path": str(ctx.experiment_root / "01_corpus" / "train_manifest.csv"),
        "corpus_manifest_hash": sha256_file(ctx.experiment_root / "01_corpus" / "train_manifest.csv"),
        "checkpoint_epoch": epoch,
        "checkpoint_path": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
        "ddpm_prediction_type": "epsilon",
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
    selected_path = ctx.experiment_root / "04_selected_configuration" / "selected_score_config.yaml"
    pass_marker = ctx.experiment_root / "04_selected_configuration" / "PASS.marker"
    if not selected_path.exists() or not pass_marker.exists() or not marker_path(ctx, "score_select").exists():
        raise SystemExit("Filtering/downstream blocked: missing successful score_select PASS artifacts")
    return read_yaml(selected_path)


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
    scores.to_csv(out / "scores_full.csv", index=False)
    ratio = float(deep_get(ctx.cfg, "filtering.ratio", 0.10))
    subset_seed = int(deep_get(ctx.cfg, "filtering.random_seed", deep_get(ctx.cfg, "seeds.subset_seed", 42)))
    validation_path = ctx.experiment_root / "00_protocol" / "downstream_validation_indices.npy"
    downstream_validation = np.load(validation_path).astype(np.int64) if validation_path.exists() else np.asarray([], dtype=np.int64)
    eligible_scores = scores[~scores["original_index"].isin(set(downstream_validation.tolist()))].copy()
    n_subset = max(1, min(int(round(ratio * len(eligible_scores))), len(eligible_scores)))
    subsets = {
        "full": np.sort(eligible_scores["original_index"].to_numpy(dtype=np.int64)),
        "topk_10": select_topk_exact(eligible_scores, n_subset),
        "quantile_spread_10": select_quantile_spread_exact(eligible_scores, n_subset, seed=subset_seed, min_points_per_bin=int(deep_get(ctx.cfg, "filtering.quantile_min_points_per_bin", 30))),
        "random_10": select_random_exact(eligible_scores, n_subset, seed=subset_seed),
    }
    idx_dir = out / "subset_indices"
    idx_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = out / "subset_metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    source_pool = np.sort(eligible_scores["original_index"].to_numpy(dtype=np.int64))
    for name, arr in subsets.items():
        np.save(idx_dir / VARIANTS[name], np.sort(arr.astype(np.int64)))
        sub = scores[scores["original_index"].isin(set(arr.tolist()))]
        method = {"full": "full", "topk_10": "topk", "quantile_spread_10": "quantile_spread", "random_10": "random"}[name]
        metadata = {
            "subset": name,
            "method": method,
            "fraction": 1.0 if name == "full" else ratio,
            "K": int(len(arr)),
            "source_pool_size": int(len(source_pool)),
            "excluded_downstream_validation_count": int(len(downstream_validation)),
            "seed": None if name == "full" else subset_seed,
            "without_replacement": True,
            "selection_independent_of_score": bool(name == "random_10"),
            "score_used_for_selection": bool(name in {"topk_10", "quantile_spread_10"}),
            "class_coverage": float(sub["class_id"].nunique() / max(1, scores["class_id"].nunique())),
            "represented_classes": int(sub["class_id"].nunique()),
            "indices_sha256": sha256_file(idx_dir / VARIANTS[name]),
        }
        write_json(meta_dir / f"{name}.json", metadata)
    summary = []
    for name, arr in subsets.items():
        sub = scores[scores["original_index"].isin(set(arr.tolist()))]
        summary.append(
            {
                "subset": name,
                "n": int(len(arr)),
                "classes": int(sub["class_id"].nunique()),
                "class_coverage": float(sub["class_id"].nunique() / max(1, scores["class_id"].nunique())),
                "median_realized_psnr": float(sub["realized_psnr_db_postclip"].median()),
            }
        )
    pd.DataFrame(summary).to_csv(out / "selection_summary.csv", index=False)
    mark_done(ctx, "filter", [out / "scores_full.csv", idx_dir])


def ensure_initial_tdncnn(ctx: Context) -> Path:
    path = ctx.experiment_root / "06_downstream" / "tdncnn" / "initial_state.pt"
    if path.exists():
        return path
    torch.manual_seed(int(deep_get(ctx.cfg, "seeds.model_seed", 42)))
    model = TinyTDnCNN(in_channels=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def variant_from_stage(stage: str) -> str:
    return {
        "tdncnn_full": "full",
        "tdncnn_topk": "topk_10",
        "tdncnn_quantile": "quantile_spread_10",
        "tdncnn_random": "random_10",
    }[stage]


def stage_tdncnn_train(ctx: Context, stage: str) -> None:
    if is_done(ctx, stage):
        stage_skip(stage, "completed marker found")
        return
    if ctx.dry_run:
        print(f"{stage}: would train TDnCNN variant")
        return
    require_pass(ctx)
    variant = variant_from_stage(stage)
    indices = np.load(ctx.experiment_root / "05_filtering" / "subset_indices" / VARIANTS[variant])
    corpus = MaterializedCorpusDataset(
        ctx.experiment_root / "01_corpus" / "train_corpus.pt",
        ctx.experiment_root / "01_corpus" / "train_manifest.csv",
        original_indices=indices,
        return_original_index=True,
    )
    train_ds = CorpusTaskNoiseDataset(
        corpus,
        dataset_name="mnist",
        task_seed=int(deep_get(ctx.cfg, "seeds.task_noise_seed", 4242)),
        task_noise_version=str(deep_get(ctx.cfg, "downstream.task_noise.version", "mnist_task_awgn_v1")),
        sigma_min=float(deep_get(ctx.cfg, "downstream.task_noise.sigma_min", 0.1)),
        sigma_max=float(deep_get(ctx.cfg, "downstream.task_noise.sigma_max", 0.8)),
    )
    batch_size = int(deep_get(ctx.cfg, "smoke.downstream_batch_size" if ctx.smoke else "downstream.batch_size", 128))
    max_steps = int(deep_get(ctx.cfg, "smoke.downstream_max_optimizer_steps" if ctx.smoke else "downstream.max_optimizer_steps", 2000))
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
        dataset_name="mnist",
        task_seed=int(deep_get(ctx.cfg, "seeds.task_noise_seed", 4242)) + 100003,
        task_noise_version=str(deep_get(ctx.cfg, "downstream.task_noise.version", "mnist_task_awgn_v1")) + "_validation",
        sigma_min=float(deep_get(ctx.cfg, "downstream.task_noise.sigma_min", 0.1)),
        sigma_max=float(deep_get(ctx.cfg, "downstream.task_noise.sigma_max", 0.8)),
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model = TinyTDnCNN(in_channels=1).to(ctx.device)
    model.load_state_dict(torch.load(ensure_initial_tdncnn(ctx), map_location=ctx.device), strict=True)
    opt = torch.optim.Adam(model.parameters(), lr=float(deep_get(ctx.cfg, "downstream.lr", 1e-3)))
    rows: list[dict[str, Any]] = []
    step = 0
    best = math.inf
    best_step = 0
    checks_without_improvement = 0
    stop_reason = "max_steps"
    out = ctx.experiment_root / "06_downstream" / "tdncnn" / variant
    out.mkdir(parents=True, exist_ok=True)
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
                if step >= warmup_steps and val_loss < best - min_delta:
                    best = val_loss
                    best_step = step
                    checks_without_improvement = 0
                    torch.save({"model_state_dict": model.state_dict(), "step": step, "variant": variant, "best_val_loss": best}, out / "best.pt")
                elif step >= warmup_steps:
                    checks_without_improvement += 1
                    if checks_without_improvement >= patience:
                        stop_reason = "early_stopping"
            rows.append(row)
            if step >= max_steps:
                break
            if stop_reason == "early_stopping":
                break
        if stop_reason == "early_stopping":
            break
    if not (out / "best.pt").exists():
        torch.save({"model_state_dict": model.state_dict(), "step": step, "variant": variant, "best_val_loss": best}, out / "best.pt")
    torch.save({"model_state_dict": model.state_dict(), "step": step, "variant": variant}, out / "last.pt")
    pd.DataFrame(rows).to_csv(out / "training_history.csv", index=False)
    write_json(
        out / "run_meta.json",
        {
            "variant": variant,
            "train_size": int(len(indices)),
            "optimizer_steps": int(step),
            "initial_state": str(ensure_initial_tdncnn(ctx)),
            "early_stopping_enabled": True,
            "early_stopping_monitor": "val_loss",
            "early_stopping_patience": int(patience),
            "early_stopping_min_delta": float(min_delta),
            "validation_interval_steps": int(val_interval),
            "best_step": int(best_step),
            "best_val_loss": float(best),
            "stop_reason": stop_reason,
        },
    )
    mark_done(ctx, stage, [out / "best.pt", out / "training_history.csv"])


def psnr_from_mse(mse: float) -> float:
    return float(10.0 * math.log10(1.0 / max(float(mse), EPS)))


def ssim_grayscale(pred: torch.Tensor, target: torch.Tensor) -> float:
    if skimage_ssim is None:
        return float("nan")
    p = pred.detach().cpu().squeeze(0).numpy()
    t = target.detach().cpu().squeeze(0).numpy()
    return float(skimage_ssim(t, p, data_range=1.0))


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


@torch.no_grad()
def stage_tdncnn_evaluate(ctx: Context) -> None:
    if is_done(ctx, "tdncnn_evaluate"):
        print("tdncnn_evaluate: resume skip")
        return
    if ctx.dry_run:
        print("tdncnn_evaluate: would evaluate TDnCNN variants")
        return
    test_ds = load_clean_dataset(ctx, train=False)
    eval_indices = np.load(ctx.experiment_root / "00_protocol" / "evaluation_indices.npy")
    paths = materialize_eval_noise_cache(
        dataset=test_ds,
        dataset_name="mnist",
        original_indices=eval_indices,
        output_dir=ctx.experiment_root / "07_evaluation" / "eval_noise_cache",
        sigma_min=float(deep_get(ctx.cfg, "evaluation.eval_noise.sigma_min", 0.1)),
        sigma_max=float(deep_get(ctx.cfg, "evaluation.eval_noise.sigma_max", 0.8)),
        eval_seed=int(deep_get(ctx.cfg, "seeds.eval_noise_seed", 4242)),
        eval_noise_version=str(deep_get(ctx.cfg, "evaluation.eval_noise.version", "mnist_eval_awgn_v1")),
        overwrite=False,
    )
    eval_ds = EvalNoiseDataset(paths[0], paths[1])
    loader = DataLoader(eval_ds, batch_size=int(deep_get(ctx.cfg, "evaluation.batch_size", 256)), shuffle=False)
    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for variant in VARIANTS:
        ckpt = ctx.experiment_root / "06_downstream" / "tdncnn" / variant / "best.pt"
        model = TinyTDnCNN(in_channels=1).to(ctx.device)
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
                row = {"variant": variant, "image_index": idx, "mse": mse, "psnr": psnr_from_mse(mse), "ssim": ssim_grayscale(p, t), "lpips": float("nan")}
                rows.append(row)
                values.append(row)
                idx += 1
        frame = pd.DataFrame(values)
        train_meta = read_yaml(ctx.experiment_root / "04_selected_configuration" / "selected_score_config.yaml")
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
    out = ctx.experiment_root / "07_evaluation" / "tdncnn"
    out.mkdir(parents=True, exist_ok=True)
    per = pd.DataFrame(rows)
    summ = pd.DataFrame(summary)
    full_psnr = float(summ[summ["variant"] == "full"]["psnr"].iloc[0])
    summ["delta_psnr_vs_full"] = summ["psnr"] - full_psnr
    per.to_csv(out / "per_image_metrics.csv", index=False)
    summ.to_csv(out / "summary_metrics.csv", index=False)
    mark_done(ctx, "tdncnn_evaluate", [out / "summary_metrics.csv"])


def stage_summary(ctx: Context) -> None:
    if is_done(ctx, "summary"):
        print("summary: resume skip")
        return
    if ctx.dry_run:
        print("summary: would write final summary")
        return
    out = ctx.experiment_root / "08_summary"
    out.mkdir(parents=True, exist_ok=True)
    selected = read_yaml(ctx.experiment_root / "04_selected_configuration" / "selected_score_config.yaml")
    metrics = pd.read_csv(ctx.experiment_root / "07_evaluation" / "tdncnn" / "summary_metrics.csv")
    lines = [
        "# MNIST Image-Space Variable-AWGN Summary",
        "",
        f"run_id: `{ctx.run_id}`",
        f"dataset: MNIST, grayscale `[B,1,28,28]`",
        f"selected checkpoint epoch: `{selected['checkpoint_epoch']}`",
        f"selected report timestep: `{selected['report_timestep']}`",
        f"MC K: `{selected['mc_k']}`",
        "",
        dataframe_markdown(metrics),
        "",
        "DRUNet for MNIST is deferred while CIFAR DRUNet frozen runtime is active.",
    ]
    (out / "final_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out / "summary.json", {"run_id": ctx.run_id, "selected": selected, "tdncnn_rows": metrics.to_dict(orient="records")})
    mark_done(ctx, "summary", [out / "final_summary.md"])


def run_stage(ctx: Context, stage: str) -> None:
    if stage == "prepare":
        stage_prepare(ctx)
    elif stage == "ddpm_train":
        stage_ddpm_train(ctx)
    elif stage == "score_validate":
        stage_score_validate(ctx)
    elif stage == "score_select":
        stage_score_select(ctx)
    elif stage == "filter":
        stage_filter(ctx)
    elif stage in {"tdncnn_full", "tdncnn_topk", "tdncnn_quantile", "tdncnn_random"}:
        stage_tdncnn_train(ctx, stage)
    elif stage == "tdncnn_evaluate":
        stage_tdncnn_evaluate(ctx)
    elif stage == "summary":
        stage_summary(ctx)
    else:
        raise ValueError(f"Unknown stage: {stage}")


def run_smoke_checks(ctx: Context) -> None:
    if ctx.dry_run:
        print("smoke_checks: dry-run")
        return
    tmp = ctx.experiment_root
    stages = ["prepare", "ddpm_train", "score_validate", "score_select", "filter", "tdncnn_full", "tdncnn_topk", "tdncnn_quantile", "tdncnn_random", "tdncnn_evaluate", "summary"]
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
    topk = np.load(idx_root / "topk_10_indices.npy")
    quantile = np.load(idx_root / "quantile_spread_10_indices.npy")
    random = np.load(idx_root / "random_10_indices.npy")
    assert len(random) == len(topk) == len(quantile)
    assert len(np.unique(random)) == len(random)
    assert set(random.tolist()).issubset(set(full.tolist()))
    score_df = pd.read_csv(tmp / "05_filtering" / "scores_full.csv")
    validation_indices = np.load(tmp / "00_protocol" / "downstream_validation_indices.npy").astype(np.int64)
    score_df = score_df[~score_df["original_index"].isin(set(validation_indices.tolist()))].copy()
    seed = int(deep_get(ctx.cfg, "filtering.random_seed", deep_get(ctx.cfg, "seeds.subset_seed", 42)))
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
    parser = argparse.ArgumentParser(description="Run canonical MNIST current research pipeline.")
    parser.add_argument("--config", default="configs/experiments/mnist.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--experiment-root", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--stage", default="all", choices=["all", "smoke_checks", *STAGES])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Use synthetic tiny MNIST-like data and smoke budgets.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--force-fail-gate", action="store_true", help="Testing hook: force score_select to fail.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ctx = resolve_context(args)
    validate_current_protocol("MNIST", ctx.cfg, downstream_models=["tdncnn"])
    print_protocol_summary("MNIST", ctx.cfg, run_id=ctx.run_id, device=str(ctx.device), downstream_models=["tdncnn"])
    if args.stage == "smoke_checks":
        run_smoke_checks(ctx)
        return
    stages = STAGES if args.stage == "all" else [args.stage]
    for stage in stages:
        with stage_logging("MNIST", stage, run_id=ctx.run_id):
            run_stage(ctx, stage)


if __name__ == "__main__":
    main()
