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
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as skimage_ssim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.evaluate_external import (
    BENCHMARKS,
    audit_datasets,
    compute_metrics,
    load_pairs_for_status,
    load_tdncnn_checkpoint,
    predict_model,
    write_panel,
)
from src.DRUNet_image import build_drunet
from src.TDnCNN_image import DnCNN_2D, align_lpips_scaling_layer, prepare_for_lpips
from src.image_lab_replication_datasets import EvalNoiseDataset, MaterializedCorpusDataset, stable_int_seed


VARIANTS = {
    "full": {"label": "Full", "dir": "01_full", "subset_file": "full_indices.npy"},
    "topk_10": {"label": "Top-K", "dir": "02_topk_10", "subset_file": "topk_10_indices.npy"},
    "quantile_spread_10": {"label": "Quantile", "dir": "03_quantile_10", "subset_file": "quantile_spread_10_indices.npy"},
}
STAGE_ALIASES = {
    "train_full": "drunet_full",
    "train_topk": "drunet_topk",
    "train_quantile": "drunet_quantile",
    "evaluate": "drunet_evaluate",
    "external": "drunet_external",
    "summary": "drunet_summary",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the current CIFAR-10 selected-score DRUNet branch.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "experiments" / "cifar10_drunet_e18_t100_mc16.yaml")
    parser.add_argument("--current-cifar-root", type=Path, default=None)
    parser.add_argument("--filtering-dir", type=Path, default=None)
    parser.add_argument("--tdncnn-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--external-benchmark-root", type=Path, default=None)
    parser.add_argument(
        "--stage",
        choices=[
            "audit",
            "drunet_full",
            "drunet_topk",
            "drunet_quantile",
            "drunet_evaluate",
            "drunet_external",
            "drunet_summary",
            "drunet_all",
            "smoke",
            "all",
            *STAGE_ALIASES,
        ],
        default="drunet_all",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-eval-images", type=int, default=None)
    parser.add_argument("--max-external-images", type=int, default=None)
    parser.add_argument("--lpips", action="store_true", help="Enable LPIPS even when the config disables it.")
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
    for key, arg_name in [
        ("current_cifar_root", "current_cifar_root"),
        ("filtering_dir", "filtering_dir"),
        ("tdncnn_root", "tdncnn_root"),
        ("output_dir", "output_dir"),
        ("external_benchmark_root", "external_benchmark_root"),
    ]:
        override = getattr(args, arg_name)
        if override is not None:
            cfg[key] = str(override)
        cfg[key] = str(resolve_path(cfg[key]))
    if args.smoke or args.stage == "smoke":
        cfg["output_dir"] = str(resolve_path(cfg["smoke"]["output_dir"]))
        cfg["model"]["official"] = bool(cfg["smoke"].get("official", False))
        cfg["training"]["total_steps"] = int(cfg["smoke"]["train_steps"])
        cfg["training"]["batch_size"] = min(int(cfg["training"]["batch_size"]), 4)
        cfg["evaluation"]["lpips_enabled"] = False
        cfg["external"]["lpips_enabled"] = False
    return cfg


def dirs(root: Path) -> dict[str, Path]:
    paths = {
        "audit": root / "00_audit",
        "internal": root / "04_internal_evaluation",
        "external": root / "05_external_evaluation",
        "summary": root / "06_summary",
    }
    for name, meta in VARIANTS.items():
        paths[name] = root / str(meta["dir"])
    return paths


def ensure_dirs(paths: dict[str, Path], dry_run: bool) -> None:
    if dry_run:
        return
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def short_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_info(device: torch.device) -> dict[str, Any]:
    info = {"device": str(device), "gpu": None, "peak_cuda_memory": None}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["peak_cuda_memory"] = int(torch.cuda.max_memory_allocated(device))
    return info


def subset_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    root = Path(cfg["filtering_dir"]) / "02_subsets"
    return {name: root / str(meta["subset_file"]) for name, meta in VARIANTS.items()}


def load_indices(cfg: dict[str, Any], variant: str, *, smoke: bool = False) -> np.ndarray:
    path = subset_paths(cfg)[variant]
    if not path.exists():
        raise FileNotFoundError(f"Missing selected subset indices: {path}")
    indices = np.load(path).astype(np.int64)
    indices = np.sort(indices.copy())
    if smoke:
        indices = indices[: int(cfg["smoke"]["max_train_items"])]
    return indices


def load_resolved_tdncnn_config(cfg: dict[str, Any]) -> dict[str, Any]:
    path = Path(cfg["tdncnn_root"]) / "00_protocol" / "resolved_downstream_config.yaml"
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def canonical_artifact_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    current = Path(cfg["current_cifar_root"])
    tdn_cfg = load_resolved_tdncnn_config(cfg)
    return {
        "corpus_cache": current / "01_corpus" / "train_corpus.pt",
        "corpus_manifest": current / "01_corpus" / "train_manifest.csv",
        "scores_full": Path(cfg["filtering_dir"]) / "01_scores" / "scores_full.csv",
        "subset_manifest": Path(cfg["filtering_dir"]) / "02_subsets" / "subset_indices.json",
        "eval_input": resolve_path(tdn_cfg.get("eval_input")) or Path(tdn_cfg["eval_input"]),
        "eval_target": resolve_path(tdn_cfg.get("eval_target")) or Path(tdn_cfg["eval_target"]),
        "eval_manifest": resolve_path(tdn_cfg.get("eval_manifest")) or Path(tdn_cfg["eval_manifest"]),
        "validation_indices": current / "00_protocol" / "score_tuning_original_indices.npy",
    }


class CorpusTaskNoiseWithSigmaDataset(Dataset):
    def __init__(self, corpus_dataset: MaterializedCorpusDataset, *, dataset_name: str, task_seed: int, task_noise_version: str, sigma_min: float, sigma_max: float) -> None:
        self.corpus_dataset = corpus_dataset
        self.dataset_name = dataset_name
        self.task_seed = int(task_seed)
        self.task_noise_version = str(task_noise_version)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

    def __len__(self) -> int:
        return len(self.corpus_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_target, original_index = self.corpus_dataset[index]
        sample_seed = stable_int_seed(self.dataset_name, "train_task_noise", int(original_index), self.task_seed, self.task_noise_version)
        generator = torch.Generator().manual_seed(sample_seed)
        u = torch.rand((), generator=generator).item()
        sigma = float(self.sigma_min + (self.sigma_max - self.sigma_min) * u)
        noise = torch.randn(x_target.shape, generator=generator, dtype=x_target.dtype) * sigma
        x_input = (x_target + noise).clamp(0.0, 1.0)
        return x_input.to(torch.float32), x_target.to(torch.float32), torch.tensor(sigma, dtype=torch.float32), torch.tensor(int(original_index), dtype=torch.long)


class EvalNoiseWithSigmaDataset(Dataset):
    def __init__(self, input_path: Path, target_path: Path, manifest_path: Path, max_images: int | None = None) -> None:
        base = EvalNoiseDataset(input_path, target_path)
        self.inputs = base.inputs
        self.targets = base.targets
        self.manifest = pd.read_csv(manifest_path)
        if "sigma" not in self.manifest.columns:
            raise ValueError(f"Eval manifest must contain sigma for DRUNet conditioning: {manifest_path}")
        if max_images is not None:
            self.inputs = self.inputs[:max_images]
            self.targets = self.targets[:max_images]
            self.manifest = self.manifest.iloc[:max_images].reset_index(drop=True)

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.manifest.iloc[int(index)]
        return self.inputs[int(index)], self.targets[int(index)], torch.tensor(float(row["sigma"]), dtype=torch.float32), torch.tensor(int(row.get("original_index", index)), dtype=torch.long)


def build_model(cfg: dict[str, Any]) -> nn.Module:
    mcfg = cfg["model"]
    return build_drunet(
        in_channels=int(mcfg.get("in_channels", 3)),
        official=bool(mcfg.get("official", True)),
        nc=mcfg.get("nc"),
        nb=int(mcfg.get("nb", 4)),
        act_mode=str(mcfg.get("act_mode", "R")),
        downsample_mode=str(mcfg.get("downsample_mode", "strideconv")),
        upsample_mode=str(mcfg.get("upsample_mode", "convtranspose")),
    )


def model_state(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"Unsupported checkpoint format: {type(payload)!r}")


def load_model(path: Path, cfg: dict[str, Any], device: torch.device) -> nn.Module:
    model = build_model(cfg).to(device)
    state = model_state(torch.load(path, map_location=device))
    if any(str(key).startswith("module.") for key in state):
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def ensure_initial_state(cfg: dict[str, Any], paths: dict[str, Path]) -> Path:
    path = paths["audit"] / "initial_state_dict.pt"
    if path.exists():
        return path
    torch.manual_seed(int(cfg["training"]["model_seed"]))
    model = build_model(cfg)
    torch.save(model.state_dict(), path)
    return path


def make_train_dataset(cfg: dict[str, Any], variant: str, *, validation: bool = False, smoke: bool = False) -> CorpusTaskNoiseWithSigmaDataset:
    artifacts = canonical_artifact_paths(cfg)
    if validation:
        indices = np.load(artifacts["validation_indices"]).astype(np.int64)
        if smoke:
            indices = indices[: int(cfg["smoke"]["max_train_items"])]
        task_version = str(cfg["training"]["task_noise"]["version"]) + str(cfg["training"].get("validation_task_noise_version_suffix", "_validation"))
    else:
        indices = load_indices(cfg, variant, smoke=smoke)
        task_version = str(cfg["training"]["task_noise"]["version"])
    corpus = MaterializedCorpusDataset(
        artifacts["corpus_cache"],
        artifacts["corpus_manifest"],
        original_indices=np.sort(indices.copy()),
        normalized=False,
        return_original_index=True,
    )
    noise = cfg["training"]["task_noise"]
    return CorpusTaskNoiseWithSigmaDataset(
        corpus,
        dataset_name=str(cfg["dataset"]),
        task_seed=int(cfg["training"]["task_seed"]),
        task_noise_version=task_version,
        sigma_min=float(noise["sigma_min"]),
        sigma_max=float(noise["sigma_max"]),
    )


def cycle(loader: DataLoader):
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


def aggregate(per_image: pd.DataFrame, variant_col: str = "Variant") -> pd.DataFrame:
    rows = []
    for key, group in per_image.groupby(variant_col, sort=False):
        row: dict[str, Any] = {variant_col: key, "Images": int(len(group))}
        for metric in ["MSE", "PSNR", "SSIM", "LPIPS"]:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[metric] = float(values.mean()) if values.notna().any() else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else np.nan
            row[f"{metric}_median"] = float(values.median()) if values.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def audit_stage(cfg: dict[str, Any], paths: dict[str, Path], *, dry_run: bool = False) -> dict[str, Any]:
    artifacts = canonical_artifact_paths(cfg)
    for name, path in artifacts.items():
        if name != "validation_indices" and not path.exists():
            raise FileNotFoundError(f"Missing required artifact {name}: {path}")
    subsets = {}
    for variant, path in subset_paths(cfg).items():
        if not path.exists():
            raise FileNotFoundError(f"Missing subset file: {path}")
        arr = np.load(path).astype(np.int64)
        subsets[variant] = {"path": str(path.resolve()), "sha256": sha256_file(path), "size": int(len(arr)), "unique": bool(len(np.unique(arr)) == len(arr))}
    full_size = subsets["full"]["size"]
    selected_size = subsets["topk_10"]["size"]
    total_steps = int(cfg["training"]["total_steps"])
    if int(cfg["training"]["reference_epochs"]) == 15 and total_steps == 42195 and full_size != 45000:
        print(f"warning: DRUNet budget was resolved for 45k Full split, actual Full size is {full_size}")
    resolved = {
        "architecture_files": ["src/DRUNet_image.py", "src/external/drunet/network_unet.py"],
        "old_training_files": ["scripts/train_drunet.py", "scripts/internal/train_DRUNet_image.py", "scripts/internal/drunet_image_runs_config.py"],
        "external_helpers": ["scripts/evaluate_external.py"],
        "why_absent_from_current_cifar_launcher": "the active selected CIFAR downstream runner is TDnCNN-only and the old DRUNet suite uses online clean-target sigma25 training, not the current x_corpus target protocol",
        "minimal_change": "new selected-score DRUNet runner reusing the existing DRUNet model and external helpers, reading the current corpus/subset/eval artifacts without recomputing DDPM scores or filtering",
        "score_config": cfg["selected_score_config"],
        "subsets": subsets,
        "training_budget": {
            "batch_size": int(cfg["training"]["batch_size"]),
            "optimizer": cfg["training"]["optimizer"],
            "lr": float(cfg["training"]["lr"]),
            "scheduler": cfg["training"]["scheduler"],
            "reference_epochs": int(cfg["training"]["reference_epochs"]),
            "optimizer_steps": total_steps,
            "full_effective_epochs": float(total_steps * int(cfg["training"]["batch_size"]) / max(full_size, 1)),
            "topk_effective_epochs": float(total_steps * int(cfg["training"]["batch_size"]) / max(selected_size, 1)),
        },
        "protocol": {
            "target": "x_target = x_corpus",
            "input": "x_input = clip(x_corpus + task_noise, 0, 1)",
            "task_noise": cfg["training"]["task_noise"],
            "sigma_conditioning": "DRUNet receives the actual per-sample task sigma as forward(x, sigma=sigma); official adapter converts it to a spatial noise-level map",
            "loss": "MSE(pred, x_corpus)",
            "checkpoint_selection": cfg["training"]["checkpoint_selection"],
            "internal_eval": "same cached CIFAR test eval input/target and sigma manifest as current TDnCNN branch",
            "external_eval": "Kodak24/CBSD68/Urban100, AWGN sigma=25/255, seed=42, same noisy input for Full/Top-K/Quantile",
        },
        "artifact_paths": {key: str(path.resolve()) for key, path in artifacts.items()},
        "config_hash": short_hash(cfg),
    }
    if dry_run:
        print(json.dumps(resolved, indent=2, ensure_ascii=False, default=str))
        return resolved
    paths["audit"].mkdir(parents=True, exist_ok=True)
    write_json(paths["audit"] / "drunet_audit.json", resolved)
    lines = [
        "# DRUNet Audit",
        "",
        "## Found DRUNet implementation",
        "- Architecture: `src/DRUNet_image.py`, official DPIR network in `src/external/drunet/network_unet.py`.",
        "- Old training entrypoints: `scripts/train_drunet.py`, `scripts/internal/train_DRUNet_image.py`, `scripts/internal/run_DRUNet_image_suite.py`.",
        "- Old config: `scripts/internal/drunet_image_runs_config.py`.",
        "- External evaluator/helpers: `scripts/evaluate_external.py`.",
        "",
        "## Current decision",
        f"- Current CIFAR corpus: `{resolved['artifact_paths']['corpus_cache']}`.",
        f"- Current subsets: `{Path(cfg['filtering_dir']).resolve() / '02_subsets'}`.",
        "- No DDPM training, score validation, score recomputation, or subset recomputation is performed.",
        "- The old DRUNet suite is not used directly because it trains clean-target online-noise denoisers; this branch trains with `x_corpus` as target.",
        "",
        "## Training Budget",
        markdown_table(pd.DataFrame([resolved["training_budget"]])),
        "",
        "## Subsets",
        markdown_table(pd.DataFrame.from_dict(subsets, orient="index").reset_index(names="variant")),
        "",
        "## Protocol",
    ]
    for key, value in resolved["protocol"].items():
        lines.append(f"- {key}: {value}")
    (paths["audit"] / "drunet_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    OmegaConf.save(config=OmegaConf.create({**cfg, "config_hash": resolved["config_hash"]}), f=paths["audit"] / "resolved_config.yaml")
    ensure_initial_state(cfg, paths)
    return resolved


def train_variant(cfg: dict[str, Any], paths: dict[str, Path], variant: str, *, device: torch.device, resume: bool, dry_run: bool, smoke: bool) -> None:
    branch_dir = paths[variant]
    if dry_run:
        print(f"[dry-run] train {variant} -> {branch_dir}")
        return
    branch_dir.mkdir(parents=True, exist_ok=True)
    best_path = branch_dir / "best.pt"
    last_path = branch_dir / "last.pt"
    status_path = branch_dir / "status.json"
    history_path = branch_dir / "training_history.csv"
    if resume and status_path.exists() and best_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "complete":
            print(f"resume: skip completed {variant}")
            return

    initial_path = ensure_initial_state(cfg, paths)
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(initial_path, map_location=device), strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"].get("weight_decay", 0.0)))
    start_step = 0
    best_loss = float("inf")
    rows: list[dict[str, Any]] = []
    if resume and last_path.exists():
        payload = torch.load(last_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_step = int(payload.get("step", 0))
        best_loss = float(payload.get("best_train_loss", best_loss))
        if history_path.exists():
            rows = pd.read_csv(history_path).to_dict("records")

    dataset = make_train_dataset(cfg, variant, smoke=smoke)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"].get("num_workers", 0)),
        generator=torch.Generator().manual_seed(int(cfg["training"]["model_seed"])),
    )
    iterator = cycle(loader)
    criterion = nn.MSELoss()
    total_steps = int(cfg["training"]["total_steps"])
    checkpoint_interval = int(cfg["training"].get("checkpoint_interval_steps", 100))
    write_json(status_path, {"status": "running", "variant": variant, "step": start_step, "total_steps": total_steps})
    start_time = time.perf_counter()
    for step in tqdm(range(start_step + 1, total_steps + 1), desc=f"DRUNet {variant}", mininterval=10.0):
        x_input, x_target, sigma, _ = next(iterator)
        x_input = x_input.to(device)
        x_target = x_target.to(device)
        sigma = sigma.to(device)
        pred = model(x_input, sigma=sigma)
        loss = criterion(pred, x_target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.item())
        if loss_value < best_loss:
            best_loss = loss_value
            torch.save(model.state_dict(), best_path)
        rows.append({"step": int(step), "train_loss": loss_value})
        if step % checkpoint_interval == 0 or step == total_steps:
            pd.DataFrame(rows).to_csv(history_path, index=False)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": int(step),
                    "best_train_loss": best_loss,
                    "config_hash": short_hash(cfg),
                },
                last_path,
            )
            write_json(status_path, {"status": "running", "variant": variant, "step": int(step), "total_steps": total_steps})

    val_loss = validate_variant(cfg, paths, variant, best_path, device=device, smoke=smoke)
    elapsed = time.perf_counter() - start_time
    indices = load_indices(cfg, variant, smoke=smoke)
    meta = {
        "variant": variant,
        "label": VARIANTS[variant]["label"],
        "subset_size": int(len(indices)),
        "subset_sha256": sha256_file(subset_paths(cfg)[variant]),
        "initial_state": str(initial_path.resolve()),
        "initial_state_sha256": sha256_file(initial_path),
        "batch_size": int(cfg["training"]["batch_size"]),
        "optimizer_steps": total_steps,
        "total_seen_samples": int(total_steps * int(cfg["training"]["batch_size"])),
        "effective_epochs": float(total_steps * int(cfg["training"]["batch_size"]) / max(len(indices), 1)),
        "best_train_loss": best_loss,
        "validation_loss": float(val_loss),
        "wall_clock_seconds": elapsed,
        "task_noise": cfg["training"]["task_noise"],
        "best_checkpoint": str(best_path.resolve()),
        "last_checkpoint": str(last_path.resolve()),
        "config_hash": short_hash(cfg),
        **device_info(device),
    }
    OmegaConf.save(config=OmegaConf.create(meta), f=branch_dir / "resolved_config.yaml")
    write_json(branch_dir / "run_meta.json", meta)
    write_json(status_path, {"status": "complete", **meta})
    print(f"DRUNet {variant} complete: best={best_path}")


@torch.no_grad()
def validate_variant(cfg: dict[str, Any], paths: dict[str, Path], variant: str, best_path: Path, *, device: torch.device, smoke: bool) -> float:
    model = load_model(best_path, cfg, device)
    dataset = make_train_dataset(cfg, variant, validation=True, smoke=smoke)
    loader = DataLoader(dataset, batch_size=int(cfg["training"]["batch_size"]), shuffle=False, num_workers=0)
    criterion = nn.MSELoss(reduction="sum")
    total = 0.0
    count = 0
    for x_input, x_target, sigma, _ in loader:
        x_input = x_input.to(device)
        x_target = x_target.to(device)
        sigma = sigma.to(device)
        total += float(criterion(model(x_input, sigma=sigma), x_target).item())
        count += int(np.prod(x_target.shape))
    return total / max(count, 1)


@torch.no_grad()
def evaluate_internal(cfg: dict[str, Any], paths: dict[str, Path], *, device: torch.device, dry_run: bool, max_images: int | None, lpips_enabled: bool) -> None:
    if dry_run:
        print(f"[dry-run] internal evaluate -> {paths['internal']}")
        return
    artifacts = canonical_artifact_paths(cfg)
    dataset = EvalNoiseWithSigmaDataset(artifacts["eval_input"], artifacts["eval_target"], artifacts["eval_manifest"], max_images=max_images)
    loader = DataLoader(dataset, batch_size=int(cfg["evaluation"]["batch_size"]), shuffle=False, num_workers=0)
    lpips_model = create_lpips_model(device, lpips_enabled)
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        model = load_model(paths[variant] / "best.pt", cfg, device)
        offset = 0
        for x_input, x_target, sigma, original_index in tqdm(loader, desc=f"Eval {variant}", mininterval=10.0):
            x_input = x_input.to(device)
            x_target = x_target.to(device)
            sigma = sigma.to(device)
            pred = model(x_input, sigma=sigma)
            mse, psnr, ssim, lpips_values = batch_metrics(pred, x_target, lpips_model)
            for i in range(int(x_input.shape[0])):
                rows.append(
                    {
                        "Architecture": "DRUNet",
                        "Variant": VARIANTS[variant]["label"],
                        "variant_key": variant,
                        "sample_index": int(offset + i),
                        "original_index": int(original_index[i].item()),
                        "MSE": float(mse[i]),
                        "PSNR": float(psnr[i]),
                        "SSIM": float(ssim[i]),
                        "LPIPS": float(lpips_values[i]),
                    }
                )
            offset += int(x_input.shape[0])
    paths["internal"].mkdir(parents=True, exist_ok=True)
    per = pd.DataFrame(rows)
    per.to_csv(paths["internal"] / "drunet_per_image_metrics.csv", index=False)
    summary = aggregate(per)
    train_meta = {VARIANTS[key]["label"]: json.loads((paths[key] / "run_meta.json").read_text(encoding="utf-8")) for key in VARIANTS}
    summary["Architecture"] = "DRUNet"
    summary["Train size"] = summary["Variant"].map(lambda v: int(train_meta[v]["subset_size"]))
    summary["Optimizer steps"] = summary["Variant"].map(lambda v: int(train_meta[v]["optimizer_steps"]))
    summary["Train time"] = summary["Variant"].map(lambda v: float(train_meta[v]["wall_clock_seconds"]))
    full = summary[summary["Variant"] == "Full"].iloc[0]
    for metric in ["PSNR", "SSIM", "LPIPS"]:
        summary[f"Delta {metric} vs DRUNet Full"] = summary[metric] - float(full[metric])
    summary.to_csv(paths["internal"] / "drunet_summary_metrics.csv", index=False)
    (paths["internal"] / "drunet_summary_metrics.md").write_text(markdown_table(summary) + "\n", encoding="utf-8")
    write_json(paths["internal"] / "evaluation_meta.json", {"eval_input": str(artifacts["eval_input"]), "eval_target": str(artifacts["eval_target"]), "eval_manifest": str(artifacts["eval_manifest"]), "lpips_enabled": bool(lpips_model is not None), **device_info(device)})
    write_per_image_deltas(per, paths["internal"] / "drunet_per_image_deltas.csv")
    write_internal_combined(cfg, paths)


def write_per_image_deltas(per: pd.DataFrame, output: Path) -> None:
    full = per[per["Variant"] == "Full"].set_index("sample_index")
    rows = []
    for variant in ["Top-K", "Quantile"]:
        current = per[per["Variant"] == variant].set_index("sample_index")
        joined = current[["PSNR", "SSIM", "LPIPS", "MSE"]].join(full[["PSNR", "SSIM", "LPIPS", "MSE"]], lsuffix="", rsuffix="_Full")
        for metric in ["PSNR", "SSIM", "LPIPS", "MSE"]:
            delta = joined[metric] - joined[f"{metric}_Full"]
            rows.append(
                {
                    "Variant": variant,
                    "Metric": metric,
                    "mean_delta": float(delta.mean()),
                    "median_delta": float(delta.median()),
                    "fraction_variant_gt_full": float((delta > 0).mean()),
                }
            )
    pd.DataFrame(rows).to_csv(output, index=False)


def read_tdncnn_internal_table(cfg: dict[str, Any]) -> pd.DataFrame:
    path = Path(cfg["tdncnn_root"]) / "04_summary" / "downstream_comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    rows = []
    mapping = [
        ("full_45000_e18_t100", "Full"),
        ("topk_10_e18_t100", "Top-K"),
        ("quantile_spread_10_e18_t100", "Quantile"),
    ]
    for variant_name, subset in mapping:
        row = df[df["variant"] == variant_name]
        if row.empty:
            continue
        item = row.iloc[0]
        rows.append(
            {
                "Architecture": "TDnCNN",
                "Subset": subset,
                "Train size": int(item.get("subset_size", item.get("train_size", np.nan))),
                "PSNR": float(item["PSNR"]),
                "SSIM": float(item["SSIM"]),
                "LPIPS": float(item["LPIPS"]),
                "MSE": float(item["MSE"]) if "MSE" in item else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_internal_combined(cfg: dict[str, Any], paths: dict[str, Path]) -> None:
    drunet = pd.read_csv(paths["internal"] / "drunet_summary_metrics.csv")
    drows = drunet.rename(columns={"Variant": "Subset"})[["Architecture", "Subset", "Train size", "PSNR", "SSIM", "LPIPS", "MSE"]]
    tdn = read_tdncnn_internal_table(cfg)
    combined = pd.concat([tdn, drows], ignore_index=True)
    combined.to_csv(paths["internal"] / "internal_all_denoisers.csv", index=False)
    (paths["internal"] / "internal_all_denoisers.md").write_text(markdown_table(combined) + "\n", encoding="utf-8")
    delta_rows = []
    for arch, group in combined.groupby("Architecture", sort=False):
        full = group[group["Subset"] == "Full"]
        if full.empty:
            continue
        full_psnr = float(full.iloc[0]["PSNR"])
        for _, row in group.iterrows():
            delta_rows.append({"Architecture": arch, "Subset": row["Subset"], "Delta PSNR vs Full": float(row["PSNR"] - full_psnr)})
    pd.DataFrame(delta_rows).to_csv(paths["internal"] / "internal_delta_topk_quantile_vs_full.csv", index=False)


def tdncnn_checkpoints(cfg: dict[str, Any]) -> dict[str, Path]:
    root = Path(cfg["tdncnn_root"]) / "02_training"
    return {
        "Full": root / "full_45000" / "best.pt",
        "Top-K": root / "topk_10" / "best.pt",
        "Quantile": root / "quantile_spread_10" / "best.pt",
    }


def variant_train_sizes(cfg: dict[str, Any], paths: dict[str, Path]) -> dict[tuple[str, str], int]:
    sizes: dict[tuple[str, str], int] = {}
    for key, meta in VARIANTS.items():
        label = str(meta["label"])
        run_meta = paths[key] / "run_meta.json"
        if run_meta.exists():
            sizes[("DRUNet", label)] = int(json.loads(run_meta.read_text(encoding="utf-8"))["subset_size"])
        else:
            sizes[("DRUNet", label)] = int(len(load_indices(cfg, key)))
    tdn_meta = {
        "Full": Path(cfg["tdncnn_root"]) / "02_training" / "full_45000" / "run_meta.json",
        "Top-K": Path(cfg["tdncnn_root"]) / "02_training" / "topk_10" / "run_meta.json",
        "Quantile": Path(cfg["tdncnn_root"]) / "02_training" / "quantile_spread_10" / "run_meta.json",
    }
    for label, path in tdn_meta.items():
        if path.exists():
            sizes[("TDnCNN", label)] = int(json.loads(path.read_text(encoding="utf-8"))["subset_size"])
    return sizes


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
    image = Image.fromarray((arr * 255).round().astype(np.uint8))
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return canvas


def write_combined_panel(output: Path, representatives: dict[str, dict[str, torch.Tensor]]) -> None:
    cols = ["Clean", "Noisy", "TDnCNN Full", "TDnCNN Top-K", "DRUNet Full", "DRUNet Top-K"]
    img_w, img_h = 330, 230
    left_w, header_h = 130, 78
    row_h = img_h + 44
    canvas = Image.new("RGB", (left_w + len(cols) * img_w + 40, header_h + len(BENCHMARKS) * row_h + 20), "white")
    draw = ImageDraw.Draw(canvas)
    header_font = font(22, True)
    for c, col in enumerate(cols):
        draw.text((left_w + c * img_w + 10, 24), col, font=header_font, fill=(0, 0, 0))
    for r, benchmark in enumerate(BENCHMARKS):
        y = header_h + r * row_h
        draw.text((18, y + 88), benchmark, font=header_font, fill=(0, 0, 0))
        item = representatives.get(benchmark)
        if not item:
            continue
        for c, col in enumerate(cols):
            key = col if col not in {"Clean", "Noisy"} else col.lower()
            canvas.paste(tensor_image(item[key], img_w - 18, img_h), (left_w + c * img_w + 8, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


@torch.no_grad()
def evaluate_external(cfg: dict[str, Any], paths: dict[str, Path], *, device: torch.device, dry_run: bool, max_images: int | None, lpips_enabled: bool) -> None:
    if dry_run:
        print(f"[dry-run] external evaluate -> {paths['external']}")
        return
    root = Path(cfg["external_benchmark_root"])
    audit = audit_datasets(root, paths["external"])
    missing = [name for name in BENCHMARKS if not audit["benchmarks"][name]["exists"] or int(audit["benchmarks"][name]["file_count"]) == 0]
    if missing:
        raise FileNotFoundError(f"Missing external benchmark data: {missing} under {root}")
    lpips_model = create_lpips_model(device, lpips_enabled)
    drunet_models = {meta["label"]: load_model(paths[key] / "best.pt", cfg, device) for key, meta in VARIANTS.items()}
    tdn_paths = tdncnn_checkpoints(cfg)
    tdn_models = {label: load_tdncnn_checkpoint(path, device) for label, path in tdn_paths.items() if path.exists()}
    sigma = int(cfg["external"]["sigma"])
    sigma_value = sigma / 255.0
    seed = int(cfg["external"]["seed"])
    per_rows = []
    drunet_reps: dict[str, dict[str, torch.Tensor]] = {}
    combined_reps: dict[str, dict[str, torch.Tensor]] = {}
    for benchmark in BENCHMARKS:
        pairs = load_pairs_for_status(root, benchmark, audit["benchmarks"][benchmark], sigma, seed, max_images)
        for image_idx, pair in enumerate(tqdm(pairs, desc=f"External {benchmark}", mininterval=10.0)):
            clean = pair["clean"]
            noisy = pair["noisy"]
            rep_d = {"clean": clean, "noisy": noisy}
            rep_c = {"clean": clean, "noisy": noisy}
            for variant, model in drunet_models.items():
                pred = predict_model(model, noisy, "drunet", sigma_value, device)
                metrics = compute_metrics(pred, clean, lpips_model)
                per_rows.append({"Benchmark": benchmark, "Architecture": "DRUNet", "Variant": variant, "Image": pair["image"], **metrics})
                if image_idx == 0:
                    rep_d[variant] = pred
                    if variant in {"Full", "Top-K"}:
                        rep_c[f"DRUNet {variant}"] = pred
            for variant, model in tdn_models.items():
                pred = predict_model(model, noisy, "tdncnn", sigma_value, device)
                metrics = compute_metrics(pred, clean, lpips_model)
                per_rows.append({"Benchmark": benchmark, "Architecture": "TDnCNN", "Variant": variant, "Image": pair["image"], **metrics})
                if image_idx == 0 and variant in {"Full", "Top-K"}:
                    rep_c[f"TDnCNN {variant}"] = pred
            if image_idx == 0:
                drunet_reps[benchmark] = rep_d
                combined_reps[benchmark] = rep_c
    per = pd.DataFrame(per_rows)
    paths["external"].mkdir(parents=True, exist_ok=True)
    per.to_csv(paths["external"] / "external_per_image_metrics.csv", index=False)
    summary_rows = []
    for (benchmark, arch, variant), group in per.groupby(["Benchmark", "Architecture", "Variant"], sort=False):
        row = {"Benchmark": benchmark, "Architecture": arch, "Variant": variant, "Images": int(len(group))}
        for metric in ["PSNR", "SSIM", "LPIPS", "MSE"]:
            row[metric] = float(pd.to_numeric(group[metric], errors="coerce").mean())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    train_sizes = variant_train_sizes(cfg, paths)
    summary["Train size"] = summary.apply(lambda row: train_sizes.get((str(row["Architecture"]), str(row["Variant"])), np.nan), axis=1)
    for (benchmark, arch), group in summary.groupby(["Benchmark", "Architecture"], sort=False):
        full = group[group["Variant"] == "Full"]
        if full.empty:
            continue
        idx = group.index
        for metric in ["PSNR", "SSIM", "LPIPS"]:
            summary.loc[idx, f"Delta {metric} vs same-architecture Full"] = summary.loc[idx, metric] - float(full.iloc[0][metric])
    all_table = summary.rename(columns={"Variant": "Subset", "Delta PSNR vs same-architecture Full": "Delta PSNR vs Full"})
    preferred_cols = ["Benchmark", "Architecture", "Subset", "Train size", "PSNR", "SSIM", "LPIPS", "MSE", "Delta PSNR vs Full"]
    all_table = all_table[[col for col in preferred_cols if col in all_table.columns]]
    all_table.to_csv(paths["external"] / "external_all_denoisers.csv", index=False)
    (paths["external"] / "external_all_denoisers.md").write_text(markdown_table(all_table) + "\n", encoding="utf-8")
    drunet_summary = summary[summary["Architecture"] == "DRUNet"].copy()
    drunet_summary.to_csv(paths["external"] / "drunet_external_summary.csv", index=False)
    (paths["external"] / "drunet_external_summary.md").write_text(markdown_table(drunet_summary) + "\n", encoding="utf-8")
    write_per_image_external_deltas(per[per["Architecture"] == "DRUNet"], paths["external"] / "drunet_external_per_image_deltas.csv")
    write_panel(paths["external"] / "drunet_external_panel.png", drunet_reps, drunet_summary.rename(columns={"Architecture": "Architecture"}), has_reference=True)
    write_combined_panel(paths["external"] / "combined_tdncnn_drunet_panel.png", combined_reps)


def write_per_image_external_deltas(per: pd.DataFrame, output: Path) -> None:
    rows = []
    for benchmark, group in per.groupby("Benchmark", sort=False):
        full = group[group["Variant"] == "Full"].set_index("Image")
        for variant in ["Top-K", "Quantile"]:
            current = group[group["Variant"] == variant].set_index("Image")
            joined = current[["PSNR", "SSIM", "LPIPS", "MSE"]].join(full[["PSNR", "SSIM", "LPIPS", "MSE"]], lsuffix="", rsuffix="_Full")
            for metric in ["PSNR", "SSIM", "LPIPS", "MSE"]:
                delta = joined[metric] - joined[f"{metric}_Full"]
                rows.append({"Benchmark": benchmark, "Variant": variant, "Metric": metric, "mean_delta": float(delta.mean()), "median_delta": float(delta.median()), "fraction_variant_gt_full": float((delta > 0).mean())})
    pd.DataFrame(rows).to_csv(output, index=False)


def summary_stage(cfg: dict[str, Any], paths: dict[str, Path], *, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] summary -> {paths['summary']}")
        return
    paths["summary"].mkdir(parents=True, exist_ok=True)
    internal = paths["internal"] / "internal_all_denoisers.csv"
    external = paths["external"] / "external_all_denoisers.csv"
    inventory = {
        "audit": str((paths["audit"] / "drunet_audit.md").resolve()),
        "internal_all_denoisers": str(internal.resolve()) if internal.exists() else None,
        "external_all_denoisers": str(external.resolve()) if external.exists() else None,
        "checkpoints": {key: str((paths[key] / "best.pt").resolve()) for key in VARIANTS},
    }
    write_json(paths["summary"] / "artifact_inventory.json", inventory)
    lines = [
        "# CIFAR-10 DRUNet Selected-Score Branch Summary",
        "",
        "- DDPM epoch/timestep/MC were reused: epoch 18, report timestep 100, code timestep 99, MC K=16.",
        "- Score files and subset indices were reused from the current filtering artifacts.",
        "- Random subset is intentionally not part of this DRUNet branch.",
        "",
    ]
    if internal.exists():
        lines.extend(["## Internal", markdown_table(pd.read_csv(internal)), ""])
    if external.exists():
        lines.extend(["## External", markdown_table(pd.read_csv(external)), ""])
    (paths["summary"] / "drunet_final_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage(args: argparse.Namespace, cfg: dict[str, Any], paths: dict[str, Path]) -> None:
    stage = STAGE_ALIASES.get(args.stage, args.stage)
    smoke = bool(args.smoke or stage == "smoke")
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    ensure_dirs(paths, args.dry_run)
    if stage == "smoke":
        stages = ["audit", "drunet_full", "drunet_topk", "drunet_quantile", "drunet_evaluate", "drunet_external", "drunet_summary"]
    elif stage in {"all", "drunet_all"}:
        stages = ["audit", "drunet_full", "drunet_topk", "drunet_quantile", "drunet_evaluate", "drunet_external", "drunet_summary"]
    else:
        stages = [stage]

    for item in stages:
        if item == "audit":
            audit_stage(cfg, paths, dry_run=args.dry_run)
        elif item == "drunet_full":
            train_variant(cfg, paths, "full", device=device, resume=args.resume, dry_run=args.dry_run, smoke=smoke)
        elif item == "drunet_topk":
            train_variant(cfg, paths, "topk_10", device=device, resume=args.resume, dry_run=args.dry_run, smoke=smoke)
        elif item == "drunet_quantile":
            train_variant(cfg, paths, "quantile_spread_10", device=device, resume=args.resume, dry_run=args.dry_run, smoke=smoke)
        elif item == "drunet_evaluate":
            max_images = args.max_eval_images if args.max_eval_images is not None else (int(cfg["smoke"]["max_eval_images"]) if smoke else None)
            evaluate_internal(cfg, paths, device=device, dry_run=args.dry_run, max_images=max_images, lpips_enabled=bool(args.lpips or cfg["evaluation"].get("lpips_enabled", True)))
        elif item == "drunet_external":
            max_images = args.max_external_images if args.max_external_images is not None else (int(cfg["smoke"]["max_external_images"]) if smoke else None)
            evaluate_external(cfg, paths, device=device, dry_run=args.dry_run, max_images=max_images, lpips_enabled=bool(args.lpips or cfg["external"].get("lpips_enabled", True)))
        elif item == "drunet_summary":
            summary_stage(cfg, paths, dry_run=args.dry_run)
        else:
            raise ValueError(f"Unsupported stage: {item}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    paths = dirs(Path(cfg["output_dir"]))
    run_stage(args, cfg, paths)


if __name__ == "__main__":
    main()
