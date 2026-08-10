from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity as skimage_ssim
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml

from src.DRUNet_image import build_drunet
from src.TDnCNN_image import align_lpips_scaling_layer, prepare_for_lpips
from src.evaluation import cifar10_score_validation as validation_runner
from src.image_lab_replication_datasets import EvalNoiseDataset, MaterializedCorpusDataset, stable_int_seed
from src.utils.current_protocol import (
    print_protocol_summary,
    stage_logging,
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
    parser.add_argument("--stage", choices=[*STAGES, "all", "validation_all", "tdncnn_all", "drunet_all"], default="all")
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
    return validation_runner.parse_args_from(argv) if hasattr(validation_runner, "parse_args_from") else _parse_validation_argv(argv)


def _parse_validation_argv(argv: list[str]) -> argparse.Namespace:
    old_argv = sys.argv
    try:
        sys.argv = ["cifar10_score_validation.py", *argv]
        return validation_runner.parse_args()
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
    def __init__(self, corpus_dataset: MaterializedCorpusDataset, *, task_seed: int, task_noise_version: str, sigma_min: float, sigma_max: float) -> None:
        self.corpus_dataset = corpus_dataset
        self.task_seed = int(task_seed)
        self.task_noise_version = str(task_noise_version)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

    def __len__(self) -> int:
        return len(self.corpus_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_target, original_index = self.corpus_dataset[index]
        sample_seed = stable_int_seed("cifar10", "train_task_noise", int(original_index), self.task_seed, self.task_noise_version)
        generator = torch.Generator().manual_seed(sample_seed)
        u = torch.rand((), generator=generator).item()
        sigma = float(self.sigma_min + (self.sigma_max - self.sigma_min) * u)
        noise = torch.randn(x_target.shape, generator=generator, dtype=x_target.dtype) * sigma
        x_input = (x_target + noise).clamp(0.0, 1.0)
        return x_input.to(torch.float32), x_target.to(torch.float32), torch.tensor(sigma, dtype=torch.float32), torch.tensor(int(original_index), dtype=torch.long)


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


def drunet_train_dataset(root: Path, cfg: dict[str, Any], subset: str, *, validation: bool = False) -> CorpusTaskNoiseWithSigmaDataset:
    if validation:
        indices_path = root / "00_protocol" / "score_tuning_original_indices.npy"
        version_suffix = "_validation"
    else:
        indices_path = root / "08_downstream_commands" / f"{subset}_indices.npy"
        version_suffix = ""
    if not indices_path.exists():
        raise FileNotFoundError(f"Missing CIFAR DRUNet indices: {indices_path}")
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
        task_noise_version=str(noise_cfg.get("version", "cifar10_current_task_awgn_v1")) + version_suffix,
        sigma_min=float(noise_cfg.get("sigma_min", 0.1)),
        sigma_max=float(noise_cfg.get("sigma_max", 0.8)),
    )


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
        if step == 1 or step == int(max_steps) or step % val_interval == 0:
            print(f"DRUNet {subset}: step={step}/{int(max_steps)} train_loss={train_loss:.6f} val_loss={val_loss:.6f} best_val_loss={best_val_loss:.6f} best_step={best_step} patience={checks_without_improvement}/{patience}")
            pd.DataFrame(rows).to_csv(out_dir / "training_history.csv", index=False)
        if stop_reason == "early_stopping":
            break
    if not best_path.exists():
        torch.save(model.state_dict(), best_path)
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "step": int(step)}, out_dir / "last.pt")
    pd.DataFrame(rows).to_csv(out_dir / "training_history.csv", index=False)
    subset_indices = np.load(root / "08_downstream_commands" / f"{subset}_indices.npy")
    meta = {
        "model": "drunet",
        "subset": subset,
        "variant": SUBSET_LABELS[subset],
        "subset_size": int(len(subset_indices)),
        "optimizer_steps": int(step),
        "batch_size": int(batch_size),
        "initial_state": str(initial),
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


def run_stage(cfg: dict[str, Any], args: argparse.Namespace, root: Path, stage: str) -> None:
    if args.dry_run:
        print(f"[dry-run] CIFAR-10 stage={stage} root={root}")
        return
    if stage in VALIDATION_STAGE_MAP:
        validation_runner.run_stage(validation_args(cfg, args, root, stage))
    elif stage == "tdncnn_evaluate":
        write_tdncnn_summary(root)
        print(f"tdncnn_evaluate: wrote {root / '07_evaluation' / 'tdncnn' / 'summary_table.csv'}")
    elif stage in DRUNET_SUBSETS:
        run_drunet_train(cfg, args, root, DRUNET_SUBSETS[stage])
    elif stage == "drunet_evaluate":
        run_drunet_evaluate(cfg, args, root)
        print(f"drunet_evaluate: wrote {root / '07_evaluation' / 'drunet' / 'summary_table.csv'}")
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
    validate_current_protocol("CIFAR-10", cfg, downstream_models=["tdncnn", "drunet"])
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
        downstream_models=["tdncnn", "drunet"],
    )
    for stage in expanded_stages(args.stage):
        with stage_logging("CIFAR-10", stage, run_id=args.run_id or str(deep_get(cfg, "experiment.run_id", "cifar10_001"))):
            run_stage(cfg, args, root, stage)


if __name__ == "__main__":
    main()
