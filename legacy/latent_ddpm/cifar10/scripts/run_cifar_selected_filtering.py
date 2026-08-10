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
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.ddpm.image_space import build_image_space_ddpm_model
from src.dataset_registry import build_torchvision_split, load_dataset_config
from src.evaluation.image_lab_replication_analysis import select_quantile_spread_exact
from src.image_lab_replication_datasets import MaterializedCorpusDataset


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
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Selected CIFAR-10 score-based filtering artifacts.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-epoch", type=int, default=18)
    parser.add_argument("--report-timestep", type=int, default=100)
    parser.add_argument("--code-timestep", type=int, default=99)
    parser.add_argument("--mc-k", type=int, default=16)
    parser.add_argument("--subset-fraction", type=float, default=0.10)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--stage", choices=["score", "subsets", "statistics", "figures", "panels", "summary", "all"], default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--visual-seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "protocol": output_dir / "00_protocol",
        "scores": output_dir / "01_scores",
        "chunks": output_dir / "01_scores" / "chunks",
        "subsets": output_dir / "02_subsets",
        "summary": output_dir / "03_summary",
        "figures": output_dir / "04_figures",
        "panels": output_dir / "05_panels",
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


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_config(experiment_root: Path) -> dict[str, Any]:
    path = experiment_root / "00_protocol" / "resolved_score_validation_config.yaml"
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected mapping config: {path}")
    return cfg


def build_model(cfg: dict[str, Any], checkpoint_path: Path, device: torch.device):
    dataset_cfg = cfg["dataset_cfg"]
    ddpm_cfg = cfg["ddpm"]
    model = build_image_space_ddpm_model(
        base_dim=int(ddpm_cfg.get("base_dim", 16)),
        deep=int(ddpm_cfg.get("deep", 3)),
        device=str(device),
        dataset_cfg=dataset_cfg,
        n_steps=int(ddpm_cfg.get("n_steps", 1000)),
        beta_start=float(ddpm_cfg.get("beta_start", 1e-4)),
        beta_end=float(ddpm_cfg.get("beta_end", 0.02)),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    legacy_keys = {"autoencoder_checkpoint_path", "autoencoder_kind", "latent_noise_mode"}
    present = sorted(key for key in legacy_keys if key in checkpoint)
    if present:
        raise RuntimeError(
            f"Refusing to load latent/autoencoder DDPM checkpoint into image-space CIFAR filtering: {checkpoint_path}. "
            f"Legacy metadata keys present: {present}."
        )
    model.model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def normalize_for_ddpm(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32) * 2.0 - 1.0


def load_artifacts(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    root = args.experiment_root
    checkpoint = root / "03_ddpm" / "checkpoints" / f"epoch_{args.checkpoint_epoch:03d}.pt"
    manifest = root / "01_corpus" / "train_manifest.csv"
    corpus = root / "01_corpus" / "train_corpus.pt"
    ddpm_indices = root / "00_protocol" / "ddpm_filter_train_original_indices.npy"
    tuning_indices = root / "00_protocol" / "score_tuning_original_indices.npy"
    confirmation_indices = root / "00_protocol" / "score_confirmation_original_indices.npy"
    for path in [checkpoint, manifest, corpus, ddpm_indices, tuning_indices, confirmation_indices]:
        if not path.exists():
            raise FileNotFoundError(path)
    return {
        "checkpoint": checkpoint.resolve(),
        "manifest": manifest.resolve(),
        "corpus": corpus.resolve(),
        "ddpm_indices": ddpm_indices.resolve(),
        "tuning_indices": tuning_indices.resolve(),
        "confirmation_indices": confirmation_indices.resolve(),
    }


def forward_noise_seeds(seed: int, mc_k: int) -> list[int]:
    return [int(hashlib.sha256(f"cifar-selected-filtering|{seed}|{idx}".encode()).hexdigest()[:8], 16) for idx in range(mc_k)]


def write_protocol(args: argparse.Namespace, dirs: dict[str, Path], cfg: dict[str, Any], artifacts: dict[str, Any], model, device: torch.device) -> None:
    idx = int(args.code_timestep)
    beta_t = float(model.betas[idx].detach().cpu())
    alpha_bar_t = float(model.alphas_cumprod[idx].detach().cpu())
    payload = {
        "dataset": "CIFAR-10",
        "status": "tuning-selected, preliminary, confirmation pending",
        "checkpoint_path": str(artifacts["checkpoint"]),
        "checkpoint_sha256": sha256_file(artifacts["checkpoint"]),
        "epoch": int(args.checkpoint_epoch),
        "report_timestep": int(args.report_timestep),
        "code_timestep_index": int(args.code_timestep),
        "beta_t": beta_t,
        "alpha_bar_t": alpha_bar_t,
        "effective_diffusion_noise": float(math.sqrt(max(0.0, 1.0 - alpha_bar_t))),
        "mc_k": int(args.mc_k),
        "score_formula": "mean_K sum(flatten(eps_pred) ** 2)",
        "topk_direction": "minimum score",
        "subset_fraction": float(args.subset_fraction),
        "random_seed": int(args.random_seed),
        "forward_noise_seeds": forward_noise_seeds(args.random_seed, args.mc_k),
        "corpus_manifest_path": str(artifacts["manifest"]),
        "corpus_cache_path": str(artifacts["corpus"]),
        "ddpm_filter_train_indices_path": str(artifacts["ddpm_indices"]),
        "score_tuning_indices_path": str(artifacts["tuning_indices"]),
        "score_confirmation_indices_path": str(artifacts["confirmation_indices"]),
        "started_or_resumed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda) if torch.version.cuda is not None else None,
    }
    stable = dict(payload)
    stable.pop("started_or_resumed_at", None)
    payload["config_hash"] = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()[:16]
    OmegaConf.save(config=OmegaConf.create(payload), f=dirs["protocol"] / "filtering_config.yaml")
    (dirs["protocol"] / "run_meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


@torch.inference_mode()
def score_batch(model, x_01: torch.Tensor, code_timestep: int, seeds: list[int], batch_offset: int, device: torch.device, use_amp: bool) -> torch.Tensor:
    x = normalize_for_ddpm(x_01).to(device, non_blocking=True)
    batch_size = int(x.shape[0])
    t = torch.full((batch_size,), int(code_timestep), dtype=torch.long, device=device)
    acc = torch.zeros(batch_size, device=device, dtype=torch.float32)
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(use_amp and device.type == "cuda"))
    for seed in seeds:
        # Derive a batch-local seed from the global MC seed to keep chunk resume deterministic.
        gen = torch.Generator(device=device).manual_seed((int(seed) + int(batch_offset) * 1000003) % (2**63 - 1))
        eps = torch.randn(x.shape, generator=gen, device=device, dtype=x.dtype)
        x_t = model.q_sample(x, t, eps)
        with amp_ctx:
            eps_pred = model.model(x_t, t)
        acc += eps_pred.float().flatten(start_dim=1).pow(2).sum(dim=1)
    return acc / float(len(seeds))


def stage_score(args: argparse.Namespace, dirs: dict[str, Path], cfg: dict[str, Any], artifacts: dict[str, Any]) -> dict[str, Any]:
    scores_path = dirs["scores"] / "scores_full.csv"
    if scores_path.exists() and args.resume:
        existing = pd.read_csv(scores_path, usecols=["original_index"])
        return {"score_seconds": 0.0, "resumed_existing": True, "n_images": int(len(existing))}
    device = resolve_device(args.device)
    model = build_model(cfg, artifacts["checkpoint"], device)
    write_protocol(args, dirs, cfg, artifacts, model, device)
    manifest = pd.read_csv(artifacts["manifest"])
    ddpm_indices = np.load(artifacts["ddpm_indices"]).astype(np.int64)
    tuning = set(np.load(artifacts["tuning_indices"]).astype(np.int64).tolist())
    confirmation = set(np.load(artifacts["confirmation_indices"]).astype(np.int64).tolist())
    if tuning.intersection(ddpm_indices.tolist()) or confirmation.intersection(ddpm_indices.tolist()):
        raise RuntimeError("Validation indices overlap ddpm_filter_train.")
    dataset = MaterializedCorpusDataset(
        artifacts["corpus"],
        artifacts["manifest"],
        original_indices=ddpm_indices,
        normalized=False,
        return_original_index=True,
    )
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), pin_memory=device.type == "cuda")
    seeds = forward_noise_seeds(args.random_seed, args.mc_k)
    start = time.perf_counter()
    done_chunks = {path.stem for path in dirs["chunks"].glob("chunk_*.csv")} if args.resume else set()
    for chunk_id, (x, original) in enumerate(tqdm(loader, desc="Scoring ddpm_filter_train")):
        chunk_name = f"chunk_{chunk_id:05d}"
        chunk_path = dirs["chunks"] / f"{chunk_name}.csv"
        if chunk_name in done_chunks:
            continue
        scores = score_batch(
            model,
            x,
            code_timestep=int(args.code_timestep),
            seeds=seeds,
            batch_offset=chunk_id,
            device=device,
            use_amp=True,
        ).detach().cpu().numpy().astype(np.float64)
        pd.DataFrame({"original_index": original.numpy().astype(np.int64), "score_value": scores}).to_csv(chunk_path, index=False)
    score_parts = [pd.read_csv(path) for path in sorted(dirs["chunks"].glob("chunk_*.csv"))]
    score_df = pd.concat(score_parts, ignore_index=True)
    score_df = score_df.merge(
        manifest[
            [
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
        ],
        on="original_index",
        how="left",
    )
    score_df["class_name"] = score_df["class_id"].map(lambda x: CIFAR10_CLASSES[int(x)] if int(x) in range(10) else str(x))
    score_df = score_df.sort_values(["score_value", "original_index"], ascending=[True, True]).reset_index(drop=True)
    score_df["score_rank"] = np.arange(1, len(score_df) + 1, dtype=np.int64)
    score_df["score_percentile"] = (score_df["score_rank"] - 1) / max(len(score_df) - 1, 1)
    columns = [
        "original_index",
        "class_id",
        "class_name",
        "score_value",
        "score_rank",
        "score_percentile",
        "target_psnr_db",
        "realized_psnr_db_postclip",
        "realized_snr_db_postclip",
        "sigma",
        "corruption_mse",
        "clipped_fraction",
        "signal_power",
    ]
    score_df[columns].to_csv(scores_path, index=False)
    validate_scores(score_df, ddpm_indices, tuning, confirmation)
    return {"score_seconds": time.perf_counter() - start, "resumed_existing": False, "n_images": int(len(score_df))}


def validate_scores(score_df: pd.DataFrame, ddpm_indices: np.ndarray, tuning: set[int], confirmation: set[int]) -> None:
    if len(score_df) != len(ddpm_indices):
        raise RuntimeError(f"Score rows mismatch: {len(score_df)} vs {len(ddpm_indices)}")
    if score_df["original_index"].nunique() != len(score_df):
        raise RuntimeError("original_index is not unique")
    if not np.isfinite(score_df["score_value"].to_numpy()).all():
        raise RuntimeError("Non-finite score values detected")
    first = score_df.sort_values(["score_rank", "original_index"]).iloc[0]
    if int(first["score_rank"]) != 1 or float(first["score_value"]) != float(score_df["score_value"].min()):
        raise RuntimeError("Rank 1 is not the minimum score")
    ids = set(score_df["original_index"].astype(int).tolist())
    if ids & tuning or ids & confirmation:
        raise RuntimeError("score_tuning/score_confirmation indices found in filtering scores")


def stage_subsets(args: argparse.Namespace, dirs: dict[str, Path]) -> dict[str, np.ndarray]:
    scores = pd.read_csv(dirs["scores"] / "scores_full.csv")
    n_full = len(scores)
    n_subset = int(round(float(args.subset_fraction) * n_full))
    topk = scores.sort_values(["score_value", "original_index"], ascending=[True, True]).head(n_subset)["original_index"].to_numpy(dtype=np.int64)
    q_df = scores.rename(columns={"score_value": "score_value"})
    quantile = select_quantile_spread_exact(q_df, n_subset, seed=int(args.random_seed), min_points_per_bin=30)
    rng = np.random.default_rng(int(args.random_seed))
    random = np.sort(rng.choice(np.sort(scores["original_index"].to_numpy(dtype=np.int64)), size=n_subset, replace=False))
    subsets = {"full": np.sort(scores["original_index"].to_numpy(dtype=np.int64)), "topk_10": np.sort(topk), "quantile_spread_10": np.sort(quantile), "random_10": random}
    membership = scores[["original_index", "score_value", "score_rank"]].copy()
    membership["is_topk_10"] = membership["original_index"].isin(set(subsets["topk_10"]))
    membership["is_quantile_spread_10"] = membership["original_index"].isin(set(subsets["quantile_spread_10"]))
    membership["is_random_10"] = membership["original_index"].isin(set(subsets["random_10"]))
    membership.to_csv(dirs["subsets"] / "subset_membership.csv", index=False)
    serializable = {name: [int(item) for item in values.tolist()] for name, values in subsets.items()}
    (dirs["subsets"] / "subset_indices.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    for name, values in subsets.items():
        np.save(dirs["subsets"] / f"{name}_indices.npy", values)
    validate_subsets(scores, subsets, n_subset, int(args.random_seed))
    return subsets


def validate_subsets(scores: pd.DataFrame, subsets: dict[str, np.ndarray], n_subset: int, seed: int) -> None:
    train_set = set(subsets["full"].tolist())
    for name in ["topk_10", "quantile_spread_10", "random_10"]:
        if len(subsets[name]) != n_subset:
            raise RuntimeError(f"{name} size mismatch")
        if not set(subsets[name].tolist()).issubset(train_set):
            raise RuntimeError(f"{name} has indices outside full")
    expected_topk = set(scores.sort_values(["score_value", "original_index"], ascending=[True, True]).head(n_subset)["original_index"].astype(int))
    if set(subsets["topk_10"].tolist()) != expected_topk:
        raise RuntimeError("Top-K is not the exact minimum-score subset")
    rng = np.random.default_rng(seed)
    expected_random = set(np.sort(rng.choice(np.sort(scores["original_index"].to_numpy(dtype=np.int64)), size=n_subset, replace=False)).tolist())
    if set(subsets["random_10"].tolist()) != expected_random:
        raise RuntimeError("Random subset is not reproducible")


def js_divergence(full_counts: np.ndarray, subset_counts: np.ndarray) -> float:
    p = full_counts.astype(np.float64) / max(float(full_counts.sum()), 1.0)
    q = subset_counts.astype(np.float64) / max(float(subset_counts.sum()), 1.0)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log((p + EPS) / (m + EPS))) + 0.5 * np.sum(q * np.log((q + EPS) / (m + EPS))))


def bootstrap_delta_ci(full: np.ndarray, subset: np.ndarray, seed: int, n: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    full = np.asarray(full, dtype=np.float64)
    subset = np.asarray(subset, dtype=np.float64)
    deltas = np.empty(int(n), dtype=np.float64)
    for idx in range(int(n)):
        f = full[rng.integers(0, len(full), size=len(full))]
        s = subset[rng.integers(0, len(subset), size=len(subset))]
        deltas[idx] = np.median(s) - np.median(f)
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def subset_df(scores: pd.DataFrame, indices: np.ndarray) -> pd.DataFrame:
    return scores[scores["original_index"].isin(set(indices.tolist()))].copy()


def stage_statistics(args: argparse.Namespace, dirs: dict[str, Path], subsets: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    scores = pd.read_csv(dirs["scores"] / "scores_full.csv")
    if subsets is None:
        subsets = load_subset_indices(dirs)
    full_df = subset_df(scores, subsets["full"])
    full_counts = full_df["class_id"].value_counts().reindex(range(10), fill_value=0).to_numpy()
    rows = []
    for name in ["full", "topk_10", "quantile_spread_10", "random_10"]:
        df = subset_df(scores, subsets[name])
        counts = df["class_id"].value_counts().reindex(range(10), fill_value=0).to_numpy()
        row = {
            "subset": name,
            "size": int(len(df)),
            "score_mean": float(df["score_value"].mean()),
            "score_median": float(df["score_value"].median()),
            "target_psnr_mean": float(df["target_psnr_db"].mean()),
            "target_psnr_median": float(df["target_psnr_db"].median()),
            "realized_psnr_mean": float(df["realized_psnr_db_postclip"].mean()),
            "realized_psnr_median": float(df["realized_psnr_db_postclip"].median()),
            "realized_snr_mean": float(df["realized_snr_db_postclip"].mean()),
            "realized_snr_median": float(df["realized_snr_db_postclip"].median()),
            "sigma_mean": float(df["sigma"].mean()),
            "sigma_median": float(df["sigma"].median()),
            "corruption_mse_mean": float(df["corruption_mse"].mean()),
            "clipped_fraction_mean": float(df["clipped_fraction"].mean()),
            "class_js_divergence_vs_full": 0.0 if name == "full" else js_divergence(full_counts, counts),
        }
        rows.append(row)
    summary = pd.DataFrame(rows)
    full = summary[summary["subset"] == "full"].iloc[0]
    delta_columns = {
        "target_psnr_median": "delta_target_psnr_median",
        "realized_psnr_median": "delta_realized_psnr_median",
        "realized_snr_median": "delta_realized_snr_median",
        "sigma_median": "delta_sigma_median",
        "corruption_mse_mean": "delta_corruption_mse_mean",
    }
    for col, out_col in delta_columns.items():
        summary[out_col] = summary[col] - float(full[col])
    for metric, out_prefix in [
        ("realized_psnr_db_postclip", "delta_realized_psnr_median"),
        ("realized_snr_db_postclip", "delta_realized_snr_median"),
        ("sigma", "delta_sigma_median"),
    ]:
        lows, highs = [], []
        full_arr = full_df[metric].to_numpy()
        for idx, name in enumerate(summary["subset"]):
            if name == "full":
                lows.append(0.0)
                highs.append(0.0)
            else:
                arr = subset_df(scores, subsets[str(name)])[metric].to_numpy()
                low, high = bootstrap_delta_ci(full_arr, arr, seed=int(args.random_seed) + idx * 101, n=int(args.bootstrap_resamples))
                lows.append(low)
                highs.append(high)
        summary[f"{out_prefix}_ci_low"] = lows
        summary[f"{out_prefix}_ci_high"] = highs
    summary.to_csv(dirs["summary"] / "filtering_summary.csv", index=False)
    write_filtering_summary_md(summary, dirs)
    return summary


def write_filtering_summary_md(summary: pd.DataFrame, dirs: dict[str, Path]) -> None:
    lines = [
        "# CIFAR-10 Selected Score Filtering Summary",
        "",
        "Status: tuning-selected, preliminary, confirmation pending.",
        "",
        "Previous tuning Top-K median PSNR shift = +2.844338 dB.",
    ]
    topk = summary[summary["subset"] == "topk_10"].iloc[0]
    lines.append(f"Current ddpm_filter_train Top-K median PSNR shift = {float(topk['delta_realized_psnr_median']):+.6f} dB.")
    lines.extend(["", dataframe_to_markdown(summary)])
    (dirs["summary"] / "filtering_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns]
    rows = []
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            val = row[col]
            values.append(f"{val:.6f}" if isinstance(val, (float, np.floating)) else str(val))
        rows.append(values)
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(v)) for w, v in zip(widths, row)]
    out = ["| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"]
    out.append("| " + " | ".join("-" * w for w in widths) + " |")
    out.extend("| " + " | ".join(v.ljust(w) for v, w in zip(row, widths)) + " |" for row in rows)
    return "\n".join(out)


def load_subset_indices(dirs: dict[str, Path]) -> dict[str, np.ndarray]:
    return {name: np.load(dirs["subsets"] / f"{name}_indices.npy").astype(np.int64) for name in ["full", "topk_10", "quantile_spread_10", "random_10"]}


def maybe_kde(ax, values: np.ndarray, xlim: tuple[float, float], color: str, linewidth: float = 2.0) -> None:
    try:
        from scipy.stats import gaussian_kde

        xs = np.linspace(xlim[0], xlim[1], 300)
        kde = gaussian_kde(np.asarray(values, dtype=np.float64))
        ax.plot(xs, kde(xs), color=color, linewidth=linewidth)
    except Exception:
        return


def plot_full_vs_subset(ax, full: pd.Series, subset: pd.Series, bins: np.ndarray, xlim: tuple[float, float], title: str) -> None:
    full_arr = full.to_numpy(dtype=np.float64)
    sub_arr = subset.to_numpy(dtype=np.float64)
    ax.hist(full_arr, bins=bins, density=True, alpha=0.38, color="#1f77b4", label=f"Full (n={len(full_arr)})")
    ax.hist(sub_arr, bins=bins, density=True, alpha=0.45, color="#ff7f0e", label=f"Subset (n={len(sub_arr)})")
    maybe_kde(ax, full_arr, xlim, "#1f77b4")
    maybe_kde(ax, sub_arr, xlim, "#ff7f0e")
    fm = float(np.median(full_arr))
    sm = float(np.median(sub_arr))
    ax.axvline(fm, color="#1f77b4", linestyle="--", linewidth=1.8)
    ax.axvline(sm, color="#ff7f0e", linestyle="--", linewidth=1.8)
    ax.set_xlim(*xlim)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(fontsize=10)


def stage_figures(args: argparse.Namespace, dirs: dict[str, Path], summary: pd.DataFrame | None = None) -> None:
    scores = pd.read_csv(dirs["scores"] / "scores_full.csv")
    subsets = load_subset_indices(dirs)
    if summary is None:
        summary = pd.read_csv(dirs["summary"] / "filtering_summary.csv")
    full_df = subset_df(scores, subsets["full"])
    method_names = [("topk_10", "Top-K 10%"), ("quantile_spread_10", "Quantile-spread 10%"), ("random_10", "Random 10%")]
    psnr_full = full_df["realized_psnr_db_postclip"]
    bins = np.histogram_bin_edges(psnr_full, bins=50)
    xlim = (float(psnr_full.min()), float(psnr_full.max()))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    for ax, (name, label) in zip(axes, method_names):
        sub = subset_df(scores, subsets[name])
        row = summary[summary["subset"] == name].iloc[0]
        title = (
            f"Full vs {label}\n"
            f"Full med {float(summary.loc[summary['subset']=='full','realized_psnr_median'].iloc[0]):.2f}; "
            f"Subset med {float(row['realized_psnr_median']):.2f}; "
            f"Delta {float(row['delta_realized_psnr_median']):+.2f} dB\n"
            f"95% CI [{float(row['delta_realized_psnr_median_ci_low']):+.2f}, {float(row['delta_realized_psnr_median_ci_high']):+.2f}]"
        )
        plot_full_vs_subset(ax, psnr_full, sub["realized_psnr_db_postclip"], bins, xlim, title)
        ax.set_xlabel("realized PSNR, dB", fontsize=12)
    fig.savefig(dirs["figures"] / "01_realized_psnr_shift_lab_style.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    plot_quality_shift_all_metrics(scores, subsets, dirs)
    plot_sigma_shift(scores, subsets, summary, dirs)
    plot_ecdf(scores, subsets, "realized_psnr_db_postclip", "realized PSNR, dB", dirs["figures"] / "04_realized_psnr_ecdf.png")
    plot_ecdf(scores, subsets, "realized_snr_db_postclip", "realized SNR, dB", dirs["figures"] / "05_realized_snr_ecdf.png")
    plot_deciles(scores, dirs)


def plot_quality_shift_all_metrics(scores: pd.DataFrame, subsets: dict[str, np.ndarray], dirs: dict[str, Path]) -> None:
    methods = [("topk_10", "Top-K 10%"), ("quantile_spread_10", "Quantile-spread 10%"), ("random_10", "Random 10%")]
    metrics = [("target_psnr_db", "target PSNR"), ("realized_psnr_db_postclip", "realized PSNR"), ("realized_snr_db_postclip", "realized SNR")]
    full = subset_df(scores, subsets["full"])
    bins_by_metric = {m: np.histogram_bin_edges(full[m], bins=45) for m, _ in metrics}
    xlim_by_metric = {m: (float(full[m].min()), float(full[m].max())) for m, _ in metrics}
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), constrained_layout=True)
    for r, (method, method_label) in enumerate(methods):
        sub = subset_df(scores, subsets[method])
        for c, (metric, metric_label) in enumerate(metrics):
            ax = axes[r, c]
            plot_full_vs_subset(ax, full[metric], sub[metric], bins_by_metric[metric], xlim_by_metric[metric], f"{method_label}: {metric_label}")
            delta = float(sub[metric].median() - full[metric].median())
            ax.text(0.04, 0.90, f"Delta median {delta:+.2f}", transform=ax.transAxes, fontsize=13, weight="bold")
            ax.set_xlabel(metric_label)
    fig.savefig(dirs["figures"] / "02_quality_shift_all_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sigma_shift(scores: pd.DataFrame, subsets: dict[str, np.ndarray], summary: pd.DataFrame, dirs: dict[str, Path]) -> None:
    methods = [("topk_10", "Top-K"), ("quantile_spread_10", "Quantile"), ("random_10", "Random")]
    full = subset_df(scores, subsets["full"])
    bins = np.histogram_bin_edges(full["sigma"], bins=45)
    xlim = (float(full["sigma"].min()), float(full["sigma"].max()))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, (method, label) in zip(axes, methods):
        sub = subset_df(scores, subsets[method])
        delta = float(sub["sigma"].median() - full["sigma"].median())
        plot_full_vs_subset(ax, full["sigma"], sub["sigma"], bins, xlim, f"Full vs {label}\nDelta median sigma {delta:+.4f}")
        ax.set_xlabel("sigma")
    fig.savefig(dirs["figures"] / "03_sigma_shift.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_ecdf(scores: pd.DataFrame, subsets: dict[str, np.ndarray], metric: str, xlabel: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5), constrained_layout=True)
    labels = [("full", "Full"), ("topk_10", "Top-K 10%"), ("quantile_spread_10", "Quantile-spread 10%"), ("random_10", "Random 10%")]
    for name, label in labels:
        df = subset_df(scores, subsets[name])
        vals = np.sort(df[metric].to_numpy(dtype=np.float64))
        y = np.arange(1, len(vals) + 1) / len(vals)
        med = float(np.median(vals))
        ax.plot(vals, y, linewidth=2, label=f"{label} n={len(vals)} med={med:.2f}")
        ax.axvline(med, linestyle="--", linewidth=1.1, alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Empirical CDF")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_deciles(scores: pd.DataFrame, dirs: dict[str, Path]) -> None:
    df = scores.sort_values(["score_value", "original_index"], ascending=[True, True]).copy()
    df["score_decile"] = pd.qcut(np.arange(len(df)), 10, labels=False) + 1
    rows = []
    for decile, group in df.groupby("score_decile", sort=True):
        rows.append(
            {
                "score_decile": int(decile),
                "realized_psnr_median": float(group["realized_psnr_db_postclip"].median()),
                "realized_psnr_q25": float(group["realized_psnr_db_postclip"].quantile(0.25)),
                "realized_psnr_q75": float(group["realized_psnr_db_postclip"].quantile(0.75)),
                "realized_snr_median": float(group["realized_snr_db_postclip"].median()),
                "sigma_median": float(group["sigma"].median()),
            }
        )
    dec = pd.DataFrame(rows)
    dec.to_csv(dirs["figures"] / "06_score_deciles_vs_quality.csv", index=False)
    fig, ax1 = plt.subplots(figsize=(13, 6.5), constrained_layout=True)
    x = dec["score_decile"].to_numpy()
    y = dec["realized_psnr_median"].to_numpy()
    yerr = np.vstack([y - dec["realized_psnr_q25"].to_numpy(), dec["realized_psnr_q75"].to_numpy() - y])
    ax1.errorbar(x, y, yerr=yerr, marker="o", linewidth=2, capsize=4, label="median realized PSNR + IQR")
    ax1.plot(x, dec["realized_snr_median"], marker="s", linewidth=2, label="median realized SNR")
    ax1.set_xlabel("score decile (lowest score to highest score)")
    ax1.set_ylabel("dB")
    ax2 = ax1.twinx()
    ax2.plot(x, dec["sigma_median"], marker="^", color="#2ca02c", linewidth=2, label="median sigma")
    ax2.set_ylabel("sigma")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    ax1.grid(alpha=0.25)
    fig.savefig(dirs["figures"] / "06_score_deciles_vs_quality.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_clean_dataset(cfg: dict[str, Any]):
    dataset_cfg = cfg["dataset_cfg"]
    data_root = Path(str(dataset_cfg.get("paths", {}).get("data_root", PROJECT_ROOT / "data")))
    return build_torchvision_split(dataset_cfg=dataset_cfg, train=True, data_root=data_root, transform_profile="tensor", download=False)


def make_panel(
    rows: list[dict[str, Any]],
    clean_dataset,
    corpus: torch.Tensor,
    pos_by_original: dict[int, int],
    output: Path,
    title: str,
    block_titles: list[str] | None = None,
    block_size: int = 8,
    figsize: tuple[float, float] = (15, 7.5),
) -> None:
    n_rows = math.ceil(len(rows) / block_size)
    image_size = 104
    pair_gap = 10
    tile_w = 318
    title_h = 58
    block_title_h = 32
    caption_h = 74
    tile_h = block_title_h + 18 + image_size + caption_h + 22
    margin_x = 28
    canvas_w = margin_x * 2 + block_size * tile_w
    canvas_h = title_h + n_rows * tile_h + 16
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    def font(size: int, bold: bool = False):
        candidates = [
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        ]
        for candidate in candidates:
            if Path(candidate).exists():
                return ImageFont.truetype(candidate, size=size)
        return ImageFont.load_default()

    title_font = font(26, bold=True)
    block_font = font(21, bold=True)
    label_font = font(14, bold=True)
    caption_font = font(15)

    def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
        arr = tensor.detach().cpu().permute(1, 2, 0).numpy()
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        return Image.fromarray(arr).resize((image_size, image_size), Image.Resampling.NEAREST)

    bbox = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((canvas_w - (bbox[2] - bbox[0])) // 2, 14), title, fill=(0, 0, 0), font=title_font)
    for idx, item in enumerate(rows):
        r = idx // block_size
        c = idx % block_size
        original = int(item["original_index"])
        x0 = margin_x + c * tile_w
        y0 = title_h + r * tile_h
        if block_titles and c == 0:
            draw.text((x0, y0), block_titles[r], fill=(0, 0, 0), font=block_font)
        image_y = y0 + block_title_h + 18
        clean_x = x0
        corrupt_x = x0 + image_size + pair_gap
        clean_img = tensor_to_image(clean_dataset[original][0])
        corrupted_img = tensor_to_image(corpus[pos_by_original[original]])
        draw.text((clean_x, image_y - 17), "Clean", fill=(35, 35, 35), font=label_font)
        draw.text((corrupt_x, image_y - 17), "Corpus", fill=(35, 35, 35), font=label_font)
        canvas.paste(clean_img, (clean_x, image_y))
        canvas.paste(corrupted_img, (corrupt_x, image_y))
        draw.rectangle([clean_x, image_y, clean_x + image_size - 1, image_y + image_size - 1], outline=(90, 90, 90))
        draw.rectangle([corrupt_x, image_y, corrupt_x + image_size - 1, image_y + image_size - 1], outline=(90, 90, 90))
        draw.multiline_text((x0, image_y + image_size + 8), str(item["caption"]), fill=(0, 0, 0), font=caption_font, spacing=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def stage_panels(args: argparse.Namespace, dirs: dict[str, Path], cfg: dict[str, Any]) -> None:
    scores = pd.read_csv(dirs["scores"] / "scores_full.csv")
    subsets = load_subset_indices(dirs)
    clean_dataset = load_clean_dataset(cfg)
    corpus = torch.load(args.experiment_root / "01_corpus" / "train_corpus.pt", map_location="cpu").to(torch.float32)
    manifest = pd.read_csv(args.experiment_root / "01_corpus" / "train_manifest.csv")
    pos_by_original = {int(r.original_index): int(r.cache_pos) for r in manifest[["original_index", "cache_pos"]].itertuples(index=False)}

    def caption(row, mode: str) -> str:
        if mode == "rank":
            return f"idx {int(row.original_index)}, {row.class_name}\nscore {row.score_value:.4f}, rank {int(row.score_rank)}\nPSNR {row.realized_psnr_db_postclip:.2f}, sigma {row.sigma:.4f}"
        if mode == "full":
            return f"idx {int(row.original_index)}, {row.class_name}\nscore {row.score_value:.4f}\nPSNR {row.realized_psnr_db_postclip:.2f}, SNR {row.realized_snr_db_postclip:.2f}, sigma {row.sigma:.4f}"
        return f"{row.class_name}\nPSNR {row.realized_psnr_db_postclip:.2f}\nscore pct {row.score_percentile:.3f}"

    lowest = scores.nsmallest(8, "score_value")
    highest = scores.nlargest(8, "score_value")
    rows = []
    for _, row in pd.concat([lowest, highest]).iterrows():
        item = row.to_dict()
        item["caption"] = caption(row, "rank")
        rows.append(item)
    make_panel(rows, clean_dataset, corpus, pos_by_original, dirs["panels"] / "01_topk_selected_vs_rejected.png", "Top-K selected vs rejected", ["Top-K selected: lowest score", "Top-K rejected: highest score"], figsize=(16, 7.5))

    for subset_name, filename, title in [
        ("topk_10", "02_topk_cleanest_noisiest_selected.png", "Top-K selected cleanest/noisiest"),
        ("quantile_spread_10", "03_quantile_cleanest_noisiest_selected.png", "Quantile-spread selected cleanest/noisiest"),
        ("random_10", "04_random_cleanest_noisiest_selected.png", "Random selected cleanest/noisiest"),
    ]:
        df = subset_df(scores, subsets[subset_name])
        selected_rows = pd.concat([df.nlargest(8, "realized_psnr_db_postclip"), df.nsmallest(8, "realized_psnr_db_postclip")])
        rows = []
        for _, row in selected_rows.iterrows():
            item = row.to_dict()
            item["caption"] = caption(row, "full")
            rows.append(item)
        make_panel(rows, clean_dataset, corpus, pos_by_original, dirs["panels"] / filename, title, ["Cleanest among selected by realized PSNR", "Noisiest among selected by realized PSNR"], figsize=(16, 7.5))

    rng = np.random.default_rng(int(args.visual_seed))
    rows = []
    block_titles = []
    for subset_name, label in [("topk_10", "Top-K 10%"), ("quantile_spread_10", "Quantile-spread 10%"), ("random_10", "Random 10%")]:
        df = subset_df(scores, subsets[subset_name]).sort_values("original_index")
        chosen = df.iloc[np.sort(rng.choice(np.arange(len(df)), size=8, replace=False))]
        block_titles.append(label)
        for _, row in chosen.iterrows():
            item = row.to_dict()
            item["caption"] = caption(row, "rep")
            rows.append(item)
    make_panel(rows, clean_dataset, corpus, pos_by_original, dirs["panels"] / "05_methods_representative_selected.png", "Representative selected examples", block_titles, figsize=(16, 10.5))

    rows = []
    block_titles = []
    for subset_name, label in [("topk_10", "Top-K noisiest selected"), ("quantile_spread_10", "Quantile noisiest selected"), ("random_10", "Random noisiest selected")]:
        df = subset_df(scores, subsets[subset_name]).nsmallest(8, "realized_psnr_db_postclip")
        block_titles.append(label)
        for _, row in df.iterrows():
            item = row.to_dict()
            item["caption"] = f"{row.class_name}\nPSNR {row.realized_psnr_db_postclip:.2f}\nscore {row.score_value:.4f}, sigma {row.sigma:.4f}"
            rows.append(item)
    make_panel(rows, clean_dataset, corpus, pos_by_original, dirs["panels"] / "06_noisiest_selected_comparison.png", "Noisiest selected comparison", block_titles, figsize=(16, 10.5))


def stage_summary(args: argparse.Namespace, dirs: dict[str, Path]) -> None:
    summary = pd.read_csv(dirs["summary"] / "filtering_summary.csv")
    full = summary[summary["subset"] == "full"].iloc[0]
    topk = summary[summary["subset"] == "topk_10"].iloc[0]
    quantile = summary[summary["subset"] == "quantile_spread_10"].iloc[0]
    random = summary[summary["subset"] == "random_10"].iloc[0]
    pngs = sorted([*dirs["figures"].glob("*.png"), *dirs["panels"].glob("*.png")])
    rel_pngs = [str(path.relative_to(args.output_dir)).replace("\\", "/") for path in pngs]
    lines = [
        "# Meeting Summary",
        "",
        "## 1. Selected DDPM configuration",
        "Epoch 18, report timestep 100, code timestep index 99, MC K=16. Status: tuning-selected, preliminary, confirmation pending.",
        "",
        "## 2. Dataset and subset sizes",
        f"Full working split: {int(full['size'])}. 10% subsets: {int(topk['size'])}.",
        "",
        "## 3. Full quality distribution",
        f"Full median realized PSNR: {float(full['realized_psnr_median']):.4f} dB.",
        "",
        "## 4. Top-K quality shift",
        f"Top-K median realized PSNR: {float(topk['realized_psnr_median']):.4f} dB; delta {float(topk['delta_realized_psnr_median']):+.4f} dB.",
        "",
        "## 5. Quantile quality shift",
        f"Quantile-spread median realized PSNR: {float(quantile['realized_psnr_median']):.4f} dB; delta {float(quantile['delta_realized_psnr_median']):+.4f} dB.",
        "",
        "## 6. Random quality shift",
        f"Random median realized PSNR: {float(random['realized_psnr_median']):.4f} dB; delta {float(random['delta_realized_psnr_median']):+.4f} dB.",
        "",
        "## 7. Comparison of selected examples",
        *[f"- {item}" for item in rel_pngs],
        "",
        "## 8. Main preliminary conclusion",
        "The selected DDPM score configuration was chosen on the tuning split. This run applies score-based filtering to the independent ddpm_filter_train working split only. The Top-K subset should be interpreted as a preliminary score-selected corpus for discussion, not as a downstream-validated training result.",
        "",
        "## 9. Methodological limitation",
        "Downstream denoising is absent in this run. Confirmation of the selected configuration remains a separate next stage.",
    ]
    (dirs["summary"] / "meeting_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    validate_artifacts(dirs, args.output_dir)


def validate_artifacts(dirs: dict[str, Path], output_dir: Path) -> None:
    expected = [
        "04_figures/01_realized_psnr_shift_lab_style.png",
        "04_figures/02_quality_shift_all_metrics.png",
        "04_figures/03_sigma_shift.png",
        "04_figures/04_realized_psnr_ecdf.png",
        "04_figures/05_realized_snr_ecdf.png",
        "04_figures/06_score_deciles_vs_quality.png",
        "05_panels/01_topk_selected_vs_rejected.png",
        "05_panels/02_topk_cleanest_noisiest_selected.png",
        "05_panels/03_quantile_cleanest_noisiest_selected.png",
        "05_panels/04_random_cleanest_noisiest_selected.png",
        "05_panels/05_methods_representative_selected.png",
        "05_panels/06_noisiest_selected_comparison.png",
    ]
    checks = []
    for rel in expected:
        path = output_dir / rel
        item = {"path": rel, "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0}
        if path.exists():
            with Image.open(path) as image:
                item["width"] = image.width
                item["height"] = image.height
                item["width_at_least_2200"] = image.width >= 2200
                item["landscape_or_square"] = image.height <= image.width
            item["size_gt_50kb"] = item["size_bytes"] > 50_000
        checks.append(item)
    (dirs["summary"] / "artifact_validation.json").write_text(json.dumps({"checks": checks}, indent=2), encoding="utf-8")


def print_priority1(summary: pd.DataFrame, dirs: dict[str, Path]) -> None:
    full = summary[summary["subset"] == "full"].iloc[0]
    topk = summary[summary["subset"] == "topk_10"].iloc[0]
    print("PRIORITY_1_DONE")
    print(f"full_size={int(full['size'])}")
    print(f"topk_size={int(topk['size'])}")
    print(f"full_median_realized_psnr={float(full['realized_psnr_median']):.6f}")
    print(f"topk_median_realized_psnr={float(topk['realized_psnr_median']):.6f}")
    print(f"delta_median={float(topk['delta_realized_psnr_median']):+.6f}")
    print(f"main_png={dirs['figures'] / '01_realized_psnr_shift_lab_style.png'}")


def run(args: argparse.Namespace) -> None:
    dirs = ensure_dirs(args.output_dir)
    cfg = read_config(args.experiment_root)
    artifacts = load_artifacts(args, cfg)
    stages = ["score", "subsets", "statistics", "figures", "panels", "summary"] if args.stage == "all" else [args.stage]
    score_meta = {}
    subsets = None
    summary = None
    if "score" in stages:
        score_meta = stage_score(args, dirs, cfg, artifacts)
    if "subsets" in stages:
        subsets = stage_subsets(args, dirs)
    if "statistics" in stages:
        summary = stage_statistics(args, dirs, subsets)
        print_priority1(summary, dirs)
    if "figures" in stages:
        if summary is None and (dirs["summary"] / "filtering_summary.csv").exists():
            summary = pd.read_csv(dirs["summary"] / "filtering_summary.csv")
        stage_figures(args, dirs, summary)
    if "panels" in stages:
        stage_panels(args, dirs, cfg)
    if "summary" in stages:
        stage_summary(args, dirs)
    if args.stage == "all":
        final = pd.read_csv(dirs["summary"] / "filtering_summary.csv")
        meta_path = dirs["protocol"] / "run_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.update(score_meta)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        filtering_config_path = dirs["protocol"] / "filtering_config.yaml"
        filtering_config = OmegaConf.load(filtering_config_path)
        for key, value in score_meta.items():
            filtering_config[key] = value
        OmegaConf.save(config=filtering_config, f=filtering_config_path)
        print_priority1(final, dirs)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
