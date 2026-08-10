from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.imagenet100_pipeline import (
    OnDemandCorpusDataset,
    PipelineContext,
    build_dataset,
    deterministic_corruption,
    read_yaml,
    stage_score_select,
)


def make_cfg() -> dict:
    cfg = read_yaml(Path("configs/experiments/imagenet100.yaml"))
    cfg["_smoke_runtime"] = True
    cfg["_real_smoke_runtime"] = False
    cfg["dataset"]["type"] = "synthetic"
    cfg["runtime"]["num_workers"] = 0
    return cfg


def deterministic_worker_check() -> None:
    cfg = make_cfg()
    ds = build_dataset(cfg, split="train", smoke=True)
    rows = []
    for idx in range(6):
        x, cls = ds[idx]
        _corrupt, meta = deterministic_corruption(
            x,
            dataset_name=cfg["dataset"]["name"],
            split="train",
            original_index=idx,
            corpus_seed=cfg["seeds"]["corpus_seed"],
            corruption_version=cfg["corpus"]["version"],
            psnr_min=cfg["corpus"]["target_psnr_min_db"],
            psnr_max=cfg["corpus"]["target_psnr_max_db"],
        )
        rows.append({"original_index": idx, "class_id": int(cls), "class_name": str(cls), "relative_path": str(idx), **meta})
    manifest = pd.DataFrame(rows)
    corpus = OnDemandCorpusDataset(ds, manifest, return_index=True)
    expected = float(corpus[3][0].sum())
    seen = {}
    for x, original, _cls in DataLoader(corpus, batch_size=2, shuffle=False, num_workers=2):
        for item, idx in zip(x, original):
            seen[int(idx)] = float(item.sum())
    assert abs(seen[3] - expected) < 1e-6


def failing_gate_check() -> None:
    cfg = make_cfg()
    ctx = PipelineContext(
        cfg=cfg,
        config_path=Path("configs/experiments/imagenet100.yaml"),
        run_id="standalone_bad_gate",
        experiment_root=Path("experiments/imagenet100/standalone_bad_gate"),
        smoke=True,
        real_smoke=False,
        dry_run=False,
        resume=False,
    )
    ctx.stage_root("score_validate").mkdir(parents=True, exist_ok=True)
    ctx.stage_root("prepare").mkdir(parents=True, exist_ok=True)
    (ctx.stage_root("ddpm_train") / "checkpoints").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"original_index": [0], "sample_seed": [1]}).to_csv(ctx.stage_root("prepare") / "corpus_manifest.csv", index=False)
    torch.save({"model_state_dict": {}}, ctx.stage_root("ddpm_train") / "checkpoints" / "epoch_001.pt")
    pd.DataFrame(
        [
            {
                "checkpoint_epoch": 1,
                "report_timestep": 1,
                "code_timestep_index": 0,
                "scan_phase": "final",
                "bootstrap_ci_low": -0.1,
                "bootstrap_ci_high": 0.2,
                "median_within_image_spearman": 0.1,
                "topk_median_realized_psnr_shift": 1.0,
                "class_coverage": 1.0,
                "global_spearman_score_sigma": 1.0,
                "global_spearman_score_corruption_mse": 1.0,
                "global_spearman_score_realized_psnr": -1.0,
                "topk_jaccard": 1.0,
                "class_distribution_max_abs_deviation": 0.0,
            }
        ]
    ).to_csv(ctx.stage_root("score_validate") / "candidate_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "report_timestep": 1,
                "code_timestep_index": 0,
                "beta_t": 0.1,
                "alpha_t": 0.9,
                "alpha_bar_t": 0.9,
                "sqrt_alpha_bar_t": 0.9,
                "sqrt_one_minus_alpha_bar_t": 0.1,
                "effective_diffusion_noise": 0.1,
            }
        ]
    ).to_csv(ctx.stage_root("score_validate") / "resolved_timestep_grid.csv", index=False)
    try:
        stage_score_select(ctx)
    except SystemExit:
        return
    raise AssertionError("score_select gate unexpectedly passed")


if __name__ == "__main__":
    deterministic_worker_check()
    failing_gate_check()
    print("standalone_smoke_checks=ok")
