from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src import imagenet100_pipeline as pipe


def base_cfg(tmp_path: Path) -> dict:
    cfg = OmegaConf.to_container(
            OmegaConf.load(Path("configs/experiments/imagenet100.yaml")),
        resolve=True,
    )
    cfg["dataset"]["type"] = "synthetic"
    cfg["dataset"]["root"] = str(tmp_path / "synthetic")
    cfg["runtime"]["num_workers"] = 0
    cfg["smoke"]["classes"] = 4
    cfg["smoke"]["samples_per_class"] = 4
    cfg["smoke"]["oracle_scores"] = True
    cfg["ddpm"]["base_dim"] = 4
    cfg["ddpm"]["depth"] = 1
    cfg["ddpm"]["T"] = 10
    cfg["score_validation"]["timestep_scan"]["report_timesteps"] = [1, 100, 500]
    cfg["score_validation"]["bootstrap_iters_smoke"] = 8
    return cfg


def make_ctx(tmp_path: Path, cfg: dict | None = None) -> pipe.PipelineContext:
    cfg = cfg or base_cfg(tmp_path)
    return pipe.PipelineContext(
        cfg=cfg,
        config_path=tmp_path / "config.yaml",
        run_id="test",
        experiment_root=tmp_path / "exp",
        smoke=True,
        real_smoke=False,
        dry_run=False,
        resume=False,
    )


def test_deterministic_corruption_repeat_and_workers(tmp_path: Path) -> None:
    cfg = base_cfg(tmp_path)
    ds = pipe.build_dataset(cfg, split="train", smoke=True)
    rows = []
    for idx in range(8):
        x, cls = ds[idx]
        _, metrics = pipe.deterministic_corruption(
            x,
            dataset_name=cfg["dataset"]["name"],
            split="train",
            original_index=idx,
            corpus_seed=cfg["seeds"]["corpus_seed"],
            corruption_version=cfg["corpus"]["version"],
            psnr_min=cfg["corpus"]["target_psnr_min_db"],
            psnr_max=cfg["corpus"]["target_psnr_max_db"],
        )
        rows.append({"original_index": idx, "class_id": cls, "class_name": str(cls), "relative_path": str(idx), **metrics})
    manifest = pd.DataFrame(rows)
    corpus = pipe.OnDemandCorpusDataset(ds, manifest, return_index=True)
    a = corpus[3][0]
    b = corpus[3][0]
    assert torch.equal(a, b)

    loader = DataLoader(corpus, batch_size=2, shuffle=False, num_workers=2)
    sums = {}
    for x, original, _cls in loader:
        for item, idx in zip(x, original):
            sums[int(idx)] = float(item.sum())
    assert pytest.approx(float(a.sum())) == sums[3]


def test_manifest_metrics_recompute(tmp_path: Path) -> None:
    cfg = base_cfg(tmp_path)
    ds = pipe.build_dataset(cfg, split="train", smoke=True)
    x, _ = ds[0]
    x_corpus, metrics = pipe.deterministic_corruption(
        x,
        dataset_name=cfg["dataset"]["name"],
        split="train",
        original_index=0,
        corpus_seed=cfg["seeds"]["corpus_seed"],
        corruption_version=cfg["corpus"]["version"],
        psnr_min=cfg["corpus"]["target_psnr_min_db"],
        psnr_max=cfg["corpus"]["target_psnr_max_db"],
    )
    mse = float(((x_corpus - x) ** 2).mean())
    assert pytest.approx(pipe.psnr_from_mse(mse), rel=1e-6) == metrics["realized_psnr_db_postclip"]
    assert pytest.approx(mse, rel=1e-6) == metrics["corruption_mse"]


def test_timestep_grid_and_score_shape(tmp_path: Path) -> None:
    cfg = base_cfg(tmp_path)
    model = pipe.build_ddpm(cfg, torch.device("cpu"))
    grid = pipe.timestep_grid(cfg, model)
    assert grid[0]["report_timestep"] == 1
    assert grid[0]["code_timestep_index"] == 0
    assert all(0 <= row["code_timestep_index"] < cfg["ddpm"]["T"] for row in grid)
    x = torch.rand(3, 3, cfg["dataset"]["image_size"], cfg["dataset"]["image_size"])
    score, std = pipe.score_batch(
        model,
        x,
        original_indices=np.asarray([0, 1, 2], dtype=np.int64),
        code_timestep_index=0,
        seed_bank=[1, 2],
    )
    assert score.shape == (3,)
    assert std.shape == (3,)
    assert np.isfinite(score).all()


def test_ladder_uses_one_base_noise_direction(tmp_path: Path) -> None:
    cfg = base_cfg(tmp_path)
    x = torch.full((3, cfg["dataset"]["image_size"], cfg["dataset"]["image_size"]), 0.5)
    bank, rows = pipe.ladder_corruption(
        x,
        dataset_name=cfg["dataset"]["name"],
        split="val",
        original_index=0,
        ladder_seed=cfg["seeds"]["ladder_seed"],
        psnr_levels=[20.0, 30.0],
    )
    n0 = (bank[0] - x) / rows[0]["sigma"]
    n1 = (bank[1] - x) / rows[1]["sigma"]
    assert torch.allclose(n0, n1, atol=1e-5)


def test_score_select_pass_and_fail(tmp_path: Path) -> None:
    cfg = base_cfg(tmp_path)
    ctx = make_ctx(tmp_path, cfg)
    (ctx.stage_root("score_validate")).mkdir(parents=True)
    (ctx.stage_root("prepare")).mkdir(parents=True)
    (ctx.stage_root("ddpm_train") / "checkpoints").mkdir(parents=True)
    pd.DataFrame({"original_index": [0], "sample_seed": [1]}).to_csv(ctx.stage_root("prepare") / "corpus_manifest.csv", index=False)
    torch.save({"model_state_dict": {}}, ctx.stage_root("ddpm_train") / "checkpoints" / "epoch_001.pt")
    pd.DataFrame(
        [
            {
                "checkpoint_epoch": 1,
                "report_timestep": 1,
                "code_timestep_index": 0,
                "bootstrap_ci_low": 0.1,
                "bootstrap_ci_high": 1.0,
                "median_within_image_spearman": 1.0,
                "topk_median_realized_psnr_shift": 1.0,
                "class_coverage": 0.25,
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
    pipe.stage_score_select(ctx)
    assert (ctx.stage_root("score_select") / "selected_score_config.yaml").exists()

    bad = make_ctx(tmp_path / "bad", cfg)
    (bad.stage_root("score_validate")).mkdir(parents=True)
    (bad.stage_root("prepare")).mkdir(parents=True)
    (bad.stage_root("ddpm_train") / "checkpoints").mkdir(parents=True)
    pd.DataFrame({"original_index": [0], "sample_seed": [1]}).to_csv(bad.stage_root("prepare") / "corpus_manifest.csv", index=False)
    torch.save({"model_state_dict": {}}, bad.stage_root("ddpm_train") / "checkpoints" / "epoch_001.pt")
    pd.DataFrame([{**pd.read_csv(ctx.stage_root("score_validate") / "candidate_metrics.csv").iloc[0].to_dict(), "bootstrap_ci_low": -0.1}]).to_csv(
        bad.stage_root("score_validate") / "candidate_metrics.csv",
        index=False,
    )
    pd.read_csv(ctx.stage_root("score_validate") / "resolved_timestep_grid.csv").to_csv(bad.stage_root("score_validate") / "resolved_timestep_grid.csv", index=False)
    with pytest.raises(SystemExit):
        pipe.stage_score_select(bad)


def test_filter_subsets_and_quantile_reference(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = base_cfg(tmp_path)
    ctx = make_ctx(tmp_path, cfg)
    pipe.stage_prepare(ctx)
    (ctx.stage_root("score_select")).mkdir(parents=True)
    manifest = ctx.stage_root("prepare") / "corpus_manifest.csv"
    selected = {
        "config_hash": ctx.cfg_hash,
        "corpus_manifest_hash": pipe.sha256_file(manifest),
        "checkpoint_epoch": 1,
        "report_timestep": 1,
        "code_timestep_index": 0,
        "mc_k": 2,
        "mc_seed_bank": [1, 2],
    }
    pipe.save_yaml(ctx.stage_root("score_select") / "selected_score_config.yaml", selected)
    pipe.write_json(ctx.marker_path("score_select"), {"run_id": ctx.run_id, "config_hash": ctx.cfg_hash})
    called = {"value": False}

    def fake_quantile(df, n_subset, *, seed, min_points_per_bin):
        called["value"] = True
        return df.sort_values(["score_value", "original_index"]).tail(n_subset)["original_index"].to_numpy(dtype=np.int64)

    monkeypatch.setattr(pipe, "select_quantile_spread_exact", fake_quantile)
    pipe.stage_filter(ctx)
    assert called["value"]
    topk = np.load(ctx.stage_root("filter") / "subset_indices" / "topk_10_indices.npy")
    quantile = np.load(ctx.stage_root("filter") / "subset_indices" / "quantile_10_indices.npy")
    random = np.load(ctx.stage_root("filter") / "subset_indices" / "random_10_indices.npy")
    assert len(topk) == len(quantile) == len(random)
    scores = pd.read_csv(ctx.stage_root("filter") / "scores_full.csv")
    expected = scores.sort_values(["score", "original_index"]).head(len(topk))["original_index"].to_numpy()
    assert np.array_equal(np.sort(topk), np.sort(expected))
