"""Run ImageNet-100 corpus-noise replicates for seeds 43 and 44.

Only run creates experiment roots. Preflight checks data and GPU access.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

import numpy as np
import pandas as pd
import torch
from src import imagenet100_pipeline as pipe

FAMILY = "imagenet100_image_space_variable_awgn_v1"
BASE_ID = "imagenet100_image_space_variable_awgn_001"
STAGES = (
    "prepare", "ddpm_train", "score_validate", "score_select", "filter",
    "downstream_early_stopping_audit", "filter_grid", "tdncnn_grid",
    "drunet_grid", "evaluate_grid", "figures_grid", "summary_grid",
)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def differences(a, b, prefix=""):
    result = set()
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a.keys() | b.keys():
            result |= differences(a.get(k), b.get(k), f"{prefix}.{k}".strip("."))
    elif a != b:
        result.add(prefix)
    return result


def runtime_cfg(cfg, data):
    cfg = copy.deepcopy(cfg)
    cfg["dataset"]["root"] = str(data)
    cfg.update(_smoke_runtime=False, _real_smoke_runtime=False)
    return cfg


def context(cfg, path, root, run_id, resume=False):
    return pipe.PipelineContext(cfg, path.resolve(), run_id, root, False, False, False, resume)


def layout():
    output = Path(os.environ.get("OUTPUT_BASE", str(PROJECT / "experiments"))).resolve()
    base = Path(os.environ.get("SEED42_ROOT", str(output / FAMILY / BASE_ID))).resolve()
    data = Path(os.environ.get("DATA_ROOT", str(PROJECT / "data/imagenet100"))).resolve()
    items = []
    for seed in (43, 44):
        run_id = f"{BASE_ID}_seed{seed}"
        path = PROJECT / f"configs/experiments/imagenet100_seed{seed}.yaml"
        root = (output / FAMILY / run_id).resolve()
        require(root != base and root.parent == base.parent, "Roots must be distinct siblings of seed42")
        require(not (output / FAMILY / run_id).is_symlink(), "Symlink run root forbidden")
        cfg = runtime_cfg(pipe.read_yaml(path), data)
        items.append((seed, run_id, path, root, cfg))
    return base, data, items


def smoke(cfg):
    # Compare corpus noise using the same clean images and classes.
    dataset = pipe.SyntheticImageNet100(split="train", image_size=64, classes=3, samples_per_class=2, seed=20260809)
    for index in (0, 2, 5):
        clean, cls = dataset[index]
        outputs, base_noises, psnrs = [], [], []
        for seed in (42, 43, 44):
            kw = dict(dataset_name="imagenet100", split="train", original_index=index,
                      corpus_seed=seed, corruption_version=cfg["corpus"]["version"], psnr_min=10., psnr_max=40.)
            x, meta = pipe.deterministic_corruption(clean, **kw)
            again, meta2 = pipe.deterministic_corruption(clean, **kw)
            require(torch.equal(x, again) and meta == meta2, "Corruption not reproducible")
            require(x.shape == clean.shape and 10 <= meta["target_psnr_db"] <= 40, "Invalid corruption")
            require(meta["sigma"] == 10 ** (-meta["target_psnr_db"] / 20), "Invalid sigma")
            gen = torch.Generator().manual_seed(meta["sample_seed"])
            torch.rand((), generator=gen)
            base_noises.append(torch.randn(clean.shape, generator=gen, dtype=clean.dtype))
            outputs.append(x)
            psnrs.append(meta["target_psnr_db"])
        require(len(set(psnrs)) == 3, "PSNR realization did not change")
        for a, b in ((0, 1), (0, 2), (1, 2)):
            require(not torch.equal(outputs[a], outputs[b]), "Corpus realization did not change")
            require(not torch.equal(base_noises[a], base_noises[b]), "Base noise did not change")
        print(f"corruption_smoke index={index} class={cls} shape={tuple(clean.shape)} PSNRs={psnrs} PASS")


def audit(base, data, items):
    current = runtime_cfg(pipe.read_yaml(PROJECT / "configs/experiments/imagenet100_seed42.yaml"), data)
    require(current["seeds"]["corpus_seed"] == 42, "Production baseline seed must be 42")
    for seed, run_id, path, root, cfg in items:
        require(differences(current, cfg) == {"seeds.corpus_seed", "provenance.allow_downstream_reuse_with_mismatched_config_hash"}, "Unexpected replicate config differences")
        require(cfg["seeds"]["corpus_seed"] == seed, "Wrong corpus seed")
        require(cfg["provenance"]["allow_downstream_reuse_with_mismatched_config_hash"] is False, "Unsafe provenance")
        pipe.validate_current_protocol("ImageNet-100", cfg, downstream_models=["tdncnn", "drunet"])
        print(json.dumps(dict(run_id=run_id, corpus_seed=seed, config=str(path), experiment_root=str(root),
                              data_root=str(data), config_hash=context(cfg, path, root, run_id).cfg_hash)))
    require(pipe.config_hash(items[0][4]) != pipe.config_hash(items[1][4]), "Seed absent from config hash")
    require(items[0][3] != items[1][3], "Replicate roots collide")
    print("stage_graph=" + " -> ".join(STAGES))
    print("external=False SIDD=False seed42_root=" + str(base))
    smoke(current)


def code_hash():
    # Include local source changes in the run hash.
    paths = sorted([*PROJECT.glob("src/**/*.py"), *PROJECT.glob("scripts/remote/*noise_seed*"),
                    PROJECT / "scripts/run_imagenet100_pipeline.py", PROJECT / "requirements.txt"])
    return pipe.config_hash({str(p.relative_to(PROJECT)): pipe.sha256_file(p) for p in paths if p.is_file()})


def ownership(item, data):
    seed, run_id, path, root, cfg = item
    return dict(run_id=run_id, corpus_seed=seed, experiment_root=str(root), data_root=str(data),
                config_hash=context(cfg, path, root, run_id).cfg_hash, code_hash=code_hash())


def check_root(item, data, resume):
    _, run_id, path, root, cfg = item
    if not root.exists() or not any(root.iterdir()):
        return
    require(resume, f"First run requires empty root: {root}")
    require(not any(p.is_symlink() for p in root.rglob("*")), f"Symlink artifacts forbidden: {root}")
    owner = root / "00_protocol/noise_replicate_owner.json"
    require(owner.exists(), f"Unowned artifacts; refusing reuse: {root}")
    require(json.loads(owner.read_text(encoding="utf-8")) == ownership(item, data), f"Resume identity/config/code mismatch: {root}")
    resolved = root / "00_protocol/resolved_config.yaml"
    require(resolved.exists() and pipe.read_yaml(resolved) == cfg, "Resume resolved config mismatch")
    expected = context(cfg, path, root, run_id).cfg_hash
    for marker in root.rglob("_SUCCESS.json"):
        payload = json.loads(marker.read_text(encoding="utf-8"))
        require(payload.get("run_id") == run_id and payload.get("config_hash") == expected, f"Foreign/stale marker: {marker}")
    selected = root / "04_selected_score_config/selected_score_config.yaml"
    if selected.exists():
        payload = pipe.read_yaml(selected)
        require(payload.get("run_id") == run_id and payload.get("config_hash") == expected, "Foreign selected score config")
        ckpt = Path(payload["checkpoint_path"]).resolve()
        require(ckpt.is_relative_to(root / "02_ddpm"), "Selected checkpoint outside own root")


def preflight(base, data, items, resume):
    require(base.is_dir(), f"Seed42 root missing: {base}")
    for rel in ("00_protocol/resolved_config.yaml", "01_corpus/corpus_manifest.csv", "09_summary/final_summary.md"):
        require((base / rel).is_file(), f"Missing seed42 artifact: {rel}")
    old = pipe.read_yaml(base / "00_protocol/resolved_config.yaml")
    cfg = items[0][4]
    require(old["seeds"] == {**cfg["seeds"], "corpus_seed": 42}, "Scientific seeds differ from seed42")
    for key in ("topk_ratio", "random_ratio", "quantile_ratio", "quantile_min_points_per_bin", "include_topk5", "full_train_topk_requires_all_classes"):
        require(old["filtering"][key] == cfg["filtering"][key], f"Filtering differs: {key}")
    for key in ("corpus", "ddpm", "score_validation", "score_selection", "evaluation"):
        require(old[key] == cfg[key], f"Protocol differs from seed42: {key}")
    # Check the final training metadata for all branches.
    for model in ("tdncnn", "drunet"):
        for variant in ("full", "topk_p10", "topk_p20", "topk_p30", "quantile_p10", "quantile_p20", "quantile_p30", "random_p10", "random_p20", "random_p30"):
            meta = json.loads((base / f"06_downstream/{model}/{variant}/run_meta.json").read_text(encoding="utf-8"))
            protocol = meta["training_protocol"]
            ctx = context(cfg, items[0][2], items[0][3], items[0][1])
            proposed = pipe.downstream_training_protocol(ctx, model, variant)
            for key in ("subset_indices_sha256", "initial_state_sha256"):
                proposed.pop(key); protocol.pop(key)
            require(proposed == protocol, f"Final seed42 training protocol differs: {model}/{variant}")
    require(torch.cuda.is_available(), "CUDA unavailable")
    print("GPU=" + torch.cuda.get_device_name(0))
    train = pipe.build_dataset(cfg, split="train", smoke=False)
    val = pipe.build_dataset(cfg, split="val", smoke=False)
    require(len(train.classes) == len(val.classes) == 100 and train.classes == val.classes, "Expected same 100 classes")
    manifest = pd.read_csv(base / "01_corpus/corpus_manifest.csv")
    require(len(train) == len(manifest), "Train dataset differs from seed42")
    summary = json.loads((base / "00_protocol/protocol_summary.json").read_text(encoding="utf-8"))
    require(len(val) == summary["val_samples"], "Val dataset differs from seed42")
    expected_counts = {int(k): int(v) for k, v in summary["val_class_counts"].items()}
    require(pipe.split_class_counts(val) == expected_counts, "Val class counts differ")
    for row in manifest.itertuples():
        index = int(row.original_index)
        require(train.targets[index] == int(row.class_id), "Train class/index drift")
        if hasattr(row, "relative_path"):
            require(pipe.sample_identifier(train, index, "train") == row.relative_path, "Train path/index drift")
    indices = pipe.class_stratified_indices(val, per_class=int(cfg["score_validation"]["samples_per_class"]), seed=int(cfg["seeds"]["split_seed"]) + 1009)
    require(np.array_equal(indices, np.load(base / "00_protocol/downstream_evaluation_indices.npy")), "Evaluation indices differ from seed42")
    eval_manifest = pd.read_csv(base / "07_evaluation/combined/eval_noise_manifest.csv")
    require(eval_manifest.original_index.tolist() == indices.tolist(), "Seed42 evaluation manifest indices differ")
    require(np.allclose(eval_manifest.sigma, cfg["evaluation"]["eval_noise"]["sigma"]), "Evaluation sigma differs")
    for row in eval_manifest.itertuples():
        n = cfg["evaluation"]["eval_noise"]
        expected = pipe.stable_int_seed("imagenet100", "eval_noise", int(row.original_index), n["seed"], n["version"])
        require(int(row.sample_seed) == expected and val.targets[int(row.original_index)] == row.class_id, "Evaluation noise/class differs")
    parent = items[0][3].parent
    require(parent.is_dir() and os.access(parent, os.W_OK), f"Output parent not writable: {parent}")
    # Reserve space for checkpoints and evaluation outputs.
    free = shutil.disk_usage(parent).free / 1024**3
    minimum = float(os.environ.get("MIN_FREE_GB", "200"))
    require(free >= minimum, f"Only {free:.1f} GiB free; require {minimum} GiB for both runs")
    for item in items:
        check_root(item, data, resume)
    print(f"preflight=PASS train={len(train)} val={len(val)} classes=100 disk_free_GiB={free:.1f} required_GiB={minimum}")


def status(items):
    for seed, run_id, _, root, _ in items:
        print(f"\nseed={seed} run_id={run_id} root={root}")
        state = root / "00_protocol/noise_replicate_status.json"
        if state.exists():
            payload = json.loads(state.read_text(encoding="utf-8"))
            print(json.dumps(payload))
            if payload["status"] == "RUNNING" and os.environ.get("STATUS_EXTERNAL_CONTAINER") != "1":
                try:
                    os.kill(payload["pid"], 0)
                    print("launcher_pid_alive=True (see timestamp/log for progress)")
                except ProcessLookupError:
                    print("INTERRUPTED: launcher PID is absent; RESUME=1 required")
        else:
            print("NOT_STARTED")
        log = root / "00_protocol/logs" / f"imagenet100_seed{seed}.log"
        if log.exists():
            from collections import deque
            with log.open(encoding="utf-8", errors="replace") as handle:
                print("".join(deque(handle, maxlen=15)))


def run(items, data, resume):
    for item in items:
        seed, run_id, path, root, cfg = item
        check_root(item, data, resume)
        protocol = root / "00_protocol"
        protocol.mkdir(parents=True, exist_ok=True)
        pipe.write_json(protocol / "noise_replicate_owner.json", ownership(item, data))
        pipe.save_yaml(protocol / "resolved_config.yaml", cfg)
        log = protocol / "logs" / f"imagenet100_seed{seed}.log"
        log.parent.mkdir(exist_ok=True)
        state_path = protocol / "noise_replicate_status.json"
        def state(stage, value, **extra):
            temporary = state_path.with_suffix(".tmp")
            pipe.write_json(temporary, dict(seed=seed, stage=stage, status=value, timestamp=dt.datetime.now(dt.timezone.utc).isoformat(), pid=os.getpid(), **extra))
            temporary.replace(state_path)
        with log.open("a", buffering=1, encoding="utf-8") as handle:
            header = dict(**ownership(item, data), config_path=str(path), git_commit=os.environ.get("PROJECT_GIT_COMMIT") or pipe.git_commit(),
                          GPU=torch.cuda.get_device_name(0), timestamp=dt.datetime.now(dt.timezone.utc).isoformat(), resume=resume)
            handle.write(json.dumps(header) + "\n")
            for stage in STAGES:
                state(stage, "RUNNING")
                handle.write(f"\n===== {stage} =====\n")
                cmd = [sys.executable, "-u", str(PROJECT / "scripts/run_imagenet100_pipeline.py"),
                       "--config", str(path), "--run-id", run_id, "--experiment-root", str(root),
                       "--data-root", str(data), "--stage", stage]
                if resume:
                    cmd.append("--resume")
                try:
                    subprocess.run(cmd, check=True, stdout=handle, stderr=subprocess.STDOUT, cwd=PROJECT)
                    marker = context(cfg, path, root, run_id).marker_path(stage)
                    require(marker.is_file(), f"Stage returned without success marker: {stage}")
                    outputs = json.loads(marker.read_text(encoding="utf-8")).get("output_artifact_list", [])
                    require(bool(outputs) and all(Path(p).is_file() for p in outputs), f"Missing stage outputs: {stage}")
                    check_root(item, data, True)
                except BaseException as exc:
                    state(stage, "FAILED", error=str(exc))
                    raise
            state("summary_grid", "COMPLETE")
            print(f"seed={seed} COMPLETE log={log}", flush=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "preflight"
    require(mode in {"audit", "dry-run", "preflight", "run", "status"}, "Unknown mode")
    raw_resume = os.environ.get("RESUME", "0")
    require(raw_resume in {"0", "1"}, "RESUME must be 0 or 1")
    resume = raw_resume == "1"
    base, data, items = layout()
    if mode == "status":
        status(items)
        return
    audit(base, data, items)
    if mode == "dry-run":
        for seed, run_id, path, root, cfg in items:
            ctx = context(cfg, path, root, run_id, resume)
            ctx.dry_run = True
            for stage in STAGES:
                # Skip audit because it writes files even when ctx.dry_run is set.
                if stage == "downstream_early_stopping_audit":
                    print("[dry-run] would audit downstream early stopping")
                else:
                    pipe.run_stage(ctx, stage)
        return
    if mode in {"preflight", "run"}:
        preflight(base, data, items, resume)
    if mode == "run":
        run(items, data, resume)


if __name__ == "__main__":
    main()
