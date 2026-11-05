"""Run CIFAR-10 corpus-noise replicates with resume checks.

Only run writes experiment roots or starts training.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from src import cifar10_pipeline as pipe

BASE_ID = 'cifar10_001_final_20260901'
BASE_ROOT = PROJECT / 'experiments/cifar10' / BASE_ID
STAGES = ['prepare', 'ddpm_train', 'score_validate', 'score_select', 'filter_grid',
          *pipe.expanded_stages('tdncnn_grid'), *pipe.expanded_stages('drunet_grid'),
          'downstream_early_stopping_audit', 'evaluate_grid', 'figures_grid', 'summary_grid']


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def layout(output=None):
    output = Path(output or PROJECT / 'experiments/cifar10').resolve()
    items = []
    baseline = pipe.read_yaml(PROJECT / 'configs/experiments/cifar10_seed42.yaml')
    for seed in (43, 44):
        path = PROJECT / f'configs/experiments/cifar10_seed{seed}.yaml'
        cfg = pipe.read_yaml(path)
        expected = copy.deepcopy(baseline)
        expected['seeds']['corpus_seed'] = seed
        run_id = f'{BASE_ID}_seed{seed}'
        expected['experiment']['run_id'] = run_id
        require(cfg == expected, 'Only corpus_seed and run_id metadata may differ')
        raw_root = output / run_id
        root = raw_root.resolve()
        require(not raw_root.is_symlink() and root.parent == output and root != BASE_ROOT.resolve(), 'Unsafe root')
        items.append((seed, run_id, path, root, cfg))
    return items


def scientific_args(item, resume=False):
    _, run_id, path, root, _ = item
    return argparse.Namespace(config=path, run_id=run_id, experiment_root=root, data_root=None,
                              stage='', resume=resume, dry_run=False, smoke=False, device='cuda')


def validation_config(item):
    args = scientific_args(item)
    va = pipe.validation_args(item[4], args, item[3], 'prepare')
    return pipe.cfg_to_dict(va, pipe.load_dataset_config('cifar10', PROJECT))


def identity(item):
    seed, run_id, path, root, cfg = item
    source = {p.relative_to(PROJECT).as_posix(): pipe.sha256_file(p)
              for p in sorted((PROJECT / 'src').rglob('*.py'))}
    for p in (Path(__file__), PROJECT / 'scripts/run_cifar10_pipeline.py', PROJECT / 'configs/dataset/cifar10.yaml', PROJECT / 'requirements.txt'):
        source[p.relative_to(PROJECT).as_posix()] = pipe.sha256_file(p)
    return dict(run_id=run_id, corpus_seed=seed, root=str(root), config_hash=digest(cfg),
                runtime_config_hash=digest(validation_config(item)), source_hash=digest(source))


def guard(item, resume):
    root, cfg = item[3:]
    if not root.exists() or not any(root.iterdir()):
        return
    require(resume, f'First launch requires an empty root: {root}')
    require(not any(p.is_symlink() for p in root.rglob('*')), 'Symlink artifacts forbidden')
    owner = root / '00_protocol/noise_replicate_owner.json'
    require(owner.exists(), f'Foreign/unowned artifacts: {root}')
    require(json.loads(owner.read_text(encoding='utf-8')) == identity(item), 'Resume config/source/root identity mismatch')
    require(pipe.read_yaml(root / '00_protocol/resolved_config.yaml') == cfg, 'Resolved config mismatch')
    selected = root / '05_selected_configuration/selected_score_config.yaml'
    if selected.exists():
        data = pipe.read_yaml(selected)
        ckpt = Path(data['selected_ddpm_checkpoint_path']).resolve()
        require(ckpt.is_relative_to(root / '03_ddpm'), 'Foreign selected DDPM checkpoint')
        require(ckpt.is_file() and pipe.sha256_file(ckpt) == data['selected_ddpm_checkpoint_sha256'], 'Selected checkpoint hash mismatch')


def smoke(vcfg):
    class Tiny:
        def __getitem__(self, index):
            return torch.full((3, 32, 32), .5), index % 10
    with tempfile.TemporaryDirectory(prefix='cifar_noise_smoke_') as temporary:
        results = {}
        for seed in (42, 43, 44):
            cfg = copy.deepcopy(vcfg)
            cfg['seeds']['corpus_seed'] = seed
            for repeat in (0, 1):
                folder = Path(temporary) / f'{seed}_{repeat}'
                folder.mkdir()
                pipe.materialize_corpus(Tiny(), np.array([0, 17, 49999]), folder, cfg)
                x = torch.load(folder / 'train_corpus.pt', map_location='cpu', weights_only=False)
                manifest = pd.read_csv(folder / 'train_manifest.csv')
                if repeat:
                    require(torch.equal(results[seed][0], x), 'Corruption is not reproducible')
                else:
                    results[seed] = x, manifest
        for a, b in ((42, 43), (42, 44), (43, 44)):
            xa, ma = results[a]; xb, mb = results[b]
            require(xa.shape == xb.shape and ma.class_id.equals(mb.class_id), 'Shape/class changed')
            require(not torch.equal(xa, xb), 'Corpus noise did not change')
            require(np.all(ma.target_psnr_db != mb.target_psnr_db) and np.all(ma.sigma != mb.sigma), 'PSNR/sigma did not change')
            require(np.allclose(ma.sigma, 10 ** (-ma.target_psnr_db / 20)), 'Sigma protocol changed')
    print('corruption_smoke=PASS seed42/43/44 differ; repeat is bitwise identical', flush=True)


def audit(items):
    for item in items:
        print(json.dumps(dict(**identity(item), config=str(item[2])), ensure_ascii=False), flush=True)
        pipe.validate_current_protocol('CIFAR-10', item[4], downstream_models=['tdncnn', 'drunet'])
    require(identity(items[0])['config_hash'] != identity(items[1])['config_hash'], 'Seeds missing from hash')
    print('stage_graph=' + ' -> '.join(STAGES), flush=True)
    print('external=False SIDD=False baseline=' + str(BASE_ROOT), flush=True)
    smoke(validation_config(items[0]))


def preflight(items, resume, minimum):
    require(BASE_ROOT.is_dir(), 'Seed42 root missing')
    old = pipe.read_yaml(BASE_ROOT / '00_protocol/resolved_score_validation_config.yaml')
    proposed = validation_config(items[0])
    for key in ('split_sizes', 'target_psnr_db', 'psnr_levels', 'ddpm', 'scan', 'num_workers'):
        require(old[key] == proposed[key], f'Seed42 protocol differs: {key}')
    require(old['seeds'] == {**proposed['seeds'], 'corpus_seed': 42}, 'Seed42 scientific seeds differ')
    require(torch.cuda.is_available(), 'Local CUDA is unavailable')
    print('GPU=' + torch.cuda.get_device_name(0), flush=True)
    dcfg = proposed['dataset_cfg']
    train = pipe.load_cifar10_train(dcfg, download=False)
    test = pipe.build_torchvision_split(dataset_cfg=dcfg, train=False,
                                      data_root=Path(dcfg['paths']['data_root']), transform_profile='tensor', download=False)
    require(len(train) == 50000 and len(test) == 10000 and len(train.classes) == 10, 'Unexpected CIFAR data')
    splits = pipe.stratified_three_way_split(pipe.class_labels(train), proposed['split_sizes'], old['seeds']['split_seed'])
    for name, indices in splits.items():
        require(np.array_equal(indices, np.load(BASE_ROOT / f'00_protocol/{name}_original_indices.npy')), 'Split indices differ')
    cfg = items[0][4]
    for model, stage_map in (('tdncnn', pipe.TDNCNN_SUBSETS), ('drunet', pipe.DRUNET_SUBSETS)):
        for stage in pipe.expanded_stages(model + '_grid'):
            subset = stage_map[stage]
            meta_path = pipe.downstream_meta_path(BASE_ROOT, model, subset)
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            current = cfg['downstream'][model]
            for field, value in (('batch_size', current['batch_size']), ('max_optimizer_steps', current['max_optimizer_steps']),
                                 ('validation_interval_steps', current['validation_interval_steps'])):
                require(meta[field] == value, f'Downstream protocol differs: {model}/{subset}/{field}')
            for k, v in current['early_stopping'].items():
                require(meta['early_stopping_' + k] == v, f'Early stopping differs: {model}/{subset}/{k}')
    cache = BASE_ROOT / '08_downstream_commands/eval_noise_cache'
    manifest = pd.read_csv(cache / 'test_noise_manifest.csv')
    require(len(manifest) == 10000 and manifest.original_index.tolist() == list(range(10000)), 'Evaluation must cover all test images')
    noise = cfg['evaluation']['eval_noise']
    require(set(manifest.eval_seed) == {noise['seed']}, 'Eval seed changed')
    targets = torch.load(cache / 'test_eval_target.pt', map_location='cpu', weights_only=False)
    inputs = torch.load(cache / 'test_eval_input.pt', map_location='cpu', weights_only=False)
    for index in (0, 17, 9999):
        clean, cls = test[index]
        row = manifest.iloc[index]
        version = str(row.eval_noise_version)
        require(version == 'cifar10_internal_eval_awgn_v1', 'Unexpected eval version')
        seed = pipe.stable_int_seed('cifar10', 'test_eval_noise', index, noise['seed'], version)
        gen = torch.Generator().manual_seed(seed)
        sigma = noise['sigma_min'] + (noise['sigma_max'] - noise['sigma_min']) * torch.rand((), generator=gen).item()
        x = (clean + torch.randn(clean.shape, generator=gen, dtype=clean.dtype) * sigma).clamp(0, 1)
        require(int(row.sample_seed) == seed and int(row.class_id) == cls, 'Eval indices/seeds differ')
        require(torch.equal(clean, targets[index]) and torch.equal(x, inputs[index]), 'Evaluation realization differs from seed42')
    parent = items[0][3].parent
    require(parent.is_dir() and os.access(parent, os.W_OK), 'Output parent is not writable')
    free = shutil.disk_usage(parent).free / 1024**3
    require(free >= minimum, f'Insufficient disk: {free:.1f} GiB < {minimum} GiB')
    for item in items:
        guard(item, resume)
    print(f'preflight=PASS train=50000 test=10000 splits=45000/2500/2500 disk_free_GiB={free:.1f}', flush=True)


def outputs(root, stage):
    simple = {
        'prepare': ['01_corpus/train_corpus.pt', '01_corpus/train_manifest.csv',
                    '00_protocol/ddpm_filter_train_original_indices.npy', '00_protocol/score_tuning_original_indices.npy',
                    '00_protocol/score_confirmation_original_indices.npy', '02_paired_banks/score_tuning_paired_bank.pt',
                    '02_paired_banks/score_confirmation_paired_bank.pt'],
        'ddpm_train': ['03_ddpm/checkpoints/epoch_040.pt', '03_ddpm/ddpm_epoch_metrics.csv'],
        'score_validate': ['04_timestep_scan/score_validation_metrics.csv'],
        'score_select': ['05_selected_configuration/selected_score_config.yaml', '06_confirmation/PASS.marker'],
        'filter_grid': ['05_filtering/scores_full.csv', '08_downstream_commands/eval_noise_cache/test_eval_input.pt',
                        '08_downstream_commands/eval_noise_cache/test_eval_target.pt',
                        '08_downstream_commands/eval_noise_cache/test_noise_manifest.csv',
                        '08_downstream_commands/topk_p30_indices.npy', '08_downstream_commands/quantile_spread_p30_indices.npy',
                        '08_downstream_commands/random_p30_indices.npy'],
        'downstream_early_stopping_audit': ['00_protocol/downstream_early_stopping_audit/downstream_early_stopping_audit.csv'],
        'evaluate_grid': ['07_evaluation/combined/subset_grid_results_all.csv'],
        'figures_grid': ['08_figures/tdncnn_subset_fraction_vs_psnr.png', '08_figures/drunet_subset_fraction_vs_psnr.png'],
        'summary_grid': ['09_summary/final_summary.md'],
    }
    if stage in simple:
        return [root / p for p in simple[stage]]
    model = 'tdncnn' if stage in pipe.TDNCNN_SUBSETS else 'drunet'
    subset = (pipe.TDNCNN_SUBSETS if model == 'tdncnn' else pipe.DRUNET_SUBSETS)[stage]
    return [pipe.downstream_checkpoint_path(root, model, subset), pipe.downstream_meta_path(root, model, subset)]


def execute_stage(item, stage, resume):
    # Allow the pipeline to open the run root after the launcher writes metadata.
    args = scientific_args(item, resume=True)
    pipe.run_stage(item[4], args, item[3], stage)


def run(items, resume):
    for item in items:
        seed, run_id, path, root, cfg = item
        guard(item, resume)
        protocol = root / '00_protocol'
        protocol.mkdir(parents=True, exist_ok=True)
        write_json(protocol / 'noise_replicate_owner.json', identity(item))
        OmegaConf.save(OmegaConf.create(cfg), protocol / 'resolved_config.yaml')
        logs = protocol / 'logs'
        logs.mkdir(exist_ok=True)
        with (logs / f'cifar10_seed{seed}.log').open('a', encoding='utf-8', buffering=1) as log:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip()
            log.write(json.dumps(dict(**identity(item), config=str(path), data=validation_config(item)['dataset_cfg']['paths']['data_root'],
                                      git_commit=commit, GPU=torch.cuda.get_device_name(0), timestamp=dt.datetime.now().isoformat())) + '\n')
            for stage in STAGES:
                marker = protocol / 'noise_stages' / f'{stage}.json'
                if resume and marker.exists():
                    saved = json.loads(marker.read_text(encoding='utf-8'))
                    require(saved['identity'] == identity(item), 'Foreign stage marker')
                    require(all(p.is_file() for p in outputs(root, stage)), f'Missing completed outputs: {stage}')
                    require(all(pipe.sha256_file(root / p) == h for p, h in saved['outputs_sha256'].items()), f'Stale/corrupt stage outputs: {stage}')
                    print(f'[resume] seed={seed} skip={stage}', flush=True)
                    continue
                # Archive incomplete branch outputs so resume restarts training.
                if stage in pipe.TDNCNN_SUBSETS or stage in pipe.DRUNET_SUBSETS:
                    branch = outputs(root, stage)[0].parent
                    if branch.exists():
                        archive = protocol / 'interrupted' / f'{stage}_{time.time_ns()}'
                        archive.parent.mkdir(exist_ok=True)
                        require(branch.resolve().is_relative_to(root.resolve()), 'Unsafe branch archive')
                        shutil.move(str(branch), str(archive))
                state = dict(seed=seed, stage=stage, status='RUNNING', pid=os.getpid(), timestamp=dt.datetime.now().isoformat())
                write_json(protocol / 'noise_replicate_status.json', state)
                log.write(f'\n===== seed={seed} stage={stage} =====\n')
                print(f'seed={seed} stage={stage} log={logs}', flush=True)
                cmd = [sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--seed', str(seed),
                       '--output-base', str(root.parent), '--worker-stage', stage]
                try:
                    subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=PROJECT)
                    guard(item, True)
                    require(all(p.is_file() for p in outputs(root, stage)), f'Stage outputs missing: {stage}')
                    hashes = {p.relative_to(root).as_posix(): pipe.sha256_file(p) for p in outputs(root, stage)}
                    write_json(marker, dict(identity=identity(item), stage=stage, completed=dt.datetime.now().isoformat(), outputs_sha256=hashes))
                except BaseException as error:
                    write_json(protocol / 'noise_replicate_status.json', {**state, 'status': 'FAILED', 'error': str(error)})
                    raise
            write_json(protocol / 'noise_replicate_status.json', dict(seed=seed, stage=STAGES[-1], status='COMPLETE',
                       pid=os.getpid(), timestamp=dt.datetime.now().isoformat()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['audit', 'preflight', 'dry-run', 'run', 'status', 'worker'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output-base', type=Path)
    parser.add_argument('--min-free-gb', type=float, default=20.)
    parser.add_argument('--seed', type=int, choices=[43,44])
    parser.add_argument('--worker-stage', choices=STAGES)
    args = parser.parse_args()
    items = layout(args.output_base)
    if args.action == 'worker':
        require(args.seed is not None and args.worker_stage is not None, 'Worker seed/stage required')
        item = next(i for i in items if i[0] == args.seed)
        guard(item, True)
        execute_stage(item, args.worker_stage, True)
        return
    if args.action == 'status':
        for seed, run_id, _, root, _ in items:
            state = root / '00_protocol/noise_replicate_status.json'
            print(run_id, state.read_text(encoding='utf-8') if state.exists() else 'NOT_STARTED')
        return
    audit(items)
    if args.action == 'dry-run':
        for item in items:
            a = scientific_args(item)
            a.dry_run = True
            for stage in STAGES:
                pipe.run_stage(item[4], a, item[3], stage)
        return
    if args.action in ('preflight', 'run'):
        preflight(items, args.resume, args.min_free_gb)
    if args.action == 'run':
        # The OS releases the lock when the process exits.
        import msvcrt
        lock_path = items[0][3].parent / '.cifar10_noise_replicates.lock'
        with lock_path.open('a+b') as lock:
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('Another CIFAR replicate launcher is running') from exc
            run(items, args.resume)


if __name__ == '__main__':
    main()
