"""Run Colored MNIST corpus-noise replicates after checking baseline files.
"""
from __future__ import annotations
import argparse, copy, datetime as dt, inspect, json, os, shutil, subprocess, sys, tempfile, time
from dataclasses import replace
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from src import mnist_pipeline as pipe
from src.corruption_datasets import materialize_variable_awgn_corpus
from run_cifar10_noise_seed_replicates import require, digest, write_json
BASE_ID = 'colored_mnist_001'
BASE_ROOT = PROJECT / 'experiments/colored_mnist' / BASE_ID
TD_STAGES = pipe.expand_stage_selection('downstream_grid')
DR_STAGES = pipe.expand_stage_selection('drunet_grid')
STAGES = pipe.expand_stage_selection('full_grid')
LADDER_SEED = 42


def layout(output=None):
    parent = Path(output or BASE_ROOT.parent).resolve()
    baseline = pipe.read_yaml(PROJECT / 'configs/experiments/colored_mnist.yaml')
    items=[]
    for seed in (42,43,44):
        suffix = 'seed42_canonical' if seed == 42 else f'seed{seed}'
        path=PROJECT / f'configs/experiments/colored_mnist_{suffix}.yaml'
        cfg=pipe.read_yaml(path)
        run_id=f'{BASE_ID}_{suffix}'
        expected=copy.deepcopy(baseline)
        expected['seeds']['corpus_seed']=seed
        expected['experiment']['run_id']=run_id
        require(cfg==expected, 'Only corpus_seed and run_id metadata may differ')
        raw=parent/run_id; root=raw.resolve()
        require(not raw.is_symlink() and root.parent==parent and not root.is_relative_to(BASE_ROOT.resolve()), 'Unsafe root')
        items.append((seed,run_id,path,root,cfg))
    return items


def context(item, resume=False, dry_run=False, smoke=False):
    seed,run_id,path,root,cfg=item
    return pipe.Context(copy.deepcopy(cfg),path.resolve(),run_id,root,PROJECT/cfg['dataset']['data_root'],
                        torch.device('cpu' if smoke else 'cuda'),resume,dry_run,smoke,False)


def identity(item):
    source={p.relative_to(PROJECT).as_posix():pipe.sha256_file(p) for p in (PROJECT/'src').rglob('*.py')}
    for p in (Path(__file__),PROJECT/'scripts/run_cifar10_noise_seed_replicates.py',PROJECT/'scripts/run_colored_mnist_pipeline.py'):
        source[p.relative_to(PROJECT).as_posix()]=pipe.sha256_file(p)
    policy=PROJECT/'configs/experiments/colored_mnist_replicate_protocol.json'
    source[policy.relative_to(PROJECT).as_posix()]=pipe.sha256_file(policy)
    return dict(run_id=item[1],corpus_seed=item[0],root=str(item[3]),config_hash=digest(item[4]),
                source_hash=digest(source),ladder_seed=LADDER_SEED,color_seed=item[4]['dataset']['colorization']['color_seed'])


def guard(item,resume):
    root=item[3]
    if not root.exists() or not any(root.iterdir()): return
    require(resume,f'First launch requires empty root: {root}')
    require(not any(p.is_symlink() for p in root.rglob('*')),'Symlink artifacts forbidden')
    owner=root/'00_protocol/noise_replicate_owner.json'
    require(owner.exists(),'Foreign/unowned artifacts')
    require(json.loads(owner.read_text(encoding='utf-8'))==identity(item),'Resume identity mismatch')
    require(pipe.read_yaml(root/'00_protocol/resolved_config.yaml')==item[4],'Resolved config mismatch')
    ctx=context(item)
    selected=pipe.selected_score_root(ctx)/'selected_score_config.yaml'
    if selected.exists():
        s=pipe.read_yaml(selected)
        require(s['run_id']==item[1] and s['config_hash']==pipe.sha256_file(item[2]),'Foreign selected config')
        ck=Path(s['checkpoint_path']).resolve()
        require(ck.is_relative_to(root/'02_ddpm') and ck.is_file(),'Foreign DDPM checkpoint')
        require(pipe.sha256_file(ck)==s['checkpoint_sha256'],'Selected checkpoint hash mismatch')
        require(pipe.sha256_file(root/'01_corpus/train_manifest.csv')==s['corpus_manifest_hash'],'Foreign corpus manifest')
    for marker in root.rglob('*_SUCCESS.json'):
        m=pipe.read_json(marker)
        require(m.get('run_id')==item[1],f'Foreign marker: {marker}')


def fixed_ladder(original):
    def call(ctx,*args,**kwargs):
        cfg=copy.deepcopy(ctx.cfg)
        cfg['seeds']['corpus_seed']=LADDER_SEED
        return original(replace(ctx,cfg=cfg),*args,**kwargs)
    return call


def execute_stage(item,stage,resume):
    ctx=context(item,resume=False)
    # Use launcher markers to decide which stages are complete.
    original=pipe.ladder_scores
    pipe.ladder_scores=fixed_ladder(original)
    try:
        pipe.run_stage(ctx,stage)
        folder=branch(item,stage)
        if folder is not None:
            hist=pd.read_csv(folder/'training_history.csv').dropna(subset=['val_loss'])
            best=hist.loc[hist.val_loss.idxmin()]
            payload=torch.load(folder/'best.pt',map_location='cpu',weights_only=False,mmap=True)
            require(int(payload['step'])==int(best.step),'Checkpoint is not true minimum val_loss')
            require(np.isclose(payload['best_val_loss'],best.val_loss,rtol=1e-10,atol=1e-12),'Best checkpoint loss mismatch')
    finally: pipe.ladder_scores=original


def outputs(root,stage):
    dummy=list(layout(root.parent)[0]);dummy[3]=root
    ctx=context(tuple(dummy))
    marker=pipe.marker_path(ctx,stage)
    require(marker.is_file(),f'Missing stage marker: {stage}')
    result=[marker]
    for name in pipe.read_json(marker)['outputs']:
        path=Path(name)
        require(path.resolve().is_relative_to(root.resolve()),'Output outside own root')
        if path.is_dir(): result.extend(sorted(p for p in path.rglob('*') if p.is_file()))
        else: result.append(path)
    return list(dict.fromkeys(result))


def branch(item,stage):
    ctx=context(item)
    if stage in TD_STAGES: return ctx.experiment_root / '06_downstream/tdncnn' / pipe.variant_from_stage(stage)
    if stage in DR_STAGES: return pipe.drunet_out_dir(ctx,pipe.variant_from_drunet_stage(stage))
    return None


def smoke(item,real=False):
    records={}
    with tempfile.TemporaryDirectory(prefix='colored_noise_smoke_') as tmp:
        for seed in (42,43,44):
            ctx=context(item,smoke=not real)
            ctx.cfg['seeds']['corpus_seed']=seed
            ds=pipe.load_clean_dataset(ctx,train=True)
            indices=np.array([0,7,17])
            folder=Path(tmp)/str(seed)
            tensor,manifest=materialize_variable_awgn_corpus(dataset=ds,dataset_name='colored_mnist',split='train',
                original_indices=indices,output_dir=folder,target_psnr_min_db=10.,target_psnr_max_db=40.,
                corpus_seed=seed,corruption_version=ctx.cfg['corpus']['version'])
            x=torch.load(tensor,map_location='cpu',weights_only=False);m=pd.read_csv(manifest)
            again,_=materialize_variable_awgn_corpus(dataset=ds,dataset_name='colored_mnist',split='train',
                original_indices=indices,output_dir=folder/'repeat',target_psnr_min_db=10.,target_psnr_max_db=40.,
                corpus_seed=seed,corruption_version=ctx.cfg['corpus']['version'])
            require(torch.equal(x,torch.load(again,weights_only=False)),'Reproducibility failed')
            base=[]
            for row in m.itertuples():
                gen=torch.Generator().manual_seed(int(row.sample_seed));torch.rand((),generator=gen)
                base.append(torch.randn((3,28,28),generator=gen))
            records[seed]=(torch.stack([ds[int(i)][0] for i in indices]),m,x,torch.stack(base))
        for a,b in ((42,43),(42,44),(43,44)):
            ca,ma,xa,na=records[a]; cb,mb,xb,nb=records[b]
            require(torch.equal(ca,cb),'RGB/colorization drift')
            for col in ('color_r','color_g','color_b','class_id','color_seed','background_seed'):
                require(ma[col].equals(mb[col]),f'{col} drift')
            require(np.all(ma.target_psnr_db!=mb.target_psnr_db) and np.all(ma.sigma!=mb.sigma),'PSNR/sigma unchanged')
            require(not torch.equal(na,nb) and not torch.equal(xa,xb),'Noise unchanged')
            require(np.allclose(ma.sigma,10**(-ma.target_psnr_db/20)),'Distribution changed')
    print('color_identity/corpus_noise/reproducibility=PASS real_data='+str(real),flush=True)


def blockers(item):
    # All three runs start from scratch; check their budgets in the run configs.
    cfg=item[4];problems=[]
    if cfg['downstream']['max_optimizer_steps']!=20000:problems.append('TDnCNN cap must be 20000')
    if cfg['drunet']['max_steps']!=20000:problems.append('DRUNet cap must be 20000')
    expected=dict(enabled=True,monitor='val_loss',mode='min',patience=10,min_delta=.0001,warmup_steps=100)
    for section in ('downstream','drunet'):
        if cfg[section]['early_stopping']!=expected or cfg[section]['validation_interval_steps']!=100:
            problems.append('Noncanonical early stopping: '+section)
    policy=pipe.read_json(PROJECT/'configs/experiments/colored_mnist_replicate_protocol.json')
    if policy.get('upstream_reuse') is not False:problems.append('Legacy upstream reuse is forbidden')
    return problems


def audit(items):
    for item in items:
        print(json.dumps(dict(**identity(item),config=str(item[2])),ensure_ascii=False),flush=True)
        pipe.validate_current_protocol('Colored MNIST',item[4],downstream_models=['tdncnn','drunet'])
    require(len({identity(i)['config_hash'] for i in items})==3,'Config hashes equal')
    print('stage_graph='+' -> '.join(STAGES),flush=True)
    print('ladder_seed=42 (replicate-only adapter); external=False SIDD=False',flush=True)
    smoke(items[0])
    print('seed42/43/44: FULL RECOMPUTE; legacy upstream reuse=BLOCKED',flush=True)
    for item in items:
        for problem in blockers(item): print('PRODUCTION BLOCKER: '+problem,flush=True)


def preflight(items,resume,minimum):
    item=items[0];ctx=context(item);old=pipe.read_yaml(BASE_ROOT/'00_protocol/resolved_config.yaml')
    cfg=copy.deepcopy(item[4]);cfg['seeds']['corpus_seed']=42
    require(cfg['seeds']==old['seeds'],'Non-corpus seeds differ')
    for key in ('dataset','ddpm','score_validation','corpus','evaluation'):
        require(cfg[key]==old[key],f'Baseline protocol drift: {key}')
    require(torch.cuda.is_available(),'CUDA unavailable')
    print('GPU='+torch.cuda.get_device_name(0),flush=True)
    smoke(item,real=True)
    train=pipe.load_clean_dataset(ctx,train=True);test=pipe.load_clean_dataset(ctx,train=False)
    require(len(train)==60000 and len(test)==10000,'Unexpected MNIST lengths')
    indices=np.load(BASE_ROOT/'00_protocol/evaluation_indices.npy')
    require(np.array_equal(indices,np.arange(len(test))),'Evaluation indices drift')
    cache=BASE_ROOT/'07_evaluation/eval_noise_cache'
    manifest=pd.read_csv(cache/'test_noise_manifest.csv')
    require(manifest.original_index.tolist()==indices.tolist(),'Eval cache indices drift')
    inp=torch.load(cache/'test_eval_input.pt',weights_only=False);tar=torch.load(cache/'test_eval_target.pt',weights_only=False)
    noise=cfg['evaluation']['eval_noise']
    for index in (0,7,9999):
        clean,label=test[index];row=manifest.iloc[index]
        seed=pipe.stable_int_seed('colored_mnist','test_eval_noise',index,cfg['seeds']['eval_noise_seed'],noise['version'])
        gen=torch.Generator().manual_seed(seed)
        sigma=noise['sigma_min']+(noise['sigma_max']-noise['sigma_min'])*torch.rand((),generator=gen).item()
        x=(clean+sigma*torch.randn(clean.shape,generator=gen)).clamp(0,1)
        require(torch.equal(clean,tar[index]) and torch.equal(x,inp[index]),'Seed42 eval/color realization drift')
    parent=item[3].parent
    require(os.access(parent,os.W_OK),'Output not writable')
    free=shutil.disk_usage(parent).free/1024**3
    require(free>=minimum,f'Disk free {free:.1f} GiB < {minimum}')
    for i in items: guard(i,resume)
    print(f'data_preflight=PASS classes=10 train=60000 test=10000 disk_free_GiB={free:.1f}',flush=True)
    problems=[p for i in items for p in blockers(i)]
    require(not problems,'PRODUCTION BLOCKED: '+'; '.join(problems))
    print('preflight=PASS',flush=True)


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
        with (logs / f'colored_mnist_seed{seed}.log').open('a', encoding='utf-8', buffering=1) as log:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip()
            log.write(json.dumps(dict(**identity(item), config=str(path), data=str(context(item).data_root), background_seed=pipe.background_seed(context(item)), device='cuda',
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
                if stage in TD_STAGES or stage in DR_STAGES:
                    branch = globals()['branch'](item,stage)
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
                    subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=PROJECT, env={**os.environ,'PYTHONIOENCODING':'utf-8'})
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
    require(os.environ.get('RESUME','0') in ('0','1'),'RESUME must be 0 or 1')
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['audit', 'preflight', 'dry-run', 'run', 'status', 'worker'])
    parser.add_argument('--resume', action='store_true', default=os.environ.get('RESUME','0')=='1')
    parser.add_argument('--output-base', type=Path)
    parser.add_argument('--min-free-gb', type=float, default=30.)
    parser.add_argument('--seed', type=int, choices=[42,43,44])
    parser.add_argument('--worker-stage', choices=STAGES)
    args = parser.parse_args()
    items = layout(args.output_base)
    if args.action == 'worker':
        require(args.seed is not None and args.worker_stage is not None, 'Worker seed/stage required')
        item = next(i for i in items if i[0] == args.seed)
        guard(item, True)
        require(not blockers(item), 'Unresolved seed42 baseline')
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
            a = context(item,dry_run=True)
            for stage in STAGES:
                pipe.run_stage(a, stage)
        return
    if args.action in ('preflight', 'run'):
        preflight(items, args.resume, args.min_free_gb)
    if args.action == 'run':
        # The OS releases the lock when the process exits.
        import msvcrt
        lock_path = items[0][3].parent / '.colored_mnist_noise_replicates.lock'
        with lock_path.open('a+b') as lock:
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('Another Colored MNIST replicate launcher is running') from exc
            run(items, args.resume)


if __name__ == '__main__':
    main()
