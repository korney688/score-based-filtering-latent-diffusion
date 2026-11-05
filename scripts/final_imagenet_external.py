"""Evaluate DRUNet on synthetic benchmarks and SIDD.
"""
from pathlib import Path
import argparse,copy,datetime as dt,hashlib,json,os,shutil,subprocess,sys,tarfile,time,tempfile
import importlib.metadata
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
import torch
import yaml
from scripts import evaluate_external as ext
from src.evaluation import sidd
from collect_imagenet_final_results import DOCKER,HELPER

OUT=PROJECT/'results_final/imagenet100_external'
SEEDS=[42,43,44]
VARIANTS=['full',*[f'{s}_p{p}' for s in ('topk','quantile','random') for p in (10,20,30)]]
BENCHMARKS=['Kodak24','CBSD68','Urban100','SIDD']
OLD=PROJECT/'experiments/imagenet100_image_space_variable_awgn_v1/imagenet100_image_space_variable_awgn_001'
OLD_SIDD=PROJECT/'experiments/sidd/imagenet100/sidd_full_local_001'


def sha(p):return sidd.sha256_file(p)
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8');tmp.replace(p)
def require(ok,msg):
    if not ok:raise RuntimeError(msg)
def archive_root(seed,project=None):return Path(project or PROJECT)/'results_final/imagenet100'/f'seed{seed}'
def checkpoint(seed,v,project=None):
    p=archive_root(seed,project)/'06_downstream/drunet'/v/'best.pt'
    if p.exists():return p
    if seed==42:
        original=OLD if project is None else Path(project)/'experiments/imagenet100_image_space_variable_awgn_v1/imagenet100_image_space_variable_awgn_001'
        return original/'06_downstream/drunet'/v/'best.pt'
    return p


def copy_remote_checkpoint(container,source,destination,size):
    """Download the exec stream after checking any saved partial data."""
    offset=destination.stat().st_size if destination.exists() else 0
    if offset>size:offset=0
    if offset:
        prefix_code="""import sys,hashlib
h=hashlib.sha256();left=int(sys.argv[2])
with open(sys.argv[1],'rb') as f:
 while left:
  b=f.read(min(left,1048576))
  if not b:raise RuntimeError('Source prefix truncated')
  h.update(b);left-=len(b)
print(h.hexdigest())"""
        source_hash=subprocess.check_output(DOCKER+['exec',container,'python','-c',prefix_code,source,str(offset)],timeout=120,text=True).strip()
        if source_hash!=sha(destination):offset=0
        else:print('CHECKPOINT PARTIAL VERIFIED',source,offset,'/',size,flush=True)
    if offset==size:return
    code="""import sys
with open(sys.argv[1],'rb') as f:
 f.seek(int(sys.argv[2]))
 for b in iter(lambda:f.read(1048576),b''):sys.stdout.buffer.write(b)
sys.stdout.buffer.flush()"""
    errors=tempfile.TemporaryFile()
    proc=subprocess.Popen(DOCKER+['exec',container,'python','-u','-c',code,source,str(offset)],stdout=subprocess.PIPE,stderr=errors)
    try:
        written=offset;next_report=(offset//(32*1024*1024)+1)*32*1024*1024
        with destination.open('ab' if offset else 'wb') as target:
            while written<size:
                chunk=proc.stdout.read(min(1024*1024,size-written))
                require(bool(chunk),'Truncated checkpoint transfer')
                target.write(chunk);written+=len(chunk)
                require(written<=size,'Oversized checkpoint transfer')
                if written>=next_report:
                    print('TRANSFER BYTES',source,written,'/',size,flush=True);next_report+=32*1024*1024
        require(written==size,'Truncated checkpoint transfer')
        try:rc=proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # Check the source SHA256 even when the file size matches.
            proc.kill();proc.wait();rc=0
            print('TRANSFER BYTES COMPLETE; closed delayed Docker client',source,flush=True)
        errors.seek(0)
        require(rc==0,'Checkpoint exec transfer failed: '+errors.read(4096).decode(errors='replace'))
    finally:
        if proc.poll() is None:proc.kill();proc.wait()
        proc.stdout.close()
        errors.close()


def download():
    inv=[{k:r[k] for k in ('seed','run_id','root')} for r in js(PROJECT/'results_final/imagenet100_remote_inventory.json') if r['run_id']==Path(r['root']).name]
    require(sorted(r['seed'] for r in inv)==SEEDS,'Expected exactly three canonical remote runs')
    code="""import sys,json,hashlib,yaml
from pathlib import Path
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
out=[]
for r in json.loads(sys.argv[1]):
 root=Path(r['root']);cp=root/'00_protocol/resolved_config.yaml'
 assert yaml.safe_load(cp.read_text())['seeds']['corpus_seed']==r['seed'],str(cp)
 for v in json.loads(sys.argv[2]):
  p=root/'06_downstream/drunet'/v/'best.pt'
  assert p.is_file() and p.stat().st_size>0,'Missing/empty checkpoint: '+str(p)
  marker=json.loads((p.parent/'_SUCCESS.json').read_text())
  assert marker['run_id']==r['run_id'],str(p)
  out.append(dict(seed=r['seed'],variant=v,remote=str(p),size=p.stat().st_size,sha256=digest(p),
   config_sha256=digest(cp),metadata_sha256=digest(p.parent/'run_meta.json'),success_sha256=digest(p.parent/'_SUCCESS.json')))
print(json.dumps(out))"""
    rows=json.loads(subprocess.check_output(DOCKER+HELPER+[code,json.dumps(inv),json.dumps(VARIANTS)],timeout=120))
    require(len(rows)==30 and {(r['seed'],r['variant']) for r in rows}=={(s,v) for s in SEEDS for v in VARIANTS},'Incomplete remote checkpoint inventory')
    # Check source files and metadata before downloading weights.
    for row in rows:
        root=archive_root(row['seed']);branch=root/'06_downstream/drunet'/row['variant']
        for path,key in [(root/'00_protocol/resolved_config.yaml','config_sha256'),(branch/'run_meta.json','metadata_sha256'),(branch/'_SUCCESS.json','success_sha256')]:
            require(sha(path)==row[key],'Local/source metadata mismatch: '+str(path))
        print('REMOTE CHECKPOINT',row['seed'],row['variant'],row['size'],row['sha256'],flush=True)
    dump(OUT/'remote_checkpoint_manifest.json',rows)
    downloaded=[];container=None
    try:
        for row in rows:
            seed,v=row['seed'],row['variant'];p=archive_root(seed)/'06_downstream/drunet'/v/'best.pt';p.parent.mkdir(parents=True,exist_ok=True)
            existing=checkpoint(seed,v)
            if existing.is_file() and sha(existing)==row['sha256']:
                print('CHECKPOINT VERIFIED',seed,v,flush=True);continue
            require(not p.exists(),'Existing checkpoint differs; refusing overwrite '+str(p))
            if container is None:
                container=subprocess.check_output(DOCKER+HELPER[:1]+['-d']+HELPER[1:]+['import time;time.sleep(86400)'],timeout=30,text=True).strip()
                require(len(container)==64 and all(c in '0123456789abcdef' for c in container),'Invalid download helper ID')
            tmp=p.with_suffix('.download-part')
            print('CHECKPOINT TRANSFER START',seed,v,flush=True)
            copy_remote_checkpoint(container,row['remote'],tmp,row['size'])
            require(tmp.stat().st_size==row['size'] and sha(tmp)==row['sha256'],'Checkpoint transfer integrity failed')
            tmp.replace(p);print('CHECKPOINT DOWNLOADED',seed,v,flush=True)
            downloaded.append(dict(seed=seed,variant=v,bytes=row['size'],path=str(p)))
    finally:
        if container:
            subprocess.run(DOCKER+['stop','--time','1',container],stdout=subprocess.DEVNULL,timeout=30,check=True)
    dump(OUT/'checkpoint_download_report.json',dict(timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),downloaded=downloaded,
         downloaded_bytes=sum(r['bytes'] for r in downloaded),training_started=False,inference_started=False))
    verify_checkpoints(rows)


def verify_checkpoints(rows=None):
    rows=js(OUT/'remote_checkpoint_manifest.json') if rows is None else rows
    require(len(rows)==30 and {(r['seed'],r['variant']) for r in rows}=={(s,v) for s in SEEDS for v in VARIANTS},'Expected 30 unique canonical checkpoints')
    results=[];details=[]
    torch.set_num_threads(2)
    for row in rows:
        seed,v=row['seed'],row['variant'];p=checkpoint(seed,v);root=archive_root(seed)
        entry={'Training seed':seed,'Variant':v,'Local path':str(p),'File size':p.stat().st_size if p.exists() else 0,
               'SHA256':'','Remote/source path':row['remote'],'Integrity status':'FAIL'}
        try:
            require(p.is_file() and entry['File size']>0,'Missing/empty checkpoint')
            require(entry['File size']==row['size'],'Source size mismatch')
            entry['SHA256']=sha(p);require(entry['SHA256']==row['sha256'],'Source SHA256 mismatch')
            cfg_path=root/'00_protocol/resolved_config.yaml';branch=root/'06_downstream/drunet'/v
            for path,key in [(cfg_path,'config_sha256'),(branch/'run_meta.json','metadata_sha256'),(branch/'_SUCCESS.json','success_sha256')]:
                require(sha(path)==row[key],'Source metadata SHA256 mismatch: '+str(path))
            cfg=sidd.read_yaml(cfg_path);meta=js(branch/'run_meta.json');marker=js(branch/'_SUCCESS.json')
            require(cfg['dataset']['name']=='imagenet100' and cfg['seeds']['corpus_seed']==seed,'Wrong training dataset/seed')
            require(marker['run_id']==Path(row['remote']).parents[3].name,'Wrong canonical run ID')
            require(meta['model']=='drunet' and meta['variant']==v,'Wrong architecture/variant metadata')
            protocol=meta['training_protocol']
            require(protocol['model']=='drunet' and protocol['variant']==v,'Wrong training protocol model/variant')
            require(hashlib.sha256(json.dumps(protocol,sort_keys=True,default=str).encode()).hexdigest()==meta['training_protocol_hash'],'Training protocol hash mismatch')
            payload=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
            require(payload.get('variant')==v,'Checkpoint variant mismatch')
            require(payload.get('training_protocol_hash')==meta['training_protocol_hash'],'Checkpoint training protocol mismatch')
            require(payload.get('step')==meta['best_step'],'Checkpoint best step mismatch')
            require(np.isclose(payload.get('best_val_loss'),meta['best_val_loss'],rtol=1e-10,atol=1e-12),'Checkpoint best loss mismatch')
            model=sidd.build_model('drunet',cfg,torch.device('cpu'))
            model.load_state_dict(sidd.extract_state_dict(payload),strict=True)
            del model,payload
            entry['Integrity status']='PASS'
            details.append(dict(seed=seed,variant=v,status='PASS',best_step=meta['best_step'],training_protocol_hash=meta['training_protocol_hash']))
        except Exception as exc:
            details.append(dict(seed=seed,variant=v,status='FAIL',error=str(exc)))
        results.append(entry)
        print('CHECKPOINT INTEGRITY',seed,v,entry['Integrity status'],flush=True)
    pd.DataFrame(results).to_csv(OUT/'checkpoint_inventory.csv',index=False)
    dump(OUT/'checkpoint_integrity_report.json',dict(timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),checks=details,
         passed=sum(r['Integrity status']=='PASS' for r in results),total=len(results),training_started=False,inference_started=False))
    require(all(r['Integrity status']=='PASS' for r in results),'Checkpoint integrity audit failed; see checkpoint_integrity_report.json')
    print('30/30 CANONICAL DRUNET CHECKPOINTS PASS',flush=True)


def inputs():
    cfg=sidd.read_yaml(PROJECT/'configs/evaluation/imagenet100_sidd.yaml')
    synthetic={}
    for b in BENCHMARKS[:3]:
        info=ext.audit_benchmark(PROJECT/'data/external_benchmarks',b)
        require(info['status']=='clean_only',f'Unexpected synthetic dataset: {b}')
        require(info['file_count']=={'Kodak24':24,'CBSD68':68,'Urban100':100}[b],'Unexpected image count')
        synthetic[b]=[dict(path=str(p.resolve()),sha256=sha(p)) for p in ext.list_images(PROJECT/'data/external_benchmarks'/b)]
    return dict(synthetic=synthetic,sidd_noisy_sha256=sha(PROJECT/cfg['sidd']['root']/cfg['sidd']['noisy_mat_filename']),
                sidd_gt_sha256=sha(PROJECT/cfg['sidd']['root']/cfg['sidd']['gt_mat_filename']),
                sigma=25,noise_seed=42,sidd_primary_sigma_conditioning=25,sidd_added_noise=False,
                synthetic_noise_algorithm=hashlib.sha256(__import__('inspect').getsource(ext.load_pairs_for_status).encode()).hexdigest(),
                synthetic_evaluator_sha256=sha(Path(ext.__file__)),sidd_evaluator_sha256=sha(Path(sidd.__file__)),
                lpips_preprocessing_sha256=sha(PROJECT/'src/TDnCNN_image.py'),
                package_versions={p:importlib.metadata.version(p) for p in ('torch','torchvision','lpips','numpy','scikit-image')})


def old_sidd_reuse(seed,v,identity):
    if seed!=42 or v not in ['full','topk_p10','topk_p20','topk_p30']:return None
    marker=OLD_SIDD/'00_protocol/markers'/f'evaluate__drunet__{v}__sigma25.success.json'
    source=OLD_SIDD/'02_evaluation/drunet'/v/'sigma_025/per_block_metrics.csv'
    if not marker.exists() or not source.exists():return None
    m=js(marker);cfg=sidd.read_yaml(OLD_SIDD/'00_protocol/resolved_config.yaml')
    for key,value in [('checkpoint_sha256',identity['checkpoint_sha256']),('sidd_noisy_sha256',identity['inputs']['sidd_noisy_sha256']),
                      ('sidd_gt_sha256',identity['inputs']['sidd_gt_sha256']),('sigma',25),('model','drunet'),('variant',v),
                      ('workload_mode','full'),('max_blocks',None),('expected_total_blocks',1280)]:
        if m.get(key)!=value:return None
    if cfg['drunet']['primary_sigma_255']!=25 or not cfg['runtime']['lpips_enabled']:return None
    df=pd.read_csv(source)
    if len(df)!=1280 or df.global_block_index.nunique()!=1280 or not np.isfinite(df[['PSNR','SSIM','LPIPS','MSE']]).all().all():return None
    if set(df.model)!={'drunet'} or set(df.variant)!={v} or set(df.assumed_sigma_255)!={25}:return None
    primary=pd.read_csv(OLD_SIDD/'03_analysis/primary_sidd_table.csv')
    label='Full' if v=='full' else 'Top-K '+v.rsplit('p',1)[1]+'%'
    selected=primary[(primary.Model=='DRUNet')&(primary.Variant==label)]
    if len(selected)!=1 or not np.allclose(df[['PSNR','SSIM','LPIPS','MSE']].mean(),selected[['PSNR','SSIM','LPIPS','MSE']].iloc[0],rtol=1e-8,atol=1e-10):return None
    return source


def audit():
    OUT.mkdir(parents=True,exist_ok=True);protocol=inputs();rows=[];checkpoints=[]
    for seed in SEEDS:
        cfg=sidd.read_yaml(archive_root(seed)/'00_protocol/resolved_config.yaml')
        require(cfg['seeds']['corpus_seed']==seed,'Cross-seed source config')
        for v in VARIANTS:
            p=checkpoint(seed,v);h=sha(p) if p.exists() else None
            if h:
                meta=js(archive_root(seed)/'06_downstream/drunet'/v/'run_meta.json')
                payload=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
                require(payload.get('training_protocol_hash')==meta['training_protocol_hash'],'Checkpoint protocol mismatch')
                require(payload['step']==meta['best_step'],'Wrong best checkpoint')
                del payload
            checkpoints.append(dict(seed=seed,variant=v,path=str(p),sha256=h,exists=p.is_file()))
            identity=dict(seed=seed,variant=v,checkpoint_sha256=h,inputs=protocol)
            reuse=old_sidd_reuse(seed,v,identity) if h else None
            for b in BENCHMARKS:
                exists=seed==42 and v in ['full','topk_p10','topk_p20','topk_p30']
                rows.append(dict(seed=seed,variant=v,benchmark=b,existing=exists,
                                 decision='REUSE' if b=='SIDD' and reuse else 'RECOMPUTE_UNPROVEN' if exists else 'MISSING',
                                 checkpoint_available=p.is_file(),source=str(reuse) if b=='SIDD' and reuse else None))
    dump(OUT/'input_protocol.json',protocol);dump(OUT/'checkpoint_manifest.json',checkpoints)
    pd.DataFrame(rows).to_csv(OUT/'existing_combinations_audit.csv',index=False)
    print(pd.DataFrame(rows).groupby(['seed','decision']).size().to_string(),flush=True)
    print('missing checkpoints',sum(not r['exists'] for r in checkpoints),flush=True)
    return rows,protocol


def combo_identity(seed,v,b,protocol,ckpt):
    return dict(seed=seed,variant=v,benchmark=b,architecture='DRUNet',checkpoint_sha256=sha(ckpt),inputs=protocol)


def infer(seeds=SEEDS):
    rows,protocol=audit()
    missing=[r for r in js(OUT/'checkpoint_manifest.json') if r['seed'] in seeds and not r['exists']]
    require(not missing,'BLOCKED: missing 20 canonical seed43/44 checkpoints; connect VPN and run download-checkpoints')
    require(torch.cuda.is_available(),'Local CUDA unavailable')
    device=torch.device('cuda');lpips=ext.create_lpips_model(device)
    require(lpips is not None,'LPIPS unavailable; do not produce incomplete final metrics')
    sidd_cfg=sidd.read_yaml(PROJECT/'configs/evaluation/imagenet100_sidd.yaml')
    require(sidd_cfg['drunet']['primary_sigma_255']==25,'SIDD protocol changed')
    for seed in seeds:
        cfg=sidd.read_yaml(archive_root(seed)/'00_protocol/resolved_config.yaml')
        scfg=copy.deepcopy(sidd_cfg);scfg['variants']=VARIANTS
        ctx=sidd.SiddContext(scfg,PROJECT/'configs/evaluation/imagenet100_sidd.yaml',OUT/f'seed{seed}/sidd_native',archive_root(seed),
                             PROJECT/sidd_cfg['sidd']['root'],cfg,archive_root(seed)/'00_protocol/resolved_config.yaml','evaluate',resume=True,
                             source_config_sha256=sha(archive_root(seed)/'00_protocol/resolved_config.yaml'))
        dataset=sidd.open_dataset(ctx)
        for v in VARIANTS:
            ckpt=checkpoint(seed,v);todo=[]
            for b in BENCHMARKS:
                folder=OUT/f'seed{seed}'/b/v;marker=folder/'provenance.json';csv=folder/'per_observation_metrics.csv'
                identity=combo_identity(seed,v,b,protocol,ckpt)
                if marker.exists():
                    saved=js(marker);require(saved['identity']==identity,'Stale combo provenance')
                    require(csv.exists() and sha(csv)==saved['output_sha256'],'Stale combo metrics')
                    print('REUSE FINAL',seed,v,b,flush=True);continue
                old=old_sidd_reuse(seed,v,identity) if b=='SIDD' else None
                if old:
                    folder.mkdir(parents=True,exist_ok=True);shutil.copy2(old,csv)
                    dump(marker,dict(identity=identity,output_sha256=sha(csv),reused_source=str(old),reused_source_sha256=sha(old)))
                    print('REUSE EXISTING SIDD',seed,v,flush=True);continue
                todo.append(b)
            if not todo:continue
            model=sidd.load_checkpoint_strict('drunet',ckpt,cfg,device)
            for b in todo:
                folder=OUT/f'seed{seed}'/b/v;folder.mkdir(parents=True,exist_ok=True)
                dump(OUT/'status.json',dict(status='RUNNING',seed=seed,variant=v,benchmark=b,pid=os.getpid(),timestamp=dt.datetime.now().isoformat()))
                print('EVALUATE',seed,v,b,flush=True);start=time.time()
                if b=='SIDD':
                    row=dict(checkpoint=str(ckpt),checkpoint_sha256=sha(ckpt))
                    source=sidd.evaluate_loaded_combination(ctx,dataset,model,row,'drunet',v,25,lpips)
                    # Save SIDD results in the per-block format.
                    shutil.copy2(source,folder/'per_observation_metrics.csv')
                else:
                    info=ext.audit_benchmark(PROJECT/'data/external_benchmarks',b)
                    pairs=ext.load_pairs_for_status(PROJECT/'data/external_benchmarks',b,info,25,42,None)
                    result=[]
                    for i,pair in enumerate(pairs):
                        pred=ext.predict_model(model,pair['noisy'],'drunet',25/255,device)
                        result.append(dict(Image=pair['image'],Benchmark=b,Variant=v,**ext.compute_metrics(pred,pair['clean'],lpips)))
                        if i%10==0:print(f'{seed}/{v}/{b} {i+1}/{len(pairs)} elapsed={time.time()-start:.1f}s',flush=True)
                    pd.DataFrame(result).to_csv(folder/'per_observation_metrics.csv',index=False)
                    del pairs
                csv=folder/'per_observation_metrics.csv'
                expected=1280 if b=='SIDD' else {'Kodak24':24,'CBSD68':68,'Urban100':100}[b]
                d=pd.read_csv(csv);require(len(d)==expected and np.isfinite(d[['PSNR','SSIM','LPIPS','MSE']]).all().all(),'Incomplete metrics')
                dump(folder/'provenance.json',dict(identity=combo_identity(seed,v,b,protocol,ckpt),output_sha256=sha(csv),seconds=time.time()-start))
            del model;torch.cuda.empty_cache()
        dataset.close()
    dump(OUT/'status.json',dict(status='COMPLETE' if set(seeds)==set(SEEDS) else 'PARTIAL_COMPLETE',seeds=seeds,timestamp=dt.datetime.now().isoformat()))
    if set(seeds)==set(SEEDS):aggregate()


def aggregate(source_root=None,output_dir=None,artifact_root=None):
    source_root=Path(source_root) if source_root is not None else OUT
    output=Path(output_dir) if output_dir is not None else source_root
    if output_dir is not None:require(not output.exists(),'Output directory already exists')
    rows=[];signatures=[]
    for seed in SEEDS:
        for v in VARIANTS:
            for b in BENCHMARKS:
                folder=source_root/f'seed{seed}'/b/v;csv=folder/'per_observation_metrics.csv';marker=folder/'provenance.json'
                require(csv.exists() and marker.exists(),f'BLOCKED: missing {seed}/{v}/{b}')
                m=js(marker);identity=m['identity'];signatures.append(json.dumps(identity['inputs'],sort_keys=True))
                require(identity['seed']==seed and identity['variant']==v and identity['benchmark']==b,'Cross-seed/variant metadata')
                require(sha(csv)==m['output_sha256'] and sha(checkpoint(seed,v,artifact_root))==identity['checkpoint_sha256'],'Checkpoint/output hash drift')
                d=pd.read_csv(csv)
                require(len(d)==(1280 if b=='SIDD' else {'Kodak24':24,'CBSD68':68,'Urban100':100}[b]),'Observation count mismatch')
                key='global_block_index' if b=='SIDD' else 'Image'
                require(d[key].is_unique,'Duplicate observation')
                require(np.isfinite(d[['PSNR','SSIM','LPIPS','MSE']]).all().all(),'NaN metric')
                strategy='Full' if v=='full' else {'topk':'Top-K','quantile':'Quantile','random':'Random'}[v.split('_')[0]]
                fraction=1. if v=='full' else int(v.rsplit('p',1)[1])/100
                rows.append(dict(Benchmark=b,Architecture='DRUNet',**{'Training corpus seed':seed,'Strategy':strategy,'Fraction':fraction,'Variant':v,
                    'Checkpoint':str(checkpoint(seed,v,artifact_root)),'Source artifact':str(csv)},**d[['PSNR','SSIM','LPIPS','MSE']].mean().to_dict()))
    require(len(set(signatures))==1,'Benchmark input protocols differ')
    all_=pd.DataFrame(rows);require(len(all_)==120 and not all_.duplicated(['Benchmark','Training corpus seed','Variant']).any(),'Bad aggregate grid')
    full=all_[all_.Variant=='full'].set_index(['Benchmark','Training corpus seed']).PSNR
    all_['Delta PSNR vs Full']=[r['PSNR']-full.loc[(r['Benchmark'],r['Training corpus seed'])] for _,r in all_.iterrows()]
    mean=[]
    for (b,s,f,v),d in all_.groupby(['Benchmark','Strategy','Fraction','Variant'],sort=False):
        require(set(d['Training corpus seed'])==set(SEEDS),'Need three paired seeds')
        row=dict(Benchmark=b,Architecture='DRUNet',Strategy=s,Fraction=f,Variant=v)
        for metric in ('PSNR','SSIM','LPIPS','MSE','Delta PSNR vs Full'):
            name='Delta PSNR' if metric=='Delta PSNR vs Full' else metric
            row[name+' mean']=d[metric].mean();row[name+' std']=d[metric].std(ddof=1)
            require(np.isclose(row[name+' mean'],np.mean(d[metric])) and np.isclose(row[name+' std'],np.std(d[metric],ddof=1)),'Mean/std validation failed')
        mean.append(row)
    mean=pd.DataFrame(mean)
    require((all_.loc[all_.Variant=='full','Delta PSNR vs Full']==0).all(),'Full delta mismatch')
    pub=mean.pivot(index='Benchmark',columns='Variant',values='PSNR mean').reindex(index=BENCHMARKS,columns=VARIANTS)
    fmt=pub.astype(object)
    for b in BENCHMARKS:
        for v in VARIANTS:
            r=mean[(mean.Benchmark==b)&(mean.Variant==v)].iloc[0];fmt.loc[b,v]=f"{r['PSNR mean']:.4f} ± {r['PSNR std']:.4f}"
    labels={v:('Full' if v=='full' else {'topk':'Top-K','quantile':'Quantile','random':'Random'}[v.split('_')[0]]+' '+v.rsplit('p',1)[1]+'%') for v in VARIANTS}
    pub=pub.rename(columns=labels);fmt=fmt.rename(columns=labels)
    output.mkdir(parents=True,exist_ok=True)
    pub.to_csv(output/'imagenet100_external_psnr_publication.csv')
    with pd.ExcelWriter(output/'imagenet100_external_psnr_publication.xlsx') as w:
        pub.to_excel(w,sheet_name='Mean PSNR');fmt.to_excel(w,sheet_name='Mean ± std')
    wide=all_.pivot(index=['Benchmark','Variant'],columns='Training corpus seed',values='PSNR').rename(columns={s:f'Seed{s}' for s in SEEDS}).reset_index()
    wide=wide.merge(mean[['Benchmark','Variant','PSNR mean','PSNR std','Delta PSNR mean']],on=['Benchmark','Variant'])
    wide.to_excel(output/'imagenet100_external_psnr_by_seed.xlsx',index=False)
    all_.to_csv(output/'imagenet100_external_all_seeds.csv',index=False);mean.to_csv(output/'imagenet100_external_mean_3seeds.csv',index=False)
    dump(output/'validation.json',dict(status='IMAGENET EXTERNAL FULL GRID READY',rows=120,mean_rows=40,std_ddof=1,
                                 large_spread=mean[mean['PSNR std']>=.5].to_dict('records')))
    dump(output/'output_files_sha256.json',{name:sha(output/name) for name in (
        'imagenet100_external_all_seeds.csv','imagenet100_external_mean_3seeds.csv','validation.json')})
    print(pub.to_string());print('IMAGENET EXTERNAL FULL GRID READY',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['audit','download-checkpoints','verify-checkpoints','evaluate','aggregate']);p.add_argument('--seeds',type=int,nargs='+',choices=SEEDS,default=SEEDS)
    p.add_argument('--artifact-root',type=Path,default=None)
    p.add_argument('--source-root',type=Path,default=None)
    p.add_argument('--output-dir',type=Path,default=None)
    a=p.parse_args()
    if a.action=='aggregate':
        if (a.artifact_root is not None or a.source_root is not None) and a.output_dir is None:
            p.error('Use --source-root/--artifact-root together with a new --output-dir')
        aggregate(a.source_root,a.output_dir,a.artifact_root)
        return
    if any(value is not None for value in (a.artifact_root,a.source_root,a.output_dir)):
        p.error('Artifact/output overrides are supported by aggregate only')
    OUT.mkdir(parents=True,exist_ok=True)
    if a.action=='audit':audit()
    elif a.action=='download-checkpoints':download()
    elif a.action=='verify-checkpoints':verify_checkpoints()
    elif a.action=='evaluate':
        import msvcrt
        with (OUT/'.evaluation.lock').open('a+b') as lock:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
            infer(a.seeds)


if __name__=='__main__':main()
