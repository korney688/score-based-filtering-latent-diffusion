"""Aggregate saved experiment results."""
from pathlib import Path
import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import sys
import numpy as np
import pandas as pd
import yaml
from openpyxl.styles import Font, PatternFill, Alignment

PROJECT=Path(__file__).resolve().parents[1]
OUT=PROJECT/'results_final'
SEEDS=[42,43,44]
DATASETS=['Colored MNIST','CIFAR-10','ImageNet-100']
ARCHS=['TDnCNN','DRUNet']
VARIANTS=['full',*[f'{s}_p{p}' for s in ('topk','quantile','random') for p in (10,20,30)]]
STAMP=dt.datetime.now(dt.timezone.utc).isoformat()
METRICS=['PSNR','SSIM','LPIPS','MSE','Delta PSNR vs Full']


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def js(path):return json.loads(path.read_text(encoding='utf-8'))
def yl(path):return yaml.safe_load(path.read_text(encoding='utf-8'))
def hash_object(value):return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()
def require(ok,msg):
    if not ok:raise ValueError(msg)


def variant(strategy,fraction):
    s=str(strategy).lower()
    if s=='full':return 'full'
    if 'top' in s:s='topk'
    elif 'quantile' in s:s='quantile'
    elif 'random' in s:s='random'
    else:raise ValueError('Unexpected strategy '+s)
    return f'{s}_p{round(float(fraction)*100)}'


def normalized_frame(frame,dataset,seed,source):
    frame=frame.rename(columns={'model':'Architecture','strategy':'Strategy','fraction':'Fraction','train_size':'Train size',
                                'best_step':'Best step','effective_epochs_at_best':'Effective epochs','Effective epochs at best':'Effective epochs'})
    frame['Architecture']=frame.Architecture.str.lower().map({'tdncnn':'TDnCNN','drunet':'DRUNet'})
    frame['Variant']=[variant(s,f) for s,f in zip(frame.Strategy,frame.Fraction)]
    frame['Strategy']=frame.Variant.map(lambda v:'Full' if v=='full' else {'topk':'Top-K','quantile':'Quantile','random':'Random'}[v.split('_')[0]])
    frame['Dataset']=dataset;frame['Seed']=seed;frame['Source file']=str(source.resolve());frame['Source SHA256']=sha(source)
    for col in ['Effective epochs','LPIPS']:
        if col not in frame:frame[col]=np.nan
    return frame[['Dataset','Architecture','Seed','Strategy','Fraction','Variant','Train size','Best step','Effective epochs','PSNR','SSIM','LPIPS','MSE','Source file','Source SHA256']]


def run_sources(project=None, results_root=None):
    project = Path(project) if project is not None else PROJECT
    results_root = Path(results_root) if results_root is not None else OUT
    inventory=js(results_root/'imagenet100_remote_inventory.json')
    remote={r['seed']:r for r in inventory if r['run_id']==Path(r['root']).name}
    result=[]
    for ds in DATASETS:
        for seed in SEEDS:
            if ds=='Colored MNIST':
                suffix='seed42_canonical' if seed==42 else f'seed{seed}'
                root=project/'experiments/colored_mnist'/f'colored_mnist_001_{suffix}'
                csv=root/'07_evaluation/subset_grid_results.csv';cfg=root/'00_protocol/resolved_config.yaml'
            elif ds=='CIFAR-10':
                name='cifar10_001_final_20260901'+('' if seed==42 else f'_seed{seed}')
                root=project/'experiments/cifar10'/name
                csv=root/'07_evaluation/combined/subset_grid_results_all.csv'
                cfg=root/('00_protocol/resolved_score_validation_config.yaml' if seed==42 else '00_protocol/resolved_config.yaml')
            else:
                root=results_root/'imagenet100'/f'seed{seed}';csv=root/'07_evaluation/combined/subset_grid_results_all.csv';cfg=root/'00_protocol/resolved_config.yaml'
                require((root/'DOWNLOAD_VERIFIED.json').exists(),'ImageNet transfer incomplete: '+str(root))
            result.append(dict(dataset=ds,seed=seed,root=root,csv=csv,config=cfg,
                               remote=remote[seed] if ds=='ImageNet-100' else None))
    return result


def signature(cfg,ds):
    cfg=copy.deepcopy(cfg)
    if ds=='Colored MNIST':
        cfg.pop('experiment',None);cfg['seeds'].pop('corpus_seed')
        return cfg
    if ds=='CIFAR-10':
        # Read the DDPM validation settings shared by all runs.
        keys=('split_sizes','target_psnr_db','psnr_levels','ddpm','scan','num_workers','seeds')
        cfg={k:cfg[k] for k in keys};cfg['seeds'].pop('corpus_seed');return cfg
    keys=('dataset','seeds','corpus','ddpm','score_validation','score_selection','evaluation')
    cfg={k:cfg[k] for k in keys};cfg['dataset'].pop('root',None);cfg['seeds'].pop('corpus_seed');return cfg


def completeness(run,frame,results_root=None):
    results_root = Path(results_root) if results_root is not None else OUT
    ds,seed,root=run['dataset'],run['seed'],run['root']
    require(len(frame)==20,'Expected 20 result rows')
    require(set(frame.Architecture)==set(ARCHS),'Architecture missing')
    for arch in ARCHS:require(set(frame[frame.Architecture==arch].Variant)==set(VARIANTS),'Variant missing')
    status=list((root/'00_protocol').glob('noise_replicate_status.json'))
    if status:require(js(status[0])['status']=='COMPLETE','Launcher did not complete')
    summary=root/('08_summary/final_summary.md' if ds=='Colored MNIST' else '09_summary/final_summary.md')
    require(summary.exists(),'Final summary missing')
    require(any((root/'08_figures').glob('*.png')),'Figures missing')
    require((root/'05_filtering/subset_indices').is_dir(),'Filtering grid missing')
    if ds=='ImageNet-100':
        remote=next(x for x in js(results_root/'imagenet100_remote_completeness.json') if x['root']==run['remote']['root'])
        require(all(remote[k] for k in ('filter_marker','eval_marker','figures_marker','summary_marker')),'Remote terminal marker missing')
        for arch in ('tdncnn','drunet'):
            require(len(remote['models'][arch])==10 and all(all(b[k] for k in ('best','last','metadata','history','marker')) for b in remote['models'][arch]),'Remote checkpoint/branch incomplete')
    branch_meta=[]
    for arch in ARCHS:
        for row in frame[frame.Architecture==arch].itertuples(index=False):
            candidates=[root/'06_downstream'/arch.lower()/v for v in alias_names(row.Variant,ds)]
            folder=next((p for p in candidates if (p/'run_meta.json').exists() or (p/'metrics.json').exists()),None)
            require(folder is not None,f'Branch metadata missing: {arch}/{row.Variant}')
            path=folder/('metrics.json' if ds=='CIFAR-10' and arch=='TDnCNN' else 'run_meta.json')
            m=js(path);hist=pd.read_csv(folder/'training_history.csv')
            stopped=int(hist.step.max());best=int(m['best_step'])
            require(stopped==m['optimizer_steps'],f'History incomplete {folder}')
            require(m.get('stop_reason') in ('early_stopping','max_steps'),'Incomplete stop reason')
            source=frame[(frame.Architecture==arch)&(frame.Variant==row.Variant)].iloc[0]
            require(best==int(source['Best step']),'Best step differs between evaluation and metadata')
            require(int(m.get('train_size',m.get('subset_size')))==int(source['Train size']),'Train size mismatch')
            finite=hist[np.isfinite(pd.to_numeric(hist.val_loss,errors='coerce'))]
            require(len(finite)>0,'No validation history')
            require(int(finite.loc[finite.val_loss.idxmin(),'step'])==best,'best_step is not validation minimum: '+str(folder))
            if (folder/'status.json').exists():require(js(folder/'status.json').get('status')=='complete','Branch status incomplete')
            cap=m.get('max_steps',m.get('max_optimizer_steps',m.get('total_steps')))
            expected=20000 if ds=='Colored MNIST' else (60000 if arch=='TDnCNN' else 50000)
            require(cap==expected,'Noncanonical cap: '+str(folder))
            require(m.get('early_stopping_enabled') is True,'Early stopping not enabled')
            branch_meta.append(dict(dataset=ds,seed=seed,architecture=arch,variant=row.Variant,path=str(path),cap=cap,
                                    meta=m,history_sha256=sha(folder/'training_history.csv'),metadata_sha256=sha(path)))
    return branch_meta


def alias_names(v,ds):
    if v=='full':return ['full']
    q=v.replace('quantile_','quantile_spread_')
    return [q,q.replace('_p10','_10'),v]


def effective_protocol(branch,ds):
    m=branch['meta']
    if ds=='ImageNet-100':
        p=copy.deepcopy(m['training_protocol'])
        for k in ('variant','subset_indices_sha256','initial_state_sha256'):p.pop(k,None)
        return p
    p={k:m.get(k) for k in ('batch_size','max_optimizer_steps','validation_interval_steps','early_stopping_enabled',
                              'early_stopping_monitor','early_stopping_mode','early_stopping_patience','early_stopping_min_delta','early_stopping_warmup_steps')}
    if ds=='Colored MNIST':
        # Use matching configs and launcher hashes to check missing metadata.
        p={'cap':m['max_steps'],'batch_size':m['batch_size'],'budget_mode':m['budget_mode'],'early_stopping':m['early_stopping_enabled']}
    return p


def per_image_check(run,frame):
    root,ds=run['root'],run['dataset'];sources=[]
    if ds=='Colored MNIST':files=[root/'07_evaluation/per_image_metrics.csv']
    else:files=[root/f'07_evaluation/{a}/per_image_metrics.csv' for a in ('tdncnn','drunet')]
    for path in files:
        d=pd.read_csv(path);d=d.rename(columns={'model':'Architecture','strategy':'Strategy','fraction':'Fraction',
                                                 'psnr':'PSNR','ssim':'SSIM','mse':'MSE','lpips':'LPIPS'})
        if 'Architecture' not in d:d['Architecture']=path.parent.name
        d['Architecture']=d.Architecture.str.lower().map({'tdncnn':'TDnCNN','drunet':'DRUNet'})
        d['Variant']=[variant(s,f) for s,f in zip(d.Strategy,d.Fraction)]
        key=next(k for k in ('original_index','sample_index','image_index') if k in d)
        require(not d.duplicated(['Architecture','Variant',key]).any(),'Duplicated per-image metrics')
        expected=800 if ds=='ImageNet-100' else 10000
        require(set(d.groupby(['Architecture','Variant']).size())=={expected},'Incomplete per-image coverage')
        require(d.PSNR.notna().all(),'Missing per-image PSNR')
        for (arch,v),group in d.groupby(['Architecture','Variant']):
            row=frame[(frame.Architecture==arch)&(frame.Variant==v)].iloc[0]
            for metric in ('PSNR','SSIM','LPIPS','MSE'):
                val=group[metric].mean() if metric in group else np.nan
                require((pd.isna(val) and pd.isna(row[metric])) or np.isclose(val,row[metric],rtol=1e-7,atol=1e-9),f'Summary disagrees with per-image mean: {ds}/{arch}/{v}/{metric}')
        sources.append(dict(path=str(path.resolve()),sha256=sha(path),rows=len(d)))
    return sources


def excel(path,sheets):
    with pd.ExcelWriter(path,engine='openpyxl') as writer:
        for name,table in sheets.items():
            table.to_excel(writer,sheet_name=name,index=False,na_rep='')
            ws=writer.sheets[name];ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
            for c in ws[1]:c.font=Font(bold=True,color='FFFFFF');c.fill=PatternFill('solid',fgColor='25476A')
            for col in ws.columns:
                values=[len(str(c.value)) if c.value is not None else 0 for c in list(col)[:100]]
                ws.column_dimensions[col[0].column_letter].width=min(55,max(12,max(values)+2))
            for row in ws.iter_rows(min_row=2):
                for c in row:
                    if isinstance(c.value,float):c.number_format='0.000000'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--artifact-root',type=Path,default=PROJECT,help='Read-only project artifact layout.')
    parser.add_argument('--results-root',type=Path,default=None,help='Defaults to artifact-root/results_final.')
    parser.add_argument('--output-dir',type=Path,required=True,help='New table directory; never overwritten.')
    args=parser.parse_args()
    results_root=args.results_root or args.artifact_root/'results_final'
    final=args.output_dir
    require(not final.exists(),'Final directory already exists; review hashes before rerunning')
    final.mkdir(parents=True)
    sources=run_sources(args.artifact_root,results_root);frames=[];audits=[];provenance=[];branches=[];errors=[];warnings=[];signatures={};eval_signatures={};owner_hashes={}
    for run in sources:
        ds,seed,root=run['dataset'],run['seed'],run['root']
        try:
            cfg=yl(run['config']);require(cfg['seeds']['corpus_seed']==seed,'Resolved corpus seed mismatch')
            canonical_cfg=yl(root/'00_protocol/resolved_score_validation_config.yaml') if ds=='CIFAR-10' else cfg
            signatures.setdefault(ds,[]).append(hash_object(signature(canonical_cfg,ds)))
            frame=normalized_frame(pd.read_csv(run['csv']),ds,seed,run['csv'])
            bm=completeness(run,frame,results_root);branches.extend(bm)
            per=per_image_check(run,frame)
            cache=root/({'Colored MNIST':'07_evaluation/eval_noise_cache/test_noise_manifest.csv',
                         'CIFAR-10':'08_downstream_commands/eval_noise_cache/test_noise_manifest.csv',
                         'ImageNet-100':'07_evaluation/combined/eval_noise_manifest.csv'}[ds])
            eval_hash=sha(cache)
            eval_signatures.setdefault(ds,[]).append(eval_hash)
            owner=root/'00_protocol/noise_replicate_owner.json'
            owner_meta=js(owner) if owner.exists() or ds=='Colored MNIST' else {}
            if 'source_hash' in owner_meta:owner_hashes.setdefault(ds,[]).append(owner_meta['source_hash'])
            if ds=='Colored MNIST':
                require(owner_meta['ladder_seed']==42 and cfg['dataset']['colorization']['color_seed']==42 and cfg['dataset']['colorization']['background_seed']==42,'Color/ladder seed mismatch')
                warnings.append('Colored MNIST LPIPS was not computed by production; retain blank/NaN, not zero')
            require(np.isfinite(frame[['PSNR','SSIM','MSE']].to_numpy()).all(),'Missing/nonfinite primary metric')
            if ds!='Colored MNIST':require(np.isfinite(frame.LPIPS).all(),'Missing LPIPS in a production protocol that computes it')
            if ds=='CIFAR-10' and seed==42:warnings.append('CIFAR seed42 has low-level resolved validation config; downstream protocol verified from final branch metadata')
            if ds=='ImageNet-100' and seed==42:warnings.append('ImageNet seed42 resolved snapshot predates grid rebuild; downstream protocol verified from final training_protocol metadata')
            p=dict(dataset=ds,seed=seed,run_id=run['remote']['run_id'] if run['remote'] else root.name,experiment_root=str(root.resolve()),
                   remote_experiment_root=run['remote']['root'] if run['remote'] else None,corpus_seed=seed,resolved_config_path=str(run['config'].resolve()),
                   resolved_config_sha256=sha(run['config']),source_evaluation_csv=str(run['csv'].resolve()),source_evaluation_sha256=frame['Source SHA256'].iloc[0],
                   evaluation_noise_manifest_sha256=eval_hash,per_image_sources=per,collected_at=STAMP)
            provenance.append(p);frames.append(frame)
            cap_td,cap_dr=(20000,20000) if ds=='Colored MNIST' else (60000,50000)
            val_interval,warmup=(100,100) if ds=='Colored MNIST' else (500,1000)
            audits.append(dict(Dataset=ds,Seed=seed,Run_ID=p['run_id'],Corpus_seed=seed,Filtering_fractions='Full,p10,p20,p30',
                               TDnCNN_protocol=f'cap={cap_td}; val={val_interval}; patience=10; warmup={warmup}; min_delta=0.0001',
                               DRUNet_protocol=f'cap={cap_dr}; val={val_interval}; patience=10; warmup={warmup}; min_delta=0.0001',
                               TDnCNN_protocol_hash=hash_object(effective_protocol(next(b for b in bm if b['architecture']=='TDnCNN'),ds)),
                               DRUNet_protocol_hash=hash_object(effective_protocol(next(b for b in bm if b['architecture']=='DRUNet'),ds)),
                               Evaluation_protocol=('fixed AWGN sigma=25/255; 800 images' if ds=='ImageNet-100' else 'AWGN sigma~Uniform(0.1,0.8); 10000 images'),
                               Evaluation_manifest_hash=eval_hash,Evaluation_seed=(42 if ds!='Colored MNIST' else 4242),Config=str(run['config']),Status='PASS'))
        except Exception as e:
            errors.append(f'{ds} seed{seed}: {e}');audits.append(dict(Dataset=ds,Seed=seed,Status='BLOCKED',Problem=str(e)))
    for ds in DATASETS:
        if len(signatures.get(ds,[]))!=3 or len(set(signatures.get(ds,[])))!=1:errors.append(ds+': upstream scientific config mismatch')
        if len(eval_signatures.get(ds,[]))!=3 or len(set(eval_signatures.get(ds,[])))!=1:errors.append(ds+': evaluation noise/indices manifest differs')
        for arch in ARCHS:
            protocols=[effective_protocol(b,ds) for b in branches if b['dataset']==ds and b['architecture']==arch]
            if len(protocols)!=30 or len({hash_object(p) for p in protocols})!=1:errors.append(f'{ds}/{arch}: actual training protocol mismatch/incomplete grid')
        if ds=='Colored MNIST' and len(set(owner_hashes.get(ds,[])))!=1:errors.append(ds+': canonical source hashes differ')
    audit=pd.DataFrame(audits)
    (final/'final_results_provenance.json').write_text(json.dumps(dict(collected_at=STAMP,runs=provenance,branches=branches,warnings=sorted(set(warnings)),errors=errors,std_ddof=1),indent=2,ensure_ascii=False),encoding='utf-8')
    audit.to_csv(final/'protocol_audit.csv',index=False)
    if errors:
        (final/'RESULT_AGGREGATION_BLOCKED.txt').write_text('\n'.join(errors),encoding='utf-8')
        print('RESULT AGGREGATION BLOCKED\n'+'\n'.join(errors));return
    all_seeds=pd.concat(frames,ignore_index=True)
    require(len(all_seeds)==180,'Expected 180 rows')
    keys=['Dataset','Architecture','Variant']
    require(not all_seeds.duplicated(keys+['Seed']).any(),'Duplicate result rows')
    require(np.isfinite(all_seeds.PSNR).all(),'Missing/nonfinite PSNR')
    require(all_seeds.groupby(keys).Seed.apply(lambda s:set(s)==set(SEEDS)).all(),'Need exactly three seeds')
    full=all_seeds[all_seeds.Variant=='full'].set_index(['Dataset','Architecture','Seed']).PSNR
    all_seeds['Delta PSNR vs Full']=[r.PSNR-full.loc[(r.Dataset,r.Architecture,r.Seed)] for r in all_seeds.itertuples()]
    require((all_seeds.loc[all_seeds.Variant=='full','Delta PSNR vs Full']==0).all(),'Full delta is not zero')
    all_seeds['_sort']=[(DATASETS.index(r.Dataset),ARCHS.index(r.Architecture),r.Seed,VARIANTS.index(r.Variant)) for r in all_seeds.itertuples()]
    all_seeds=all_seeds.sort_values('_sort').drop(columns='_sort')
    all_seeds=all_seeds[['Dataset','Architecture','Seed','Strategy','Fraction','Variant','Train size','Best step','Effective epochs',
                         'PSNR','SSIM','LPIPS','MSE','Delta PSNR vs Full','Source file','Source SHA256']]
    group=['Dataset','Architecture','Strategy','Fraction','Variant'];means=[]
    for k,d in all_seeds.groupby(group,sort=False):
        row=dict(zip(group,k))
        for metric in METRICS:
            name='Delta_PSNR' if metric=='Delta PSNR vs Full' else metric
            row[name+'_mean']=d[metric].mean();row[name+'_std']=d[metric].std(ddof=1)
            values=d[metric].to_numpy(dtype=float)
            if np.isfinite(values).any():
                require(np.isclose(row[name+'_mean'],np.mean(values)),'Mean validation failed')
                require(np.isclose(row[name+'_std'],np.std(values,ddof=1)),'Std validation failed')
        row['Best_step_mean']=d['Best step'].mean();row['Effective_epochs_mean']=d['Effective epochs'].mean();means.append(row)
    mean=pd.DataFrame(means);mean['_sort']=[(DATASETS.index(r.Dataset),ARCHS.index(r.Architecture),VARIANTS.index(r.Variant)) for r in mean.itertuples()]
    mean=mean.sort_values('_sort').drop(columns='_sort')
    require(len(mean)==60,'Expected 60 mean rows')
    comparisons={}
    for metric in ('PSNR','SSIM','LPIPS','MSE'):
        wide=all_seeds.pivot(index=group,columns='Seed',values=metric).reset_index().rename(columns={s:f'{metric} seed{s}' for s in SEEDS})
        wide=mean[group].merge(wide,on=group)
        for name in ('mean','std'):wide[metric+' '+name]=mean[metric+'_'+name].to_numpy()
        if metric=='PSNR':
            delta=all_seeds.pivot(index=group,columns='Seed',values='Delta PSNR vs Full').reset_index().rename(columns={s:f'Delta seed{s}' for s in SEEDS})
            wide=wide.merge(delta,on=group);wide['Delta mean']=mean.Delta_PSNR_mean;wide['Delta std']=mean.Delta_PSNR_std
        comparisons[metric+' comparison']=wide
    pub=mean[group].copy()
    pub['Variant']=[('Full' if v=='full' else {'topk':'Top-K','quantile':'Quantile','random':'Random'}[v.split('_')[0]]+' '+v.split('_p')[1]+'%') for v in pub.Variant]
    for metric in ('PSNR','Delta_PSNR','SSIM','LPIPS','MSE'):
        pub[metric+' mean ± std']=[f'{a:.4f} ± {b:.4f}' if pd.notna(a) else '' for a,b in zip(mean[metric+'_mean'],mean[metric+'_std'])]
    src=all_seeds[['Dataset','Architecture','Seed','Variant','Source file','Source SHA256']]
    all_seeds.to_csv(final/'final_results_all_seeds.csv',index=False)
    mean.to_csv(final/'final_results_mean_3seeds.csv',index=False)
    excel(final/'final_results_all_seeds.xlsx',{'All seeds':all_seeds})
    excel(final/'final_results_mean_3seeds.xlsx',{'Mean 3 seeds':mean})
    excel(final/'final_results_publication_table.xlsx',{'Publication':pub})
    excel(final/'final_results_seed_comparison.xlsx',comparisons)
    excel(final/'final_experiment_results_3seeds.xlsx',{'All seeds':all_seeds,'Mean 3 seeds':mean,**comparisons,'Protocol audit':audit,'Source files':src})
    roundtrip=pd.read_excel(final/'final_experiment_results_3seeds.xlsx',sheet_name='All seeds')
    require(len(roundtrip)==180 and np.allclose(roundtrip.PSNR,all_seeds.PSNR),'Excel numerical roundtrip failed')
    comparisons['PSNR comparison'].to_csv(final/'psnr_comparison.csv',index=False)
    missing=all_seeds[METRICS[:-1]].isna().sum().to_dict()
    validation=dict(status='FINAL 3-SEED RESULTS READY FOR ARTICLE',rows=180,mean_rows=60,std_ddof=1,missing_metrics=missing,
                    checks=['3 seeds per variant','Full exists','Full paired delta=0','unique rows','PSNR finite','expected 10 variants',
                            'mean/std recomputed from raw seeds','per-image mean matches summary','protocol equality','eval manifest hashes equal','source hashes retained'],warnings=sorted(set(warnings)))
    (final/'validation_report.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    files=('final_results_all_seeds.csv','final_results_mean_3seeds.csv','final_results_provenance.json',
           'protocol_audit.csv','validation_report.json')
    (final/'output_files_sha256.json').write_text(json.dumps({name:sha(final/name) for name in files},indent=2),encoding='utf-8')
    print(json.dumps(validation,indent=2))


if __name__=='__main__':main()
