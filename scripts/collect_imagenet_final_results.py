"""List and download saved results over Docker TLS."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import sys

PROJECT=Path(__file__).resolve().parents[1]
DEST=PROJECT/'results_final'
# Docker reads DOCKER_HOST and DOCKER_CERT_PATH from the caller's environment.
DOCKER=['docker','--tlsverify']
for option,variable in [('tlscacert','RESEARCH_DOCKER_CA'),('tlscert','RESEARCH_DOCKER_CERT'),('tlskey','RESEARCH_DOCKER_KEY')]:
    if os.environ.get(variable):DOCKER.append('--'+option+'='+os.environ[variable])
HELPER=['run','--rm','--pull=never','--read-only','--network=none','--cpus=2','--memory=8g','--memory-swap=8g',
        '-e','NVIDIA_VISIBLE_DEVICES=void','-e','PYTHONDONTWRITEBYTECODE=1',
        '-v','imagenet100_experiments:/workspace/experiments:ro',os.environ.get('RESEARCH_DOCKER_IMAGE','research-imagenet100:latest'),'python','-c']
INVENTORY=r'''
import json,hashlib
from pathlib import Path
import yaml
base=Path('/workspace/experiments');runs=[]
for cp in base.glob('*/*/00_protocol/resolved_config.yaml'):
 cfg=yaml.safe_load(cp.read_text());r=cp.parent.parent
 if cfg.get('dataset',{}).get('name')!='imagenet100' or cfg.get('seeds',{}).get('corpus_seed') not in (42,43,44):continue
 files=[];total=0;count=0;weights=0;groups={}
 for p in sorted(r.rglob('*')):
  if not p.is_file():continue
  rel=p.relative_to(r);size=p.stat().st_size;total+=size;count+=1
  groups[rel.parts[0]]=groups.get(rel.parts[0],0)+size
  if p.suffix in ('.pt','.pth','.ckpt','.safetensors'):weights+=size;continue
  if any('backup' in x or x=='interrupted' for x in rel.parts):continue
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
  files.append(dict(path=rel.as_posix(),size=size,sha256=h.hexdigest()))
 protocol=r/'00_protocol/protocol_summary.json'
 owner=r/'00_protocol/noise_replicate_owner.json'
 meta=json.loads(owner.read_text()) if owner.exists() else json.loads(protocol.read_text()) if protocol.exists() else {}
 runs.append(dict(root=str(r),run_id=meta.get('run_id',r.name),seed=cfg['seeds']['corpus_seed'],bytes=total,file_count=count,
                  tensor_checkpoint_bytes=weights,directories=groups,download_bytes=sum(x['size'] for x in files),files=files))
print(json.dumps(runs))
'''
ARCHIVE=r'''
import sys,json,tarfile
from pathlib import Path
root=Path(sys.argv[1]);names=json.load(sys.stdin)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as t:
 for name in names:
  p=root/name
  if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):raise RuntimeError('Unsafe source')
  t.add(p,arcname=name,recursive=False)
'''


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['inventory','download']);args=parser.parse_args()
    DEST.mkdir(exist_ok=True)
    manifest=DEST/'imagenet100_remote_inventory.json'
    if args.action=='inventory':
        runs=json.loads(subprocess.check_output(DOCKER+HELPER+[INVENTORY],timeout=300))
        if manifest.exists() and json.loads(manifest.read_text())!=runs:raise RuntimeError('Existing inventory differs; review before replacing')
        manifest.write_text(json.dumps(runs,indent=2),encoding='utf-8')
        for r in runs:print(json.dumps({k:v for k,v in r.items() if k!='files'}),flush=True)
        return
    runs=[r for r in json.loads(manifest.read_text()) if r['run_id']==Path(r['root']).name]
    if sorted(r['seed'] for r in runs)!=[42,43,44]:raise RuntimeError('Expected exactly three remote roots')
    for r in runs:
        target=DEST/'imagenet100'/f"seed{r['seed']}";target.mkdir(parents=True,exist_ok=True)
        byname={x['path']:x for x in r['files']};missing=[]
        for name,x in byname.items():
            p=target/name
            if p.exists():
                if p.stat().st_size!=x['size'] or sha(p)!=x['sha256']:raise RuntimeError('Existing destination differs: '+str(p))
            else:missing.append(name)
        print(f"seed={r['seed']} files={len(byname)} missing={len(missing)} bytes={r['download_bytes']}",flush=True)
        if missing:
            proc=subprocess.Popen(DOCKER+HELPER[:1]+['-i']+HELPER[1:]+[ARCHIVE,r['root']],stdout=subprocess.PIPE,stdin=subprocess.PIPE)
            proc.stdin.write(json.dumps(missing).encode());proc.stdin.close()
            try:
                with tarfile.open(fileobj=proc.stdout,mode='r|') as archive:
                    for i,member in enumerate(archive):
                        if member.name not in byname or not member.isfile():raise RuntimeError('Unexpected archive entry')
                        p=target/member.name
                        if not p.resolve().is_relative_to(target.resolve()):raise RuntimeError('Unsafe destination')
                        p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.download-part')
                        with archive.extractfile(member) as source,tmp.open('wb') as dest:
                            for b in iter(lambda:source.read(1024*1024),b''):dest.write(b)
                        x=byname[member.name]
                        if tmp.stat().st_size!=x['size'] or sha(tmp)!=x['sha256']:raise RuntimeError('Transfer verification failed')
                        tmp.replace(p)
                        if i%50==0:print(f"seed={r['seed']} downloaded={i+1}/{len(missing)}",flush=True)
                if proc.wait()!=0:raise RuntimeError('Remote archive failed')
            finally:
                if proc.poll() is None:proc.terminate()
        for name,x in byname.items():
            p=target/name
            if not p.is_file() or sha(p)!=x['sha256']:raise RuntimeError('Final hash verification failed: '+name)
        (target/'DOWNLOAD_VERIFIED.json').write_text(json.dumps(dict(remote_root=r['root'],files=len(byname),bytes=r['download_bytes'],all_sha256_verified=True),indent=2))
        print(f"seed={r['seed']} ALL_HASHES_VERIFIED",flush=True)


if __name__=='__main__':main()
