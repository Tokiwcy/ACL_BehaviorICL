"""Independent P0 zero-answer collector. No retrieval training or legacy queue.

Default phase is read-only preflight; capture requires an explicit GPU flag and
exact expected device. Original retrieval manifests/predictions remain untouched.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from contextlib import contextmanager

ROOT=Path(__file__).resolve().parents[1]
REVISION='ebb281ec70b05090aa6165b016eac8ec08e71b17'
MODEL='Qwen/Qwen3-VL-4B-Instruct'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def same_persisted_identity(saved, expected):
    # Manifest tuples become JSON arrays when the first smoke is persisted.
    # Compare their canonical serialized forms, not Python list/tuple types.
    return canonical(saved)==canonical(expected)

def atomic(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)

@contextmanager
def lock(path):
    with path.open('a+b') as f:
        if f.tell()==0: f.write(b'0'); f.flush()
        f.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: yield
        finally:
            f.seek(0)
            if os.name=='nt': msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(f.fileno(),fcntl.LOCK_UN)

def prefix(path,samples,job_hash):
    rows=[]
    if not path.exists(): return rows
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.endswith('\n'): raise ValueError('Truncated JSONL: retain evidence, do not silently repair')
            r=json.loads(line); i=len(rows)
            if i>=len(samples): raise ValueError('Too many zero output rows')
            s=samples[i]
            if (r['sample_id'],r['true_label'],r['split'],r['job_hash'])!=(s.sample_id,s.label,s.split,job_hash):
                raise ValueError('Zero prefix identity/order mismatch')
            if not isinstance(r['raw_zero_shot_answer'],str): raise ValueError('Missing raw answer')
            if digest(s.path)!=r['image_sha256']: raise ValueError('Image bytes changed since collection')
            rows.append(r)
    return rows

def prepare(dataset,datasets_root,cache_dir):
    from multidataset_protocol import load_dataset
    from run_behavior_priority_analysis import source_catalog, identity, read
    bank,query=load_dataset(dataset,datasets_root); manifest={'bank':[asdict(s) for s in bank],'query':[asdict(s) for s in query]}
    source=source_catalog()[dataset]['base']/'manifest.json'
    if identity(read(source))!=identity(manifest): raise ValueError('Local official ID/label/split/order differs from verified ICL bank/test')
    snapshot=cache_dir/f'models--Qwen--Qwen3-VL-4B-Instruct/snapshots/{REVISION}'
    for name in ('config.json','tokenizer.json','preprocessor_config.json','chat_template.json','model.safetensors.index.json'):
        if not (snapshot/name).is_file(): raise FileNotFoundError(snapshot/name)
    index=read(snapshot/'model.safetensors.index.json')
    for name in set(index['weight_map'].values()):
        if not (snapshot/name).is_file() or (snapshot/name).stat().st_size<1000: raise FileNotFoundError(snapshot/name)
    labels=sorted({s.label for s in bank})
    spec={'experiment':'behavior_priority_zero_v1','dataset':dataset,'model':MODEL,'revision':REVISION,
          'samples':identity(manifest),'labels':labels,'source_manifest_sha256':digest(source),
          'prompt_helper':'multimodal_model_adapter.zero_shot_messages','image_size_hint':224,'vision_pixels_override':None,
          'decoding':{'do_sample':False,'max_new_tokens':12,'legal_label_constraint':False,'batch_size':1},
          'purpose':'P0 full bank/test zero predictions; no representation regeneration, training, or label repair',
          'adapter_sha256':digest(ROOT/'scripts/multimodal_model_adapter.py'),
          'collector_sha256':digest(Path(__file__)),'dataset_loader_sha256':digest(ROOT/'scripts/multidataset_protocol.py')}
    spec['job_hash']=canonical(spec)
    return bank+query,manifest,snapshot,spec

def main():
    p=argparse.ArgumentParser(); p.add_argument('--phase',choices=['preflight','capture','verify'],default='preflight')
    p.add_argument('--dataset',choices=['cub','pets','dogs'],required=True)
    p.add_argument('--datasets-root',type=Path,default=ROOT/'datasets'); p.add_argument('--cache-dir',type=Path,default=ROOT/'.hf_cache')
    p.add_argument('--output-root',type=Path,default=ROOT/'results/behavior_priority_zero_v1')
    p.add_argument('--allow-gpu',action='store_true'); p.add_argument('--expected-device'); p.add_argument('--max-samples',type=int)
    a=p.parse_args()
    samples,manifest,snapshot,spec=prepare(a.dataset,a.datasets_root,a.cache_dir)
    dest=a.output_root.resolve()/a.dataset
    if not dest.is_relative_to(ROOT/'results/behavior_priority_zero_v1'): raise ValueError('Use separate P0 output group only')
    if a.phase=='preflight':
        print(json.dumps({'phase':'preflight_passed_no_gpu','dataset':a.dataset,'total':len(samples),
                          'bank':len(manifest['bank']),'query':len(manifest['query']),
                          'job_hash':spec['job_hash'],'cached_snapshot':str(snapshot)},indent=2)); return
    if not dest.is_dir() and a.phase=='verify': raise FileNotFoundError(dest)
    if a.phase=='capture' and (not a.allow_gpu or not a.expected_device): raise ValueError('Capture requires explicit GPU authorization and exact expected device')
    dest.mkdir(parents=True,exist_ok=True)
    with lock(dest/'capture.lock'):
        ip=dest/'identity.json'; outputs=dest/'zero_outputs.jsonl'
        if ip.exists():
            if not same_persisted_identity(json.loads(ip.read_text(encoding='utf-8')),spec):
                raise ValueError('P0 job identity changed')
        elif a.phase=='verify': raise FileNotFoundError(ip)
        else: atomic(ip,spec); atomic(dest/'manifest.json',manifest)
        rows=prefix(outputs,samples,spec['job_hash']); start=len(rows)
        if a.phase=='verify':
            if start!=len(samples): raise ValueError(f'Incomplete zero cache {start}/{len(samples)}')
            atomic(dest/'verification.json',{'status':'all_complete_verified','rows':start,'job_hash':spec['job_hash'],'outputs_sha256':digest(outputs)})
            print(f'Full P0 zero cache verified: {start}'); return
        if start==len(samples): print('Already complete; use --phase verify'); return
        # Do not acquire another model when an unrelated compute process is active.
        active=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
        if any(line.strip().isdigit() and int(line.strip())!=os.getpid() for line in active.stdout.splitlines()):
            raise RuntimeError('Existing GPU compute process: no second GPU job permitted')
        os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError('Require one CUDA GPU')
        device=torch.cuda.get_device_name(0)
        if device!=a.expected_device: raise RuntimeError(f'Unexpected GPU {device}')
        from multimodal_model_adapter import load_model, MODEL_SPECS, zero_shot_messages, prepare_batch
        import transformers
        atomic(dest/'runtime.json',{'device':device,'torch':torch.__version__,'transformers':transformers.__version__,'pid':os.getpid(),
                                   'original_gpu_environment_equivalence':'not_asserted','generation_compatibility':'must_audit_after_smoke'})
        model,processor=load_model(MODEL_SPECS['qwen3vl4b'],a.cache_dir,None)
        stop=len(samples) if a.max_samples is None else min(len(samples),start+a.max_samples)
        if stop<=start: raise ValueError('max-samples must be positive')
        with outputs.open('a',encoding='utf-8') as f:
            for i in range(start,stop):
                s=samples[i]; before=digest(s.path); t=time.perf_counter()
                inputs=prepare_batch(processor,[zero_shot_messages(s,a.dataset,spec['labels'],224)])
                n=int(inputs['input_ids'].shape[1]); torch.cuda.synchronize()
                with torch.inference_mode():
                    result=model.generate(**inputs,max_new_tokens=12,do_sample=False,use_cache=True)
                torch.cuda.synchronize(); tail=result[:,n:]; raw=processor.batch_decode(tail,skip_special_tokens=True)[0].strip()
                if digest(s.path)!=before: raise ValueError('Image changed during generation')
                legal={v.casefold():v for v in spec['labels']}; pred=legal.get(raw.casefold())
                r={'sample_id':s.sample_id,'true_label':s.label,'split':s.split,'job_hash':spec['job_hash'],
                   'raw_zero_shot_answer':raw,'zero_shot_pred':pred,'zero_shot_valid':pred is not None,
                   'zero_shot_correct':pred==s.label,'image_sha256':before,'output_tokens':int(tail.numel()),
                   'elapsed_seconds':time.perf_counter()-t}
                f.write(json.dumps(r,ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())
                atomic(dest/'progress.json',{'completed':i+1,'total':len(samples),'job_hash':spec['job_hash'],'status':'complete_capture_pending_verify' if i+1==len(samples) else 'running'})
                del inputs,result,tail
                if (i+1)%20==0: torch.cuda.empty_cache()
                if (i+1)%50==0 or i+1==stop: print(f'{a.dataset} zero {i+1}/{len(samples)}',flush=True)

if __name__=='__main__': main()
