"""CPU-safe identity and execution contracts for six methods at seeds 74/75."""
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
from behavior_task_prompt_policy import read, canonical, digest, immutable, atomic, now
from behavior_p2_matched_policy import native_identity

ROOT=Path(__file__).resolve().parents[1]
SCIENCE='behavior_qwen_three_seed_v1'
OPERATION='behavior_qwen_three_seed_cloud_v1'
PLAN=ROOT/'analysis/paper_qwen_three_seed_plan_v1/plan_lock.json'
MODEL='Qwen/Qwen3-VL-4B-Instruct'
REVISION='ebb281ec70b05090aa6165b016eac8ec08e71b17'
CLIP='openai/clip-vit-base-patch32'
DATASETS=('dtd','aircraft','cub','pets','dogs')
SEEDS=(74,75)
METHODS=('rices','gpt_mm','detriever','decision_zero','projection_only','decision_learn')
COUNTS={'dtd':(3760,1880),'aircraft':(6667,3333),'cub':(5994,5794),'pets':(3680,3669),'dogs':(12000,8580)}
PHASES=('features','baseline_retrieve','baseline_generate','decision_retrieve','decision_generate',
    'proj_retrieve','proj_generate','proxy_capture','detriever_retrieve','detriever_generate')
PROTOCOL={'experiment':SCIENCE,'model':MODEL,'revision':REVISION,'datasets':list(DATASETS),
    'seeds':[73,74,75],'new_seeds':list(SEEDS),'methods':list(METHODS),
    'seed73':'reuse only independently audited 30 historical fixed256 complete official-test groups; no GPU rerun',
    'new_predictions':279072,'new_groups':60,'shots':4,
    'source':'fresh tuple-scoped native36x2560 last-input-token decoder states, embedding excluded, FP16 storage; query truth excluded',
    'source_reuse':'only within identical dataset/model/seed; no cross-seed states/checkpoints/selections/predictions',
    'vision':'existing model native processor defaults; historical resized224 message hints retained, no explicit pixel-budget override',
    'rices':'fixed CLIP ViT-B/32 L2 cosine, all bank',
    'gpt_mm':'historical generated12-token output embedding, no query gold',
    'raw':'existing FP32 per-layer L2 cosine uniform arithmetic mean; no training',
    'proj':'existing projection_only class and train_control; only fixed bias-free256 shared W, no layer weights',
    'learn':'existing projected_state train_variant; shared256 W plus36 softmax layer weights',
    'behavior_training':{'steps':1500,'projection_dim':256,'development_per_class':10,
        'classes_per_batch':16,'samples_per_class':4,'temperature':.07,'optimizer':'AdamW',
        'projection_lr':.0003,'layer_weights_lr':.0003,'weight_decay':.01,
        'development':'bank-only; training indices exclude dev; evaluate at0,100,...1500; earliest best dev top4 label fraction',
        'final_bank':'complete bank including dev; no test tuning or refit'},
    'detriever':{'adaptation':'historical bank_only_gold_input_answer_eos_dot_product_v1',
        'steps':10000,'batch_size':64,'positive_count':40,'negative_count':100,'memory_refresh':100,
        'lr':.0001,'temperature':.07,'optimizer':'AdamW betas(.9,.98) weight_decay.01',
        'layers_one_based':[1,6,11,16,21,26,31,36],'checkpoint':'final10000-step same-tuple weights',
        'gold_proxy':'bank only; fresh for each tuple; no query output supervision'},
    'retrieval':'existing topk4 descending then reverse; full official bank; historical tie behavior retained',
    'generation':'existing greedy legal-label trie, batch1, independent complete method predictions',
    'batching':'new extraction1/generation1 for padding and memory control; history may use larger batches and different runtime',
    'non_goals':'no selected-width-v2/gate, cross-tuple artifacts, favorable seed/subset or test-guided checkpoint'}


def split_bank(rows,seed):
    if seed not in SEEDS:raise ValueError('New bank split seed74/75 only')
    import numpy as np
    rng=np.random.default_rng(seed);labels=np.asarray([r['label'] for r in rows]);train=[];dev=[]
    for label in sorted(set(labels)):
        indices=np.flatnonzero(labels==label)
        if len(indices)<=11:raise ValueError('Insufficient examples for bank-only development split')
        order=rng.permutation(indices)
        dev.extend(order[:10].tolist());train.extend(order[10:].tolist())
    return {'train_ids':[rows[i]['sample_id'] for i in sorted(train)],
        'development_ids':[rows[i]['sample_id'] for i in sorted(dev)]}


def validate_split(manifest,split,seed):
    bank={r['sample_id'] for r in manifest['bank']};query={r['sample_id'] for r in manifest['query']}
    train=set(split['train_ids']);dev=set(split['development_ids'])
    if (split!=split_bank(manifest['bank'],seed) or bank & query or train & dev
        or train | dev != bank or len(train)!=len(split['train_ids']) or len(dev)!=len(split['development_ids'])):
        raise ValueError('Invalid, changed, cross-seed or leaking bank split')


def tuple_identity(dataset,seed):
    if dataset not in DATASETS or seed not in SEEDS:raise ValueError('Only new complete official tuples seed74/75')
    return {'dataset':dataset,'model_slug':'qwen3vl4b','model_id':MODEL,'seed':seed}


def tuple_directory(component,dataset,seed):
    tuple_identity(dataset,seed)
    if component not in ('baselines','decision','proj','detriever'):raise ValueError('Independent scientific component required')
    return ROOT/'results'/SCIENCE/component/dataset/'qwen3vl4b'/('seed_'+str(seed))


def validate_plan(plan):
    if plan.get('plan_hash')!=canonical({k:v for k,v in plan.items() if k!='plan_hash'}) or plan.get('protocol')!=PROTOCOL:
        raise ValueError('Exact three-seed six-method protocol changed')
    if tuple(plan['datasets'])!=DATASETS:raise ValueError('All five official datasets required')
    for dataset,data in plan['datasets'].items():
        manifest=data['manifest']
        if tuple(len(manifest[s]) for s in ('bank','query'))!=COUNTS[dataset]:raise ValueError('Incomplete test or bank')
        for seed in SEEDS:validate_split(manifest,data['splits'][str(seed)],seed)
        if data['class_labels']!=sorted({r['label'] for r in manifest['bank']}):raise ValueError('Class inventory differs')
    if len(plan['jobs'])!=10*len(PHASES) or len({r['job_hash'] for r in plan['jobs']})!=len(plan['jobs']):
        raise ValueError('Independent tuple/phase jobs required')
    expected=[(d,s,phase) for s in SEEDS for d in DATASETS for phase in PHASES]
    if [(r['dataset'],r['seed'],r['phase']) for r in plan['jobs']]!=expected:raise ValueError('Complete ordered jobs required')
    for job in plan['jobs']:
        if job['identity']!=tuple_identity(job['dataset'],job['seed']) or job['job_hash']!=canonical({k:v for k,v in job.items() if k!='job_hash'}):
            raise ValueError('Scientific job identity differs')
    for rel,sha in plan['code_sha256'].items():
        if digest(ROOT/rel)!=sha:raise ValueError('Frozen scientific source changed: '+rel)
    if digest(ROOT/plan['seed73_reference']['audit_path'])!=plan['seed73_reference']['audit_SHA']:
        raise ValueError('Audited history reference differs')
    return plan


def validate_source_identity(record,dataset,seed):
    if record!=tuple_identity(dataset,seed):raise ValueError('Frozen feature cache belongs to another tuple')


def gpu_guard(plan):
    if sys.platform!='linux' or str(ROOT)!='/workspace/BehaviorICL' or Path.cwd()!=ROOT:
        raise ValueError('Exact cloud only; local CUDA forbidden')
    validate_plan(plan);out=ROOT/'results'/OPERATION;auth=read(out/'execution_authorization.json')
    if (auth.get('experiment')!=SCIENCE or auth.get('plan_hash')!=plan['plan_hash']
        or auth.get('implementation_hash')!=plan['plan_hash'] or not auth.get('launch_id')
        or auth.get('actual_api_pod_id')!=auth.get('pod_id')):raise ValueError('Independent exact launch required')
    proof=out/'native_identity_receipt.json'
    if digest(proof)!=auth.get('native_identity_receipt_sha256') or read(proof)['pod_id']!=auth['pod_id']:
        raise ValueError('Fresh native/API proof binding absent')
    native_identity(auth['pod_id'])
    for filename in ('backup_ready.json','backup_heartbeat.json'):
        backup=read(out/filename)
        if (backup.get('status')!='actual_backup_succeeded' or backup.get('watcher_identity_verified') is not True
            or backup.get('file_count',0)<=0 or len(backup.get('archive_sha256',''))!=64
            or any(not auth.get(k) or auth[k]!=backup.get(k) for k in ('experiment','plan_hash','implementation_hash','pod_id','launch_id'))):
            raise ValueError('Actual successful independent backup absent')
        if not 0<=(datetime.now(timezone.utc)-datetime.fromisoformat(backup['utc'])).total_seconds()<=300:
            raise ValueError('Stale/future actual backup')
    if (out/'stop_request.json').exists():raise ValueError('Safety stop requested')
    if {name:importlib.metadata.version(name) for name in plan['environment']}!=plan['environment']:
        raise ValueError('Exact pinned runtime required')
    for model,inventory in plan['models'].items():
        cache=ROOT/'.hf_cache'/('models--'+model.replace('/','--'))/'snapshots'/inventory['revision']
        for name,item in inventory['files'].items():
            if digest(cache/name)!=item['sha256']:raise ValueError('Original model bytes differ')
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    if any(x.strip().isdigit() and int(x.strip())!=os.getpid() for x in active.splitlines()):raise ValueError('Other GPU compute present')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or torch.cuda.get_device_name(0)!=auth['device']:
        raise ValueError('Exact sole GPU required')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    return auth
