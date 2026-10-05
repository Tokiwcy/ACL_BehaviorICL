"""Freeze complete six-method seed74/75 science and verified seed73 references.

CPU-only source/manifest/provenance preparation. No model import or CUDA.
The historical fixed256 sources and results stay unchanged.
"""
import argparse
import ast
import copy
from pathlib import Path
import behavior_qwen_three_seed_policy as p
import behavior_qwenvision_host_policy as vision
import behavior_internvl_five_v2_policy as second
import behavior_paper_queue_v2_policy as queue
from prepare_behavior_internvl_five_v2 import change_once

PROJ='scripts/run_behavior_qwen_three_seed_proj.py'
HISTORY='analysis/paper_p1_recheck_v1/audit.json'


def projected_source():
    text=(p.ROOT/'scripts/run_behavior_ablation.py').read_text(encoding='utf-8')
    text=change_once(text,'METHODS = ("final_projected", "layer_weights_only", "projection_only")','METHODS = ("projection_only",)')
    text=change_once(text,'if (args.seed, args.shots, args.train_steps, args.projection_dim) != (73, 4, 1500, 256):',
        'if args.seed not in (74,75) or (args.shots, args.train_steps, args.projection_dim) != (4, 1500, 256):')
    text=change_once(text,'Fixed main protocol requires seed73, 4-shot, 1500 steps, 256 dimensions','Independent multiseed protocol requires seed74/75, 4-shot, 1500 steps, 256 dimensions')
    return text


def local_import_closure(entries):
    modules={path.stem:path.relative_to(p.ROOT).as_posix() for path in (p.ROOT/'scripts').glob('*.py')}
    pending=list(entries);seen=set()
    while pending:
        relative=pending.pop()
        if relative in seen:continue
        seen.add(relative)
        tree=ast.parse((p.ROOT/relative).read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            names=[n.name.split('.')[0] for n in node.names] if isinstance(node,ast.Import) else [node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else []
            pending.extend(modules[name] for name in names if name in modules)
    return sorted(seen)


def checked_history(audit):
    if audit.get('status')!='all_30_groups_verified' or len(audit['results'])!=30:
        raise ValueError('Only complete independently verified historical main table may be reused')
    expected={(d,m) for d in p.DATASETS for m in ('RICES','GPT-MM','DeTriever','BehaviorICL-Raw','BehaviorICL-Proj','BehaviorICL-Learn')}
    if {(r['dataset'],r['method']) for r in audit['results']}!=expected:
        raise ValueError('Historical six-method five-dataset coverage differs')
    if any(r['total']!=p.COUNTS[r['dataset']][1] for r in audit['results']):raise ValueError('Sampled history cannot count as full test')
    for name,sha in audit['current_group_source_sha256'].items():
        if p.digest(p.ROOT/name)!=sha:raise ValueError('Previously audited seed73 source changed')
    return {'audit_path':HISTORY,'audit_SHA':p.digest(p.ROOT/HISTORY),
        'full_groups':copy.deepcopy(audit['results']),'source_SHA':copy.deepcopy(audit['current_group_source_sha256']),
        'GPU_rerun_authorized':False,'history_environment_difference_retained':True}


def prepare(finish_preserved=False):
    if p.PLAN.parent.exists():raise ValueError('Preserve independent seed protocol; never overwrite')
    if (p.ROOT/PROJ).exists() and not (finish_preserved and (p.ROOT/PROJ).read_text(encoding='utf-8')==projected_source()):
        raise ValueError('Preserve independently prepared projection source')
    queue.authority(p.read(p.ROOT/'.tools/behavior_priority_experiments_request.json'),p.read(p.ROOT/queue.QUEUE_PATH))
    full=vision.validate_plan(vision.read(vision.PLAN_PATH));binding=vision.validate_binding(full)
    public=second.validate_plan(p.read(second.PLAN))
    history=checked_history(p.read(p.ROOT/HISTORY))
    text=projected_source();compile(text,PROJ,'exec')
    if not (p.ROOT/PROJ).exists():
        with (p.ROOT/PROJ).open('x',encoding='utf-8',newline='\n') as handle:handle.write(text)
    data={d:{'manifest':copy.deepcopy(full['datasets'][d]['manifest']),
        'class_labels':copy.deepcopy(full['datasets'][d]['class_labels']),
        'splits':{str(seed):p.split_bank(full['datasets'][d]['manifest']['bank'],seed=seed) for seed in p.SEEDS}} for d in p.DATASETS}
    jobs=[]
    for seed in p.SEEDS:
        for dataset in p.DATASETS:
            for phase in p.PHASES:
                job={'dataset':dataset,'seed':seed,'phase':phase,'identity':p.tuple_identity(dataset,seed),
                    'protocol_SHA':p.canonical(p.PROTOCOL),'manifest_SHA':p.canonical(data[dataset]['manifest'])}
                job['job_hash']=p.canonical(job);jobs.append(job)
    entries=('scripts/behavior_qwen_three_seed_policy.py','scripts/run_behavior_qwen_three_seed.py',PROJ,
        'scripts/run_multidataset_main.py','scripts/run_dtd_decision_states.py','scripts/run_detriever_output_proxy.py')
    codes=local_import_closure(entries)+['scripts/prepare_behavior_qwen_three_seed.py','tests/test_behavior_qwen_three_seed.py']
    weights={'model':p.MODEL,'revision':p.REVISION,
        'files':{name:{'sha256':sha} for name,sha in binding['model_inventory']['files_SHA256'].items()}}
    value={'utc':p.now(),'protocol':p.PROTOCOL,'datasets':data,'jobs':jobs,
        'seed73_reference':history,'models':{p.MODEL:weights,p.CLIP:copy.deepcopy(public['models'][p.CLIP])},
        'environment':binding['environment'],'code_sha256':{r:p.digest(p.ROOT/r) for r in codes},
        'queue_hash':queue.QUEUE_HASH,'new_official_queries_per_seed':23256,'new_predictions':279072,
        'readiness':{'CPU_protocol_and_source_prepared':True,'cloud_operation_implemented':False,
            'actual_native_hidden_multidemo_legal_smoke_passed':False,'actual_new_prediction_count':0},
        'historical_provenance_verified_now':True,'test_tuning':False,'local_CUDA_execution':False}
    value['plan_hash']=p.canonical(value);p.validate_plan(value);p.immutable(p.PLAN,value)
    print({'CPU_plan_frozen':value['plan_hash'],'new_predictions':279072,'reuse_seed73_groups':30,
        'new_tuple_jobs':len(jobs),'GPU_requested':False},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--finish-preserved-cpu-preparation',action='store_true');args=parser.parse_args()
    if args.prepare or args.finish_preserved_cpu_preparation:prepare(args.finish_preserved_cpu_preparation)
    else:compile(projected_source(),PROJ,'exec');print({'CPU_source_preparation_check':True,'GPU_requested':False})
