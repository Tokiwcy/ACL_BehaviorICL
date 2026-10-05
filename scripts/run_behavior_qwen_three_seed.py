"""Guarded independent tuple orchestration using unchanged historical recipes.

Imports model/scientific modules only after the cloud/native/backup guard.
Legacy mains run inside this sole scientific process, without subprocess GPU
children. Artifacts remain scoped by component/dataset/model/seed.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
from dataclasses import asdict, replace
import behavior_qwen_three_seed_policy as p

BASE=p.ROOT/'results'/p.SCIENCE
RECIPES={
    'features':('run_multidataset_main','baselines','features',False),
    'baseline_retrieve':('run_multidataset_main','baselines','retrieval',True),
    'baseline_generate':('run_multidataset_main','baselines','generation',True),
    'decision_retrieve':('run_dtd_decision_states','decision',None,False),
    'decision_generate':('run_dtd_decision_states','decision',None,True),
    'proj_retrieve':('run_behavior_qwen_three_seed_proj','proj','retrieval',False),
    'proj_generate':('run_behavior_qwen_three_seed_proj','proj','generation',True),
    'proxy_capture':('run_detriever_output_proxy','detriever','proxy',False),
    'detriever_retrieve':('run_detriever_output_proxy','detriever','retrieval',True),
    'detriever_generate':('run_detriever_output_proxy','detriever','generation',True)}


def command(dataset,seed,phase):
    p.tuple_identity(dataset,seed)
    if phase not in RECIPES:raise ValueError('Unknown frozen tuple stage')
    module,component,stage,resume=RECIPES[phase]
    args=['--dataset',dataset,'--seed',str(seed),'--output-root',str(BASE/component),
        '--cache-dir',str(p.ROOT/'.hf_cache'),'--shots','4','--generation-batch-size','1']
    if module=='run_multidataset_main':
        args+=['--model','qwen3vl4b','--methods','rices','gpt_mm','--extract-batch-size','1','--gpt-mm-new-tokens','12']
    else:args+=['--source-root',str(BASE/'baselines')]
    if module in ('run_dtd_decision_states','run_behavior_qwen_three_seed_proj'):args+=['--train-steps','1500']
    if module=='run_behavior_qwen_three_seed_proj':args+=['--projection-dim','256']
    if module=='run_detriever_output_proxy':args+=['--extract-batch-size','1','--detriever-steps','10000']
    if stage:args+=['--stage',stage]
    if phase=='decision_retrieve':args+=['--selection-only']
    if resume:args+=['--resume']
    return module,args


def official_samples(plan,dataset):
    from multidataset_protocol import load_dataset
    bank,query=load_dataset(dataset,p.ROOT/'datasets')
    saved=plan['datasets'][dataset]['manifest']
    for split,rows in (('bank',bank),('query',query)):
        if [(r.sample_id,r.label,r.split) for r in rows]!=[(r['sample_id'],r['label'],r['split']) for r in saved[split]]:
            raise ValueError('Actual complete dataset order differs')
        for sample,record in zip(rows,saved[split]):
            if p.digest(Path(sample.path))!=record['image_sha256']:raise ValueError('Official original image bytes changed')
    return bank,query


def smoke(plan,dataset,seed):
    p.gpu_guard(plan)
    import torch
    import multimodal_model_adapter as a
    bank,query=official_samples(plan,dataset);labels=plan['datasets'][dataset]['class_labels']
    sample=query[0]
    if a.zero_shot_messages(sample,dataset,labels,224)!=a.zero_shot_messages(replace(sample,label='TRUTH_EXCLUDED'),dataset,labels,224):
        raise ValueError('Query gold leaked into source prompt')
    model,processor=a.load_model(a.MODEL_SPECS['qwen3vl4b'],p.ROOT/'.hf_cache')
    if a.model_dimensions(model)!=(36,2560):raise ValueError('Native model dimensions differ')
    message=a.zero_shot_messages(sample,dataset,labels,224)
    inputs=a.prepare_batch(processor,[message]);states=a.anchor_hidden_states(model,inputs)
    if tuple(states.shape)!=(1,36,2560) or not torch.isfinite(states).all():raise ValueError('Native pre-answer smoke failed')
    zero=a.generate_legal_labels(model,processor,[message],labels)[0]
    context=a.icl_messages(bank[:4],sample,dataset,labels,224)
    if context!=a.icl_messages(bank[:4],replace(sample,label='TRUTH_EXCLUDED'),dataset,labels,224):raise ValueError('Query gold leaked into ICL')
    icl=a.generate_legal_labels(model,processor,[context],labels)[0]
    if zero[0] not in labels or icl[0] not in labels:raise ValueError('Real native legal-label generation failed')
    out=BASE/'smoke'/dataset/('seed_'+str(seed))
    p.immutable(out/'native_hidden_four_demo_legal.json',{'identity':p.tuple_identity(dataset,seed),
        'plan_hash':plan['plan_hash'],'states_shape':[1,36,2560],'query_id':sample.sample_id,
        'bank_demo_ids':[r.sample_id for r in bank[:4]],'zero':zero,'icl':icl,'formal_prediction_count':0,'utc':p.now()})


def execute(plan,dataset,seed,phase):
    auth=p.gpu_guard(plan);bank,query=official_samples(plan,dataset)
    module,args=command(dataset,seed,phase)
    job=next(r for r in plan['jobs'] if (r['dataset'],r['seed'],r['phase'])==(dataset,seed,phase))
    jobdir=BASE/'jobs'/dataset/('seed_'+str(seed))/phase
    if jobdir.exists():raise ValueError('Preserve consumed scientific phase; independent explicit recovery required')
    if phase!='features':
        source=p.tuple_directory('baselines',dataset,seed)
        p.validate_source_identity(p.read(source/'run_identity.json'),dataset,seed)
        progress=p.read(source/'anchor_states_progress.json')
        if progress.get('shape')!=[len(bank)+len(query),36,2560] or progress.get('completed')!=len(bank)+len(query):
            raise ValueError('Only complete same-tuple frozen states may be reused')
    if phase in ('features','decision_retrieve','proj_retrieve','proxy_capture'):
        component=RECIPES[phase][1]
        if p.tuple_directory(component,dataset,seed).exists():raise ValueError('Fresh tuple/component root required')
    p.immutable(jobdir/'job.json',{**job,'launch_id':auth['launch_id'],'utc':p.now(),
        'module':module,'argv':args,'source_tuple':p.tuple_identity(dataset,seed)})
    previous=sys.argv
    try:
        sys.argv=[module+'.py',*args]
        importlib.import_module(module).main()
    finally:sys.argv=previous
    p.immutable(jobdir/'complete.json',{'job_hash':job['job_hash'],'plan_hash':plan['plan_hash'],
        'identity':p.tuple_identity(dataset,seed),'utc':p.now(),'status':'phase_complete_not_full_test_audit'})


def verify(plan):
    import numpy as np
    groups=[];identities=set()
    for seed in p.SEEDS:
        for dataset in p.DATASETS:
            bank,query=official_samples(plan,dataset);labels=plan['datasets'][dataset]['class_labels']
            for phase in p.PHASES:
                complete=p.read(BASE/'jobs'/dataset/('seed_'+str(seed))/phase/'complete.json')
                job=next(r for r in plan['jobs'] if (r['dataset'],r['seed'],r['phase'])==(dataset,seed,phase))
                if complete['job_hash']!=job['job_hash'] or complete['identity']!=p.tuple_identity(dataset,seed):raise ValueError('Complete independent tuple job absent')
            source=p.tuple_directory('baselines',dataset,seed);p.validate_source_identity(p.read(source/'run_identity.json'),dataset,seed)
            array=np.load(source/'anchor_states.npy',mmap_mode='r',allow_pickle=False)
            if array.shape!=(len(bank)+len(query),36,2560):raise ValueError('Source array denominator differs')
            for component,methods in [('baselines',('rices','gpt_mm')),('decision',('decision_zero','decision_learn')),
                                     ('proj',('projection_only',)),('detriever',('detriever',))]:
                folder=p.tuple_directory(component,dataset,seed);identity=p.read(folder/'run_identity.json')
                if any(identity.get(k)!=v for k,v in p.tuple_identity(dataset,seed).items() if k!='model_slug'):
                    raise ValueError('Cross-tuple retriever/prediction artifact')
                saved=p.read(folder/'manifest.json')
                for split,rows in (('bank',bank),('query',query)):
                    if [(r['sample_id'],r['label'],r['split']) for r in saved[split]]!=[(r.sample_id,r.label,r.split) for r in rows]:
                        raise ValueError('Actual generated tuple manifest differs')
                selections=p.read(folder/'selections.json');records=[]
                with (folder/'predictions.jsonl').open(encoding='utf-8') as handle:
                    for line in handle:
                        if not line.endswith('\n'):raise ValueError('Truncated predictions')
                        records.append(json.loads(line))
                if len(records)!=len(methods)*len(query):raise ValueError('All official predictions required')
                by_bank={r.sample_id:r.label for r in bank}
                for method in methods:
                    chosen=[r for r in records if r['method']==method]
                    if [r['query_id'] for r in chosen]!=[r.sample_id for r in query]:raise ValueError('Duplicate/missing/unordered query')
                    if set(selections.get(method,{}))!={r.sample_id for r in query}:raise ValueError('Incomplete demonstration selection')
                    for sample,row in zip(query,chosen):
                        if row['target']!=sample.label or row['prediction'] not in labels or row['correct']!=(row['prediction']==sample.label):
                            raise ValueError('Illegal/misbound official prediction')
                        demos=selections[method][sample.sample_id]
                        if len(demos)!=4 or len({d['sample_id'] for d in demos})!=4 or any(by_bank.get(d['sample_id'])!=d['label'] for d in demos):
                            raise ValueError('Gold full-bank four-shot context differs')
                        if row['demo_ids']!=[d['sample_id'] for d in demos]:raise ValueError('Prediction/context identity differs')
                    groups.append({'dataset':dataset,'seed':seed,'method':method,'rows':len(chosen),
                        'predictions_SHA':p.digest(folder/'predictions.jsonl'),'selections_SHA':p.digest(folder/'selections.json')})
                    identities.add((dataset,seed,method))
            for component,metadata_name in [('decision','projected_state_metadata.json'),('proj','projection_only_metadata.json')]:
                metadata=p.read(p.tuple_directory(component,dataset,seed)/metadata_name)
                if (component=='proj' and metadata.get('train_steps')!=1500) or [r['step'] for r in metadata['history']]!=list(range(0,1501,100)):
                    raise ValueError('Complete fixed1500-step training required')
            detr=p.read(p.tuple_directory('detriever',dataset,seed)/'detriever_metadata.json')
            if detr['steps']!=10000 or detr['proxy']!='gold_input_answer_eos' or detr['layers_one_based']!=[1,6,11,16,21,26,31,36]:
                raise ValueError('Historical task-adapted DeTriever required')
    if len(identities)!=60 or sum(r['rows'] for r in groups)!=279072:raise ValueError('Complete new six-method five-dataset two-seed science required')
    p.immutable(BASE/'verification.json',{'status':'all_complete_verified','plan_hash':plan['plan_hash'],
        'groups':groups,'new_predictions':279072,'new_groups':60,'seed73_rerun':False,'independent_post_stop_audit':'pending','utc':p.now()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('preflight','smoke',*p.PHASES,'verify'),default='preflight')
    parser.add_argument('--dataset',choices=p.DATASETS);parser.add_argument('--seed',type=int,choices=p.SEEDS);parser.add_argument('--allow-gpu',action='store_true')
    args=parser.parse_args();plan=p.validate_plan(p.read(p.PLAN))
    if args.phase=='preflight':
        for dataset in p.DATASETS:official_samples(plan,dataset)
        print({'CPU_full_official_preflight':True,'new_predictions':279072,'GPU_execution':False})
    elif args.phase=='verify':verify(plan)
    else:
        if not args.allow_gpu or not args.dataset or args.seed is None:raise ValueError('Exact tuple and explicit cloud GPU guard required')
        if args.phase=='smoke':smoke(plan,args.dataset,args.seed)
        else:execute(plan,args.dataset,args.seed,args.phase)
