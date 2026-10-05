"""Create an independent five-dataset science from the frozen reviewed recipe.

CPU file preparation only. Exact counted source edits expand scope and nouns;
the old two-dataset policy, adapter, runner and package are never modified.
No weights, author import, CUDA, network or cloud start.
"""
import argparse
import copy
import json
from pathlib import Path

import behavior_internvl_policy as old
import behavior_qwenvision_host_policy as visual
import behavior_paper_queue_v2_policy as authority

ROOT=old.ROOT
PLAN=ROOT/'analysis/second_vlm_five_dataset_v2/plan_lock.json'
DATASETS=('dtd','aircraft','cub','pets','dogs')
COUNTS={'dtd':(3760,1880),'aircraft':(6667,3333),'cub':(5994,5794),'pets':(3680,3669),'dogs':(12000,8580)}


def change_once(text,before,after):
    if text.count(before)!=1:raise ValueError('Exact original source span must occur once: '+before)
    return text.replace(before,after,1)


def new_sources():
    policy=(ROOT/'scripts/behavior_internvl_policy.py').read_text(encoding='utf-8')
    edits=[("SCIENCE = 'behavior_internvl_generalization_v1'","SCIENCE = 'behavior_internvl_five_dataset_v2'"),
        ("OPERATION = 'behavior_internvl_generalization_cloud_v1'","OPERATION = 'behavior_internvl_five_dataset_cloud_v2'"),
        ("analysis/second_vlm_dtd_cub_plan_v1/plan_lock.json","analysis/second_vlm_five_dataset_v2/plan_lock.json"),
        ("DATASETS = ('dtd', 'cub')","DATASETS = ('dtd', 'aircraft', 'cub', 'pets', 'dogs')"),
        ("'official_queries': 7674, 'planned_predictions': 30696","'official_queries': 23256, 'planned_predictions': 93024"),
        ("expected = (3760, 1880) if dataset == 'dtd' else (5994, 5794)","expected = {'dtd': (3760,1880), 'aircraft': (6667,3333), 'cub': (5994,5794), 'pets': (3680,3669), 'dogs': (12000,8580)}[dataset]")]
    for before,after in edits:policy=change_once(policy,before,after)
    adapter=(ROOT/'scripts/behavior_internvl_adapter.py').read_text(encoding='utf-8')
    adapter=change_once(adapter,"NOUNS = {'dtd': 'texture category', 'cub': 'bird species'}",
        "NOUNS = {'dtd': 'texture category', 'aircraft': 'aircraft variant', 'cub': 'bird species', 'pets': 'pet breed', 'dogs': 'dog breed'}")
    runner=(ROOT/'scripts/run_behavior_internvl_generalization.py').read_text(encoding='utf-8')
    for before,after in [("full DTD+CUB","full five-dataset"),
        ('import behavior_internvl_policy as p','import behavior_internvl_five_v2_policy as p'),
        ('import behavior_internvl_adapter as adapter','import behavior_internvl_five_v2_adapter as adapter'),
        ("if sum(row['rows'] for row in groups) != 30696 or len(groups) != 8:","if sum(row['rows'] for row in groups) != 93024 or len(groups) != 20:"),
        ('Need every method on both complete official tests','Need every method on all five complete official tests'),
        ("'groups': groups, 'predictions': 30696","'groups': groups, 'predictions': 93024")]:
        runner=change_once(runner,before,after)
    return {'scripts/behavior_internvl_five_v2_policy.py':policy,
        'scripts/behavior_internvl_five_v2_adapter.py':adapter,
        'scripts/run_behavior_internvl_five_v2.py':runner}


def prepare():
    if PLAN.parent.exists():raise ValueError('Preserve new five-dataset science; never reprepare')
    queue=old.read(ROOT/authority.QUEUE_PATH)
    authority.authority(old.read(ROOT/'.tools/behavior_priority_experiments_request.json'),queue)
    previous=old.validate_plan(old.read(old.PLAN))
    full=visual.validate_plan(visual.read(visual.PLAN_PATH));visual.validate_binding(full)
    original_sha={name:old.digest(ROOT/name) for name in ('scripts/behavior_internvl_policy.py',
        'scripts/behavior_internvl_adapter.py','scripts/run_behavior_internvl_generalization.py')}
    outputs=new_sources()
    for relative,text in outputs.items():
        compile(text,relative,'exec')
        path=ROOT/relative
        if path.exists():raise ValueError('Preserve new scientific source: '+relative)
    for relative,text in outputs.items():
        with (ROOT/relative).open('x',encoding='utf-8',newline='\n') as handle:handle.write(text)
    import behavior_internvl_five_v2_policy as p
    data={d:{'manifest':copy.deepcopy(full['datasets'][d]['manifest']),
        'class_labels':copy.deepcopy(full['datasets'][d]['class_labels']),
        'split':p.split_bank(full['datasets'][d]['manifest']['bank'])} for d in DATASETS}
    code=set(previous['code_sha256'])-set(original_sha)
    code.update(outputs);code.add('scripts/prepare_behavior_internvl_five_v2.py')
    code.add('tests/test_behavior_internvl_five_v2.py')
    images=0
    for d,value in data.items():
        if tuple(len(value['manifest'][s]) for s in ('bank','query'))!=COUNTS[d]:raise ValueError('Full official counts differ')
        for s in ('bank','query'):
            for row in value['manifest'][s]:
                if p.digest(ROOT/row['path'])!=row['image_sha256']:raise ValueError('Actual full image byte SHA differs')
                images+=1
    plan={'protocol':p.PROTOCOL,'datasets':data,'environment':copy.deepcopy(previous['environment']),
        'models':copy.deepcopy(previous['models']),'code_sha256':{name:p.digest(ROOT/name) for name in sorted(code)},
        'original_public_source_inventory_SHA':previous['original_public_source_inventory_SHA'],
        'old_two_dataset_plan_SHA':p.digest(old.PLAN),'old_recipe_source_SHA':original_sha,
        'full_manifest_origin_SHA':p.digest(visual.PLAN_PATH),'human_queue_hash':authority.QUEUE_HASH,
        'actual_weights_verified':False,'actual_cloud_compatibility':False,'planned_source_rows':55357,
        'planned_official_predictions':93024,'actual_source_rows':0,'actual_predictions':0,
        'operational_entry_implemented':False,'inheritance':'reviewed recipe; only explicit scope/count/noun changes in new files'}
    plan['plan_hash']=p.canonical(plan);p.validate_plan(plan)
    PLAN.parent.mkdir();p.immutable(PLAN,plan)
    p.immutable(PLAN.parent/'CPU_preparation_audit.json',{'utc':p.now(),'status':'CPU_full_five_protocol_frozen_not_cloud_compatibility',
        'plan_hash':plan['plan_hash'],'images_SHA_verified':images,'full_groups':20,'planned_predictions':93024,
        'old_scientific_sources_unchanged':all(p.digest(ROOT/n)==sha for n,sha in original_sha.items()),
        'author_imported':False,'weights_downloaded':False,'local_CUDA':False,'operation_ready':False})
    print(json.dumps({'CPU_new_five_dataset_plan_frozen':True,'plan_hash':plan['plan_hash'],'images':images,'predictions':93024}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    else:print(json.dumps({'CPU_only':True,'requires_explicit_prepare':True,'GPU':False}))
