"""CPU-only immutable task/generic prompt protocol, not a GPU execution entry."""
from pathlib import Path
from dataclasses import asdict
import argparse
import json
import behavior_p2_matched_policy as p

DATASETS=('dtd','cub')
GENERIC='Describe the image briefly.'
PLAN=p.ROOT/'analysis/task_conditioning_prompt_plan_v1/protocol_lock.json'
CODE=('scripts/prepare_behavior_task_prompt_plan.py','scripts/multimodal_model_adapter.py',
      'scripts/legal_label_decoding.py','scripts/multidataset_protocol.py')
PROTOCOL={
    'experiment':'behavior_task_conditioning_raw_v1','datasets':list(DATASETS),
    'arms':['task','generic'],'generic_prompt':GENERIC,
    'task_prompt':'unchanged original adapter zero_shot_messages/task_instruction with complete fixed class vocabulary',
    'representation':'fresh last input token before answer, all36 decoder blocks,2560 each, same position rule both arms',
    'model':'Qwen/Qwen3-VL-4B-Instruct','revision':'ebb281ec70b05090aa6165b016eac8ec08e71b17',
    'vision':'same original processor and224 message hints; no explicit pixel budget change',
    'capture':'two fresh arm sources on each identical ordered official bank/query image; no description/label generation during extraction',
    'retrieval':'untrained Raw: per-layer FP32 L2/cosine then uniform mean; top4, ascending actual score order; original bank order resolves exact ties',
    'training':'none; no W, learned layer weights, width grid or dev-ICL search',
    'downstream':'same original classification task/true bank labels/four-demo ICL/legal greedy decoding in both arms',
    'test':'all official DTD and CUB query IDs; labels evaluation only, never retrieval or test tuning',
    'comparison':'paired full-query accuracy/rescue/harm plus retrieval content;10k query-paired bootstrap CI',
    'selection_disclosure':'two datasets fixed before this experiment; old results already observed, so exploratory not a new unseen benchmark',
    'claim_limit':'joint effect of task instruction and label vocabulary on decoder representation, not isolated label-vocabulary or same-failure causality',
    'sources':'new independent arrays; old task arrays/predictions are not substituted or overwritten',
}


def messages(sample,dataset,labels,arm):
    from multimodal_model_adapter import image_part,zero_shot_messages
    if arm=='task':return zero_shot_messages(sample,dataset,labels,224)
    if arm!='generic':raise ValueError('Unknown frozen prompt arm')
    return [{'role':'user','content':[image_part(sample.path,224),{'type':'text','text':GENERIC}]}]


def prepare():
    old=p.read(p.ROOT/'.tools/paper_p2_cloud/package.json')
    for rel in CODE[1:]:
        if p.digest(p.ROOT/rel)!=old['code_sha256'][rel]:raise ValueError('Original downstream source changed: '+rel)
    origin=p.read(p.PLAN_PATH);runtime=p.read(p.ROOT/'analysis/paper_p2_qwenvision_fixed256_plan_v1/plan_lock.json')
    from multidataset_protocol import load_dataset
    from run_behavior_priority_analysis import identity
    data={}
    for dataset in DATASETS:
        selected=origin['datasets'][dataset]
        bank,query=load_dataset(dataset,p.ROOT/'datasets')
        if identity({'bank':[asdict(s) for s in bank],'query':[asdict(s) for s in query]})!=identity(selected['manifest']):
            raise ValueError('Full official manifest/order changed')
        for row in selected['manifest']['bank']+selected['manifest']['query']:
            if p.digest(p.ROOT/row['path'])!=row['image_sha256']:raise ValueError('Image SHA changed')
        data[dataset]={k:selected[k] for k in ('manifest','class_labels','original_manifest_sha256','original_manifest_path')}
    counts={dataset:{split:len(data[dataset]['manifest'][split]) for split in ('bank','query')} for dataset in DATASETS}
    value={'protocol':PROTOCOL,'datasets':data,'counts':counts,'origin_manifest_plan_sha256':p.digest(p.PLAN_PATH),
        'model_inventory':runtime['model_weight_and_processor_inventory'],'expected_environment':runtime['expected_environment'],
        'code_sha256':{rel:p.digest(p.ROOT/rel) for rel in CODE},
        'implementation_status':'CPU protocol lock only; GPU runner/source capture/generation compatibility pending',
        'actual_new_predictions':0,'new_GPU_entry_authorized_by_this_file':False}
    value['protocol_hash']=p.canonical(value);p.immutable(PLAN,value)
    print(json.dumps({'status':value['implementation_status'],'protocol_hash':value['protocol_hash'],'counts':counts,
        'planned_ICL_generations':2*sum(v['query'] for v in counts.values())}))


if __name__=='__main__':
    argparse.ArgumentParser().parse_args();prepare()
