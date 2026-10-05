"""Independent post-stop full official-test audit for QV, InternVL and seeds.

Own archive, array, checkpoint, context, ordered-output and recipe checks.
Does not call the scientific runner's verify or execute models/retrievers.
CPU tokenizer use is restricted to the original pinned public source bytes.
"""
import argparse
import ast
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

import behavior_task_prompt_policy as p
import behavior_paper_queue_v3_canonical_v4_policy as queue
import behavior_paper_post_stop_slots_v2 as slots
from analyze_behavior_repair_all_queries import paired_bootstrap

ROOT=p.ROOT
BINDING=ROOT/'analysis/paper_qwenvision_two_complete_CPU_v3/implementation_lock.json'
ROLES=('qwenvision',)
FILES=('scripts/analyze_behavior_qwenvision_two_complete_cpu_v3.py','scripts/prepare_behavior_qwenvision_two_audit_v3.py','scripts/behavior_paper_post_stop_slots_v2.py','tests/test_behavior_qwenvision_two_audit_v3.py')


def config(role):
    if role=='qwenvision':from behavior_qwenvision_two_cloud_v1 import config as selected
    elif role=='internvl_five':from behavior_internvl_five_native_cloud_v4 import config as selected
    elif role=='qwen_three_seed':from behavior_qwen_three_seed_cloud_v1 import config as selected
    else:raise ValueError('Only complete new queue roles')
    return selected(role)


def freeze():
    if BINDING.parent.exists():raise ValueError('Preserve independent CPU audit binding')
    scopes={}
    for role in ROLES:
        cfg=config(role);scopes[role]={'plan_hash':cfg['plan_hash'],'implementation_hash':cfg['implementation_hash'],
            'environment':cfg['environment'],'models':cfg['models'],'expected':slots.EXPECTED[role]}
    from prepare_behavior_qwen_three_seed import local_import_closure
    bound=local_import_closure(FILES)
    value={'utc':p.now(),'code_sha256':{n:p.digest(ROOT/n) for n in bound},'scopes':scopes,
        'CPU_only':True,'bootstrap_repetitions':10000,'bootstrap_seed':73,
        'numerical_retrieval_kernel_or_model_replay':False,'cloud_execution':False}
    value['audit_implementation_hash']=p.canonical(value);p.immutable(BINDING,value);return value


def implementation_check():
    value=p.read(BINDING)
    if value['audit_implementation_hash']!=p.canonical({k:v for k,v in value.items() if k!='audit_implementation_hash'}):raise ValueError('CPU audit binding changed')
    for name,sha in value['code_sha256'].items():
        if p.digest(ROOT/name)!=sha:raise ValueError('Frozen CPU audit code changed')
    for role,saved in value['scopes'].items():
        cfg=config(role)
        if saved!={'plan_hash':cfg['plan_hash'],'implementation_hash':cfg['implementation_hash'],
            'environment':cfg['environment'],'models':cfg['models'],'expected':slots.EXPECTED[role]}:
            raise ValueError('Frozen audited scope changed')
    return value


def lines(path):
    result=[]
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if not line.endswith('\n'):raise ValueError('Truncated scientific record; preserve and reject')
            result.append(json.loads(line))
    return result


def array_audit(path,shape,sha=None,dtype='float16'):
    import numpy as np
    if sha is not None and p.digest(path)!=sha:raise ValueError('Original array SHA changed')
    array=np.load(path,mmap_mode='r',allow_pickle=False)
    if list(array.shape)!=list(shape) or array.dtype!=np.dtype(dtype):raise ValueError('Wrong full array shape or dtype')
    for offset in range(0,len(array),64):
        if not np.isfinite(array[offset:offset+64]).all():raise ValueError('Nonfinite original full source array')
    return {'sha256':p.digest(path),'shape':list(shape),'dtype':dtype,'all_rows_finite':True}


def numeric_array_SHA(path):
    """Historical DeTriever hashes array values, excluding the NPY header."""
    import numpy as np
    array=np.load(path,mmap_mode='r',allow_pickle=False);sha=hashlib.sha256()
    for start in range(0,len(array),64):sha.update(np.ascontiguousarray(array[start:start+64]).tobytes())
    return sha.hexdigest()


def full_history(metadata,nested=False):
    history=metadata['history']
    if [r['step'] for r in history]!=list(range(0,1501,100)):raise ValueError('Complete 1500-step history required')
    metric=lambda r:r['validation']['top4_same_label_fraction'] if nested else r['dev_top4_same_label_fraction']
    if any(not math.isfinite(float(metric(r))) or not 0<=metric(r)<=1 for r in history):raise ValueError('Invalid development metric')
    best=min(history,key=lambda r:(-metric(r),r['step']))['step']
    if metadata['best_step']!=best:raise ValueError('Checkpoint must use earliest best bank-development step')
    return best


def checkpoint(path,width,layers,best,learned=False,context=None):
    import torch
    value=torch.load(path,map_location='cpu',weights_only=True);model=value['model']
    if value.get('step')!=best or (context is not None and value.get('context')!=context):raise ValueError('Exact chosen checkpoint identity differs')
    if model.get('projection.weight') is None or tuple(model['projection.weight'].shape)!=(256,width):raise ValueError('Bias-free fixed256 projection shape differs')
    if any('bias' in name for name in model) or set(model)-{'projection.weight','layer_logits'}:raise ValueError('Unexpected trained projection model parameters')
    for value in model.values():
        if value.device.type!='cpu' or not torch.isfinite(value).all():raise ValueError('Finite CPU checkpoint tensors required')
    logits=model.get('layer_logits')
    if learned:
        if logits is None or tuple(logits.shape)!=(layers,):raise ValueError('Learn requires its own complete layer weights')
    elif logits is not None and (tuple(logits.shape)!=(layers,) or not torch.equal(logits,torch.zeros_like(logits))):
        raise ValueError('Projection-only context must have frozen uniform layers')
    return {'sha256':p.digest(path),'chosen_step':best,'projection_shape':[256,width],'CPU_tensor_check':True}


def selections(data,selected,method,legacy=False):
    bank={r['sample_id']:r['label'] for r in data['manifest']['bank']};query=data['manifest']['query']
    if set(selected)!={r['sample_id'] for r in query}:raise ValueError('Incomplete or extra selection query')
    result={}
    for row in query:
        value=selected[row['sample_id']]
        if legacy:
            ids=[v['sample_id'] for v in value];labels=[v['label'] for v in value]
        else:ids=value['demo_ids'];labels=value['demo_labels']
        expected=0 if method=='zero' else 4
        if len(ids)!=expected or len(set(ids))!=expected or labels!=[bank.get(i) for i in ids] or any(i not in bank for i in ids):
            raise ValueError('Original full-bank distinct demo IDs/labels/order required')
        if not legacy:
            scores=value['scores']
            if len(scores)!=expected or any(not math.isfinite(float(x)) for x in scores) or any(a>b+1e-6 for a,b in zip(scores,scores[1:])):
                raise ValueError('Saved similarity scores must be finite in ascending demo order')
        result[row['sample_id']]={'demo_ids':ids,'demo_labels':labels}
    return result


def prediction_audit(data,rows,selected,context=None,plan_hash=None):
    query=data['manifest']['query'];labels=data['class_labels']
    if len(rows)!=len(query) or [r.get('query_id') for r in rows]!=[r['sample_id'] for r in query]:raise ValueError('All official ordered queries required')
    for target,row in zip(query,rows):
        demos=selected[target['sample_id']]
        if (row.get('target')!=target['label'] or row.get('prediction') not in labels or type(row.get('correct')) is not bool
            or row['correct']!=(row['prediction']==target['label']) or row.get('demo_ids')!=demos['demo_ids']
            or row.get('demo_labels')!=demos['demo_labels']):raise ValueError('Official target/legality/context/correctness differs')
        if context is not None:
            if row.get('context_hash')!=context or row.get('plan_hash')!=plan_hash or row.get('input_tokens',0)<=0 or row.get('output_tokens',0)<=0 or row.get('seconds',0)<=0:
                raise ValueError('Actual measured generation/context evidence absent')
    return rows


def shared_training(cfg,science,dataset,rep,width,layers,provenance):
    data=cfg['plan']['datasets'][dataset];folder=science/dataset/'trained'/rep/'d256'
    source=science/dataset/'sources'/(rep+'.npy');context=p.read(folder/'context.json')
    if context!={'plan_hash':cfg['plan_hash'],'source_sha256':p.digest(source),'dataset':dataset,
        'representation':rep,'width':256,'split_hash':p.canonical(data['split'])}:raise ValueError('Training source/split/context differs')
    verified=p.read(folder/'training_verified.json');best=full_history(verified)
    if verified['context_hash']!=p.canonical(context) or verified['best_sha256']!=p.digest(folder/'best.pt') or verified['trainable_parameters']!=width*256:
        raise ValueError('Original training receipt/model capacity differs')
    provenance[str(folder.relative_to(science))]=checkpoint(folder/'best.pt',width,layers,best,context=context)
    last=__import__('torch').load(folder/'last.pt',map_location='cpu',weights_only=True)
    if last['step']!=1500 or last['context']!=context or last['history']!=verified['history'] or last['best_step']!=best:
        raise ValueError('Complete optimizer/checkpoint/1500-step history differs')


def native_cpu_tokenizer(cfg):
    model=cfg['policy'].MODEL;inventory=cfg['models'][model]
    folder=ROOT/'.tools/paper_queue_CPU_tokenizers_v1/internvl'/inventory['revision']
    from prepare_behavior_queue_CPU_tokenizers_v1 import FILES as tokenizer_files
    for name in tokenizer_files:
        if p.digest(folder/name)!=inventory['files'][name]['sha256']:raise ValueError('Pinned CPU tokenizer bytes differ')
    sys.path.insert(0,str(ROOT/'.tools/paper_CPU_audit_dependencies_v1'))
    # This exact reviewed tokenizer class only tokenizes text on the CPU.
    spec=importlib.util.spec_from_file_location('behavior_fixed_CPU_tokenizer_v1',folder/'tokenization_internlm2.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module.InternLM2Tokenizer.from_pretrained(folder,local_files_only=True)


def native_prompt(cfg,dataset,demo_labels,patch_counts):
    import behavior_internvl_five_v2_adapter as adapter
    path=ROOT/'analysis/second_vlm_source_inventory_v1/source/conversation.py'
    if p.digest(path)!=cfg['models'][cfg['policy'].MODEL]['files']['conversation.py']['sha256']:raise ValueError('Original native conversation source changed')
    tree=ast.parse(path.read_text(encoding='utf-8'));template=None
    for node in tree.body:
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='register_conv_template':
            call=node.value.args[0];keys={v.arg:v.value for v in call.keywords}
            if ast.literal_eval(keys['name'])=='internvl2_5':template=keys;break
    if template is None or ast.unparse(template['sep_style'])!='SeparatorStyle.MPT':raise ValueError('Frozen native MPT template required')
    literals={k:ast.literal_eval(template[k]) for k in ('system_template','system_message','roles','sep')}
    text=adapter.question(dataset,cfg['plan']['datasets'][dataset]['class_labels'],demo_labels)
    prompt=literals['system_template'].format(system_message=literals['system_message'])+literals['sep']+literals['roles'][0]+text+literals['sep']+literals['roles'][1]
    if len(patch_counts)!=(5 if demo_labels else 1) or any(type(v) is not int or not 1<=v<=3 for v in patch_counts):raise ValueError('Actual native patch budget differs')
    for count in patch_counts:prompt=prompt.replace('<image>','<img>'+'<IMG_CONTEXT>'*(256*count)+'</img>',1)
    if '<image>' in prompt:raise ValueError('Native image placeholders remain')
    return prompt,literals['sep'].strip()


def fresh_source_guard(cfg,science):
    source=ROOT/'analysis/qwenvision_two_fresh_v1/implementation_lock.json'
    binding=p.read(source)
    if binding['source_contract_hash']!=p.canonical({k:v for k,v in binding.items() if k!='source_contract_hash'}):
        raise ValueError('Independent native source contract changed')
    for name,sha in binding['code_sha256'].items():
        if p.digest(ROOT/name)!=sha:raise ValueError('Exact fresh-source implementation changed')
    if (binding['original_plan_hash']!=cfg['plan_hash'] or binding['original_host_implementation_hash']!=cfg['implementation_hash']
        or binding['original_manifest_hash']!=p.canonical(cfg['manifest'])
        or binding['images']!=17428 or binding['predictions']!=7674 or binding['fixed_width']!=256
        or binding['seed']!=73 or binding['train_steps']!=1500 or binding['numerical_tolerance_added'] is not False):
        raise ValueError('Original scientific scope or exact source contract differs')
    auth=cfg['audit_auth'];receipt=p.read(science/'source_execution_contract.json')
    expected={'plan_hash':cfg['plan_hash'],'source_contract_hash':binding['source_contract_hash'],
        'pod_id':auth['pod_id'],'launch_id':auth['launch_id'],'all_two_sources_newly_captured':True,
        'original_cache_used':False,'original_formula_preserved':True}
    if receipt!=expected or auth.get('source_contract_hash')!=binding['source_contract_hash']:
        raise ValueError('Actual native launch/source contract identity differs')
    return binding['source_contract_hash']


def modern_groups(cfg,science,provenance):
    role=cfg['mode'];plan=cfg['plan'];groups={};native=role=='internvl_five'
    tokenizer=native_cpu_tokenizer(cfg) if native else None
    fresh_source=fresh_source_guard(cfg,science) if role=='qwenvision' and (science/'source_execution_contract.json').exists() else None
    for dataset,data in plan['datasets'].items():
        cfg['policy'].validate_split(data['manifest'],data['split'])
        src=science/dataset/'sources';verified=p.read(src/'verification.json');count=len(data['manifest']['bank'])+len(data['manifest']['query'])
        if verified.get('status')!='all_complete_verified' or verified.get('plan_hash')!=cfg['plan_hash']:raise ValueError('Complete source identity required')
        expected={'behavior':(count,32,4096),'clip':(count,1,512)} if native else {'qwenvision':(count,1,1024)}
        if set(verified['sources'])!=set(expected):raise ValueError('Frozen representation inventory differs')
        for rep,shape in expected.items():provenance[str((src/(rep+'.npy')).relative_to(science))]=array_audit(src/(rep+'.npy'),shape,verified['sources'][rep]['sha256'])
        reuse=plan.get('complete_cache_reuse',{}).get(dataset) if fresh_source is None else None
        if fresh_source is not None:
            smoke=p.read(src/'smoke.json')
            if verified.get('reuse') is not None or smoke.get('status')!='actual_native_QwenVision_passed' or smoke.get('cache_exact_FP16_reproduction') is not False:
                raise ValueError('Each fresh source needs real native smoke and no cache origin')
        logs=lines(src/('capture.jsonl' if native else 'vision_capture.jsonl'))
        if not reuse:
            ordered=data['manifest']['bank']+data['manifest']['query']
            if len(logs)!=count or any((r['sample_id'],r['image_sha256'],r['plan_hash'])!=(ordered[i]['sample_id'],ordered[i]['image_sha256'],cfg['plan_hash']) for i,r in enumerate(logs)):
                raise ValueError('Actual complete source rows/ordered image SHA differ')
            for row in logs:
                if row.get('input_tokens',0)<=0 or row.get('seconds',0)<=0:raise ValueError('Actual source measurement absent')
                if native:
                    prompt,_=native_prompt(cfg,dataset,[],row['patch_counts'])
                    if row.get('prompt_fingerprint')!=p.canonical(prompt) or row['input_tokens']!=len(tokenizer.encode(prompt)) or row.get('visual_tokens')!=sum(row['patch_counts'])*256:
                        raise ValueError('Actual complete native source prompt/token/visual evidence differs')
                elif row.get('native_vision_token_count',0)<=0:raise ValueError('Actual native vision token capture absent')
        else:
            if verified['sources']['qwenvision']['sha256']!=reuse['sha256'] or p.digest(ROOT/reuse['local_folder']/'qwen_capture.jsonl')!=reuse['log_SHA']:
                raise ValueError('Original allowed complete cache provenance differs')
            originals=lines(ROOT/reuse['local_folder']/'qwen_capture.jsonl');ordered=data['manifest']['bank']+data['manifest']['query']
            if len(originals)!=count or any((r['sample_id'],r['image_sha256'],r['plan_hash'])!=(ordered[i]['sample_id'],ordered[i]['image_sha256'],reuse['origin_plan_hash']) for i,r in enumerate(originals)):
                raise ValueError('Complete original cache ordered image provenance differs')
            smoke=p.read(src/'smoke.json')
            if smoke.get('cache_exact_FP16_reproduction') is not True or smoke.get('status')!='actual_native_QwenVision_passed':raise ValueError('Actual original-cache numerical first-row smoke absent')
        shared_training(cfg,science,dataset,'behavior' if native else 'qwenvision',4096 if native else 1024,32 if native else 1,provenance)
        methods=cfg['policy'].METHODS if native else ('supcon_qwenvision',)
        for method in methods:
            folder=science/dataset/'test'/method
            if not native:folder=folder/'d256'
            context=p.read(folder/'context.json')
            if context.get('context_hash')!=p.canonical({k:v for k,v in context.items() if k!='context_hash'}) or context.get('plan_hash')!=cfg['plan_hash'] or context.get('candidate_ids')!=[r['sample_id'] for r in data['manifest']['bank']] or context.get('query_ids')!=[r['sample_id'] for r in data['manifest']['query']]:
                raise ValueError('Full final candidate/test context differs')
            if native:
                expected_sources=verified['sources'] if method!='zero' else None
                expected_checkpoint=p.digest(science/dataset/'trained/behavior/d256/best.pt') if method=='behavior_proj' else None
                if context.get('source_hash')!=expected_sources or context.get('checkpoint_sha256')!=expected_checkpoint:raise ValueError('Actual retrieval source/checkpoint binding differs')
            elif context.get('source_hash')!=verified['sources']['qwenvision']['sha256'] or context.get('checkpoint_SHA')!=p.digest(science/dataset/'trained/qwenvision/d256/best.pt'):
                raise ValueError('Actual vision-control source/checkpoint binding differs')
            saved=p.read(folder/'selections.json');selected=selections(data,saved,method)
            rows=prediction_audit(data,lines(folder/'predictions.jsonl'),selected,context['context_hash'],cfg['plan_hash'])
            receipt=p.read(folder/'verification.json')
            if receipt.get('status')!='all_complete_verified' or receipt['predictions_sha256']!=p.digest(folder/'predictions.jsonl'):raise ValueError('Independent full predictions SHA differs')
            if not (folder/'smoke.json').exists():raise ValueError('Real legal consumer smoke absent')
            if native:
                import behavior_internvl_five_v2_adapter as adapter
                codec=p.read(folder/'codec.json');_,sep=native_prompt(cfg,dataset,[],[1]);owners,eos=adapter.label_codec(tokenizer,data['class_labels'],sep)
                if codec!={'eos':eos,'owners':[{'tokens':list(k),'label':v} for k,v in owners.items()]}:raise ValueError('Actual label codec differs from original native tokenizer')
                for row in rows:
                    tokens=row['actual_output_token_ids'];prompt,_=native_prompt(cfg,dataset,row['demo_labels'],row['patch_counts'])
                    if (len(tokens)<2 or tokens[-1]!=eos or eos in tokens[:-1] or owners.get(tuple(tokens[:-1]))!=row['prediction']
                        or tokenizer.decode(tokens[:-1],skip_special_tokens=False)!=row['raw_answer']
                        or row['output_tokens']!=len(tokens) or row['prompt_fingerprint']!=p.canonical(prompt)
                        or row['input_tokens']!=len(tokenizer.encode(prompt)) or row['visual_tokens']!=sum(row['patch_counts'])*256
                        or len(row.get('measured_native_generation_prefix',[]))>1 or row.get('actual_prefix_callback_calls',0)<=0):
                        raise ValueError('Original native prompt/token/visual/prefix evidence differs')
            else:
                if any(r.get('raw_answer')!=r['prediction'] or r.get('scores')!=saved[r['query_id']]['scores'] for r in rows):raise ValueError('Original legal text/similarity evidence differs')
            groups[dataset,73,method]=rows
    return groups


def legacy_groups(cfg,science,provenance):
    plan=cfg['plan'];policy=cfg['policy'];groups={}
    methods={'baselines':('rices','gpt_mm'),'decision':('decision_zero','decision_learn'),
        'proj':('projection_only',),'detriever':('detriever',)}
    import run_behavior_qwen_three_seed as frozen_commands
    for seed in policy.SEEDS:
        for dataset,data in plan['datasets'].items():
            source=science/'baselines'/dataset/'qwen3vl4b'/('seed_'+str(seed));n=len(data['manifest']['bank'])+len(data['manifest']['query'])
            source_SHA=p.digest(source/'anchor_states.npy');policy.validate_source_identity(p.read(source/'run_identity.json'),dataset,seed)
            for name,shape,progress in [('anchor_states.npy',(n,36,2560),'anchor_states_progress.json'),
                ('clip_features.npy',(n,512),'clip_features.progress.json'),('gpt_mm_embeddings.npy',(n,2560),'gpt_mm_progress.json')]:
                status=p.read(source/progress)
                if status.get('shape')!=list(shape) or status.get('completed')!=n:raise ValueError('Complete same-tuple feature commits required')
                provenance[str((source/name).relative_to(science))]=array_audit(source/name,shape,dtype='float32' if name=='clip_features.npy' else 'float16')
            outputs=lines(source/'gpt_mm_zero_shot_outputs.jsonl')
            if [r['sample_id'] for r in outputs]!=[r['sample_id'] for r in data['manifest']['bank']+data['manifest']['query']]:raise ValueError('Actual GPT-MM source outputs incomplete')
            for phase in policy.PHASES:
                folder=science/'jobs'/dataset/('seed_'+str(seed))/phase;job=p.read(folder/'job.json');done=p.read(folder/'complete.json')
                expected=next(j for j in plan['jobs'] if (j['dataset'],j['seed'],j['phase'])==(dataset,seed,phase))
                module,argv=frozen_commands.command(dataset,seed,phase);argv=[v.replace(str(ROOT),'/workspace/BehaviorICL').replace('\\','/') for v in argv]
                if (any(job.get(k)!=v for k,v in expected.items()) or job.get('module')!=module or job.get('argv')!=argv
                    or job.get('source_tuple')!=policy.tuple_identity(dataset,seed) or job.get('launch_id')!=cfg['audit_auth']['launch_id']
                    or done.get('job_hash')!=expected['job_hash'] or done.get('plan_hash')!=cfg['plan_hash'] or done.get('identity')!=policy.tuple_identity(dataset,seed)):
                    raise ValueError('Actual independent tuple job/source/module/arguments differ')
            for component,names in methods.items():
                folder=science/component/dataset/'qwen3vl4b'/('seed_'+str(seed));identity=p.read(folder/'run_identity.json');config=p.read(folder/'config.json')
                if any(identity.get(k)!=v for k,v in policy.tuple_identity(dataset,seed).items() if k!='model_slug') or config.get('seed')!=seed or config.get('dataset')!=dataset or config.get('shots')!=4:
                    raise ValueError('Cross-tuple component/checkpoint/prediction evidence')
                if component!='baselines' and config.get('source_root')!='/workspace/BehaviorICL/results/'+cfg['science']+'/baselines':raise ValueError('Only own-tuple source root may feed training')
                manifest=p.read(folder/'manifest.json')
                for split in ('bank','query'):
                    if [(r['sample_id'],r['label'],r['split']) for r in manifest[split]]!=[(r['sample_id'],r['label'],r['split']) for r in data['manifest'][split]]:raise ValueError('Actual tuple manifest/order differs')
                saved=p.read(folder/'selections.json');rows=lines(folder/'predictions.jsonl')
                if set(saved)!=set(names) or len(rows)!=len(names)*len(data['manifest']['query']):raise ValueError('Every complete component method required')
                for method in names:
                    chosen=[r for r in rows if r.get('method')==method];selected=selections(data,saved[method],method,legacy=True)
                    prediction_audit(data,chosen,selected)
                    if any(r.get('raw_output')!=r['prediction'] for r in chosen):raise ValueError('Original constrained raw output differs')
                    groups[dataset,seed,method]=chosen
                if component in ('decision','proj'):
                    name='projected_state' if component=='decision' else 'projection_only';metadata=p.read(folder/(name+'_metadata.json'));best=full_history(metadata,True)
                    if config.get('train_steps')!=1500:raise ValueError('Exact fixed1500 recipe required')
                    if component=='proj':
                        split=p.read(folder/'development_split.json');expected=data['splits'][str(seed)]
                        if split!={'train_ids':expected['train_ids'],'validation_ids':expected['development_ids']} or identity.get('source_cache_sha256')!=source_SHA:raise ValueError('Seed-specific bank-only dev/source cache differs')
                    provenance[str((folder/(name+'_best.pt')).relative_to(science))]=checkpoint(folder/(name+'_best.pt'),2560,36,best,component=='decision')
                if component=='detriever':
                    proxy=folder/'gold_input_answer_eos.npy';sha=numeric_array_SHA(proxy);metadata=p.read(folder/'detriever_metadata.json');status=p.read(folder/'gold_input_answer_eos_progress.json')
                    if status.get('completed')!=len(data['manifest']['bank']) or status.get('manifest_sha256')!=p.digest(source/'manifest.json'):
                        raise ValueError('Gold proxy must be complete same-tuple bank only')
                    provenance[str(proxy.relative_to(science))]={**array_audit(proxy,(len(data['manifest']['bank']),2560)),'numeric_values_sha256':sha}
                    if (metadata.get('steps')!=10000 or metadata.get('proxy')!='gold_input_answer_eos' or metadata.get('proxy_sha256')!=sha or metadata.get('layers_one_based')!=[1,6,11,16,21,26,31,36]):raise ValueError('Original task-adapted DeTriever recipe/proxy differs')
                    import torch
                    saved_model=torch.load(folder/'detriever_checkpoint.pt',map_location='cpu',weights_only=True)
                    if saved_model.get('step')!=10000 or saved_model.get('proxy_sha256')!=sha or saved_model.get('proxy_mode')!='gold_input_answer_eos' or saved_model.get('layers_zero_based')!=[0,5,10,15,20,25,30,35] or len(saved_model.get('losses',[]))!=10000:
                        raise ValueError('Fresh complete same-tuple DeTriever final checkpoint differs')
                    if any(t.device.type!='cpu' or not torch.isfinite(t).all() for t in saved_model['model'].values()):raise ValueError('Invalid CPU retriever checkpoint tensors')
    return groups


def csv_out(path,rows):
    with path.open('x',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def analyze(role,session,out):
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    binding=implementation_check();cfg=config(role);plan=cfg['plan'];session=Path(session).resolve();out=Path(out).resolve()
    if not session.is_relative_to(ROOT/'results/cloud_trial') or not out.is_relative_to(ROOT/'analysis') or out.exists():raise ValueError('Actual retrieved session and fresh analysis root required')
    row=next(r for r in p.read(ROOT/'.tools/paper_active_jobs_v1.json')['active_jobs']+p.read(ROOT/'.tools/paper_active_jobs_v1.json').get('ended_jobs',[]) if r['backup_dir']==session.relative_to(ROOT).as_posix() and r['role']==role)
    take,stop,account=(p.read(session/n) for n in ('full_takeback_receipt.json','stop_confirmation.json','runtime_accounting.json'))
    # Validate terminal evidence before parsing any scientific arrays.
    preliminary={'status':'complete_verified','complete_official_test':True,'session':row['backup_dir'],**slots.EXPECTED[role],
        'utc':p.now(),'independent_CPU_audit_after_stop':True,'closed_archive_member_extracted_SHA_verified':True}
    slots.terminal_guard(row,take,stop,account,preliminary)
    members=slots.archive_audit(session,take,role);science=session/'snapshot/results'/cfg['science'];operation=session/'snapshot/results'/cfg['operation']
    auth=p.read(operation/'execution_authorization.json');native=p.read(operation/'native_identity_receipt.json');env=p.read(operation/'environment_audit.json')
    if (auth.get('pod_id')!=row['pod_id'] or auth.get('launch_id')!=row['launch_id'] or auth.get('plan_hash')!=cfg['plan_hash'] or auth.get('implementation_hash')!=cfg['implementation_hash']
        or p.digest(operation/'native_identity_receipt.json')!=auth.get('native_identity_receipt_sha256')
        or native.get('actual_api_pod_id')!=row['pod_id'] or native.get('proc1_pod_id')!=row['pod_id'] or native.get('export_file_pod_id')!=row['pod_id'] or native.get('expected_Pod_field_injected') is not False
        or auth.get('code_sha256')!=cfg['code_sha256']
        or env.get('versions')!=cfg['environment'] or env.get('models')!=cfg['models']):raise ValueError('Original exact native/runtime/model/code identity differs')
    if p.read(science/'verification.json').get('status')!='all_complete_verified':raise ValueError('Real whole-scope scientific terminal absent')
    inputs=p.read(operation/'inputs_ready.json')
    if inputs.get('images')!=17428 or inputs.get('plan_hash')!=cfg['plan_hash'] or inputs.get('launch_id')!=row['launch_id']:raise ValueError('Complete original byte-audited images required')
    for dataset,data in plan['datasets'].items():
        for split in ('bank','query'):
            for sample in data['manifest'][split]:
                if p.digest(ROOT/sample['path'])!=sample['image_sha256']:raise ValueError('Original full official image SHA changed')
    provenance={};cfg={**cfg,'audit_auth':auth};groups=legacy_groups(cfg,science,provenance) if role=='qwen_three_seed' else modern_groups(cfg,science,provenance)
    expected=slots.EXPECTED[role]
    if len(groups)!=expected['groups'] or sum(len(r) for r in groups.values())!=expected['predictions']:raise ValueError('No partial official groups may count as complete')
    summaries=[];pairs=[]
    for (dataset,seed,method),rows in groups.items():
        summaries.append({'dataset':dataset,'seed':seed,'method':method,'correct':sum(r['correct'] for r in rows),'total':len(rows),'accuracy_percent':sum(r['correct'] for r in rows)*100/len(rows)})
    if role=='qwenvision':
        for dataset,reference in plan['references'].items():
            folder=ROOT/reference['source_folder']
            for name,sha in reference['files_SHA'].items():
                if p.digest(folder/name)!=sha:raise ValueError('Historical paired reference byte provenance changed')
            rows=[r for r in lines(folder/'predictions.jsonl') if r.get('method')=='projection_only']
            query=plan['datasets'][dataset]['manifest']['query']
            if len(rows)!=len(query) or [r['query_id'] for r in rows]!=[r['sample_id'] for r in query] or any(r['target']!=q['label'] or r['correct']!=(r['prediction']==q['label']) for r,q in zip(rows,query)):raise ValueError('Historical full official paired reference differs')
            pairs.append({'dataset':dataset,'seed':73,'first':'supcon_qwenvision','second':'historical_fixed256_Proj',**paired_bootstrap([r['correct'] for r in groups[dataset,73,'supcon_qwenvision']],[r['correct'] for r in rows])})
    else:
        comparisons=[('behavior_proj',other) for other in ('zero','rices','behavior_raw')] if role=='internvl_five' else [(m,b) for m in ('decision_zero','projection_only','decision_learn') for b in ('rices','gpt_mm','detriever')]+[('decision_learn','projection_only')]
        seeds=(73,) if role=='internvl_five' else (74,75)
        for seed in seeds:
            for dataset in plan['datasets']:
                for first,second in comparisons:pairs.append({'dataset':dataset,'seed':seed,'first':first,'second':second,**paired_bootstrap([r['correct'] for r in groups[dataset,seed,first]],[r['correct'] for r in groups[dataset,seed,second]])})
    out.mkdir(parents=True);csv_out(out/'full_official_accuracy.csv',summaries);csv_out(out/'paired_accuracy_CI.csv',pairs)
    if role=='qwen_three_seed':
        from prepare_behavior_qwen_three_seed import checked_history
        history=checked_history(p.read(ROOT/plan['seed73_reference']['audit_path']))
        names={'RICES':'rices','GPT-MM':'gpt_mm','DeTriever':'detriever','BehaviorICL-Raw':'decision_zero','BehaviorICL-Proj':'projection_only','BehaviorICL-Learn':'decision_learn'}
        combined=summaries+[{'dataset':r['dataset'],'seed':73,'method':names[r['method']],'correct':r['correct'],'total':r['total'],'accuracy_percent':r['accuracy_percent']} for r in history['full_groups']]
        stats=[]
        for dataset in plan['datasets']:
            for method in policy_methods(cfg):
                values=[r['accuracy_percent'] for r in combined if r['dataset']==dataset and r['method']==method]
                if len(values)!=3:raise ValueError('All three audited seeds required')
                stats.append({'dataset':dataset,'method':method,'seeds':'73/74/75','mean_accuracy_percent':statistics.mean(values),'sample_std_accuracy_percent':statistics.stdev(values),'history_environment_difference':True})
        csv_out(out/'three_seed_mean_std.csv',stats)
    audit={**preliminary,'utc':p.now(),'plan_hash':cfg['plan_hash'],'implementation_hash':cfg['implementation_hash'],
        'audit_implementation_hash':binding['audit_implementation_hash'],'archive_SHA':take['archive_sha256'],
        'archive_members_SHA':members,'source_and_checkpoint_audit':provenance,'source_SHA':{n:p.digest(ROOT/n) for n in FILES},
        'CPU_only_no_model_or_GPU_execution':True,'retrieval_numerical_kernel_replay':False,
        'CI':'10000 query-paired bootstrap seed73; descriptive, no multiplicity adjustment; separate dataset/seed comparisons',
        'limits':'Exact frozen source/bytes/arrays/checkpoints/tuple recipes/contexts/outputs checked; no model-generation or full retrieval-kernel replay. Historical Qwen comparisons retain runtime/source/batching differences. QV and legacy Qwen outputs lack saved actual token IDs and rendered prompt fingerprints; native InternVL does retain them.'}
    p.immutable(out/'audit.json',audit);return {'status':'complete_verified','role':role,**expected}


def policy_methods(cfg):return cfg['policy'].METHODS


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('freeze','preflight','analyze'),default='preflight')
    parser.add_argument('--role',choices=ROLES);parser.add_argument('--session',type=Path);parser.add_argument('--output-root',type=Path);args=parser.parse_args()
    if args.phase=='freeze':print({'CPU_audit_frozen':freeze()['audit_implementation_hash'],'GPU_requested':False})
    elif args.phase=='preflight':implementation_check();print({'CPU_audit_identity_ready':True,'GPU_requested':False})
    else:
        if not all((args.role,args.session,args.output_root)):raise ValueError('Exact stopped role/session and fresh output required')
        os.environ.update(CUDA_VISIBLE_DEVICES='-1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        print(analyze(args.role,args.session,args.output_root))
