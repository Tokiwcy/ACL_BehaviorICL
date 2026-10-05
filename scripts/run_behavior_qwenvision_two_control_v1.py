"""One QwenVision+SupCon fixed256 control. No old scheduler/width grid is run."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import time
import behavior_qwenvision_two_policy_v1 as p
import run_behavior_p2_matched as shared
from collect_behavior_priority_zero import lock

ROOT=p.ROOT; OUT=ROOT/'results'/p.SCIENCE
# Reuse frozen implementation helpers in this independent process, never mutate
# the historical source files or call its main/scheduler.
shared.p=p; shared.ROOT=ROOT; shared.OUT=OUT


def source_arrays(plan,dataset):
    import numpy as np
    folder=OUT/dataset/'sources'; record=p.read(folder/'verification.json')
    item=record['sources']['qwenvision']; path=folder/'qwenvision.npy'
    if record['status']!='all_complete_verified' or record['plan_hash']!=plan['plan_hash'] or p.digest(path)!=item['sha256']:
        raise ValueError('Complete new/cache source verification changed')
    array=np.load(path,mmap_mode='r')
    if list(array.shape)!=item['shape'] or str(array.dtype)!=item['dtype'] or not np.isfinite(array).all():
        raise ValueError('Nonfinite or wrong complete QwenVision source')
    return {'qwenvision':array}


shared.sources=source_arrays


def capture(plan,dataset,smoke=False):
    import numpy as np
    import torch
    from multimodal_model_adapter import load_model, MODEL_SPECS, zero_shot_messages, prepare_batch
    data=shared.preflight(plan,dataset); rows=data['manifest']['bank']+data['manifest']['query']
    folder=OUT/dataset/'sources'; folder.mkdir(parents=True,exist_ok=True)
    p.immutable(folder/'identity.json',{'plan_hash':plan['plan_hash'],'dataset':dataset,'manifest_hash':p.canonical(data['manifest'])})
    path=folder/'qwenvision.npy'; pp=folder/'qwenvision_progress.json'; shape=[len(rows),1,1024]
    reuse=plan['complete_cache_reuse'].get(dataset)
    if reuse and not path.exists():
        origin=ROOT/reuse['cloud_folder']
        if (p.digest(origin/'qwenvision.npy')!=reuse['sha256'] or p.digest(origin/'verification.json')!=reuse['verification_SHA'] or
                p.digest(origin/'qwen_capture.jsonl')!=reuse['log_SHA']):
            raise ValueError('Complete reused QwenVision cache provenance changed')
        shutil.copyfile(origin/'qwenvision.npy',path)
        p.atomic(pp,{'plan_hash':plan['plan_hash'],'completed':len(rows),'shape':shape,'dtype':'float16','origin':reuse})
    previous=p.read(pp) if pp.exists() else None
    if path.exists()!=bool(previous): raise ValueError('Unpaired source/progress; no silent repair')
    if previous and (previous['plan_hash']!=plan['plan_hash'] or previous['shape']!=shape or previous['dtype']!='float16'):
        raise ValueError('Source prefix identity changed')
    array=np.lib.format.open_memmap(path,mode='r+' if previous else 'w+',dtype='float16',shape=tuple(shape))
    count=previous['completed'] if previous else 0
    if not 0<=count<=len(rows): raise ValueError('Invalid committed source prefix')
    if not previous: p.atomic(pp,{'plan_hash':plan['plan_hash'],'completed':0,'shape':shape,'dtype':'float16'})
    logs=[]; logpath=folder/'vision_capture.jsonl'
    if logpath.exists():
        with logpath.open(encoding='utf-8') as handle:
            for line in handle:
                if not line.endswith('\n'): raise ValueError('Truncated source prefix; preserve evidence')
                logs.append(json.loads(line))
    if not reuse and (len(logs)!=count or any((r['sample_id'],r['image_sha256'],r['plan_hash'])!=
            (rows[i]['sample_id'],rows[i]['image_sha256'],plan['plan_hash']) for i,r in enumerate(logs))):
        raise ValueError('Source prefix log/ID/image/count disagreement')
    need_smoke=not (folder/'smoke.json').exists()
    indices=([0] if need_smoke and reuse else list(range(count,1 if smoke else len(rows))))
    if indices:
        model,processor=load_model(MODEL_SPECS['qwen3vl4b'],ROOT/'.hf_cache',None); pooled=[]
        def hook(module,args,result):
            value=result[0] if isinstance(result,tuple) else result
            if value.ndim!=2 or value.shape[-1]!=1024: raise ValueError('Wrong native final vision token dimensions')
            pooled.append((value.detach().float().mean(0),int(value.shape[0])))
        handle=model.model.visual.blocks[-1].register_forward_hook(hook)
        with logpath.open('a',encoding='utf-8') as log,torch.inference_mode():
            for i in indices:
                before=time.perf_counter(); pooled.clear(); row=rows[i]
                inputs=prepare_batch(processor,[zero_shot_messages(shared.sample(row),dataset,data['class_labels'],224)])
                output=model(**inputs,output_hidden_states=False,use_cache=False,return_dict=True,logits_to_keep=1)
                if len(pooled)!=1: raise ValueError('Vision hook did not capture exactly one image')
                value,n=pooled[0]
                if n!=int(inputs['image_grid_thw'].prod(-1).sum()) or not torch.isfinite(value).all():
                    raise ValueError('Native token grid/nonfinite feature smoke rejected')
                encoded=value.cpu().numpy().astype(np.float16)
                if reuse:
                    if not np.array_equal(encoded,array[0,0]): raise ValueError('Cached vision first-row exact FP16 reproduction failed')
                else:
                    array[i,0]=encoded; array.flush()
                    torch.cuda.synchronize()
                    shared.write_row(log,{'sample_id':row['sample_id'],'image_sha256':row['image_sha256'],
                        'plan_hash':plan['plan_hash'],'native_vision_token_count':n,'input_tokens':int(inputs['input_ids'].shape[1]),
                        'seconds':time.perf_counter()-before})
                    p.atomic(pp,{'plan_hash':plan['plan_hash'],'completed':i+1,'shape':shape,'dtype':'float16'})
                if i==0:
                    p.atomic(folder/'smoke.json',{'status':'actual_native_QwenVision_passed','utc':shared.now(),'plan_hash':plan['plan_hash'],
                        'sample_id':row['sample_id'],'shape':[1,1024],'native_token_count':n,'cache_exact_FP16_reproduction':bool(reuse)})
                if (i+1)%100==0: print(f'{dataset} QwenVision {i+1}/{len(rows)}',flush=True)
                del inputs,output,value; pooled.clear()
        handle.remove(); del model,processor; gc.collect(); torch.cuda.empty_cache()
    if smoke: return
    if p.read(pp)['completed']!=len(rows) or not np.isfinite(array).all(): raise ValueError('Source incomplete or nonfinite')
    p.atomic(folder/'verification.json',{'status':'all_complete_verified','plan_hash':plan['plan_hash'],
        'sources':{'qwenvision':{'sha256':p.digest(path),'shape':shape,'dtype':'float16','rows':len(rows)}},'reuse':reuse})


def retrieve(plan,dataset):
    import numpy as np
    import torch
    import run_dtd_learned_velocity as learned
    data=plan['datasets'][dataset]; bank=data['manifest']['bank']; queries=data['manifest']['query']
    arrays=source_arrays(plan,dataset); ck=OUT/dataset/'trained/qwenvision/d256/best.pt'
    folder=OUT/dataset/'test/supcon_qwenvision/d256'; folder.mkdir(parents=True,exist_ok=True)
    context={'plan_hash':plan['plan_hash'],'dataset':dataset,'phase':'test','method':'supcon_qwenvision','width':256,
        'source_hash':p.read(OUT/dataset/'sources/verification.json')['sources']['qwenvision']['sha256'],
        'checkpoint_SHA':p.digest(ck),'candidate_ids':[r['sample_id'] for r in bank],'query_ids':[r['sample_id'] for r in queries]}
    context['context_hash']=p.canonical(context); p.immutable(folder/'context.json',context)
    path=folder/'selections.json'
    if path.exists():
        selections=p.read(path)
        if list(selections)!=context['query_ids']: raise ValueError('Selection full query order changed')
        for v in selections.values(): p.validate_top4(v['demo_ids'],v['scores'],context['candidate_ids'])
        return folder,queries,selections,context
    before=time.perf_counter(); model=shared.retriever('qwenvision',256,arrays)
    saved=torch.load(ck,map_location='cuda',weights_only=True); model.load_state_dict(saved['model']); model.eval()
    left=learned.encode_indices(model,arrays['qwenvision'],np.arange(len(bank),len(bank)+len(queries)))
    right=learned.encode_indices(model,arrays['qwenvision'],np.arange(len(bank))); selections={}
    with torch.inference_mode():
        for start in range(0,len(queries),64):
            scores=model.score(left[start:start+64],right).float()
            top=scores.topk(4,dim=1).indices.flip(1); actual=torch.gather(scores,1,top).cpu().tolist()
            for j,(indices,values) in enumerate(zip(top.cpu().tolist(),actual)):
                demos=[bank[i] for i in indices]
                selections[queries[start+j]['sample_id']]={'demo_ids':[r['sample_id'] for r in demos],
                    'demo_labels':[r['label'] for r in demos],'scores':values}
                p.validate_top4([r['sample_id'] for r in demos],values,context['candidate_ids'])
    torch.cuda.synchronize(); p.immutable(path,selections)
    p.atomic(folder/'retrieval_time.json',{'seconds':time.perf_counter()-before,'query_count':len(queries),
        'scope':'cached source encoding/scoring, excludes frozen source extraction; not end-to-end online latency'})
    del model,left,right; gc.collect(); torch.cuda.empty_cache()
    return folder,queries,selections,context


def test(plan,dataset,smoke=False):
    folder,queries,selections,context=retrieve(plan,dataset)
    if not (folder/'smoke.json').exists():
        rows=shared.generate(plan,dataset,folder,queries,selections,context,1)
        p.atomic(folder/'smoke.json',{'status':'actual_four_demo_legal_generation_passed','utc':shared.now(),
            'context_hash':context['context_hash'],'input_tokens':rows[0]['input_tokens'],'output_tokens':rows[0]['output_tokens'],
            'first_official_query_counted_not_duplicate':True})
    if not smoke: shared.generate(plan,dataset,folder,queries,selections,context)


def verify(plan):
    groups=[]
    for dataset in p.DATASETS:
        source_arrays(plan,dataset)
        folder=OUT/dataset/'test/supcon_qwenvision/d256'; context=p.read(folder/'context.json')
        if context['plan_hash']!=plan['plan_hash']: raise ValueError('Wrong test plan identity')
        selection=p.read(folder/'selections.json'); data=plan['datasets'][dataset]
        rows=p.prediction_prefix(folder/'predictions.jsonl',data['manifest']['query'],selection,data['class_labels'],context['context_hash'],complete=True)
        v=p.read(folder/'verification.json'); train=OUT/dataset/'trained/qwenvision/d256'
        tv=p.read(train/'training_verified.json')
        if (v['predictions_sha256']!=p.digest(folder/'predictions.jsonl') or tv['best_sha256']!=p.digest(train/'best.pt') or
                [x['step'] for x in tv['history']]!=list(range(0,1501,100)) or
                p.read(folder/'smoke.json')['status']!='actual_four_demo_legal_generation_passed' or
                p.read(OUT/dataset/'sources/smoke.json')['status']!='actual_native_QwenVision_passed'):
            raise ValueError('Full prediction/checkpoint/history/smoke gate rejected')
        groups.append({'dataset':dataset,'rows':len(rows),'predictions_SHA':v['predictions_sha256']})
    p.atomic(OUT/'verification.json',{'status':'all_complete_verified','plan_hash':plan['plan_hash'],'full_groups':groups})


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--phase',choices=('preflight','capture','train','test','verify'),default='preflight')
    parser.add_argument('--dataset',choices=p.DATASETS); parser.add_argument('--smoke',action='store_true'); parser.add_argument('--allow-gpu',action='store_true')
    a=parser.parse_args(); plan=p.validate_plan(p.read(p.PLAN_PATH))
    if a.phase=='preflight':
        for dataset in ([a.dataset] if a.dataset else p.DATASETS): shared.preflight(plan,dataset)
        print('CPU image/code/split preflight passed; no model/CUDA'); return
    if a.phase!='verify':
        if not a.allow_gpu or not a.dataset: raise ValueError('Explicit independent cloud GPU flag/dataset required')
        p.gpu_guard(plan)
    with lock(OUT/'scientific_single_process.lock'):
        if a.phase=='capture': capture(plan,a.dataset,a.smoke)
        elif a.phase=='train': shared.train(plan,a.dataset,'qwenvision',256)
        elif a.phase=='test': test(plan,a.dataset,a.smoke)
        elif a.phase=='verify': verify(plan)


if __name__=='__main__': main()
