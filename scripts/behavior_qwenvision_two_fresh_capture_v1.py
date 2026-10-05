"""Fresh native capture: same formula, every image newly computed.
Original cache acceptance code remains preserved and is never invoked.
"""
import gc,json,shutil,time
import behavior_qwenvision_two_host_policy_v1 as p
import run_behavior_qwenvision_two_control_v1 as original
shared=original.shared;OUT=original.OUT;ROOT=p.ROOT

def capture(plan,dataset,smoke=False):
    import numpy as np
    import torch
    from multimodal_model_adapter import load_model, MODEL_SPECS, zero_shot_messages, prepare_batch
    data=shared.preflight(plan,dataset); rows=data['manifest']['bank']+data['manifest']['query']
    folder=OUT/dataset/'sources'; folder.mkdir(parents=True,exist_ok=True)
    p.immutable(folder/'identity.json',{'plan_hash':plan['plan_hash'],'dataset':dataset,'manifest_hash':p.canonical(data['manifest'])})
    path=folder/'qwenvision.npy'; pp=folder/'qwenvision_progress.json'; shape=[len(rows),1,1024]
    reuse=None
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
