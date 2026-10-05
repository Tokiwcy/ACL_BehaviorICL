"""Independent cloud-only TAM pilot; one native GPU process, no training."""
import hashlib,json,os
from pathlib import Path
import sys,time,traceback,subprocess
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'
def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()
def atomic(path,value):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
def stage(name,**extra):atomic(OUT/'progress.json',{'utc':time.time(),'pid':os.getpid(),'stage':name,**extra})
def guard(auth):
    if sys.platform!='linux' or Path.cwd()!=ROOT or not str(ROOT).startswith('/workspace/BehaviorICL-TAM/'):
        raise ValueError('Independent native cloud only; no local model')
    from query_tam_cloud_ops_v1 import NATIVE_REMOTE,assess_native
    # The standalone native probe reads real container/export identities.
    import subprocess
    value=json.loads(subprocess.check_output([sys.executable,'-c',NATIVE_REMOTE],text=True,timeout=20))
    assess_native(value,auth['pod_id'],empty_gpu=False)
    if any(pid!=os.getpid() for pid in value['GPU_compute_pids']):
        raise ValueError('Only this exact TAM worker may use the single GPU')
    heartbeat=json.loads((OUT/'backup_heartbeat.json').read_text())
    if (heartbeat['pod_id'],heartbeat['launch_id'])!=(auth['pod_id'],auth['launch_id']) or not 0<=time.time()-heartbeat['utc']<=300:
        raise ValueError('Exact successful backup heartbeat stale')
    if (OUT/'stop_request.json').exists():raise ValueError('TAM controller requested stop')
def bootstrap():
    auth=json.loads((ROOT/'auth.json').read_text());guard(auth)
    import importlib.metadata as meta
    plan=json.loads((ROOT/'protocol.json').read_text())
    if sys.prefix!=str(ROOT/'.venv'):
        stage('CPU_environment_not_science')
        subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(ROOT/'.venv')],check=True)
        os.execv(str(ROOT/'.venv/bin/python'),[str(ROOT/'.venv/bin/python'),'-u',str(Path(__file__)), '--isolated'])
    versions={}
    for name,version in plan['environment'].items():
        try:actual=meta.version(name)
        except meta.PackageNotFoundError:actual=None
        if actual!=version:
            if name in ('torch','torchvision'):raise ValueError('Native CUDA template differs; do not install an unknown CUDA runtime')
            subprocess.run([sys.executable,'-m','pip','install',name+'=='+version],check=True)
        versions[name]=meta.version(name)
    if versions!=plan['environment']:raise ValueError('Exact original prompt environment required')
    stage('CPU_model_download_and_SHA_not_science',versions=versions)
    from huggingface_hub import snapshot_download
    inventory=plan['model_inventory']
    location=Path(snapshot_download(plan['model'],revision=plan['revision'],cache_dir=ROOT/'.hf_cache',
        allow_patterns=list(inventory['files_SHA256']),token=False))
    for name,expected in inventory['files_SHA256'].items():
        guard(auth)
        if sha(location/name)!=expected:raise ValueError('Original complete model bytes differ: '+name)
    atomic(OUT/'actual_model_environment_SHA.json',{'versions':versions,'model':plan['model'],'revision':plan['revision'],'files_SHA':inventory['files_SHA256'],'model_generation_compatible':False})
    for name,expected in auth['sources'].items():
        if sha(ROOT/name)!=expected:raise ValueError('TAM source bytes differ')
    run(plan,auth,location)


def run(plan,auth,location):
    import torch,numpy as np
    from PIL import Image
    from scipy.optimize import minimize_scalar
    from transformers import AutoProcessor,AutoModelForImageTextToText
    import query_tam_adapter_v1 as adapter
    torch.manual_seed(73)
    processor=AutoProcessor.from_pretrained(location,local_files_only=True)
    model=AutoModelForImageTextToText.from_pretrained(location,dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True).eval().to('cuda')
    config=model.config.text_config
    if (config.num_hidden_layers,config.hidden_size)!=(36,2560):raise ValueError('Original Qwen decoder dimensions differ')
    tam,receipt=adapter.load(ROOT/'tam/tam.py',torch,np,minimize_scalar)
    atomic(OUT/'adapter_receipt.json',receipt)
    img_id=int(model.config.image_token_id)
    vision_end=processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
    footer=processor.tokenizer.encode('<|im_end|>\n<|im_start|>assistant',add_special_tokens=False)
    answer_start=processor.tokenizer.encode('<|im_end|>\n<|im_start|>assistant\n',add_special_tokens=False)
    count=0
    with torch.inference_mode():
        for example in plan['examples']:
            path=ROOT/example['bundle_path']
            if sha(path)!=example['image_SHA']:raise ValueError('Exact fixed pilot image differs')
            image=Image.open(path).convert('RGB')
            previous_pixels=previous_grid=None
            for arm,query in example['queries'].items():
                guard(auth)
                messages=[{'role':'user','content':[{'type':'image','image':str(path),'resized_height':224,'resized_width':224},{'type':'text','text':query}]}]
                original_padding=processor.tokenizer.padding_side
                try:
                    processor.tokenizer.padding_side='left'
                    inputs=processor.apply_chat_template([messages],tokenize=True,add_generation_prompt=True,
                        return_dict=True,return_tensors='pt',processor_kwargs={'padding':True})
                finally:processor.tokenizer.padding_side=original_padding
                grid=inputs['image_grid_thw'][0].tolist()
                if grid[0]!=1:raise ValueError('Single still-image grid required')
                merge=int(model.config.vision_config.spatial_merge_size)
                shape=(grid[1]//merge,grid[2]//merge)
                if grid[1]%merge or grid[2]%merge or int((inputs['input_ids']==img_id).sum())!=shape[0]*shape[1]:
                    raise ValueError('Actual image-token/grid alignment differs')
                pixels_SHA=hashlib.sha256(inputs['pixel_values'].contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
                if previous_pixels is not None and (pixels_SHA!=previous_pixels or grid!=previous_grid):raise ValueError('Changing query changed the processed image')
                previous_pixels,previous_grid=pixels_SHA,grid
                inputs={k:v.to('cuda') for k,v in inputs.items()}
                L=int(inputs['input_ids'].shape[1]);base_ids=inputs['input_ids'][0].tolist()
                if base_ids[-len(answer_start):]!=answer_start:raise ValueError('Actual Qwen assistant prefix differs')
                special={'img_id':[img_id],'prompt_id':[vision_end,footer],'answer_id':[answer_start,-1]}
                for mode in ('natural','fixed_target'):
                    guard(auth);start=time.time()
                    stage('TAM_forward',example=example['sample_id'],arm=arm,mode=mode,committed_runs=count)
                    kwargs={}
                    if mode=='fixed_target':
                        forced=processor.tokenizer.encode(example['target'],add_special_tokens=False)
                        def force(_batch,ids):return [forced[len(ids)-L]] if len(ids)-L<len(forced) else [processor.tokenizer.eos_token_id]
                        kwargs={'prefix_allowed_tokens_fn':force,'max_new_tokens':len(forced)+1}
                    else:kwargs={'max_new_tokens':plan['natural_generation']['max_new_tokens']}
                    output=model.generate(**inputs,do_sample=False,use_cache=True,output_hidden_states=True,return_dict_in_generate=True,**kwargs)
                    tokens=output.sequences[0].tolist();tail=tokens[L:]
                    if mode=='fixed_target' and tail[:len(forced)]!=forced:raise ValueError('Fixed target probe token sequence differs')
                    rounds=min(len(output.hidden_states),len(tail),6 if mode=='natural' else len(forced))
                    # Remove EOS/special token explanation while preserving positions.
                    while rounds and tail[rounds-1] in processor.tokenizer.all_special_ids:rounds-=1
                    if rounds<1:raise ValueError('No explainable answer token')
                    states=output.hidden_states[0]
                    if len(states)!=37 or states[-1].shape[1]!=L:raise ValueError('Actual full prefill hidden-state contract differs')
                    anchor=torch.stack([x[0,-1].float().cpu() for x in states[1:]]).numpy()
                    logits=[model.lm_head(step[-1]).detach().cpu() for step in output.hidden_states[:rounds]]
                    # Hidden states/vision tokens are exported at actual native grid.
                    del output
                    rows=adapter.maps(tam,tokens,shape,logits,special,image,processor,rounds)
                    key=example['dataset']+'_'+example['sample_id']+'_'+arm+'_'+mode
                    folder=OUT/'runs'/key;folder.mkdir(parents=True,exist_ok=False)
                    np.savez_compressed(folder/'maps.npz',maps=np.stack([r['map'] for r in rows]),
                        raw_CAM=np.stack([r['raw_CAM'] for r in rows]),post_ECI=np.stack([r['post_ECI_before_filter'] for r in rows]),anchor_states=anchor)
                    record={'key':key,'sample_id':example['sample_id'],'dataset':example['dataset'],'historical_group':example['historical_group'],
                        'arm':arm,'mode':mode,'query':query,'image_SHA':example['image_SHA'],'processed_pixels_SHA':pixels_SHA,
                        'image_grid_thw':grid,'map_shape':list(shape),'input_ids':base_ids,'output_ids':tail,
                        'target_ids':[r['token_id'] for r in rows],'target_tokens':processor.tokenizer.convert_ids_to_tokens([r['token_id'] for r in rows]),
                        'output_text':processor.decode(tail,skip_special_tokens=True),'fixed_target_is_not_prediction':mode=='fixed_target',
                        'possibly_truncated':mode=='natural' and len(tail)>=24 and tail[-1] not in processor.tokenizer.all_special_ids,
                        'constant_activation':[r['constant_multimodal_activation'] for r in rows],
                        'maps_SHA':sha(folder/'maps.npz'),'seconds':time.time()-start,'source_purpose':'source-query mechanistic probe; no ICL demos'}
                    atomic(folder/'record.json',record);count+=1
                    stage('TAM_committed',committed_runs=count,last_key=key)
                    del logits,states,anchor,rows
                    torch.cuda.empty_cache()
                del inputs
    stage('complete',committed_runs=count,expected_runs=len(plan['examples'])*8)
    atomic(OUT/'complete.json',{'protocol_hash':plan['protocol_hash'],'runs':count,'images':len(plan['examples']),
        'claim':'exploratory mechanism pilot, not full-test accuracy or attention/causal proof'})


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    try:bootstrap()
    except BaseException as error:
        stage('failed',error_class=type(error).__name__)
        atomic(OUT/'failure.json',{'error_class':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        raise
