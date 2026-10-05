"""Pinned tokenizer-only bytes for independent CPU audits; no model loading."""
import argparse
from pathlib import Path
import urllib.request
from behavior_task_prompt_policy import ROOT,read,digest,immutable,now

BASE=ROOT/'.tools/paper_queue_CPU_tokenizers_v1'
FILES=('tokenizer.model','tokenizer_config.json','added_tokens.json','special_tokens_map.json','tokenization_internlm2.py')


def prepare():
    plan=read(ROOT/'analysis/second_vlm_five_dataset_v2/plan_lock.json')
    model='OpenGVLab/InternVL2_5-8B';inventory=plan['models'][model];folder=BASE/'internvl'/inventory['revision']
    folder.mkdir(parents=True,exist_ok=True);members={}
    for name in FILES:
        item=inventory['files'][name];path=folder/name
        if not path.exists():
            local=ROOT/'analysis/second_vlm_source_inventory_v1/source'/name
            if local.exists():blob=local.read_bytes()
            else:
                with urllib.request.urlopen('https://huggingface.co/'+model+'/resolve/'+inventory['revision']+'/'+name,timeout=60) as response:
                    blob=response.read(12*1024*1024+1)
            import hashlib
            if len(blob)>12*1024*1024 or hashlib.sha256(blob).hexdigest()!=item['sha256']:
                raise ValueError('Exact fixed tokenizer public bytes differ')
            with path.open('xb') as handle:handle.write(blob)
        if path.is_symlink() or digest(path)!=item['sha256']:raise ValueError('Preserved tokenizer identity differs')
        members[name]=item['sha256']
    value={'utc':now(),'model':model,'revision':inventory['revision'],'files_SHA':members,
        'tokenizer_only':True,'model_weights_downloaded_or_loaded':False,'CUDA_execution':False}
    receipt=folder/'CPU_tokenizer_receipt.json'
    if receipt.exists():
        if read(receipt)['files_SHA']!=members:raise ValueError('Original tokenizer receipt differs')
    else:immutable(receipt,value)
    print({'fixed_CPU_tokenizer_bytes_ready':True,'files':len(members),'GPU_requested':False},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    else:print({'CPU_tokenizer_preparation_only':True,'GPU_requested':False})
