"""CPU-only fixed-image/query pilot tied to already audited prompt results."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/query_tam_pilot_v1'
SAMPLE=ROOT/'results/cloud_trial/czscrdd6bcut9k-paper-task-prompt-autonomous/session_1/snapshot/results/behavior_task_conditioning_raw_v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def prepare():
    if (OUT/'protocol.json').exists():raise ValueError('Preserve fixed TAM pilot identity')
    plan=read(ROOT/'analysis/task_conditioning_prompt_plan_v1/protocol_lock.json')
    audit=read(ROOT/'analysis/paper_task_prompt_complete_results_v1/audit.json')
    source=read(ROOT/'.tools/tam_official_v1/source_receipt.json')
    # Execute only pure original prompt formatting, without importing models.
    tree=ast.parse((ROOT/'scripts/multimodal_model_adapter.py').read_text())
    namespace={'TASK_NOUNS':{'dtd':'texture category','cub':'bird species'}}
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='task_instruction')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'original_pure_task_prompt','exec'),namespace)
    examples=[];prediction_sha={}
    for d in ('cub','dtd'):
        predictions={}
        for arm in ('task','generic'):
            path=SAMPLE/d/'test'/arm/'predictions.jsonl'
            relative=path.relative_to(SAMPLE.parent.parent).as_posix()
            if audit['source_archive_members_SHA'][relative]!=sha(path):raise ValueError('Audited full prediction bytes differ')
            prediction_sha[d+'/'+arm]=sha(path)
            predictions[arm]={row['query_id']:row for row in map(json.loads,path.read_text().splitlines())}
        rows={row['sample_id']:row for row in plan['datasets'][d]['manifest']['query']}
        groups={key:[] for key in ('task_only','generic_only','both_correct','both_wrong')}
        for sid,row in rows.items():
            a,b=predictions['task'][sid],predictions['generic'][sid]
            if a['target']!=row['label'] or b['target']!=row['label']:raise ValueError('Exact query identity differs')
            key=('both_correct' if a['correct'] and b['correct'] else 'task_only' if a['correct'] else 'generic_only' if b['correct'] else 'both_wrong')
            groups[key].append(sid)
        for group,ids in groups.items():
            sid=min(ids,key=lambda x:hashlib.sha256(('query-tam-pilot-seed73/'+d+'/'+group+'/'+x).encode()).hexdigest())
            row=rows[sid];image=ROOT/row['path']
            if sha(image)!=row['image_sha256']:raise ValueError('Original image bytes differ')
            queries={
                'generic':plan['protocol']['generic_prompt'],
                'task':namespace['task_instruction'](d,plan['datasets'][d]['class_labels']),
                'focus_a':("Describe the bird's head and beak." if d=='cub' else 'Describe the local surface pattern and texture.'),
                'focus_b':("Describe the bird's wings and tail." if d=='cub' else 'Describe the overall shape and background.')}
            examples.append({'dataset':d,'sample_id':sid,'image_path':row['path'],'image_SHA':row['image_sha256'],
                'bundle_path':'inputs/'+d+'-'+sid+image.suffix,'historical_group':group,'target':row['label'],
                'queries':queries,'historical_ICL':{arm:{k:predictions[arm][sid][k] for k in ('prediction','correct','demo_ids','demo_labels','scores')} for arm in ('task','generic')}})
    OUT.mkdir(parents=True,exist_ok=False)
    value={'version':1,'purpose':'exploratory fixed-image changed-query TAM; not new full-test accuracy or causal proof',
        'user_request':'改变query，复用Token Activation Map代码，比较query+image空间响应；优先于InternVL',
        'model':plan['protocol']['model'],'revision':plan['protocol']['revision'],'model_inventory':plan['model_inventory'],
        'environment':plan['expected_environment'],'TAM':source,'examples':examples,
        'selection':'one SHA-deterministic query from each of four historical outcome strata per dataset; no visual/heatmap selection',
        'input_processing':'same original apply_chat_template and 224 message hints; no new pixel budget; assert same pixels/grid across queries',
        'natural_generation':{'do_sample':False,'max_new_tokens':24,'maps_first_non_special_tokens':6},
        'fixed_target_probe':'same full gold-label token sequence teacher-forced for each query; explicitly not a model prediction',
        'measures':['TAM post-ECI maps','raw class-to-vision CAM','per-layer last-input decoder state','same-target map distances'],
        'hypothesis_limits':['Original image precedes query in causal decoder; vision-token raw maps for a fixed target may stay unchanged.',
            'Post-ECI differences can reflect changed textual context subtraction, not attention weights.',
            'Natural answer changes confound target token changes; fixed-target control is required.',
            'Existing improvement changes retrieval representations/demo selection; source-query TAM is a mechanistic probe, not a replay of downstream four-demo ICL.'],
        'original_protocol_SHA':sha(ROOT/'analysis/task_conditioning_prompt_plan_v1/protocol_lock.json'),
        'original_complete_audit_SHA':sha(ROOT/'analysis/paper_task_prompt_complete_results_v1/audit.json'),
        'historical_predictions_SHA':prediction_sha,'local_CUDA':False,'pretrained_weights_loaded_locally':False}
    value['protocol_hash']=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (OUT/'protocol.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'images':len(examples),'query_arms':4,'natural_and_fixed_target_runs':len(examples)*8,'protocol_hash':value['protocol_hash']}))
if __name__=='__main__':prepare()
