"""Independent, post-stop complete DTD+CUB prompt audit and paired analysis.

No model execution, CUDA, remote mutation, source substitution or partial
accuracy. Full archive, source/rows, actual prompt fingerprints, legal token
codec and original identities must all pass before a fresh analysis is saved.
"""
import argparse
import ast
from datetime import datetime
import hashlib
import math
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
import tarfile

import behavior_task_prompt_policy as p
import behavior_task_prompt_cloud as cloud
import run_behavior_task_prompt as runner
from analyze_behavior_repair_all_queries import paired_bootstrap
from run_behavior_priority_analysis import csv_out, rate


def terminal_guard(takeback,stop,account,auth):
    if (takeback.get('stage')!='all_complete_verified' or stop.get('reason')!='all_complete_verified'
        or stop.get('pod_id')!=auth['pod_id'] or account.get('pod_id')!=auth['pod_id']
        or account.get('phase')!='stopped' or account.get('confirmed_utc')!=stop.get('confirmed_utc')
        or not stop.get('disks_preserved') or not takeback.get('all_new_scientific_arrays_retrieved')
        or datetime.fromisoformat(takeback['utc'])>datetime.fromisoformat(stop['confirmed_utc'])):
        raise ValueError('Complete successful full takeback and earliest exact stop required; no partial analysis')


def archive_audit(session,takeback):
    path=session/'full_takeback.tar.gz'
    if p.digest(path)!=takeback['archive_sha256']:raise ValueError('Full scientific takeback archive changed')
    records=takeback['members'];manifest={r['name']:r for r in records}
    if len(manifest)!=len(records) or takeback['files']!=len(records):raise ValueError('Duplicate/missing full archive member')
    result={}
    with tarfile.open(path) as tar:
        members=cloud.safe_members(tar,True)
        if {r.name for r in members}!=set(manifest):raise ValueError('Closed full archive members differ')
        for member in members:
            item=manifest[member.name];sha=hashlib.sha256()
            if member.size!=item['bytes']:raise ValueError('Full member size differs')
            with tar.extractfile(member) as handle:
                while chunk:=handle.read(8*1024*1024):sha.update(chunk)
            dest=session/'snapshot'/member.name
            if dest.is_symlink() or sha.hexdigest()!=item['sha256'] or p.digest(dest)!=item['sha256']:
                raise ValueError('Full archive/member/extracted scientific bytes differ')
            result[member.name]=item['sha256']
    return result


def frozen_message_builders(plan):
    """Use only exact original pure message AST, without importing torch/model."""
    path=p.ROOT/'scripts/multimodal_model_adapter.py'
    if p.digest(path)!=plan['code_sha256']['scripts/multimodal_model_adapter.py']:raise ValueError('Original adapter changed')
    tree=ast.parse(path.read_text(encoding='utf-8'));names={'image_part','task_instruction','zero_shot_messages','icl_messages'}
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    nouns=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TASK_NOUNS' for t in n.targets)]
    if len(selected)!=4 or len(nouns)!=1:raise ValueError('Exact pure message definitions absent')
    namespace={'TASK_NOUNS':ast.literal_eval(nouns[0].value),'Sample':SimpleNamespace}
    body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*selected]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(path),'exec'),namespace)
    return namespace


def at_cloud(row,label=''):
    relative=PurePosixPath(row['path'])
    if relative.is_absolute() or '..' in relative.parts or relative.parts[0]!='datasets':raise ValueError('Unsafe original fixed image path')
    return SimpleNamespace(path=str(PurePosixPath('/workspace/BehaviorICL')/relative),label=label)


def prompt_audit(plan,binding,science,builders):
    rows_by_group={};checks=0;old=runner.OUT;runner.OUT=science
    try:
        for dataset in p.DATASETS:
            runner.require_compatibility(binding,dataset);data=plan['datasets'][dataset]
            manifest=data['manifest'];bank={r['sample_id']:r for r in manifest['bank']}
            runner.preflight(plan,dataset)
            for arm in p.ARMS:
                _,source=runner.source(plan,binding,dataset,arm)
                folder=science/dataset/'sources'/arm
                for row,logged in zip(manifest['bank']+manifest['query'],p.strict_jsonl(folder/'capture.jsonl')):
                    sample=at_cloud(row)
                    message=(builders['zero_shot_messages'](sample,dataset,data['class_labels'],224) if arm=='task' else
                             [{'role':'user','content':[builders['image_part'](sample.path,224),{'type':'text','text':plan['protocol']['generic_prompt']}]}])
                    if logged['prompt_fingerprint']!=p.canonical(message):raise ValueError('Actual pre-answer prompt fingerprint changed')
                    checks+=1
                folder=science/dataset/'test'/arm
                context,selections,codec=[p.read(folder/name) for name in ('context.json','selections.json','codec.json')]
                if (context['source_array_sha256']!=source['array_sha256'] or context['implementation_hash']!=binding['implementation_hash']
                    or context['job_hash']!=binding['job_hashes'][dataset+'/'+arm+'/generate']
                    or context['context_hash']!=p.canonical({k:v for k,v in context.items() if k!='context_hash'})
                    or context.get('protocol_hash')!=p.PROTOCOL_HASH or context.get('dataset')!=dataset or context.get('arm')!=arm
                    or context['query_ids']!=[r['sample_id'] for r in manifest['query']]
                    or context['candidate_ids']!=list(bank) or list(selections)!=context['query_ids']
                    or context['retrieval_rule']!=plan['protocol']['retrieval']):raise ValueError('Complete official retrieval/source identity differs')
                for value in selections.values():p.validate_selection(value,manifest['bank'])
                rows=p.prediction_prefix(folder/'predictions.jsonl',manifest['query'],selections,context,codec,complete=True)
                receipt,smoke,runtime,timing=[p.read(path) for path in
                    (folder/'verification.json',folder/'smoke.json',science/dataset/'runtime.json',folder/'retrieval_time.json')]
                if (receipt['status']!='all_complete_verified' or receipt['rows']!=len(rows)
                    or receipt['context_hash']!=context['context_hash'] or receipt['predictions_sha256']!=p.digest(folder/'predictions.jsonl')
                    or receipt['selections_sha256']!=p.digest(folder/'selections.json') or smoke['context_hash']!=context['context_hash']
                    or smoke['first_row_sha256']!=p.canonical(rows[0]) or runtime['implementation_hash']!=binding['implementation_hash']
                    or runtime['environment']!=plan['expected_environment'] or timing['context_hash']!=context['context_hash']
                    or timing['query_count']!=len(rows) or not math.isfinite(timing['seconds'])
                    or timing['seconds']<=0):raise ValueError('Complete result/runtime/actual smoke bytes differ')
                for row,query in zip(rows,manifest['query']):
                    demos=[at_cloud(bank[name],bank[name]['label']) for name in row['demo_ids']]
                    message=builders['icl_messages'](demos,at_cloud(query),dataset,data['class_labels'],224)
                    if row['prompt_fingerprint']!=p.canonical(message):raise ValueError('Actual downstream four-demo prompt differs')
                    checks+=1
                if (receipt['input_tokens']!=sum(r['input_tokens'] for r in rows)
                    or receipt['output_tokens']!=sum(r['output_tokens'] for r in rows)):
                    raise ValueError('Actual token accounting differs')
                rows_by_group[dataset,arm]=rows
    finally:runner.OUT=old
    if checks!=34856+15348:raise ValueError('Missing complete actual source/generation prompt fingerprints')
    return rows_by_group,checks


def comparison_rows(queries,task,generic):
    ids=[r['sample_id'] for r in queries]
    if (len(task)!=len(ids) or len(generic)!=len(ids)
        or [r['query_id'] for r in task]!=ids or [r['query_id'] for r in generic]!=ids):raise ValueError('No partial/unaligned paired official accuracy')
    if any(t['target']!=q['label'] or g['target']!=q['label'] for q,t,g in zip(queries,task,generic)):
        raise ValueError('Evaluation targets differ')
    result=paired_bootstrap([r['correct'] for r in task],[r['correct'] for r in generic])
    # The shared bootstrap kernel was written for rescue analysis. This is
    # full-test accuracy, so do not inherit its misleading "rescued" labels.
    rename={'first_rescued':'task_correct','second_rescued':'generic_correct',
            'first_only_rescued':'task_only_correct','second_only_rescued':'generic_only_correct'}
    return {rename.get(key,key):value for key,value in result.items()}


def complete_group_audit(full,plan,science):
    expected={(dataset,arm) for dataset in p.DATASETS for arm in p.ARMS}
    groups=full['full_groups']
    identities=[(r['dataset'],r['arm']) for r in groups]
    if len(identities)!=len(expected) or set(identities)!=expected:
        raise ValueError('All four distinct official dataset/arm groups required')
    for group in groups:
        dataset,arm=group['dataset'],group['arm']
        if (group['rows']!=len(plan['datasets'][dataset]['manifest']['query'])
            or group['predictions_sha256']!=p.digest(science/dataset/'test'/arm/'predictions.jsonl')
            or group['source_array_sha256']!=p.read(science/dataset/'sources'/arm/'verification.json')['array_sha256']):
            raise ValueError('Global complete group source/result identities differ')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--session',type=Path,required=True);parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args();session=args.session.resolve();out=args.output_root.resolve()
    if (not session.is_relative_to(p.ROOT/'results/cloud_trial') or not out.is_relative_to(p.ROOT/'analysis') or out.exists()):
        raise ValueError('Exact local takeback and fresh independent analysis root required')
    plan=p.validate_plan(p.read(p.PLAN));binding=p.validate_binding(plan,p.read(p.BINDING))
    science=session/'snapshot/results'/p.SCIENCE;operation=session/'snapshot/results'/p.OPERATION
    auth=p.read(operation/'execution_authorization.json');launch=p.read(operation/'launch.json');cloud.exact_identity(launch,auth)
    terminal_guard(p.read(session/'full_takeback_receipt.json'),p.read(session/'stop_confirmation.json'),p.read(session/'runtime_accounting.json'),auth)
    native=p.read(session/'native_identity_receipt.json')
    if (native['actual_api_pod_id']!=auth['pod_id'] or native['proc1_pod_id']!=auth['pod_id'] or native['export_file_pod_id']!=auth['pod_id']
        or native.get('ssh_process_pod_id') not in (None,auth['pod_id']) or native['expected_Pod_field_injected'] is not False
        or not native['export_root_owned_nonwritable_regular'] or p.digest(operation/'native_identity_receipt.json')!=auth['native_identity_receipt_sha256']):
        raise ValueError('Wrong original actual native source/launch proof')
    env=p.read(operation/'environment_audit.json')
    if (env['status']!='CPU_pinned_runtime_full_model_SHA_verified_not_generation' or env['versions']!=plan['expected_environment']
        or env['model_inventory']!=plan['model_inventory']):raise ValueError('Original full model/runtime evidence differs')
    full=p.read(science/'verification.json')
    if (full['status']!='all_complete_verified' or full['protocol_hash']!=p.PROTOCOL_HASH
        or full['implementation_hash']!=binding['implementation_hash'] or full['total_test_predictions']!=15348 or len(full['full_groups'])!=4):
        raise ValueError('All four complete official groups required before analysis')
    members=archive_audit(session,p.read(session/'full_takeback_receipt.json'))
    complete_group_audit(full,plan,science)
    groups,checks=prompt_audit(plan,binding,science,frozen_message_builders(plan))
    # Tokenizer-only CPU decoding from the complete original local pinned bytes.
    cache=p.ROOT/'.hf_cache/models--Qwen--Qwen3-VL-4B-Instruct/snapshots'/plan['protocol']['revision']
    for name,sha in plan['model_inventory']['files_SHA256'].items():
        if p.digest(cache/name)!=sha:raise ValueError('Original local complete model/processor identity differs')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(cache,local_files_only=True,trust_remote_code=False)
    summary=[];pairs=[]
    for dataset in p.DATASETS:
        data=plan['datasets'][dataset];queries=data['manifest']['query']
        pair=comparison_rows(queries,groups[dataset,'task'],groups[dataset,'generic'])
        pairs.append({'dataset':dataset,**pair,'first':'task','second':'generic','reference':'paired complete official test accuracy'})
        for arm in p.ARMS:
            rows=groups[dataset,arm];folder=science/dataset/'test'/arm
            codec,_=runner.codec_for(tokenizer,data['class_labels'])
            if p.read(folder/'codec.json')!=codec:raise ValueError('Actual legal codec differs from original pinned tokenizer')
            for row in rows:
                if tokenizer.decode(row['output_token_ids'],skip_special_tokens=True).strip()!=row['raw_answer']:
                    raise ValueError('Actual legal output IDs/literal answer differ')
            summary.append({'dataset':dataset,'arm':arm,'queries':len(rows),'correct':sum(r['correct'] for r in rows),
                'accuracy':rate(sum(r['correct'] for r in rows),len(rows)),'input_tokens':sum(r['input_tokens'] for r in rows),
                'output_tokens':sum(r['output_tokens'] for r in rows),'actual_generation_seconds':sum(r['seconds'] for r in rows),
                'actual_retrieval_seconds':p.read(folder/'retrieval_time.json')['seconds']})
    # Results are written only after every complete group and identity passed.
    out.mkdir(parents=True);csv_out(out/'full_official_accuracy.csv',summary);csv_out(out/'paired_task_generic.csv',pairs)
    p.immutable(out/'audit.json',{'status':'complete_verified','complete_official_test':True,'protocol_hash':p.PROTOCOL_HASH,
        'implementation_hash':binding['implementation_hash'],'queries':7674,'predictions':15348,'fresh_source_rows':34856,
        'actual_prompt_fingerprints_verified':checks,'source_archive_members_SHA':members,'session':session.relative_to(p.ROOT).as_posix(),
        'CI':'10000 query-paired bootstrap seed73, descriptive without multiplicity adjustment',
        'claim_limit':'joint task-instruction plus vocabulary conditioning; not identical thinking or unique same-failure causality',
        'full_retrieval_kernel_numerical_replay_performed':False,
        'retrieval_audit_scope':'frozen actual-score code and source/result bytes, all contexts/order/ties/scores, not independent full-array kernel replay',
        'CPU_only_no_model_or_GPU_execution':True,'audit_code_SHA':p.digest(Path(__file__))})
    print('Complete post-stop independent prompt audit and paired analysis passed')


if __name__=='__main__':main()
