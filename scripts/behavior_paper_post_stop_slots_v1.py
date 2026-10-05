"""Independent accepted-slot proof for every latest queue role, CPU only."""
from datetime import datetime
import hashlib
from pathlib import Path
import tarfile
import behavior_task_prompt_policy as p
import behavior_task_prompt_cloud as native
import behavior_paper_queue_v2_policy as queue

ROOT=p.ROOT
DONE='complete_official_test_full_takeback_stopped_independently_audited'
EXPECTED={'task_prompt':{'queries':7674,'predictions':15348,'fresh_source_rows':34856},
    'full_labels':{'queries':7674,'records':30696,'generated':30526,'explicit_missing':170},
    'qwenvision':{'queries':23256,'predictions':23256,'groups':5},
    'internvl_five':{'queries':23256,'predictions':93024,'groups':20},
    'qwen_three_seed':{'queries':46512,'predictions':279072,'groups':60}}


def terminal_guard(row,take,stop,account,audit):
    role=row['role'];expected=EXPECTED[role]
    if (take.get('stage')!='all_complete_verified' or stop.get('reason')!='all_complete_verified'
        or take.get('all_new_scientific_arrays_retrieved' if role=='task_prompt' else 'all_new_arrays_and_checkpoints_retrieved') is not True
        or not take.get('archive_sha256') or not take.get('members')
        or stop.get('pod_id')!=row['pod_id'] or account.get('pod_id')!=row['pod_id']
        or account.get('launch_id')!=row['launch_id'] or account.get('phase')!='stopped'
        or account.get('confirmed_utc')!=stop.get('confirmed_utc') or stop.get('disks_preserved') is not True
        or audit.get('status') not in ('complete_verified','all_complete_verified')
        or audit.get('complete_official_test') is not True or audit.get('session')!=row['backup_dir']
        or any(audit.get(k)!=v for k,v in expected.items())):
        raise ValueError('Complete independent official-test audit, FULL takeback and exact stop required')
    took,ended=datetime.fromisoformat(take['utc']),datetime.fromisoformat(stop['confirmed_utc'])
    if took.tzinfo is None or ended.tzinfo is None or took>ended:raise ValueError('Full byte takeback must precede exact stop')
    if role in ('qwenvision','internvl_five','qwen_three_seed'):
        audited=datetime.fromisoformat(audit['utc'])
        if audited.tzinfo is None or audited<ended or audit.get('independent_CPU_audit_after_stop') is not True:
            raise ValueError('Independent CPU audit after exact stop required')
        if audit.get('closed_archive_member_extracted_SHA_verified') is not True:
            raise ValueError('Independent closed-byte audit required')
    return {'previous_role':role,'pod_id':row['pod_id'],'launch_id':row['launch_id'],
        'confirmed_utc':ended.isoformat(),'previous_scientific_root':queue.SCOPES[role]['science']}


def archive_audit(session,take,role):
    path=session/'full_takeback.tar.gz'
    if p.digest(path)!=take['archive_sha256']:raise ValueError('Closed full archive SHA changed')
    manifest={r['name']:r for r in take['members']};target=queue.SCOPES[role]
    if len(manifest)!=len(take['members']) or take['files']!=len(manifest):raise ValueError('Duplicate full archive members')
    prefixes=('results/'+target['science']+'/','results/'+target['operation']+'/');result={}
    with tarfile.open(path) as tar:
        rows=tar.getmembers()
        if len(rows)!=len(manifest) or {r.name for r in rows}!=set(manifest):raise ValueError('Closed archive set differs')
        for member in rows:
            native.safe_relative(member.name);item=manifest[member.name]
            if (not member.isfile() or member.issym() or member.islnk() or not member.name.startswith(prefixes)
                or member.size!=item['bytes']):raise ValueError('Unsafe or wrong archive member')
            sha=hashlib.sha256()
            with tar.extractfile(member) as handle:
                while chunk:=handle.read(8*1024*1024):sha.update(chunk)
            dest=session/'snapshot'/member.name
            if dest.is_symlink() or not dest.resolve().is_relative_to(session/'snapshot') or sha.hexdigest()!=item['sha256'] or p.digest(dest)!=item['sha256']:
                raise ValueError('Archive/member/extracted scientific SHA differs')
            result[member.name]=item['sha256']
    return result


def predecessor(row,audit_path):
    session=queue.old.relative_path(ROOT,row['backup_dir'],'results');location=Path(audit_path).resolve()
    if not location.is_relative_to(ROOT/'analysis') or location.is_symlink():raise ValueError('Real local scientific audit required')
    take,stop,account,audit=(p.read(path) for path in (session/'full_takeback_receipt.json',
        session/'stop_confirmation.json',session/'runtime_accounting.json',location))
    slot=terminal_guard(row,take,stop,account,audit)
    if row['role'] in ('qwenvision','internvl_five','qwen_three_seed'):
        if row['role']=='qwenvision':
            from behavior_critical_followup_cloud import config
        elif row['role']=='internvl_five':
            from behavior_internvl_five_native_cloud_v4 import config
        else:
            from behavior_qwen_three_seed_cloud_v1 import config
        cfg=config(row['role'])
        if (audit.get('plan_hash')!=cfg['plan_hash'] or audit.get('implementation_hash')!=cfg['implementation_hash']
            or audit.get('archive_SHA')!=take['archive_sha256']):raise ValueError('Exact accepted science and byte provenance differs')
    else:
        expected={'task_prompt':('c4620bfd0984879ae5ef318411ed88a83a6ec2a8b3271173ef648b8e56a6dbb8','4e5ab5d4389b2b43ed391edcff0ebebb56d1bf2d6ece51a93a6524db9fc1d580'),
            'full_labels':('6ce80bf47d662198c9d2f593bce04cf83ddc04791dffa07401eb8f3a82f48096','5054d46fc48e07f9e2be17f1cb87677e09358ffedea8fed6ed4a673d27befe8b')}[row['role']]
        if (audit.get('protocol_hash'),audit.get('implementation_hash'))!=expected:raise ValueError('Predecessor original identity differs')
    archive_audit(session,take,row['role'])
    return {**slot,'audit_path':location.relative_to(ROOT).as_posix(),'audit_SHA':p.digest(location),
        'takeback_archive_SHA':take['archive_sha256'],'CPU_audit_confirmed_after_stop':True}


def completed_roles(registry):
    roles=set()
    for row in registry.get('ended_jobs',[]):
        if row.get('status')==DONE:
            predecessor(row,ROOT/row['audit_path']);roles.add(row['role'])
    return roles


def completed_candidates(registry,selected):
    candidates=[selected];consumed=set()
    for row in registry.get('active_jobs',[])+registry.get('ended_jobs',[]):
        path=ROOT/row['deployment_receipt']
        if path.exists():
            old=p.read(path).get('predecessor',{})
            if old.get('pod_id') and old.get('launch_id'):consumed.add((old['pod_id'],old['launch_id']))
    if (selected['pod_id'],selected['launch_id']) in consumed:raise ValueError('Logical predecessor slot has already been consumed')
    for row in registry.get('ended_jobs',[]):
        identity=(row['pod_id'],row['launch_id'])
        if row.get('status')==DONE and identity not in consumed and identity!=(selected['pod_id'],selected['launch_id']):
            candidates.append(predecessor(row,ROOT/row['audit_path']))
    return candidates
