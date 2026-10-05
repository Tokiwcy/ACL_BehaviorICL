"""CPU-safe independent serial operations for the other three frozen sciences.

This is not an automatic queue. Each mode has its own package, launch, native
proof, successful backup, parent/watch, complete takeback, and exact stop.
Scientific plans/runners and old deployment entries are never patched/called.
"""
from datetime import datetime,timezone
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import tarfile
import time
import zipfile

import behavior_task_prompt_cloud as native

p=native.p
ROOT=p.ROOT
MODES=('qwenvision',)
PARENT='scripts/bootstrap_behavior_qwenvision_two_v1.py'
WATCHER='scripts/watch_behavior_qwenvision_two_v1.py'
ENTRY='scripts/start_behavior_qwenvision_two_retained_v21.py'
CHECKER='scripts/check_behavior_paper_two_qv_v1_live.py'
BINDING=ROOT/'analysis/qwenvision_two_cloud_v1/implementation_lock.json'
EXTENSIONS=('.json','.jsonl','.log')
FULL_EXTENSIONS=EXTENSIONS+('.npy','.pt')
AUTH_KEYS=('mode','experiment','plan_hash','implementation_hash','operation_hash','pod_id','launch_id')


def config(mode):
    if mode!='qwenvision':raise ValueError('Only human-authorized DTD+CUB QwenVision')
    import behavior_qwenvision_two_host_policy_v1 as policy
    plan=policy.validate_plan(policy.read(policy.PLAN_PATH));binding=policy.validate_binding(plan)
    source=p.read(ROOT/'analysis/qwenvision_two_fresh_v1/implementation_lock.json')
    if source['original_plan_hash']!=plan['plan_hash'] or source['original_host_implementation_hash']!=binding['implementation_hash']:
        raise ValueError('Exact new two-dataset identities required')
    return {'mode':mode,'science':policy.SCIENCE,'operation':policy.OPERATION,
        'runner':'scripts/run_behavior_qwenvision_two_native_v1.py','plan':plan,'binding':binding,'policy':policy,
        'plan_hash':plan['plan_hash'],'implementation_hash':binding['implementation_hash'],
        'code_sha256':binding['code_sha256'],'paths':[policy.PLAN_PATH,policy.BINDING,ROOT/'analysis/qwenvision_two_fresh_v1/implementation_lock.json'],
        'environment':binding['environment'],'models':{policy.MODEL:{'revision':binding['model_inventory']['revision'],
        'files':{name:{'sha256':sha} for name,sha in binding['model_inventory']['files_SHA256'].items()}}},
        'manifest':{d:v['manifest'] for d,v in plan['datasets'].items()},'venv':'.venv-qwenvision-two-v1',
        'predictions':7674,'records':7674,'source_contract_hash':source['source_contract_hash']}


def work(mode):
    if mode not in MODES:raise ValueError('Unknown followup')
    return ROOT/'.tools'/('paper_'+mode+'_two_v1')


def exact(record,auth):
    if any(not auth.get(key) or record.get(key)!=auth[key] for key in AUTH_KEYS):
        raise ValueError('Wrong independent followup launch/operation identity')


def schedule(cfg,python):
    base=[str(python),'-u',cfg['runner']];jobs=[]
    if cfg['mode']=='full_labels':
        for d in cfg['manifest']:
            selected=base+['--dataset',d]
            jobs.append(('CPU_full_official_preflight_not_science',selected+['--phase','preflight']))
            jobs.append(('actual_fixed_four_arm_smoke',selected+['--phase','capture','--smoke','--allow-gpu']))
            jobs.append(('complete_official_fixed_label_capture',selected+['--phase','capture','--allow-gpu']))
            jobs.append(('complete_official_verification',selected+['--phase','verify']))
    elif cfg['mode']=='qwenvision':
        jobs.append(('CPU_full_two_official_preflight_not_science',base+['--phase','preflight']))
        for d in cfg['manifest']:
            selected=base+['--dataset',d,'--allow-gpu']
            jobs.append(('actual_native_vision_smoke',selected+['--phase','capture','--smoke']))
            jobs.append(('complete_native_vision_capture',selected+['--phase','capture']))
            jobs.append(('fixed256_1500step_SupCon',selected+['--phase','train']))
            jobs.append(('actual_retrieved_four_demo_smoke',selected+['--phase','test','--smoke']))
            jobs.append(('complete_official_test',selected+['--phase','test']))
        jobs.append(('complete_two_group_verification',base+['--phase','verify']))
    else:
        jobs.append(('CPU_full_official_preflight_not_science',base+['--phase','preflight']))
        for d in cfg['manifest']:
            selected=base+['--dataset',d,'--allow-gpu']
            jobs.extend([('actual_native_source_and_five_image_smoke',selected+['--phase','smoke']),
                         ('complete_native_and_CLIP_capture',selected+['--phase','capture']),
                         ('fixed256_1500step_SupCon',selected+['--phase','train'])])
            for method in cfg['policy'].METHODS:
                scoped=selected+['--method',method]
                jobs.extend([('actual_full_retrieval',scoped+['--phase','retrieve']),
                             ('actual_native_legal_consumer_smoke',scoped+['--phase','generate','--smoke']),
                             ('complete_official_test',scoped+['--phase','generate'])])
        jobs.append(('complete_eight_group_verification',base+['--phase','verify']))
    return jobs


def image_inventory(cfg):
    result={}
    for data in cfg['manifest'].values():
        for split in ('bank','query'):
            for row in data[split]:
                relative=row['path'].replace('\\','/')
                # Full-label retrospective manifests retain their immutable
                # historical Windows anchor. Attached execution paths are
                # portable; normalize only the transport destination, never
                # rewrite that frozen plan or allow arbitrary absolute paths.
                if cfg['mode']=='full_labels':
                    anchors=('C:/Users/ASUS/Documents/ChatGPT/BehaviorICL/','/workspace/BehaviorICL/')
                    matches=[anchor for anchor in anchors if relative.startswith(anchor)]
                    if len(matches)!=1:raise ValueError('Unknown historical full-label image anchor')
                    relative=relative[len(matches[0]):]
                native.safe_relative(relative)
                if not relative.startswith('datasets/') or relative in result:raise ValueError('Duplicate/out-of-scope fixed image')
                result[relative]=row['image_sha256']
    if len(result)!=(17428 if cfg['mode']=='qwenvision' else 17428):raise ValueError('Wrong complete image denominator')
    return result


def additional_inputs(cfg):
    metadata=('dtd/labels/train1.txt','dtd/labels/val1.txt','dtd/labels/test1.txt',
        'CUB_200_2011/images.txt','CUB_200_2011/image_class_labels.txt','CUB_200_2011/classes.txt','CUB_200_2011/train_test_split.txt')
    return {'datasets/'+name:{'local_path':str(ROOT/'datasets'/name),'sha256':p.digest(ROOT/'datasets'/name)} for name in metadata}


def validate_package(package,cfg):
    if (package.get('package_hash')!=p.canonical({k:v for k,v in package.items() if k!='package_hash'})
            or package.get('mode')!=cfg['mode'] or package.get('experiment')!=cfg['science']
            or package.get('operation')!=cfg['operation'] or package.get('plan_hash')!=cfg['plan_hash']
            or package.get('implementation_hash')!=cfg['implementation_hash']
            or package.get('images')!=image_inventory(cfg) or package.get('environment')!=cfg['environment']
            or package.get('models')!=cfg['models'] or package.get('source_contract_hash')!=cfg['source_contract_hash']):raise ValueError('Independent package science/scope changed')
    for relative,sha in package['code_sha256'].items():
        native.safe_relative(relative)
        if p.digest(ROOT/relative)!=sha:raise ValueError('Frozen independent followup code/input changed: '+relative)
    if any(package['code_sha256'].get(k)!=v for k,v in cfg['code_sha256'].items()):raise ValueError('Frozen scientific code is not bound')


def implementation_check():
    binding=p.read(BINDING)
    if binding.get('implementation_hash')!=p.canonical({k:v for k,v in binding.items() if k!='implementation_hash'}):
        raise ValueError('Independent followup CPU implementation binding changed')
    for rel,sha in binding['code_sha256'].items():
        if p.digest(ROOT/rel)!=sha:raise ValueError('Frozen independent followup code changed: '+rel)
    for mode,saved in binding['scientific_bindings'].items():
        cfg=config(mode)
        if saved!={'plan_hash':cfg['plan_hash'],'scientific_implementation_hash':cfg['implementation_hash'],
                   'environment_hash':p.canonical(cfg['environment']),'model_inventory_hash':p.canonical(cfg['models']),
                   'images':len(image_inventory(cfg)),'predictions':cfg['predictions'],'records':cfg['records']}:
            raise ValueError('Frozen independent followup scientific scope changed')
    return binding


def freeze(mode):
    implementation_check()
    cfg=config(mode);folder=work(mode)
    if folder.exists():raise ValueError('Independent followup package already exists; no overwrite')
    files={**cfg['code_sha256'],**p.read(BINDING)['code_sha256']}
    from prepare_behavior_qwen_three_seed import local_import_closure
    files.update({n:p.digest(ROOT/n) for n in local_import_closure(tuple(files))})
    # Common imports are pure CPU/native helpers; no old entry/main is invoked.
    for relative in ('scripts/behavior_qwen_three_seed_cloud_v1.py',PARENT,WATCHER,ENTRY,CHECKER,
                     'scripts/behavior_task_prompt_cloud.py','scripts/behavior_task_prompt_policy.py',
                     'scripts/behavior_p2_matched_policy.py','scripts/collect_behavior_priority_zero.py',
                     'scripts/diagnose_behavior_prompt_transport_once.py'):
        files[relative]=p.digest(ROOT/relative)
    for path in cfg['paths']:files[path.relative_to(ROOT).as_posix()]=p.digest(path)
    files[BINDING.relative_to(ROOT).as_posix()]=p.digest(BINDING)
    images=image_inventory(cfg);extra=additional_inputs(cfg)
    for relative,sha in images.items():
        if p.digest(ROOT/relative)!=sha:raise ValueError('Complete image byte identity changed')
    # Registered provenance is audited locally; not uploaded as unrelated jobs.
    if mode=='full_labels':
        for relative,sha in p.read(cfg['policy'].INPUT)['source_sha256'].items():
            if p.digest(ROOT/relative)!=sha:raise ValueError('Registered original full-label provenance changed')
    folder.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(folder/'inputs.zip','x',zipfile.ZIP_STORED) as z:
        for relative in images:z.writestr(zipfile.ZipInfo(relative,date_time=(1980,1,1,0,0,0)),(ROOT/relative).read_bytes())
        for relative,item in extra.items():
            if p.digest(item['local_path'])!=item['sha256']:raise ValueError('Complete allowed extra input changed')
            z.writestr(zipfile.ZipInfo(relative,date_time=(1980,1,1,0,0,0)),Path(item['local_path']).read_bytes())
    package={'mode':mode,'experiment':cfg['science'],'operation':cfg['operation'],'plan_hash':cfg['plan_hash'],
        'implementation_hash':cfg['implementation_hash'],'code_sha256':files,'images':images,'source_contract_hash':cfg['source_contract_hash'],
        'extra_inputs':{rel:{'sha256':item['sha256']} for rel,item in extra.items()},
        'inputs_zip_sha256':p.digest(folder/'inputs.zip'),'environment':cfg['environment'],'models':cfg['models'],
        'planned_predictions':cfg['predictions'],'planned_records':cfg['records'],
        'scientific_changes':False,'new_Pod_creation':False,'GPU_execution':False,'CPU_tests_not_cloud_compatibility':True}
    package['package_hash']=p.canonical(package);validate_package(package,cfg)
    p.immutable(folder/'package.json',package)
    with zipfile.ZipFile(folder/'code.zip','x',zipfile.ZIP_DEFLATED) as z:
        for relative in files:z.write(ROOT/relative,relative)
        z.write(folder/'package.json',(folder/'package.json').relative_to(ROOT).as_posix())
    p.immutable(folder/'package_receipt.json',{'utc':p.now(),'package_hash':package['package_hash'],
        'code_zip_SHA':p.digest(folder/'code.zip'),'inputs_zip_SHA':package['inputs_zip_sha256'],
        'complete_images':len(images),'code_and_input_files':len(files)+1,'cloud_compatibility':False})
    return package


def backup_guard(cfg,auth):
    folder=ROOT/'results'/cfg['operation']
    native.p.native_identity(auth['pod_id']);exact(p.read(folder/'execution_authorization.json'),auth)
    ready,heartbeat=p.read(folder/'backup_ready.json'),p.read(folder/'backup_heartbeat.json')
    for record in (ready,heartbeat):
        exact(record,auth)
        if (record.get('status')!='actual_backup_succeeded' or not record.get('watcher_identity_verified')
                or record.get('file_count',0)<=0 or len(record.get('archive_sha256',''))!=64):
            raise ValueError('No actual new successful backup protection')
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(heartbeat['utc'])).total_seconds()
    if not 0<=age<=300:raise ValueError('Backup stale/future')
    if (folder/'stop_request.json').exists():raise ValueError('Independent safety stop requested')


def process_state(cfg,launch,terminal=False):
    parent=None
    try:
        parent=native.process_info(launch['pid'])
        if (parent['cwd']!=str(ROOT) or PARENT not in parent['args'] or cfg['mode'] not in parent['args']
                or parent['start_ticks']!=launch['start_ticks']):raise ValueError('Wrong exact parent process/cwd/start-time')
    except (FileNotFoundError,ProcessLookupError):
        if not terminal:raise ValueError('Independent parent disappeared before terminal state')
    children=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:info=native.process_info(int(proc.name))
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if info['ppid']==launch['pid'] and cfg['runner'] in info['args']:
            if info['cwd']!=str(ROOT):raise ValueError('Wrong scientific cwd')
            children.append(info)
    pids=sorted({int(v.strip()) for v in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines() if v.strip().isdigit()})
    if len(children)>1 or len(pids)>1 or any(v not in {r['pid'] for r in children} for v in pids):
        raise ValueError('Unrelated/parallel GPU process')
    return parent,children,pids


def stop_bound(parent,children):
    bound=children+([parent] if parent else [])
    for info in bound:
        try:os.kill(info['pid'],signal.SIGTERM)
        except ProcessLookupError:pass
    before=time.monotonic()
    while time.monotonic()-before<10 and any(Path('/proc',str(r['pid'])).exists() for r in bound):time.sleep(.2)
    for info in bound:
        try:
            if native.process_info(info['pid'])['start_ticks']==info['start_ticks']:os.kill(info['pid'],signal.SIGKILL)
        except (FileNotFoundError,ProcessLookupError):pass


def committed(cfg,science):
    result={}
    for path in science.rglob('*.jsonl'):
        if path.name not in ('predictions.jsonl','vision_capture.jsonl','capture.jsonl'):continue
        count=0
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                if not line.endswith('\n'):break
                row=json.loads(line)
                if cfg['mode']=='full_labels':
                    if row.get('job_hash') not in cfg['binding']['job_hashes'].values():raise ValueError('Wrong full-label committed job')
                elif row.get('plan_hash')!=cfg['plan_hash']:raise ValueError('Wrong committed source/prediction plan')
                count+=1
        if count:result[path.relative_to(science).as_posix()]=count
    for path in science.glob('*/trained/*/d256/progress.json'):
        value=p.read(path);steps=value.get('completed_steps')
        if type(steps) is not int or not 0<=steps<=1500:raise ValueError('Wrong committed fixed256 training progress')
        if steps:result[path.relative_to(science).as_posix()]=steps
    return result


def safe_members(tar,cfg,full):
    names=set();prefixes=('results/'+cfg['science']+'/','results/'+cfg['operation']+'/')
    for member in tar.getmembers():
        rel=native.safe_relative(member.name)
        if (member.name in names or not member.name.startswith(prefixes) or not member.isfile()
                or member.issym() or member.islnk() or rel.suffix not in (FULL_EXTENSIONS if full else EXTENSIONS)):
            raise ValueError('Unsafe duplicate/out-of-scope scientific backup member')
        names.add(member.name)
    return tar.getmembers()


def snapshot(mode,auth,previous,reason=None):
    if str(ROOT)!='/workspace/BehaviorICL':raise ValueError('Cloud exact snapshot only')
    cfg=config(mode);out=ROOT/'results'/cfg['operation'];science=ROOT/'results'/cfg['science']
    native.p.native_identity(auth['pod_id']);exact(p.read(out/'execution_authorization.json'),auth)
    launch=p.read(out/'launch.json');exact(launch,auth)
    stage=p.read(out/'sequence_progress.json') if (out/'sequence_progress.json').exists() else {'stage':'waiting_CPU_parent_not_science'}
    if (out/'stop_request.json').exists():
        stop=p.read(out/'stop_request.json');exact(stop,auth);reason=reason or stop['reason']
    parent=children=None;pids=[]
    try:parent,children,pids=process_state(cfg,launch,stage['stage'] in ('failed','all_complete_verified'))
    except (ValueError,FileNotFoundError,ProcessLookupError) as error:
        reason=reason or 'exact_process_GPU_fault';stage={'stage':'failed','error_class':type(error).__name__}
        # Never signal an unbound PID. The local watcher still stops exact Pod.
    if reason or stage['stage']=='failed':
        if children is not None:stop_bound(parent,children)
        stage={**stage,'stage':'failed','reason':reason or 'scientific_parent_failure','utc':p.now()};p.atomic(out/'sequence_progress.json',stage)
    full=stage['stage'] in ('failed','all_complete_verified')
    signature=committed(cfg,science);files=[];signatures={}
    for group in (science,out):
        for path in group.rglob('*'):
            if not path.is_file() or path.suffix not in (FULL_EXTENSIONS if full else EXTENSIONS):continue
            if path.is_symlink() or not path.resolve().is_relative_to(group):raise ValueError('Linked scientific evidence refused')
            name=path.relative_to(ROOT).as_posix();info=path.stat();signatures[name]=[info.st_size,info.st_mtime_ns]
            if full or previous.get(name)!=signatures[name]:files.append(path)
    archive=out/('takeback_'+auth['launch_id']+'.tar.gz');members=[]
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for path in files:
            name=path.relative_to(ROOT).as_posix()
            if path.suffix in ('.npy','.pt'):
                before=path.stat();tar.add(path,arcname=name,recursive=False);sha=p.digest(path);after=path.stat()
                if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Full quiescent scientific array/checkpoint changed')
                size=before.st_size
            else:
                blob=path.read_bytes();member=tarfile.TarInfo(name);member.size=len(blob);member.mode=0o600
                tar.addfile(member,io.BytesIO(blob));size=len(blob);sha=hashlib.sha256(blob).hexdigest()
            members.append({'name':name,'bytes':size,'sha256':sha})
    return {'utc':p.now(),'stage':stage,'full':full,'archive':str(archive),'archive_sha256':p.digest(archive),
            'members':members,'signatures':signatures,'science_signature':signature,'remote_committed_coverage':signature,
            'actual_parent':parent,'actual_children':children,'GPU_compute_pids':pids,'launch':launch}


def complete(cfg):
    science=ROOT/'results'/cfg['science']
    if cfg['mode']=='full_labels':
        rows=[p.read(science/d/'verification.json') for d in cfg['manifest']]
        if any(v.get('status')!='all_complete_verified' for v in rows) or sum(v['records'] for v in rows)!=30696 or sum(v['generated'] for v in rows)!=30526:
            raise ValueError('Both complete official label groups required')
        return {'records':30696,'generations':30526,'explicit_missing':170}
    value=p.read(science/'verification.json')
    if value.get('status')!='all_complete_verified':raise ValueError('Scientific complete verification absent')
    groups=value['full_groups'] if cfg['mode']=='qwenvision' else value['groups']
    if sum(v['rows'] for v in groups)!=cfg['records'] or len(groups)!=(2 if cfg['mode']=='qwenvision' else 8):raise ValueError('Complete official method groups required')
    return {'records':cfg['records'],'generations':cfg['predictions']}


def package_audit(mode):
    implementation_check();cfg=config(mode);folder=work(mode)
    package=p.read(folder/'package.json');validate_package(package,cfg)
    receipt=p.read(folder/'package_receipt.json')
    if receipt['package_hash']!=package['package_hash'] or receipt['code_zip_SHA']!=p.digest(folder/'code.zip'):
        raise ValueError('Closed independent code package differs')
    if receipt['inputs_zip_SHA']!=p.digest(folder/'inputs.zip') or receipt['inputs_zip_SHA']!=package['inputs_zip_sha256']:
        raise ValueError('Closed independent input package differs')
    return cfg,package,receipt


def frozen_scope_check(mode,cfg):
    if mode!='qwenvision' or cfg['science']!='behavior_qwenvision_supcon_dtd_cub_v1' or tuple(cfg['manifest'])!=('dtd','cub'):
        raise ValueError('Human authorized two-group science required')
    if cfg['predictions']!=7674 or cfg['records']!=7674 or len(image_inventory(cfg))!=17428:
        raise ValueError('Complete official DTD+CUB denominator required')
    config(mode)
