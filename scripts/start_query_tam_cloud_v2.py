"""Once-only independent small TAM pilot and exact local watchdog.

Default CPU audit never creates a resource. One new volume, one native GPU
worker, actual 120-second closed SHA backups, immediate stop before CPU audit.
The old paper science registry and frozen entry points are never modified.
"""
import argparse,base64,hashlib,json,os,shlex,subprocess,sys,time,traceback,uuid,zipfile
from contextlib import contextmanager
from pathlib import Path
import query_tam_cloud_ops_v1 as ops
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.tools/query_tam_cloud_v2'
PLAN=ROOT/'analysis/query_tam_pilot_v1/protocol.json'
CTL=ROOT/'.tools/runpodctl/runpodctl.exe'
SCRIPTS=('scripts/run_query_tam_pilot_v1.py','scripts/query_tam_adapter_v1.py',
         'scripts/query_tam_cloud_ops_v1.py','scripts/start_query_tam_cloud_v2.py')
FIELDS=('id','desiredStatus','runtimeStatus','runtimeStatusReason','gpuCount','costPerHr')

@contextmanager
def lock(path):
    with Path(path).open('a+b') as f:
        if f.tell()==0:f.write(b'0');f.flush()
        f.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            f.seek(0)
            if os.name=='nt':msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f.fileno(),fcntl.LOCK_UN)

def command(args,*,input=None,timeout=60,output=None,label='command'):
    try:
        result=subprocess.run(args,input=input.encode() if isinstance(input,str) else input,
            stdout=output if output is not None else subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
    except subprocess.TimeoutExpired as error:
        ops.atomic(WORK/('error_'+str(time.time_ns())+'.json'),{'utc':ops.now(),'label':label,
            'error_class':'TimeoutExpired','partial_stdout_bytes':len(error.stdout or b''),
            'partial_stdout_SHA':hashlib.sha256(error.stdout or b'').hexdigest(),
            'partial_stderr_bytes':len(error.stderr or b''),'secret_content_saved':False})
        raise RuntimeError(label+' timed out; no automatic replay') from None
    if result.returncode:
        ops.atomic(WORK/('error_'+str(time.time_ns())+'.json'),{'utc':ops.now(),'label':label,
            'returncode':result.returncode,'stdout_bytes':len(result.stdout or b''),
            'stderr_SHA':hashlib.sha256(result.stderr).hexdigest(),'secret_content_saved':False})
        raise RuntimeError(label+' failed; returncode='+str(result.returncode))
    return result.stdout.decode('utf-8') if output is None else None

def api(pod):
    actual=json.loads(command([str(CTL),'pod','get',pod],timeout=30,label='fresh_exact_API'))
    if actual.get('id')!=pod:raise ValueError('API exact identity differs')
    ops.atomic(WORK/'public_API'/('actual_'+str(time.time_ns())+'.json'),
        {'utc':ops.now(),'actual':{key:actual.get(key) for key in FIELDS},
         'ssh_authority_present':bool(actual.get('ssh')),'expected_fields_injected':False})
    return actual

def account(expected=()):
    actual=json.loads(command([str(CTL),'pod','list'],label='account_read_only'))
    running=[x for x in actual if x.get('desiredStatus')=='RUNNING']
    if len(running)>2 or any(x.get('gpuCount')!=1 for x in running) or {x['id'] for x in running}!=set(expected):
        raise ValueError('Unregistered/third/multi-GPU resource; no new action')
    return actual

def ssh(pod):
    actual=api(pod);endpoint=actual.get('ssh') or {}
    if (actual.get('desiredStatus'),actual.get('runtimeStatus'),actual.get('gpuCount'))!=('RUNNING','running',1) or not endpoint.get('ip') or not endpoint.get('port'):
        raise ValueError('Fresh exact running single-GPU SSH authority absent')
    return ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=accept-new',
        '-i','C:/Users/ASUS/.runpod/ssh/runpodctl-ssh-key','-p',str(endpoint['port']),'root@'+endpoint['ip']]

def remote(pod,code,label,timeout=60):
    return json.loads(command(ssh(pod)+['python -u -'],input=code,timeout=timeout,label=label))

def cim(launch):
    text=command(['powershell','-NoProfile','-Command',
        "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -like '*start_query_tam_cloud_v2.py*' -and $_.CommandLine -like '*"+launch+"*' } | Select-Object ProcessId,ParentProcessId,CommandLine,@{Name='CreationUTC';Expression={$_.CreationDate.ToUniversalTime().ToString('o')}} | ConvertTo-Json -Compress"],label='exact_CIM_watch',timeout=20)
    rows=json.loads(text) if text.strip() else []
    if isinstance(rows,dict):rows=[rows]
    if len(rows)!=1 or rows[0]['ProcessId']!=os.getpid():raise ValueError('Exactly one bound local TAM watchdog required')
    return rows[0]

def package():
    WORK.mkdir(parents=True,exist_ok=True)
    if (WORK/'package_receipt.json').exists():return audit()
    plan=ops.read(PLAN)
    if plan['protocol_hash']!='568f17e6b054bda816d41a18a7e7c8a434b6d9255cdb9a486298cece90a2830f' or len(plan['examples'])!=8:raise ValueError('Fixed pilot protocol differs')
    files={name:ROOT/name for name in SCRIPTS}
    files['protocol.json']=PLAN
    for name in ('tam.py','demo.py','qwen_utils.py','requirements.txt','README.md','source_receipt.json'):
        files['tam/'+name]=ROOT/'.tools/tam_official_v1'/name
    for example in plan['examples']:
        file=ROOT/example['image_path']
        if ops.digest(file)!=example['image_SHA']:raise ValueError('Exact pilot image SHA differs')
        files[example['bundle_path']]=file
    members={name:ops.digest(file) for name,file in files.items()}
    for name in members:ops.relative(name)
    with zipfile.ZipFile(WORK/'package.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name,file in files.items():z.write(file,name)
    receipt={'protocol_hash':plan['protocol_hash'],'members':members,'zip_SHA':ops.digest(WORK/'package.zip'),
        'zip_bytes':(WORK/'package.zip').stat().st_size,'single_cloud_GPU':True,'local_weights_or_CUDA':False,
        'old_science_registry_unchanged':True,'new_scientific_entry_not_consumed':True}
    receipt['operation_hash']=hashlib.sha256(json.dumps(receipt,sort_keys=True).encode()).hexdigest()
    ops.atomic(WORK/'package_receipt.json',receipt)
    return audit()

def audit():
    receipt=ops.read(WORK/'package_receipt.json')
    if ops.digest(WORK/'package.zip')!=receipt['zip_SHA']:raise ValueError('Frozen package bytes differ')
    with zipfile.ZipFile(WORK/'package.zip') as z:
        if set(z.namelist())!=set(receipt['members']):raise ValueError('Package closure differs')
        for name,expected in receipt['members'].items():
            data=z.read(name)
            if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Package member SHA differs')
            local=PLAN if name=='protocol.json' else ROOT/name if name in SCRIPTS else ROOT/'.tools/tam_official_v1'/name[4:] if name.startswith('tam/') else None
            if local and ops.digest(local)!=expected:raise ValueError('Frozen executable/local source differs')
    return receipt

def code(root,auth,body):
    return 'import sys,json\nfrom pathlib import Path\nroot=Path('+repr(root)+')\nsys.path.insert(0,str(root/"scripts"))\nimport query_tam_cloud_ops_v1 as c\nauth='+repr(auth)+'\n'+body

def upload_code(root,auth,receipt):
    payload=base64.b64encode((WORK/'package.zip').read_bytes()).decode()
    upload='import pathlib,json,base64,io,zipfile,hashlib\nroot=pathlib.Path('+repr(root)+')\n'
    upload+='if root.exists() or root.is_symlink() or pathlib.Path("/workspace/BehaviorICL").exists():raise ValueError("New independent volume/root required")\nroot.mkdir(parents=True)\n'
    upload+='data=base64.b64decode('+repr(payload)+')\nexpected='+repr(receipt['members'])+'\n'
    upload+='if hashlib.sha256(data).hexdigest()!='+repr(receipt['zip_SHA'])+':raise ValueError("Package stream SHA differs")\n'
    upload+='with zipfile.ZipFile(io.BytesIO(data)) as z:\n if set(z.namelist())!=set(expected):raise ValueError("Closed member set differs")\n for name,sha in expected.items():\n  if name.startswith("/") or ".." in name.split("/") or "\\\\" in name:raise ValueError("Unsafe package path")\n  value=z.read(name)\n  if hashlib.sha256(value).hexdigest()!=sha:raise ValueError("Package member SHA differs")\n  target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(value)\n'
    upload+='(root/"results").mkdir()\n(root/"auth.json").write_text(json.dumps('+repr(auth)+'),encoding="utf-8")\n(root/"upload.json").write_text(json.dumps({"zip_SHA":'+repr(receipt['zip_SHA'])+',"member_SHA_verified":True}))\nprint(json.dumps({"independent_package_accepted":True,"members":len(expected)}),flush=True)\n'
    return upload


def backup(pod,root,auth,folder,full=False):
    owner=cim(auth['launch_id'])
    record=remote(pod,code(root,auth,'print(json.dumps(c.archive(root,auth,'+repr(full)+')),flush=True)'), 'closed_FULL' if full else 'closed_SHA_backup',timeout=90)
    if (record['pod_id'],record['launch_id'],record['full'])!=(pod,auth['launch_id'],full):raise ValueError('Archive exact identity differs')
    if not record['path'].startswith(root+'/archives/backup_') or len(record['archive_SHA'])!=64:raise ValueError('Independent archive path differs')
    number=len(list(folder.glob('cycle_*.json')))+1
    path=folder/('FULL.tar.gz' if full else 'backup_'+str(number)+'.tar.gz')
    transfer="import pathlib,sys;sys.stdout.buffer.write(pathlib.Path("+repr(record['path'])+").read_bytes())"
    with path.open('wb') as f:command(ssh(pod)+['python -c '+shlex.quote(transfer)],output=f,timeout=180,label='archive_binary_transfer')
    if path.stat().st_size!=record['bytes']:raise ValueError('Closed transfer byte count differs')
    destination=ROOT/'.tools/tam_snapshots_v1'/auth['launch_id']/str(number)
    members=ops.unpack(path,destination,record)
    value={**record,'utc':ops.now(),'snapshot':str(destination),'local_archive':str(path),
        'closed_archive_member_extracted_SHA_verified':True,'file_count':len(members),'exact_CIM_watch':owner}
    ops.atomic(folder/('cycle_'+str(number)+'.json'),value)
    if full:ops.atomic(folder/'full_takeback_receipt.json',value)
    heartbeat={'pod_id':pod,'launch_id':auth['launch_id'],'utc':time.time(),'archive_SHA':record['archive_SHA'],
               'closed_archive_member_extracted_SHA_verified':True}
    remote(pod,code(root,auth,'c.atomic(root/"results/backup_heartbeat.json",'+repr(heartbeat)+')\nprint(json.dumps({"written":True}))'), 'actual_successful_backup_heartbeat')
    ops.atomic(folder/'actual_backup_heartbeat.json',heartbeat)
    return value

def stop(pod,auth,folder,reason):
    if (pod,auth.get('pod_id'))!=(pod,pod) or not auth.get('launch_id'):raise ValueError('Complete exact stop identity required')
    actual=api(pod)
    if actual.get('desiredStatus')=='RUNNING':command([str(CTL),'pod','stop',pod],label='exact_STOP',timeout=40)
    deadline=time.monotonic()+180
    while True:
        actual=api(pod)
        if (actual.get('desiredStatus'),actual.get('runtimeStatus'))==('EXITED','stopped'):break
        if time.monotonic()>deadline:raise ValueError('Exact fresh stopped confirmation missing')
        time.sleep(5)
    fee=ops.read(folder/'runtime_accounting.json');seconds=max(0,time.time()-fee['start_epoch'])
    record={'utc':ops.now(),'pod_id':pod,'launch_id':auth['launch_id'],'reason':reason,'disks_preserved':True,
        'fresh_API':{key:actual.get(key) for key in FIELDS},'GPU_seconds_estimate':seconds,
        'GPU_cost_estimate_USD':seconds*fee['cost_per_hour']/3600,'estimate_not_invoice':True,'storage_included':False}
    ops.atomic(folder/'stop_confirmation.json',record)
    deployment=ops.read(WORK/'deployment_receipt.json');deployment.update(phase='stopped_'+reason,stop=record)
    ops.atomic(WORK/'deployment_receipt.json',deployment)
    print(json.dumps({'TAM_exact_stopped':True,'reason':reason,'segment_GPU_estimate_USD':record['GPU_cost_estimate_USD']}),flush=True)

def choose_card(catalog):
    for name in ('NVIDIA GeForce RTX 5090','NVIDIA RTX PRO 4500 Blackwell','NVIDIA RTX PRO 4500 Blackwell Server Edition'):
        rows=[x for x in catalog if x.get('gpuId')==name and x.get('available') is True and x.get('secureCloud') is True]
        if len(rows)==1:return rows[0]
    raise ValueError('Actually available secure compatible single GPU absent; no create')


def launch(launch_id):
    receipt=audit()
    if (WORK/'launch_consumed.json').exists():raise ValueError('This entry already consumed; monitor it, never repeat')
    if len(launch_id)!=32 or any(x not in '0123456789abcdef' for x in launch_id):raise ValueError('Exact launch UUID required')
    ops.atomic(WORK/'launch_consumed.json',{'utc':ops.now(),'launch_id':launch_id,'operation_hash':receipt['operation_hash']})
    owner=cim(launch_id);ops.atomic(WORK/'watch_launch.json',owner)
    pod=folder=root=None;auth={'launch_id':launch_id};started=time.time()
    try:
        with lock(ROOT/'.tools/cloud_resource_creation.lock'):
            account()
            catalog=json.loads(command([str(CTL),'gpu','list'],label='available_GPU_read_only'))
            gpu=choose_card(catalog);name='BehaviorICL-TAM-'+launch_id[:12]
            deployment={'utc':ops.now(),'launch_id':launch_id,'name':name,'operation_hash':receipt['operation_hash'],
                'protocol_hash':receipt['protocol_hash'],'phase':'new_capacity_pending_not_science','single_GPU':True}
            ops.atomic(WORK/'deployment_receipt.json',deployment)
            try:
                created=json.loads(command([str(CTL),'pod','create','--template-id','runpod-torch-v280',
                    '--gpu-id',gpu['gpuId'],'--gpu-count','1','--cloud-type','SECURE','--name',name,
                    '--container-disk-in-gb','60','--volume-in-gb','60','--ports','22/tcp'],timeout=60,label='once_new_TAM_capacity'))
                pod=created.get('id')
            except BaseException:
                matches=[x for x in json.loads(command([str(CTL),'pod','list'],label='creation_reconciliation')) if x.get('name')==name]
                if len(matches)!=1:raise ValueError('Unknown creation outcome; no repeated create')
                pod=matches[0]['id']
            if not pod:raise ValueError('Actual Pod missing')
            folder=ROOT/'results/cloud_trial'/(pod+'-tam')/'session_1';folder.mkdir(parents=True,exist_ok=False)
            root='/workspace/BehaviorICL-TAM/'+launch_id
            auth.update(pod_id=pod,protocol_hash=receipt['protocol_hash'],operation_hash=receipt['operation_hash'],sources=receipt['members'])
            ops.atomic(folder/'runtime_accounting.json',{'pod_id':pod,'launch_id':launch_id,'start_epoch':started,
                'start_request_utc':deployment['utc'],'cost_per_hour':gpu['securePricePerHr'],'new_GPU_segment_started':True,
                'estimate_not_invoice':True,'storage_included':False})
            deployment.update(pod_id=pod,backup_dir=str(folder),remote_root=root,watch=owner);ops.atomic(WORK/'deployment_receipt.json',deployment)
            deadline=time.monotonic()+300
            while True:
                live=api(pod)
                if live.get('runtimeStatus')=='running' and live.get('ssh'):break
                if time.monotonic()>deadline:raise ValueError('New capacity running SSH deadline')
                time.sleep(5)
            actual=remote(pod,ops.NATIVE_REMOTE,'standalone_actual_native',timeout=40)
            ops.assess_native(actual,pod);ops.atomic(folder/'actual_native.json',actual)
            cost=live.get('costPerHr',gpu['securePricePerHr']);fee=ops.read(folder/'runtime_accounting.json');fee['cost_per_hour']=cost;ops.atomic(folder/'runtime_accounting.json',fee)
            upload=upload_code(root,auth,receipt)
            accepted=remote(pod,upload,'small_fixed_image_package',timeout=90);ops.atomic(folder/'input_package_acceptance.json',accepted)
            backup(pod,root,auth,folder)
            body='import subprocess,sys,os\nlog=(root/"results/run.log").open("ab",buffering=0)\nworker=subprocess.Popen([sys.executable,"-u",str(root/"scripts/run_query_tam_pilot_v1.py")],cwd=root,stdout=log,stderr=log,start_new_session=True)\ninfo=c.process(worker.pid)\nc.atomic(root/"launch.json",info)\nprint(json.dumps(info),flush=True)\n'
            actual=remote(pod,code(root,auth,body),'once_exact_TAM_worker',timeout=30);ops.atomic(folder/'launch_receipt.json',actual)
            deployment.update(phase='cloud_worker_running_compatibility_pending',auth=auth,worker=actual);ops.atomic(WORK/'deployment_receipt.json',deployment)
        last_backup=last_commit=time.monotonic();previous_count=0;failures=0;last_stage=None
        while True:
            account((pod,))
            actual=remote(pod,code(root,auth,'print(json.dumps(c.state(root,auth)),flush=True)'), 'actual_worker_and_science_state',timeout=40)
            ops.atomic(folder/'actual_live.json',actual)
            progress=actual.get('progress') or {};count=progress.get('committed_runs',0)
            if count>previous_count:last_commit=time.monotonic();previous_count=count
            if progress.get('stage')!=last_stage or count:
                if progress.get('stage')!=last_stage:print(json.dumps({'actual_TAM_stage':progress.get('stage'),'committed_runs':count}),flush=True)
                last_stage=progress.get('stage')
            if actual['worker'] is None:
                if actual['native']['GPU_compute_pids']:raise ValueError('Worker ended with unexpected GPU compute')
                backup(pod,root,auth,folder,True)
                stop(pod,auth,folder,'pilot_complete' if actual['complete'] and not actual['failed'] else 'worker_failure')
                return
            if time.monotonic()-last_commit>10800:raise ValueError('180 minutes without a real committed TAM run')
            if time.monotonic()-last_backup>=120:
                try:backup(pod,root,auth,folder);last_backup=time.monotonic();failures=0
                except BaseException as error:
                    failures+=1;ops.atomic(folder/('backup_failure_'+str(time.time_ns())+'.json'),{'utc':ops.now(),'class':type(error).__name__,'message':str(error),'consecutive':failures})
                    if failures>=5:raise ValueError('Five consecutive real backup failures')
                    time.sleep(5)
            time.sleep(15)
    except BaseException as error:
        ops.atomic(WORK/'controller_failure.json',{'utc':ops.now(),'error_class':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        if pod and folder:
            if root:
                try:
                    body='import os,signal,time\nc.atomic(root/"results/stop_request.json",{"utc":c.now()})\nactual=c.state(root,auth)\nif actual["worker"]:os.killpg(actual["worker"]["pid"],signal.SIGTERM)\ntime.sleep(3)\nprint(json.dumps(c.state(root,auth)))\n'
                    remote(pod,code(root,auth,body),'exact_failure_worker_quiescence',timeout=30)
                    backup(pod,root,auth,folder,True)
                except BaseException as forensic:
                    ops.atomic(folder/'FULL_failure.json',{'utc':ops.now(),'class':type(forensic).__name__,'message':str(forensic),'closed_FULL_asserted':False})
            stop(pod,auth,folder,'controller_failure')
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('package','audit','launch'),default='audit');parser.add_argument('--launch-id');args=parser.parse_args()
    WORK.mkdir(parents=True,exist_ok=True)
    if args.phase=='package':print(json.dumps(package()))
    elif args.phase=='audit':print(json.dumps({'CPU_package_audit':True,'operation_hash':audit()['operation_hash'],'GPU_requested':False}))
    else:
        with lock(WORK/'controller.lock'):launch(args.launch_id)
