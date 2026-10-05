"""Independent TAM operational helpers; standard library only, no model import."""
import hashlib,io,json,os,stat,subprocess,tarfile,time,uuid
from contextlib import redirect_stdout
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath

def now():return datetime.now(timezone.utc).isoformat()
def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):value.update(chunk)
    return value.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
def relative(name):
    p=PurePosixPath(name)
    if not name or '\\' in name or ':' in name or p.is_absolute() or any(x in ('','..','.') for x in name.split('/')):raise ValueError('Unsafe member path')
    return p
def actual_native():
    out=io.StringIO()
    with redirect_stdout(out):exec(NATIVE_REMOTE,{})
    return json.loads(out.getvalue())
def process(pid):
    proc=Path('/proc')/str(pid);tail=(proc/'stat').read_text().rsplit(')',1)[1].split()
    return {'pid':pid,'ppid':int(tail[1]),'state':tail[0],'start_ticks':int(tail[19]),
            'cwd':str((proc/'cwd').resolve()),'args':[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]}
def state(root,auth):
    root=Path(root);native=actual_native();assess_native(native,auth['pod_id'],empty_gpu=False)
    launch=read(root/'launch.json') if (root/'launch.json').exists() else None
    worker=None
    if launch:
        try:worker=process(launch['pid'])
        except FileNotFoundError:pass
        if worker and worker['state']=='Z':worker=None
        if worker and (worker['start_ticks']!=launch['start_ticks'] or worker['cwd']!=str(root) or str(root/'scripts/run_query_tam_pilot_v1.py') not in worker['args']):raise ValueError('Exact worker PID/start/cwd/command differs')
    if any(pid!=(worker or {}).get('pid') for pid in native['GPU_compute_pids']):raise ValueError('Unbound or multiple GPU computations')
    result={'utc':now(),'pod_id':auth['pod_id'],'launch_id':auth['launch_id'],'native':native,'worker':worker,
            'progress':read(root/'results/progress.json') if (root/'results/progress.json').exists() else None,
            'complete':(root/'results/complete.json').exists(),'failed':(root/'results/failure.json').exists()}
    return result
def archive(root,auth,full=False):
    root=Path(root);actual=state(root,auth)
    if full and (actual['worker'] or actual['native']['GPU_compute_pids']):raise ValueError('FULL needs quiescent exact worker and empty GPU')
    inventory={}
    for prefix in ('scripts','tam','inputs','results'):
        base=root/prefix
        if not base.exists():continue
        for file in base.rglob('*'):
            if file.is_symlink():raise ValueError('Archive symlink refused')
            if not file.is_file() or file.name.endswith('.tmp') or '__pycache__' in file.parts:continue
            if prefix=='results' and 'runs' in file.parts and not (file.parent/'record.json').exists():continue
            name=file.relative_to(root).as_posix();relative(name);inventory[name]=file
    for name in ('auth.json','protocol.json','launch.json','upload.json'):
        file=root/name
        if file.exists():inventory[name]=file
    destination=root/'archives'/('backup_'+uuid.uuid4().hex+'.tar.gz');destination.parent.mkdir(exist_ok=True)
    manifest={}
    with tarfile.open(destination,'w:gz') as tf:
        for name,file in sorted(inventory.items()):
            data=file.read_bytes();manifest[name]=hashlib.sha256(data).hexdigest()
            member=tarfile.TarInfo(name);member.size=len(data);member.mode=0o600
            tf.addfile(member,io.BytesIO(data))
    return {'utc':now(),'pod_id':auth['pod_id'],'launch_id':auth['launch_id'],'full':full,
            'remote_actual':actual,'path':str(destination),'archive_SHA':digest(destination),
            'bytes':destination.stat().st_size,'members':manifest,
            'model_weights_excluded_verified_inventory_in_results':True}
def unpack(path,destination,expected):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    if digest(path)!=expected['archive_SHA']:raise ValueError('Closed archive SHA differs')
    seen={}
    with tarfile.open(path,'r:gz') as tf:
        for member in tf:
            relative(member.name)
            if not member.isfile() or member.name in seen or member.name not in expected['members']:raise ValueError('Unexpected/unsafe archive member')
            data=tf.extractfile(member).read();actual=hashlib.sha256(data).hexdigest()
            if actual!=expected['members'][member.name]:raise ValueError('Member SHA differs')
            target=destination/member.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            if digest(target)!=actual:raise ValueError('Extracted SHA differs')
            seen[member.name]=actual
    if seen!=expected['members']:raise ValueError('Closed archive members missing')
    return seen

# Original standalone native probe and native guards are appended by the
# CPU packaging step from their original AST; no expected fields are injected.

NATIVE_REMOTE = '\nimport json,os,pathlib,re,stat,subprocess\ndef one(values):\n    if len(values)>1: raise ValueError(\'Duplicate native Pod field\')\n    if values and not re.fullmatch(r\'[A-Za-z0-9_-]{1,80}\',values[0]): raise ValueError(\'Malformed Pod field\')\n    return values[0] if values else None\nproc=one([v.decode(\'ascii\') for item in pathlib.Path(\'/proc/1/environ\').read_bytes().split(b\'\\0\')\n          for k,sep,v in [item.partition(b\'=\')] if k==b\'RUNPOD_POD_ID\' and sep])\nexport=pathlib.Path(\'/etc/rp_environment\'); exists=export.exists() or export.is_symlink(); trusted=None; native=None\nif exists:\n    info=export.lstat();trusted=stat.S_ISREG(info.st_mode) and info.st_uid==0 and not info.st_mode&0o022\n    if not trusted: raise ValueError(\'Untrusted native export\')\n    values=[]\n    for line in export.read_text().splitlines():\n        if line.startswith(\'export RUNPOD_POD_ID=\'):\n            match=re.fullmatch(r\'export RUNPOD_POD_ID=["\\x27]?([A-Za-z0-9_-]{1,80})["\\x27]?\',line)\n            if not match:raise ValueError(\'Malformed native export\')\n            values.append(match.group(1))\n    native=one(values)\npids=subprocess.check_output([\'nvidia-smi\',\'--query-compute-apps=pid\',\'--format=csv,noheader,nounits\'],text=True)\nprint(json.dumps({\'proc1_pod_id\':proc,\'export_file_pod_id\':native,\'native_export_exists\':exists,\n \'export_root_owned_nonwritable_regular\':trusted,\'ssh_process_pod_id\':os.environ.get(\'RUNPOD_POD_ID\'),\n \'GPU_compute_pids\':sorted({int(v.strip()) for v in pids.splitlines() if v.strip().isdigit()}),\n \'devices\':subprocess.check_output([\'nvidia-smi\',\'--query-gpu=name\',\'--format=csv,noheader\'],text=True).strip().splitlines()}))\n'

def assess_native(fields, expected, *, empty_gpu=True):
    values = [fields.get(k) for k in ('proc1_pod_id', 'export_file_pod_id') if fields.get(k) is not None]
    if not values or any((v != expected for v in values)) or fields.get('ssh_process_pod_id') not in (None, expected) or (fields.get('native_export_exists') and (not fields.get('export_root_owned_nonwritable_regular'))):
        raise ValueError('Actual native Pod proof absent/conflicting/untrusted')
    if empty_gpu and fields.get('GPU_compute_pids'):
        raise ValueError('Existing computation; never parallel GPU')
    device = fields.get('devices', [])
    if len(device) != 1 or not any((s in device[0] for s in ('4090', '5090', 'RTX PRO 4500'))):
        raise ValueError('Wrong actual authorized single GPU')
    return {'actual_api_pod_id': expected, 'proc1_pod_id': fields.get('proc1_pod_id'), 'export_file_pod_id': fields.get('export_file_pod_id'), 'ssh_process_pod_id': fields.get('ssh_process_pod_id'), 'export_root_owned_nonwritable_regular': fields.get('export_root_owned_nonwritable_regular'), 'device': device[0], 'utc': now(), 'expected_Pod_field_injected': False}
