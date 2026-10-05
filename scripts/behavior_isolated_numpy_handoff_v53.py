"""CPU-only prerequisite at the exact original isolated-interpreter handoff.

Original bootstrap/config/runners stay byte-identical; no config override is
installed. The first os.execv call prepares exactly the pinned NumPy, then
execs the unchanged bootstrap in the isolated interpreter. The wrapper does
not survive exec, change PID/start ticks, or run weights/CUDA/model inference.
"""
import importlib.metadata,json,os,subprocess,sys,time
from pathlib import Path

VERSION_CODE="import importlib.metadata,json,sys;\ntry:v=importlib.metadata.version('numpy')\nexcept importlib.metadata.PackageNotFoundError:v=None\nprint(json.dumps({'numpy':v,'prefix':sys.prefix}))"

def ensure_numpy(python,version,constraint,guard,deadline_seconds=300):
    """Used directly in clean CPU integration tests as well as cloud handoff."""
    python=Path(python);constraint=Path(constraint)
    if not python.is_file() or not (python.parent.parent/'pyvenv.cfg').is_file() or constraint.is_symlink() or not constraint.is_file():raise ValueError('Exact isolated interpreter/constraint required')
    lines=constraint.read_text(encoding='utf-8').splitlines()
    if lines.count('numpy=='+version)!=1 or any(line.startswith('numpy') and line!='numpy=='+version for line in lines):raise ValueError('Exact original pinned NumPy constraint required')
    def current():
        guard();value=subprocess.run([str(python),'-I','-c',VERSION_CODE],capture_output=True,text=True,check=True,timeout=30)
        actual=json.loads(value.stdout)
        if Path(actual['prefix']).resolve()!=python.parent.parent.resolve():raise ValueError('Actual isolated interpreter prefix differs')
        return actual['numpy']
    before=current()
    if before is not None and before!=version:raise ValueError('Existing isolated NumPy differs; never overwrite')
    installed=False
    if before is None:
        guard();child=subprocess.Popen([str(python),'-m','pip','install','--no-deps','--only-binary=:all:','--constraint',str(constraint),'numpy=='+version],stdin=subprocess.DEVNULL)
        end=time.monotonic()+deadline_seconds
        try:
            while child.poll() is None:
                guard()
                if time.monotonic()>end:raise TimeoutError('Bounded CPU NumPy install')
                time.sleep(.5)
            if child.returncode:raise ValueError('Pinned NumPy prerequisite install failed')
        except BaseException:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
            raise
        installed=True
    after=current()
    if after!=version:raise ValueError('Exact pinned NumPy metadata not accepted')
    return {'numpy_before':before,'numpy_after':after,'new_CPU_prerequisite_installed':installed,'weights_or_CUDA_executed':False}

def install_handoff(c,auth):
    if sys.platform!='linux' or str(c.ROOT)!='/workspace/BehaviorICL' or Path.cwd()!=c.ROOT:raise ValueError('Actual cloud CPU bootstrap only')
    mode=auth['mode']
    if mode not in ('internvl_five','qwen_three_seed'):raise ValueError('Exact pending queue role only')
    c.native.p.native_identity(auth['pod_id']);cfg=c.config(mode)
    folder=c.ROOT/'results'/cfg['operation'];c.exact(c.p.read(folder/'execution_authorization.json'),auth)
    constraint=Path(os.environ['PIP_CONSTRAINT'])
    if constraint.is_symlink() or not constraint.resolve().is_relative_to(c.ROOT/'.tools') or c.p.digest(constraint)!=auth['runtime_constraints_sha256']:raise ValueError('Original authorized isolated constraints differ')
    expected=''.join(n+'=='+v+'\n' for n,v in sorted(cfg['environment'].items()))
    if constraint.read_text(encoding='utf-8')!=expected:raise ValueError('Original whole environment recipe differs')
    target=c.ROOT/cfg['venv']/'bin/python';original=os.execv;used=False
    def handoff(path,argv):
        nonlocal used
        if used or str(path)!=str(target) or argv!=[str(target),'-u',c.PARENT,'--mode',mode] or Path(sys.prefix).resolve()==target.parent.parent:raise ValueError('Only one exact original first isolated exec handoff')
        used=True
        def guard():
            c.native.p.native_identity(auth['pod_id']);c.backup_guard(cfg,auth)
            launch=c.p.read(folder/'launch.json');c.exact(launch,auth)
            if launch['pid']!=os.getpid():raise ValueError('Exact parent PID handoff only')
            parent,children,pids=c.process_state(cfg,launch)
            if children or pids:raise ValueError('No scientific computation before isolated NumPy prerequisite')
        result=ensure_numpy(target,cfg['environment']['numpy'],constraint,guard)
        c.p.immutable(folder/'isolated_numpy_before_exec.json',{'utc':c.p.now(),'pod_id':auth['pod_id'],'launch_id':auth['launch_id'],
            'pid':os.getpid(),'parent_start_ticks':launch_ticks(c,folder),'original_parent':c.PARENT,'original_parent_SHA':c.p.digest(c.ROOT/c.PARENT),'constraint_SHA':c.p.digest(constraint),'scientific_progress':False,**result})
        guard();return original(path,argv)
    os.execv=handoff

def launch_ticks(c,folder):return c.p.read(folder/'launch.json')['start_ticks']
