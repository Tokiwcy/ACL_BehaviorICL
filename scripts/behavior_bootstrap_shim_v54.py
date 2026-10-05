"""Bind the CPU NumPy prerequisite to the original parent without editing it.

The process command still contains the exact original parent and mode. The
handoff module intercepts only its first isolated os.execv; that exec removes
the wrapper. No configuration replacement enters the scientific interpreter.
"""
import hashlib

HELPER='scripts/behavior_isolated_numpy_handoff_v53.py'
SHIM_HELPER='scripts/behavior_bootstrap_shim_v54.py'
ROLES={
    'internvl_five':('behavior_internvl_five_native_cloud_v4','scripts/bootstrap_behavior_internvl_native_v4.py'),
    'qwen_three_seed':('behavior_qwen_three_seed_cloud_v1','scripts/bootstrap_behavior_qwen_three_seed_cloud_v1.py'),
}

def code(c,auth):
    mode=auth['mode']
    if mode not in ROLES or c.PARENT!=ROLES[mode][1]:raise ValueError('Exact original pending parent required')
    sha=auth.get('isolated_numpy_handoff_SHA')
    if not isinstance(sha,str) or len(sha)!=64:raise ValueError('Closed prerequisite source SHA required')
    module,parent=ROLES[mode]
    return ('import sys,runpy\n'
        'sys.path.insert(0,"/workspace/BehaviorICL/scripts")\n'
        'import '+module+' as c\n'
        'auth='+repr(auth)+'\n'
        'if sys.argv[1:]!='+repr([parent,'--mode',mode])+':raise ValueError("Original parent argv differs")\n'
        'if c.p.digest(c.ROOT/'+repr(HELPER)+')!=auth["isolated_numpy_handoff_SHA"]:raise ValueError("CPU prerequisite bytes differ")\n'
        'import behavior_isolated_numpy_handoff_v53 as handoff\n'
        'handoff.install_handoff(c,auth)\n'
        'sys.argv=sys.argv[1:]\n'
        'runpy.run_path(c.PARENT,run_name="__main__")\n')

def parent_stage(original,mode):
    """Only replace the original child's command; retain every stage guard."""
    if mode not in ROLES:raise ValueError('Pending role required')
    old="child=subprocess.Popen([sys.executable,'-u',c.PARENT,'--mode',"+repr(mode)+"],cwd=root,"
    if original.count(old)!=1:raise ValueError('One exact original child spawn required')
    before=("if c.p.digest(root/"+repr(SHIM_HELPER)+")!=auth['bootstrap_shim_SHA']:raise ValueError('Bound bootstrap shim changed')\n"
        "import behavior_bootstrap_shim_v54 as bootstrap_shim\n"
        "bootstrap_code=bootstrap_shim.code(c,auth)\n")
    marker="with (out/'sequence.log').open('a') as log:\n"
    if original.count(marker)!=1:raise ValueError('Original closed log stage required')
    changed=original.replace(marker,before+marker).replace(old,
        "child=subprocess.Popen([sys.executable,'-u','-c',bootstrap_code,c.PARENT,'--mode',"+repr(mode)+"],cwd=root,")
    compile(changed,'bound_original_parent_spawn','exec')
    return changed

def original_stage(changed,mode):
    """CPU proof: inverse substitution must recover the exact original bytes."""
    new="child=subprocess.Popen([sys.executable,'-u','-c',bootstrap_code,c.PARENT,'--mode',"+repr(mode)+"],cwd=root,"
    old="child=subprocess.Popen([sys.executable,'-u',c.PARENT,'--mode',"+repr(mode)+"],cwd=root,"
    before=("if c.p.digest(root/"+repr(SHIM_HELPER)+")!=auth['bootstrap_shim_SHA']:raise ValueError('Bound bootstrap shim changed')\n"
        "import behavior_bootstrap_shim_v54 as bootstrap_shim\n"
        "bootstrap_code=bootstrap_shim.code(c,auth)\n")
    if changed.count(new)!=1 or changed.count(before)!=1:raise ValueError('Exact independent substitution required')
    return changed.replace(new,old).replace(before,'')

def supplement(c):
    return {name:c.p.digest(c.ROOT/name) for name in (HELPER,SHIM_HELPER)}

def audited_failure(c):
    """Read-only recheck of the preceding closed, stopped bootstrap failure."""
    import analyze_behavior_internvl_bootstrap_failure_v52 as failure
    path=c.ROOT/'analysis/paper_internvl_bootstrap_failure_v52/audit.json'
    record=c.p.read(path)
    if ((record.get('pod_id'),record.get('launch_id'))!=(failure.POD,failure.LAUNCH)
        or any(record.get(key) is not True for key in ('closed_FULL_verified','fresh_exact_stopped_verified','independent_CPU_audit_passed'))
        or record.get('scientific_complete') is not False or record.get('source_SHA')!=c.p.digest(c.ROOT/'scripts/analyze_behavior_internvl_bootstrap_failure_v52.py')):
        raise ValueError('Exact independent NumPy bootstrap failure audit required')
    actual=failure.verify(failure.SESSION)
    if any(record.get(key)!=value for key,value in actual.items()):raise ValueError('Preceding closed FULL bytes changed')
    return record
