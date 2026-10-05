"""Read-only actual API/SSH/CIM check for independent TAM, never starts a job."""
import json,os,time
from pathlib import Path
import start_query_tam_cloud_v2 as c

def inspect():
    d=c.ops.read(c.WORK/'deployment_receipt.json');pod=d['pod_id'];launch=d['launch_id']
    actual=c.api(pod);folder=Path(d['backup_dir'])
    value={'utc':c.ops.now(),'pod_id':pod,'launch_id':launch,'phase':d['phase'],
        'actual_API':{k:actual.get(k) for k in c.FIELDS},'expected_fields_injected':False}
    if actual.get('desiredStatus')!='RUNNING':
        if (actual.get('desiredStatus'),actual.get('runtimeStatus'))!=('EXITED','stopped'):raise ValueError('Unexpected actual capacity state')
        c.account();value['stop']=c.ops.read(folder/'stop_confirmation.json')
        value['FULL_verified']=(folder/'full_takeback_receipt.json').exists()
    else:
        c.account((pod,))
        text=c.command(['powershell','-NoProfile','-Command',
            "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -like '*start_query_tam_cloud_v2.py*' -and $_.CommandLine -like '*"+launch+"*' } | Select-Object ProcessId,ParentProcessId,CommandLine,@{Name='CreationUTC';Expression={$_.CreationDate.ToUniversalTime().ToString('o')}} | ConvertTo-Json -Compress"],label='read_only_exact_CIM',timeout=20)
        rows=json.loads(text) if text.strip() else []
        if isinstance(rows,dict):rows=[rows]
        if len(rows)!=1 or rows[0]['ProcessId']!=d['watch']['ProcessId'] or rows[0]['CreationUTC']!=d['watch']['CreationUTC']:raise ValueError('Exact original unique TAM watchdog differs')
        value['exact_CIM_watch']=rows
        if actual.get('runtimeStatus')=='running' and actual.get('ssh'):
            auth=d.get('auth')
            if auth:
                value['remote_actual']=c.remote(pod,c.code(d['remote_root'],auth,'print(json.dumps(c.state(root,auth)),flush=True)'), 'read_only_actual_TAM_state',timeout=40)
                h=c.ops.read(folder/'actual_backup_heartbeat.json');value['backup_age_seconds']=time.time()-h['utc']
                if not 0<=value['backup_age_seconds']<=300:raise ValueError('Actual successful SHA backup stale')
            else:
                value['actual_native']=c.remote(pod,c.ops.NATIVE_REMOTE,'startup_actual_native_only',timeout=40)
                c.ops.assess_native(value['actual_native'],pod,empty_gpu=False)
                value['startup_not_model_compatibility']=True
    c.ops.atomic(c.WORK/('independent_actual_live_'+str(time.time_ns())+'.json'),value)
    return value

if __name__=='__main__':
    v=inspect();print(json.dumps({'utc':v['utc'],'pod_id':v['pod_id'],'API':v['actual_API'],
        'remote_actual':v.get('remote_actual'),'startup_not_model_compatibility':v.get('startup_not_model_compatibility'),
        'backup_age_seconds':v.get('backup_age_seconds'),'FULL_verified':v.get('FULL_verified')}))
