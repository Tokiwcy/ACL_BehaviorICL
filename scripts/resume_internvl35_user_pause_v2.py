"""Explicit same-boot pause resume; frozen science and guarded ICL plan."""
import argparse
import csv
import json
import subprocess
from pathlib import Path

import internvl35_joint_config_v2 as config
import run_internvl35_joint_pipeline_v2 as base
import resume_internvl35_checked_v1 as checked


def desktop_only(allowed):
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name',
                             '--format=csv,noheader'], capture_output=True, text=True, check=True)
    code = "@(Get-CimInstance Win32_Process | Select-Object ProcessId,Name,ExecutablePath,@{n='CreatedUtc';e={$_.CreationDate.ToUniversalTime().ToString('o')}}) | ConvertTo-Json -Compress"
    result2 = subprocess.run(['powershell', '-NoProfile', '-Command', code],
                             capture_output=True, text=True, check=True)
    processes = json.loads(result2.stdout)
    if isinstance(processes, dict):
        processes = [processes]
    by_pid = {p['ProcessId']: p for p in processes}
    observed = []
    for row in csv.reader(result.stdout.splitlines()):
        p = by_pid.get(int(row[0].strip()))
        if p is None or not any(all(p.get(k) == a.get(k) for k in
                                    ('ProcessId', 'Name', 'CreatedUtc')) for a in allowed):
            raise RuntimeError('Unattested GPU context; refuse concurrent worker')
        if p['Name'].lower() != 'dwm.exe':
            raise RuntimeError('Only explicitly attested Windows DWM is allowed')
        observed.append(p)
    return observed


def validate(auth_path):
    auth = json.loads(auth_path.read_text(encoding='utf-8'))
    if auth.get('authorization') != 'Human explicitly requested continue after 2026-10-09 22:44 pause':
        raise RuntimeError('Missing explicit user authorization')
    status_path = base.LOG_ROOT / 'status.json'
    status = json.loads(status_path.read_text(encoding='utf-8'))
    if status['state'] != 'paused' or base.boot_identity() != auth['current_boot']:
        raise RuntimeError('Not paused or boot changed since authorization checks')
    if config.digest(status_path) != auth['status_sha256']:
        raise RuntimeError('Paused status changed')
    pause_path = Path(auth['pause_receipt'])
    if config.digest(pause_path) != auth['pause_receipt_sha256']:
        raise RuntimeError('Pause identity changed')
    pause = json.loads(pause_path.read_text(encoding='utf-8'))
    from datetime import datetime
    delta = abs((datetime.fromisoformat(status['boot_identity'].replace('Z', '+00:00')) -
                 datetime.fromisoformat(auth['current_boot'].replace('Z', '+00:00'))).total_seconds())
    if delta > 10 or auth['restart_events_since_pause']:
        raise RuntimeError('New boot or restart incident; explicit diagnosis required')
    code = "$ErrorActionPreference='Stop'; try {$e=@(Get-WinEvent -FilterHashtable @{LogName='System';Id=12,41,6008,1074;StartTime=[datetime]'" + pause['paused_utc'] + "'} -ErrorAction Stop)} catch {if($_.FullyQualifiedErrorId -notlike 'NoMatchingEventsFound*'){throw}; $e=@()}; if($e.Count){throw 'Restart event since pause'}"
    subprocess.run(['pwsh', '-NoProfile', '-Command', code], check=True)
    for name, expected in auth['operational_code_sha256'].items():
        if config.digest(config.ROOT / 'scripts' / name) != expected:
            raise RuntimeError('Operational code identity changed')
    dataset, method = checked.validate(status, auth['expected_rows'])
    for name, item in pause['prediction_files'].items():
        p = config.OUTPUT / dataset / 'predictions' / f'{name}.jsonl'
        if config.digest(p) != item['sha256']:
            raise RuntimeError('Prediction file differs from pause receipt')
    model_files = json.loads((config.legacy.ANALYSIS / 'model_files.json').read_text(encoding='utf-8'))
    for name, item in model_files.items():
        st = (config.legacy.SNAPSHOT / name).stat()
        if (st.st_size, st.st_mtime_ns) != (item['size'], item['mtime_ns']):
            raise RuntimeError('Model file changed')
    planned = list(checked.guarded_stages())
    done = [r for r in status['stages'] if r['state'] == 'complete']
    for record, (d, stage, argv) in zip(done, planned):
        if (record['dataset'], record['stage'], record['command'], record['return_code']) != (d, stage, argv, 0):
            raise RuntimeError('Completed guarded stage plan changed')
    if (dataset, 'icl_' + method) != planned[len(done)][:2]:
        raise RuntimeError('Current stage differs from next guarded plan entry')
    desktop = desktop_only(auth['allowed_desktop'])
    return auth, status, dataset, method, desktop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--authorization', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    auth, status, dataset, method, desktop = validate(args.authorization)
    if args.check:
        print(json.dumps({'state': 'CPU_checked', 'dataset': dataset, 'method': method,
                          'rows': auth['expected_rows'], 'desktop_contexts': desktop}))
        return
    with base.lock_transition(), base.lock_gpu():
        auth, status, dataset, method, desktop = validate(args.authorization)
        key = config.digest(args.authorization)[:16]
        receipt_path = config.ANALYSIS / f'user_resume_{key}.json'
        if receipt_path.exists():
            raise RuntimeError('Consumed authorization; never automatically relaunch')
        path = base.LOG_ROOT / 'status.json'
        (config.ANALYSIS / f'status.pre_user_resume_{key}.json').write_bytes(path.read_bytes())
        receipt = {'state': 'running', 'started_utc': config.now(),
                   'authorization': auth['authorization'], 'authorization_sha256': config.digest(args.authorization),
                   'code_sha256': config.digest(Path(__file__)), 'protocol_sha256': config.digest(config.PROTOCOL_PATH),
                   'boot_identity': auth['current_boot'], 'old_boot_identity': status['boot_identity'],
                   'saved_predictions': auth['expected_rows'], 'desktop_contexts': desktop,
                   'same_boot_time_field_drift_seconds': auth['boot_time_field_drift_seconds'],
                   'note': 'No new reboot; preserve frozen science and passed gate; no claim Hypervisor root cause fixed'}
        config.write_json(receipt_path, receipt)
        flag = base.LOG_ROOT / 'STOP_REQUESTED'
        if flag.exists():
            flag.rename(flag.with_name(f'STOP_REQUESTED.user_resume_{key}'))
        status.update(boot_identity=auth['current_boot'], operation_receipt=str(receipt_path),
                      resumed_utc=config.now(), resume_reason=auth['authorization'])
        config.write_json(path, status)
        tp = config.ANALYSIS / 'transition.json'
        transition = json.loads(tp.read_text(encoding='utf-8'))
        transition.update(state='running_joint_pipeline', boot_identity=auth['current_boot'],
                          operation_receipt=str(receipt_path), resumed_utc=config.now())
        config.write_json(tp, transition)
        base.stages = checked.guarded_stages
        try:
            base.pipeline(True)
            final = json.loads(path.read_text(encoding='utf-8'))
            receipt.update(state=final['state'], finished_utc=config.now())
            config.write_json(receipt_path, receipt)
        except BaseException as error:
            receipt.update(state='failed', error=repr(error), finished_utc=config.now())
            config.write_json(receipt_path, receipt)
            raise


if __name__ == '__main__':
    main()
