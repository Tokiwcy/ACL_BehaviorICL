"""Reusable explicit-user recovery with cache identity and GPU replay gates."""
import argparse
import json
from pathlib import Path
import sys
import internvl35_joint_config_v2 as config
import run_internvl35_joint_pipeline_v2 as base

def guarded_stages():
    for dataset,stage,argv in original_stages():
        if stage.startswith('icl_'):
            argv=list(argv);argv[2]=str(config.ROOT/'scripts/internvl35_icl_cache_guard_v1.py')
        yield dataset,stage,argv

original_stages=base.stages

def validate(status,expected):
    config.require_gate()
    current=status['current'];dataset=current['dataset'];method=current['stage'][4:]
    if not current['stage'].startswith('icl_'):
        raise ValueError('Checked recovery currently supports interrupted ICL')
    from run_internvl35_joint_methods_v2 import load_context
    folder,bank,queries,identity=load_context(dataset)
    data=(folder/'predictions'/f'{method}.jsonl').read_bytes()
    if not data.endswith(b'\n'):
        raise ValueError('Truncated final row must be preserved and diagnosed first')
    rows=[json.loads(line) for line in data.splitlines()]
    if len(rows)!=expected:
        raise ValueError('Expected committed cursor changed')
    selected=folder/'selections'/f'{method}.json'
    selection=json.loads(selected.read_text(encoding='utf-8'))
    pred_identity=json.loads((folder/'predictions'/f'{method}.identity.json').read_text(encoding='utf-8'))
    if (pred_identity['full_protocol_sha256']!=config.digest(config.PROTOCOL_PATH)
        or pred_identity['selections_sha256']!=config.digest(selected)
        or pred_identity['source_manifest_sha256']!=identity['manifest_sha256']):
        raise ValueError('Prediction identity differs')
    for i,row in enumerate(rows):
        q=queries[i];demos=[bank[j] for j in selection[q.sample_id]]
        if (row['query_id'],row['target'],row['demo_ids'],row['demo_labels'],row['correct'])!=(q.sample_id,q.label,[d.sample_id for d in demos],[d.label for d in demos],row['prediction']==q.label):
            raise ValueError('Saved prediction order/context differs')
    return dataset,method

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--incident',type=Path,required=True)
    parser.add_argument('--expected-rows',type=int,required=True)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    status_path=base.LOG_ROOT/'status.json'
    status=json.loads(status_path.read_text(encoding='utf-8'))
    dataset,method=validate(status,args.expected_rows)
    incident=json.loads(args.incident.read_text(encoding='utf-8-sig'))
    boot=base.boot_identity()
    if boot!=incident['boot_identity'] or 'Human explicitly requested' not in incident['restart_authorization']:
        raise RuntimeError('Recovery requires explicit human request and diagnosed current boot')
    if args.check:
        print(f'CPU verified {dataset}/{method}: {args.expected_rows} complete rows, same scientific identity')
        return
    with base.lock_transition(),base.lock_gpu():
        base.gpu_free()
        key=config.digest(args.incident)[:16]
        receipt_path=config.ANALYSIS/f'checked_resume_{key}.json'
        if receipt_path.exists():
            raise RuntimeError('Consumed recovery identity; never loop after failure')
        snapshot=config.ANALYSIS/f'status.pre_checked_resume_{key}.json'
        snapshot.write_bytes(status_path.read_bytes())
        receipt={'state':'replay','started_utc':config.now(),'authorization':incident['restart_authorization'],
                 'incident_sha256':config.digest(args.incident),'protocol_sha256':config.digest(config.PROTOCOL_PATH),
                 'code_sha256':config.digest(Path(__file__)),'guard_code_sha256':config.digest(config.ROOT/'scripts/internvl35_icl_cache_guard_v1.py'),
                 'replay_code_sha256':config.digest(config.ROOT/'scripts/check_internvl35_checked_restart_v1.py'),
                 'saved_predictions':args.expected_rows,'old_boot':status['boot_identity'],'boot_identity':boot,
                 'root_cause_limit':'Hypervisor fatal error confirmed; concrete driver/firmware cause unresolved; cache cleanup is operational mitigation, not proof of a fix'}
        config.write_json(receipt_path,receipt)
        try:
            report=config.ANALYSIS/f'checked_replay_{key}.json'
            base.execute(base.cmd('check_internvl35_checked_restart_v1.py','--dataset',dataset,'--method',method,'--rows',str(args.expected_rows),'--report',str(report)),
                         config.ANALYSIS/f'checked_replay_{key}.log')
            if json.loads(report.read_text(encoding='utf-8'))['state']!='complete':
                raise RuntimeError('GPU replay/next-input gate failed')
            base.gpu_free()
            status.update(state='paused',boot_identity=boot,operation_receipt=str(receipt_path))
            config.write_json(status_path,status)
            transition_path=config.ANALYSIS/'transition.json'
            transition=json.loads(transition_path.read_text(encoding='utf-8'))
            transition.update(state='running_joint_pipeline',boot_identity=boot,operation_receipt=str(receipt_path),checked_utc=config.now())
            config.write_json(transition_path,transition)
            receipt.update(state='running',replay_sha256=config.digest(report),checked_utc=config.now())
            config.write_json(receipt_path,receipt)
            base.stages=guarded_stages
            base.pipeline(True)
            receipt.update(state='complete',finished_utc=config.now());config.write_json(receipt_path,receipt)
        except BaseException as error:
            receipt.update(state='failed',error=repr(error),finished_utc=config.now());config.write_json(receipt_path,receipt)
            raise

if __name__=='__main__':
    main()
