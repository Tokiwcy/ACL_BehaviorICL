"""CPU-only dispatch decision for the latest first-freed-card instruction.

No cloud, model, startup or overwrite. Successful full takeback/exact stop and
independent complete-test audit remain prerequisites. Frozen old serial order
and entries are untouched; a new audited role-aware entry is still required.
"""
from datetime import datetime
from pathlib import Path

import behavior_paper_four_scope_policy as scope

ORDER = ('internvl', 'qwenvision')


def authority(request):
    scope.authority(request)
    if (request.get('first_free_card_dispatch_authorized') is not True
        or request.get('first_free_card_dispatch_roles') != list(ORDER)
        or request.get('first_free_card_dispatch_human_quote') !=
            '大概还要多久。固定 demo 标签干预新版DTD＋CUB 完整 test第二 VLMDTD＋CUB 完整 test  这俩也要接上，4090/pro 4500哪个跑完就接哪个后面'):
        raise ValueError('Latest direct first-freed-card order is not recorded')


def ended_success(row, takeback, stop, account, audit):
    role = row['role']
    if role not in ('task_prompt', 'full_labels'):
        raise ValueError('Only a completed current independent ablation supplies the first free card')
    target = scope.SCOPES[role]
    expected = {'task_prompt': {'queries': 7674, 'predictions': 15348, 'fresh_source_rows': 34856},
                'full_labels': {'queries': 7674, 'records': 30696, 'generated': 30526, 'explicit_missing': 170}}[role]
    if (takeback.get('stage') != 'all_complete_verified'
        or takeback.get('all_new_scientific_arrays_retrieved' if role == 'task_prompt'
                        else 'all_new_arrays_and_checkpoints_retrieved') is not True
        or not takeback.get('archive_sha256') or not takeback.get('members')
        or stop.get('pod_id') != row['pod_id'] or stop.get('reason') != 'all_complete_verified'
        or stop.get('disks_preserved') is not True or account.get('pod_id') != row['pod_id']
        or account.get('launch_id') != row['launch_id'] or account.get('phase') != 'stopped'
        or account.get('confirmed_utc') != stop.get('confirmed_utc')
        or audit.get('status') not in ('complete_verified', 'all_complete_verified')
        or audit.get('complete_official_test') is not True
        or any(audit.get(key) != value for key, value in expected.items())
        or audit.get('session') != row['backup_dir']):
        raise ValueError('Actual complete official-test takeback/stop/independent audit required; no sample or smoke substitution')
    ended = datetime.fromisoformat(stop['confirmed_utc'])
    took = datetime.fromisoformat(takeback['utc'])
    if ended.tzinfo is None or took.tzinfo is None or took > ended:
        raise ValueError('Full takeback precedes earliest confirmed stop; no paid analysis wait')
    return {'previous_role': role, 'pod_id': row['pod_id'], 'launch_id': row['launch_id'],
            'confirmed_utc': ended.isoformat(), 'previous_scientific_root': target['science']}


def select_next(registry, completed_slots, completed_roles, root):
    rows = scope.validate_registry(registry, root)
    active = {row['role'] for row in rows}
    done = set(completed_roles)
    if len(rows) >= scope.MAX_CARDS:
        return {'action': 'wait_existing_jobs_no_third_GPU', 'cloud_execution': False}
    next_role = next((role for role in ORDER if role not in active and role not in done), None)
    if not next_role:
        return {'action': 'no_queued_role', 'cloud_execution': False}
    if not completed_slots:
        return {'action': 'wait_successful_full_takeback_stop_and_audit', 'role': next_role, 'cloud_execution': False}
    candidates = sorted(completed_slots, key=lambda row: datetime.fromisoformat(row['confirmed_utc']))
    for slot in candidates:
        if (not slot.get('pod_id') or slot['pod_id'] in {row['pod_id'] for row in rows}
            or slot.get('previous_role') not in ('task_prompt', 'full_labels')):
            raise ValueError('Do not reuse a running/unbound/failed scientific slot')
    return {'action': 'prepare_new_independent_role_aware_entry', 'role': next_role,
            'preferred_existing_pod_id': candidates[0]['pod_id'],
            'previous_exact_launch_id': candidates[0]['launch_id'],
            'current_capacity_must_be_freshly_verified': True,
            'capacity_switch_autonomous_within_max_two': True,
            'old_serial_or_consumed_entry_invoked': False, 'cloud_execution': False,
            'required': 'actual native/new backup/exact isolated runtime/full input SHA/native legal smoke and frozen audited new entry'}
