"""Latest human queue authority and max-two dispatch, CPU only.

The frozen earlier ordering and scientific entries remain unchanged. A queued
scope is not executable science; each entry must separately bind its package.
"""
from datetime import datetime
import hashlib
import json

import behavior_paper_four_scope_policy as old
from behavior_paper_next_free_slot_policy import ended_success

QUEUE_HASH = '0e503c34fb99f240e2cb4f362acca82edf2ce0a6ea346149b2323ad9a575548d'
QUEUE_PATH = 'analysis/paper_supervision_queue_v2/queue.json'
ORDER = ('qwenvision', 'internvl_five', 'qwen_three_seed')
SCOPES = {**{k: v for k, v in old.SCOPES.items() if k != 'internvl'},
    'internvl_five': {'science': 'behavior_internvl_five_dataset_v2',
        'operation': 'behavior_internvl_five_dataset_cloud_v2',
        'datasets': ('dtd', 'aircraft', 'cub', 'pets', 'dogs'),
        'predictions': 93024, 'records': 93024},
    'qwen_three_seed': {'science': 'behavior_qwen_three_seed_v1',
        'operation': 'behavior_qwen_three_seed_cloud_v1',
        'datasets': ('dtd', 'aircraft', 'cub', 'pets', 'dogs'),
        'predictions': 279072, 'records': 279072}}


def authority(request, queue):
    old.legacy.authority(request)
    calculated = hashlib.sha256(json.dumps({k: v for k, v in queue.items() if k != 'queue_hash'},
        ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if (queue.get('queue_hash') != QUEUE_HASH or calculated != QUEUE_HASH
        or request.get('latest_supervision_queue') != QUEUE_PATH
        or request.get('latest_supervision_queue_hash') != QUEUE_HASH
        or request.get('latest_supervision_human_authorized') is not True
        or queue.get('human_authorized_execution_and_continuation') is not True
        or queue.get('maximum_cloud_GPUs') != 2 or queue.get('local_CUDA_execution') is not False
        or queue['scheduler']['remaining_priority'] != list(ORDER)
        or queue.get('first_VLM_methods_confirmed') is not True):
        raise ValueError('Latest direct human queue and order must be intact')
    return queue


def validate_registry(value, root):
    # Reuse every frozen path/accounting invariant, with independently defined
    # new five-dataset/multiseed scopes rather than aliasing the old InternVL.
    if (value.get('direct_human_parallel_limit') != 2
        or value.get('ended_GPU_baseline_USD') != old.ENDED_BASELINE):
        raise ValueError('Max-two authority and independent fee baseline required')
    rows = value.get('active_jobs')
    if not isinstance(rows, list) or len(rows) > 2:
        raise ValueError('At most two current jobs')
    seen = {k: set() for k in ('role', 'pod_id', 'launch_id', 'backup_dir', 'scientific_root', 'deployment_receipt')}
    for row in rows:
        role = row.get('role')
        if role not in SCOPES or row.get('scientific_root') != SCOPES[role]['science']:
            raise ValueError('Old two-dataset or unbound scientific scope refused')
        for k, identities in seen.items():
            if not isinstance(row.get(k), str) or not row[k] or row[k] in identities:
                raise ValueError('Independent role/Pod/launch/backup/operation/science required')
            identities.add(row[k])
        backup = old.relative_path(root, row['backup_dir'], 'results')
        trial = root/'results/cloud_trial'
        receipt = old.relative_path(root, row['deployment_receipt'], '.tools')
        if (not backup.is_relative_to(trial) or len(backup.relative_to(trial).parts) != 2
            or backup.name != 'session_1' or not backup.parent.name.startswith(row['pod_id']+'-paper-')
            or receipt.name != 'deployment_receipt.json' or row.get('single_GPU_count') != 1
            or row.get('actual_backup_required') is not True
            or row.get('scientific_compatibility_required') is not True):
            raise ValueError('Exact sole-compute backed-up scientific slot required')
    fees = value.get('fee_segment_receipts')
    if not isinstance(fees, list) or len(set(fees)) != len(fees):
        raise ValueError('Retain unique independent historical fee segments')
    for name in fees:
        path = old.relative_path(root, name, 'results')
        if not path.is_relative_to(root/'results/cloud_trial') or path.name != 'runtime_accounting.json':
            raise ValueError('Wrong segment receipt')
    if any(row['backup_dir']+'/runtime_accounting.json' not in fees for row in rows):
        raise ValueError('Every active start must have its own fee segment')
    return rows


def select_next(registry, completed_slots, completed_roles, root):
    rows = validate_registry(registry, root)
    if len(rows) == 2:
        return {'action': 'wait_existing_jobs_no_third_GPU'}
    active = {row['role'] for row in rows}
    role = next((r for r in ORDER if r not in active and r not in completed_roles), None)
    if role is None:
        return {'action': 'no_queued_role'}
    if not completed_slots:
        return {'action': 'wait_full_takeback_stop_and_audit', 'role': role}
    for slot in completed_slots:
        if (slot.get('previous_role') not in SCOPES or not slot.get('CPU_audit_confirmed_after_stop')
            or not slot.get('audit_SHA') or not slot.get('takeback_archive_SHA')
            or slot.get('pod_id') in {row['pod_id'] for row in rows}):
            raise ValueError('Only fully audited stopped independent slots may be reused')
    first = min(completed_slots, key=lambda r: datetime.fromisoformat(r['confirmed_utc']))
    return {'action': 'start_new_audited_entry', 'role': role,
            'preferred_existing_pod_id': first['pod_id'], 'previous_launch_id': first['launch_id']}


def chunk_manifest(manifest, expected_bytes):
    if not manifest or sum(item['bytes'] for item in manifest) != expected_bytes:
        raise ValueError('Complete input byte count required')
    for index, item in enumerate(manifest):
        if (item['name'] != str(index).zfill(6)+'.bin' or item['bytes'] <= 0
            or len(item['sha256']) != 64):
            raise ValueError('Exact contiguous input chunk identity required')
    return manifest
