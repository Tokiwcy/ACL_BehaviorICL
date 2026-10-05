"""CPU-only registry policy for the four already agreed complete-test scopes.

This does not launch anything or widen scientific protocols. Frozen initial
two-role/single-card entries are untouched; future independent entries must
bind this policy, their exact package/runtime/native proof and own watcher.
"""
from pathlib import Path, PurePosixPath

import behavior_paper_parallel_policy as legacy

MAX_CARDS = legacy.MAX_CARDS
ENDED_BASELINE = legacy.ENDED_BASELINE
SCOPES = {
    'task_prompt': {'science': 'behavior_task_conditioning_raw_v1',
                    'operation': 'behavior_task_conditioning_raw_cloud_v1',
                    'datasets': ('dtd', 'cub'), 'predictions': 15348, 'records': 15348},
    'full_labels': {'science': 'behavior_corrective_labels_full_dtd_cub_v1',
                    'operation': 'behavior_corrective_labels_full_dtd_cub_cloud_v1',
                    'datasets': ('dtd', 'cub'), 'predictions': 30526, 'records': 30696},
    'qwenvision': {'science': 'behavior_qwenvision_supcon_v1',
                   'operation': 'behavior_qwenvision_supcon_cloud_host_v1',
                   'datasets': ('dtd', 'aircraft', 'cub', 'pets', 'dogs'),
                   'predictions': 23256, 'records': 23256},
    'internvl': {'science': 'behavior_internvl_generalization_v1',
                 'operation': 'behavior_internvl_generalization_cloud_v1',
                 'datasets': ('dtd', 'cub'), 'predictions': 30696, 'records': 30696},
}


def authority(request):
    # Reuse, do not modify, the actual human max-two/no-local-CUDA authority.
    legacy.authority(request)
    allocation = request.get('first_four_confirmed_dataset_allocation')
    expected = {
        'task_prompt_vs_generic_prompt': list(SCOPES['task_prompt']['datasets']),
        'QwenVision_SupCon_fixed256': list(SCOPES['qwenvision']['datasets']),
        'fixed_demo_label_intervention_full_official_test_new_protocol': list(SCOPES['full_labels']['datasets']),
        'independent_second_VLM': list(SCOPES['internvl']['datasets']),
    }
    if allocation != expected or request.get('first_four_full_official_test_required') is not True:
        raise ValueError('The four settled complete official test scopes changed')


def relative_path(root, name, prefix):
    if not isinstance(name, str) or '\\' in name:
        raise ValueError('Portable exact relative registry path required')
    value = PurePosixPath(name)
    if (not value.parts or value.is_absolute() or ':' in name
        or any(part in ('..', '.') for part in value.parts) or value.parts[0] != prefix):
        raise ValueError('Unsafe/out-of-scope registry path')
    candidate = root.joinpath(*value.parts)
    if (candidate.is_symlink() or any(parent.is_symlink() for parent in candidate.parents
        if parent.is_relative_to(root)) or not candidate.resolve().is_relative_to(root/prefix)):
        raise ValueError('Linked/escaping registry path refused')
    return candidate.resolve()


def validate_registry(value, root):
    root = Path(root).resolve()
    if (value.get('direct_human_parallel_limit') != MAX_CARDS
        or value.get('ended_GPU_baseline_USD') != ENDED_BASELINE):
        raise ValueError('Max-two direct authority or independent accounting baseline changed')
    rows = value.get('active_jobs')
    if not isinstance(rows, list) or len(rows) > MAX_CARDS:
        raise ValueError('At most two cloud jobs, never a third card')
    distinct = {key: set() for key in ('role', 'pod_id', 'launch_id', 'backup_dir', 'scientific_root', 'deployment_receipt')}
    for row in rows:
        role = row.get('role')
        if role not in SCOPES or row.get('scientific_root') != SCOPES[role]['science']:
            raise ValueError('Only the exact four agreed role/scientific identities')
        for key, seen in distinct.items():
            if not isinstance(row.get(key), str) or not row[key] or row[key] in seen:
                raise ValueError('Independent role/Pod/launch/science/receipt/backup required')
            seen.add(row[key])
        backup = relative_path(root, row['backup_dir'], 'results')
        trial = root/'results/cloud_trial'
        if (not backup.is_relative_to(trial) or len(backup.relative_to(trial).parts) != 2
            or backup.name != 'session_1'
            or not backup.parent.name.startswith(row['pod_id']+'-paper-')):
            raise ValueError('Backup must be the exact per-Pod independent session')
        receipt = relative_path(root, row['deployment_receipt'], '.tools')
        if receipt.name != 'deployment_receipt.json':
            raise ValueError('Exact independent deployment receipt required')
        if (row.get('single_GPU_count') != 1 or row.get('actual_backup_required') is not True
            or row.get('scientific_compatibility_required') is not True):
            raise ValueError('Each Pod retains sole compute, real backup and compatibility guards')
    fees = value.get('fee_segment_receipts')
    if not isinstance(fees, list) or len(set(fees)) != len(fees):
        raise ValueError('Independent historical fee segments cannot be lost or duplicated')
    for name in fees:
        target = relative_path(root, name, 'results')
        if not target.is_relative_to(root/'results/cloud_trial') or target.name != 'runtime_accounting.json':
            raise ValueError('Wrong exact runtime accounting receipt')
    for row in rows:
        if row['backup_dir']+'/runtime_accounting.json' not in fees:
            raise ValueError('Every active start has an independent additive fee segment')
    return rows


def frozen_scope_check(role, cfg):
    scope = SCOPES.get(role)
    if (role == 'task_prompt' or scope is None or cfg.get('mode') != role
        or cfg.get('science') != scope['science'] or cfg.get('operation') != scope['operation']
        or tuple(cfg.get('manifest', {})) != scope['datasets']
        or cfg.get('predictions') != scope['predictions'] or cfg.get('records') != scope['records']):
        raise ValueError('Frozen followup scope/denominators cannot be widened or substituted')
    return scope


def cumulative(segments, at=None):
    return legacy.cumulative(segments, at)

