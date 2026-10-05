"""Fresh host operation around unchanged one-control QwenVision science.

No old root-repair/start entry is reused. Scientific plan/runner stays byte
identical. New wrapper binds exact native API proof and fresh successful backup.
"""
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import os
import sys
import subprocess

from behavior_qwenvision_two_policy_v1 import *

OPERATION = 'behavior_qwenvision_supcon_dtd_cub_cloud_v1'
BINDING = ROOT / 'analysis/qwenvision_two_host_v1/implementation_lock.json'


def validate_binding(plan):
    binding = read(BINDING)
    if (binding.get('implementation_hash') != canonical({k: v for k, v in binding.items() if k != 'implementation_hash'})
            or binding.get('plan_hash') != plan['plan_hash'] or binding.get('scientific_root') != SCIENCE
            or binding.get('operational_root') != OPERATION):
        raise ValueError('Wrong independent host wrapper binding')
    for rel, sha in binding['code_sha256'].items():
        if digest(ROOT / rel) != sha: raise ValueError('Frozen wrapper/scientific code changed: ' + rel)
    return binding


def validate_cache(plan, local=True):
    import numpy as np
    for dataset, item in plan['complete_cache_reuse'].items():
        folder = ROOT / item['local_folder' if local else 'cloud_folder']
        expected = {'qwenvision.npy': item['sha256'], 'verification.json': item['verification_SHA'], 'qwen_capture.jsonl': item['log_SHA']}
        for name, sha in expected.items():
            if digest(folder / name) != sha: raise ValueError('Complete source-cache bytes/provenance changed: ' + name)
        array = np.load(folder / 'qwenvision.npy', mmap_mode='r')
        if list(array.shape) != item['shape'] or str(array.dtype) != item['dtype'] or not np.isfinite(array).all():
            raise ValueError('Only complete byte/shape/finite-verified original cache is reusable')
        rows = plan['datasets'][dataset]['manifest']['bank'] + plan['datasets'][dataset]['manifest']['query']
        records = []
        with (folder / 'qwen_capture.jsonl').open(encoding='utf-8') as handle:
            for line in handle:
                if not line.endswith('\n'): raise ValueError('Truncated original source evidence')
                import json
                records.append(json.loads(line))
        if len(records) != len(rows) or any((v.get('sample_id'), v.get('image_sha256'), v.get('plan_hash')) !=
            (rows[i]['sample_id'], rows[i]['image_sha256'], item['origin_plan_hash']) for i, v in enumerate(records)):
            raise ValueError('Original cache ordered-image/source identity differs')
    return True


def validate_backup(auth, ready, heartbeat, at=None):
    for record in (ready, heartbeat):
        if (record.get('status') != 'actual_backup_succeeded' or len(record.get('archive_sha256', '')) != 64
                or any(not auth.get(k) or record.get(k) != auth[k] for k in ('experiment', 'plan_hash', 'implementation_hash', 'pod_id', 'launch_id'))):
            raise ValueError('Missing exact new independent successful backup')
    age = ((at or datetime.now(timezone.utc))-datetime.fromisoformat(heartbeat['utc'])).total_seconds()
    if not 0 <= age <= 300: raise ValueError('Stale/future backup heartbeat')


def gpu_guard(plan):
    if sys.platform != 'linux' or str(ROOT) != '/workspace/BehaviorICL' or Path.cwd() != ROOT:
        raise ValueError('Cloud exact cwd only; local GPU forbidden')
    validate_plan(plan)
    binding = validate_binding(plan)
    auth = read(ROOT / 'results' / OPERATION / 'execution_authorization.json')
    if (auth.get('experiment') != SCIENCE or auth.get('plan_hash') != plan['plan_hash']
            or auth.get('implementation_hash') != binding['implementation_hash']
            or auth.get('actual_api_pod_id') != auth.get('pod_id')):
        raise ValueError('Wrong independent current-host scientific authorization')
    proof = ROOT / 'results' / OPERATION / 'native_identity_receipt.json'
    if digest(proof) != auth.get('native_identity_receipt_sha256') or read(proof).get('pod_id') != auth['pod_id']:
        raise ValueError('Native proof is not bound to fresh API/launch')
    native_identity(auth['pod_id'])
    validate_backup(auth, read(ROOT / 'results' / OPERATION / 'backup_ready.json'),
                    read(ROOT / 'results' / OPERATION / 'backup_heartbeat.json'))
    actual = {name: importlib.metadata.version(name) for name in binding['environment']}
    if actual != binding['environment']: raise ValueError('Frozen original Qwen pinned runtime differs')
    inventory = binding['model_inventory']
    cache = ROOT / '.hf_cache/models--Qwen--Qwen3-VL-4B-Instruct/snapshots' / inventory['revision']
    for name, sha in inventory['files_SHA256'].items():
        if digest(cache / name) != sha: raise ValueError('Full original model/processor bytes differ')
    active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    if any(v.strip().isdigit() and int(v.strip()) != os.getpid() for v in active.splitlines()):
        raise ValueError('Parallel GPU computation exists')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != auth['device']:
        raise ValueError('Wrong exact authorized single GPU')
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    return auth
