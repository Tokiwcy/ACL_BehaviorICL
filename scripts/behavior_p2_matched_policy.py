"""Independent P2 matched representation protocol; CPU integrity policy only."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from collect_behavior_priority_zero import digest, canonical, atomic

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('dtd', 'aircraft', 'cub', 'pets', 'dogs')
WIDTHS = (64, 128, 256, 512)
REPRESENTATIONS = ('clip', 'qwenvision', 'behavior')
METHODS = ('raw_clip', 'supcon_clip', 'raw_qwenvision', 'supcon_qwenvision', 'behavior_raw', 'behavior_proj')
SCIENCE = 'behavior_p2_matched_v1'
OPERATION = 'behavior_p2_matched_cloud_v1'
QWEN_REVISION = 'ebb281ec70b05090aa6165b016eac8ec08e71b17'
CLIP_REVISION = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
MODEL = 'Qwen/Qwen3-VL-4B-Instruct'
CLIP_MODEL = 'openai/clip-vit-base-patch32'
PLAN_PATH = ROOT / 'analysis/paper_p2_matched_plan_v1/plan_lock.json'
PROTOCOL = {
    'experiment': SCIENCE, 'datasets': list(DATASETS), 'seed': 73, 'shots': 4,
    'representations': list(REPRESENTATIONS), 'widths': list(WIDTHS),
    'Qwen_revision': QWEN_REVISION, 'CLIP_revision': CLIP_REVISION,
    'sources': 'new independent extraction for all three; never substitute or overwrite original sources',
    'clip': 'frozen CLIP-ViT-B/32 projected image feature, standard pinned CLIP processor, FP32; cosine',
    'qwenvision': 'mean FP32 over all final vision block tokens BEFORE merger and language decoder; 1024 input dimensions',
    'behavior': '36 decoder layers at last input token of original task/label prompt, before answer; 2560 each',
    'Qwen_vision': 'original AutoProcessor + resized_height/width224 message hints, no pixel override; hints are not guaranteed resize limits',
    'cache_dtype': {'clip': 'float32', 'qwenvision': 'float16', 'behavior': 'float16'},
    'normalization': 'per input vector L2 before W; per projected vector L2 after W; fixed uniform layer-mean cosine',
    'train_steps': 1500, 'optimizer': 'AdamW', 'learning_rate': 0.0003,
    'weight_decay': 0.01, 'betas': [0.9, 0.999], 'temperature': 0.07,
    'objective': 'existing supervised_contrastive_loss: -log(sum positive exp / sum all nonself exp); same implementation for all three',
    'classes_per_batch': 16, 'samples_per_class': 4, 'batch_rng_seed': 10073,
    'initialization': 'orthogonal bias-free W, torch seed73 separately each tuple; no learned layer weights',
    'matched_parameters': 'only input representation and consequent W input width/layer count differ',
    'development_per_class': 10, 'split_seed': 73,
    'development_candidate': 'bank training only; exclude every development ID',
    'final_test_candidate': 'complete official bank including dev; no refitting, same for all methods',
    'checkpoint_rule': 'max dev top4 same-label fraction at 0,100,...1500; earliest exact tie',
    'width_rule': 'max full development ICL correct count; smallest width on exact tie',
    'test_gate': 'all 60 trained tuples and their complete development ICL/15 immutable width locks before any official test ICL',
    'raw_methods': 'no training or width selection; same representation arrays and final candidate bank',
    'RICES_alias': 'raw_clip is exact direct CLIP cosine RICES in this new cohort; do not generate duplicate predictions',
    'ICL': 'original task/vision/four-demo order; constrained legal greedy labels; batch1',
    'order': 'lowest to highest actual similarity among top4; stable original bank order resolves exact retrieval ties',
    'knn': 'same actual top4: majority -> summed actual similarity -> nearest; weighted exp(score/.07)',
    'precision': 'projection training/encoding CUDA BF16 autocast, saved embeddings FP32 scoring; raw normalization/scoring FP32',
    'interpretation': 'independent matched cohort; exploratory because old official test results already observed; no claim of old-source reproduction',
    'test_label_use': 'evaluation only, never retrieval input, fitting, width or checkpoint selection',
}


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if canonical(read(path)) != canonical(value): raise ValueError('Immutable identity changed: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True); atomic(path, value)


def split_bank(rows):
    import numpy as np
    rng = np.random.default_rng(73)
    labels = np.asarray([s['label'] for s in rows]); train, dev = [], []
    for label in sorted(set(labels)):
        ids = np.flatnonzero(labels == label)
        if len(ids) < 14: raise ValueError('Insufficient per-class training/development rows')
        order = rng.permutation(ids)
        dev.extend(order[:10].tolist()); train.extend(order[10:].tolist())
    return {'train_ids': [rows[i]['sample_id'] for i in sorted(train)],
        'development_ids': [rows[i]['sample_id'] for i in sorted(dev)]}


def validate_split(manifest, split):
    bank = [s['sample_id'] for s in manifest['bank']]; test = [s['sample_id'] for s in manifest['query']]
    train, dev = split['train_ids'], split['development_ids']
    if (len(bank) != len(set(bank)) or len(test) != len(set(test)) or set(bank) & set(test)
            or len(train) != len(set(train)) or len(dev) != len(set(dev)) or set(train) & set(dev)
            or set(train) | set(dev) != set(bank) or split != split_bank(manifest['bank'])):
        raise ValueError('Invalid, changed or leaking matched split')


def select_width(records):
    if len(records) != 4 or {r['width'] for r in records} != set(WIDTHS) or len({r['total'] for r in records}) != 1:
        raise ValueError('Need all four complete equal-denominator development evaluations')
    if any(type(r['correct']) is not int or r['total'] <= 0 or not 0 <= r['correct'] <= r['total'] for r in records):
        raise ValueError('Invalid complete development counts')
    return min(records, key=lambda r: (-r['correct'], r['width']))['width']


def validate_top4(demos, scores, bank_ids):
    import math
    if (len(demos) != 4 or len(set(demos)) != 4 or not set(demos) <= set(bank_ids)
            or len(scores) != 4 or any(not math.isfinite(v) for v in scores)
            or scores != sorted(scores)):
        raise ValueError('Invalid bank top4/actual score/order')


def prediction_prefix(path, queries, selections, labels, context_hash, complete=False):
    if not Path(path).exists():
        if complete: raise ValueError('Missing full predictions')
        return []
    rows = []
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if not line.endswith('\n'): raise ValueError('Truncated prediction; preserve, never silently repair')
            row = json.loads(line); i = len(rows)
            if i >= len(queries): raise ValueError('Too many predictions')
            sample, selection = queries[i], selections[queries[i]['sample_id']]
            if (row.get('query_id') != sample['sample_id'] or row.get('target') != sample['label']
                    or row.get('context_hash') != context_hash or row.get('demo_ids') != selection['demo_ids']
                    or row.get('scores') != selection['scores'] or row.get('prediction') not in labels
                    or type(row.get('correct')) is not bool or row['correct'] != (row['prediction'] == sample['label'])
                    or row.get('input_tokens', 0) <= 0 or row.get('output_tokens', 0) <= 0):
                raise ValueError('Prediction identity/legality/count/order mismatch')
            rows.append(row)
    if complete and len(rows) != len(queries): raise ValueError('Incomplete output; no partial accuracy')
    return rows


def native_identity(expected):
    """Actual native identity, never assign expected Pod ID to the environment."""
    proc = Path('/proc/1/environ'); values = []
    for item in proc.read_bytes().split(b'\0'):
        key, _, value = item.partition(b'=')
        if key == b'RUNPOD_POD_ID': values.append(value.decode('ascii'))
    if len(values) > 1: raise ValueError('Duplicate native Pod field')
    export = Path('/etc/rp_environment')
    if export.exists() or export.is_symlink():
        info = export.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
            raise ValueError('Untrusted native environment export')
        exports = []
        for line in export.read_text().splitlines():
            if line.startswith('export RUNPOD_POD_ID='):
                match = re.fullmatch(r'export RUNPOD_POD_ID=["\x27]?([A-Za-z0-9_-]{1,80})["\x27]?', line)
                if not match: raise ValueError('Malformed native export Pod field')
                exports.append(match.group(1))
        if len(exports) > 1: raise ValueError('Duplicate export Pod field')
        values.extend(exports)
    if not values or any(v != expected for v in values): raise ValueError('Native Pod identity absent/conflicting')
    inherited = os.environ.get('RUNPOD_POD_ID')
    if inherited is not None and inherited != expected: raise ValueError('Conflicting inherited Pod identity')
    return {'native_pod_ids': values, 'ssh_inherited_id_present': inherited is not None,
        'expected_not_injected': True}


def gpu_guard(plan):
    import subprocess
    import sys
    if sys.platform != 'linux': raise ValueError('P2 GPU execution is cloud Linux only; no local CUDA')
    auth = read(ROOT / 'results' / OPERATION / 'execution_authorization.json')
    ready = read(ROOT / 'results' / OPERATION / 'backup_ready.json')
    if (auth.get('plan_hash') != plan['plan_hash'] or auth.get('experiment') != SCIENCE
            or ready.get('plan_hash') != plan['plan_hash'] or ready.get('launch_id') != auth.get('launch_id')
            or ready.get('pod_id') != auth.get('pod_id') or ready.get('status') != 'actual_backup_succeeded'):
        raise ValueError('Wrong independent launch or no actual P2 backup readiness')
    native_identity(auth['pod_id'])
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    if any(v.strip().isdigit() and int(v.strip()) != os.getpid() for v in active.splitlines()):
        raise ValueError('Other computation exists; no parallel GPU job')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != auth['device']:
        raise ValueError('Wrong exact single authorized GPU')
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    return auth


def validate_plan(plan):
    if plan.get('protocol') != PROTOCOL or plan.get('plan_hash') != canonical({k: v for k, v in plan.items() if k != 'plan_hash'}):
        raise ValueError('P2 plan protocol/hash changed')
    if set(plan['datasets']) != set(DATASETS): raise ValueError('Five datasets required')
    for rel, sha in plan['code_sha256'].items():
        if digest(ROOT / rel) != sha: raise ValueError('P2 locked code changed: ' + rel)
    return plan
