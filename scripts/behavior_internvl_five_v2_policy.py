"""CPU-safe independent second-VLM protocol and strict execution gates."""
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys

from behavior_task_prompt_policy import read, now, immutable, canonical, digest, atomic
from behavior_p2_matched_policy import native_identity, split_bank, validate_split, validate_top4

ROOT = Path(__file__).resolve().parents[1]
SCIENCE = 'behavior_internvl_five_dataset_v2'
OPERATION = 'behavior_internvl_five_dataset_cloud_v2'
PLAN = ROOT / 'analysis/second_vlm_five_dataset_v2/plan_lock.json'
DATASETS = ('dtd', 'aircraft', 'cub', 'pets', 'dogs')
METHODS = ('zero', 'rices', 'behavior_raw', 'behavior_proj')
MODEL = 'OpenGVLab/InternVL2_5-8B'
REVISION = 'e9e4c0dc1db56bfab10458671519b7fa3dd29463'
CLIP = 'openai/clip-vit-base-patch32'
CLIP_REVISION = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
ENVIRONMENT = {'torch': '2.8.0+cu128', 'torchvision': '0.23.0+cu128',
               'transformers': '4.37.2', 'tokenizers': '0.15.2',
               'huggingface-hub': '0.36.0', 'safetensors': '0.6.2',
               'sentencepiece': '0.2.1', 'einops': '0.8.1', 'timm': '1.0.15',
               'numpy': '2.5.1', 'scipy': '1.18.0', 'Pillow': '12.2.0'}
PROTOCOL = {
    'experiment': SCIENCE, 'datasets': list(DATASETS), 'methods': list(METHODS),
    'model': MODEL, 'revision': REVISION, 'clip_revision': CLIP_REVISION,
    'official_queries': 23256, 'planned_predictions': 93024,
    'pre_answer_source': '32 decoder last-input-token states4096; exclude embedding, final state after native final norm; saveFP16',
    'vision': 'author native448 dynamic grid max2 plus thumbnail iff grid>1, deterministic area/width/height ties; identical within model',
    'maximum_tiles_per_image': 3, 'image_context_tokens_per_tile': 256,
    'native_input': 'author internvl2_5 system/conversation template, one or five separate ordered image placeholders',
    'generation': 'batch1 greedy native inputs_embeds cached generation; actual prefix offset measured; exact class tokens plus native conversation EOS',
    'zero': 'same constrained legal decoder as ICL; not free zero decoding or Qwen P0 reuse',
    'rices': 'new fixed CLIP-ViT-B/32 native image features512, raw L2 cosine',
    'raw': 'FP32 per-layer L2 cosine then uniform arithmetic layer mean; no trained weights',
    'proj': 'same Raw layers with shared bias-free orthogonal4096x256 W; L2 before/after; uniform layers',
    'training': {'seed': 73, 'batch_rng_seed': 10073, 'train_steps': 1500, 'width': 256,
                 'loss': 'existing supervised_contrastive_loss, positive logsumexp / all nonself logsumexp',
                 'optimizer': 'AdamW', 'lr': .0003, 'weight_decay': .01,
                 'temperature': .07, 'classes_per_batch': 16, 'samples_per_class': 4,
                 'development_per_class': 10, 'dev_candidate': 'training only',
                 'best': 'earliest maximal complete dev top4 same-label fraction at0,100,...1500',
                 'final_candidate': 'complete official bank including dev; no refit or test tuning'},
    'retrieval_order': 'stable actual-score descending top4 then reverse for ascending demo score; ties original bank order',
    'test_truth': 'evaluation only, never query extraction/prompt/retrieval/selection/training',
    'interpretation': 'within-model contrasts; InternVL8B vs Qwen4B not size matched; exploratory choice after existing Qwen results',
    'non_goals': 'no Learn, grid, text/audio, old Qwen source/checkpoint substitution or favorable subset reporting',
}


def validate_plan(plan):
    if plan.get('plan_hash') != canonical({k: v for k, v in plan.items() if k != 'plan_hash'}) or plan.get('protocol') != PROTOCOL:
        raise ValueError('Independent second-VLM plan/protocol changed')
    if list(plan['datasets']) != list(DATASETS):
        raise ValueError('Wrong agreed full-test allocation')
    for dataset, data in plan['datasets'].items():
        expected = {'dtd': (3760,1880), 'aircraft': (6667,3333), 'cub': (5994,5794), 'pets': (3680,3669), 'dogs': (12000,8580)}[dataset]
        manifest = data['manifest']
        if tuple(len(manifest[v]) for v in ('bank', 'query')) != expected:
            raise ValueError('Incomplete official image splits')
        validate_split(manifest, data['split'])
        if data['class_labels'] != sorted({row['label'] for row in manifest['bank']}):
            raise ValueError('Class inventory differs')
        for split in ('bank', 'query'):
            for row in manifest[split]:
                location = (ROOT / row['path']).resolve()
                if row['split'] != split or not location.is_relative_to(ROOT / 'datasets') or len(row['image_sha256']) != 64:
                    raise ValueError('Official image path/split/hash differs')
    for rel, sha in plan['code_sha256'].items():
        if digest(ROOT / rel) != sha:
            raise ValueError('Independent science code bytes changed: ' + rel)
    for model, revision in ((MODEL, REVISION), (CLIP, CLIP_REVISION)):
        inventory = plan['models'][model]
        if inventory['model'] != model or inventory['revision'] != revision or not inventory['files']:
            raise ValueError('Exact public model-byte inventory absent')
        for name, item in inventory['files'].items():
            if '/' in name or '\\' in name or '..' in name or len(item['sha256']) != 64 or item['bytes'] <= 0:
                raise ValueError('Wrong closed model-file identity')
    inventory = read(ROOT / 'analysis/second_vlm_source_inventory_v1/inventory.json')
    for name, item in inventory['files'].items():
        if name == 'README.md':
            continue
        if plan['models'][MODEL]['files'].get(name, {}).get('sha256') != item['sha256']:
            raise ValueError('Previously reviewed author bytes changed')
    return plan


def validate_ready(auth, ready, heartbeat, at=None):
    fields = ('experiment', 'plan_hash', 'pod_id', 'launch_id')
    for item in (ready, heartbeat):
        if item.get('status') != 'actual_backup_succeeded' or any(not auth.get(k) or item.get(k) != auth[k] for k in fields):
            raise ValueError('Missing new exact successful backup')
        if len(item.get('snapshot_archive_sha256', '')) != 64:
            raise ValueError('Missing actual archiveSHA')
    age = ((at or datetime.now(timezone.utc)) - datetime.fromisoformat(heartbeat['utc'])).total_seconds()
    if not 0 <= age <= 300:
        raise ValueError('Stale or future independent backup heartbeat')


def model_path(plan, name):
    folder = ROOT / '.hf_cache' / ('models--' + name.replace('/', '--')) / 'snapshots' / plan['models'][name]['revision']
    for relative, item in plan['models'][name]['files'].items():
        path = folder / relative
        # HF snapshot symlinks to its byte-hashed blob store are expected.
        if not path.is_file() or path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            raise ValueError('Actual complete model/tokenizer/author file bytes differ: ' + relative)
    return folder


def live_guard():
    # Cheap per-row guard; full code/model/env verification happens before each
    # child phase. The independent parent additionally monitors sole GPU/backup.
    if sys.platform != 'linux' or str(ROOT) != '/workspace/BehaviorICL' or Path.cwd() != ROOT:
        raise ValueError('Exact cloud Linux only; no local CUDA/model/customcode')
    auth = read(ROOT / 'results' / OPERATION / 'execution_authorization.json')
    if auth.get('experiment') != SCIENCE or not auth.get('plan_hash') or auth.get('actual_api_pod_id') != auth.get('pod_id'):
        raise ValueError('Wrong independent native launch')
    native_identity(auth['pod_id'])
    validate_ready(auth, read(ROOT / 'results' / OPERATION / 'backup_ready.json'),
                   read(ROOT / 'results' / OPERATION / 'backup_heartbeat.json'))
    if (ROOT / 'results' / OPERATION / 'stop_request.json').exists():
        raise ValueError('Independent watcher/controller requested safety stop')
    return auth


def gpu_guard(plan):
    if sys.platform != 'linux' or str(ROOT) != '/workspace/BehaviorICL' or Path.cwd() != ROOT:
        raise ValueError('Exact cloud Linux only; no local CUDA/model/customcode')
    validate_plan(plan)
    auth = read(ROOT / 'results' / OPERATION / 'execution_authorization.json')
    if (auth.get('experiment') != SCIENCE or auth.get('plan_hash') != plan['plan_hash']
            or auth.get('actual_api_pod_id') != auth.get('pod_id') or not auth.get('launch_id')
            or auth.get('author_code_execution_authorized') is not True):
        raise ValueError('Missing independently audited second-model execution authorization')
    proof = ROOT / 'results' / OPERATION / 'native_identity_receipt.json'
    if digest(proof) != auth.get('native_identity_receipt_sha256') or read(proof).get('pod_id') != auth['pod_id']:
        raise ValueError('Actual native/API proof not bound to this launch')
    native_identity(auth['pod_id'])
    validate_ready(auth, read(ROOT / 'results' / OPERATION / 'backup_ready.json'),
                   read(ROOT / 'results' / OPERATION / 'backup_heartbeat.json'))
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    if any(v.strip().isdigit() and int(v.strip()) != os.getpid() for v in active.splitlines()):
        raise ValueError('Parallel computation detected')
    if {name: importlib.metadata.version(name) for name in ENVIRONMENT} != ENVIRONMENT:
        raise ValueError('Independent isolated pinned runtime differs')
    for name in (MODEL, CLIP): model_path(plan, name)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != auth['device']:
        raise ValueError('Wrong exact single GPU')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    return auth
