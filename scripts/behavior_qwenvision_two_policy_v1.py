"""New single visual SupCon control; old P2/Behavior identities stay immutable."""
from pathlib import Path
import os
import subprocess
import sys
from behavior_p2_matched_policy import (ROOT, DATASETS, QWEN_REVISION, MODEL,
    read, digest, canonical, atomic, immutable, validate_split, validate_top4,
    native_identity, prediction_prefix)

DATASETS = ('dtd', 'cub')
SCIENCE = 'behavior_qwenvision_supcon_dtd_cub_v1'
OPERATION = 'behavior_qwenvision_supcon_dtd_cub_cloud_v1'
PLAN_PATH = ROOT / 'analysis/paper_qwenvision_dtd_cub_v1/plan_lock.json'
REPRESENTATIONS = ('qwenvision',)
WIDTHS = (256,)
PROTOCOL = {
    'experiment': SCIENCE, 'datasets': list(DATASETS), 'seed': 73, 'shots': 4,
    'representations': ['qwenvision'], 'widths': [256],
    'Qwen_revision': QWEN_REVISION,
    'qwenvision': 'FP32 mean of all final vision-block tokens before merger/decoder; 1024 input dimensions, saved FP16',
    'normalization': 'L2 before shared bias-free W and after W; cosine; uniform layers; no trained layer weights',
    'train_steps': 1500, 'optimizer': 'AdamW', 'learning_rate': .0003,
    'weight_decay': .01, 'betas': [.9, .999], 'temperature': .07,
    'objective': 'existing supervised_contrastive_loss: -log(sum positive exp / sum all nonself exp)',
    'classes_per_batch': 16, 'samples_per_class': 4, 'batch_rng_seed': 10073,
    'initialization': 'fresh orthogonal W, torch seed73 separately each dataset',
    'development_per_class': 10, 'split_seed': 73,
    'development_candidate': 'training bank only; exclude all development IDs',
    'final_test_candidate': 'complete official bank including dev; no refitting',
    'checkpoint_rule': 'max dev top4 same-label fraction at 0,100,...1500; earliest exact tie',
    'width_rule': 'fixed256 before new results; no width grid or development ICL selection',
    'ICL': 'original task/vision/four-demo order; constrained legal greedy labels; batch1',
    'order': 'torch.topk(4).indices.flip(1), as historical Proj; save actual similarities, not ranks',
    'Qwen_vision': 'original AutoProcessor; resized_height/width224 message hints, no pixel override',
    'precision': 'CUDA BF16 projection training/encoding; encoded vectors FP32 scoring',
    'test_label_use': 'evaluation only; never retrieval, fitting or checkpoint selection',
    'reuse': 'audited historical fixed256 Behavior references, not reclassified as new P2; only complete byte-verified QwenVision caches',
    'interpretation': 'exploratory historical-reference comparison; source/environment differences disclosed, not a bitwise replay of all old Behavior runs',
}


def validate_reference(data, config, split, metadata, protocol):
    if (config.get('seed'), config.get('shots'), config.get('train_steps'), config.get('projection_dim')) != (73, 4, 1500, 256):
        raise ValueError('Historical Proj recipe does not match fixed256 control')
    if (split.get('train_ids') != data['split']['train_ids'] or
            split.get('validation_ids') != data['split']['development_ids']):
        raise ValueError('Historical training/development ID/order mismatch')
    if (metadata.get('method') != 'projection_only' or metadata.get('train_steps') != 1500 or
            metadata.get('projection_dim') != 256 or metadata.get('learning_rate') != .0003 or
            metadata.get('selection_metric') != 'bank development top4_same_label_fraction' or
            [r['step'] for r in metadata['history']] != list(range(0, 1501, 100))):
        raise ValueError('Historical training/checkpoint selection mismatch')
    if protocol.get('model') != MODEL or protocol.get('bank_count') != len(data['manifest']['bank']) or protocol.get('query_count') != len(data['manifest']['query']):
        raise ValueError('Historical model/candidate/test protocol mismatch')
    return True


def validate_plan(plan):
    if plan.get('protocol') != PROTOCOL or plan.get('plan_hash') != canonical({k:v for k,v in plan.items() if k != 'plan_hash'}):
        raise ValueError('New single-control plan hash/protocol changed')
    if tuple(plan['datasets']) != DATASETS or plan['planned_test_generations'] != 7674 or plan['planned_training_tuples'] != 2:
        raise ValueError('Scope is not the authorized five-dataset one-control plan')
    for rel, expected in plan['code_sha256'].items():
        if digest(ROOT / rel) != expected: raise ValueError('New locked code changed: ' + rel)
    return plan


def gpu_guard(plan):
    if sys.platform != 'linux': raise ValueError('Cloud Linux only; local CUDA forbidden')
    auth = read(ROOT/'results'/OPERATION/'execution_authorization.json')
    ready = read(ROOT/'results'/OPERATION/'backup_ready.json')
    if (auth.get('experiment') != SCIENCE or auth.get('plan_hash') != plan['plan_hash'] or
            any(ready.get(k) != auth.get(k) for k in ('pod_id', 'launch_id', 'plan_hash')) or
            ready.get('status') != 'actual_backup_succeeded'):
        raise ValueError('Wrong independent authorization or no actual backup')
    native_identity(auth['pod_id'])
    active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'], text=True)
    if any(v.strip().isdigit() and int(v.strip()) != os.getpid() for v in active.splitlines()):
        raise ValueError('Another GPU computation exists')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != auth['device']:
        raise ValueError('Wrong exact authorized single GPU')
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    return auth
