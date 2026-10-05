"""Read-only, full-test P1 recheck; no model, training, CUDA or source mutation."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from datetime import datetime, timezone
import run_behavior_priority_analysis as evidence

ROOT = Path(__file__).resolve().parents[1]
KEYS = {'RICES': ('base', 'rices'), 'GPT-MM': ('base', 'gpt_mm'),
        'DeTriever': ('detriever', 'detriever'),
        'BehaviorICL-Raw': ('behavior', 'decision_zero'),
        'BehaviorICL-Proj': ('proj', 'projection_only'),
        'BehaviorICL-Learn': ('behavior', 'decision_learn')}


def full_counts(rows, ordered_ids):
    if len(rows) != len(ordered_ids) or set(rows) != set(ordered_ids):
        raise ValueError('Incomplete official query set; no partial accuracy')
    return sum(rows[i]['correct'] for i in ordered_ids), len(ordered_ids)


def audit(out):
    if out.exists():
        raise RuntimeError('Fresh audit root required; do not overwrite')
    prior = evidence.read(ROOT / 'analysis/priority_experiments/audit.json')
    pinned = prior['source_sha256']
    for path, expected in pinned.items():
        if evidence.sha(ROOT / path.replace('\\', '/')) != expected:
            raise ValueError('Registered source changed: ' + path)
    with (ROOT / 'analysis/priority_experiments/behavior_raw_results.csv').open(encoding='utf-8-sig') as f:
        published = {(r['dataset'], r['method']): r for r in csv.DictReader(f)}
    if len(published) != 30:
        raise ValueError('Expected 30 existing full-test groups')
    results, provenance, raw_protocols = [], {}, {}
    for dataset, sources in evidence.source_catalog().items():
        manifest = evidence.read(sources['base'] / 'manifest.json')
        ids = [r['sample_id'] for r in manifest['query']]
        raw_identity = evidence.read(sources['behavior'] / 'run_identity.json')
        raw_config = evidence.read(sources['behavior'] / 'config.json')
        raw_protocol = evidence.read(sources['behavior'] / 'protocol.json')
        if (raw_identity != {'dataset': dataset, 'model_slug': 'qwen3vl4b',
                'model_id': evidence.MODEL, 'seed': 73, 'ablation': '36_raw_layer_states_v1'}
                or raw_config['shots'] != 4 or raw_config['seed'] != 73
                or raw_config['state_shape'] != [len(manifest['bank']) + len(ids), 36, 2560]
                or raw_protocol['ablation'] != '36 normalized raw layer states; no adjacent-layer subtraction'):
            raise ValueError('Raw recorded protocol differs: ' + dataset)
        raw_protocols[dataset] = {'identity': raw_identity, 'state_shape': raw_config['state_shape'],
            'recorded_protocol': raw_protocol['ablation'],
            'config_sha256': evidence.sha(sources['behavior'] / 'config.json')}
        for method, (source_key, key) in KEYS.items():
            rows = evidence.load_group(sources[source_key], key, manifest, provenance)
            correct, total = full_counts(rows, ids)
            old = published[dataset, method]
            if int(old['correct']) != correct or int(old['total']) != total or abs(float(old['accuracy']) - correct / total) > 1e-8:
                raise ValueError('Published count differs: ' + dataset + '/' + method)
            results.append({'dataset': dataset, 'method': method, 'correct': correct,
                'total': total, 'accuracy_percent': 100 * correct / total,
                'cohort': evidence.COHORT, 'separate_retrieval_seconds': None,
                'separate_ICL_seconds': None, 'source_folder': str(sources[source_key].relative_to(ROOT))})
    out.mkdir(parents=True)
    evidence.csv_out(out / 'full_test_counts.csv', results)
    receipt = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'all_30_groups_verified',
        'registered_source_SHA_checks': len(pinned), 'current_group_source_sha256': provenance,
        'official_queries': sum(r['total'] for r in results if r['method'] == 'BehaviorICL-Raw'),
        'raw_recorded_protocols': raw_protocols, 'script_sha256': evidence.sha(Path(__file__)),
        'raw_kernel_inspection_sha256': evidence.sha(ROOT / 'scripts/run_dtd_decision_states.py'),
        'raw_kernel': 'per-layer L2 normalize; mean of 36 layer cosine scores; no W, no label input',
        'kernel_replay': 'not performed; original arrays not all locally available',
        'timing': 'separate retrieval/ICL time not recoverable from existing group artifacts; remains missing',
        'model_or_CUDA_or_training': False, 'source_files_changed': False,
        'old_RESULTS_sha256': evidence.sha(ROOT / 'RESULTS.md'), 'results': results}
    (out / 'audit.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output-root', type=Path, default=ROOT / 'analysis/paper_p1_recheck_v1')
    r = audit(p.parse_args().output_root)
    print(json.dumps({k: r[k] for k in ('status', 'registered_source_SHA_checks', 'official_queries', 'timing')}, ensure_ascii=False))
