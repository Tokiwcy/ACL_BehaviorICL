"""Independent post-stop full DTD+CUB fixed-context label audit/analysis.

Never executes capture, imports a model, contacts a Pod or reports partial
accuracy. Original sample sources/analysis/locks are read-only dependencies.
"""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
import tarfile

import run_behavior_full_test_labels as runner
import behavior_critical_followup_cloud as cloud
import analyze_behavior_label_intervention as tables
import analyze_behavior_task_prompt_complete_cpu as messages

p = runner.common
ROOT = p.ROOT
IMPLEMENTATION = ROOT/'analysis/paired_label_full_test_audit_implementation_v1/implementation_lock.json'
SOURCES = ('scripts/analyze_behavior_full_test_labels_complete_cpu.py',
           'tests/test_behavior_full_test_labels_complete_cpu.py',
           'scripts/analyze_behavior_label_intervention.py',
           'scripts/analyze_behavior_task_prompt_complete_cpu.py',
           'scripts/analyze_behavior_repair_all_queries.py',
           'scripts/behavior_critical_followup_cloud.py',
           'scripts/run_behavior_full_test_labels.py', 'scripts/run_behavior_label_intervention.py')


def implementation_value():
    return {'scientific_plan_hash': runner.PLAN_HASH, 'attachment_hash': runner.INPUT_HASH,
            'scientific_implementation_hash': p.read(runner.BINDING)['implementation_hash'],
            'sources': {name: p.digest(ROOT/name) for name in SOURCES},
            'required': 'complete official test, full archive/member/extracted byteSHA, earliest exact stop before independent CPU audit',
            'queries': 7674, 'records': 30696, 'generated': 30526, 'explicit_missing': 170,
            'bootstrap_repetitions': 10000, 'seed': 73, 'GPU_execution': False,
            'sample_prediction_reuse': False, 'scientific_source_changes': False}


def freeze():
    if IMPLEMENTATION.parent.exists():
        raise ValueError('Preserve independent audit implementation; no repeat freeze')
    value = implementation_value(); value['audit_implementation_hash'] = p.canonical(value)
    p.immutable(IMPLEMENTATION, value)
    return value


def implementation_check():
    value = p.read(IMPLEMENTATION)
    if ({k: v for k, v in value.items() if k != 'audit_implementation_hash'} != implementation_value()
        or value['audit_implementation_hash'] != p.canonical({k: v for k, v in value.items() if k != 'audit_implementation_hash'})):
        raise ValueError('Independent full-label audit code/science identity differs')
    return value


def terminal_guard(take, stop, account, auth):
    at = datetime.fromisoformat(take['utc']); ended = datetime.fromisoformat(stop['confirmed_utc'])
    if (take.get('stage') != 'all_complete_verified' or stop.get('reason') != 'all_complete_verified'
        or take.get('all_new_arrays_and_checkpoints_retrieved') is not True
        or stop.get('pod_id') != auth['pod_id'] or stop.get('disks_preserved') is not True
        or account.get('pod_id') != auth['pod_id'] or account.get('launch_id') != auth['launch_id']
        or account.get('phase') != 'stopped' or account.get('confirmed_utc') != stop['confirmed_utc']
        or at.tzinfo is None or ended.tzinfo is None or at > ended):
        raise ValueError('Successful full official takeback and earliest exact stopped identity required')


def archive_audit(session, take, cfg):
    archive = session/'full_takeback.tar.gz'
    if p.digest(archive) != take['archive_sha256']:
        raise ValueError('Complete original archive SHA changed')
    manifest = {r['name']: r for r in take['members']}
    if len(manifest) != len(take['members']) or len(manifest) != take['files']:
        raise ValueError('Duplicate or missing takeback manifest members')
    result = {}
    with tarfile.open(archive) as tar:
        members = cloud.safe_members(tar, cfg, True)
        if {m.name for m in members} != set(manifest):
            raise ValueError('Closed complete archive member set differs')
        for member in members:
            expected = manifest[member.name]; h = hashlib.sha256()
            if member.size != expected['bytes']:
                raise ValueError('Scientific member byte count differs')
            with tar.extractfile(member) as handle:
                while chunk := handle.read(8*1024*1024): h.update(chunk)
            path = session/'snapshot'/member.name
            if (path.is_symlink() or not path.resolve().is_relative_to(session/'snapshot')
                or h.hexdigest() != expected['sha256'] or p.digest(path) != expected['sha256']):
                raise ValueError('Actual archive/member/extracted scientific byteSHA differs')
            result[member.name] = expected['sha256']
    return result


def native_audit(auth, launch, native, operation):
    cloud.exact(launch, auth)
    if (auth.get('mode') != 'full_labels' or auth.get('experiment') != runner.SCIENCE
        or auth.get('plan_hash') != runner.PLAN_HASH or auth.get('protocol_hash') != runner.PLAN_HASH
        or auth.get('actual_api_pod_id') != auth.get('pod_id') or launch.get('cwd') != '/workspace/BehaviorICL'
        or native.get('actual_api_pod_id') != auth['pod_id'] or native.get('proc1_pod_id') != auth['pod_id']
        or native.get('export_file_pod_id') != auth['pod_id']
        or native.get('ssh_process_pod_id') not in (None, auth['pod_id'])
        or native.get('expected_Pod_field_injected') is not False
        or native.get('export_root_owned_nonwritable_regular') is not True
        or p.digest(operation/'native_identity_receipt.json') != auth.get('native_identity_receipt_sha256')
        or p.read(operation/'native_identity_receipt.json') != native):
        raise ValueError('Actual original Pod/native/launch proof differs; no injected expected identity')


def original_messages(row, arm, builders):
    request = runner.original.prompt_spec(row, arm)
    query = messages.at_cloud({'path': request['query_image']['relative_path']})
    if arm == 'same_constrained_zero':
        return builders['zero_shot_messages'](query, row['dataset'], row['legal_classes'], 224)
    demos = [messages.at_cloud({'path': image['relative_path']}, label)
             for image, label in zip(request['demo_images'], request['demo_labels'])]
    return builders['icl_messages'](demos, query, row['dataset'], row['legal_classes'], 224)


def actual_smoke_audit(smoke, records, rows, job):
    count = runner.smoke_count(rows)
    if (smoke.get('status') != 'actual_generated_all_four_arms' or smoke.get('job_hash') != job['job_hash']
        or smoke.get('prefix_records') != count or smoke.get('prefix_hash') != p.canonical(records[:count])
        or smoke.get('counted_once_as_official_prefix') is not True
        or {r['arm'] for r in records[:count] if r['status'] == 'generated'} != set(runner.original.ARMS)):
        raise ValueError('Actual original four-arm smoke prefix missing or double counted')


def runtime_audit(value, auth, job, plan, binding):
    if (value.get('pod_id') != auth['pod_id'] or value.get('launch_id') != auth['launch_id']
        or value.get('device') != auth['device'] or value.get('job_hash') != job['job_hash']
        or value.get('implementation_hash') != binding['implementation_hash']
        or value.get('environment') != plan['expected_environment']
        or not math.isfinite(value.get('model_load_seconds', float('nan')))
        or value['model_load_seconds'] <= 0):
        raise ValueError('Actual model/generation runtime has another Pod/job or environment')


def record_accounting(record):
    if record['status'] == 'generated':
        if len(record['generated_token_ids']) != record['output_tokens']:
            raise ValueError('Actual output token count differs')
    elif (record['status'] != 'missing_illegal_demo_zero' or record['arm'] != 'demo_own_legal_zero_labels'
          or any(record.get(k) is not None for k in ('raw_answer', 'prediction', 'correct', 'generated_token_ids', 'prompt_messages_sha256', 'demo_label_token_counts'))
          or any(record.get(k) != 0 for k in ('input_tokens', 'output_tokens', 'elapsed_seconds'))):
        raise ValueError('Explicit illegal own-label missing arm must have no generation/prompt/tokens/time')


def dataset_audit(dataset, rows, folder, plan, inputs, binding, auth, builders, tokenizer):
    if len(rows) != runner.COUNTS[dataset]:
        raise ValueError('Complete official dataset only, never original sampled query cohort')
    runner.original.verify_images(rows)
    job = runner.job(plan, inputs, binding, dataset)
    if p.read(folder/'identity.json') != job:
        raise ValueError('Original independent full-test job/context changed')
    codec, _ = messages.runner.codec_for(tokenizer, rows[0]['legal_classes'])
    if p.read(folder/'codec.json') != codec:
        raise ValueError('Original fixed tokenizer/legal codec differs')
    records = runner.verified_prefix(folder/'predictions.jsonl', rows, job, codec, complete=True)
    actual_smoke_audit(p.read(folder/'actual_smoke.json'), records, rows, job)
    for name in ('runtime_smoke.json', 'runtime_full.json'):
        runtime_audit(p.read(folder/name), auth, job, plan, binding)
    verification = p.read(folder/'verification.json')
    expected = {'status': 'all_complete_verified', 'job_hash': job['job_hash'], 'queries': len(rows),
                'records': 4*len(rows), 'generated': 4*len(rows)-runner.MISSING[dataset],
                'explicit_missing': runner.MISSING[dataset], 'outputs_sha256': p.digest(folder/'predictions.jsonl'),
                'input_tokens': sum(r['input_tokens'] for r in records),
                'output_tokens': sum(r['output_tokens'] for r in records),
                'elapsed_seconds': sum(r['elapsed_seconds'] for r in records)}
    measured = expected.pop('elapsed_seconds')
    recorded_seconds = verification.get('elapsed_seconds', float('nan'))
    if ({k: v for k, v in verification.items() if k != 'elapsed_seconds'} != expected
        or not math.isfinite(recorded_seconds) or not math.isclose(recorded_seconds, measured, rel_tol=1e-12, abs_tol=1e-8)
        or sum(r['status'] != 'generated' for r in records) != runner.MISSING[dataset]):
        raise ValueError('Complete dataset verification/denominators/actual token and time totals differ')
    for (row, arm), record in zip(runner.original.schedule(rows), records):
        record_accounting(record)
        if record['status'] == 'generated' and p.canonical(original_messages(row, arm, builders)) != record['prompt_messages_sha256']:
            raise ValueError('Actual frozen image/order/labels/classification prompt fingerprint differs')
    tables.audit_generated_tokens(records, rows, tokenizer)
    return records


def denominator_guard(groups):
    if set(groups) != set(runner.DATASETS):
        raise ValueError('Both complete official datasets required before any accuracy')
    for dataset, records in groups.items():
        if (len(records) != runner.COUNTS[dataset]*4
            or sum(r['status'] == 'generated' for r in records) != runner.COUNTS[dataset]*4-runner.MISSING[dataset]):
            raise ValueError('Full official record/generation/missing denominator differs')
    if (sum(len(r) for r in groups.values()) != 30696
        or sum(r['status'] == 'generated' for rows in groups.values() for r in rows) != 30526):
        raise ValueError('No partial, sampled, doubled smoke or dropped missing-arm analysis')


def analyze(session, out):
    session = session.resolve(); out = out.resolve()
    if (not session.is_relative_to(ROOT/'results/cloud_trial') or not out.is_relative_to(ROOT/'analysis')
        or out.exists() or session.is_symlink()):
        raise ValueError('Exact stopped takeback and fresh analysis root required')
    audit_binding = implementation_check()
    plan, inputs, binding = (p.read(path) for path in (runner.PLAN, runner.INPUT, runner.BINDING))
    runner.validate(plan, inputs); runner.binding_check(plan, inputs, binding)
    cfg = cloud.config('full_labels'); package = p.read(cloud.work('full_labels')/'package.json')
    cloud.validate_package(package, cfg)
    science = session/'snapshot/results'/runner.SCIENCE; operation = session/'snapshot/results'/runner.OPERATION
    auth, launch = (p.read(operation/name) for name in ('execution_authorization.json', 'launch.json'))
    if auth.get('implementation_hash') != binding['implementation_hash'] or auth.get('code_sha256') != binding['code_sha256']:
        raise ValueError('Original independent scientific authorization differs')
    terminal_guard(p.read(session/'full_takeback_receipt.json'), p.read(session/'stop_confirmation.json'),
                   p.read(session/'runtime_accounting.json'), auth)
    members = archive_audit(session, p.read(session/'full_takeback_receipt.json'), cfg)
    native_audit(auth, launch, p.read(session/'native_identity_receipt.json'), operation)
    env = p.read(operation/'environment_audit.json')
    if env.get('versions') != plan['expected_environment'] or env.get('models') != cfg['models']:
        raise ValueError('Full actual original model/runtime byte evidence differs')
    stage = p.read(operation/'sequence_progress.json')
    if (stage.get('mode') != 'full_labels' or stage.get('stage') != 'all_complete_verified'
        or (stage.get('records'), stage.get('generations'), stage.get('explicit_missing')) != (30696, 30526, 170)):
        raise ValueError('Actual complete cloud scientific sequence absent')
    ready = p.read(operation/'inputs_ready.json'); cloud.exact(ready, auth)
    if ready.get('images') != 17428 or ready.get('inputs_zip_sha256') != package['inputs_zip_sha256']:
        raise ValueError('Actual complete original fixed image readiness differs')
    provenance = dict(inputs['source_sha256'])
    for name, expected in provenance.items():
        if p.digest(ROOT/name) != expected: raise ValueError('Registered original input/context/P0/code source bytes differ')
    for name, expected in package['images'].items():
        if p.digest(ROOT/name) != expected: raise ValueError('Original complete official image bytes differ')
    cache = ROOT/'.hf_cache/models--Qwen--Qwen3-VL-4B-Instruct/snapshots'/plan['revision']
    for name, expected in plan['model_weight_and_processor_inventory']['files_SHA256'].items():
        if p.digest(cache/name) != expected: raise ValueError('Complete original local model/tokenizer/processor files differ')
    if importlib.metadata.version('transformers') != plan['expected_environment']['transformers']:
        raise ValueError('CPU tokenizer library differs from the original pinned decoding runtime')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cache, local_files_only=True, trust_remote_code=False)
    builders = messages.frozen_message_builders({'code_sha256': binding['code_sha256']})
    groups = {}
    for dataset in runner.DATASETS:
        rows = [r for r in inputs['all_queries'] if r['dataset'] == dataset]
        groups[dataset] = dataset_audit(dataset, rows, science/dataset, plan, inputs, binding, auth, builders, tokenizer)
    denominator_guard(groups)
    free_csv = ROOT/'analysis/paper_repair_five_dataset_v1/query_level_results.csv'
    free = tables.load_free_zero(free_csv, inputs['all_queries'])
    outcomes, summaries, pairs, consistency = [], [], [], []
    for dataset in runner.DATASETS:
        rows = [r for r in inputs['all_queries'] if r['dataset'] == dataset]
        values = tables.make_tables(rows, groups[dataset], {r['query_id']: free[(dataset, r['query_id'])] for r in rows})
        outcomes.extend(values[0]); summaries.extend(values[1]); pairs.extend(values[2]); consistency.append(values[3])
    # Only after all two complete official groups pass; never current increments.
    for name, expected in provenance.items():
        if p.digest(ROOT/name) != expected: raise ValueError('Original evidence changed during CPU analysis')
    for path in (runner.PLAN, runner.INPUT, runner.BINDING, IMPLEMENTATION, free_csv, Path(__file__),
                 session/'full_takeback_receipt.json', session/'stop_confirmation.json', session/'runtime_accounting.json'):
        provenance[path.relative_to(ROOT).as_posix()] = p.digest(path)
    out.mkdir(parents=True)
    for name, values in (('query_arm_outcomes.csv', outcomes), ('arm_summary.csv', summaries),
                         ('paired_label_effects.csv', pairs), ('identical_label_consistency.csv', consistency)):
        tables.a.csv_out(out/name, values)
    p.immutable(out/'audit.json', {'status': 'complete_verified', 'complete_official_test': True,
        'session': session.relative_to(ROOT).as_posix(), 'protocol_hash': runner.PLAN_HASH,
        'attachment_hash': runner.INPUT_HASH, 'implementation_hash': binding['implementation_hash'],
        'audit_implementation_hash': audit_binding['audit_implementation_hash'],
        'queries': 7674, 'records': 30696, 'generated': 30526, 'explicit_missing': 170,
        'actual_generated_prompt_fingerprints_verified': 30526,
        'actual_four_arm_smokes': list(runner.DATASETS), 'full_model_and_processor_file_SHA_checks': len(cfg['models'][plan['model']]['files']),
        'complete_original_image_SHA_checks': 17428, 'source_sha256': provenance, 'archive_members_SHA': members,
        'primary_zero_reference': 'same-environment newly generated constrained zero; free P0 zero sensitivity only',
        'CI': '10000 query-paired bootstrap seed73, descriptive without multiplicity adjustment',
        'causal_scope': 'label content within fixed retrieved image contexts; not Behavior-unique correction, identical reasoning or independently established same-failure causality',
        'label_length_confound': 'actual input/output and label token counts retained; label length not separately causally isolated',
        'harm_and_missingness_retained': True, 'CPU_only_no_model_or_GPU_execution': True})
    return {'status': 'complete_verified', 'queries': 7674, 'records': 30696}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=('preflight', 'freeze', 'analyze'), default='preflight')
    parser.add_argument('--session', type=Path); parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    if args.phase == 'freeze':
        print({'CPU_implementation_frozen_not_scientific_results': True, 'hash': freeze()['audit_implementation_hash']})
    elif args.phase == 'preflight':
        implementation_check(); print({'CPU_code_identity_verified_only': True, 'GPU_execution': False})
    else:
        if args.session is None or args.output_root is None: raise ValueError('Actual stopped session and fresh output required')
        os.environ.update(CUDA_VISIBLE_DEVICES='-1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        print(analyze(args.session, args.output_root))
