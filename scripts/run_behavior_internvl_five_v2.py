"""Independent InternVL full five-dataset zero/RICES/Raw/Proj scientific runner.

Default is CPU preflight. No deployment, environment install, network/model
download, author-code import or CUDA until separately authorized guarded phase.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import behavior_internvl_five_v2_policy as p
import behavior_internvl_five_v2_adapter as adapter
from collect_behavior_priority_zero import lock

OUT = p.ROOT / 'results' / p.SCIENCE


def progress(dataset, stage, **fields):
    p.atomic(OUT / dataset / 'progress.json', {'plan_hash': p.read(p.PLAN)['plan_hash'],
             'dataset': dataset, 'stage': stage, 'utc': p.now(), **fields})


def write_row(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False) + '\n')
    handle.flush()


def preflight(plan):
    p.validate_plan(plan)
    images = 0
    for data in plan['datasets'].values():
        for split in ('bank', 'query'):
            for row in data['manifest'][split]:
                if p.digest(p.ROOT / row['path']) != row['image_sha256']:
                    raise ValueError('Full official image bytes differ')
                images += 1
    return images


def load_native(plan):
    p.gpu_guard(plan)
    import torch
    from transformers import AutoModel, AutoTokenizer
    location = p.model_path(plan, p.MODEL)
    # Fixed, fully byte-verified, reviewed local author source only. No remote
    # dynamic code or latest revision lookup. Isolated4.37.2 native eager backend.
    tokenizer = AutoTokenizer.from_pretrained(location, trust_remote_code=True,
                                               use_fast=False, local_files_only=True)
    model = AutoModel.from_pretrained(location, trust_remote_code=True,
                                     torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,
                                     use_flash_attn=False, local_files_only=True).eval().to('cuda')
    if (len(model.language_model.model.layers) != 32 or model.num_image_token != 256
            or int(model.config.llm_config.hidden_size) != 4096 or not callable(model.language_model.generate)):
        raise ValueError('Actual native author interface mismatch')
    return model, tokenizer


def absolute_rows(rows):
    # Query truth is not passed to adapter/model; bank truth exists only when
    # explicitly passed as demonstration labels, never as extraction targets.
    return [{'sample_id': row['sample_id'], 'path': str(p.ROOT / row['path'])} for row in rows]


def load_clip(plan):
    import torch
    from transformers import CLIPConfig, CLIPModel, CLIPProcessor
    location = p.model_path(plan, p.CLIP)
    config = CLIPConfig.from_pretrained(location, local_files_only=True)
    model = CLIPModel(config)
    state = torch.load(location / 'pytorch_model.bin', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    processor = CLIPProcessor.from_pretrained(location, local_files_only=True)
    return model.eval().to('cuda'), processor


def captures(plan, dataset, smoke=False):
    import numpy as np
    import torch
    data = plan['datasets'][dataset]
    rows = data['manifest']['bank'] + data['manifest']['query']
    folder = OUT / dataset / 'sources'
    folder.mkdir(parents=True, exist_ok=True)
    status_path = folder / 'source_progress.json'
    status = p.read(status_path) if status_path.exists() else {'plan_hash': plan['plan_hash'], 'completed': 0}
    if status['plan_hash'] != plan['plan_hash'] or not 0 <= status['completed'] <= len(rows):
        raise ValueError('Source committed prefix identity changed')
    count = status['completed']
    arrays = {}
    for name, shape in (('behavior', (len(rows), 32, 4096)), ('clip', (len(rows), 1, 512))):
        location = folder / (name + '.npy')
        if location.exists():
            array = np.load(location, mmap_mode='r+')
            if array.shape != shape or array.dtype != np.float16:
                raise ValueError('Source array shape/dtype changed')
        else:
            if count:
                raise ValueError('Committed source array missing')
            array = np.lib.format.open_memmap(location, mode='w+', dtype='float16', shape=shape)
        arrays[name] = array
    log = folder / 'capture.jsonl'
    logged = []
    if log.exists():
        for line in log.read_text(encoding='utf-8').splitlines(keepends=True):
            if not line.endswith('\n'):
                raise ValueError('Truncated source log; never silently repair')
            logged.append(json.loads(line))
    if len(logged) != count or any((entry['sample_id'], entry['image_sha256'], entry['plan_hash']) !=
                                  (rows[i]['sample_id'], rows[i]['image_sha256'], plan['plan_hash']) for i, entry in enumerate(logged)):
        raise ValueError('Source log/progress/ordered-ID/image prefix mismatch')
    stop = min(len(rows), 1) if smoke else len(rows)
    if count < stop:
        model, tokenizer = load_native(plan)
        clip, processor = load_clip(plan)
        from PIL import Image
        with log.open('a', encoding='utf-8') as handle, torch.inference_mode():
            for i in range(count, stop):
                p.live_guard()
                row, before = rows[i], time.perf_counter()
                text = adapter.question(dataset, data['class_labels'])
                batch, metadata = adapter.prepare(model, tokenizer, absolute_rows([row]), text)
                values = adapter.hidden(model, batch).cpu().numpy().astype(np.float16)
                with Image.open(p.ROOT / row['path']) as image:
                    inputs = processor(images=image.convert('RGB'), return_tensors='pt')
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
                visual = clip.get_image_features(**inputs).float()
                if visual.shape != (1, 512) or not torch.isfinite(visual).all() or not np.isfinite(values).all():
                    raise ValueError('Actual native source smoke/dimensions/finite gate failed')
                arrays['behavior'][i] = values
                arrays['clip'][i, 0] = visual[0].cpu().numpy().astype(np.float16)
                for array in arrays.values(): array.flush()
                torch.cuda.synchronize()
                write_row(handle, {'sample_id': row['sample_id'], 'image_sha256': row['image_sha256'],
                          'plan_hash': plan['plan_hash'], 'input_tokens': metadata['input_tokens'],
                          'visual_tokens': metadata['visual_tokens'], 'patch_counts': metadata['patch_counts'],
                          'prompt_fingerprint': p.canonical(metadata['prompt']), 'seconds': time.perf_counter() - before})
                p.atomic(status_path, {'plan_hash': plan['plan_hash'], 'completed': i + 1, 'expected': len(rows)})
                progress(dataset, 'actual_pre_answer_and_CLIP_source_capture', completed=i + 1, expected=len(rows))
                del batch, values, visual, inputs
        del model, tokenizer, clip, processor
        gc.collect(); torch.cuda.empty_cache()
    if not smoke:
        if p.read(status_path)['completed'] != len(rows) or any(not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError('Full independent source is missing/nonfinite')
        p.immutable(folder / 'verification.json', {'status': 'all_complete_verified', 'plan_hash': plan['plan_hash'],
            'sources': {name: {'sha256': p.digest(folder / (name + '.npy')), 'shape': list(array.shape), 'dtype': 'float16'}
                        for name, array in arrays.items()}})


def source_arrays(plan, dataset):
    import numpy as np
    folder = OUT / dataset / 'sources'
    verified = p.read(folder / 'verification.json')
    if verified.get('status') != 'all_complete_verified' or verified.get('plan_hash') != plan['plan_hash']:
        raise ValueError('Complete independent sources absent')
    arrays = {}
    for name, item in verified['sources'].items():
        location = folder / (name + '.npy')
        if p.digest(location) != item['sha256']:
            raise ValueError('Full source bytes changed')
        array = np.load(location, mmap_mode='r')
        if list(array.shape) != item['shape'] or array.dtype != np.float16 or not np.isfinite(array).all():
            raise ValueError('Full source shape/dtype/finite mismatch')
        arrays[name] = array
    return arrays


def train(plan, dataset):
    import run_behavior_p2_matched as implementation
    # Reuse ONLY the audited CPU/GPU training helper, not its original main,
    # queue, plan, output root, Qwen model, checkpoints or hidden dimensions.
    original_out, original_p, original_sources = implementation.OUT, implementation.p, implementation.sources
    try:
        implementation.OUT, implementation.p = OUT, p
        implementation.sources = source_arrays
        implementation.train(plan, dataset, 'behavior', 256)
    finally:
        implementation.OUT, implementation.p, implementation.sources = original_out, original_p, original_sources


def retrieve(plan, dataset, method):
    import torch
    import numpy as np
    import torch.nn.functional as F
    import run_behavior_p2_matched as implementation
    import run_dtd_learned_velocity as learned
    data = plan['datasets'][dataset]
    bank, queries = data['manifest']['bank'], data['manifest']['query']
    folder = OUT / dataset / 'test' / method
    folder.mkdir(parents=True, exist_ok=True)
    arrays = source_arrays(plan, dataset) if method != 'zero' else {}
    checkpoint = OUT / dataset / 'trained/behavior/d256/best.pt'
    context = {'plan_hash': plan['plan_hash'], 'dataset': dataset, 'method': method,
               'source_hash': p.read(OUT / dataset / 'sources/verification.json')['sources'] if method != 'zero' else None,
               'checkpoint_sha256': p.digest(checkpoint) if method == 'behavior_proj' else None,
               'candidate_ids': [r['sample_id'] for r in bank], 'query_ids': [r['sample_id'] for r in queries]}
    context['context_hash'] = p.canonical(context)
    p.immutable(folder / 'context.json', context)
    path = folder / 'selections.json'
    if path.exists():
        selected = p.read(path)
        validate_selections(data, method, selected)
        return folder, context, selected
    before = time.perf_counter()
    selected = {}
    if method == 'zero':
        selected = {row['sample_id']: {'demo_ids': [], 'demo_labels': [], 'scores': []} for row in queries}
    else:
        rep, width = ('clip', None) if method == 'rices' else ('behavior', 256 if method == 'behavior_proj' else None)
        retriever = implementation.retriever(rep, width, arrays)
        if width:
            saved = torch.load(checkpoint, map_location='cuda', weights_only=True)
            retriever.load_state_dict(saved['model'])
        # Exact FP32 encoding/scoring, unlike projection-training autocast.
        def encode(indices):
            parts = []
            with torch.inference_mode():
                for start in range(0, len(indices), 32):
                    values = torch.from_numpy(np.asarray(arrays[rep][indices[start:start+32]], dtype=np.float32)).to('cuda')
                    parts.append(retriever(values).float())
            return torch.cat(parts)
        left, right = encode(np.arange(len(bank), len(bank)+len(queries))), encode(np.arange(len(bank)))
        with torch.inference_mode():
            for start in range(0, len(queries), 32):
                p.live_guard()
                scores = retriever.score(left[start:start+32], right).float()
                top = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :4].flip(1)
                actual = torch.gather(scores, 1, top).cpu().tolist()
                for j, (indices, values) in enumerate(zip(top.cpu().tolist(), actual)):
                    demos = [bank[i] for i in indices]
                    selected[queries[start+j]['sample_id']] = {'demo_ids': [r['sample_id'] for r in demos],
                            'demo_labels': [r['label'] for r in demos], 'scores': values}
        del retriever, left, right
        gc.collect(); torch.cuda.empty_cache()
    validate_selections(data, method, selected)
    p.immutable(path, selected)
    p.immutable(folder / 'retrieval_time.json', {'seconds': time.perf_counter()-before,
        'context_hash': context['context_hash'], 'scope': 'cached full-source encoding/scoring only; not end-to-end extraction/generation'})
    return folder, context, selected


def validate_selections(data, method, selections):
    if method not in p.METHODS or list(selections) != [r['sample_id'] for r in data['manifest']['query']]:
        raise ValueError('Selection scope/order/complete test differs')
    bank = {r['sample_id']: r for r in data['manifest']['bank']}
    for selected in selections.values():
        if method == 'zero':
            if selected != {'demo_ids': [], 'demo_labels': [], 'scores': []}:
                raise ValueError('Zero arm must not receive demos or test truth')
        else:
            p.validate_top4(selected['demo_ids'], selected['scores'], bank)
            if selected['demo_labels'] != [bank[i]['label'] for i in selected['demo_ids']]:
                raise ValueError('Original true bank labels/order changed')


def prediction_prefix(plan, dataset, method, folder, context, selections, complete=False):
    data = plan['datasets'][dataset]
    queries = data['manifest']['query']
    path = folder / 'predictions.jsonl'
    rows = []
    codec = p.read(folder / 'codec.json')
    owners = {tuple(row['tokens']): row['label'] for row in codec['owners']}
    if path.exists():
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                if not line.endswith('\n'):
                    raise ValueError('Truncated scientific output; preserve and reject')
                row = json.loads(line)
                if len(rows) >= len(queries): raise ValueError('Too many official predictions')
                query = queries[len(rows)]
                selected = selections[query['sample_id']]
                tokens = row['actual_output_token_ids']
                if (row.get('query_id') != query['sample_id'] or row.get('target') != query['label']
                        or row.get('plan_hash') != plan['plan_hash'] or row.get('context_hash') != context['context_hash']
                        or any(row.get(k) != selected[k] for k in ('demo_ids', 'demo_labels', 'scores'))
                        or row.get('prediction') not in data['class_labels'] or type(row.get('correct')) is not bool
                        or row['correct'] != (row['prediction'] == query['label'])
                        or len(tokens) < 2 or tokens[-1] != codec['eos'] or codec['eos'] in tokens[:-1]
                        or owners.get(tuple(tokens[:-1])) != row['prediction']
                        or row.get('raw_answer') not in (row['prediction'], ' '+row['prediction'])
                        or row.get('input_tokens', 0) <= 0 or row.get('output_tokens') != len(tokens)
                        or row.get('seconds', -1) <= 0 or len(row.get('prompt_fingerprint', '')) != 64
                        or len(row.get('measured_native_generation_prefix', [])) > 1
                        or row.get('actual_prefix_callback_calls', 0) <= 0
                        or len(row.get('patch_counts', [])) != (1 if method == 'zero' else 5)
                        or any(not 1 <= v <= 3 for v in row['patch_counts'])
                        or row.get('visual_tokens') != sum(row['patch_counts']) * 256):
                    raise ValueError('Actual native full prediction/token/context/denominator mismatch')
                rows.append(row)
    if complete and len(rows) != len(queries):
        raise ValueError('Full official predictions incomplete; no partial accuracy')
    return rows


def generate(plan, dataset, method, smoke=False):
    import torch
    data = plan['datasets'][dataset]
    folder, context, selections = retrieve(plan, dataset, method)
    compatibility = p.read(OUT / dataset / 'compatibility_smoke.json')
    if compatibility.get('status') != 'actual_native_source_and_five_image_legal_smoke_passed' or compatibility['plan_hash'] != plan['plan_hash']:
        raise ValueError('Actual executed native compatibility smoke absent')
    model, tokenizer = load_native(plan)
    owners, eos = adapter.label_codec(tokenizer, data['class_labels'], model.conv_template.sep.strip())
    p.immutable(folder / 'codec.json', {'eos': eos, 'owners': [{'tokens': list(tokens), 'label': value} for tokens, value in owners.items()]})
    rows = prediction_prefix(plan, dataset, method, folder, context, selections)
    queries, bank = data['manifest']['query'], {r['sample_id']: r for r in data['manifest']['bank']}
    stop = min(len(queries), 1) if smoke else len(queries)
    with (folder / 'predictions.jsonl').open('a', encoding='utf-8') as handle:
        for i in range(len(rows), stop):
            query = queries[i]
            selected = selections[query['sample_id']]
            original_rows = [bank[identifier] for identifier in selected['demo_ids']] + [query]
            text = adapter.question(dataset, data['class_labels'], selected['demo_labels'])
            batch, metadata = adapter.prepare(model, tokenizer, absolute_rows(original_rows), text)
            result = adapter.generate(model, tokenizer, batch, metadata, data['class_labels'])
            output = {**result, **selected, 'query_id': query['sample_id'], 'target': query['label'],
                      'correct': result['prediction'] == query['label'], 'plan_hash': plan['plan_hash'],
                      'context_hash': context['context_hash'], 'prompt_fingerprint': p.canonical(metadata['prompt'])}
            write_row(handle, output)
            rows.append(output)
            progress(dataset, 'actual_complete_test_generation', method=method, completed=len(rows), expected=len(queries))
            del batch
    prediction_prefix(plan, dataset, method, folder, context, selections, complete=not smoke)
    if smoke:
        if not rows: raise ValueError('No actual retrieved context generation smoke')
        p.immutable(folder / 'smoke.json', {'status': 'actual_native_legal_generation_passed', 'plan_hash': plan['plan_hash'],
                    'first_query_counted_once': True, 'input_tokens': rows[0]['input_tokens'],
                    'output_tokens': rows[0]['output_tokens'], 'actual_prefix': rows[0]['measured_native_generation_prefix']})
    else:
        if not (folder / 'smoke.json').exists(): raise ValueError('Actual method-specific smoke absent')
        p.immutable(folder / 'verification.json', {'status': 'all_complete_verified', 'plan_hash': plan['plan_hash'],
                    'context_hash': context['context_hash'], 'rows': len(rows), 'predictions_sha256': p.digest(folder / 'predictions.jsonl'),
                    'correct': sum(row['correct'] for row in rows), 'actual_seconds': sum(row['seconds'] for row in rows)})
    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()


def smoke(plan, dataset):
    import torch
    captures(plan, dataset, smoke=True)
    data = plan['datasets'][dataset]
    model, tokenizer = load_native(plan)
    demos, query = data['manifest']['bank'][:4], data['manifest']['query'][0]
    text = adapter.question(dataset, data['class_labels'], [row['label'] for row in demos])
    batch, metadata = adapter.prepare(model, tokenizer, absolute_rows(demos + [query]), text)
    result = adapter.generate(model, tokenizer, batch, metadata, data['class_labels'])
    p.immutable(OUT / dataset / 'compatibility_smoke.json', {'status': 'actual_native_source_and_five_image_legal_smoke_passed',
                'plan_hash': plan['plan_hash'], 'diagnostic_demo_ids': [row['sample_id'] for row in demos],
                'diagnostic_query_id': query['sample_id'], 'result': result,
                'prompt_fingerprint': p.canonical(metadata['prompt']), 'counted_as_official_prediction': False,
                'decoder_state_shape': [32, 4096], 'native_order_and_visual_count_checked': True})
    del model, tokenizer, batch
    gc.collect(); torch.cuda.empty_cache()


def verify(plan):
    groups = []
    for dataset in p.DATASETS:
        source_arrays(plan, dataset)
        training = OUT / dataset / 'trained/behavior/d256'
        trained = p.read(training / 'training_verified.json')
        history = trained['history']
        earliest = min(history, key=lambda row: (-row['dev_top4_same_label_fraction'], row['step']))['step']
        if (p.digest(training / 'best.pt') != trained['best_sha256'] or trained['best_step'] != earliest
                or [row['step'] for row in history] != list(range(0, 1501, 100)) or trained['trainable_parameters'] != 4096*256):
            raise ValueError('New independent fixed256 training/checkpoint/full-history identity differs')
        for method in p.METHODS:
            folder = OUT / dataset / 'test' / method
            context, selected = p.read(folder / 'context.json'), p.read(folder / 'selections.json')
            if context['context_hash'] != p.canonical({k: v for k, v in context.items() if k != 'context_hash'}):
                raise ValueError('Immutable retrieval context changed')
            validate_selections(plan['datasets'][dataset], method, selected)
            rows = prediction_prefix(plan, dataset, method, folder, context, selected, complete=True)
            verified = p.read(folder / 'verification.json')
            if (verified.get('status') != 'all_complete_verified' or verified['plan_hash'] != plan['plan_hash']
                    or verified['predictions_sha256'] != p.digest(folder / 'predictions.jsonl')
                    or p.read(folder / 'smoke.json').get('status') != 'actual_native_legal_generation_passed'):
                raise ValueError('Incomplete actual method/source/generation evidence')
            groups.append({'dataset': dataset, 'method': method, 'rows': len(rows), 'predictions_sha256': verified['predictions_sha256']})
    if sum(row['rows'] for row in groups) != 93024 or len(groups) != 20:
        raise ValueError('Need every method on all five complete official tests')
    p.immutable(OUT / 'verification.json', {'status': 'all_complete_verified', 'plan_hash': plan['plan_hash'],
                'groups': groups, 'predictions': 93024,
                'next_action': 'sole watcher full arrays/checkpoints/evidence takeback then exact stop before independent CPU science audit'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('preflight', 'smoke', 'capture', 'train', 'retrieve', 'generate', 'verify'), default='preflight')
    parser.add_argument('--dataset', choices=p.DATASETS)
    parser.add_argument('--method', choices=p.METHODS)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--allow-gpu', action='store_true')
    args = parser.parse_args()
    plan = p.validate_plan(p.read(p.PLAN))
    if args.phase == 'preflight':
        print(json.dumps({'phase': 'CPU_preflight_not_cloud_compatibility', 'images': preflight(plan)}))
        return
    if args.phase == 'verify':
        verify(plan); return
    if not args.allow_gpu or not args.dataset:
        raise ValueError('Explicit independent GPU phase/dataset required')
    if args.phase in ('retrieve', 'generate') and args.method not in p.METHODS:
        raise ValueError('Explicit prespecified method required')
    p.gpu_guard(plan)
    OUT.mkdir(parents=True, exist_ok=True)
    with lock(OUT / 'scientific_single_process.lock'):
        if args.phase == 'smoke': smoke(plan, args.dataset)
        elif args.phase == 'capture':
            status = p.read(OUT / args.dataset / 'compatibility_smoke.json')
            if status.get('status') != 'actual_native_source_and_five_image_legal_smoke_passed':
                raise ValueError('Actual compatibility smoke required before full source capture')
            captures(plan, args.dataset)
        elif args.phase == 'train': train(plan, args.dataset)
        elif args.phase == 'retrieve': retrieve(plan, args.dataset, args.method)
        elif args.phase == 'generate': generate(plan, args.dataset, args.method, args.smoke)


if __name__ == '__main__': main()
