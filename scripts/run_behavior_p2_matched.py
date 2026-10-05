"""New matched P2 GPU runner; explicit independent cloud gate, CPU default.

Never imports an old scheduler or reuses old anchor arrays/checkpoints. Stages
are serial. Official test generation requires all 60 devs and 15 width locks.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import time
import behavior_p2_matched_policy as p
from collect_behavior_priority_zero import lock

ROOT = p.ROOT
OUT = ROOT / 'results' / p.SCIENCE


def now(): return datetime.now(timezone.utc).isoformat()


def progress(folder, stage, **values):
    folder.mkdir(parents=True, exist_ok=True)
    p.atomic(folder / 'progress.json', {'utc': now(), 'stage': stage, 'pid': os.getpid(), **values})


def write_row(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False) + '\n'); handle.flush(); os.fsync(handle.fileno())


def sample(row):
    from multidataset_protocol import Sample
    return Sample(row['sample_id'], str(ROOT / row['path']), row['label'], row['split'])


def preflight(plan, dataset):
    data = plan['datasets'][dataset]; manifest = data['manifest']
    p.validate_split(manifest, data['split'])
    from multidataset_protocol import load_dataset
    from run_behavior_priority_analysis import identity
    from dataclasses import asdict
    bank, query = load_dataset(dataset, ROOT / 'datasets')
    if identity({'bank': [asdict(s) for s in bank], 'query': [asdict(s) for s in query]}) != identity(manifest):
        raise ValueError('Official ID/label/split/order differs')
    for row in manifest['bank'] + manifest['query']:
        if p.digest(ROOT / row['path']) != row['image_sha256']: raise ValueError('Image SHA changed: ' + row['sample_id'])
    return data


def capture(plan, dataset, smoke=False):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    from multimodal_model_adapter import load_model, MODEL_SPECS, zero_shot_messages, prepare_batch, model_dimensions
    data = preflight(plan, dataset); rows = data['manifest']['bank'] + data['manifest']['query']
    dest = OUT / dataset / 'sources'; dest.mkdir(parents=True, exist_ok=True)
    p.immutable(dest / 'identity.json', {'plan_hash': plan['plan_hash'], 'dataset': dataset,
        'manifest_hash': p.canonical(data['manifest']), 'source_origin': 'new_independent_P2_extraction'})
    if (dest / 'verification.json').exists(): return sources(plan, dataset)
    arrays = {}
    shapes = {'clip': (len(rows), 1, 512), 'qwenvision': (len(rows), 1, 1024), 'behavior': (len(rows), 36, 2560)}
    completed = {}
    for rep, shape in shapes.items():
        path = dest / (rep + '.npy'); pp = dest / (rep + '_progress.json')
        old = p.read(pp) if pp.exists() else None
        if path.exists() != bool(old): raise ValueError('Unpaired array/progress; preserve failure evidence')
        dtype = p.PROTOCOL['cache_dtype'][rep]
        if old and (old['plan_hash'] != plan['plan_hash'] or old['shape'] != list(shape) or old['dtype'] != dtype):
            raise ValueError('Cache prefix context/shape/dtype changed')
        arrays[rep] = np.lib.format.open_memmap(path, mode='r+' if old else 'w+', dtype=dtype, shape=shape)
        completed[rep] = old['completed'] if old else 0
        if not 0 <= completed[rep] <= len(rows): raise ValueError('Invalid source prefix length')
        if not old: p.atomic(pp, {'plan_hash': plan['plan_hash'], 'completed': 0, 'shape': list(shape), 'dtype': dtype})
    if completed['behavior'] != completed['qwenvision']: raise ValueError('Paired Qwen source prefix mismatch')
    for name, count in [('clip_capture', completed['clip']), ('qwen_capture', completed['behavior'])]:
        logpath = dest / (name + '.jsonl')
        logs = []
        if logpath.exists():
            with logpath.open(encoding='utf-8') as handle:
                for line in handle:
                    if not line.endswith('\n'): raise ValueError('Truncated source log; preserve evidence')
                    logs.append(json.loads(line))
        if len(logs) != count or any((r['sample_id'], r['image_sha256'], r['plan_hash']) !=
                (rows[i]['sample_id'], rows[i]['image_sha256'], plan['plan_hash']) for i, r in enumerate(logs)):
            raise ValueError('Source prefix row/order/image/log disagrees with committed progress')
    target = 1 if smoke else len(rows)
    if completed['clip'] < target:
        processor = CLIPProcessor.from_pretrained(p.CLIP_MODEL, revision=p.CLIP_REVISION, cache_dir=ROOT / '.hf_cache', local_files_only=True)
        model = CLIPModel.from_pretrained(p.CLIP_MODEL, revision=p.CLIP_REVISION, cache_dir=ROOT / '.hf_cache', dtype=torch.float32, local_files_only=True).eval().to('cuda')
        with (dest / 'clip_capture.jsonl').open('a', encoding='utf-8') as handle, torch.inference_mode():
            for i in range(completed['clip'], target):
                row = rows[i]; t = time.perf_counter()
                with Image.open(ROOT / row['path']) as image: inputs = processor(images=[image.convert('RGB')], return_tensors='pt')
                value = model.get_image_features(pixel_values=inputs['pixel_values'].to('cuda'))
                if not torch.is_tensor(value): value = value.pooler_output
                value = F.normalize(value.float(), dim=-1)
                if list(value.shape) != [1, 512] or not torch.isfinite(value).all(): raise ValueError('CLIP actual feature smoke failed')
                arrays['clip'][i, 0] = value[0].cpu().numpy(); arrays['clip'].flush(); torch.cuda.synchronize()
                write_row(handle, {'sample_id': row['sample_id'], 'image_sha256': row['image_sha256'],
                    'plan_hash': plan['plan_hash'], 'seconds': time.perf_counter() - t})
                p.atomic(dest / 'clip_progress.json', {'plan_hash': plan['plan_hash'], 'shape': list(shapes['clip']),
                    'dtype': 'float32', 'completed': i + 1})
                if (i + 1) % 100 == 0: print(f'{dataset} CLIP {i+1}/{len(rows)}', flush=True)
        del model, processor; gc.collect(); torch.cuda.empty_cache()
    if completed['behavior'] < target:
        model, processor = load_model(MODEL_SPECS['qwen3vl4b'], ROOT / '.hf_cache', None)
        if model_dimensions(model) != (36, 2560): raise ValueError('Actual decoder hidden dimensions changed')
        tokens = []
        def final_vision_hook(module, args, result):
            value = result[0] if isinstance(result, tuple) else result
            if value.ndim != 2 or value.shape[-1] != 1024: raise ValueError('Unexpected final pre-merger vision token shape')
            tokens.append((value.detach().float().mean(0), int(value.shape[0])))
        hook = model.model.visual.blocks[-1].register_forward_hook(final_vision_hook)
        with (dest / 'qwen_capture.jsonl').open('a', encoding='utf-8') as handle, torch.inference_mode():
            for i in range(completed['behavior'], target):
                row = rows[i]; t = time.perf_counter(); tokens.clear()
                inputs = prepare_batch(processor, [zero_shot_messages(sample(row), dataset, data['class_labels'], 224)])
                output = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True, logits_to_keep=1)
                if len(tokens) != 1 or output.hidden_states is None or len(output.hidden_states) != 37:
                    raise ValueError('Actual paired vision/decoder capture failed')
                states = torch.stack([v[0, -1].detach() for v in output.hidden_states[1:]])
                pooled, vision_n = tokens[0]
                if list(states.shape) != [36, 2560] or not torch.isfinite(states).all() or not torch.isfinite(pooled).all():
                    raise ValueError('Nonfinite/wrong actual representation')
                expected_n = int(inputs['image_grid_thw'].prod(-1).sum())
                if vision_n != expected_n: raise ValueError('Final vision tokens do not match native pre-merger grid')
                arrays['behavior'][i] = states.float().cpu().numpy().astype(np.float16)
                arrays['qwenvision'][i, 0] = pooled.cpu().numpy().astype(np.float16)
                for rep in ('behavior', 'qwenvision'): arrays[rep].flush()
                torch.cuda.synchronize()
                write_row(handle, {'sample_id': row['sample_id'], 'image_sha256': row['image_sha256'],
                    'plan_hash': plan['plan_hash'], 'input_tokens': int(inputs['input_ids'].shape[1]),
                    'native_vision_token_count': vision_n, 'native_grid': inputs['image_grid_thw'].cpu().tolist(),
                    'seconds_paired_capture': time.perf_counter() - t})
                for rep in ('behavior', 'qwenvision'):
                    p.atomic(dest / (rep + '_progress.json'), {'plan_hash': plan['plan_hash'], 'shape': list(shapes[rep]),
                        'dtype': 'float16', 'completed': i + 1})
                del inputs, output, states, pooled; tokens.clear()
                if (i + 1) % 100 == 0: print(f'{dataset} Qwen paired sources {i+1}/{len(rows)}', flush=True)
        hook.remove(); del model, processor; gc.collect(); torch.cuda.empty_cache()
    if smoke:
        p.atomic(dest / 'smoke.json', {'status': 'actual_three_representations_passed', 'plan_hash': plan['plan_hash'],
            'sample_id': rows[0]['sample_id'], 'CLIP_shape': [1,512], 'vision_shape': [1,1024], 'behavior_shape': [36,2560], 'utc': now()})
        return
    result = {}
    for rep, array in arrays.items():
        if not np.isfinite(array).all(): raise ValueError('Nonfinite complete source: ' + rep)
        result[rep] = {'sha256': p.digest(dest / (rep + '.npy')), 'shape': list(array.shape), 'dtype': str(array.dtype), 'rows': len(rows)}
    p.atomic(dest / 'verification.json', {'status': 'all_complete_verified', 'plan_hash': plan['plan_hash'], 'sources': result})
    return result


def sources(plan, dataset):
    import numpy as np
    dest = OUT / dataset / 'sources'; value = p.read(dest / 'verification.json')
    if value['status'] != 'all_complete_verified' or value['plan_hash'] != plan['plan_hash']: raise ValueError('Unverified source context')
    arrays = {}
    for rep in p.REPRESENTATIONS:
        path = dest / (rep + '.npy'); record = value['sources'][rep]
        if p.digest(path) != record['sha256']: raise ValueError('Complete source SHA changed')
        array = np.load(path, mmap_mode='r')
        if list(array.shape) != record['shape'] or str(array.dtype) != record['dtype']: raise ValueError('Source shape changed')
        arrays[rep] = array
    return arrays


def retriever(rep, width, arrays):
    import torch
    import torch.nn.functional as F
    from run_dtd_learned_velocity import StateRetriever
    torch.manual_seed(73)
    class Raw(StateRetriever):
        def forward(self, states): return F.normalize(states.float(), dim=-1)
    cls = StateRetriever if width else Raw
    model = cls(arrays[rep].shape[2], arrays[rep].shape[1], width).to('cuda')
    model.layer_logits.requires_grad_(False); model.layer_logits.data.zero_()
    return model


def save_torch(path, value):
    import torch
    temp = path.with_suffix('.pt.tmp'); torch.save(value, temp); temp.replace(path)


def train(plan, dataset, rep, width):
    import numpy as np
    import torch
    import run_dtd_learned_velocity as learned
    data = plan['datasets'][dataset]; arrays = sources(plan, dataset); states = arrays[rep]
    bank = data['manifest']['bank']; index = {s['sample_id']: i for i, s in enumerate(bank)}
    train_ids = np.asarray([index[i] for i in data['split']['train_ids']]); dev_ids = np.asarray([index[i] for i in data['split']['development_ids']])
    labels = [s['label'] for s in bank]; label_index = {s: i for i, s in enumerate(sorted(set(labels)))}
    by_label = {v: np.asarray([i for i in train_ids if label_index[labels[i]] == v]) for v in label_index.values()}
    folder = OUT / dataset / 'trained' / rep / ('d' + str(width)); folder.mkdir(parents=True, exist_ok=True)
    context = {'plan_hash': plan['plan_hash'], 'source_sha256': p.read(OUT / dataset / 'sources/verification.json')['sources'][rep]['sha256'],
        'dataset': dataset, 'representation': rep, 'width': width, 'split_hash': p.canonical(data['split'])}
    p.immutable(folder / 'context.json', context)
    if (folder / 'training_verified.json').exists():
        verified = p.read(folder / 'training_verified.json')
        if verified['context_hash'] != p.canonical(context) or p.digest(folder / 'best.pt') != verified['best_sha256']:
            raise ValueError('Completed checkpoint changed')
        return
    model = retriever(rep, width, arrays); params = [v for v in model.parameters() if v.requires_grad]
    expected = states.shape[2] * width
    if sum(v.numel() for v in params) != expected: raise ValueError('Unexpected trained layer weights/bias/parameter count')
    optimizer = torch.optim.AdamW(params, lr=.0003, weight_decay=.01, betas=(.9,.999))
    rng = np.random.default_rng(10073); start = 0; history = []; best_metric = -1.; best_step = 0
    best_state = None; training_seconds = 0.; current = folder / 'last.pt'
    if current.exists():
        saved = torch.load(current, map_location='cuda', weights_only=True)
        if saved['context'] != context: raise ValueError('Resume checkpoint context changed')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['numpy_rng']; torch.set_rng_state(saved['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])
        start, history, best_metric, best_step = saved['step'], saved['history'], saved['best_metric'], saved['best_step']
        best_state, training_seconds = saved['best_state'], saved['training_seconds']
    before = time.perf_counter()
    for step in range(start, 1501):
        loss_value = None
        if step > start:
            indices, targets = learned.balanced_batch(by_label, 16, 4, rng)
            batch = torch.from_numpy(np.asarray(states[indices], dtype=np.float32)).to('cuda')
            target = torch.as_tensor(targets, device='cuda'); model.train()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                encoded = model(batch); loss = learned.supervised_contrastive_loss(model.score(encoded, encoded), target, .07)
            if not torch.isfinite(loss): raise ValueError('Nonfinite matched contrastive loss')
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); loss_value = float(loss.detach())
            del batch, target, encoded, loss
        if step % 100 == 0:
            if history and history[-1]['step'] == step: continue
            metric = learned.retrieval_metrics(model, states, train_ids, dev_ids, labels, 4)['top4_same_label_fraction']
            history.append({'step': step, 'loss': loss_value, 'dev_top4_same_label_fraction': metric})
            if metric > best_metric:
                best_metric, best_step = metric, step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.cuda.synchronize()
            save_torch(current, {'context': context, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'step': step, 'history': history, 'best_metric': best_metric, 'best_step': best_step, 'best_state': best_state,
                'numpy_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'training_seconds': training_seconds + time.perf_counter() - before})
            progress(folder, 'training', completed_steps=step, expected_steps=1500, trainable_parameters=expected)
            print(f'{dataset}/{rep}/{width} step {step}/1500 dev_top4={metric:.6f}', flush=True)
    if [r['step'] for r in history] != list(range(0, 1501, 100)): raise ValueError('Incomplete training history')
    if best_state is None or any(not torch.isfinite(v).all() for v in best_state.values()):
        raise ValueError('Invalid best checkpoint tensors')
    save_torch(folder / 'best.pt', {'context': context, 'step': best_step, 'model': best_state})
    p.atomic(folder / 'training_verified.json', {'context_hash': p.canonical(context), 'best_sha256': p.digest(folder / 'best.pt'),
        'best_step': best_step, 'history': history, 'trainable_parameters': expected, 'training_seconds': training_seconds + time.perf_counter() - before})
    del model, optimizer; gc.collect(); torch.cuda.empty_cache()


def retrieve(plan, dataset, rep, width, phase):
    import numpy as np
    import torch
    import run_dtd_learned_velocity as learned
    data = plan['datasets'][dataset]; bank = data['manifest']['bank']; arrays = sources(plan, dataset)
    index = {s['sample_id']: i for i, s in enumerate(bank)}
    if phase == 'development':
        queries = [bank[index[i]] for i in data['split']['development_ids']]
        candidates = [index[i] for i in data['split']['train_ids']]; query_indices = [index[r['sample_id']] for r in queries]
    else:
        queries = data['manifest']['query']; candidates = list(range(len(bank))); query_indices = list(range(len(bank), len(bank) + len(queries)))
    method = ('supcon_' + rep if rep != 'behavior' else 'behavior_proj') if width else ('raw_' + rep if rep != 'behavior' else 'behavior_raw')
    folder = OUT / dataset / phase / method / ('d' + str(width) if width else 'raw')
    folder.mkdir(parents=True, exist_ok=True)
    context = {'plan_hash': plan['plan_hash'], 'dataset': dataset, 'phase': phase, 'method': method, 'width': width,
        'source_hash': p.read(OUT / dataset / 'sources/verification.json')['sources'][rep]['sha256'],
        'candidate_ids': [bank[i]['sample_id'] for i in candidates], 'query_ids': [r['sample_id'] for r in queries],
        'checkpoint_SHA': p.digest(OUT / dataset / 'trained' / rep / ('d' + str(width)) / 'best.pt') if width else None}
    context['context_hash'] = p.canonical(context); p.immutable(folder / 'context.json', context)
    selection_path = folder / 'selections.json'
    if selection_path.exists():
        selections = p.read(selection_path)
        if list(selections) != context['query_ids']: raise ValueError('Selection order/coverage changed')
        for s in selections.values(): p.validate_top4(s['demo_ids'], s['scores'], context['candidate_ids'])
        return folder, queries, selections, context
    before = time.perf_counter(); model = retriever(rep, width, arrays)
    if width:
        saved = torch.load(OUT / dataset / 'trained' / rep / ('d' + str(width)) / 'best.pt', map_location='cuda', weights_only=True)
        model.load_state_dict(saved['model'])
    model.eval()
    left = learned.encode_indices(model, arrays[rep], np.asarray(query_indices))
    right = learned.encode_indices(model, arrays[rep], np.asarray(candidates))
    selections = {}
    with torch.inference_mode():
        for start in range(0, len(queries), 32):
            scores = model.score(left[start:start+32], right).float()
            # Stable tie order fixed before outcomes; retain actual scores, never ranks as scores.
            top = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :4].flip(1)
            values = torch.gather(scores, 1, top).cpu().tolist(); top = top.cpu().tolist()
            for j, (indices, actual) in enumerate(zip(top, values)):
                demos = [bank[candidates[i]] for i in indices]
                selections[queries[start+j]['sample_id']] = {'demo_ids': [r['sample_id'] for r in demos],
                    'demo_labels': [r['label'] for r in demos], 'scores': actual}
                p.validate_top4(selections[queries[start+j]['sample_id']]['demo_ids'], actual, context['candidate_ids'])
    torch.cuda.synchronize(); p.immutable(selection_path, selections)
    p.atomic(folder / 'retrieval_time.json', {'seconds': time.perf_counter()-before, 'query_count': len(queries),
        'includes': 'retriever encoding and scoring, excludes frozen source extraction; offline cached retrieval, not end-to-end online cost'})
    del model, left, right; gc.collect(); torch.cuda.empty_cache()
    return folder, queries, selections, context


def generate(plan, dataset, folder, queries, selections, context, limit=None):
    import torch
    from multimodal_model_adapter import load_model, MODEL_SPECS, icl_messages, prepare_batch
    from legal_label_decoding import LegalLabelTrie, label_token_sequences, exact_generated_label
    from run_behavior_priority_analysis import majority, weighted
    data = plan['datasets'][dataset]; labels = data['class_labels']; bank = {r['sample_id']: r for r in data['manifest']['bank']}
    path = folder / 'predictions.jsonl'; rows = p.prediction_prefix(path, queries, selections, labels, context['context_hash'])
    if len(rows) == len(queries): return rows
    model, processor = load_model(MODEL_SPECS['qwen3vl4b'], ROOT / '.hf_cache', None)
    sequences, max_tokens = label_token_sequences(processor.tokenizer, labels); trie = LegalLabelTrie(sequences, processor.tokenizer.eos_token_id)
    stop = len(queries) if limit is None else min(len(queries), len(rows)+limit)
    with path.open('a', encoding='utf-8') as handle, torch.inference_mode():
        for i in range(len(rows), stop):
            row = queries[i]; selected = selections[row['sample_id']]; demos = [sample(bank[v]) for v in selected['demo_ids']]
            before = time.perf_counter(); inputs = prepare_batch(processor, [icl_messages(demos, sample(row), dataset, labels, 224)])
            n = int(inputs['input_ids'].shape[1])
            generated = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, use_cache=True,
                prefix_allowed_tokens_fn=trie.prefix_allowed_tokens_fn(n))
            torch.cuda.synchronize(); tail = generated[:, n:]; raw = processor.batch_decode(tail, skip_special_tokens=True)[0].strip()
            prediction = exact_generated_label(raw, labels)
            if prediction not in labels: raise ValueError('Constrained generation smoke produced illegal label')
            output = {'query_id': row['sample_id'], 'target': row['label'], 'prediction': prediction, 'correct': prediction == row['label'],
                'raw_answer': raw, 'demo_ids': selected['demo_ids'], 'demo_labels': selected['demo_labels'], 'scores': selected['scores'],
                'majority_knn': majority(selected['demo_labels'], selected['scores']), 'weighted_knn': weighted(selected['demo_labels'], selected['scores']),
                'input_tokens': n, 'output_tokens': int(tail.shape[1]), 'seconds': time.perf_counter()-before,
                'context_hash': context['context_hash'], 'plan_hash': plan['plan_hash']}
            write_row(handle, output); rows.append(output)
            progress(folder, 'ICL_generation', completed=len(rows), expected=len(queries))
            if len(rows) % 25 == 0: print(f'{dataset}/{context["method"]}/{context["phase"]} {len(rows)}/{len(queries)}', flush=True)
            del inputs, generated, tail
    del model, processor; gc.collect(); torch.cuda.empty_cache()
    if len(rows) == len(queries):
        p.prediction_prefix(path, queries, selections, labels, context['context_hash'], complete=True)
        p.atomic(folder / 'verification.json', {'status': 'all_complete_verified', 'context_hash': context['context_hash'],
            'predictions_sha256': p.digest(path), 'selections_sha256': p.digest(folder / 'selections.json'),
            'rows': len(rows), 'correct': sum(r['correct'] for r in rows), 'total_ICL_seconds': sum(r['seconds'] for r in rows)})
    return rows


def dev(plan, dataset, rep, width):
    folder, queries, selections, context = retrieve(plan, dataset, rep, width, 'development')
    if not (folder / 'smoke.json').exists():
        rows = generate(plan, dataset, folder, queries, selections, context, 1)
        p.atomic(folder / 'smoke.json', {'status': 'actual_four_demo_legal_generation_passed', 'utc': now(),
            'input_tokens': rows[0]['input_tokens'], 'output_tokens': rows[0]['output_tokens'], 'context_hash': context['context_hash']})
    generate(plan, dataset, folder, queries, selections, context)


def lock_widths(plan):
    locks = {}
    for dataset in p.DATASETS:
        for rep in p.REPRESENTATIONS:
            method = 'supcon_' + rep if rep != 'behavior' else 'behavior_proj'; records = []
            for width in p.WIDTHS:
                folder = OUT / dataset / 'development' / method / ('d' + str(width))
                verification = p.read(folder / 'verification.json'); context = p.read(folder / 'context.json')
                if context['plan_hash'] != plan['plan_hash'] or verification['status'] != 'all_complete_verified': raise ValueError('Wrong/incomplete dev identity')
                if p.digest(folder / 'predictions.jsonl') != verification['predictions_sha256']: raise ValueError('Dev output SHA changed')
                records.append({'width': width, 'correct': verification['correct'], 'total': verification['rows'],
                    'verification_sha256': p.digest(folder / 'verification.json')})
            value = {'plan_hash': plan['plan_hash'], 'dataset': dataset, 'representation': rep,
                'selected_width': p.select_width(records), 'all_four_dev_records': records}
            path = OUT / dataset / 'width_locks' / (rep + '.json'); p.immutable(path, value)
            locks[dataset + '/' + rep] = value
    p.immutable(OUT / 'all_width_locks.json', {'plan_hash': plan['plan_hash'], 'locks': locks})
    return locks


def test(plan, dataset):
    locks = lock_widths(plan)  # Refuses official test until every 60 full dev is present/unchanged.
    for rep in p.REPRESENTATIONS:
        for width in (None, locks[dataset + '/' + rep]['selected_width']):
            folder, queries, selections, context = retrieve(plan, dataset, rep, width, 'test')
            generate(plan, dataset, folder, queries, selections, context)


def verify(plan):
    locks = lock_widths(plan); verified = []
    for dataset in p.DATASETS:
        sources(plan, dataset)
        for rep in p.REPRESENTATIONS:
            for width in (None, locks[dataset + '/' + rep]['selected_width']):
                method = ('supcon_'+rep if rep != 'behavior' else 'behavior_proj') if width else ('raw_'+rep if rep != 'behavior' else 'behavior_raw')
                folder = OUT / dataset / 'test' / method / ('d'+str(width) if width else 'raw')
                context = p.read(folder / 'context.json'); selections = p.read(folder / 'selections.json')
                rows = p.prediction_prefix(folder / 'predictions.jsonl', plan['datasets'][dataset]['manifest']['query'], selections,
                    plan['datasets'][dataset]['class_labels'], context['context_hash'], complete=True)
                old = p.read(folder / 'verification.json')
                if old['predictions_sha256'] != p.digest(folder / 'predictions.jsonl') or context['plan_hash'] != plan['plan_hash']:
                    raise ValueError('Completed test context/SHA changed')
                verified.append({'dataset': dataset, 'method': method, 'rows': len(rows), 'predictions_SHA': old['predictions_sha256']})
    p.atomic(OUT / 'verification.json', {'status': 'all_complete_verified', 'plan_hash': plan['plan_hash'], 'full_groups': verified})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('preflight','capture','train','development','lock-widths','test','verify'), default='preflight')
    parser.add_argument('--dataset', choices=p.DATASETS); parser.add_argument('--representation', choices=p.REPRESENTATIONS)
    parser.add_argument('--width', choices=p.WIDTHS, type=int); parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--allow-gpu', action='store_true'); args = parser.parse_args()
    plan = p.validate_plan(p.read(p.PLAN_PATH))
    if args.phase == 'preflight':
        for d in ([args.dataset] if args.dataset else p.DATASETS): preflight(plan, d)
        print(json.dumps({'status': 'CPU_images_split_code_preflight_passed', 'plan_hash': plan['plan_hash']})); return
    if args.phase not in ('lock-widths', 'verify'):
        if not args.allow_gpu: raise ValueError('Explicit independent GPU execution flag required')
        p.gpu_guard(plan)
    OUT.mkdir(parents=True, exist_ok=True)
    with lock(OUT / 'scientific_single_process.lock'):
        if args.phase == 'capture': capture(plan, args.dataset, args.smoke)
        elif args.phase == 'train': train(plan, args.dataset, args.representation, args.width)
        elif args.phase == 'development': dev(plan, args.dataset, args.representation, args.width)
        elif args.phase == 'lock-widths': lock_widths(plan)
        elif args.phase == 'test': test(plan, args.dataset)
        elif args.phase == 'verify': verify(plan)


if __name__ == '__main__': main()
