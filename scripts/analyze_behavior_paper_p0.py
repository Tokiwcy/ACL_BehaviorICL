"""CPU-only integration of three COMPLETE new zero caches with old ICL evidence.

Never read partial caches as a completed result. Never modify old sources or the
cloud's fixed collector. Reuse the original numerical analysis in this process
only, and record the new provenance and complete-case limitations explicitly.
"""
import argparse
import json
import math
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_behavior_priority_analysis as analysis
from collect_behavior_priority_zero import canonical, digest, same_persisted_identity

NEW_DATASETS=('cub','pets','dogs')


def validated_rows(raw, samples, labels, job_hash, image_hashes):
    """Pure validation: legality is exact, and illegal text remains untouched."""
    if len(raw)!=len(samples): raise ValueError('Incomplete full bank/test zero cache')
    legal={label.casefold():label for label in labels}
    if len(legal)!=len(labels): raise ValueError('Casefold-colliding legal labels')
    out={}
    for r,s in zip(raw,samples):
        if (r['sample_id'],r['true_label'],r['split'],r['job_hash'])!=(s['sample_id'],s['label'],s['split'],job_hash):
            raise ValueError('New zero sample/label/split/order/job mismatch')
        if r['sample_id'] in out: raise ValueError('Duplicate zero sample')
        text=r['raw_zero_shot_answer']
        if not isinstance(text,str): raise ValueError('Missing literal answer')
        pred=legal.get(text.strip().casefold())
        if (r['zero_shot_pred']!=pred or type(r['zero_shot_valid']) is not bool
                or type(r['zero_shot_correct']) is not bool
                or r['zero_shot_valid']!=(pred is not None)
                or r['zero_shot_correct']!=(pred==s['label'])):
            raise ValueError('Predicted label/legality/correctness not derived from literal answer')
        if (type(r['output_tokens']) is not int or r['output_tokens']<=0
                or not isinstance(r['elapsed_seconds'],(int,float))
                or not math.isfinite(r['elapsed_seconds']) or r['elapsed_seconds']<=0):
            raise ValueError('No real generated output/token/time evidence')
        if not re.fullmatch('[0-9a-f]{64}',r['image_sha256']) or r['image_sha256']!=image_hashes[s['sample_id']]:
            raise ValueError('Official image bytes differ from capture')
        out[s['sample_id']]={'sample_id':s['sample_id'],'true_label':s['label'],'split':s['split'],
                            'raw_zero_shot_answer':text,'zero_shot_pred':pred,
                            'zero_shot_valid':pred is not None,'zero_shot_correct':pred==s['label'],
                            'decoding':'unconstrained_greedy_max_new_tokens12'}
    return out


def load_complete_new(folder, manifest, expected, image_hashes, provenance):
    saved=analysis.read(folder/'identity.json')
    if not same_persisted_identity(saved,expected): raise ValueError('New collector scientific identity differs from current audited source/code')
    no_hash={k:v for k,v in saved.items() if k!='job_hash'}
    if canonical(no_hash)!=saved['job_hash']: raise ValueError('Invalid job hash')
    if analysis.identity(analysis.read(folder/'manifest.json'))!=analysis.identity(manifest):
        raise ValueError('New cache manifest differs from original official cohort')
    verification=analysis.read(folder/'verification.json'); path=folder/'zero_outputs.jsonl'
    samples=manifest['bank']+manifest['query']
    if (verification.get('status')!='all_complete_verified' or verification.get('rows')!=len(samples)
            or verification.get('job_hash')!=saved['job_hash']
            or verification.get('outputs_sha256')!=digest(path)):
        raise ValueError('Missing/changed full capture verification')
    runtime=analysis.read(folder/'runtime.json')
    if not str(runtime.get('torch','')).startswith('2.8.0') or runtime.get('transformers')!='5.13.0':
        raise ValueError('New zero environment differs from pinned capture')
    data=path.read_text(encoding='utf-8')
    if not data.endswith('\n'): raise ValueError('Truncated output; preserve evidence, no repair')
    rows=[json.loads(line) for line in data.splitlines()]
    result=validated_rows(rows,samples,saved['labels'],saved['job_hash'],image_hashes)
    for name in ('identity.json','manifest.json','verification.json','runtime.json','zero_outputs.jsonl'):
        provenance[str((folder/name).relative_to(ROOT))]=digest(folder/name)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--zero-root',type=Path,required=True,help='Local verified snapshot results/behavior_priority_zero_v1')
    p.add_argument('--output-root',type=Path,required=True,help='Fresh derived analysis directory; no overwrite')
    args=p.parse_args(); zero_root=args.zero_root.resolve(); out=args.output_root.resolve()
    if not zero_root.is_relative_to(ROOT/'results') or not out.is_relative_to(ROOT/'analysis'):
        raise ValueError('Inputs/results and derived analysis must stay in the workspace')
    if (out/'audit.json').exists(): raise ValueError('Use a fresh output root; no overwrite')
    # CPU image SHA and complete-cache pre-audit happens before any derived files.
    from collect_behavior_priority_zero import prepare
    sources=analysis.source_catalog(); replacements={}; extra_provenance={}; cache_specs={}
    for dataset in NEW_DATASETS:
        samples,local_manifest,_,expected=prepare(dataset,ROOT/'datasets',ROOT/'.hf_cache')
        manifest=analysis.read(sources[dataset]['base']/'manifest.json')
        images={s.sample_id:digest(s.path) for s in samples}
        replacements[sources[dataset]['base'].resolve()]=load_complete_new(
            zero_root/dataset,manifest,expected,images,extra_provenance)
        cache_specs[dataset]={'job_hash':expected['job_hash'],'rows':len(samples),
                             'folder':str((zero_root/dataset).relative_to(ROOT)),
                             'image_SHA_check':'all local official image bytes matched capture'}
        print('Complete new zero cache audited on CPU: '+dataset,flush=True)
    original=analysis.load_zero
    def load_zero(folder,manifest,provenance):
        replacement=replacements.get(folder.resolve())
        if replacement is not None:
            provenance.update(extra_provenance)
            return replacement
        return original(folder,manifest,provenance)
    prior_argv=sys.argv; analysis.load_zero=load_zero
    try:
        sys.argv=[str(ROOT/'scripts/run_behavior_priority_analysis.py'),'--output-root',str(out)]
        analysis.main()
    finally:
        analysis.load_zero=original; sys.argv=prior_argv
    audit=analysis.read(out/'audit.json')
    audit.update({'experiment':'behavior_paper_full_zero_evidence_v1','zero_cache_scope':list(analysis.DATASETS),
                  'new_complete_zero_caches':cache_specs,
                  'P0_full_requirement':'All five full bank/test zero caches verified; primary association remains conditional on common legal query/demo complete cases, not all initial errors',
                  'environment_equivalence_between_old_and_new_zero_caches':'not asserted; pinned model/prompt/decoding/official IDs match, new runtime recorded',
                  'analysis_integration_script_sha256':digest(Path(__file__)),
                  'corrective_label_intervention':'not part of this observational analysis; separate paired experiment required'})
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__': main()
