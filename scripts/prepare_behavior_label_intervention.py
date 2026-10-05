"""Lock a CPU-only paired-label plan before seeing the new P0 outputs.

No model execution, no new retrieval, no query-label-dependent selection.
Gold/reference and wrong-label interventions share the exact existing Proj256
demo images and order. Own zero labels attach only after full zero-cache audit.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_behavior_priority_analysis as a
from collect_behavior_priority_zero import atomic,canonical,digest


def key(dataset,sample_id,seed=73):
    return hashlib.sha256(f'{dataset}:{sample_id}:{seed}'.encode()).hexdigest()


def pick_queries(dataset,queries,n=256):
    return sorted(queries,key=lambda q:(key(dataset,q['sample_id']),q['sample_id']))[:n]


def wrong_label(dataset,demo_id,gold,labels):
    alternatives=[v for v in sorted(labels) if v!=gold]
    if not alternatives or gold not in labels: raise ValueError('Invalid gold/class inventory')
    return alternatives[int(key(dataset,demo_id),16)%len(alternatives)]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output-root',type=Path,default=ROOT/'analysis/paired_label_plan_v1')
    args=p.parse_args(); out=args.output_root.resolve()
    if not out.is_relative_to(ROOT/'analysis'): raise ValueError('Derived plan stays under analysis/')
    if out.exists(): raise ValueError('Existing query/arm lock; never overwrite after observing new outputs')
    sources=a.source_catalog(); provenance={}; rows=[]; counts={}
    for dataset in a.DATASETS:
        source=sources[dataset]; manifest=a.read(source['base']/'manifest.json')
        group=a.load_group(source['proj'],'projection_only',manifest,provenance)
        labels=sorted({s['label'] for s in manifest['bank']})
        chosen=pick_queries(dataset,manifest['query'])
        counts[dataset]=len(chosen)
        for q in chosen:
            r=group[q['sample_id']]
            rows.append({'dataset':dataset,'query_id':q['sample_id'],'query_hash':key(dataset,q['sample_id']),
                         'target_for_evaluation_only':q['label'],
                         'demo_ids_in_actual_prompt_order':r['demo_ids'],
                         'gold_labels':r['demo_labels'],
                         'hashed_wrong_labels':[wrong_label(dataset,demo,label,labels)
                                                for demo,label in zip(r['demo_ids'],r['demo_labels'])],
                         'demo_own_zero_labels':None,'own_arm_status':'pending_full_zero_audit',
                         'legal_classes':labels})
        provenance[str((source['base']/'manifest.json').relative_to(ROOT))]=digest(source['base']/'manifest.json')
    spec={'experiment':'behavior_corrective_label_paired_v1','model':a.MODEL,
          'revision':'ebb281ec70b05090aa6165b016eac8ec08e71b17','cohort':a.COHORT,
          'retriever':'existing_projection_only_fixed256','seed':73,'max_queries_per_dataset':256,
          'query_selection':'sha256(dataset:sample_id:73) ascending; no label/correctness/rescue filtering',
          'all_queries':rows,'source_sha256':provenance,
          'paired_arms':['same_constrained_zero','gold_demo_labels','demo_own_legal_zero_labels','hashed_wrong_demo_labels'],
          'prompt_vision_demo_order':'same original adapter/vision and actual four demo order; no text/visual budget mutation',
          'decoding':'greedy legal-label constraint; identical per arm and zero reference',
          'gold_reference':'rerun in the same current environment, not historical gold predictions',
          'own_arm_invalid_policy':'mark missing if any demo zero is illegal; never repair; keep full preselected query set for gold and hashed-wrong',
          'same_failure_analysis':'query true label only for post-hoc subgroup definition, not retrieval or generation signals',
          'outcomes':['total_accuracy','initial_wrong_rescue','initial_correct_harm','number_of_labels_changed','actual_tokens_and_time'],
          'paired_CI':'query-level 10000 paired bootstrap; report missingness and subgroup denominators',
          'causal_scope':'effect of label content within fixed retrieved images/context; not unique causality of same-failure retrieval',
          'plan_script_sha256':digest(Path(__file__))}
    spec['plan_hash']=canonical(spec)
    out.mkdir(parents=True)
    atomic(out/'query_arm_lock.json',spec)
    atomic(out/'status.json',{'stage':'CPU_plan_locked_no_model_executed','utc':datetime.now(timezone.utc).isoformat(),
                            'plan_hash':spec['plan_hash'],'query_counts':counts,'total':len(rows),
                            'own_labels':'not attached; wait for full P0 audit','GPU_execution':'not yet implemented/launched'})
    print(json.dumps({'plan_hash':spec['plan_hash'],'queries':counts,'total':len(rows),'GPU':'not run'}))


if __name__=='__main__': main()
