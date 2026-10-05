"""Retrospective CPU comparison using the unchanged P0 failure proxy.

No new inference, feature extraction, retrieval, query filtering or test tuning.
All official queries and unknown matches retained; historical environment and
free-zero versus constrained-ICL decoding differences remain explicit.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_behavior_priority_analysis as a
from analyze_behavior_paper_p0 import validated_rows
from analyze_behavior_repair_all_queries import SPEC, failure_features, method_summary, paired_bootstrap, association_tables

QV_AUDIT = ROOT / 'analysis/paper_qwenvision_two_complete_results_v1/audit.json'
OLD_AUDIT = ROOT / 'analysis/paper_repair_five_dataset_v1/audit.json'
PLAN = ROOT / 'analysis/paper_qwenvision_dtd_cub_v1/plan_lock.json'
OUT = ROOT / 'analysis/paper_qwenvision_match_control_v1'


def derived_rows(dataset, manifest, zero, predictions, method):
    queries = manifest['query']; bank = {r['sample_id']:r for r in manifest['bank']}
    if len(predictions) != len(queries) or [r['query_id'] for r in predictions] != [r['sample_id'] for r in queries]:
        raise ValueError('Every ordered official query required, no duplicate or subset')
    rows = []
    for q, r in zip(queries, predictions):
        ids = r['demo_ids']; labels = r['demo_labels']
        if (len(ids) != 4 or len(set(ids)) != 4 or any(i not in bank for i in ids)
                or labels != [bank[i]['label'] for i in ids] or r['target'] != q['label']
                or type(r['correct']) is not bool or r['correct'] != (r['prediction'] == q['label'])):
            raise ValueError('Original full-bank gold labels, targets and actual predictions required')
        z = zero[q['sample_id']]
        if (z['true_label'],z['split']) != (q['label'],'query'):
            raise ValueError('Original same-query failure proxy differs')
        demo_preds = [zero[i]['zero_shot_pred'] for i in ids]
        f = failure_features(q['label'],z['zero_shot_pred'],labels,demo_preds)
        rows.append({'dataset':dataset,'method':method,'query_id':q['sample_id'],'true_label':q['label'],
            'zero_shot_pred':z['zero_shot_pred'],'zero_shot_valid':z['zero_shot_valid'],
            'zero_shot_correct':z['zero_shot_correct'],'initial_error':not z['zero_shot_correct'],
            'demo_ids':ids,'demo_labels':labels,'demo_zero_preds':demo_preds,
            'icl_pred':r['prediction'],'icl_correct':r['correct'],
            'rescued':not z['zero_shot_correct'] and r['correct'],
            'harmed':z['zero_shot_correct'] and not r['correct'],
            'eligible_common_complete_case_wrong':False,
            'num_same_true_label':f['sameclass_observed_count'],**f})
    return rows


def analyze():
    if OUT.exists():raise ValueError('Preserve previous derived evidence')
    previous=a.read(OLD_AUDIT); verified=a.read(QV_AUDIT); plan=a.read(PLAN)
    if (verified.get('status')!='complete_verified' or verified.get('complete_official_test') is not True
            or verified.get('independent_CPU_audit_after_stop') is not True or verified.get('predictions')!=7674):
        raise ValueError('Only independently audited complete QV official results')
    provenance={str(p.relative_to(ROOT)):a.sha(p) for p in (OLD_AUDIT,QV_AUDIT,PLAN)}
    def old_file(path):
        rel=str(path.relative_to(ROOT)); digest=a.sha(path)
        expected=previous['source_sha256'].get(rel,previous['source_sha256'].get(rel.replace('\\','/')))
        if digest!=expected:raise ValueError('Original P0 source byte mismatch: '+rel)
        provenance[rel]=digest
    def new_file(path):
        session=ROOT/verified['session']; rel=path.relative_to(session/'snapshot').as_posix(); digest=a.sha(path)
        if verified['archive_members_SHA'].get(rel)!=digest:raise ValueError('Full audited QV source differs: '+rel)
        provenance[str(path.relative_to(ROOT))]=digest
    reference_csv=ROOT/'analysis/paper_repair_five_dataset_v1/all_error_repair_summary.csv'
    provenance[str(reference_csv.relative_to(ROOT))]=a.sha(reference_csv)
    old_summaries={(r['dataset'],r['method']):r for r in csv.DictReader(reference_csv.open(encoding='utf-8-sig'))}
    catalog=a.source_catalog(); full=[]; summaries=[]; associations=[]; strata=[]; pairs=[]
    for dataset in ('dtd','cub'):
        data=plan['datasets'][dataset]; manifest=data['manifest']; base=catalog[dataset]['base']
        old_file(base/'manifest.json')
        if a.identity(a.read(base/'manifest.json'))!=a.identity(manifest):raise ValueError('Official bank/query identity mismatch')
        if dataset=='dtd':
            old_file(base/'gpt_mm_zero_shot_outputs.jsonl'); zero=a.load_zero(base,manifest,provenance)
        else:
            candidates=[ROOT/n for n in previous['source_sha256'] if 'cub' in n and n.endswith('zero_outputs.jsonl')]
            if len(candidates)!=1:raise ValueError('Unique original complete CUB failure source required')
            folder=candidates[0].parent
            for name in ('identity.json','manifest.json','verification.json','runtime.json','zero_outputs.jsonl'):old_file(folder/name)
            if a.identity(a.read(folder/'manifest.json'))!=a.identity(manifest):raise ValueError('Original complete zero identity differs')
            samples=manifest['bank']+manifest['query']; images={r['sample_id']:r['image_sha256'] for r in samples}
            zero=validated_rows(a.jsonl(folder/'zero_outputs.jsonl'),samples,data['class_labels'],a.read(folder/'identity.json')['job_hash'],images)
        grouped={}
        for method,(key,internal) in SPEC.items():
            current={}; group=a.load_group(catalog[dataset][key],internal,manifest,current)
            for name in current:old_file(ROOT/name)
            rows=derived_rows(dataset,manifest,zero,[group[q['sample_id']] for q in manifest['query']],method)
            summary=method_summary(rows); historical=old_summaries[dataset,method]
            for k,v in summary.items():
                if abs(float(historical[k])-float(v))>1e-10:raise ValueError('Original all-error decomposition not reproduced: '+k)
            grouped[method]=rows
        folder=ROOT/verified['session']/'snapshot/results/behavior_qwenvision_supcon_dtd_cub_v1'/dataset/'test/supcon_qwenvision/d256'
        for name in ('predictions.jsonl','selections.json','context.json','verification.json'):new_file(folder/name)
        predictions=a.jsonl(folder/'predictions.jsonl'); selected=a.read(folder/'selections.json'); context=a.read(folder/'context.json')
        legal=set(data['class_labels'])
        for r in predictions:
            chosen=selected[r['query_id']]
            if (r['prediction'] not in legal or r['demo_ids']!=chosen['demo_ids'] or r['demo_labels']!=chosen['demo_labels']
                    or r['context_hash']!=context['context_hash'] or r['plan_hash']!=verified['plan_hash']):
                raise ValueError('Complete audited actual context/selection differs')
        grouped['SupCon-QwenVision']=derived_rows(dataset,manifest,zero,predictions,'SupCon-QwenVision')
        cohorts=None
        for method,rows in grouped.items():
            ids=[r['query_id'] for r in rows if r['initial_error']]
            if cohorts is not None and ids!=cohorts:raise ValueError('Methods must share every initial error including illegal zero')
            cohorts=ids; full.extend(rows)
            summary=method_summary(rows)
            summary['strict_observed_coverage_all_errors_percent']=summary['strict_present_queries']*100/summary['zero_errors']
            summaries.append({'dataset':dataset,'method':method,**summary})
            assoc,fixed=association_tables(rows)
            associations.extend({'dataset':dataset,'method':method,**r} for r in assoc if r['scope']!='common_complete_case_wrong')
            strata.extend({'dataset':dataset,'method':method,**r} for r in fixed if r['scope']!='common_complete_case_wrong')
        first=grouped['SupCon-QwenVision']
        for name in ('BehaviorICL-Proj','BehaviorICL-Raw','RICES'):
            second=grouped[name]
            for outcome,subset in (('strict_observed_presence','all_initial_errors'),('rescue','all_initial_errors'),('harm','initial_correct')):
                indexes=[i for i,r in enumerate(first) if r['initial_error']==(subset=='all_initial_errors')]
                def flag(r):return r['strict_status']=='present' if outcome=='strict_observed_presence' else r['rescued'] if outcome=='rescue' else r['harmed']
                pair=paired_bootstrap([flag(first[i]) for i in indexes],[flag(second[i]) for i in indexes])
                pair={k.replace('rescued','positive'):v for k,v in pair.items()}
                pairs.append({'dataset':dataset,'first':'SupCon-QwenVision','second':name,'outcome':outcome,'cohort':subset,**pair})
    OUT.mkdir()
    for name,rows in (('query_level_results.csv',full),('all_error_repair_summary.csv',summaries),('match_associations.csv',associations),('fixed_sameclass_count_associations.csv',strata),('paired_differences.csv',pairs)):a.csv_out(OUT/name,rows)
    audit={'status':'complete_CPU_retrospective_comparison_verified','datasets':['dtd','cub'],'official_queries':7674,'query_method_rows':len(full),'methods':list(SPEC)+['SupCon-QwenVision'],
        'source_sha256':provenance,'analysis_source_SHA':{n:a.sha(ROOT/n) for n in ('scripts/analyze_behavior_qwenvision_match_control_v1.py','tests/test_behavior_qwenvision_match_control_v1.py','scripts/analyze_behavior_repair_all_queries.py','scripts/analyze_behavior_paper_p0.py','scripts/run_behavior_priority_analysis.py')},
        'outputs_SHA':{p.name:a.sha(p) for p in OUT.glob('*.csv')},'unchanged_P0_failure_proxy':True,'all_official_queries_retained':True,
        'strict_definition':'Same query/demo true class and same legal wrong original zero prediction',
        'unknown_policy':'No query removed; presence from any legal match, known absence only if query wrong/legal and all four demo answers legal; illegal query remains unknown',
        'paired_CI':'10000 paired-query bootstrap seed73, no multiplicity adjustment; observed strict presence unknown counted as unestablished, not known absence',
        'limits':'Retrospective observations using historical free-zero versus constrained-ICL decoding; historical zero and ICL environments differ from new QV. Conditional rescue cohorts may differ by method. No independent same-failure causal claim or model/feature/retriever/kernel replay.',
        'new_GPU_or_model_execution':False,'original_all_error_summaries_reproduced':True}
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':audit['status'],'output_root':str(OUT),'query_method_rows':len(full)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.parse_args();analyze()
