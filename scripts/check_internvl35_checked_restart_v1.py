"""Same-context replay and next uncommitted input under the operational guard."""
import argparse
import json
from pathlib import Path
import torch
import internvl35_joint_config_v2 as config
from run_internvl35_joint_methods_v2 import load_context
import internvl35_icl_cache_guard_v1 as guard
from multimodal_model_adapter import icl_messages

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",required=True)
    parser.add_argument("--method",required=True)
    parser.add_argument("--rows",type=int,required=True)
    parser.add_argument("--report",type=Path,required=True)
    args=parser.parse_args()
    config.require_gate()
    folder,bank,queries,identity=load_context(args.dataset)
    rows=[json.loads(line) for line in (folder/"predictions"/f"{args.method}.jsonl").read_bytes().splitlines()]
    if len(rows)!=args.rows or not 0<len(rows)<len(queries):
        raise ValueError("Interrupted prefix changed")
    selection=json.loads((folder/"selections"/f"{args.method}.json").read_text(encoding="utf-8"))
    manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    image_rows={r['sample_id']:r for part in ('bank','query') for r in manifest[part]}
    labels=sorted({b.label for b in bank})
    torch.manual_seed(73)
    model,processor=config.load_model(args.report.with_suffix(".inputs.jsonl"))
    guard.install()
    report={"state":"running","rows":[],"accuracy_used_for_tuning":False}
    config.write_json(args.report,report)
    for index in (args.rows-1,args.rows):
        query=queries[index];demos=[bank[i] for i in selection[query.sample_id]]
        for image in demos+[query]:
            if config.digest(image.path)!=image_rows[image.sample_id]['image_sha256']:
                raise ValueError("Context image bytes changed")
        torch.cuda.reset_peak_memory_stats()
        prediction,raw=guard.guarded_generate(model,processor,[icl_messages(demos,query,args.dataset,labels,None)],labels)[0]
        entry={"query_id":query.sample_id,"kind":"saved_replay" if index<args.rows else "next_uncommitted",
               "prediction":prediction,"raw":raw,"legal":prediction in labels,
               "peak_reserved_gb":torch.cuda.max_memory_reserved()/2**30,**guard.guarded_generate.last_memory}
        if index<args.rows:
            entry['exact_saved_output']=(prediction,raw)==(rows[index]['prediction'],rows[index]['raw_output'])
            if not entry['exact_saved_output']:
                raise RuntimeError("Memory-guard replay differs; do not resume")
        if not entry['legal']:
            raise RuntimeError("Illegal compatibility output")
        report['rows'].append(entry);config.write_json(args.report,report)
    report['state']='complete';config.write_json(args.report,report)

if __name__=="__main__":
    main()
