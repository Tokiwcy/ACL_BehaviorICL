"""Fixed-group target inference with raw outputs and offline resumable receipts."""
from __future__ import annotations
import json
import os
from pathlib import Path
from .hashing import file_sha256, stable_hash
from .prompts import official_content


def run_image_plan(plan, train_samples, query_samples, *, backend, labels,
                   question, output_path, method="rices", batch_size=2):
    """Execute an explicitly supplied plan; never called by fit or retrieve.

    Method names use prompts.py's canonical names; CIRCLES uses "circles".
    The full fixed batch is the cache unit, so partner changes invalidate reuse.
    """
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    train = {s.sample_id: s for s in train_samples}
    queries = {s.sample_id: s for s in query_samples}
    if len(train) != len(train_samples) or len(queries) != len(query_samples) or train.keys() & queries.keys():
        raise ValueError("sample IDs duplicate or overlap")
    if any(s.split != "train" for s in train_samples) or any(s.split != "test" for s in query_samples):
        raise ValueError("expected train demonstration pool and held-out test queries")
    if len(plan) != len({row['query_id'] for row in plan}) or {r['query_id'] for r in plan} != queries.keys():
        raise ValueError("plan must cover each query exactly once")
    # Validate image identities once, before invoking the backend.
    image_hashes = {}
    for sample in [*train_samples, *query_samples]:
        actual = file_sha256(sample.image_path)
        if sample.image_sha256 and actual != sample.image_sha256:
            raise ValueError("image identity drift: "+sample.sample_id)
        image_hashes[sample.sample_id] = actual
    groups = []
    for start in range(0,len(plan),batch_size):
        group = plan[start:start+batch_size]
        conversations = []
        for row in group:
            demos = [train[q] for q in row['demo_ids']]
            conversations.append(official_content(question, queries[row['query_id']].image_path,
                [(s.image_path,s.label) for s in demos], labels, method,
                attribute=row.get('attribute',''),caption=row.get('modified_caption','')))
        identity = stable_hash({'plan':group,'conversations':conversations,
            'image_hashes':{sid:image_hashes[sid] for r in group for sid in [r['query_id'],*r['demo_ids']]},
            'model':backend.config.__dict__,'batch_size':batch_size})
        groups.append((group,conversations,identity))
    output = Path(output_path)
    existing = []
    if output.exists():
        for line in output.read_text(encoding='utf-8').splitlines():
            record=json.loads(line)
            if len(record['answers']) != len(record['query_ids']):
                raise ValueError("incomplete fixed-group checkpoint")
            existing.append(record)
    if len(existing)>len(groups) or any(r['request_hash']!=groups[i][2] for i,r in enumerate(existing)):
        raise ValueError("saved inference belongs to a different fixed-group contract")
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('a',encoding='utf-8') as stream:
        for group,conversations,identity in groups[len(existing):]:
            answers=backend.classify(conversations,labels)
            if len(answers)!=len(group):
                raise ValueError("backend returned incomplete fixed group")
            record={'request_hash':identity,'query_ids':[r['query_id'] for r in group],'answers':answers}
            stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
            existing.append(record)
    return existing

