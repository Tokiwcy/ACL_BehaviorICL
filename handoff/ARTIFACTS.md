# Artifact availability

This repository is a portable code/evidence handoff, not the full private experiment workspace. Historical audits are preserved as records of completed independent checks; a clean clone cannot re-run those full audits without their external inputs.

## Included directly

- Current overview, code map, results, narrative, priorities, paused run status, `AGENTS.md`, `HANDOFF.md`.
- Selected current scientific implementations and their static local-import dependencies, original byte identities preserved.
- Original completion audits and compact result/paired-CI CSVs for seed73, prompt, label, QV, match control; label query-arm tables and sampled outputs.
- Fixed TAM protocol and CPU review; two deferred scientific failure audits and CPU repair evidence.
- Safe historical TAM failure/stop records in `handoff/local_evidence/`, plus the local consolidation pause snapshot. These are historical snapshots, **not** live provider authority for a new clone.
- `handoff/publish_manifest.json`: exact original bytes/SHA and compressed artifact identities. `handoff/external_artifacts.json`: omitted local context and source SHA when known.

## Materialize frozen artifacts

Large ordered-manifest plans, label query/arm lock, and full P0/QV query-level CSVs are in `artifacts/frozen/...json.gz` or `.csv.gz`. Their gzip payloads decompress to the exact original bytes, not rewritten summaries. See manifest entries with `kind=gzip_exact`.

```bash
python scripts/materialize_handoff_artifacts.py
python scripts/verify_handoff.py
```

This is offline, bounded by each recorded source byte length, validates compressed/sourceSHA, and refuses conflicting existing files. It never downloads images/weights or starts a model/cloud process. Restored paths are ignored by Git to avoid duplicate large blobs. Original frozen validators may still require omitted private evidence and a versioned path adapter; materialization alone does not make them runnable.

The main restored paths are:

| Contract/output | Why it matters |
|---|---|
| `analysis/task_conditioning_prompt_plan_v1/protocol_lock.json` | Exact task/generic source orderedbank/query, labels, input SHA and prompt protocol |
| `analysis/paired_label_full_test_dtd_cub_plan_v1/query_arm_lock.json` | Exact fixed contexts and planned own-label missingness |
| `analysis/paper_qwenvision_dtd_cub_v1/plan_lock.json` | DTD+CUB visual-control source/training/test scope |
| `analysis/second_vlm_five_dataset_v2/plan_lock.json` | Deferred five-dataset Intern scientific scope |
| `analysis/paper_qwen_three_seed_plan_v1/plan_lock.json` | Deferred seeds73/74/75, six methods, independent tuples |
| `analysis/paper_repair_five_dataset_v1/query_level_results.csv` | Complete139536 query×method retrospective rows |
| `analysis/paper_qwenvision_match_control_v1/query_level_results.csv` | Complete53718 seven-method retrospective rows |

## Local-only artifacts

| Missing material | How to proceed |
|---|---|
| Dataset image bytes under `datasets/` | Obtain original datasets under their terms; verify the frozen ordered manifests/imageSHA. TAM needs exactly eight locked images, not replacement examples |
| Fixed Qwen/Intern/CLIP pretrained files | Restore/download the exact revision/inventory from the plan. Do not load weights locally under current authorization |
| Original raw prediction JSONL, npy/pt source/checkpoints, full cloud archives | Obtain original session archives from maintainer, verify member/extracted/source identities using the original audit |
| Private operational `.tools` packages, tokenizer caches, raw API/auth/runtime evidence | Do not reconstruct from public PodIDs or fill expected fields. New resumed operation needs actual authorized native/backup/identity evidence |
| Old stopped Pod volumes and incomplete historical FULL | Preserve separately. They are not public Git assets or prerequisite permission to launch a new job |

Original completed sessions remain in the maintainer's workspace:

- `results/cloud_trial/czscrdd6bcut9k-paper-task-prompt-autonomous/session_1`
- `results/cloud_trial/yuj11of0d8gbgn-paper-full-labels-staged-pinned/session_1`
- `results/cloud_trial/t0ku94ln50lmsf-paper-qv37/session_1`

Original machine path `C:/Users/ASUS/Documents/ChatGPT/BehaviorICL` may occur inside immutable scripts/audits. Treat it as historical context; do not assume that location exists. Full original model/kernel replay was not performed by the consolidation or repository CI.

## Local history

The maintainer's `archive/project_docs_before_20261006` preserves the previous documents and their SHA. It is not included wholesale because old startup prose is not the current handoff. The current tables retain the relevant positive/negative evidence and link original structured audits. Previous public scripts are recoverable in Git history; `handoff/old_public_files_retired.json` lists which were removed from the current tree.

## Upstream TAM

Author code is fetched into the ignored `.tools/tam_official_v1` cache, never substituted by a guessed implementation. Read [TAM_UPSTREAM.md](TAM_UPSTREAM.md). The published adapter, receipt and tests do not imply that real model outputs have been generated.
