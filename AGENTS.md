# Instructions for the next coding agent

## Read this first

This is the **current BehaviorICL research handoff**, not the early velocity/spurious-task project. Read, in order:

1. `HANDOFF.md` — exact state, first useful actions, and acceptance criteria.
2. `PROJECT_OVERVIEW.md` — current idea and implemented method.
3. `CURRENT_RESULTS.md` — audited results, denominators, confidence intervals, and limits.
4. `CURRENT_CODE.md` — current scientific entry points and required old-name dependencies.
5. `NEXT_STEPS.md` and `handoff/ARTIFACTS.md` — priorities and what a clone actually contains.

Run `python scripts/verify_handoff.py` before editing. It is standard-library-only and makes no network, CUDA, model, or cloud calls. A passing check verifies the published handoff, **not** a complete model rerun or the current provider account.

## Current work and authorization

- The human explicitly **paused all experiments and automation on 2026-10-06**. Updating this GitHub repository does not revoke that pause.
- Do not start/restore a GPU, invoke old cloud entries, or restart TAM/InternVL/three-seed jobs unless the human explicitly resumes that work. Continue authorized CPU/code/documentation work.
- The latest research priority is **changed-query, fixed-image TAM explanation**, ahead of InternVL. There are **no actual TAM maps or 64-run results** yet.
- Prompt intervention, fixed-label intervention, QwenVision control, and Qwen seed73 results are complete. Do not rerun them to fill documentation gaps.
- InternVL five datasets and new Qwen seeds74/75 remain incomplete. Their failure audits are not scientific completions.

## Scientific contract

- BehaviorICL uses the target VLM's **task-conditioned layerwise pre-answer states**, at the final input position. Qwen has36 decoder layers×2560 dimensions; the embedding layer is excluded. This is not a generated reasoning trajectory, attention map, first-layer vision encoder, or velocity difference.
- Raw: per-layer L2 normalization and uniform mean cosine. Proj: shared fixed256 projection, uniform layers, bank-only SupCon. Learn: same projection plus learned softmax layer weights.
- Four demonstrations; bank-only10dev/class;1500steps and earliest best-dev checkpoint; full-bank final candidates. No test tuning, gates, width grid, or query gold label in online selection.
- Only QwenVision was reduced to DTD+CUB. InternVL and three-seed validation retain five datasets.
- Report single seed73 as single seed. Report five×256 labels as a sample. Gold−own full-test effects use only available paired queries and retain the170 explicit missing records.
- Keep harm, unknown/illegal outputs, historical environment/batch differences, fixed-sameclass-count limits, GPT-MM/QV coverage counterexamples. No independent same-failure causal claim or universal decoder superiority.

## Source/evidence discipline

- `scripts/run_dtd_learned_velocity.py` is still a dependency: current code uses `StateRetriever`, not its old `VelocityRetriever`. Read the function, not the stale historical header.
- Original scientific files and audits are published byte-for-byte. `handoff/publish_manifest.json` records their hashes. Do not silently rewrite original locks or historical receipts; version any scientific change and document its new binding.
- A public clone lacks images, model weights, raw feature/checkpoint/full archives, private cloud state and credentials. Read `handoff/ARTIFACTS.md`; do not invent missing files/fields or call a snapshot a live audit.
- Large contracts and query-level CSVs are provided as exact gzip payloads. Materialize them with `python scripts/materialize_handoff_artifacts.py`; that command is offline and does not execute models.
- Upstream TAM is pinned. Use `scripts/fetch_tam_upstream.py --fetch` for the ignored cache before synthetic CPU TAM tests. The fetcher verifies bytes and does not execute upstream code.

## If the human later resumes cloud experiments

First complete the small actual compatibility check described in `HANDOFF.md`; build a new reviewed operation identity, do not reuse a consumed launcher. At most two single-GPU cloud jobs, one scientific worker per card; independent writable roots, unique watcher, real120-second archive/member/extractedSHA backups, exact stop, separate fee segments. No local pretrained weights/CUDA/generation. Keep the real resource-creation mutex and native/identity guards. FULL SHA takeback→immediate exact stop/fresh provider confirmation→independent CPU audit before reporting completion. Install/upload/preallocation/smoke/CPU mocks are not scientific results.

## Validation and handoff

After changes, run `python scripts/verify_handoff.py` and targeted tests relevant to the change. Preserve the paused state. Report what changed, what was actually tested, missing external artifacts, and the next concrete action. Prefer one reusable checked implementation over another chain of one-off startup versions.
