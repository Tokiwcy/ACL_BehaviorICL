# Validation receipt — 2026-11-01

This release was prepared with Python 3.12.13, NumPy 2.3.5, SciPy 1.18.1 and
pytest 9.1.1. All checks were CPU-only. No target model, CUDA worker, cloud job or
automation was started.

- `python -m pytest -q`: **12 passed**. Checks include real manuscript rescue/harm
  distribution agreements against an independent relative-entropy formulation,
  CIRCLES channel maxima/ties, MUIER/MMICES adaptations, complete-pool selection,
  validation-only fitting and contract rejection, offline CLI fit/retrieve,
  fixed-batch checkpoint reuse and changed-partner rejection, matched source
  prompts, train-only discovery, VICL text examples and spatial patch scoring.
- `python tools/verify_release.py`: passed. Binds 130 package files, the complete
  main/supplement source, all 28 numerical tables, 15 main factor banks,
  180 main cells, 90 auxiliary cells/60 pairs, 180 three-run cells and 12 larger-shot cells.
- `python tools/verify_results.py`: passed. Checks the exact original evidence
  archive's 385 members, 152 recorded source hashes, 214 result rows, 149 pairs
  and the 75 primary DecisionICL comparisons.
- `python tools/recompute_artifact_results.py --review-archive artifact_bundle/decisionicl-review-artifacts.zip`:
  passed. Recomputes 214 method rows, 149 paired rows and four matched-source
  comparisons from original predictions and labels.
- `python -m pip wheel . --no-deps --no-build-isolation`: package wheel built.

Pretrained inference of the refactored model adapters has not been tested in this
task. Original published results are retained as evidence, not claimed as newly rerun.
The original third-party LaTeX style file retains its existing trailing whitespace
to preserve its recorded bytes.
