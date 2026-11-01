# DecisionICL core release

This branch is the DecisionICL CVPR manuscript's code release. It is a different
method from the older BehaviorICL state-retrieval project retained on `main`.

Read README.md and docs/PAPER_COVERAGE.md. Keep the complete train-pool,
train-only prototypes, native SigLIP group softmax, validation-only alpha fit,
equal factor weights, and exact generated-label matching contracts.
Every changed method/bank/budget requires independent validation calibration.
Do not tune on test labels or silently alter tie policies or frozen evidence.
The 2-shot coefficients reused in larger-shot curves are historical fixed-weight
extensions, not independently calibrated larger-shot results.

Only CPU code organization and verification are authorized by this release task.
Do not launch models, GPU experiments, cloud jobs, or automation as a side effect.
Original byte-identical files are recorded in provenance/source_manifest.json.
After edits run `python -m pytest -q`, `python tools/verify_release.py`, and
`python tools/verify_results.py`. No model weights are needed for those checks.

