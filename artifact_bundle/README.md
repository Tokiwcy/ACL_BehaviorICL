# Original review evidence

`decisionicl-review-artifacts.zip` is the unchanged original review archive
(predictions, test labels, plans/calibration receipts and audits) from the
[DecisionICL evidence release](https://github.com/Tokiwcy/CVPR_DecisionICL/releases/tag/decisionicl-evidence-20261009).
The release also hosts the much larger raw/cache/numeric resource archives.
This repository does not redistribute dataset images or model checkpoints.

```bash
python tools/recompute_artifact_results.py --review-archive artifact_bundle/decisionicl-review-artifacts.zip
```

The archive and `evaluation_index.json` describe the original main/control release.
Current supplement summary inputs, including completed repeated evaluations and
auxiliary baselines, are separately in `results/paper/`; do not interpret the old
export's paused repeat counters as current manuscript status.

