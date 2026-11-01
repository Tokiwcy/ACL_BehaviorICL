# DecisionICL

Core code for **DecisionICL: Selecting Visual Demonstrations with Target-Model Decision Factors**.
This branch contains the CVPR manuscript and its complete supplementary material.
The older BehaviorICL hidden-state retrieval project remains in the repository's `main` history.

目标 VLM 从训练图像提出决策因素；冻结 SigLIP 2 测量因素状态；训练集构建类别原型；
验证集独立校准融合系数；选定类别后，在该类别的**完整训练池**检索示例；最后由目标 VLM 生成答案。

## Layout

```text
src/decisionicl/          scientific implementation and compact CLI
  discovery.py, atoms.py factor discovery, schema and duplicate handling
  encoders.py, clip.py   frozen SigLIP 2 / CLIP encoders and CLIP patch maps
  evidence.py            aligned evidence stores and group probabilities
  raw_evidence.py        paper policy: native logits, no train debiasing
  selection.py           prototypes, JSD, normalization, equal-factor calibration
  pipeline.py            contract-bound validation fit and full-pool retrieval
  baselines.py           Random/RICES/MUIER/CIRCLES and class/source controls
  generic.py             Generic direct-sample control and its own calibration
  spatial.py, vicl.py    Prompt-SelF spatial and VICL text retrieval adaptations
  circles.py            CIRCLES attribute and counterfactual description generation
  prompts.py, models.py  accepted prompt builders and three target model families
  inference.py          fixed-batch raw-output checkpoints and contract-checked resume
  analysis.py            final-answer pairs, plan changes and repeated summaries
resources/               15 frozen main-table banks and paper experiment settings
results/paper/           current manuscript numerical inputs, including all repeats
results/tables/          all 28 original LaTeX numerical tables
results/main_release/   original 180-cell main release plus its published controls
artifact_bundle/         original portable predictions/labels/audits archive
paper/                   unchanged, standalone main.tex and supplement.tex
tests/                   CPU scientific/contract/CLI checks
tools/                   release verification and original-artifact recomputation
```

See [论文逐项对应表](docs/PAPER_COVERAGE.md) and [reproduction instructions](docs/REPRODUCTION.md).
The code tree contains no monitoring, cloud launch, restart, historical queue or migration scripts.

## Install and verify

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python tools/verify_release.py
python tools/verify_results.py
python tools/recompute_artifact_results.py --review-archive artifact_bundle/decisionicl-review-artifacts.zip
```

The default install and checks use CPU libraries. Model inference additionally requires
`pip install -e ".[models,quantization]"`, dataset images and the matching checkpoints.
Importing the package or running its offline checks does not load a target model.

## Main method

For each factor, softmax the encoder's native state logits. Average training probabilities
within each class to form its prototype. Agreement is `1 - JSD / log(2)`.
Standardize each base/factor channel across classes, then score with
`alpha * base + (1-alpha) * mean(factors)`. Five-fold validation fits and the final
full-validation fit use the frozen alpha grid; ties prefer larger base weight.
Use the selected class's complete training pool for final base-retriever ranking.
The VLM receives the retrieved images and labels, not the numeric class score as an answer.

```python
from decisionicl.pipeline import DecisionICLRetriever

retriever = DecisionICLRetriever(
    train_ids, train_labels, train_factor_probabilities,
    contract={"target_model": "Qwen3-VL-4B", "dataset": "aircraft",
              "bank_sha256": bank_sha, "evidence_encoder": encoder_identity,
              "parent": "rices", "seed": 42}, shots=2,
)
calibration = retriever.fit(validation_parent_scores, validation_probabilities, validation_samples)
plan = retriever.retrieve(test_parent_scores, test_probabilities, test_ids)
```

Online retrieval accepts no query labels. Calibration binds target/dataset/bank/parent/budget
and the ordered training evidence. Changing a method requires a fresh validation fit.
`baselines.py` exposes the two-channel CIRCLES rule separately; Prompt-SelF and VICL retain
their own ranking and tie policies. The compact CLI implements the single-channel main rule.

## Reported evidence

The main paper uses accepted **seed 42** results: three target models, five datasets,
one/two demonstrations, **180 method cells and 75 matched comparisons**.
Equal-cell mean gains are 2.06/3.11/3.52 percentage points for Qwen3/Qwen2.5/Gemma.
The supplement separately includes the completed Qwen3 **42/17/73** evaluations,
larger-shot curves, all controls, 90 auxiliary VICL/Prompt-SelF cells and 60 auxiliary pairs.
The historical main release's October 9 repeat counters are provenance, not current status.

This organization preserves the manuscript, result inputs and original review evidence.
The new portable package is validated on CPU, including the paper's real rescue/failure
JSD values and offline fit-to-plan flow. It has not rerun the three pretrained models.
Images, weights and the larger raw/cache/numeric evidence remain external; exact historical
outputs also depend on checkpoint revisions, prompt identities and fixed batch partners.
Original evidence downloads are listed in [artifact_bundle/README.md](artifact_bundle/README.md).

