# Reproduction

## Offline checks

Install `.[test]`, run pytest and both release verifiers from the repository root.
`tools/recompute_artifact_results.py --review-archive artifact_bundle/decisionicl-review-artifacts.zip`
recomputes original main/control accuracy, macro, invalid, rescue/harm and exact p-values
from the published predictions and labels. It checks the original audited bootstrap
interval receipts; it does not silently redraw them.

`provenance/source_manifest.json` distinguishes byte-identical original files from
extracted functions and adapted interfaces. `provenance/release_manifest.json` binds
the final package. Historical manifests retain their original `D:\` paths as identity
records. The portable runtime uses explicit inputs and does not require that drive.

## Inputs and calibration

Image manifests are JSONL `Sample` records: `sample_id`, `image_path`, `label`, `split`,
optional `question`, `answer`, `image_sha256`. Keep train, validation and test disjoint.
Use the exact ordered IDs in `EmbeddingStore` and `EvidenceStore` when loading caches.
Only training images contribute prototypes. Query labels are absent from retrieval.

The paper's raw state evidence must use `raw_evidence.build_raw_siglip_evidence`,
which validates the native-logit identity transform. `evidence.py` also understands
historical store schemas, including older calibration formats; those formats are
supported for reading and must not be silently substituted for the paper policy.

`DecisionICLRetriever` fits a single-channel parent (RICES, MUIER or another fixed
query-by-full-train score matrix). Supply the exact bank/encoder/target identities
in its contract. Fit each target/dataset/parent/shot independently. CIRCLES builds
features with `baselines.circles_features` and uses the same `calibrate_decisionicl`;
its final two channels use `circles_demonstrations`. Generic direct retrieval uses
its independently fitted demo-precision objective in `generic.fit_one`.

The array CLI accepts NPZ files, with no pickle:

```text
train_ids, train_labels        string arrays in full training-pool order
factor_count                  scalar integer G
train_probabilities_0 .. G-1   train x factor-state probabilities
query_probabilities_0 .. G-1   query x factor-state probabilities
parent_scores                 query x full train scores
query_ids                     unique string IDs, in score/evidence order
query_split                   scalar 'validation' or 'test'
query_labels                  validation only; never consumed by retrieve
```

```bash
decisionicl fit --input validation.npz --contract resources/example_contract.json --shots 2 --calibration runs/calibration.json
decisionicl retrieve --input test.npz --contract resources/example_contract.json --shots 2 --calibration runs/calibration.json --output runs/plan.jsonl
```

Replace example identities with actual frozen hashes. The CLI does not infer or
download datasets and never executes a model. `decisionicl evaluate --input answers.json
--output metrics.json` expects aligned arrays `labels`, `baseline`, `method`, `class_names`.

## Model APIs

`GenerationBackend(ModelConfig(family, model_path, revision=...))` supports `qwen3`,
`qwen25`, `gemma4`. Use `prompts.official_content` for image demonstrations and the
VICL prompt builders for text summaries. Preserve each batch's complete conversations,
ordered demonstration IDs, image hashes, raw output and batch-group identity.
Use `FactorDiscoveryBackend` for the discovery request/parse APIs.

`inference.run_image_plan` binds image hashes and complete batch membership, then
checkpoints one complete fixed group per JSONL record. Resuming rejects a changed
model, context, image, group or ordering contract. It does not launch from the
offline CLI. `circles.describe` generates the accepted attribute and manipulated
caption prompts; `ClipEncoder` encodes the image and caption retrieval channels.

`SiglipEncoder` supplies evidence embeddings and native logit scale/bias. CLIP
embeddings supply RICES and CIRCLES image/caption scores. Prompt-SelF spatial
features are CLIP ViT-L/14's final vision hidden state **without CLS**, shape
`images x 256 x 1024`, L2-normalized at each position. Its score is the mean
corresponding-position cosine. VICL descriptions are CLIP text-to-train-image
cosines, with visual similarity then training index as tie breaks.

Main answers use exact case-insensitive label matching after trimming whitespace.
Do not clean prose into a legal answer. Gemma uses its official structured response
parser; the recognized role-only empty assistant response becomes invalid.

## Exact historical runs versus new executions

This refactoring does not claim newly reproduced pretrained outputs. Exact replay
needs images, weights, original processor/revision identities, numeric evidence,
fixed batch partners, environment receipts and accepted raw outputs. The compact
review ZIP covers original main/control results, not every newer repeat/auxiliary
raw cache. Current supplementary summary inputs and their hashes are included;
additional full raw resources remain external as recorded by their source manifests.

The main results are seed 42. Qwen3 repeats 17/73 are separately reported completed
evaluations. They mostly preserve non-Random plans, so their SD measures execution
repeatability under the recorded conditions, not factor-bank sampling uncertainty.
Higher-shot results use the frozen two-shot calibration as stated in the manuscript.
Any newly changed method/bank/evidence/budget needs a fresh independent validation fit.

