# BehaviorICL

Research code for **Behavior-Zero** and **Behavior-Learn**, lightweight methods for retrieving in-context demonstrations from a vision-language model's internal response to an image and a classification task. **Behavior is the main method of this repository.** RICES, GPT-MM, and a task-adapted DeTriever are comparison baselines; earlier CDR and forward-trace experiments are exploratory code, not the proposed method.

## Method at a glance

For each bank or test image, we make one *zero-shot prefill* with the image, task instruction, and valid class list. From Qwen3-VL-4B-Instruct we take the hidden state at the answer position from each of its 36 decoder layers. This gives a `36 × 2560` representation per image. The method does **not** use a sequence of generated tokens, adjacent-layer differences, or test labels to build the representation.

- **Behavior-Zero:** L2-normalize each layer state, average the 36 same-layer cosine similarities between query and bank item, and retrieve the four highest-scoring bank images.
- **Behavior-Learn:** apply a shared `2560 → 256` projection and learn 36 layer weights. A supervised contrastive objective uses same-class versus different-class examples from the *training bank*; ten bank samples per class are held out for model selection. Training starts from a new random initialization for every dataset–model–seed tuple. At test time, the learned retriever scores a query against the bank and selects four demonstrations.

The four selected bank images, with their labels, become the VLM's few-shot prompt. Evaluation uses greedy generation with decoding constrained to the dataset's legal class names. The on-disk method identifiers `decision_zero` and `decision_learn` are legacy keys for Behavior-Zero and Behavior-Learn; they are retained so existing runs remain resumable.

## Setup

Install a CUDA-compatible PyTorch build for your GPU, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Download the datasets into `datasets/` with `scripts/download_datasets.ps1` on Windows. See [DATASETS.md](DATASETS.md) for dataset sources, splits, and licensing. Model weights and dataset files must be obtained separately; this repository does not include them.

The reported protocol uses Qwen3-VL-4B-Instruct, seed 73, four demonstrations, the complete official test split, and the same image-processing and generation settings for all compared retrieval methods. See [RESULTS.md](RESULTS.md) for completed results and limitations.

## Run Behavior retrieval

From the repository root, first extract the frozen input states for a dataset, then train/select demonstrations and evaluate both Behavior methods:

```bash
python scripts/prepare_detriever_input_states.py --dataset dtd --output-root results/behavior_source
python scripts/run_dtd_decision_states.py --dataset dtd --source-root results/behavior_source --output-root results/behavior_main --seed 73 --shots 4 --train-steps 1500
python scripts/verify_dtd_decision_states.py results/behavior_main/dtd/qwen3vl4b/seed_73 --source-run results/behavior_source/dtd/qwen3vl4b/seed_73
```

Replace `dtd` with `aircraft`, `pets`, `cub`, or `dogs` to run the other datasets. The frozen-state file can be several GB. Use `--resume` for an interrupted extraction or Behavior run **only within the same dataset–model–seed tuple**; do not reuse a checkpoint, selections, or predictions across tuples. The extractor's filename retains an older DeTriever-oriented name, but here it only creates the shared frozen input-state cache.

## Baselines and validation

- `scripts/run_multidataset_main.py` contains the original five-method baseline runner, including RICES and GPT-MM. Its legacy CDR and label-identity DeTriever outputs are **not** the paper-aligned DeTriever results reported in `RESULTS.md`.
- `scripts/run_detriever_output_proxy.py` trains the reported task-adapted DeTriever with a bank-only gold-input-plus-answer representation proxy; `scripts/verify_detriever_output_proxy.py` checks its completed output. This is an adaptation of the published method, **not the authors' official implementation**.
- `scripts/compare_decision_main.py` checks query identity and computes paired comparisons between completed Behavior, baseline, and paper-aligned DeTriever runs.

Run unit tests with:

```bash
python -m unittest discover -s tests
```

`results/`, `datasets/`, `.hf_cache/`, checkpoints, and predictions are excluded from Git. [RESULTS.md](RESULTS.md) contains verified aggregate results and points to the local artifacts used for validation.
