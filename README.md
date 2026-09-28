# BehaviorICL

Research code for retrieving in-context examples by multimodal model computation dynamics. The main experiment compares RICES, GPT-MM, DeTriever, CDR-Zero, and CDR-Learn on DTD, FGVC Aircraft, CUB-200-2011, Stanford Dogs, and Oxford-IIIT Pet.

## Setup

Use a Python environment with a CUDA-compatible PyTorch installation for your GPU, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Download the datasets into `datasets/` using `scripts/download_datasets.ps1` on Windows. See [DATASETS.md](DATASETS.md) for sources and licensing notes. Dataset images, model weights, caches, checkpoints, and predictions are not included in this repository.

## Main experiment

From the repository root, for example:

```bash
python scripts/run_multidataset_main.py --dataset dtd --model qwen3vl4b
```

The runner defaults to four shots and seed 73. It writes results under `results/cdr_main/`. Use `--resume` to continue an interrupted run with the same dataset, model, and seed. CDR-Learn and DeTriever initialize and train separately for each dataset–model–seed tuple; frozen features may be reused only within that tuple. Run `python scripts/run_multidataset_main.py --help` for stage and method options.

The adapter also defines Gemma 3 4B, whose weights require separate access approval. The current main protocol uses Qwen3-VL-4B-Instruct.

## Checks

```bash
python -m unittest discover -s tests
```

For completed runs, use `scripts/verify_main_run.py` and `scripts/analyze_retrieval_run.py` to validate outputs and summarize retrieval results.
