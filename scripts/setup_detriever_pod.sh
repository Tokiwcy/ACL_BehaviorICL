#!/usr/bin/env bash
set -euo pipefail

cd /workspace/BehaviorICL
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install --no-cache-dir -r requirements.txt huggingface_hub
export HF_HOME=/workspace/BehaviorICL/.hf_cache
.venv/bin/python scripts/download_pinned_qwen.py
.venv/bin/python -c 'import torch, transformers; print("READY", torch.__version__, transformers.__version__, torch.cuda.is_available(), flush=True)'
