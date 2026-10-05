# Environment and execution boundary

## Standard-library handoff check

Use Python3.12 (the historical worker version) for:

```bash
python scripts/verify_handoff.py
python scripts/materialize_handoff_artifacts.py
```

No pip packages, models, CUDA, provider credentials or network are required. This is the default CI and the first step for a fresh agent.

## Optional synthetic CPU TAM test

Needs an existing CPU-capable `torch`, `numpy`, and `scipy` environment, plus the pinned upstream cache. Uses random CPU tensors and a mock processor, not pretrained weights. Set `CUDA_VISIBLE_DEVICES=-1` before execution (`export` on Linux; `$env:CUDA_VISIBLE_DEVICES='-1'` on PowerShell), then:

```bash
python scripts/fetch_tam_upstream.py --fetch
python -m unittest discover -s tests -p test_query_tam_pilot_v1.py -v
```

Do not install the author's unused visual-text/LaTeX rendering stack merely to run the adapter tests. The adapter's evaluation path records that text visualization is not called.

## Historical scientific contracts, not installation advice

The current TAM/prompt experiments bind Qwen/Qwen3-VL-4B-Instruct revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`. Exact full model file hashes and environment are in `analysis/query_tam_pilot_v1/protocol.json` and its original source protocol. The captured Qwen stack includes Python3.12.3, torch2.8.0+cu128, torchvision0.23.0+cu128, transformers5.13.0, numpy2.5.1, scipy1.18.0, Pillow12.2.0. Root `requirements.txt` is the original non-torch pin file; it is not a complete CPU/GPU install recipe.

InternVL is a different isolated stack: OpenGVLab/InternVL2_5-8B revision `e9e4c0dc1db56bfab10458671519b7fa3dd29463`, transformers4.37.2, native448 dynamicmax2+thumbnail,32×4096 pre-answer states. Do not install it into the Qwen environment or guess compatibility from CPU mocks.

Future resumed science must inspect exact plan/environment/model inventory, verify fileSHA and original processor/visual-token budget, and pass a small real native/multi-demo compatibility check. No local pretrained-weight/model/CUDA/generation execution is authorized by the current pause/repository-update request.
