# Pinned upstream TAM and adapter

Upstream: [xmed-lab/TAM](https://github.com/xmed-lab/TAM), commit `7c7df1d3df418bb9467ca68ff2f00ff1f26a6f4c`; paper [Token Activation Map to Visually Explain Multimodal LLM](https://arxiv.org/abs/2506.23270).

The pinned upstream tree has no license file. This repository publishes our adapter/tests and the upstream byte receipt, and fetches the author's unchanged source into an ignored local cache rather than redistributing/relicensing it.

```bash
python scripts/fetch_tam_upstream.py --fetch
```

The fetcher reads only the pinned `tam.py`, `demo.py`, `qwen_utils.py`, `requirements.txt`, and `README.md`, verifies each SHA from [tam_upstream_receipt.json](tam_upstream_receipt.json), refuses conflicting existing files and executes none of them. It does not install requirements or download models/images.

Our byte-preserved [adapter](../scripts/query_tam_adapter_v1.py) compiles the original function AST, verifies the author's `tam.py` SHA, and changes exactly one documented class-target index: first generated answer atL, rather than last input tokenL−1. The original integer round0 evaluation returns the first prompt map, so the adapter processes prompt tuples in author order and saves the actual answer map. ECI/rank filtering functions are otherwise unchanged.

CPU tests cover actual original ECI/filter execution on synthetic tensors, correct target token positions, raw-vision invariance under text-logit changes, wrong-source/token-boundary refusal, and locked image strata/query definitions. They do not load a pretrained model, inspect real attention, or prove Qwen3 GPU compatibility.

Interpret raw CAM, post-ECI and last-input hidden states separately. Fixed gold targets are teacher-forced probes, not natural predictions. An image before a later query can have unchanged fixed-target raw CAM while the ECI text correction or final-input states change. Preserve this possible negative result.
