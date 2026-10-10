# 本机接手状态（2026-10-06）

> 本页记录最初的离线安装状态；之后用户授权的本机 GPU TAM 与检索实验见 [LOCAL_GPU_STUDY.md](LOCAL_GPU_STUDY.md)。

本机仓库在 `C:\Users\toki_\Documents\ChatGPT\BehaviorICL`，跟踪 GitHub `main` 的 `6b5b78e`。本页记录可公开取得的代码、数据、模型和离线证据的准备状态。研究实验仍按 `AGENTS.md` 暂停；本次没有加载预训练模型、调用 CUDA、生成预测或启动云任务。

## 已准备并核验

| 项目 | 本地位置 / 版本 | 核验 |
|---|---|---|
| 主线 Python 环境 | `.venv`；Python 3.12.14；torch 2.8.0+cu128、torchvision 0.23.0+cu128、transformers 5.13.0 和根目录 `requirements.txt` | `pip check` 通过 |
| InternVL 独立环境 | `.tools/venv-internvl`；Python 3.12.14；torch 2.8.0+cu128、transformers 4.37.2 及冻结计划中的 tokenizers/hub/safetensors/sentencepiece/einops/timm/numpy/scipy/Pillow 版本 | `pip check` 通过；没有导入作者模型代码 |
| Qwen3-VL-4B-Instruct | `.hf_cache`，revision `ebb281ec70b05090aa6165b016eac8ec08e71b17` | 冻结 TAM 协议列出的 12 个文件逐个 SHA-256 一致，包括两块权重 |
| CLIP ViT-B/32 | `.hf_cache`，revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` | 冻结第二 VLM 计划列出的 8 个文件逐个 SHA-256 一致，包括 PyTorch 权重 |
| InternVL2.5-8B | `.hf_cache`，revision `e9e4c0dc1db56bfab10458671519b7fa3dd29463` | 冻结第二 VLM 计划列出的 21 个文件逐个长度和 SHA-256 一致，包括四块权重及作者代码 |
| 官方数据集 | `datasets/`：DTD 5,640；Aircraft 10,000；CUB 11,788；Pets 7,349；Dogs 20,580 | 共 55,357 张使用冻结计划的逐图 SHA-256 核验；五组 bank/query 顺序、路径、标签和 split 与冻结计划一致 |
| 本地 manifest | `results/multidataset_manifests/` | 五组官方 bank/query manifest 已生成；test 合计 23,256 |
| 已发布大证据 | `analysis/` 下从 `artifacts/frozen/` 恢复的 7 个原字节 JSON/CSV | `scripts/verify_handoff.py` 通过 |
| TAM 作者源码 | `.tools/tam_official_v1`，commit `7c7df1d3df418bb9467ca68ff2f00ff1f26a6f4c` | 五个源码/文档文件与仓库收据 SHA 一致 |

本机 NVIDIA 驱动为 572.90，GPU 为 RTX 5080 Laptop，显存 16,303 MiB。**CUDA 与模型真实加载未测试**，因此上述核验是文件与 CPU 环境准备完成，不等于本机正式实验兼容通过。InternVL 四块权重共 16.15 GB，原脚本要求 BF16 模型整体 `.to('cuda')`；在这张 16 GB 卡上按原协议运行尤其可能显存不足。

## 离线复查入口

在 PowerShell 仓库根目录：

```powershell
& .\.venv\Scripts\python.exe scripts/verify_handoff.py
$env:CUDA_VISIBLE_DEVICES='-1'
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
& .\.venv\Scripts\python.exe -m pip check
& .\.tools\venv-internvl\Scripts\python.exe -m pip check
```

本次这些检查全部通过：发布自检包含 147 个原件或压缩证据、7 个已恢复大文件、30 组 seed 73 结果与 23,256 个官方 query；CPU 单元测试 18/18 通过。下载使用 Git 历史中的 `scripts/download_datasets.ps1`（本地副本在被 Git 忽略的 `.tools/download_datasets.ps1`），原脚本核验各档案 MD5/SHA。`datasets/`、`.hf_cache/`、`.tools/`、`.venv/`、`results/` 均被 Git 忽略。

## GitHub 克隆无法补齐的原始材料

公开仓库没有已完成实验的原始 prediction JSONL、npy/pt source 与 checkpoint、完整云归档、私有 `.tools` 操作包及旧运行凭据。本机也没有 `handoff/ARTIFACTS.md` 所述的原维护者目录。因此可以离线核查发布的审计、结果、协议和原始查询表，但不能声称已从本机重放完整历史实验或全部原始审计。相关缺件和取回办法见 [handoff/ARTIFACTS.md](handoff/ARTIFACTS.md)。

Git 工作树中的原科学代码、冻结协议和审计未改动。本地环境和数据可用于以后经明确恢复的工作；当前 TAM 没有实际 64-run/热图，InternVL 和新增 Qwen seed 也没有新的科学结果。
