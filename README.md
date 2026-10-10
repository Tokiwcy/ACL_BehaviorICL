# BehaviorICL

> **2026-10-10 发布更新**：Qwen五组seed73主表保留；新增本地TAM/检索研究、InternVL小样本验证，以及正在进行的InternVL全量六方法实验。当前完整结果及进度见 [CURRENT_RESULTS.md 第8节](CURRENT_RESULTS.md#8-internvl35-4b五数据集结果与四组重跑进度)，代码见 [CURRENT_CODE.md](CURRENT_CODE.md)，发布范围与离线复核见 [本次发布说明](handoff/LOCAL_PUBLICATION_20261010.md)。本地InternVL已获用户恢复授权；云端仍暂停，LFM取消。仓库记录是带时间的快照，不能代替本机实时进程状态。

> 2026-10-06 本机实验更新：Qwen 五数据集 TAM 与检索消融见 [LOCAL_GPU_STUDY.md](LOCAL_GPU_STUDY.md)；新增 dense InternVL3.5-4B 五数据集检索、四示例 ICL 与 GPT-MM-style 代理对照见 [INTERNVL35_LOCAL_STUDY.md](INTERNVL35_LOCAL_STUDY.md)。下文保留此前 GitHub 接手快照，关于 TAM/InternVL 尚未运行的语句只描述该旧快照；云端自动任务仍暂停。

> GitHub接手版：仓库内链接可移植；原科学source/audit字节保留。完整本地归档、图片、权重及云运行状态不在此仓库，参见 [artifact availability](handoff/ARTIFACTS.md)。


**当前主线：用目标 VLM 在任务文本与图片上的逐层回答前状态，选择对该 VLM 有用的 demonstration。**

2026-10-06历史整合时实验曾全部暂停。随后用户授权本地研究与InternVL恢复；当前授权和操作记录见 [HANDOFF.md](HANDOFF.md)。公开clone不能继承本机恢复授权或直接重调已消费启动器。

**新 coding agent 从 [HANDOFF.md](HANDOFF.md) 开始，并遵守 [AGENTS.md](AGENTS.md)。** 首次检查只需 Python 3.12 标准库，无需图片、模型权重、GPU 或云凭据：

```bash
python scripts/verify_handoff.py
```

该检查核验原2026-10-06 handoff的代码/审计SHA、压缩原始表、结果计数、依赖与文档链接。本次新增代码/证据另用 `python scripts/verify_local_publication_v1.py` 核验。完整实验复现还需要 [外部 artifacts](handoff/ARTIFACTS.md)。本地新版TAM已经有实测报告；旧云端64-run协议仍未完成，两者独立记录。

| 你想看什么 | 当前入口 |
|---|---|
| 想法、方法、已得到的证据与缺口 | [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) |
| 当前代码分别实现什么 | [CURRENT_CODE.md](CURRENT_CODE.md) |
| 完整结果、配对 CI、分母和原始证据 | [CURRENT_RESULTS.md](CURRENT_RESULTS.md) |
| 下一步先做什么、怎样判断有结果 | [NEXT_STEPS.md](NEXT_STEPS.md) |
| 论文可以怎样讲 | [PAPER_NARRATIVE.md](PAPER_NARRATIVE.md) |
| 停机状态与剩余工程问题 | [RUN_STATUS.md](RUN_STATUS.md) |

当前已完成：Qwen五数据集seed73主表、纠错/匹配分析、DTD+CUB任务提示对照、固定demo标签干预、QwenVision控制，以及本地新版TAM/小bank检索和InternVL五组32-query面板。InternVL全量实验已审计DTD六格、Aircraft六格和CUB四格；DeTriever/RICES及Dogs/Pets仍未完成。新增Qwen seed74/75也未完成。详细分母、设置差异和负面结果见各报告。

早期探索与启动排障不再作为当前方法入口。必要依赖保留在当前树，退役的早期公开文件保留于 Git 历史；本地原文件、锁、失败盘及整理前文档单独保留，见 [历史说明](handoff/ARTIFACTS.md#local-history)。机器可读的代码依赖、证据 SHA 和精简 CSV 在 [整合索引](analysis/current_project_consolidation_v1/evidence_index.json)。
