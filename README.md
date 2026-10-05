# BehaviorICL

> GitHub接手版：仓库内链接可移植；原科学source/audit字节保留。完整本地归档、图片、权重及云运行状态不在此仓库，参见 [artifact availability](handoff/ARTIFACTS.md)。


**当前主线：用目标 VLM 在任务文本与图片上的逐层回答前状态，选择对该 VLM 有用的 demonstration。**

项目已按 2026-10-06 的人类要求暂停。自动监督为 PAUSED，最新整合快照确认运行 GPU 为 0；TAM、InternVL 和三 seed 均不会自动恢复。

**新 coding agent 从 [HANDOFF.md](HANDOFF.md) 开始，并遵守 [AGENTS.md](AGENTS.md)。** 首次检查只需 Python 3.12 标准库，无需图片、模型权重、GPU 或云凭据：

```bash
python scripts/verify_handoff.py
```

该检查核验随仓库提供的原代码/审计 SHA、压缩原始表、结果计数、依赖与文档链接。完整实验复现还需要 [外部 artifacts](handoff/ARTIFACTS.md)。下一优先任务是固定图片改变 query 的 TAM 对照；目前没有实际热图。

| 你想看什么 | 当前入口 |
|---|---|
| 想法、方法、已得到的证据与缺口 | [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) |
| 当前代码分别实现什么 | [CURRENT_CODE.md](CURRENT_CODE.md) |
| 完整结果、配对 CI、分母和原始证据 | [CURRENT_RESULTS.md](CURRENT_RESULTS.md) |
| 下一步先做什么、怎样判断有结果 | [NEXT_STEPS.md](NEXT_STEPS.md) |
| 论文可以怎样讲 | [PAPER_NARRATIVE.md](PAPER_NARRATIVE.md) |
| 停机状态与剩余工程问题 | [RUN_STATUS.md](RUN_STATUS.md) |

当前已完成：Qwen 五数据集 seed 73 主表、完整纠错/匹配分析、DTD+CUB 任务提示对照、固定 demo 标签干预，以及 DTD+CUB QwenVision+SupCon 视觉控制。**TAM 尚无实际热图；InternVL 和新增 seed 74/75 尚无完整科学结果。**

早期探索与启动排障不再作为当前方法入口。必要依赖保留在当前树，退役的早期公开文件保留于 Git 历史；本地原文件、锁、失败盘及整理前文档单独保留，见 [历史说明](handoff/ARTIFACTS.md#local-history)。机器可读的代码依赖、证据 SHA 和精简 CSV 在 [整合索引](analysis/current_project_consolidation_v1/evidence_index.json)。
