# 当前实验计划入口

> GitHub接手版：仓库内链接可移植；原科学source/audit字节保留。完整本地归档、图片、权重及云运行状态不在此仓库，参见 [artifact availability](handoff/ARTIFACTS.md)。


2026-10-06：人类暂停实验，要求整合。当前计划以 [NEXT_STEPS.md](NEXT_STEPS.md) 为准；此页不再累积几十版启动修复。

## 保留的研究顺序

| 优先级 | 工作 | 状态 |
|---|---|---|
| 1 | 同图改变query的TAM/逐层状态解释pilot，64runs | 暂停；官方代码、协议与CPU实现已有，无实际热图 |
| 2 | 将已有task/generic表示、top4 demo变化与最终ICL收益连接 | 计划；已有完整预测与source身份，不重跑已完成实验 |
| 3 | 完善论文framing和primary-source related-work对照 | 当前整合已提供叙事；novelty矩阵尚未完成 |
| 后续 | Qwen五组三seed稳定性、InternVL五组跨模型验证 | 原科学范围保留；暂停自动接续，先解决已知启动链问题 |

TAM优先于InternVL是人类最新的研究决定。此前冻结科学队列仍保留 [queue.json](analysis/paper_supervision_queue_v3/queue.json) 和 [canonical policy](scripts/behavior_paper_queue_v3_canonical_v4_policy.py)，本次不改其字节，也不由它触发恢复。原Intern→first顺序如需重排，之后由人类确定。

已完成的seed73主表、提示、标签、QV不重跑；固定256/完整bank/bank-only dev/无test调参保持。所有旧已消耗cloud entry不重复。未来仅在人类明确恢复后，先实际兼容和精确身份检查，再云执行；最多两单GPU、每卡唯一科学worker、本机不运行权重/CUDA，完整归档后停机再CPU审计。

当前信息分工：想法读 [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)，代码读 [CURRENT_CODE.md](CURRENT_CODE.md)，结果读 [CURRENT_RESULTS.md](CURRENT_RESULTS.md)，停机/故障读 [RUN_STATUS.md](RUN_STATUS.md)。

历次部署准备、冻结测试和故障时间线见 [原计划归档](handoff/ARTIFACTS.md#local-only-artifacts)。这些是来源记录，不是当前待执行指令。
