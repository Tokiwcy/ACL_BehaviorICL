# 论文结果入口

> GitHub接手版：仓库内链接可移植；原科学source/audit字节保留。完整本地归档、图片、权重及云运行状态不在此仓库，参见 [artifact availability](handoff/ARTIFACTS.md)。


2026-10-06整合后，当前完整结果以 [CURRENT_RESULTS.md](CURRENT_RESULTS.md) 为准。那里统一列出主表、task/generic、完整和样本标签干预、QV控制、纠错/match反例，以及未完成项目；配对分母、CI和来源链接均保留。

## 仍可直接使用的证据

| 结果 | 实际范围 | 原始审计 |
|---|---|---|
| Qwen六方法主表 | 五数据集完整test，seed73，30组 | [P1](analysis/paper_p1_recheck_v1/audit.json) |
| 纠错与strict-match | 五组23256query，139536 query×method记录 | [P0完整分析](analysis/paper_repair_five_dataset_v1/audit.json) |
| 任务条件化source | DTD+CUB完整test，15348预测 | [prompt audit](analysis/paper_task_prompt_complete_results_v1/audit.json) |
| 固定demo标签 | DTD+CUB完整test；30696计划记录、170 own缺失 | [full-label audit](analysis/paper_full_test_labels_complete_results_v1/audit.json) |
| 固定标签样本 | 五组×256固定query，非完整test | [sampled audit](analysis/paired_label_sampled_results_v1/audit.json) |
| QwenVision+SupCon | 仅DTD+CUB完整test，7674预测 | [QV audit](analysis/paper_qwenvision_two_complete_results_v1/audit.json) |
| QV匹配比较 | DTD+CUB七方法53718记录 | [match control](analysis/paper_qwenvision_match_control_v1/audit.json) |

TAM无实际maps，InternVL和新增seed74/75未完成，不列结果数字。结论不能变成same-failure独立因果、decoder普遍更优或三seed稳定性证明。论文叙事见 [PAPER_NARRATIVE.md](PAPER_NARRATIVE.md)。

## 原详细分解仍保留

旧页约五百行的P0共同合法子集、条件bank参考、固定同类数量、无严格match仍rescue、各arm harm、P3可选kNN与缺项记录已逐字节存入 [整理前RESULTS_PAPER.md](handoff/ARTIFACTS.md#local-only-artifacts)。原CSV/audit没有移动。该归档用于查表和溯源；若其中旧计划状态与当前不同，以当前结果页和运行状态为准。

[证据SHA索引](analysis/current_project_consolidation_v1/evidence_index.json) 和 [未舍入统计数值](analysis/current_project_consolidation_v1/summary_facts.json) 可用于后续排版或独立复核。
