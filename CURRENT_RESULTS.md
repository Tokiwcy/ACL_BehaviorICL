# 当前可用结果与证据

> GitHub接手版：仓库内链接可移植；原科学source/audit字节保留。完整本地归档、图片、权重及云运行状态不在此仓库，参见 [artifact availability](handoff/ARTIFACTS.md)。


更新：2026-10-06。此处只把已有独立核验结果整合为当前表格，未新增训练、预测或统计重跑。pp 表示百分点；下列配对区间来自原 10000 次 query bootstrap，seed 73，95% 描述性 CI，未做多重比较校正。

## 1. 主表：Qwen 五数据集，seed 73，完整官方 test

| 方法 | DTD | Aircraft | CUB | Pets | Dogs | 五数据集等权均值 |
|---|---:|---:|---:|---:|---:|---:|
| RICES | 78.72 | 59.68 | 71.21 | 91.52 | 74.32 | 75.09 |
| GPT-MM | 81.60 | 69.37 | 72.64 | 92.86 | 78.68 | 79.03 |
| DeTriever（任务适配） | 78.24 | 68.38 | 68.81 | 92.40 | 77.68 | 77.10 |
| BehaviorICL-Raw | 84.04 | 73.84 | 81.50 | 94.96 | 82.97 | 83.46 |
| BehaviorICL-Proj | 85.90 | 78.67 | 83.36 | 95.42 | 85.20 | 85.71 |
| BehaviorICL-Learn | 86.49 | 78.28 | 83.83 | 95.42 | 85.07 | 85.82 |
| 官方 test 数 | 1880 | 3333 | 5794 | 3669 | 8580 | 共23256 |

这是 **30 个完整组的单 seed 表**，不是三 seed mean±std。Proj 与 Learn 很接近，Learn 并非每个数据集更高；目前不能把学习层权重写成必不可少的收益来源。DeTriever 是任务适配实现，训练预算与 Behavior 不同，不称作者官方复现或全方法同预算对照。

来源：[计数 CSV](analysis/paper_p1_recheck_v1/full_test_counts.csv)、[P1 审计](analysis/paper_p1_recheck_v1/audit.json)。审计核验 102 项来源 SHA、完整 query 和计数；没有重放全部检索 kernel，原数组并非全部本地可用，分离的 retrieval/ICL 时间也未恢复。

## 2. 任务提示 vs 通用提示：完整 DTD+CUB

改变的是 **bank/query source capture 的 prompt**，使用同一 Raw 检索规则；下游 ICL 的分类任务保持一致。task 同时包含任务指令和完整类别词表，generic 为 `Describe the image briefly.`。

| 数据集 | task 正确数/全test | generic 正确数/全test | task/generic 准确率% | task−generic，pp [95% CI] | task-only / generic-only 正确 |
|---|---:|---:|---:|---:|---:|
| DTD | 1579/1880 | 1483/1880 | 83.99 / 78.88 | +5.11 [3.51, 6.76] | 171 / 75 |
| CUB | 4704/5794 | 2717/5794 | 81.19 / 46.89 | +34.29 [32.86, 35.69] | 2218 / 231 |

已取回并核验 34856 个全新 source 和 15348 条正式预测，完整 test/配对 CI 均通过。这支持 **任务指令与词表联合条件化的检索效用**；尚不能单独归因于哪部分文本或空间注意变化。它也不是直接把最终生成 query 换成通用提示的实验。

来源：[accuracy](analysis/paper_task_prompt_complete_results_v1/full_official_accuracy.csv)、[paired effects](analysis/paper_task_prompt_complete_results_v1/paired_task_generic.csv)、[独立完整审计](analysis/paper_task_prompt_complete_results_v1/audit.json)。新 task arm 与历史 Raw 使用的环境/source 不完全一致，不当成历史预测逐字复现。

## 3. 固定 demo 标签干预：完整 DTD+CUB

锁定同一 query 的四张 demo 图片及其顺序，只替换 demo 标签：正确标签 Gold、demo 原合法 zero 预测标签 own、确定性错误标签 wrong；另有同环境约束 zero。计划 7674×4=30696 条记录，实际生成 30526 条，**own 预定缺失 170 条完整记录在案**。

| 数据集 | Gold−own 可用配对数 | Gold/own 配对正确数 | 配对准确率差，pp [95% CI] | own 预定缺失 |
|---|---:|---:|---:|---:|
| DTD | 1809 | 1554 / 1211 | +18.96 [16.92, 20.95] | 71 |
| CUB | 5695 | 4741 / 2191 | +44.78 [43.34, 46.16] | 99 |

**完整 test 已完成，不代表 Gold−own 的分母就是全部 test。** 这两项效应用同 query 可用配对；不把缺失 own 预测擅自补成错误或删去其缺失证据。

| 数据集 | 同环境约束 zero 准确率% | Gold 全test准确率% | wrong 全test准确率% | Gold 伤害初始正确 query |
|---|---:|---:|---:|---:|
| DTD | 66.76（1255/1880） | 85.90（1615/1880） | 8.40（158/1880） | 52/1255 |
| CUB | 37.37（2165/5794） | 83.36（4830/5794） | 13.89（805/5794） | 161/2165 |

支持的干预结论限定为：**固定这些上下文时，demo 标签内容显著影响结果**。它不独立证明 Behavior 检索特有价值、same-failure 因果或相同推理过程；Gold 仍会造成 harm，标签长度作用也未独立分离。

来源：[各 arm 全表](analysis/paper_full_test_labels_complete_results_v1/arm_summary.csv)、[配对效应](analysis/paper_full_test_labels_complete_results_v1/paired_label_effects.csv)、[独立完整审计](analysis/paper_full_test_labels_complete_results_v1/audit.json)。主要 zero 参照是本实验同环境约束 zero；历史 P0 自由 zero 只作敏感性分析，不混用分母。

### 保留的五组×256标签实验：样本证据，不是五组完整test

| 数据集 | 预先固定 query 数 | Gold−own 可用配对 | Gold−own，pp [95% CI] |
|---|---:|---:|---:|
| DTD | 256 | 244 | +22.54 [16.80, 28.28] |
| Aircraft | 256 | 246 | +48.78 [41.46, 55.69] |
| CUB | 256 | 247 | +49.39 [42.51, 55.87] |
| Pets | 256 | 256 | +10.55 [6.25, 14.84] |
| Dogs | 256 | 240 | +24.17 [17.50, 30.83] |

共 1280 个固定 query、5120 条计划 arm 记录。Gold 有 harm，Pets/Dogs 的部分比较中 harm 高于 own。它补充跨数据集证据，不能替代完整 test。来源：[sampled effects](analysis/paired_label_sampled_results_v1/paired_label_effects.csv)、[sampled audit](analysis/paired_label_sampled_results_v1/audit.json)。

## 4. QwenVision+SupCon：强视觉控制，完整 DTD+CUB

这是目标 VLM 视觉模块 pooled feature 的固定256 SupCon 检索控制，随后仍做四示例 ICL。全部 17428 新 source、两次1500步训练、7674正式预测及合法4demo smoke完成，FULL归档全SHA核验、停机后独立CPU审计通过。

| 数据集 | 新 QwenVision 正确数/全test | 新 QV准确率% | 历史 fixed256 Proj准确率% | QV−Proj，pp [95% CI] |
|---|---:|---:|---:|---:|
| DTD | 1604/1880 | 85.32 | 85.90 | -0.59 [-1.81, 0.64] |
| CUB | 4886/5794 | 84.33 | 83.36 | +0.97 [0.07, 1.85] |

**没有支持 decoder 表示普遍优于监督视觉表示。** 新 QV 与历史 Proj 在环境、batch/source 来源上存在差异，CI 只描述这两个已记录结果的 query 配对，不消除环境混杂。

来源：[accuracy](analysis/paper_qwenvision_two_complete_results_v1/full_official_accuracy.csv)、[paired CI](analysis/paper_qwenvision_two_complete_results_v1/paired_accuracy_CI.csv)、[完整审计](analysis/paper_qwenvision_two_complete_results_v1/audit.json)。原配对 CSV 的 `first_rescued` 在该表实际表示全 query 正确计数，不是初始错误上的 rescue。

## 5. 完整纠错与 strict-match：解释性支持与反例

P0 主分母包含全部官方 test 的自由 zero 初始错误，包括非法输出。`rescue` 是初始错而 ICL 对，`harm` 是初始对而 ICL 错。最终正确数恒等于初始正确−harm+rescue。

| 数据集 | 初始错误数（含非法） | Proj rescue | Proj−RICES rescue，pp [95% CI] | Proj / RICES harm（共同初始正确分母） |
|---|---:|---:|---:|---:|
| DTD | 664（50） | 453/664，68.22% | +11.45 [7.98, 14.91] | 54 / 113，分母1216 |
| Aircraft | 2262（125） | 1662/2262，73.47% | +21.09 [18.79, 23.30] | 111 / 267，分母1071 |
| CUB | 3633（60） | 2829/3633，77.87% | +14.67 [13.02, 16.30] | 160 / 331，分母2161 |
| Pets | 645（24） | 523/645，81.09% | +5.12 [1.71, 8.37] | 46 / 156，分母3024 |
| Dogs | 3934（347） | 2917/3934，74.15% | +13.52 [11.92, 15.07] | 253 / 654，分母4646 |

strict-match 的事后定义：query 与 demo 真类别相同，且两者原 zero 的**合法错误预测类别相同**。非法/未知保留，unknown 不当成 without-match。这不是在线使用 query 金标签的检索算法。

Proj 的 strict 覆盖高于 RICES，但不能说覆盖单独解释收益：GPT-MM 在 DTD/Pets 覆盖更高却 rescue 较低；固定同类别 demo 数量后，没有一致支持独立 same-failure 作用。

### QV 反例：同一 P0 错误代理下的七方法完整对照

共 7674 query×7方法=53718 条完整记录，原六方法计数已实际复现。

| 数据集 | Proj / QV strict覆盖数 | Proj−QV覆盖，pp [95% CI] | QV / Proj rescue数 | QV−Proj rescue，pp [95% CI] |
|---|---:|---:|---:|---:|
| DTD | 383 / 367，分母664 | +2.41 [0.00, 4.82] | 446 / 453，分母664 | -1.05 [-3.77, 1.66] |
| CUB | 2838 / 2633，分母3633 | +5.64 [4.38, 6.91] | 2890 / 2829，分母3633 | +1.68 [0.41, 2.95] |

CUB 中 Proj 覆盖更高、QV 总 rescue 更高；DTD 覆盖差的 CI 包含零。不同方法的有/无 match 条件子组不同，不能用子组差当干预因果。自由 zero 与约束 ICL、新旧环境差异仍保留。

来源：[P0 完整审计](analysis/paper_repair_five_dataset_v1/audit.json)、[repair summary](analysis/paper_repair_five_dataset_v1/all_error_repair_summary.csv)、[repair paired CI](analysis/paper_repair_five_dataset_v1/paired_all_error_differences.csv)、[QV match audit](analysis/paper_qwenvision_match_control_v1/audit.json)、[七方法 paired effects](analysis/paper_qwenvision_match_control_v1/paired_differences.csv)。

## 6. 尚无可报告科学结果的项目

| 项目 | 已有材料 | 没有完成什么 |
|---|---|---|
| TAM query+image | 官方代码固定commit、八图64run协议、adapter/worker/CPU图表代码与测试 | 无实际64run、无实际热图、无空间/hidden变化结论；最后一次初始化失败已停机 |
| InternVL 五数据集 | 原计划/科学代码、完整失败FULL与CPU审计、部分合成CPU修复 | native smoke未通过；正式source提交0、训练/预测未完成；预分配NPY不算结果 |
| Qwen 三 seed | seed73主表、74/75冻结方案/科学代码 | 新74/75未完成；不能给三seed均值或标准差 |

失败来源见 [RUN_STATUS.md](RUN_STATUS.md)，未来研究顺序见 [NEXT_STEPS.md](NEXT_STEPS.md)。

## 7. 原始结果和整合文件在哪里

提示原session：[czscr… task-prompt](handoff/ARTIFACTS.md#local-only-artifacts)；标签原session：[yuj… full-labels](handoff/ARTIFACTS.md#local-only-artifacts)；QV原session：[t0ku… qv37](handoff/ARTIFACTS.md#local-only-artifacts)。原预测、来源身份和审计引用保留，不只保留论文表格。

[机器可读证据索引](analysis/current_project_consolidation_v1/evidence_index.json) 记录各audit/plan/CSV实际SHA；[summary_facts.json](analysis/current_project_consolidation_v1/summary_facts.json) 保留未四舍五入数值；[精简CSV目录](analysis/current_project_consolidation_v1) 的CSV是原表逐字节复制。旧详细分解见 [RESULTS_PAPER.md](RESULTS_PAPER.md)，当前结论与状态以本页为准。
