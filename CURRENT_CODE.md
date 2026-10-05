# 当前代码地图

> GitHub接手版：仓库内链接可移植；原科学source/audit字节保留。完整本地归档、图片、权重及云运行状态不在此仓库，参见 [artifact availability](handoff/ARTIFACTS.md)。


更新：2026-10-06。这里只列当前研究需要理解的科学代码和证据消费者。**文件可用于阅读，不代表云启动入口可以重跑。** 项目已暂停，旧已消耗 launch 不复用，也不在本机运行模型权重或 CUDA。

## 核心方法：先从这里读

| 功能 | 文件与具体入口 | 使用范围 |
|---|---|---|
| 数据集、bank/query、类别与有序身份 | [multidataset_protocol.py](scripts/multidataset_protocol.py) | 五数据集共用协议 |
| VLM 消息、任务文本、特征和生成 | [multimodal_model_adapter.py](scripts/multimodal_model_adapter.py)：`task_instruction`、`zero_shot_messages`、`prepare_batch`、`anchor_hidden_states`、`generate_legal_labels` | `states[1:]` 的最后输入位置；当前 Qwen 为 36×2560 |
| 合法标签约束 | [legal_label_decoding.py](scripts/legal_label_decoding.py) | ICL 的标签 trie 与 EOS |
| Raw 与 Learn | [run_dtd_decision_states.py](scripts/run_dtd_decision_states.py)：`zero_selections` 与学习分支 | Raw 等权 cosine；Learn 共享投影及层权重 |
| 共享训练内核 | [run_dtd_learned_velocity.py](scripts/run_dtd_learned_velocity.py)：**`StateRetriever`**、`supervised_contrastive_loss`、`train_variant` | 当前使用 StateRetriever；旧 VelocityRetriever 不是当前方法 |
| seed 73 Proj | [run_behavior_ablation.py](scripts/run_behavior_ablation.py)：`BehaviorControl('projection_only')` | 固定256共享 W、等权层；其他旧控制不是当前主表方法 |
| 主实验共享运行与基线 | [run_multidataset_main.py](scripts/run_multidataset_main.py)：`extract_anchor_cache`、CLIP/GPT-MM source、检索和 ICL | RICES/GPT-MM 与共享 I/O、feature/generation |
| 预答案 source 及任务适配 DeTriever | [prepare_detriever_input_states.py](scripts/prepare_detriever_input_states.py)、[run_detriever_output_proxy.py](scripts/run_detriever_output_proxy.py) | 已有任务适配基线；不是作者官方复现 |

`run_dtd_*` 是历史文件名，实际当前调用覆盖五数据集；不能从文件名推断当前科学范围。`run_dtd_learned_velocity.py` 的早期说明也不能代替现在被调用的 `StateRetriever` 定义。

`run_behavior_projection_dimension.py` 对应早期宽度探索，**不是 Proj 的当前入口**。当前固定 256，不恢复宽度网格、selected-width 或 test 选参数。

## 已完成控制：代码、协议、结果一一对应

| 实验 | 科学代码 | 协议及结果入口 |
|---|---|---|
| 任务 vs 通用 source prompt | [run_behavior_task_prompt.py](scripts/run_behavior_task_prompt.py)、[behavior_task_prompt_policy.py](scripts/behavior_task_prompt_policy.py) | [协议](handoff/ARTIFACTS.md#materialize-frozen-artifacts)；[完整审计](analysis/paper_task_prompt_complete_results_v1/audit.json) |
| 固定 demo 标签干预，完整 DTD+CUB | [run_behavior_full_test_labels.py](scripts/run_behavior_full_test_labels.py) | [query/arm 锁](handoff/ARTIFACTS.md#materialize-frozen-artifacts)；[完整审计](analysis/paper_full_test_labels_complete_results_v1/audit.json) |
| QwenVision+SupCon，DTD+CUB | [run_behavior_qwenvision_two_native_v1.py](scripts/run_behavior_qwenvision_two_native_v1.py) → [host](scripts/run_behavior_qwenvision_two_host_v1.py) → [scientific control](scripts/run_behavior_qwenvision_two_control_v1.py)；[fresh capture](scripts/behavior_qwenvision_two_fresh_capture_v1.py) | [两组计划](handoff/ARTIFACTS.md#materialize-frozen-artifacts)；[完整审计](analysis/paper_qwenvision_two_complete_results_v1/audit.json) |
| QV 的 strict-match/rescue 对照 | [analyze_behavior_qwenvision_match_control_v1.py](scripts/analyze_behavior_qwenvision_match_control_v1.py) | [53718 条 query×method 审计](analysis/paper_qwenvision_match_control_v1/audit.json) |

上述科学 source 和冻结 lock 保持原字节。结果汇总脚本没有重新生成预测或训练模型。

## TAM：当前优先、代码已准备、没有实际结果

| 顺序 | 文件 | 做什么 |
|---|---|---|
| 1 | [prepare_query_tam_pilot_v1.py](scripts/prepare_query_tam_pilot_v1.py) | 从已完成提示实验按 SHA 固定八图、四分层与 query；生成冻结协议 |
| 2 | [protocol.json](analysis/query_tam_pilot_v1/protocol.json) | 同图四 query×自然/固定金标签两模式；64 runs 的具体定义 |
| 3 | [官方 source](handoff/TAM_UPSTREAM.md)、[query_tam_adapter_v1.py](scripts/query_tam_adapter_v1.py) | TAM commit `7c7df1d3df418bb9467ca68ff2f00ff1f26a6f4c`；保留原函数，明确修正生成目标索引 |
| 4 | [run_query_tam_pilot_v1.py](scripts/run_query_tam_pilot_v1.py) | 云 worker；实际 model/input/source SHA、token/grid/pixels；保存 raw/post-ECI/maps 和逐层状态 |
| 5 | [analyze_query_tam_pilot_v1.py](scripts/analyze_query_tam_pilot_v1.py) | CPU 检查完整64run、固定目标、同pixel/grid、FULL与停机，输出固定尺度热图和变化指标 |
| 6 | [check_query_tam_live_v1.py](scripts/check_query_tam_live_v1.py) | 独立 TAM 只读监视；不启动任务 |

CPU 兼容/索引/归档测试见 [CPU review](analysis/query_tam_pilot_v1/CPU_review.json)。它不证明 Qwen3 GPU 生成兼容。`start_query_tam_cloud_v1.py` 和 `v2.py` 都已消耗；V2 初始化失败并停机，**没有可直接复用的 launch**。未来恢复需新操作身份及最小实际兼容验证。

## 暂缓的两个科学方向

| 方向 | 当前科学代码 | 状态及保留范围 |
|---|---|---|
| 第二 VLM 五组 | [run_behavior_internvl_five_v2.py](scripts/run_behavior_internvl_five_v2.py)、[adapter](scripts/behavior_internvl_five_v2_adapter.py)、[minimal training helper](scripts/behavior_internvl_native_training_v3.py) | [原计划](handoff/ARTIFACTS.md#materialize-frozen-artifacts) 保留；没有正式完整预测；先解决 CLIP 严格加载问题 |
| Qwen 三 seed 五组 | [run_behavior_qwen_three_seed.py](scripts/run_behavior_qwen_three_seed.py)、[新 seed Proj](scripts/run_behavior_qwen_three_seed_proj.py) | [原计划](handoff/ARTIFACTS.md#materialize-frozen-artifacts) 保留；73只重核既有30组，74/75尚未完成 |

这些文件是科学实现，不是现成可用的新云部署。旧 Intern V75、first V76 以及更早启动入口均不重调。CPU 修复组件和工程失败证据见 [运行状态](RUN_STATUS.md)，不混入科学结果。

## 结果消费者与保留策略

完整科学审计分别由 [P1 recheck](scripts/audit_behavior_p1_recheck.py)、[prompt CPU audit](scripts/analyze_behavior_task_prompt_complete_cpu.py)、[label CPU audit](scripts/analyze_behavior_full_test_labels_complete_cpu.py)、[QV CPU audit](scripts/analyze_behavior_qwenvision_two_complete_cpu_v3.py) 完成；未来 Intern/first 使用 [统一完整审计](scripts/analyze_behavior_queue_complete_cpu_v2.py)。CPU 审计不等于重放模型或全部检索 kernel。

早期 velocity、trajectory pilot、宽度网格、旧两组 Intern 计划，以及数十个版本的 cloud start/watch/forensic 不属于当前方法入口。必要静态依赖仍保留原字节；不再被精选代码引用的早期公开文件从当前树退役，并可从 Git 历史恢复。本地原文件和冻结包没有删除。当前不重构共享科学 kernel。

[静态依赖索引](analysis/current_project_consolidation_v1/current_code_dependency_index.json) 将精选入口与必要本地 import 依赖分开，并记录 SHA；动态 import 和冻结包消费者仍以原锁为准。这份索引是阅读地图，不是删除清单。

GitHub 接手工具均只用标准库：[verify_handoff.py](scripts/verify_handoff.py) 离线核验已发布字节、计数、依赖与链接；[materialize_handoff_artifacts.py](scripts/materialize_handoff_artifacts.py) 恢复原字节的大型 JSON/CSV，拒绝覆盖冲突；[fetch_tam_upstream.py](scripts/fetch_tam_upstream.py) 仅在显式 `--fetch` 时获取固定 SHA 的官方源文件。它们不启动云端或加载模型，不能代替原科学完整审计。
