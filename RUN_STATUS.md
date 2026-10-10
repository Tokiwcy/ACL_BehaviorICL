# 运行状态：本地InternVL继续，云端暂停

## 最新本地状态（2026-10-10发布快照）

本地InternVL joint_v2已获用户明确继续授权，当前CUB DeTriever全量预测中；动态结果10/24格独立审计完成，DTD旧单patch另6/6。完整准确率及带时间进度以 [CURRENT_RESULTS.md](CURRENT_RESULTS.md)第8节为准。半小时监督自动化ACTIVE，唯一科学GPU worker与固定Windows桌面上下文已核对。云端GPU仍暂停，LFM已取消，旧单patch流水线冻结。

最近真实蓝屏为2026-10-09 09:08 HYPERVISOR_ERROR 0x20001；与03:31计划Windows升级重启分开记录。具体驱动/固件根因尚未定位，缓存清理只是运行保护，不能称为修复。用户暂停与恢复均保留原状态字节和独立操作receipt；当前入口见 [HANDOFF.md](HANDOFF.md)。新真实系统崩溃不自动循环恢复。

下文为2026-10-06历史云端停机快照，仍作为当时证据保留，不是当前本地GPU实时状态。

> GitHub接手版：仓库内链接可移植；原科学source/audit字节保留。完整本地归档、图片、权重及云运行状态不在此仓库，参见 [artifact availability](handoff/ARTIFACTS.md)。


更新：2026-10-06。人类要求“暂停一下，把当前想法、代码、结果和下一步整合”。本页只保留当前状态；历次部署与排障时间线已原样归档。

## 暂停与停机的实际证明

自动监督 `behavioricl` 已设置为 **PAUSED**。整合期间重新只读核验账户API和本机CIM：**运行GPU 0、科学/启动controller与watch/CPUwait 0**。没有新增云动作。实际时间、脱敏账户字段和暂停状态见 [pause_receipt.json](analysis/current_project_consolidation_v1/pause_receipt.json)。

恢复工作必须由人类明确提出；不自动启动TAM、InternVL或三seed，不重复已有launch。原云盘、科学锁、旧源码、失败收据和已完成结果保留。

## 科学项目当前状态

| 项目 | 状态 | 下一工作性质 |
|---|---|---|
| Qwen seed73五组六方法 | 完整30组已核验 | 直接用于单seed主表 |
| 五组纠错/strict-match诊断 | 完整回顾分析已核验 | 保留定义、harm、unknown及反例 |
| DTD+CUB任务提示对照 | 完整test、source/预测取回及配对CI完成 | 连接表示、demo与TAM解释 |
| 固定demo标签干预 | 完整DTD+CUB及五组×256样本均核验；own缺失显式保留 | 两个范围分开报告 |
| DTD+CUB QwenVision+SupCon | FULL/停机/独立CPU审计完成 | 作为强视觉控制，不重跑 |
| query+image TAM | 代码/64run协议和CPU测试完成；无实际maps | 暂停；未来优先取得真实模型兼容和64run |
| InternVL五组 | 未完成，正式source提交与预测均0 | 暂缓工程修复与跨VLM验证 |
| Qwen新增seed74/75 | 未完成，无完整追加组 | 暂缓启动链诊断与三seed验证 |

当前科学数字见 [CURRENT_RESULTS.md](CURRENT_RESULTS.md)；不把安装、图片验收、模型shard加载、预分配数组、smoke或CPU测试当正式完成。

## 最近三项工程失败，只保留需要记住的内容

| 项目/实际身份 | 已知故障与取证范围 | 已停止/入口状态 |
|---|---|---|
| TAM V2，`5cngvs2t9ahe5p / 2c970df6411e458a92932827e564738e` | provider初始化/等待container超期限；未取得SSH，native/worker未运行，无maps、无FULL科学归档 | 2026-10-05 16:19:12.419316 UTC精确EXITED/stopped；V2消耗，不重调 |
| Intern V75，`pdfex2yk5w4zz0 / 4734bd71bdee40ebb8e7c75e0fabe678` | 完整模型/env/55357图已验；CLIP strict load拒绝text/vision两个position_ids键，首个source提交前失败；完整失败FULL已取回并CPU审计 | 14:55:06.453020 UTC精确停止留盘；V75消耗，不重调 |
| first V76，`g6v816yvh7cd4n / 1418faa9319a4c63aef45f7b0a2ea9ac` | 原CPUvalidator SSH90秒超时，根因Unknown；发生在父进程请求前；只有本地控制审计，远端FULL/最终状态未完整取证 | 15:04:02.698232 UTC精确停止留盘；V76消耗，不重调 |

对应原证据：[TAM controller_failure](handoff/local_evidence/tam_controller_failure.json)、[TAM stop](handoff/local_evidence/tam_stop_confirmation.json)、[Intern完整失败审计](analysis/paper_internvl_five_startup_failure_v77/audit.json)、[first控制审计](analysis/paper_qwen_three_seed_startup_failure_v77/audit.json)。

Intern的 [CLIP buffer兼容helper](scripts/behavior_clip_position_ids_compat_v78.py) 和 [原loader窄wrapper](scripts/behavior_internvl_clip_runtime_v79.py) 只有合成CPU证据，真实checkpoint buffer等值和完整parent/runner/包链尚未完成。[review](analysis/paper_startup_repairs_CPU_v78/review.json) 明确没有新的可启动科学entry。已修复并实际证明的NumPy exec不当成未修复旧故障。

旧hmkv失败盘及更早少数FULL待容量的盘保留；增量不冒充FULL，旧API变化原因仍Unknown。旧盘取证与未来独立新卡分开，不再把旧盘容量当新卡科学的阻塞条件。

## 成本快照

暂停时独立GPU段累计估算 **US$23.0581**：此前已结束段US$22.965898649844448，加TAM V2独立段US$0.09217733472585678。这是运行记录估算，不是账单，未含存储；不重复相加历史cum或单段stop baseline，也不计停机间隙/明确容量拒绝零段。详细数值和源stop SHA在 [暂停收据](analysis/current_project_consolidation_v1/pause_receipt.json)。GPU停机不代表旧盘存储成本为零。

原科学角色/部署/费用段authority仍是 [.tools/paper_active_jobs_v1.json](handoff/ARTIFACTS.md#local-only-artifacts)，TAM独立authority仍在 [.tools/query_tam_cloud_v2](handoff/ARTIFACTS.md#local-only-artifacts)。本次没有改原registry或科学队列。

旧运行日志：[RUN_STATUS原样归档](handoff/ARTIFACTS.md#local-only-artifacts)。下一步：[NEXT_STEPS.md](NEXT_STEPS.md)。
