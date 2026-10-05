# 新 coding agent 接手指南

这是2026-10-06的当前研究快照。**先读本页，运行离线自检，再开始用户授权的工作。** 仓库更新不等于恢复实验；云端/自动监督仍暂停。

## 你需要理解的一句话

外部图像相似性未必等于目标VLM可利用的示例效用。BehaviorICL让冻结目标VLM联合处理任务文本与图片，用36层最后输入位置的回答前状态检索top4，再让同一VLM使用原始demo图片与标签进行ICL。Raw、固定256 Proj、Learn的区别见 [方法介绍](PROJECT_OVERVIEW.md) 和 [代码地图](CURRENT_CODE.md)。

## 目前真的有什么结果

| 工作 | 实际状态 |
|---|---|
| Qwen五组六方法主表 | seed73的30组完整test已核验；没有三seed std |
| task/generic source prompt | DTD+CUB完整test，+5.11/+34.29pp；改变检索source，最终ICL任务不变 |
| 固定demo标签 | 完整DTD+CUB，Gold−own +18.96/+44.78pp；配对n1809/5695，缺失71/99保留；另有五×256样本 |
| QwenVision+SupCon | DTD85.32/CUB84.33%；历史Proj对照不支持decoder普遍更优 |
| strict-match/rescue | 完整回顾分析；覆盖有GPT-MM/QV反例，非独立因果 |
| TAM | **无实际maps**；官方代码/八图64run协议/CPU适配与分析代码已有 |
| InternVL / 新seed74、75 | 未完成；GPU均已精确停止，失败不当科学结果 |

完整表及CI读 [CURRENT_RESULTS.md](CURRENT_RESULTS.md)。本次发布包括原audit、汇总CSV、标签query-arm输出，以及压缩的完整query-level匹配CSV；不包括权重、图片和原始npy/pt/云端FULL。

## 第一次接手：无需GPU的具体步骤

```bash
python scripts/verify_handoff.py
python scripts/materialize_handoff_artifacts.py
python scripts/verify_handoff.py
```

第一条验证发布文件、原始SHA、30组计数、配对分母、TAM协议和可移植链接；第二条仅离线恢复已发布gzip里的大JSON/CSV；第三条还会验证已恢复原件身份。缺数据/模型时，不运行原科学preflight冒充兼容。

接着读 [TAM源与索引说明](handoff/TAM_UPSTREAM.md)，检查 `scripts/query_tam_adapter_v1.py`、`run_query_tam_pilot_v1.py`、`analyze_query_tam_pilot_v1.py`。若已有CPU torch/numpy/scipy环境，可下载固定上游并执行三个合成CPU测试：

```bash
python scripts/fetch_tam_upstream.py --fetch
python -m unittest discover -s tests -p test_query_tam_pilot_v1.py -v
```

这些测试不载入预训练权重、不使用CUDA、不访问RunPod。环境和原科学版本记录见 [ENVIRONMENT.md](handoff/ENVIRONMENT.md)。没有相应CPU依赖时，先完成标准库自检，不安装/运行GPU模型来“测试”。

## 第一项研究工作：固定图改变query的TAM

协议是 `analysis/query_tam_pilot_v1/protocol.json`：DTD/CUB各4图，按历史task-only、generic-only、都对、都错分层SHA固定选图。四query×自然生成和固定gold teacher-forcing两模式=64runs。保留原Qwen3模型revision、processor、实际pixels/grid和tokenIDs。

**当前第一安全动作**是审阅/CPU测试适配器、列出缺失的八图与固定模型文件，再准备最小实际兼容检查方案。当前pause下不申请卡。若用户明确恢复TAM，先用协议内一图检查模型/native、输入/目标索引、raw/post-ECI/36层shape、真实备份，再运行完整64runs；不是把旧V2重新执行。

完整交付：八图全部固定尺度热图、自然词/固定目标分开、raw CAM/post-ECI分开、每层anchor变化指标、完整maps/token/grid身份及FULL取回/停机/CPU审计。负结果和不变图同样交付，不根据热图挑例子。

注意：image-before-query时，固定目标的raw image-token CAM可能不随后置query变化；ECI文本校正图变化不等于attention移动。原accuracy提升来自source prompt→demo变化→ICL，需进一步连接该query实际top4及预测，不能仅凭热图解释所有improvement。详见 [NEXT_STEPS.md](NEXT_STEPS.md)。

## 已知工程问题，避免重复绕圈

| 问题 | 证据和处理边界 |
|---|---|
| TAM V2 container初始化超期限 | 没有SSH/native/worker，已停止；V1/V2 consumed。新卡操作需要新身份及实际连通/最小兼容，不能直接重调 |
| Intern V75 CLIP strict load两个position_ids键 | 完整失败FULL已核验；合成CPU窄buffer修复已写，真实checkpoint buffer相等性和完整runner/parent/包链尚未证明。禁止strict=False |
| first V76原Linux CPUvalidator90s超时 | 原因Unknown，缺远端FULL；不能用Windows配置速度猜原因或单纯抬timeout |

修复参考：`behavior_clip_position_ids_compat_v78.py`、`behavior_internvl_clip_runtime_v79.py`；它们是CPU准备，**不是新的可启动云入口**。旧NumPy首次exec顺序已在真实Linux证明修复，不重新把它当未解决故障。详情 [RUN_STATUS.md](RUN_STATUS.md)。

## 仓库边界与后续更新

原冻结计划和科学代码字节保留；大锁/CSV可离线恢复。部分原consumer仍有本地路径和私有`.tools`要求，它们是历史合同消费者，不是fresh clone即可完整执行的云流水线。用新版本的可移植适配替代，不改历史SHA来掩盖缺件。

如需完整科学复核，请向维护者取得 [外部artifact清单](handoff/ARTIFACTS.md) 中的原图片/预测/source/checkpoint/FULL，核SHA后再审计。本仓库的CI只核发布内容，不宣称重放全部模型或kernel。

早期不必要的public脚本已从当前tree移除，可由Git历史恢复；必要旧文件名共享依赖保留。优先TAM及其检索证据连接，三seed/Intern按人类新优先级再恢复，不自动接续旧科学队列。
