# 当前论文叙事

> GitHub接手版：仓库内链接可移植；原科学source/audit字节保留。完整本地归档、图片、权重及云运行状态不在此仓库，参见 [artifact availability](handoff/ARTIFACTS.md)。


更新：2026-10-06。当前叙事采用人类最新的任务条件化动机；覆盖原来以same-failure为中心的表述。数字来源集中在 [CURRENT_RESULTS.md](CURRENT_RESULTS.md)，实现集中在 [CURRENT_CODE.md](CURRENT_CODE.md)。

## 论文要讲的一个问题

视觉相似性适合描述图片之间的关系，但不直接衡量“哪个示例对当前目标VLM解决这个任务更有用”。示例选择应考虑目标模型如何响应任务与图片。BehaviorICL因此用冻结目标VLM的任务条件化、逐层回答前状态定义检索空间，再让同一VLM使用检索出的四个示例预测。

可用工作标题：**BehaviorICL: Task-Conditioned Demonstration Retrieval from a VLM's Layerwise Pre-Answer States**。这是标题候选，不是已确认novelty声明。

可作introduction/abstract基础的英文段落：

> Visual similarity need not identify the demonstrations most useful to a target vision-language model on a particular task. We investigate demonstration retrieval based on the target model's own task-conditioned internal responses. BehaviorICL extracts hidden states at the final input position across decoder layers before answer generation, and retrieves examples using either an untrained layerwise similarity or a lightweight bank-supervised metric. The retrieved images and labels are then supplied to the same frozen model for in-context prediction. Full-test experiments on five image-classification datasets show gains under the recorded single-seed protocol. Paired source-prompt and fixed-context label experiments further examine task conditioning and context utility. A strong supervised native-vision control reaches comparable performance, motivating a claim about task-conditioned retrieval rather than universal superiority of decoder representations.

不能将“逐层回答前状态”替换为“完整生成推理/整个decision过程”：当前代码没有采集完整生成链条，也没有证明内部状态等同于可解释的人类决策步骤。

## 方法与证据的写作顺序

1. **问题与方法**：外部CLIP图像相似性基线；目标VLM联合处理任务指令、类别词表和图片；固定位置的36层状态；Raw、固定256 Proj与Learn；bank-only训练、dev选checkpoint、完整bank top4与同VLM ICL。
2. **主表**：五组完整test六方法seed73；明确尚无三seed方差。Proj和Learn的接近说明共享投影贡献强，不能把层权重说成唯一关键。
3. **任务条件化**：完整DTD/CUB任务source相对generic +5.11/+34.29pp。最终ICL任务prompt一致；任务指令与词表联合变化，不拆成已验证的单因素机制。
4. **上下文效用**：固定demo图片/顺序，Gold−own的可用配对 +18.96/+44.78pp；明确缺失分母与harm。这是标签内容干预，不是same-failure检索的独立因果试验。
5. **表示控制和负结果**：QwenVision+同类SupCon在DTD接近，在CUB高于历史Proj；不主张decoder普遍更好。新旧环境差异保留。
6. **纠错诊断**：完整rescue/harm、strict-match覆盖与未知情况；GPT-MM和QV反例、固定同类数量控制一并报告。覆盖帮助描述检索，但不能单独解释性能。
7. **可视化解释与推广**：TAM待实测，InternVL和新增seed待完成。未完成内容不进入results，不画虚构热图或填写计划数字。

## 每个主张需要哪份证据

| 可以写的主张 | 已有证据 | 写作边界 |
|---|---|---|
| 目标VLM逐层状态能构造有效检索器 | 主表Raw明显高于RICES，Proj/Learn进一步提高 | 单模型、单seed；历史环境限制 |
| 任务条件化source对示例选择有用 | 完整paired task/generic结果 | 指令与词表共同变化；不是直接输出prompt的对照 |
| 正确demo标签有重要上下文作用 | 固定图片/顺序的标签干预 | own可用配对；Gold仍有harm；非检索唯一性证明 |
| 错误匹配覆盖可以解释一部分现象 | P0及QV完整回顾分析 | 不独立因果，unknown保留，覆盖有反例 |
| query改变空间或逐层响应 | 目前只有TAM代码、协议和CPU测试 | 无实测结论；先完成64runs并区分raw与ECI |
| 对不同seed和VLM普遍成立 | 当前尚无完整追加结果 | 不能提前写泛化、稳定性或mean±std |

## TAM如何放进论文才有帮助

若实测支持，可以展示同图、不同query、固定目标token下的raw/post-ECI图与36层anchor差异，再展示该query的实际task/generic检索demo。它对应“任务条件化的模型响应”，不直接证明注意力因果或全部accuracy提升来源。

若同目标raw图不变、只ECI或anchor变化，也应完整报告；这符合image-before-query的因果结构，不能为了论文故事改变选图。未固定目标词的自然生成图可作为补充，不作为排除token混淆的主证据。

## Novelty与限制仍需完成

当前贡献候选是输入条件、逐层表示、轻量检索学习与完整验证的组合。需要再对照既有model-aware、hidden-state、task-conditioned和multimodal demonstration retrieval原论文/实现。只把特征从CLIP换成VLM hidden state，或用“整个decision过程”包装，尚不足以证明novelty。

保留的限制包括：固定256/4-shot、bank标签监督与模型冻结、DeTriever任务适配而非官方复现、缺少完整历史kernel重放和分离时耗、单seed、自由zero与约束ICL口径差异、环境/batch差异、非法/unknown、harm和标签own缺失。它们不阻止报告已完成结果，但决定主张的范围。

当前暂停，研究优先级和完成条件见 [NEXT_STEPS.md](NEXT_STEPS.md)。旧叙事原文已保存在 [归档](handoff/ARTIFACTS.md#local-only-artifacts)，不再与此页并列作为当前主线。
