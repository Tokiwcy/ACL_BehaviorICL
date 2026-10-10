# 下一步：先解释任务条件化，再补稳定性

## 最新执行优先级（2026-10-10）

用户已明确恢复本地InternVL全量实验，当前继续CUB DeTriever→RICES，然后Dogs→Pets，四组动态共24格。Aircraft已完成6格，CUB已完成Raw/Proj/Learn/GPT-MM四格；DTD旧单patch另6格直接保留在五数据集表。全部完成后独立CPU核验、更新配对差值并暂停该自动化。Qwen历史主表不重跑。

本地新版TAM与四提示检索实验已经有报告，先读 [LOCAL_GPU_STUDY.md](LOCAL_GPU_STUDY.md)与 [INTERNVL35_LOCAL_STUDY.md](INTERNVL35_LOCAL_STUDY.md)，保留空间归因不支持一致前景迁移、词表混杂与小样本局限。旧云端64-run TAM、三seed74/75、LFM均不因仓库发布而恢复。下文是2026-10-06保留计划，涉及“尚无本地maps/全部暂停”的语句仅描述当时状态。实际下一操作须读取本机状态，不能用GitHub快照判断活跃worker。

> GitHub接手版：仓库内链接可移植；原科学source/audit字节保留。完整本地归档、图片、权重及云运行状态不在此仓库，参见 [artifact availability](handoff/ARTIFACTS.md)。


更新：2026-10-06。**这是暂停期间保留的研究计划，不是执行指令。** 自动监督 PAUSED；不申请GPU、不继续TAM、不恢复InternVL/三seed。之后仅在人类明确恢复相应工作时执行。

## 第一优先：完成可信的 query+image TAM pilot

**问题**：同一图片、不同 query，哪些空间响应和回答前状态发生变化？变化属于图像 raw CAM、文本 ECI 校正，还是最终输入位置的任务响应？

复用 [已锁定协议](analysis/query_tam_pilot_v1/protocol.json) 和 [当前TAM代码](CURRENT_CODE.md)。保留八张SHA固定图片、四个query、自然生成/固定金标签两种模式，不按结果挑图。固定目标是teacher-forced probe，不当成模型自然预测。

恢复后先完成一张协议内图片的最小实际兼容检查：模型/processor/revision、视觉grid和pixels、目标token位置、原TAM函数及逐层状态形状。自然词变化与固定目标分开保存；确认同图不同query的预处理字节一致，再执行正式64runs。这个检查只验证实现，不提前宣称机制结论。

交付应包括：

- 八图的固定尺度对照图；raw CAM、ECI校正后图、生成词和固定目标词分别显示；全部图都交付，包括不变化或不支持假设的图。
- 各图/query的实际token IDs、grid/pixel/source/model身份；raw/post-ECI的cosine/JSD/top-region overlap，以及36层anchor-state变化指标。
- 完整64run、无重复/无缺失、FULL归档/字节核验与精确停机后CPU审计；原始maps可重新作图，CPU不重放模型。

**解释规则**：image在query之前时，后置query不能反向改变此前图像token的因果decoder状态；同目标raw CAM不变化可能符合结构预期。ECI图变化不能单独等同于attention移动；自然输出词变了也不能单独证明空间关注变了。如果仅anchor state变化，可用它说明任务条件化响应，而不强行讲空间重分配。

最近V2在container初始化期限内未获得SSH，没有执行worker。旧V1/V2都不重调；新操作必须先验证实际可用、可连通容量和最小兼容链。保持一个科学worker、及时备份/停机，不让付费GPU等待CPU整理。

## 第二优先：把解释图接到已有的 improvement

TAM pilot只看source query+image；原来的提升来自 **source prompt变化 → demo检索变化 → 同一下游任务ICL**。二者之间还缺检索层面的连接。

先用已取回的 [任务提示实验](analysis/paper_task_prompt_complete_results_v1) 做CPU证据整理：同query的task/generic top4交集与顺序、类别构成、分数、预测task-only/generic-only分层。对TAM八图展示实际被检索的demo及标签，不能仅画一张漂亮query热图。

若要进一步区分“任务指令”与“类别词表”效应，应另立预声明对照，而不是改变原冻结实验或用test选query/参数。后续固定demo/交换demo条件可以更直接检验上下文效用；其范围与成本先定义，再执行。

完成条件：同一个query能追溯prompt、表示指标、四张demo与最终结果；报告相关和干预的边界，不从热图直接推导收益因果。

## 第三优先：把论文贡献说清楚

Methods使用“目标VLM的任务条件化逐层回答前状态”，说明36层最后输入位置、共享投影与bank-only SupCon。不要使用“整个推理过程”“第一层就是视觉编码器”或“decoder普遍更优”的实现外表述。

补一份以原论文/官方实现为依据的related-work矩阵：外部视觉检索、生成/末层表示检索、逐层hidden-state检索、task-conditioned/model-aware demonstration selection。比较输入条件、提取位置、层聚合、监督来源、是否同目标VLM、验证范围。当前尚未完成这份novelty对照，不能宣称first。

现有论文可以先按 [PAPER_NARRATIVE.md](PAPER_NARRATIVE.md) 写出方法和已完成结果；TAM只在实测审计后加入结果。负面视觉控制、harm和覆盖反例也进入论文。

## 后续验证：三seed与第二VLM，均保持原范围但暂缓

| 项目 | 保留的科学范围 | 恢复前还缺什么 | 最终交付 |
|---|---|---|---|
| Qwen 三seed | 73/74/75，五组×六方法；73重核已有30组；新74/75共279072预测、100独立tuple | 原Linux CPUvalidator90秒超时路径的真实阶段诊断；完整新启动链检查，原CLIP offline-main/NumPy修复不丢失 | 新60组完整test，连同73共90组；按数据集/方法给三seed均值和sample std |
| InternVL 第二VLM | 五组，zero/RICES/Raw/Proj，seed73；55357图、23256query、93024预测 | 真实checkpoint两position_ids与模型非持久buffer值严格相等证明；原loader wrapper/parent/runner/包身份全链检查 | native视觉/hidden/合法多demo smoke后正式source、训练、完整预测及CPU pairedCI |

旧队列原顺序为InternVL→第一VLM三seed，当前均暂停。TAM之后是否恢复或重排，由人类重新确定；本计划不会自动更换顺序或申请卡。失败不是科学完成，旧消耗入口不能当可用启动器。

Intern的窄CLIP修复已有合成CPU测试，只允许删除**完全相等的两个非持久位置buffer**，保留checkpoint字节、全部learned tensors和strict=True。真实buffer还未核验；不能用strict=False绕过。first的超时根因仍Unknown，不能只抬timeout就重新跑。

## 哪些事情不再花时间做

已完成的主表、提示、标签、QV不重跑。早期velocity/宽度网格/旧两组Intern不恢复。旧失败盘取证保留为独立溯源工作，不再阻塞人类授权的全新独立卡；不重复无新证据的容量探测或已过CPU测试。kNN、文本/音频推广可放后续，不是当前TAM交付的前置条件。

暂停证明和失败证据入口见 [RUN_STATUS.md](RUN_STATUS.md)。下一次开始工作时先读本页、代码地图、实际部署/停机收据，避免重复启动历史入口。
