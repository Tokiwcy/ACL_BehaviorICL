# BehaviorICL 本地 GPU：TAM、检索消融与论文定位

更新：2026-10-06。所有新增模型计算在本机 RTX 5080 Laptop（16 GB）上完成；未使用云端 GPU。原仓库历史审计与主表保持原样。本报告的新实验是机制与小 bank 检索分析，不是重新跑五个官方 test 的四示例 ICL。

随后新增的 dense InternVL3.5-4B 五数据集检索和四示例小样本 ICL、答案状态代理对照另见 [INTERNVL35_LOCAL_STUDY.md](INTERNVL35_LOCAL_STUDY.md)。

## 1. 研究问题与实际实现

BehaviorICL 将图片和任务文本交给同一个冻结 Qwen3-VL-4B-Instruct，在**生成答案之前**保存最后输入位置的 36 层 decoder 状态（每层 2560 维）。Raw 用每层余弦相似度等权平均检索四张原始训练图片与金标签；Proj/Learn 在 bank 上用标签监督训练 256 维检索头。最终仍由该 VLM 做 ICL 分类。这里的“过程”有严格边界：它是最后输入位置的逐层状态，不是所有视觉 patch、整段生成推理或注意力权重。方法和历史完整 test 见 [项目介绍](PROJECT_OVERVIEW.md)、[当前结果](CURRENT_RESULTS.md)。

本次用官方 [TAM](https://github.com/xmed-lab/TAM) 固定提交 `7c7df1d3df418bb9467ca68ff2f00ff1f26a6f4c`，以及原仓库已审计的首生成词索引适配器，检验改变任务文本后图像归因是否更贴合独立空间标注。模型固定为 Qwen3-VL-4B-Instruct revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`。本地模型、CLIP、InternVL、五数据集与环境安装见 [本地安装说明](LOCAL_SETUP.md)。

## 2. 样本、标注、指标

五组官方 train/trainval bank 和 test 均已准备并核对全部 55,357 张图像。TAM 的主比较以 SHA 确定性选出**每组 128 张 test 图**，合计 640 张、1,280 次固定目标运行。DTD/CUB 各有 4 张沿用原历史正确/错误分层图；其余图按预先固定的 SHA 排序扩展并尽量增加类别覆盖。空间分析的类别覆盖为 DTD 47、Aircraft 100、CUB 127、Dogs 120、Pets 37。方案、每图 SHA 与完整结果分别在 [TAM 协议](analysis/query_tam_local_v3/protocol.json)、[TAM 分析](analysis/query_tam_local_v3/analysis.json)。

主对比保持**同一图片、相同处理后像素/视觉网格、相同金标签的首个 token**，只改 `Describe the image briefly.` 与简短分类任务文本。固定 token 是 teacher forcing 探针，不是模型预测。另在每组前 8 张跑完整类别词表提示与自然贪心首 token；共保存 1,400 份 TAM 记录。TAM 保存原始 CAM、ECI 后值、最终图和 36 层状态。图像先于文本进入因果 decoder，所以固定目标下的原始视觉 token CAM 理论上可能几乎不变；ECI 后图的改变也不能直接叫注意力转移。

Aircraft、CUB、Dogs 用原数据集的目标框，Pets 用 trimap 的前景值 1；将标注缩到实际视觉 token 网格。指标为 `目标区域热图质量占比 − 目标区域面积占比`，比较任务提示减通用提示，单位百分点。面积扣除用于避免大框天然获得更多质量。DTD 是纹理图片，没有同类独立目标框，报告热图相似度和空间熵，不把“铺满全图”定义为正确关注。区间是对所选 128 图做 10,000 次配对 bootstrap 的描述性 95% 区间；扩样发生在看到 32 图初步结果之后，不能把区间当成预注册的显著性检验。

## 3. TAM 的五数据集结果

| 数据集 | 图数 | 两提示 TAM 余弦均值 | 原始 CAM 完全相同 | 末层状态余弦均值 | 目标区域质量变化，百分点 [95% 区间] |
|---|---:|---:|---:|---:|---:|
| DTD | 128 | 0.9939 | 107/128 | 0.706 | 无独立区域标注 |
| Aircraft | 128 | 0.9866 | 126/128 | 0.619 | −0.518 [−0.733, −0.307] |
| CUB | 128 | 0.9943 | 108/128 | 0.641 | −0.091 [−0.279, +0.084] |
| Dogs | 128 | 0.9908 | 117/128 | 0.598 | −0.314 [−0.538, −0.094] |
| Pets | 128 | 0.9899 | 114/128 | 0.606 | −0.084 [−0.399, +0.221] |

在全部 640 对中，**572 对原始 CAM 完全一致**。最终 TAM 图通常略变，但四个有标注的数据集没有一致的前景占比提升。Aircraft 与 Dogs 的均值略降，CUB/Pets 的区间跨零。DTD 的归一化空间熵为通用 0.8814、任务 0.8785；这只能描述分布，不判断纹理证据是否合理。回答前状态却明显改变：五组的层 1 余弦均约 0.997–0.998，层 36 降至 0.598–0.706。前 8 张/组的完整类别提示仍无一致区域提升；自然输出两提示的首 token 在 **0/40** 图上相同，说明直接比较自然生成热图会混入目标 token 身份变化。[控制分析](analysis/query_tam_local_v1/controls_analysis.json)、[DTD 熵分析](analysis/query_tam_local_v3/texture_spread.json) 与[固定顺序示意图](analysis/query_tam_local_v2/contact_sheet.png)可复核。

这项结果**不支持**“当前 image-first 管线让 query 普遍把视觉归因移向更合理目标区域”。目标框也较粗，不能排除模型对框内某个鉴别部位改善；TAM 是 token 归因而非真实注意力权重或因果干预。合理的下一步是同图的目标部位标注、等面积遮挡干预和 text-before-image 顺序对照，随后重新验证检索与 ICL 输出。

## 4. 更贴近论文主张的检索消融

使用与 TAM 前 32 张/组相同的 test 图。bank 从**全部类别各取 2 张官方训练图**，只按训练 sample ID 的 SHA 排序，不使用测试标签挑候选；五组 bank 数为 94/200/400/240/74，合计 1,008 张。四种提示对每张 bank/test 图提取 36 层回答前状态，合计 4,672 次 GPU 特征提取。固定计划与机器结果在[初始消融协议](analysis/query_retrieval_local_v1/protocol.json)、[提示拆分协议](analysis/query_retrieval_local_v2/protocol.json)、[预先列明的层方法分析](analysis/query_retrieval_local_v2/analysis.json)。TAM 与检索对共有 160 张图×2 提示的状态，在 float16 下**320/320 完全一致**。

下表用原项目 Raw 的**逐层 cosine 等权平均**规则重算同类 Top-4 命中率；这是冻结特征上的补充 CPU 诊断，区别于初始协议预列的末层、池化和 velocity 方法。完整逐组 Top-1、Top-4、AP 与配对区间见[原 Raw 规则诊断](analysis/query_retrieval_local_v2/original_raw_diagnostic.json)。

| 数据集，每组 test n=32 | 通用描述 | 短任务指令，无词表 | 原完整任务提示 | 仅类别词表 |
|---|---:|---:|---:|---:|
| DTD | 59.4% | 87.5% | 90.6% | 90.6% |
| Aircraft | 18.8% | 78.1% | 78.1% | 78.1% |
| CUB | 25.0% | 75.0% | 81.3% | 84.4% |
| Dogs | 9.4% | 87.5% | 93.8% | 93.8% |
| Pets | 21.9% | 87.5% | 93.8% | 93.8% |

同一规则下，Top-1 从通用到完整任务提示依次为 DTD 31.2→78.1%、Aircraft 9.4→59.4%、CUB 12.5→68.8%、Dogs 6.2→65.6%、Pets 12.5→90.6%；两张同类正例的平均精度 AP 依次为 31.3→79.3%、11.7→64.4%、11.6→69.3%、6.5→67.5%、14.7→86.4%。这些指标与 Top-4 方向一致。

原先预列的**第 18 层、第 36 层、归一化层池化、层间变化池化**也已全部计算；短任务提示下第 18 层 Top-4 仅为 DTD 40.6%、Aircraft 21.9%、CUB 21.9%、Dogs 12.5%、Pets 31.3%，第 36 层为 87.5%、78.1%、84.4%、96.9%、93.8%。层间变化池化的短任务结果为 87.5%、78.1%、78.1%、87.5%、90.6%，并未稳定超过末层。**原论文方法不是 velocity**；这只是消融。进一步事后探索 `任务状态 − 通用状态` 的逐层相似度也未稳定超过原状态，[单列结果](analysis/query_retrieval_local_v2/query_delta_exploratory.json)不作为主贡献。

四种提示的结果表明，**给同一目标 VLM 一个分类形式的文本，确实能强烈改变检索表征与同类示例排序**。完整提示与仅类别词表在这个小 bank 上几乎一样，短指令也常接近两者。因此不能声称“完整类别词表和任务指令的独特协同”已被识别。这个结果也没有重跑完整 bank、四示例生成或最终准确率；完整官方 test 的历史 seed 73 表仍以 [CURRENT_RESULTS.md](CURRENT_RESULTS.md) 为准。历史 DTD/CUB 完整 test source-prompt 对照分别有 +5.11/+34.29 个百分点，但它同时改变了指令与词表。

## 5. Novelty 与 GPT-MM 的准确差异

[GPT-MM 原文 §6.4/§7.2](https://www.jstage.jst.go.jp/article/jnlp/33/1/33_207/_pdf)已经把**图片/音频与问题一起输入生成模型**，用答案预测位置的最终层状态做余弦检索，并给示例添加标签诱导的解释。因此不能 claim “首个把 query 和 image 联合送进 VLM 检索”，也不能 claim “GPT-MM 没考虑任务”。本仓库的 GPT-MM 列是本地 **GPT-MM-style 代理实现**，不应写成作者官方代码复现。已有 [DeTriever](https://aclanthology.org/2025.coling-main.544/) 在文本 NL2SQL 使用加权多层 decoder 状态，故“首个使用多层/全过程 hidden states”也不成立。

可守住的表述是：

> **We retrieve multimodal demonstrations from the target VLM's task-conditioned, layerwise pre-answer states. Unlike an answer-state retriever, our representation is available before decoding and lets the frozen model's response to the image and task guide example selection.**

补充贡献是 bank-only 的轻量 SupCon 检索头与五个细粒度图像数据集上的完整 seed 73 ICL 比较，而不是“视觉 attention 被任务文本转向”的机制。历史五组等权平均 Raw/Proj/Learn 为 83.46/85.71/85.82%，GPT-MM-style 为 79.03%，但仅单 seed；Proj/Learn 很接近。DTD/CUB 的 QwenVision+SupCon 完整控制达到 85.32/84.33%，不支持“decoder 普遍优于目标模型视觉特征”。这些边界见[当前结果](CURRENT_RESULTS.md)。

如要把 “利用处理 query + image 的全过程” 做成更强的新方法，应明确定义并验证**跨层交互轨迹中哪部分对 demo 效用有独立贡献**，而非把 36 层状态改名为全过程。优先实验是：在相同 bank、同一 VLM、同一 ICL 模板和训练预算下比较末层/逐层/层间变化、原始状态/任务减通用状态、GPT-MM-style 答案状态、QwenVision+SupCon；完整五组 test、多个 seed 和第二 VLM 都要完成。若论文主打空间机制，还需独立的部位标注与遮挡干预。目前证据更强的故事是**query 条件化检索改善示例选择，但 TAM 未证明前景注意转移**。

## 6. 可复核路径与运行状态

- [TAM 本地 runner](scripts/run_query_tam_local_v1.py)、[TAM 分析器](scripts/analyze_query_tam_local_v1.py)、[完整/自然控制](scripts/analyze_query_tam_controls_v1.py)。原云端 runner 与历史科学源码未改。
- [检索 GPU runner](scripts/run_query_retrieval_local_v1.py)、[四提示 CPU 分析](scripts/analyze_query_retrieval_local_v2.py)、[原 Raw 规则诊断](scripts/analyze_query_retrieval_original_raw_v1.py)、[跨流程状态核验](scripts/audit_query_retrieval_tam_alignment_v1.py)。
- 原始新增运行数组在 `results/query_tam_local_v1/` 与 `results/query_retrieval_local_v1/`，两目录由 Git 忽略；协议、分析、报告和图片可见于项目工作区。若移动工作区，请同时保留这两目录。
- 已执行代码语法检查、TAM 适配器 3 个 CPU 单元测试、全 640 对 TAM 的 SHA/网格/固定词核验、四提示检索完整进度/协议核验，以及 320 份共享状态精确一致性检查。`scripts/verify_handoff.py` 的原发布校验也通过；其 `TAM_actual_results: 0` 专指旧云端 64-run 协议，本报告的 1,400 个本地新版本 run 单独存放。云端 GPU 仍保持暂停。

