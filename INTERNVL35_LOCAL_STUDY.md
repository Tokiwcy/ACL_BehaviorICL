# 第二个 dense VLM：InternVL3.5-4B 的五数据集本地验证

更新：2026-10-06。全部新增计算使用本机 RTX 5080 Laptop（16 GB）与 BF16 权重，未启动云端 GPU。这里报告的是**每数据集 32 张官方 test 图、每类 2 张官方训练图**的检索与四示例 ICL 面板，不是原论文的全 bank、全 test 主表。历史 Qwen 主表仍见 [CURRENT_RESULTS.md](CURRENT_RESULTS.md)；此前本机 Qwen/TAM 研究见 [LOCAL_GPU_STUDY.md](LOCAL_GPU_STUDY.md)。

## 模型选择与运行条件

选择官方 [OpenGVLab/InternVL3_5-4B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-4B-HF)，固定 revision `6bd4487402110ef9889ba50eb7aefeb302526fed`。官方模型系列将此型号列为视觉约 0.3B、语言约 4.4B、合计约 **4.7B 的 dense 模型**，Apache-2.0。它使用 Qwen3 语言模型作为组件，因此是另一套多模态模型与训练管线，**不是独立语言骨干家族的验证**。本机没有接受访问许可的 Hugging Face 凭据，故未选需要授权的 Gemma 3 4B。使用标准 HF 格式、Transformers 5.13.0、torch 2.8.0+cu128；没有量化。

InternVL 默认允许每图最多 12 个 448×448 视觉块。四示例加 query 的首个 DTD 样例会成为 33 个视觉块、8,691 个输入 token。为让检索与五图 ICL **使用同一图像处理**并适应 16 GB 显存，正式 v2 协议统一固定 `image_processor.max_patches = 1`：每图 1 个 448×448 块，256 个视觉 token。五组各抽取 train/test 图检查均确实为一个块。默认切块的 v1 只在 DTD 与部分 Aircraft 完成探索计算，[v1 协议](analysis/internvl35_retrieval_local_v1/protocol.json)与可恢复的本地缓存保留，以下结论只用 v2。

## 固定面板与指标

[v2 冻结协议](analysis/internvl35_retrieval_local_v2/protocol.json)逐样本复用 Qwen 本地研究的图片、SHA、类别词表、顺序和四种提示，且模型 revision、图像预算单列。bank/test 数依次为 DTD 94/32、Aircraft 200/32、CUB 400/32、Dogs 240/32、Pets 74/32。每类恰有 2 张 bank 图片，测试标签没有用于检索。五组合计 1,008 张 bank、160 张 test；四提示共 **4,672 次**回答前状态提取。每张图使用完整的 36×2560 最后输入位置 decoder 状态，不生成答案。原项目 Raw 评分是 36 层**分别归一化做余弦，再等权平均**。Top-1/Top-4 是是否检出同类，AP 是两张同类 bank 图片的平均精度。所有状态经过形状、协议 SHA、完成计数与有限值检查。

四提示为：通用图片描述、无标签词表的短任务句、完整任务指令加标签词表、仅标签词表加输出格式。四示例 ICL 固定同一分类提示与合法标签约束，仅替换前两种检索源选出的四张原始图片及其金标签；下表还加入完整任务检索源与答案状态源。

## 五组检索结果

原 Raw 等权逐层余弦的 Top-4 同类命中率；每格分母均为 32。

| 数据集 | 通用 | 短任务 | 完整任务 | 仅标签词表 | 答案状态代理 |
|---|---:|---:|---:|---:|---:|
| DTD | 71.9% | 75.0% | 84.4% | 84.4% | 65.6% |
| Aircraft | 25.0% | 37.5% | 31.3% | 28.1% | 28.1% |
| CUB | 21.9% | 34.4% | 43.8% | 43.8% | 40.6% |
| Dogs | 9.4% | 46.9% | 65.6% | 62.5% | 50.0% |
| Pets | 25.0% | 46.9% | 90.6% | 84.4% | 84.4% |
| 等量五组平均 | 30.6% | 48.1% | 63.1% | 60.6% | 53.8% |

完整任务减通用的 Top-4 配对差值，单位百分点，及描述性 95% bootstrap 区间：DTD +12.5 [−3.1,+28.1]，Aircraft +6.3 [−9.4,+21.9]，CUB +21.9 [+3.1,+40.6]，Dogs +56.3 [+37.5,+75.0]，Pets +65.6 [+46.9,+84.4]。Top-1 与 AP 的四提示结果同存在[完整检索分析](analysis/internvl35_retrieval_local_v2/analysis.json)。同一数据中完整任务减仅标签词表的 Top-4 差值是 0、+3.1、0、+3.1、+6.3 点；各组配对区间均包含 0。**分类标签词表本身解释了大部分检索变化，不能把改善独占归因于任务句与图像的特定交互。** Aircraft 的短任务甚至高于完整任务，不能宣称更多文本总是更好。

作为跨模型方向检查，[逐查询配对 Qwen 比较](analysis/internvl35_retrieval_local_v2/qwen_comparison.json)在完全相同图片、bank 与提示上显示 Qwen 的通用→完整 Top-4 为 59.4→90.6、18.8→78.1、25.0→81.3、9.4→93.8、21.9→93.8%。InternVL 五组也全部正向，但 Aircraft/CUB 显著弱于 Qwen。两模型视觉预处理预算与训练不同；这些绝对分数不是受控的架构优劣排名。InternVL 又包含 Qwen3 语言组件，不能称为独立语言家族复制。

## 四示例 ICL 是否受益

每种检索源都向**同一个 InternVL、同一个四示例分类提示**提交四张选中的图片与金标签，使用相同合法标签约束。三列都是每组 32 张 test 图的正确数。

| 数据集 | 通用检索 | 完整任务的回答前 Raw | GPT-MM-style 答案状态 |
|---|---:|---:|---:|
| DTD | 22/32 | 23/32 | 16/32 |
| Aircraft | 4/32 | 6/32 | 4/32 |
| CUB | 6/32 | 9/32 | 7/32 |
| Dogs | 5/32 | 11/32 | 10/32 |
| Pets | 13/32 | 23/32 | 22/32 |
| 合计 | **50/160** | **72/160** | **59/160** |

回答前 Raw 减通用检索是 +13.75 个百分点，配对 95% bootstrap 区间 [+7.5,+20.6]；27 张由错变对，5 张由对变错。回答前 Raw 减答案状态代理是 +8.13 点，[+1.88,+14.38]；20 张由错变对、7 张由对变错。[逐查询 ICL 审计](analysis/internvl35_answer_state_local_v1/icl_comparison.json)保留五组配对区间：Raw 对答案状态的最大差异在 DTD (+7/32)，Aircraft/CUB/Dogs/Pets 仅 +1 至 +2/32，因此不能声称五组各自均有确定优势。完整任务与通用检索选出的四例平均只有 0.8 张相同；160 张中 77 张获得更多同类示例、11 张变少，[选择与下游审计](analysis/internvl35_retrieval_local_v2/selection_icl_audit.json)保留逐组计数。

## 表征与 GPT-MM 差异的边界

对完整任务提示，Raw / 单独第 36 层 / 归一化 36 层状态池化 / 相邻层差分池化的 Top-4 依次为：DTD 84.4/78.1/90.6/87.5，Aircraft 31.3/37.5/31.3/28.1，CUB 43.8/46.9/43.8/46.9，Dogs 65.6/68.8/65.6/71.9，Pets 90.6/93.8/96.9/90.6%。这些事后层对照见[检索分析](analysis/internvl35_retrieval_local_v2/analysis.json)。**逐层 Raw 并未稳定优于末层或其他聚合**；“利用全过程”不能仅以层数来 claim 独特有效机制，更不能称为完整生成推理过程。方法准确对象是答案前最后输入位置的逐层状态。

本地 [GPT-MM-style 代理协议](analysis/internvl35_answer_state_local_v1/protocol.json)模仿仓库既有实现：相同完整任务提示下贪心生成最多 12 token，从生成答案位置取末层状态，再以余弦检索。五组 Top-4 均低于回答前 Raw，但除 DTD 外配对区间跨 0 或碰 0；[答案状态检索分析](analysis/internvl35_answer_state_local_v1/analysis.json)和同一 ICL 下游表构成**本地代理对照**，不是 GPT-MM 作者代码或官方复现。GPT-MM 原文已经联合输入图片与问题进行检索，[先前研究定位](LOCAL_GPU_STUDY.md#5-novelty-与-gpt-mm-的准确差异)解释了为什么不能 claim “第一个把 query+image 给 VLM”。此处更可守的差异是**何时、取哪些状态**：在解码前取任务条件化的逐层状态，而代理取生成答案后的末层状态。

此前 Qwen 的 [TAM 空间归因实验](LOCAL_GPU_STUDY.md#3-tam-的五数据集结果)没有发现一致的合理目标区域迁移；本次没有把 Qwen 专用 TAM 适配器硬套到 InternVL。因此本结果支持“query 文本改变检索表征、示例与部分 ICL 预测”，**不证明**视觉注意力因 query 转向更合理部位。

## 复核入口与限制

- 正式检索：[GPU runner](scripts/run_internvl35_retrieval_local_v2.py)、[CPU 分析](scripts/analyze_internvl35_retrieval_local_v2.py)、[Qwen 配对比较](scripts/compare_internvl35_qwen_local_v1.py)。
- 四示例：[Raw/通用 runner](scripts/run_internvl35_icl_local_v2.py)、[配对分析](scripts/analyze_internvl35_icl_local_v2.py)、[示例选择审计](scripts/analyze_internvl35_selection_local_v2.py)。
- 答案状态代理：[GPU runner](scripts/run_internvl35_answer_state_local_v1.py)、[检索分析](scripts/analyze_internvl35_answer_state_local_v1.py)、[四示例 runner](scripts/run_internvl35_answer_state_icl_local_v1.py)、[三源 ICL 分析](scripts/analyze_internvl35_icl_three_arm_v1.py)。
- 原始 `.npy`、进度和 prediction JSONL 在本机 `results/internvl35_*`，模型在 `.hf_cache/models--OpenGVLab--InternVL3_5-4B-HF`；两目录被 Git 忽略。协议与汇总 JSON 在 `analysis/` 可见。原全 test 的 23,256 张查询、全训练 bank、Learn/Proj retriever、多个 seed、第二模型 TAM 均未在此面板运行。所有区间仅描述固定的每组 32 张测试样本，并非预注册显著性声明。
