# BehaviorICL

## 合作者任务书：五 seed 主表与 ablation（2026-10-10）

**请先看这一节。交付顺序是：Qwen 五数据集五 seed 主表 → 组件消融 A → 提示拆解 B → 可选状态位置消融 C。** 以下是新增研究计划与接手说明；已有结果、待执行实验和代码准备程度分别标明。本次 README 更新没有启动新实验。当前本机 InternVL 工作由其独立监督器继续，云端保持暂停；合作者的新机器运行需要自己的执行身份，不能消费本机恢复 receipt。

### 0. 拿到仓库后的第一步

在 Python 3.12 环境的仓库根目录执行，以下命令均不加载模型、不调用 CUDA、不申请云资源：

```bash
git clone https://github.com/Tokiwcy/ACL_BehaviorICL.git
cd ACL_BehaviorICL
python scripts/verify_handoff.py
python scripts/materialize_handoff_artifacts.py
python scripts/verify_handoff.py
python scripts/verify_local_publication_v1.py
```

随后按顺序读 [HANDOFF.md](HANDOFF.md)、[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)、[CURRENT_RESULTS.md](CURRENT_RESULTS.md)、[CURRENT_CODE.md](CURRENT_CODE.md)、[NEXT_STEPS.md](NEXT_STEPS.md) 和 [外部材料清单](handoff/ARTIFACTS.md)。这些文件包含历史快照，最新状态看顶部日期；旧云端八图64-run TAM 未完成，本地新版1400份 TAM 记录已完成，两者不能合并。原始发布自检中的 `TAM_actual_results: 0` 专指旧64-run协议。

**公开 clone 已有代码、原协议、审计与汇总；图片、预训练权重、完整状态数组/训练 checkpoint/原始预测/FULL 归档和私有部署包需要另行准备。** 离线 materialize 只恢复发布的 gzip JSON/CSV。缺原始文件时可以读审计，不能声称重新完成原实验审计。

| 需准备的材料 | 精确要求 / 已有入口 |
|---|---|
| Qwen | `Qwen/Qwen3-VL-4B-Instruct`，revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`；模型与 processor 文件 SHA 核对原协议 |
| RICES 的 CLIP | `openai/clip-vit-base-patch32`，revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`；核对原 inventory |
| 五数据集 | 官方 bank/test，顺序、sample ID、标签、图片 SHA 与恢复后的旧计划 manifest 一致；数量见下表 |
| seed73 原始材料 | 从维护者取回主表30组的 source、selections、预测、checkpoint、环境记录；[原独立审计](analysis/paper_p1_recheck_v1/audit.json)仅是引用入口 |
| 运行环境 | [ENVIRONMENT.md](handoff/ENVIRONMENT.md)、[LOCAL_SETUP.md](LOCAL_SETUP.md)是已有环境记录；根目录 requirements 不包含完整 torch/CUDA 部署方案。新环境保存包版本、GPU/driver、dtype、processor 配置、模型文件身份 |

### 1. 已有证据：哪些已经完成，哪些不能当完整 ICL 结果

#### 1.1 主方法拆分：五数据集完整 test，单 seed73

| 方法 | DTD | Aircraft | CUB | Pets | Dogs | 五数据集等权均值 |
|---|---:|---:|---:|---:|---:|---:|
| RICES | 78.72 | 59.68 | 71.21 | 91.52 | 74.32 | 75.09 |
| GPT-MM-style | 81.60 | 69.37 | 72.64 | 92.86 | 78.68 | 79.03 |
| DeTriever（任务适配） | 78.24 | 68.38 | 68.81 | 92.40 | 77.68 | 77.10 |
| BehaviorICL-Raw | 84.04 | 73.84 | 81.50 | 94.96 | 82.97 | 83.46 |
| BehaviorICL-Proj | 85.90 | 78.67 | 83.36 | 95.42 | 85.20 | 85.71 |
| BehaviorICL-Learn | 86.49 | 78.28 | 83.83 | 95.42 | 85.07 | 85.82 |

单位为准确率%。Raw 是36层逐层 L2 后 cosine 等权平均；Proj 是共享256维投影、等权层；Learn 加 softmax 层权重。**Learn−Proj 等权均值约 +0.11pp**，Aircraft/Dogs 为负，Pets 持平。已有主表不能证明学习层权重必不可少，也不是五 seed mean±std。原计数及审计见 [full_test_counts.csv](analysis/paper_p1_recheck_v1/full_test_counts.csv)、[audit.json](analysis/paper_p1_recheck_v1/audit.json)。

#### 1.2 source 任务提示干预：完整 DTD+CUB

bank/query source 分别用完整 task prompt 与 `Describe the image briefly.`；Raw 检索规则相同，最终 ICL 分类提示相同。改变发生在**检索表示提取**，并未把最终分类任务改成图片描述。

| 数据集 | task 正确数/全test | generic 正确数/全test | task−generic，pp [95%配对CI] |
|---|---:|---:|---:|
| DTD | 1579/1880（83.99%） | 1483/1880（78.88%） | +5.11 [3.51, 6.76] |
| CUB | 4704/5794（81.19%） | 2717/5794（46.89%） | +34.29 [32.86, 35.69] |

已有34856份 source 和15348条正式预测，完整审计通过。**task 同时包含任务指令与完整类别词表**，两部分作用尚未单独识别；历史 task arm 与主表 Raw 的环境/source 不完全相同，不能逐行冒充主表重放。证据：[提示协议](artifacts/frozen/analysis/task_conditioning_prompt_plan_v1/protocol_lock.json.gz)、[准确率](analysis/paper_task_prompt_complete_results_v1/full_official_accuracy.csv)、[配对效应](analysis/paper_task_prompt_complete_results_v1/paired_task_generic.csv)、[完整审计](analysis/paper_task_prompt_complete_results_v1/audit.json)。

#### 1.3 四提示与层/池化：五数据集小 bank 检索诊断

每类 bank **2张**，每组 test **32张**；bank 数 DTD/Aircraft/CUB/Pets/Dogs 为94/200/400/74/240，总1008张。四提示共4672次特征提取。下表是原 Raw 规则的 **Top-4 中至少一张同类示例的 query 比例**，不是最终四示例生成准确率。

| 数据集，query n=32 | 通用描述 | 短任务指令，无词表 | 完整任务提示 | 仅类别词表 |
|---|---:|---:|---:|---:|
| DTD | 59.4 | 87.5 | 90.6 | 90.6 |
| Aircraft | 18.8 | 78.1 | 78.1 | 78.1 |
| CUB | 25.0 | 75.0 | 81.3 | 84.4 |
| Pets | 21.9 | 87.5 | 93.8 | 93.8 |
| Dogs | 9.4 | 87.5 | 93.8 | 93.8 |

预列的第18层、第36层、归一化层池化、层间变化池化也已有结果；最后层在部分条件下很强，多层 Raw 未稳定优于最后层。任务减通用状态另属事后探索，未稳定优于原状态。**原方法使用状态，不使用 velocity；现有小 bank 结果不替代完整 bank/full-test ICL。**

固定短任务提示时，同一批32-query的Top4-any结果如下（%）；池化向量cosine与逐层cosine均值是不同评分规则：

| 数据集 | 第18层 | 第36层 | 归一化36层均值池化 | 归一化层间变化池化 | 原Raw逐层cosine均值 |
|---|---:|---:|---:|---:|---:|
| DTD | 40.6 | 87.5 | 87.5 | 87.5 | 87.5 |
| Aircraft | 21.9 | 78.1 | 78.1 | 78.1 | 78.1 |
| CUB | 21.9 | 84.4 | 71.9 | 78.1 | 75.0 |
| Pets | 31.25 | 93.8 | 87.5 | 90.6 | 87.5 |
| Dogs | 12.5 | 96.9 | 87.5 | 87.5 | 87.5 |

入口：[完整报告第4节](LOCAL_GPU_STUDY.md#4-更贴近论文主张的检索消融)、[提示拆分协议](analysis/query_retrieval_local_v2/protocol.json)、[层/池化结果](analysis/query_retrieval_local_v2/analysis.json)、[原Raw诊断](analysis/query_retrieval_local_v2/original_raw_diagnostic.json)、[事后状态差分](analysis/query_retrieval_local_v2/query_delta_exploratory.json)。GPU runner 是 [run_query_retrieval_local_v1.py](scripts/run_query_retrieval_local_v1.py)，CPU分析是 [analyze_query_retrieval_local_v2.py](scripts/analyze_query_retrieval_local_v2.py)；沿用该 runner 会得到小样本检索实验，不能直接充当新增 B 的正式 full-test worker。

#### 1.4 固定 demo 标签干预：已有完整结果与样本扩展

对同一 query 固定四张 demo 图片、顺序和最终 ICL，标签分 Gold、demo 自身合法 zero 预测 own、确定性错误 wrong，另有同环境约束 zero。完整 DTD+CUB 计划30696条，实际30526条；**170条 own 预定缺失保留：DTD71、CUB99**。

| 数据集 | Gold−own 可用配对 | Gold−own，pp [95%CI] | Gold / wrong 全test准确率 | Gold 对原zero正确query的 harm |
|---|---:|---:|---:|---:|
| DTD | 1809 | +18.96 [16.92, 20.95] | 85.90% / 8.40% | 52/1255 |
| CUB | 5695 | +44.78 [43.34, 46.16] | 83.36% / 13.89% | 161/2165 |

另有五组各256个预定 query 的样本扩展：Gold−own 可用配对 DTD244、Aircraft246、CUB247、Pets256、Dogs240；差值分别 +22.54、+48.78、+49.39、+10.55、+24.17pp。这不是五组完整 test。支持结论是固定上下文时标签内容影响输出，不能独立证明 Behavior 检索的特有价值或 same-failure 因果。Gold 也有 harm，缺失不能补成错误。

证据：[完整arm表](analysis/paper_full_test_labels_complete_results_v1/arm_summary.csv)、[完整配对效应](analysis/paper_full_test_labels_complete_results_v1/paired_label_effects.csv)、[完整审计](analysis/paper_full_test_labels_complete_results_v1/audit.json)、[样本效应](analysis/paired_label_sampled_results_v1/paired_label_effects.csv)、[样本审计](analysis/paired_label_sampled_results_v1/audit.json)。这项工作已经完成，无须重跑来填文档。

#### 1.5 QwenVision 控制与 strict-match 反例

同目标 Qwen 视觉特征 + SupCon，完整 DTD1604/1880=**85.32%**、CUB4886/5794=**84.33%**；CUB 高于历史 Proj83.36%。这是 DTD+CUB 两组控制，保留历史环境差异，不支持 decoder 普遍更优。证据：[准确率](analysis/paper_qwenvision_two_complete_results_v1/full_official_accuracy.csv)、[配对表](analysis/paper_qwenvision_two_complete_results_v1/paired_accuracy_CI.csv)、[审计](analysis/paper_qwenvision_two_complete_results_v1/audit.json)。

完整五组的 rescue/harm 与 strict-match 回顾分析也已完成。strict-match 使用评估时的 query 真类别和 query/demo 原合法错误预测，不能放进在线检索。GPT-MM 在 DTD/Pets 存在覆盖更高、rescue较低的反例；CUB 中 Proj strict覆盖2838/3633，高于QV2633/3633，但QV rescue2890/3633，高于Proj2829/3633。固定同类demo数后，没有一致支持独立 same-failure 作用。保留非法/unknown 与历史自由zero、约束ICL之间的差别。入口：[P0审计](analysis/paper_repair_five_dataset_v1/audit.json)、[固定同类数分析](analysis/paper_qwenvision_match_control_v1/fixed_sameclass_count_associations.csv)、[QV匹配审计](analysis/paper_qwenvision_match_control_v1/audit.json)。

#### 1.6 TAM：已完成的本地新版空间诊断

五组各128图，共640图、1280次主比较，加完整词表/自然首token控制共1400份记录。主比较固定图片、pixels/grid、金标签首token，只改通用/短任务文本；固定金token是 teacher-forced probe。640对中572对原始CAM完全相同。扣除区域面积后的热图质量变化：Aircraft −0.518pp，CUB −0.091pp，Dogs −0.314pp，Pets −0.084pp；DTD 无同类独立区域标注。末层状态余弦却降至0.598–0.706。

这支持文本能改变回答前表示，**未证明一致前景归因改善**。raw CAM 与 ECI 校正分开；自然首token在0/40控制图上相同，比较自然热图会混入目标词变化。128图扩样发生在观察32图结果之后，bootstrap CI是描述性区间。入口：[本地TAM完整说明](LOCAL_GPU_STUDY.md)、[协议](analysis/query_tam_local_v3/protocol.json)、[分析](analysis/query_tam_local_v3/analysis.json)、[控制](analysis/query_tam_local_v1/controls_analysis.json)。新增主线优先补下面的检索/ICL证据。

### 2. 五 seed 主表：固定范围与交付

**目标：seeds 73/74/75/76/77 × 五数据集 × 六方法，共150个完整结果格。** 已有73共30格；74/75尚未完成；若73合格复用，新74–77共120格、558144条预测；含73全部697680条。每格必须是完整官方 test。中途部分预测、预分配数组、smoke和失败审计不计完成。

| 数据集 ID | 类别数 | bank | test / 每方法每seed预测数 |
|---|---:|---:|---:|
| `dtd` | 47 | 3760 | 1880 |
| `aircraft` | 100 | 6667 | 3333 |
| `cub` | 200 | 5994 | 5794 |
| `pets` | 37 | 3680 | 3669 |
| `dogs` | 120 | 12000 | 8580 |
| 合计 | — | 32101 | 23256 |

| 表格名 | 旧实现结果 key | seed 控制与注意事项 |
|---|---|---|
| RICES | `rices` | CLIP L2 cosine，完整bank，无检索头训练 |
| GPT-MM-style | `gpt_mm` | 原12-token生成状态代理，无query金标签；不是作者官方复现 |
| DeTriever（任务适配） | `detriever` | 初始化/训练采样变化，bank-only gold input+answer EOS proxy；保留原10000步方案 |
| Raw | `decision_zero` | 36层逐层cosine均值，无训练 |
| Proj | `projection_only` | bank train/dev划分、投影初始化、训练采样变化 |
| Learn | `decision_learn` | 同seed匹配Proj的bank train/dev，投影初始化和训练采样；加层权重 |

**共同配置**：上述模型revision；36×2560最后输入位置的回答前状态，排除embedding层；FP16 source存储、原FP32归一化/打分；原native processor默认与224消息尺寸提示，不把224当总视觉token预算，不移植InternVL的maxpatch配置；BF16目标模型；提取/最终生成batch1；完整bank top4降序选出再反转为demo顺序；同一合法类别trie greedy解码、`do_sample=False`。测试标签仅用于评估。

**Proj/Learn**：共享无bias `2560→256` 投影，正交初始化；bank每类10dev，训练排除dev；SupCon温度0.07，16类×4样本/批，AdamW lr3e-4、weight_decay0.01；1500步，0/100/…/1500评估bank-dev Top4同类比例，保留最早严格最佳checkpoint。最终完整bank含dev，不refit，不按test选层/宽度/seed/checkpoint。

**DeTriever**：原任务适配有自己的训练合同，不能为追求“所有方法同预算”而改成1500步或强加Proj/Learn的dev checkpoint规则。bank全体参与原训练，10000步最终checkpoint；batch64，positive40/negative100，memory refresh100，lr1e-4，temperature0.07，AdamW betas(0.9,0.98)、wd0.01；层1/6/11/16/21/26/31/36。query不能使用gold答案proxy监督。主表这是任务适配对照，训练预算不同需要披露。

**缓存与73复用**：原多seed合同只允许同(dataset/model/seed)内复用完整source，禁止跨seed状态/checkpoint/selections/预测混用。新计划沿用这个保守规则。73仅在source/模型/processor/协议/SHA/完整预测审核通过后登记历史引用，不能只复制准确率。保存73与新环境的batch/软件/硬件差异；存在实质不兼容时，分别报告“历史73”和“新环境74–77”，先解决兼容性再称统一五seed结果；不要静默重跑或替换原73。

Raw/RICES/GPT-MM没有检索头训练，确定性重放可能五seed完全相同，应如实列值并说明随机性来源；不引入生成采样来制造std。五seed主要检查训练方法稳定性，尤其 **Learn−Proj**。

### 3. 新增 ablation A：多层、投影与层权重（P0）

范围固定 **DTD+CUB，完整bank/full test，seeds73/74/75**。共7 arms×2组×3seed=42格、161154条预测；其中Raw/Proj/Learn三个参考arm只有在同seed、相同source/环境/协议身份匹配时才能复用主表；四个新增arm共92088条预测。未匹配的历史参考不能直接并成配对对照。

令 `u_l = L2(h_l)`，投影后 `z_l = L2(W u_l)`；统一逐层cosine求和，最终ICL完全相同。

| arm ID | 定义 | 训练 | 预定主要比较 |
|---|---|---|---|
| `raw`（参考） | 36层 `mean_l cos(u_l(q),u_l(b))` | 无 | 参考 |
| `proj`（参考） | 共享W，36层cosine等权平均 | 原1500步SupCon | 参考 |
| `learn`（参考） | 共享W，36层cosine的softmax权重和 | 原1500步SupCon | Learn−Proj |
| `last_raw` | 仅第36层 `cos(u_36(q),u_36(b))` | 无 | Raw−LastRaw |
| `last_proj` | 仅第36层 `cos(z_36(q),z_36(b))` | 1500步SupCon | Proj−LastProj |
| `weights_only` | 无W，36层cosine的softmax权重和 | 1500步SupCon，仅学36 logits | WeightsOnly−Raw；Learn−WeightsOnly |
| `random_proj` | 固定正交随机W，36层cosine等权平均 | 无更新；冻结对应seed的Proj初始W | Proj−RandomProj；RandomProj−Raw |

LastProj与多层Proj的W均为655360个参数，Learn额外36 logits。LastProj必须单独训练，不能切片已经训练好的多层Proj假装相同训练对照。RandomProj应保存Proj step0相同初始化矩阵SHA，禁止按test挑随机矩阵；checkpoint与打分数据类型、归一化顺序完全一致。

学习arm复用同seed的bank train/dev和预定训练采样序列；投影arm lr3e-4。WeightsOnly预先固定lr1e-2，沿用已有控制实现，披露学习率差别，不能宣称所有优化设置完全相同；无训练arm没有best-dev选择。验证只在bank-dev上进行，正式检索候选包含整个bank。

已有 [run_behavior_ablation.py](scripts/run_behavior_ablation.py) 中 `final_projected` 对应LastProj、`layer_weights_only` 对应WeightsOnly、`projection_only` 对应Proj；默认没有LastRaw/RandomProj，也没有Learn reference。旧文件限制seed73；`pooled_projected`只存在于legacy class，不在默认METHODS里，且带学习层权重，不能当“仅换pooling”的干净对照。**新A需要独立版本runner与协议，保留旧源码字节。**

### 4. 新增 ablation B：任务指令 × 类别词表（P0）

范围固定 **DTD+CUB，完整bank/full test，seeds73/74/75，Raw检索**。四arm共24格、92088条预测；Raw无训练，这三个seed属于独立tuple重放，不是三个独立训练随机性样本。主结论来自同query的prompt配对干预，不用重放std冒充训练稳定性。

定义两个显式字段 `instruction_on`、`vocabulary_on`；四种prompt从同一个构造函数组合，bank与query使用同一arm。固定image-first顺序、assistant generation prefix、输出格式共同后缀、图像字节与processor；有指令时使用原dataset分类句，无指令时省略该句；有词表时列原class_labels固定顺序，无词表时省略整个词表句。共同后缀固定为 `Return a short answer and no explanation. Answer:`，不会在无词表arm额外泄漏类别名称。这是新匹配2×2模板，不声称与历史generic/task文本逐字相同。

| arm ID | 任务分类句 | 合法类别词表句 |
|---|---|---|
| `neutral` | 无 | 无 |
| `instruction_only` | 有：原 `Classify the image into exactly one {noun}.` | 无 |
| `vocabulary_only` | 无 | 有：原 `The only valid labels are: ...` |
| `instruction_vocabulary` | 有 | 有 |

模板锁定前保存五类信息：完整message JSON与SHA、token IDs/长度、pixels/grid SHA、类别列表顺序、最后输入位置索引。此协议仅运行DTD+CUB，但构造器按dataset noun映射；不要字符串替换后遗留“valid label”等不一致后缀。文本长度不同是干预的一部分，报告token数，检查截断；固定图像预算，不因为长词表缩图。

只改变source提取；最终ICL仍调用原完整分类模板，demo用同样四张检索出的原图及gold标签、固定顺序、合法解码。无指令arm也不使用query真类别。每arm都重提完整bank/query状态，不能只换query而共用另一个arm的bank。

预定效应以最终ICL准确率为主：

- 无词表下指令作用：`instruction_only − neutral`。
- 有词表下指令作用：`instruction_vocabulary − vocabulary_only`。
- 无指令下词表作用：`vocabulary_only − neutral`。
- 有指令下词表作用：`instruction_vocabulary − instruction_only`。
- 交互项：`instruction_vocabulary − instruction_only − vocabulary_only + neutral`，同query联合bootstrap，不能把四个独立CI直接相减。

辅助报告Top1、Top4-any同类、Top4同类比例、AP、top4集合/顺序变化、task-only/generic-only正确query，注意这些同类指标只在离线评估使用gold。旧小bank四提示结果用于动机，不计入新B full-test完成数。

### 5. 可选 ablation C：提取位置 × 层聚合（P1）

范围先DTD+CUB，完整bank/full test，seed73；4 arms=8格、30696条预测。若扩展73/74/75须在看到结果前登记，且说明无训练重放的含义。统一无训练cosine检索与相同task输入，分别取：`pre_last`、`pre_all`、`generation_selected_last`、`generation_selected_all`。聚合规则为同位置第36层或36层逐层L2 cosine等权平均。

回答前用原最后输入位置；生成侧复用原GPT-MM-style的12-token greedy生成与**原capturer所选的生成hidden step / token位置**，保存全部36层以比较聚合。[extract_gpt_mm_cache](scripts/run_multidataset_main.py) 的索引为 `min(length, len(hidden_steps)-1)`，length是首EOS索引（无EOS则tail长度），最小1；再取该step的最后输入位置。HF generation的hidden状态未必包含最后输出token处理后的状态，不能随意改称“完整答案后的EOS”。记录实际generated IDs、EOS/提前停止、step index与input position；两个生成arm必须来自同一次生成同一位置。query不用gold答案。此C是新匹配对照，不替换历史GPT-MM列。

主要比较：同位置多层−末层；同聚合生成侧−回答前。可进一步固定均匀层权重比较pooling，但要预先定义投影/归一化顺序。C用于分离位置与聚合因素，不能写成整段CoT轨迹；[GPT-MM原文](https://www.jstage.jst.go.jp/article/jnlp/33/1/33_207/_pdf)已使用多模态问题与答案位置表示，[DeTriever](https://aclanthology.org/2025.coling-main.544/)已有多层decoder检索，避免first/universal superiority claim。

### 6. 执行入口与实施清单：先完成新版本，再正式跑

**现在可以直接执行第0节离线接手命令；新五seed/A/B/C的一键正式runner尚未实现。** 下表指出可复用函数与需要补齐的位置，不能对旧入口传 `--seed 76` 假装已支持新计划。已有旧guard依赖私有操作身份和云文件，`--help`或CPU测试通过都不等于新GPU运行链可用。

| 工作 | 已有实现 | 合作者需要补齐的独立版本 |
|---|---|---|
| 五seed orchestration | [run_behavior_qwen_three_seed.py](scripts/run_behavior_qwen_three_seed.py)、[policy](scripts/behavior_qwen_three_seed_policy.py)、[Proj recipe](scripts/run_behavior_qwen_three_seed_proj.py) | 旧policy只接受74/75，Proj recipe只覆盖旧范围，verify硬编码60格/279072预测。新版本注册73–77、120新增格/558144预测、150总格/697680预测，历史73独立引用 |
| frozen plan preparation | [prepare_behavior_qwen_three_seed.py](scripts/prepare_behavior_qwen_three_seed.py) | 旧prepare依赖私有.tools并拒绝覆盖，不能在clone直接重调。新prepare应接受部署root、数据root、cache root和历史artifact位置，生成新SHA绑定 |
| Raw/Learn | [run_dtd_decision_states.py](scripts/run_dtd_decision_states.py)、[StateRetriever依赖](scripts/run_dtd_learned_velocity.py) | 保留原kernel/训练行为，在新运行目录绑定seed与source；勿按旧velocity文件头改科学定义 |
| Proj / A | [run_behavior_ablation.py](scripts/run_behavior_ablation.py) | 新A注册7个arm和73/74/75，新增LastRaw/RandomProj；保存共享初始化、实际训练split/采样身份；补完整ICL与CPU审计 |
| B | [四提示runner](scripts/run_query_retrieval_local_v1.py)、[prompt政策](scripts/behavior_task_prompt_policy.py)、[完整prompt实验](scripts/run_behavior_task_prompt.py) | 新2×2构造器、full-bank提取、全test最终ICL、四arm同query审计；不沿用2张/类和32query面板 |
| C | [原生成状态capturer](scripts/run_multidataset_main.py) | 新增36层同生成step保存及位置receipt，保持原capturer语义；新独立arm目录 |

建议的新实验ID是 `qwen_five_seed_v1`、`qwen_components_v1`、`qwen_prompt_factorial_v1`、`qwen_position_layers_v1`，均为本节定义的新名字，当前不是已经存在的worker或科学锁。新protocol保存：commit与源码闭包SHA、模型inventory、完整有序manifest与逐图SHA、prompt/decoder/vision参数、seed/arm矩阵、输出目录、统计方案、73复用/环境差异、运行保护的独立身份。锁必须在正式结果产生前冻结，改变协议就新版本；不编辑 [原锁](artifacts/frozen/analysis/paper_qwen_three_seed_plan_v1/plan_lock.json.gz)或历史receipt来绕过限制。

每个新tuple完整执行以下顺序，阶段完成标记与日志分开；并发卡须有独立可写root：

1. **CPU preflight**：核有序bank/query及逐图SHA、模型/processor inventory、source与训练split无query泄漏；验证完整阶段命令与新操作身份。
2. **最小实际兼容检查**：指定GPU上实际加载固定模型，确认36×2560、BF16、native pixels/grid、真实4-demo合法解码。query标签替换为sentinel时输入不变；确认只有一个科学worker。smoke不计正式结果。
3. **五seed阶段**：`features → baseline_retrieve → baseline_generate → decision_retrieve → decision_generate → proj_retrieve → proj_generate → proxy_capture → detriever_retrieve → detriever_generate`。这是原phase语义供新runner复用；`decision_*`负责Raw/Learn，baseline负责RICES/GPT-MM，DeTriever独立gold proxy和10000步。
4. **A/B/C阶段**：按各自矩阵提取完整source → 仅适用arm训练 → 完整bank top4 → 每arm完整test ICL → 保存进度/原始输出。参数、prompt模板和解码检查失败先诊断，不能调test选一个能过的版本。
5. **断点恢复**：只接受完整JSONL行和权威提交游标；复核protocol/模型/代码/source/上下文/worker身份，另立一次操作receipt，不重复消费旧launcher。预分配数组shape不等于source完成；系统崩溃保留进度并报告，不自动循环恢复。
6. **取回与停机**：若部署云端，遵守[AGENTS.md](AGENTS.md)的最多两张独立单GPU、每卡一个科学worker、真实120秒archive/member/extractedSHA备份和精确停机要求。FULL取回SHA核验后立即精确停机并核对provider，再独立CPU审计。该README不授权重启历史云端队列。

**首个实现交付**：新protocol与完整矩阵、可运行CPU preflight、阶段命令预览、可配置root的新runner、恢复身份检查、独立CPU完整性审计；列明实际缺件。先用一个预定DTD tuple检查完整链路，兼容通过再按已固定矩阵运行，不能通过pilot准确率选版本。

### 7. 统一结果格式、统计与完成标准

建议每tuple路径 `results/<experiment>/<dataset>/qwen3vl4b/seed_<seed>/<arm>/`，不同实验版本、seed、prompt arm不能共用写入目录。命名可适配旧component结构，但映射必须写进新protocol。至少交付：

- `run_identity.json` / `protocol.json`：源码/model/prompt/vision/decoder/seed/arm/source身份及SHA；native兼容receipt、环境版本、各阶段日志与进度。
- `manifest.json`：有序完整bank/query与图片SHA；学习arm另存 `development_split.json`、初始化与训练采样身份、训练history、best_step规则和checkpoint SHA；DeTriever另存其proxy进度/SHA与10000步metadata。
- `selections.json`：每query四个不同bank demo ID及gold标签、检索score与实际ICL顺序；`predictions.jsonl`保存dataset/seed/arm/query ID、target、prediction、raw_output、correct、demo IDs/labels。非法/unknown保留并计错误，不删query。
- `full_audit.json`：完整query覆盖、无重复/缺失/截断、target正确、demo来自full bank、gold标签/顺序与selections一致、source/checkpoint绑定一致、计数从原始预测重算；汇总/accuracy不能独立证明完成。
- 原始source/checkpoint/预测、逐成员SHA和FULL归档保存于约定外部存储；Git只上传协议、代码、审计、精简CSV和路径索引，遵守数据许可证，不上传凭据/模型权重。

主表CSV列建议 `dataset, method, seed, correct, total, accuracy_percent, audit_path, predictions_sha256, environment_id, historical_reference`；汇总列 `dataset, method, n_seeds, mean_percent, sample_std_percent`。每seed的五组宏均值先按五数据集等权计算，再对五个seed计算mean与sample std（ddof=1）；不能混成按query数加权均值。

配对差值CSV保存dataset/seed/first/second/queries/difference_pp。重点Learn−Proj，并报告Behavior三方法对GPT-MM/DeTriever的同seed差值；A/B/C按预定比较。每seed可补10000次同query配对bootstrap、固定统计seed的描述性95%CI，无多重校正时明确标注；query bootstrap区间与跨seed std分别报告，不把同一test在五seed下当116280个独立query。

**验收清单**：五seed150格（可登记73的30格历史引用）完整审计；A42格、B24格分别独立审计；C可选8格、未做就标未做。只在参考arm身份匹配时复用，实际新增预测数量需单列。缺seed/方法、缺外部原始文件、运行失败均明确列出，不能填入mean±std主表。保留负结果、harm、unknown、环境差异及样本范围。A/B只有DTD+CUB，不能推广为五数据集消融结论。

**优先级**：先五seed并核Learn−Proj，再A和B；C与效率表随后。效率分source提取、检索头训练、检索和最终ICL耗时，记录tokens、峰值显存、缓存字节和参数量。维度扫描、更多热图、shot扫描后置；现有完整提示/标签/QV实验无需为填文档重新跑。

---

> **2026-10-10 发布更新**：Qwen五组seed73主表保留；新增本地TAM/检索研究、InternVL小样本验证，以及正在进行的InternVL全量六方法实验。当前完整结果及进度见 [CURRENT_RESULTS.md 第8节](CURRENT_RESULTS.md#8-internvl35-4b五数据集结果与四组重跑进度)，代码见 [CURRENT_CODE.md](CURRENT_CODE.md)，发布范围与离线复核见 [本次发布说明](handoff/LOCAL_PUBLICATION_20261010.md)。本地InternVL已获用户恢复授权；云端仍暂停，LFM取消。仓库记录是带时间的快照，不能代替本机实时进程状态。

> 2026-10-06 本机实验更新：Qwen 五数据集 TAM 与检索消融见 [LOCAL_GPU_STUDY.md](LOCAL_GPU_STUDY.md)；新增 dense InternVL3.5-4B 五数据集检索、四示例 ICL 与 GPT-MM-style 代理对照见 [INTERNVL35_LOCAL_STUDY.md](INTERNVL35_LOCAL_STUDY.md)。下文保留此前 GitHub 接手快照，关于 TAM/InternVL 尚未运行的语句只描述该旧快照；云端自动任务仍暂停。

> GitHub接手版：仓库内链接可移植；原科学source/audit字节保留。完整本地归档、图片、权重及云运行状态不在此仓库，参见 [artifact availability](handoff/ARTIFACTS.md)。


**当前主线：用目标 VLM 在任务文本与图片上的逐层回答前状态，选择对该 VLM 有用的 demonstration。**

2026-10-06历史整合时实验曾全部暂停。随后用户授权本地研究与InternVL恢复；当前授权和操作记录见 [HANDOFF.md](HANDOFF.md)。公开clone不能继承本机恢复授权或直接重调已消费启动器。

**新 coding agent 从 [HANDOFF.md](HANDOFF.md) 开始，并遵守 [AGENTS.md](AGENTS.md)。** 首次检查只需 Python 3.12 标准库，无需图片、模型权重、GPU 或云凭据：

```bash
python scripts/verify_handoff.py
```

该检查核验原2026-10-06 handoff的代码/审计SHA、压缩原始表、结果计数、依赖与文档链接。本次新增代码/证据另用 `python scripts/verify_local_publication_v1.py` 核验。完整实验复现还需要 [外部 artifacts](handoff/ARTIFACTS.md)。本地新版TAM已经有实测报告；旧云端64-run协议仍未完成，两者独立记录。

| 你想看什么 | 当前入口 |
|---|---|
| 想法、方法、已得到的证据与缺口 | [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) |
| 当前代码分别实现什么 | [CURRENT_CODE.md](CURRENT_CODE.md) |
| 完整结果、配对 CI、分母和原始证据 | [CURRENT_RESULTS.md](CURRENT_RESULTS.md) |
| 下一步先做什么、怎样判断有结果 | [NEXT_STEPS.md](NEXT_STEPS.md) |
| 论文可以怎样讲 | [PAPER_NARRATIVE.md](PAPER_NARRATIVE.md) |
| 停机状态与剩余工程问题 | [RUN_STATUS.md](RUN_STATUS.md) |

当前已完成：Qwen五数据集seed73主表、纠错/匹配分析、DTD+CUB任务提示对照、固定demo标签干预、QwenVision控制，以及本地新版TAM/小bank检索和InternVL五组32-query面板。InternVL全量实验已审计DTD六格、Aircraft六格和CUB四格；DeTriever/RICES及Dogs/Pets仍未完成。新增Qwen seed74/75也未完成。详细分母、设置差异和负面结果见各报告。

早期探索与启动排障不再作为当前方法入口。必要依赖保留在当前树，退役的早期公开文件保留于 Git 历史；本地原文件、锁、失败盘及整理前文档单独保留，见 [历史说明](handoff/ARTIFACTS.md#local-history)。机器可读的代码依赖、证据 SHA 和精简 CSV 在 [整合索引](analysis/current_project_consolidation_v1/evidence_index.json)。
