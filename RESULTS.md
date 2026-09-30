# BehaviorICL 实验结果

本文只记录已完成、已校验的结果；运行进度见本地 `RUN_STATUS.md`。原始逐题输出保留在本地 `results/`，不随 Git 上传；下文指向这些产物的相对路径是本地归档位置，在 GitHub 上不可直接打开。

## 评测协议

所有下表结果使用 Qwen3-VL-4B-Instruct、seed 73、各数据集完整官方划分、4-shot、贪心生成及合法类别约束解码。同一数据集各方法使用相同的 test query 和类别集合。“—”表示尚无该方法的完整结果，不代表准确率为零。

RICES 按 CLIP 相似度取最近的四个训练样本；GPT-MM 按生成答案的最终状态检索。DeTriever 是按 [原论文](https://aclanthology.org/2025.coling-main.544/) 的分层 MLP、加权融合与训练样本“输入＋真实答案”表示相似度代理目标所做的任务适配，**不是运行作者官方代码**；每个数据集的检索器从随机初始化训练 10,000 step，test 标签不用于训练。Behavior 系列把“模型看到给定图像和分类任务输入后如何表征它”具体化为一次零样本 prefill 在回答位置的 36 层状态；这并非测量逐 token 的动态决策过程。Behavior-Zero 逐层计算余弦相似度并取平均；Behavior-Learn 使用共享 2560→256 投影及学习的层权重，只用训练 bank 的同类/异类对比信号，每个数据集重新初始化训练 1,500 step。检索阶段不读取 test 标签。现有运行文件和校验器的内部键仍为 `decision_zero`、`decision_learn`，分别对应 Behavior-Zero、Behavior-Learn；改名不改变任何预测或结果。

## 主结果

| 数据集（bank / test / 类别） | RICES | GPT-MM | DeTriever | Behavior-Zero | Behavior-Learn |
|---|---:|---:|---:|---:|---:|
| DTD（3,760 / 1,880 / 47） | 1,480/1,880 · 78.72% | 1,534/1,880 · 81.60% | 1,471/1,880 · 78.24% | 1,580/1,880 · 84.04% | **1,626/1,880 · 86.49%** |
| FGVC Aircraft variant（6,667 / 3,333 / 100） | 1,989/3,333 · 59.68% | 2,312/3,333 · 69.37% | 2,279/3,333 · 68.38% | 2,461/3,333 · 73.84% | **2,609/3,333 · 78.28%** |
| Oxford-IIIT Pet（3,680 / 3,669 / 37） | 3,358/3,669 · 91.52% | 3,407/3,669 · 92.86% | 3,390/3,669 · 92.40% | 3,484/3,669 · 94.96% | **3,501/3,669 · 95.42%** |
| CUB-200-2011（5,994 / 5,794 / 200） | 4,126/5,794 · 71.21% | 4,209/5,794 · 72.64% | 3,987/5,794 · 68.81% | 4,722/5,794 · 81.50% | **4,857/5,794 · 83.83%** |
| Stanford Dogs（12,000 / 8,580 / 120） | 6,377/8,580 · 74.32% | 6,751/8,580 · 78.68% | 6,665/8,580 · 77.68% | — | — |

目前 RICES、GPT-MM、DeTriever 在五个数据集上均有完整结果；Behavior 系列在 DTD、Aircraft、Pets、CUB 完成，Dogs 尚未完成。不能用这四个数据集的 Behavior 数字宣称五数据集一致提升；所有结果也只有一个模型、一个 seed。

## Behavior 系列内部比较

两种 Behavior 方法在每个数据集上分别完成全部 query 预测，已校验合法标签、训练 bank 示例、无重复 query、缓存及本 tuple 的训练文件。下表差值为同一批 query 上 Behavior-Learn 减 Behavior-Zero；95% CI 由 20,000 次配对 bootstrap 得到，p 值为双侧 exact McNemar。

| 数据集 | Learn − Zero | 95% CI（百分点） | McNemar p |
|---|---:|---:|---:|
| DTD | +2.45 pp | [+1.28, +3.62] | 6.74×10⁻⁵ |
| Aircraft | +4.44 pp | [+3.18, +5.70] | 4.74×10⁻¹² |
| Pets | +0.46 pp | [−0.03, +0.95] | 0.0784 |
| CUB | +2.33 pp | [+1.52, +3.16] | 3.85×10⁻⁸ |

Behavior-Learn 与 DeTriever 的准确率差在 DTD 为 **+8.25 pp**、Aircraft 为 **+9.90 pp**、Pets 为 **+3.03 pp**、CUB 为 **+15.02 pp**。Pets 上两者同 query 配对的 20,000 次 bootstrap 95% CI 为 **[+2.29,+3.76] pp**，双侧 exact McNemar p=2.26×10⁻¹⁶；CUB 为 **[+13.84,+16.17] pp**、p=3.30×10⁻¹³⁸。这里未把不同训练目标、参数规模和训练步数的影响解释为逐层匹配机制的独立因果效应；配对区间也不涵盖其他 seed 或模型。

Behavior 的原始预测与训练 checkpoint 分别保存在 [DTD 独立备份](results/cloud_trial/jiirguh96xew6j-decision/dtd/seed_73)、[Aircraft 独立备份](results/cloud_trial/nwbxy265c6vqvw-aircraft/decision/aircraft/qwen3vl4b/seed_73)、[Pets 独立备份](results/cloud_trial/pbzipnh9i4d180-decision/pets/seed_73)和 [CUB 独立备份](results/cloud_trial/pbzipnh9i4d180-decision/cub/seed_73)。Pets 和 CUB 均已通过云端含源缓存的完整校验及本地 `--skip-source-cache` 校验，两法各覆盖全部 query、合法标签、bank 示例、无重复键、训练 checkpoint/metadata 均通过；与其他三基线的 query ID、标签、划分及顺序一致。DeTriever 的五个独立 tuple 已经完成合法标签、bank 选择、完整预测与本 tuple checkpoint 校验；其 [DTD 输出](results/cloud_trial/jiirguh96xew6j-detriever/dtd/seed_73)和 [Aircraft 输出](results/cloud_trial/jiirguh96xew6j-detriever/aircraft/seed_73)与 Behavior 使用相同 query 身份。Aircraft 的 RICES 预测曾跨本机、MIG 和 RTX 4090 续跑，硬件差异应在正式报告中披露。

## Behavior 补跑时间成本

时间按云端串行日志的 UTC 阶段切换记录计算；“Behavior 运行”包含检索选择、Learn 训练和两方法测试预测，日志没有这些子阶段各自的精确起止时间，不拆分估算。总 Pod 计费时长及费用待停止并确认 `EXITED` 后填写；数据准备、传输和空闲时间计入 Pod 费用，但不计为方法运行时间。存储费用另计，当前无账单实数。

| 数据集 | 数据准备/核对 | 冻结状态抽取 | Behavior 运行 | 完整校验 | 数据集阶段合计 |
|---|---:|---:|---:|---:|---:|
| Pets | 1分30秒 | 12分50秒 | 32分46秒 | 1秒内 | 47分06秒 |
| CUB | 59秒 | 44分24秒 | 1小时19分28秒 | 约1秒 | 2小时04分52秒 |
