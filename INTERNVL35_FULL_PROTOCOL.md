# InternVL3.5-4B 全量五数据集 4-shot 对照

本文件记录本机正在执行的独立全量实验协议。先前每组 32 张 test 的探索结果见 [小样本报告](INTERNVL35_LOCAL_STUDY.md)，不能代替本实验的全量结论。历史 Qwen 结果和 TAM 结果分别见 [当前结果](CURRENT_RESULTS.md)与 [本机 TAM 报告](LOCAL_GPU_STUDY.md)。云端 GPU 保持暂停。

## 固定范围

目标模型为官方 `OpenGVLab/InternVL3_5-4B-HF`，revision `6bd4487402110ef9889ba50eb7aefeb302526fed`，BF16 dense 权重。本机 RTX 5080 Laptop 16 GB。检索源与最终 4-shot 生成均固定每图一个 448×448 视觉块；不使用量化。seed 73，所有方法共用同一官方 bank、test 顺序、分类提示、四张原图与金标签、合法标签解码。

| 数据集 | 官方 bank | 官方 test | 类别 |
|---|---:|---:|---:|
| DTD | 3,760 | 1,880 | 47 |
| Aircraft | 6,667 | 3,333 | 100 |
| CUB | 5,994 | 5,794 | 200 |
| Dogs | 12,000 | 8,580 | 120 |
| Pets | 3,680 | 3,669 | 37 |
| 合计 | 32,101 | 23,256 | — |

六种方法产生 30 组、共 139,536 条官方 test 预测。每张 bank/test 图片、顺序和标签有固定 manifest 与图像 SHA；五组图片合计 55,357 张。在线检索不使用 test 金标签。

| 方法 | 检索表示与训练 |
|---|---|
| Raw | 同一目标 VLM 的 36 层最后输入位置回答前状态；每层 L2 归一化后等权平均余弦 |
| Proj | 共享 256 维投影、均匀层权重；只用 bank 标签做 SupCon，10 dev/class，1,500 步 |
| Learn | 同一投影加学习的 softmax 层权重；同样的 bank-only 训练与 checkpoint 选择 |
| GPT-MM-style | 相同 VLM 贪心生成至多 12 token 的答案状态，末层余弦；这是本地受控算法适配，不是作者代码复现 |
| DeTriever-style | 基于目标 VLM 逐层回答前状态、bank 金答案 EOS 代理与 10,000 步训练；这是论文公式在图像分类上的适配，不是作者代码复现 |
| RICES | CLIP ViT-B/32 图片特征余弦 |

GPT-MM 原文已经把图片与问题联合输入模型，因此本实验只检验**生成前逐层状态与生成答案状态的差别**。DeTriever 已在文本 NL2SQL 使用多层 decoder 表示，因此不主张首次使用多层隐藏状态。关于已有 Qwen 和 TAM 证据的边界见 [novelty 讨论](LOCAL_GPU_STUDY.md#5-novelty-与-gpt-mm-的准确差异)。

## 执行与审计

单 GPU 顺序流水线为 [run_internvl35_full_pipeline_v1.py](scripts/run_internvl35_full_pipeline_v1.py)，数据与特征由 [source runner](scripts/run_internvl35_full_sources_v1.py) 生成，训练/检索由 [method runner](scripts/run_internvl35_full_methods_v1.py) 执行，完整预测由 [ICL runner](scripts/run_internvl35_full_icl_v1.py) 写入。冻结协议在 [protocol.json](analysis/internvl35_full_v1/protocol.json)。实时阶段、命令、返回码和日志在本机忽略 Git 的 `results/internvl35_full_v1/pipeline_logs/`；中断后可恢复，不应同时启动第二个 GPU 流水线。

[独立 CPU 审计](scripts/audit_internvl35_full_v1.py) 核对 30 组的覆盖、query 顺序、四张示例、标签、SHA、非法输出、准确率、类别准确率、检索同类率和与 GPT-MM/DeTriever 的 query 配对差值及 10,000 次 bootstrap 区间。未完成时只写 `analysis/internvl35_full_v1/progress_audit.json`；全部完成且校验通过才写 `full_audit.json`。自动 CPU 收尾见 [finish_internvl35_full_audit_v1.py](scripts/finish_internvl35_full_audit_v1.py)。区间是单 seed 73 的描述性结果，没有多重比较校正。
