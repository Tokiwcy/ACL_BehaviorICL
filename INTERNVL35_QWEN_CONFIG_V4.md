# InternVL3.5-4B：按 Qwen 正式设置从 Aircraft 重跑

2026-10-07 用户明确取消 LFM，要求从 Aircraft 开始重跑 InternVL。旧单 patch 流水线保留冻结状态；云端 GPU 保持暂停。

## 科学配置与差异

原v2在最极端五图（16,640视觉token）smoke的Windows SDPA数学路径上CUDA OOM，尚未开始正式source或训练。v3的单图cuDNN状态计算与数学路径比较通过，但GPT-MM单token解码受本机cuDNN不支持sequence length=1限制而失败。v4保持图片预算和科学参数，固定cuDNN用于prefill，数学SDPA用于单token解码；smoke预先要求单图36层回答前状态与数学路径的逐层cosine≥0.995，并记录误差。旧v2/v3协议、代码和失败smoke保留。两版均无正式source/训练/全量预测。

沿用 Qwen seed73 主实验的完整 bank/test、任务与类别文本、四示例消息、合法标签 trie 贪心解码、检索相似度较低示例在前的顺序，以及 Raw/Proj/Learn/GPT-MM-style/DeTriever-style/RICES 六种方法。所有 InternVL 图文状态、生成答案状态、bank 答案代理及可训练检索参数重新生成，独立保存在 `results/internvl35_qwen_config_v4`，不复用旧单 patch source 或 checkpoint。

- 模型固定为 `OpenGVLab/InternVL3_5-4B-HF`，revision `6bd4487402110ef9889ba50eb7aefeb302526fed`，BF16。本地权重文件与科学代码 SHA 绑定到 [协议](analysis/internvl35_qwen_config_v4/protocol.json)。
- Qwen 主实验使用原生动态处理（`vision_pixels=None`）；此轮 InternVL 对应使用其原生动态分块，`crop_to_patches=True`、`max_patches=12`、448×448、thumbnail、每块256视觉 token。thumbnail 在分块图上额外增加一块，实际每图最多13块。两个架构不会因此获得完全相同的输入 token 数；4,173不是统一上限。每张图片的预算和每次实际输入的 token、patch tensor、图片路径顺序都保存并审计。
- 回答前最后输入位置的36×2560状态（排除 embedding）。Raw逐层 L2、全36层等权 cosine；Proj共享无偏256维投影、等权层；Learn相同投影加softmax层权重。
- Proj/Learn：仅bank标签，每类10 dev，1500步，16类×4图的均衡batch，SupCon温度0.07，AdamW lr3e-4/weight_decay0.01，每100步dev评估，最早最佳dev checkpoint；最终候选为完整bank。
- GPT-MM-style：同任务零示例贪心答案状态，最多12新token。DeTriever-style：bank gold-answer EOS代理、原每五层网格，10000步、batch64、40正/100负、温度0.07、lr1e-4、每100步memory refresh。这些是原项目本地任务适配，不能称作者官方代码复现。
- RICES：相同CLIP ViT-B/32检索，最终ICL由新图像设置下的InternVL重新运行。
- 本地16GB卡使用推理batch1。历史Qwen环境和batch并非完全相同；本次不能声称严格相同的硬件执行结果。未完成的检索训练在固定seed73从头训练，避免把不一致的随机数恢复路径当原始训练。

## 覆盖与顺序

| 数据集 | 完整 test query | 方法数 |
|---|---:|---:|
| Aircraft | 3,333 | 6 |
| CUB | 5,794 | 6 |
| Dogs | 8,580 | 6 |
| Pets | 3,669 | 6 |

总计21,376个query、128,256条预测、24个方法格。DTD按照用户“从Aircraft开始”的范围保留旧结果；不能把旧单patch DTD拼进新动态处理结果并称统一五组主表。

## 执行和审计

入口为 [run_internvl35_qwen_pipeline_v4.py](scripts/run_internvl35_qwen_pipeline_v4.py)：CPU prepare→GPU smoke→单GPU顺序pipeline。CPU prepare核验图片字节及有序manifest，固定本地模型文件SHA，并计算全bank/test图像分块预算。Smoke按预算确定四张最大patch bank图和最大patch query，覆盖四组五图输入；仅以兼容性与显存判断可行性，不依据test准确率选择预算。同时检查36层状态、GPT-MM生成状态、bank gold代理和合法标签解码。

监督器持有进程锁并检查GPU占用，状态文件原子落盘，失败立即停下，不循环重启。`pipeline_logs/STOP_REQUESTED`可在阶段边界暂停；恢复须协议、缓存和GPU身份一致。系统崩溃后的原记录保留，半小时汇报不得自动反复恢复。

[独立CPU审计](scripts/audit_internvl35_qwen_config_v4.py)核对完整query、四示例身份和顺序、预测合法性、summary与manifest/selection/protocol SHA，以及实际五图视觉token和预算一致性；输出准确率、类别宏平均、Wilson区间、检索同类率、GPT-MM/DeTriever配对差值及实际token分布。24格全完成后才写完整审计。

当前进度以 `results/internvl35_qwen_config_v4/pipeline_logs/status.json`、阶段日志、缓存进度和审计文件为准。CPU准备、显存smoke、预分配缓存均不是完整科学结果。

## LFM 中断证据

Windows最近三次崩溃都记录了0x20001。最后一次LFM单图source成功得到30×2048有限状态，随后五图ICL阶段被系统重启中断。事件日志证明系统崩溃类型，尚未通过dump确定具体故障驱动；不能把时间关联直接当模型导致崩溃的结论。原smoke、事件记录和取消收据在 `analysis/lfm25_vl3b_full_v1` 保留。
