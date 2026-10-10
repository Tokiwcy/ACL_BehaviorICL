# LFM2.5-VL-3B：第二模型全量验证计划

状态（2026-10-07）：**用户已取消 LFM 运行并切回 InternVL。** LFM 单图回答前状态 smoke 成功（30×2048）；五图 4-shot 阶段发生 Windows 0x20001 系统崩溃，未完成多图 smoke，无正式训练或全量结果。保留 `analysis/lfm25_vl3b_full_v1/cancellation_receipt.json`、原 smoke 与 Windows 事件证据，未经用户再次授权不恢复。后续旧计划仅作历史记录。云端 GPU 保持暂停。

## 模型与图像预算

- 独立模型家族：`LiquidAI/LFM2.5-VL-3B`，固定 Hugging Face revision `35a118d938ce6d123ac2d371649f24a8efb69058`。官方 BF16 权重约 3.12B 参数；语言模型 30 层、隐藏维度 2048；视觉编码器为 SigLIP2 NaFlex。
- 本地环境：Transformers 5.13.0、PyTorch 2.8.0+cu128、RTX 5080 Laptop 16 GB。只使用官方 BF16 权重，不量化。正式运行前必须做单图 source、多图 4-shot、回答前 30 层状态及合法标签解码的最小 GPU smoke，并记录峰值显存。
- 候选主设置在 source 与 ICL 两侧统一设置官方处理器的 `max_tiles=4`，其余图像处理保持固定 revision 的默认值。处理器逐条记录实际视觉 token 数、tile 数、输入总 token 数和图像身份。**GPU 正式运行的视觉预算尚未锁定**：`max_tiles` 只是上限，不能保证小图达到该上限；须先完成多数据集 CPU 预算审计与显存 smoke，再版本化锁定。
- 已完成的 CPU 核验：在同一条 Aircraft RICES 四示例加 query 上，LFM `max_tiles=3` 为 3,769 视觉 token，`max_tiles=4` 为 6,329 视觉 token；原 Qwen3-VL 处理器在此条输入为 4,173。另以五组首条 query 与各组前四张 bank 图作探索性处理器抽查：LFM 4-tile 分别为 DTD 925、Aircraft 6,329、CUB 824、Dogs 851、Pets 818；旧 Qwen 处理器对应为 1,023、3,732、824、851、818。这些不是正式检索上下文，不能代表全 test。**4,173 只是某一条 Aircraft 输入的观测值，不是所有 query 的恒定预算。**
- LFM 的分块处理器不能直接把所有五图输入精确设为 4,173。不得为“凑 token”无说明地上采样小图，因为上采样不会增加原图信息。正式跨模型比较须报告各数据集实际视觉 token 分布，并明确预算差异。可预设 `max_tiles=3` 与 4 的固定上下文预算消融，不得用 test 准确率选主预算。如果选定预算在 16 GB BF16 下无法完成 smoke，先诊断并版本化修改协议，不静默降预算。

## 科学范围

- 五组官方完整 test：DTD 1,880、Aircraft 3,333、CUB 5,794、Dogs 8,580、Pets 3,669，总计 23,256 query。复用既有固定 bank/test manifest、类别词表、图片 SHA、seed 73、四示例提示及合法标签约束解码。
- 六方法均在 LFM 上完成完整 test，共 30 格和 139,536 条预测：RICES、GPT-MM-style、DeTriever-style、BehaviorICL-Raw、Proj、Learn。GPT-MM/DeTriever 是同任务本地算法适配，不称作者官方复现。
- Raw 使用 LFM 在完整任务文本与图片下生成答案前、最后输入位置的 30 层状态，每层 L2 归一化后等权余弦。Proj 为共享无偏 256 维投影与均匀层权重；Learn 为同一投影加可学习 softmax 层权重。Proj/Learn 仅用 bank 标签训练，10 dev/class、1,500 步、最早最佳 dev checkpoint，完整 bank 检索 top4；test 标签只用于最终审计。
- GPT-MM-style 在同一 LFM 上取贪心答案状态；DeTriever-style 重新按 LFM 状态及既定 bank 答案代理训练；RICES 使用相同 CLIP ViT-B/32 视觉检索规则，但最终四示例 ICL 必须由 LFM 重跑。不得复用 Qwen 或 InternVL 的 source/训练 checkpoint。
- 各方法共用同一 LFM 图像预算、提示、合法标签解码、query 顺序。独立 CPU 审计核对 30 格覆盖、每条四示例身份与顺序、图片 SHA、输出合法性、准确率、检索同类率、GPT-MM/DeTriever 的配对差值及实际视觉 token 分布。未完成格不记为完成结果。

## 进入全量运行的条件

1. InternVL 当前流水线结束，完成独立审计，确认 GPU 无其他科学 worker。
2. LFM 固定 revision 的文件 SHA 与本地环境记录齐备；最小 smoke 证明 30 层状态、五图输入、合法标签解码和 `max_tiles=4` BF16 显存可行。
3. 新 runner 使用独立输出目录与缓存身份，先 CPU 核对 manifest/protocol，再运行一次单 GPU 顺序流水线；从真实进度恢复，绝不重复启动并发 worker。

官方依据：[LFM2.5-VL-3B 模型卡](https://huggingface.co/LiquidAI/LFM2.5-VL-3B)、[Liquid AI 2026 发布说明](https://huggingface.co/blog/LiquidAI/lfm2-5-vl-3b)。
