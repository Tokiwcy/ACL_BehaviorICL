# 论文内容与代码对应

依据：`paper/main.tex` 与 `paper/supplement.tex`，2026-10-21 干净源包。
两个文件以及全部表格输入按原字节保留，正文不混入尚未执行的属性组删除实验。

| 论文内容 | 核心实现 / 资源 | 已有结果 |
|---|---|---|
| 训练图像因素发现、多视角请求、结构检查、重复处理 | `discovery.py`, `atoms.py`, `schema.py`; `resources/factor_banks/` | `factor_bank_inventory.json`，15 份主表因素库 |
| 原始 SigLIP 2 状态 softmax | `encoders.py`, `evidence.py`, `raw_evidence.py` | 原始 evidence 身份见主实验审计；不做 train-mean subtraction |
| 训练类别原型、JSD、跨类别标准化 | `selection.py` | `algorithm_cases.json` 实际成功/失败计算轨迹 |
| 独立五折与全验证集 alpha 校准 | `folds.py`, `selection.py`, `pipeline.py` | `all_model_plan_effects_20261004.json` 在 evidence archive 中；主校准记录保留 |
| 先选类，再完整类内训练池检索 | `pipeline.py`, `selection.py`, `baselines.within_class` | 主表与冻结计划；没有 RICES shortlist reranking |
| Random、CLIP RICES | `baselines.random_demonstrations`, `rices_scores`, `topk` | 180 主结果的一部分 |
| 分类适配 MUIER（SigLIP 图像+训练类名） | `baselines.muier_scores` | 不把它误当成与 RICES 只差一项文本的同编码器对照 |
| 两通道 CIRCLES 与 +DecisionICL | `circles.py`, `clip.py`, `baselines.circles_features`, `circles_demonstrations`, `prompts.official_content` | 仅 2-shot，保留两通道选中同一图像的情况 |
| 三个目标模型、精确标签与非法输出 | `models.py`, `prompts.py`, `resources/experiments.json` | Qwen3 BF16 / Qwen2.5 NF4 / Gemma4 12B NF4 |
| 同类无因素选例 | `baselines.base_class_gate` | DTD/Aircraft/CUB 三项，`factor_vs_class_gate.*` |
| 因素-类别原型循环错位 | `baselines.rotated_factor_features` | 三项，旋转方向与冻结代码一致 |
| CLIP 类质心替代因素 | `baselines.centroid_factor` + 独立 `class_features`/校准 | 三项混合结果，不声称因素普遍优于质心 |
| 匹配发现预算的 Decision/Generic/Task 来源控制 | `discovery.py` + `source_controls.py` + 同一原型/校准路径 | DTD/Aircraft 四项，`matched_source_control.json` |
| Generic 直接逐样本检索 | `generic.py` | 25 项；其目标为 demo 同类比例，与类别原型控制区分 |
| VICL 一/两示例摘要适配 | `vicl.py`, `prompts.vicl_*`, `models.GenerationBackend` | Qwen3/Qwen2.5 共 60 辅助结果，含 +D/+G |
| Prompt-SelF 对应空间 CLIP patch 适配 | `spatial.py`, `baselines.within_class(stable=True)` | 30 辅助结果；不冒称作者 DINOv2 CLS 实现 |
| MMICES question-only 与 RICES 计划等价 | `baselines.mmices_question_only` | 49,472 query-budget 计划，不另造模型结果 |
| accuracy、macro、invalid、rescue/harm、bootstrap CI、exact McNemar | `metrics.py`, `analysis.py`, `tools/recompute_artifact_results.py` | 主表全部配对、补充表及 60 辅助配对 |
| 类别选取读出与最终 VLM 答案 | `analysis.evaluate`, `plan_changes` | `selector_per_run.csv`, `selector_three_run_summary.csv` |
| 三次 Qwen3 重复 | `analysis.repeated_summary`, 固定 seed 配置 | `three_seed_method_results.csv`、`three_seed_sources.json`；逐运行先算跨数据集均值再 sample SD |
| 更高示例预算 | 同一检索 API 的 shots 参数 | `shot_curve_results.csv`；历史曲线复用 2-shot 系数，不能写成高 shot 独立校准结果 |
| 计划变化、来源和固定 batch 伙伴 | `hashing.py`, `pipeline.py`, `inference.py`, `models.GenerationBackend` | 原始审计/计划/预测 archive；相同 demo 的不同 batch 输出保留 |
| 成功与失败定性实例 | `algorithm_cases.json`, `tests/test_paper_contract.py` | 论文四幅图原始 PDF；保留事后选例规则，不当总体效果估计 |
| 全文与补充表 | `paper/`，`results/tables/`，`results/paper/` | 两个完整 tex、28 表、4 图、全部数值输入 |

代码列中的短名称均位于 `src/decisionicl/`。`results/paper/` 为本稿汇总输入，
`results/main_release/` 为旧版主实验发布证据原件，其状态字段不代表当前重复完成度。

这次工作覆盖论文全部已报告内容。新包中的模型适配器需要在实际硬件和冻结模型版本上
重新做兼容检查后才能作为新的执行入口；本次只做 CPU 整理和验证，没有新目标推理。

