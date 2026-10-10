# 2026-10-10 本地研究发布快照

本次发布补充2026-10-06之后的本地代码、版本化协议、CPU分析、运行/失败证据和TAM对照图。原 `publish_manifest.json`、原科学锁与历史Qwen结果保留原字节；新增文件由 `local_publish_manifest_20261010.json` 独立绑定。这里是发布时快照，不是实时GPU状态或新机器的执行授权。

## 科学范围

- Qwen新版本地TAM：640张固定图、1,280次固定目标主比较，加有限自然/完整提示控制，共1,400份记录。旧云端八图64-run协议仍未完成。指标不支持一致的合理前景注意转移。
- 四提示小bank检索、层/聚合消融及事后query-delta诊断独立记录，不能替代完整test ICL主表。
- InternVL五数据集每组32张query的检索/ICL/答案状态面板，保持其样本分母和单patch设置。
- InternVL全量：DTD旧单patch六格、Aircraft动态v4六格、CUB joint_v2的Raw/Proj/Learn/GPT-MM四格完整审计；CUB DeTriever进行中，RICES和Dogs/Pets未完成。动态重跑合计10/24，DTD另6/6。最新分母、完整准确率及配对CI见仓库根目录 `CURRENT_RESULTS.md` 第8节。
- v2/v3动态smoke、joint_v1 cached-gold门槛失败、LFM取消、Windows计划升级和Hypervisor蓝屏分别保留为工程证据，不计科学结果。

## 离线复核

```bash
python scripts/verify_handoff.py
python scripts/verify_local_publication_v1.py
```

两者仅核对公开字节、SHA、语法、依赖/链接及结构化计数，不加载模型、CUDA或云端。原hand-off检查输出的 `TAM_actual_results: 0` 指原云端协议，不否定新增本地实验。新增校验通过也不代表clone已重跑全部模型或现场审计。

可选CPU实现测试：`python -m unittest discover -s tests -p 'test_internvl35*.py' -v`。需要torch/numpy/transformers；Windows锁测试在其他系统上不可直接执行。测试使用合成小模型，不使用预训练权重。

## 未随Git发布的输入

`results/` 的原始预测、telemetry、特征数组和训练checkpoint，`datasets/` 图像与空间标注、固定revision权重、`.tools/`缓存/运行环境和私有云状态仍在本机。新进程PID/boot/授权receipt只是本机历史证据，不能被另一个环境当作有效恢复身份。完整审计需恢复精确外部输入；失败smoke、预分配、部分预测和GitHub快照都不当完整结果。

原文件中的本机路径为审计溯源，尚未改写成跨平台模型执行入口。公开图是报告对照图，不包含完整数据集。未完成atomic `.tmp` 和重复准备控制台输出留在本地，不作为发布证据。
