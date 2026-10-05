# 数据与外部文件

当前范围：DTD、Aircraft、CUB、Pets、Dogs。Qwen主表seed73的完整test分别1880/3333/5794/3669/8580。数据与模型不随Git分发。

有序bank/query、标签、图SHA和split由原冻结计划约束。大合同以gzip原字节提供，执行 `python scripts/materialize_handoff_artifacts.py` 可在本地恢复；不会下载图片或启动模型。具体可用性见 [handoff/ARTIFACTS.md](handoff/ARTIFACTS.md)。

不要重划分数据或按test调整示例、query、checkpoint。目标模型和原处理设置见 [handoff/ENVIRONMENT.md](handoff/ENVIRONMENT.md)。
