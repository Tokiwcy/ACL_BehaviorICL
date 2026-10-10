# 新 coding agent 接手指南

> **2026-10-10 01:47 SGT 用户再次明确“继续”**：撤销下方22:44暂停。CUB Learn 2,699条完整预测及SHA、原科学协议/模型文件、完整已完成阶段的cache_guard命令计划通过CPU核验；暂停后无启动/异常关机事件，boot字段仅与之前已观察值相差1.800661秒，并非真实重启。当前入口 `scripts/resume_internvl35_user_pause_v2.py --authorization analysis/internvl35_joint_v2/user_continue_20261010_v2.json`，操作receipt `user_resume_1075062680e57a58.json`，日志 `user_resume_20261010.stdout.log` / `user_resume_20261010.stderr.log`。固定身份DWM PID2724、创建UTC 2026-10-09T01:08:36.6710440Z允许保留，其他GPU上下文拒绝；未知worker拒绝测试通过。原36条GPU门槛不重跑，原科学文件及逐条缓存清理保护不变，从Learn断点继续，恢复半小时汇报。旧已消费入口不得重启；新的真实系统崩溃不自动循环恢复，Hypervisor具体根因仍未定位。LFM、旧单patch和云端继续暂停。

> **2026-10-09 22:44 SGT explicit human pause overrides all earlier resume authorizations**: joint_v2 supervisor and CUB Learn worker were stopped by verified PID/command/creation identity; internvl heartbeat is PAUSED. Committed CUB predictions: raw 5794/5794, proj 5794/5794, learn 2699/5794. Pause receipt: `analysis/internvl35_joint_v2/user_pause_20261009T144456Z.json`; original status and transition bytes preserved. Do not restart a GPU worker or consumed launcher until the human explicitly resumes. The 36-case gate and frozen scientific bindings remain unchanged; cloud and LFM remain paused.

> **2026-10-09最新09:08蓝屏与用户明确恢复授权**：此次不是03:31计划更新，而是WER1001/Kernel41的HYPERVISOR_ERROR 0x20001，四参数与10月7日LFM故障一致。CUB Raw保留2877完整行。VBS/Hypervisor开启；事件尚不能锁定具体驱动/固件，Windows ACL拒绝读取新minidump，本机未找到调试器；不能声称已解决根因。用户明确要求“查原因然后重启实验”，已在独立GPU子进程复验最后已保存和下一未提交上下文：原输出逐字一致，下一输入合法。当前通用恢复入口 `scripts/resume_internvl35_checked_v1.py --incident analysis/internvl35_joint_v2/hypervisor_incident_20261009.json --expected-rows 2877`，只消费本次授权一次；新运行保护 `scripts/internvl35_icl_cache_guard_v1.py` 调用原解码后逐条同步/释放空闲缓存，提示/图像预算/BF16/模型/选择/4-shot/解码均不改。运行保护有独立SHA receipt，不改冻结科学文件；复验第一条保留显存14.13GB、清理后约9GB。当前科学worker已从2877条继续，半小时报告继续；若再次系统蓝屏不自动循环启动，保留进度并报告。

> 本次证据 `analysis/internvl35_joint_v2/hypervisor_incident_20261009.json`；复验 `checked_replay_4fec08f0e9e0bd9c.json`；操作receipt `checked_resume_4fec08f0e9e0bd9c.json`；日志 `checked_resume.stdout.log` / `checked_resume.stderr.log`。后续预测阶段用cache_guard入口；再次用户明确恢复应使用通用checked入口，旧恢复启动器的plan包含原ICL脚本路径，不能直接恢复已完成guard阶段。模型相关缓存/参数继续独立joint_v2，云端/LFM暂停。没有修改VBS、安全设置、BIOS或驱动。

> **2026-10-09最新故障诊断与恢复**：03:31:16 User32/1074记录TrustedInstaller发起“操作系统升级（计划内）”，原因0x80020003；随后KB5129195/KB5126052安装成功。此次12小时事件窗口无41/1001/6008，无新minidump，不能称为GPU/Hypervisor崩溃。CUB Raw保留2688/5794完整行，协议/选择/顺序核验通过。当前恢复入口 `scripts/resume_internvl35_planned_update_v1.py`，先在独立GPU子进程复验最后已保存四示例上下文，输出逐字相同后才续跑原joint_v2；复验已通过，峰值保留13.96GB。证据 `analysis/internvl35_joint_v2/windows_update_restart_20261009.json`，操作receipt `planned_update_resume_20261009_v1.json`，日志 `planned_update_resume.stderr.log` / `stdout.log`。原boot状态字节另存，模型/科学协议/缓存不改。当前不要重启旧操作入口；确认新PID与真实状态。半小时汇报继续。系统更新设置尚未由本任务修改。

> **最新用户授权：18:58暂停后再次明确“继续”**。joint_v2协议、36条GPU门槛、2603条缓存/答案提交及同次boot身份核验通过，CUB从2603/11788断点恢复，半小时汇报恢复。当前监督入口 `scripts/resume_internvl35_joint_operations_v1.py`，日志 `analysis/internvl35_joint_v2/operation_resume.stdout.log` / `operation_resume.stderr.log`，操作receipt `explicit_resume_*.json`。操作守卫辨认本次boot固定身份的DWM与Windows官方ShellExperienceHost为桌面上下文，其余GPU计算进程拒绝；科学worker仅一个，科学代码和协议不变。旧handover入口不重启。下方18:58暂停保留为历史，已被本次授权撤销；LFM/旧单patch/云端仍暂停。

> **2026-10-08 18:58用户再次明确暂停，覆盖下方所有恢复授权**：joint_v2监督器与CUB科学worker已精确停止，GPU无计算进程；半小时`internvl`自动化设为PAUSED。CUB三份缓存和答案已提交2603/11788（bank proxy2603/5994），答案文件2603条完整行，无额外残缺尾行；原状态字节与暂停receipt在 `analysis/internvl35_joint_v2/user_pause_20261008T105830Z.json`及同目录snapshot。Aircraft六格已完整审计，合计6/24；DTD另6/6。joint_v2固定36条GPU门槛已通过，无须重复验证。未获用户再次明确恢复前，不启动GPU或自动汇报。

> **最新执行变更**：Aircraft已补完并审计6/6。joint_v1缓存gold suffix在实际GPU检查首条CUB样本cosine0.9996466，低于冻结0.9999门槛，失败记录保留。继续用户授权时使用joint_v2：共享原HF生成prefill与所有anchor，gold proxy回到原完整forward（bank两次vision、query一次）；阈值不降、训练/图像/解码不改。独立v2目录与入口 `scripts/run_internvl35_joint_pipeline_v2.py --stage handover`，先固定36条GPU验证再CUB→Dogs→Pets；结果写CURRENT_RESULTS第8节。先查v2真实状态，禁止重启v1失败gate或重复启动v2。

> **2026-10-08用户已明确“继续”**：撤销下方17:18暂停。系统实际于17:21重启；已核对GPU空闲、无实验进程、冻结协议/模型文件状态、3329条已有RICES顺序/示例身份及剩余4条图片SHA。新的显式恢复操作是 `scripts/resume_internvl35_after_user_pause_v1.py`，先补完Aircraft→独立CPU审计六格→联合GPU兼容门槛→CUB/Dogs/Pets。科学协议及原暂停/启动记录保留，新的boot身份和启动code SHA在 `analysis/internvl35_joint_v1/user_resume_20261008_v1.json`；实时日志 `user_resume.stdout.log` / `user_resume.stderr.log`。恢复半小时汇报；LFM、旧单patch与云端继续暂停。不要重复执行已消费恢复入口，先看真实进程与状态。

> **2026-10-08 17:18用户再次明确暂停**：Aircraft GPU worker、v4监督器和joint切换等待器均已精确停止，`internvl`半小时自动化已设为PAUSED。Aircraft RICES保留3329/3333条完整JSONL，末尾无残缺行，尚无完整summary；其余五方法已审计，合计5/24，DTD另6/6。CUB联合提取GPU兼容检查未开始。暂停记录为 `analysis/internvl35_joint_v1/user_pause_20261008.json`；状态原件备份保留。未获用户再次恢复授权前，不启动GPU、切换等待器或自动汇报。以下17:18以前的启动授权与计划均不能覆盖本次暂停。

> 2026-10-08最新授权：从CUB接入[联合hidden-state提取v1](INTERNVL35_JOINT_SOURCE_V1.md)。Aircraft v4先完成；旧监督器已请求阶段边界停止。新隐藏等待器完成Aircraft CPU审计与GPU独占检查后，运行固定pilot兼容门槛，通过才接CUB→Dogs→Pets。新目录 `results/internvl35_joint_v1`；状态/兼容证据未complete前不能称GPU门槛已通过。不得重复启动v4或联合流水线；继续半小时报告、更新CURRENT_RESULTS，DTD保留。LFM和云端仍暂停。

> 2026-10-07 最新用户授权：LFM已取消，旧单patch InternVL保持冻结；从Aircraft开始重跑动态分块 InternVL3.5-4B，接CUB/Dogs/Pets，4-shot六方法共24格。正式入口、CPU/GPU核验和进度见 [InternVL Qwen配置v4](INTERNVL35_QWEN_CONFIG_V4.md)。本地v4流水线已启动，半小时汇报跟踪v4；不得恢复LFM或旧InternVL目录，云端仍暂停。以下原接手快照保留其历史时间与研究结论。

> 新的本机 GPU Qwen TAM 与四提示检索消融见 [LOCAL_GPU_STUDY.md](LOCAL_GPU_STUDY.md)；dense InternVL3.5-4B 五数据集小样本验证见 [INTERNVL35_LOCAL_STUDY.md](INTERNVL35_LOCAL_STUDY.md)。以下是原接手快照，原云端自动任务仍暂停。

这是2026-10-06的当前研究快照。**先读本页，运行离线自检，再开始用户授权的工作。** 仓库更新不等于恢复实验；云端/自动监督仍暂停。

## 你需要理解的一句话

外部图像相似性未必等于目标VLM可利用的示例效用。BehaviorICL让冻结目标VLM联合处理任务文本与图片，用36层最后输入位置的回答前状态检索top4，再让同一VLM使用原始demo图片与标签进行ICL。Raw、固定256 Proj、Learn的区别见 [方法介绍](PROJECT_OVERVIEW.md) 和 [代码地图](CURRENT_CODE.md)。

## 目前真的有什么结果

| 工作 | 实际状态 |
|---|---|
| Qwen五组六方法主表 | seed73的30组完整test已核验；没有三seed std |
| task/generic source prompt | DTD+CUB完整test，+5.11/+34.29pp；改变检索source，最终ICL任务不变 |
| 固定demo标签 | 完整DTD+CUB，Gold−own +18.96/+44.78pp；配对n1809/5695，缺失71/99保留；另有五×256样本 |
| QwenVision+SupCon | DTD85.32/CUB84.33%；历史Proj对照不支持decoder普遍更优 |
| strict-match/rescue | 完整回顾分析；覆盖有GPT-MM/QV反例，非独立因果 |
| TAM | **无实际maps**；官方代码/八图64run协议/CPU适配与分析代码已有 |
| InternVL / 新seed74、75 | 未完成；GPU均已精确停止，失败不当科学结果 |

完整表及CI读 [CURRENT_RESULTS.md](CURRENT_RESULTS.md)。本次发布包括原audit、汇总CSV、标签query-arm输出，以及压缩的完整query-level匹配CSV；不包括权重、图片和原始npy/pt/云端FULL。

## 第一次接手：无需GPU的具体步骤

```bash
python scripts/verify_handoff.py
python scripts/materialize_handoff_artifacts.py
python scripts/verify_handoff.py
```

第一条验证发布文件、原始SHA、30组计数、配对分母、TAM协议和可移植链接；第二条仅离线恢复已发布gzip里的大JSON/CSV；第三条还会验证已恢复原件身份。缺数据/模型时，不运行原科学preflight冒充兼容。

接着读 [TAM源与索引说明](handoff/TAM_UPSTREAM.md)，检查 `scripts/query_tam_adapter_v1.py`、`run_query_tam_pilot_v1.py`、`analyze_query_tam_pilot_v1.py`。若已有CPU torch/numpy/scipy环境，可下载固定上游并执行三个合成CPU测试：

```bash
python scripts/fetch_tam_upstream.py --fetch
python -m unittest discover -s tests -p test_query_tam_pilot_v1.py -v
```

这些测试不载入预训练权重、不使用CUDA、不访问RunPod。环境和原科学版本记录见 [ENVIRONMENT.md](handoff/ENVIRONMENT.md)。没有相应CPU依赖时，先完成标准库自检，不安装/运行GPU模型来“测试”。

## 第一项研究工作：固定图改变query的TAM

协议是 `analysis/query_tam_pilot_v1/protocol.json`：DTD/CUB各4图，按历史task-only、generic-only、都对、都错分层SHA固定选图。四query×自然生成和固定gold teacher-forcing两模式=64runs。保留原Qwen3模型revision、processor、实际pixels/grid和tokenIDs。

**当前第一安全动作**是审阅/CPU测试适配器、列出缺失的八图与固定模型文件，再准备最小实际兼容检查方案。当前pause下不申请卡。若用户明确恢复TAM，先用协议内一图检查模型/native、输入/目标索引、raw/post-ECI/36层shape、真实备份，再运行完整64runs；不是把旧V2重新执行。

完整交付：八图全部固定尺度热图、自然词/固定目标分开、raw CAM/post-ECI分开、每层anchor变化指标、完整maps/token/grid身份及FULL取回/停机/CPU审计。负结果和不变图同样交付，不根据热图挑例子。

注意：image-before-query时，固定目标的raw image-token CAM可能不随后置query变化；ECI文本校正图变化不等于attention移动。原accuracy提升来自source prompt→demo变化→ICL，需进一步连接该query实际top4及预测，不能仅凭热图解释所有improvement。详见 [NEXT_STEPS.md](NEXT_STEPS.md)。

## 已知工程问题，避免重复绕圈

| 问题 | 证据和处理边界 |
|---|---|
| TAM V2 container初始化超期限 | 没有SSH/native/worker，已停止；V1/V2 consumed。新卡操作需要新身份及实际连通/最小兼容，不能直接重调 |
| Intern V75 CLIP strict load两个position_ids键 | 完整失败FULL已核验；合成CPU窄buffer修复已写，真实checkpoint buffer相等性和完整runner/parent/包链尚未证明。禁止strict=False |
| first V76原Linux CPUvalidator90s超时 | 原因Unknown，缺远端FULL；不能用Windows配置速度猜原因或单纯抬timeout |

修复参考：`behavior_clip_position_ids_compat_v78.py`、`behavior_internvl_clip_runtime_v79.py`；它们是CPU准备，**不是新的可启动云入口**。旧NumPy首次exec顺序已在真实Linux证明修复，不重新把它当未解决故障。详情 [RUN_STATUS.md](RUN_STATUS.md)。

## 仓库边界与后续更新

原冻结计划和科学代码字节保留；大锁/CSV可离线恢复。部分原consumer仍有本地路径和私有`.tools`要求，它们是历史合同消费者，不是fresh clone即可完整执行的云流水线。用新版本的可移植适配替代，不改历史SHA来掩盖缺件。

如需完整科学复核，请向维护者取得 [外部artifact清单](handoff/ARTIFACTS.md) 中的原图片/预测/source/checkpoint/FULL，核SHA后再审计。本仓库的CI只核发布内容，不宣称重放全部模型或kernel。

早期不必要的public脚本已从当前tree移除，可由Git历史恢复；必要旧文件名共享依赖保留。优先TAM及其检索证据连接，三seed/Intern按人类新优先级再恢复，不自动接续旧科学队列。
