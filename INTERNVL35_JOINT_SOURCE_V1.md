# InternVL 联合提取：从 CUB 接入

> **最新科学版本 joint_v2**：Aircraft已补完3333/3333，六格通过独立审计（RICES1434/3333=43.02%）。joint_v1 GPU门槛在首个固定CUB bank样本失败：anchor和GPT-MM最大绝对差均0、生成token相同；cached gold proxy cosine0.999646604低于预定0.9999，最大绝对差0.375。没有放宽门槛、选择测试准确率或修改v1证据。v2保留共享HF生成prefill与GPT-MM，bank gold proxy使用原完整teacher-forced forward；bank实际vision forward为2次、query为1次。v2独立协议、特征、参数与审计目录为 `analysis/internvl35_joint_v2` / `results/internvl35_joint_v2`，继续相同36条pilot、cosine0.9999、exact生成token与ordered top4门槛，通过后接三组全量。v1记录冻结，下面v1 cached suffix细节保留其历史身份。

> 最新操作状态：2026-10-08 17:18用户暂停，本地进程已精确停止，RICES保存3329/3333。用户随后明确“继续”，系统实际在17:21重新启动。新单一监督入口 `scripts/resume_internvl35_after_user_pause_v1.py` 以本次显式授权和当前boot记录恢复：先补4条、审计Aircraft六格，再运行原兼容门槛及三组流水线。原协议/科学代码不变，暂停与旧transition字节备份保留。恢复receipt与日志在 `analysis/internvl35_joint_v1/user_resume_20261008_v1.json`、`user_resume.stdout.log`、`user_resume.stderr.log`；半小时自动监督已恢复。17:07的时钟guard是旧操作历史，不用于跨越本次真实重启。

用户于2026-10-08授权从CUB开始接入。Aircraft当前v4 RICES完成后，旧监督器通过 `STOP_REQUESTED` 在阶段边界退出。先独立CPU审计Aircraft六方法，确认GPU空闲并获得与v4相同的操作系统锁，才做新GPU兼容检查。旧v4代码、协议、Aircraft特征/参数/预测不改写。

## 计算与科学身份

- CUB→Dogs→Pets，全bank/full test、4-shot、seed73、六方法共18格，独立目录 `results/internvl35_joint_v1`，新协议 `analysis/internvl35_joint_v1/protocol.json`。继承并核验v4冻结科学配置与全部依赖SHA，再绑定新联合提取、检查器和监督器SHA。
- 调用原HF `generate`（greedy、12 new tokens、batch1）：第一步37个states中排除embedding，保存36层最后输入位置的anchor。BehaviorICL三个版本和DeTriever输入共享它。
- GPT-MM使用同一次生成的答案和原hidden-step索引。EOS位置与达到12-token上限时的历史取法保留，不把cap修成另一科学实现。
- bank额外把生成返回的KV cache裁回原输入长度，接完整gold assistant模板的剩余token，包括 `im_end` 和末尾换行，保存最后状态作DeTriever proxy。每条确认完整gold模板的输入前缀及pixel tensor相同；query不传gold。
- 短cached gold分支使用数学SDPA，支持offset causal mask；长图像prefill和GPT-MM decode沿用v4后端。浮点运算路径改变，必须有新版本与真实GPU兼容门槛，不声称bitwise/full-bank等价。
- 每条图像只有一次实际vision forward；CPU processor可能再次读取bank图片以核对模板，但不会再次GPU编码图片。
- 原生动态图像12patch加thumbnail、提示、256投影、训练1500/10000步、bank-only开发集、最终候选、合法解码保持v4配置。CLIP另提取；模型相关参数全部在新目录重新训练。

## 兼容门槛与恢复

固定每组按SHA256(dataset/sample_id)排序取8bank+4query，共36条。不使用test准确率选预算或阈值。GPU对照原独立anchor/GPT-MM/gold-proxy：状态cosine最小值≥0.9999，生成token逐个相同，实际vision forward恰好一次，Raw/GPT-MM固定8bank的ordered top4一致；bank proxy cosine top4（排除自身）一致。记录差值、tokens、完整gold suffix、峰值显存。有限pilot不证明全bank所有检索/训练结果完全相等。

CPU测试覆盖真实小型随机初始化Qwen decoder的cache分支/full-forward等价、EOS/cap规则、模板前缀不一致拒绝、query不读取gold、已提交答案校验及中断恢复。它不是预训练InternVL的GPU验证。

三份NPY和答案行flush后，先更新 `joint_progress.json` 单一提交游标，再发布原消费者的三个progress文件。恢复时按权威游标截去未提交答案尾部，重新写未提交行；协议、manifest和兼容receipt身份变动拒绝恢复。预分配NPY不算完成。

监督器两个锁：独占transition锁防重复等待器，与v4共用GPU锁。旧监督器退出前等待器只做CPU状态检查。失败立即停止；重启Windows后的boot identity变化拒绝自动恢复。GPU兼容失败保留receipt，不循环尝试。

## 入口与记录

```powershell
.venv\Scripts\python.exe scripts/run_internvl35_joint_pipeline_v1.py --stage prepare
.venv\Scripts\python.exe -m unittest discover -s tests -p test_internvl35_joint_v1.py -v
.venv\Scripts\python.exe scripts/run_internvl35_joint_pipeline_v1.py --stage handover
```

最后一条由后台隐藏监督器执行，等待Aircraft完成后顺序兼容检查→CUB→Dogs→Pets。真实阶段在 `results/internvl35_joint_v1/pipeline_logs/status.json`，等待/切换在 `analysis/internvl35_joint_v1/transition.json`，GPU门槛在 `compatibility.json`。未出现complete之前，不报告已通过GPU门槛。

半小时监督继续；`scripts/update_internvl35_qwen_results_v4.py --audit` 保留Qwen历史主表，在CURRENT_RESULTS第8节合并DTD旧单patch6格、Aircraft v4六格、joint三组六方法。DTD设置差异保留，不给统一视觉预算的五组均值。LFM取消、旧单patchInternVL冻结、云端GPU暂停。


## 2026-10-08 17:07 操作监督修正

WMI LastBootUpTime 从2026-10-07T08:28:55.5000000Z变为08:28:58.3574150Z（偏移2.857415秒）。原Aircraft监督器和GPU worker的创建时间未变，近三小时无System41/6005/6008重启事件；正在等待的CPU进程也跨越该变化保持运行。没有证据支持实际系统重启。严格ISO比较会误停CUB切换。

因此新增操作层 `scripts/run_internvl35_joint_handover_clock_guard_v1.py`，保留冻结科学代码和协议。仅在同一已诊断waiting阶段，将启动时间字段≤10秒的偏移视为时钟校准；超过10秒仍交给原检查拒绝恢复。3项CPU边界检查通过，包括真实下一天boot与11秒偏移均拒绝归一化。只替换CPU等待器，Aircraft GPU worker不变。原transition字节保存在 `analysis/internvl35_joint_v1/transition.pre_clock_guard_v1.json`；操作wrapper的SHA、时间偏移与判断依据在 `operational_clock_guard_v1.json`，新日志为 `clock_guard.stdout.log` 和 `clock_guard.stderr.log`。原科学协议SHA不变。
