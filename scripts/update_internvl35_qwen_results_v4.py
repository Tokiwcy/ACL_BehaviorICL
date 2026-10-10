"""CPU-only update of the InternVL results section beside the historical Qwen table."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/internvl35_qwen_config_v4"
ANALYSIS = ROOT / "analysis/internvl35_qwen_config_v4"
JOINT_RUN = ROOT / "results/internvl35_joint_v1"
JOINT_ANALYSIS = ROOT / "analysis/internvl35_joint_v1"
JOINT_VERSION = 2 if (ROOT / "analysis/internvl35_joint_v2/protocol.json").exists() else 1
if JOINT_VERSION == 2:
    JOINT_RUN = ROOT / "results/internvl35_joint_v2"
    JOINT_ANALYSIS = ROOT / "analysis/internvl35_joint_v2"
DATASETS = ("aircraft", "cub", "dogs", "pets")
METHODS = ("raw", "proj", "learn", "gpt_mm", "detriever", "rices")
COUNTS = {"aircraft": 3333, "cub": 5794, "dogs": 8580, "pets": 3669}
NAMES = {"aircraft": "Aircraft", "cub": "CUB", "dogs": "Dogs", "pets": "Pets"}
METHOD_NAMES = {"raw": "Raw", "proj": "Proj", "learn": "Learn", "gpt_mm": "GPT-MM-style", "detriever": "DeTriever-style", "rices": "RICES"}
START = "<!-- INTERNVL_QWEN_V4_RESULTS_START -->"
END = "<!-- INTERNVL_QWEN_V4_RESULTS_END -->"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def progress(status):
    if status.get("version") == "joint_transition":
        return f"Aircraft结束后切换CUB联合提取：{status['state']}"
    current = status.get("current")
    if not current:
        return status["state"]
    dataset, stage = current["dataset"], current["stage"]
    run = JOINT_RUN if status.get("version") in ("joint_v1", "joint_v2") else RUN
    folder = run / dataset
    files = {"anchor": folder / "anchor_states_progress.json",
             "gpt_mm_source": folder / "gpt_mm_progress.json",
             "detriever_proxy": folder / "trained/detriever/gold_input_answer_eos_progress.json",
             "clip_source": folder / "clip_features.progress.json"}
    files["joint_source"] = folder / "joint_progress.json"
    text = f"{NAMES.get(dataset, dataset)} / `{stage}`"
    if stage in files and files[stage].exists():
        item = read(files[stage])
        text += f"，{item['completed']:,}/{item['shape'][0]:,}"
    elif stage.startswith("icl_"):
        path = folder / "predictions" / f"{stage[4:]}.jsonl"
        count = 0
        if path.exists():
            with path.open("rb") as handle:
                count = sum(line.endswith(b"\n") for line in handle)
        text += f"，{count:,}/{COUNTS[dataset]:,}"
    elif stage.startswith("train_"):
        path = run / "pipeline_logs" / f"{dataset}_{stage}.log"
        if path.exists():
            content = path.read_text(encoding="utf-8", errors="replace")
            steps = re.findall(r"(?:step\s+|train\s+)(\d+)", content)
            if steps:
                text += f"，日志最近记录 {int(steps[-1]):,}/{10000 if stage == 'train_detriever' else 1500:,} 步"
    return text


def block(status, audit, dtd_audit):
    time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
    complete = sum(audit["datasets"][d]["methods"][m]["state"] == "complete" for d in DATASETS for m in METHODS)
    dtd_methods = dtd_audit["datasets"]["dtd"]["methods"]
    if any(dtd_methods[m]["state"] != "complete" or dtd_methods[m]["total"] != 1880
           or dtd_methods[m]["completed"] != 1880 for m in METHODS):
        raise ValueError("DTD requires all six independently audited full-test cells")
    lines = [START, "## 8. InternVL3.5-4B：五数据集结果与四组重跑进度", "",
             f"记录更新：**{time}（新加坡时间）**。流水线状态：`{status['state']}`；当前阶段：**{progress(status)}**；本轮独立 CPU 审计完成 **{complete}/24 格**，已保留 DTD **6/6 格**。",
             "",
             "DTD已审计的完整test结果直接列入下表；用户确认不重跑DTD。Aircraft、CUB、Dogs、Pets均全量重跑六方法，本轮共21,376 query、128,256条预测。下表仅填独立审计通过的完整方法格，未完成显示“待审计”。",
             "",
             "| 方法 | DTD（1880，旧单patch） | Aircraft（3333，动态重跑） | CUB（5794，动态重跑） | Dogs（8580，动态重跑） | Pets（3669，动态重跑） |",
             "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        dtd = dtd_methods[method]
        cells = [f"{100 * dtd['accuracy']:.2f}%（{dtd['correct']}/1880）"]
        for dataset in DATASETS:
            item = audit["datasets"][dataset]["methods"][method]
            if item["state"] == "complete":
                if item["total"] != COUNTS[dataset] or item["completed"] != COUNTS[dataset]:
                    raise ValueError("Full-result denominator differs from official test")
                cells.append(f"{100 * item['accuracy']:.2f}%（{item['correct']}/{item['total']}）")
            else:
                cells.append("待审计")
        lines.append("| " + METHOD_NAMES[method] + " | " + " | ".join(cells) + " |")
    lines += ["", "配对差值：Raw/Proj/Learn分别对GPT-MM-style和DeTriever-style；仅在同数据集两个完整方法格都通过审计后记录，单位pp，95% query配对bootstrap CI。", ""]
    pairs = [(p, dtd_audit, 1880, "DTD（旧单patch）") for p in dtd_audit.get("pairs", []) if p["dataset"] == "dtd"]
    pairs += [(p, audit, COUNTS[p["dataset"]], NAMES[p["dataset"]]) for p in audit.get("pairs", [])]
    if pairs:
        lines += ["| 数据集 | 比较（前者−后者） | 配对query | 差值pp [95% CI] |",
                  "|---|---|---:|---:|"]
        for pair, source, total, label in pairs:
            dataset, first, second = pair["dataset"], pair["first"], pair["second"]
            if (pair["queries"] != total
                    or source["datasets"][dataset]["methods"][first]["state"] != "complete"
                    or source["datasets"][dataset]["methods"][second]["state"] != "complete"):
                raise ValueError("Pair lacks two audited full-test method cells")
            effect = f"{pair['difference_pp']:+.2f} [{pair['paired_bootstrap_low_pp']:.2f}, {pair['paired_bootstrap_high_pp']:.2f}]"
            lines.append(f"| {label} | {METHOD_NAMES[first]}−{METHOD_NAMES[second]} | {pair['queries']} | {effect} |")
    else:
        lines.append("当前尚无完整方法对可报告。")
    lines += ["", "协议：固定BF16 InternVL revision；原生动态分块max_patches12加thumbnail、4-shot、seed73、Proj/Learn共享256维投影及1500步bank-only训练、DeTriever10000步、相同分类提示和合法标签贪心解码。重新提取全部模型source与答案proxy、训练模型相关检索参数；旧单patch缓存未复用。Windows执行后端固定cuDNN prefill加数学SDPA单token解码。",
              "",
              "四组最大五图兼容检查完成（每条16,640视觉token，峰值保留显存最高14.39GB）；该检查不计入准确率。表中DTD使用旧单patch设置，其余四组使用动态处理；表格保留该设置差异，不计算统一设置的五组均值。LFM已取消、旧InternVL流水线冻结、云端GPU暂停。",
              "",
              "Aircraft证据入口：[运行协议](INTERNVL35_QWEN_CONFIG_V4.md)、[冻结科学协议](analysis/internvl35_qwen_config_v4/protocol.json)、[独立CPU审计](analysis/internvl35_qwen_config_v4/" + ("full_audit.json" if (ANALYSIS / "full_audit.json").exists() else "progress_audit.json") + ")。Aircraft原始预测与实际视觉token记录在本地 `results/internvl35_qwen_config_v4/aircraft/predictions/` 和 `telemetry/`。",
              "", "DTD来源：[旧单patch协议](INTERNVL35_FULL_PROTOCOL.md)、[DTD独立CPU审计](analysis/internvl35_full_v1/progress_audit.json)。DTD六格均覆盖1,880 query，非法/未知输出为0。",
              "", "此节由 `scripts/update_internvl35_qwen_results_v4.py --audit` 根据真实文件更新；Qwen历史数字和原始审计保留。", END]
    if (JOINT_ANALYSIS / "protocol.json").exists():
        gate_path = JOINT_ANALYSIS / "compatibility.json"
        gate = read(gate_path).get("state") if gate_path.exists() else "尚未运行GPU兼容检查"
        proxy_policy = "bank金标签proxy保留原完整forward，KV suffix方案未通过旧门槛" if JOINT_VERSION == 2 else "bank金标签短suffix使用数学SDPA"
        lines[-1:-1] = ["", f"从CUB开始接入[联合提取v{JOINT_VERSION}](INTERNVL35_JOINT_SOURCE_V1.md)：Aircraft保留v4；CUB/Dogs/Pets独立缓存与参数在 `results/internvl35_joint_v{JOINT_VERSION}/`，独立审计在 `analysis/internvl35_joint_v{JOINT_VERSION}/`。共享GPT-MM生成的prefill供所有回答前anchor，保留GPT-MM原生成规则，{proxy_policy}。GPU兼容门槛当前状态：`{gate}`。该优化有独立代码与协议身份；有限pilot不能证明全bank检索排名逐条相等。"]
    return "\n".join(lines)


def refresh_incomplete(audit):
    """Derive the composite pending list from its actual merged method states."""
    audit["incomplete"] = [
        f"{d}/{m}: {r['state']} {r.get('completed', 0)}/{r.get('total', 0)}"
        for d in DATASETS for m in METHODS
        for r in [audit["datasets"][d]["methods"][m]]
        if r["state"] != "complete"
    ]
    audit["state"] = "incomplete" if audit["incomplete"] else "complete"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true", help="Run independent CPU audit first; no CUDA/model calls")
    args = parser.parse_args()
    if args.audit:
        subprocess.run([sys.executable, str(ROOT / "scripts/audit_internvl35_qwen_config_v4.py"), "--allow-incomplete"], cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(ROOT / "scripts/audit_internvl35_full_v1.py"), "--allow-incomplete"], cwd=ROOT, check=True)
        if (JOINT_ANALYSIS / "protocol.json").exists():
            subprocess.run([sys.executable, str(ROOT / f"scripts/audit_internvl35_joint_v{JOINT_VERSION}.py"), "--allow-incomplete"], cwd=ROOT, check=True)
    status = read(RUN / "pipeline_logs/status.json")
    path = ANALYSIS / "full_audit.json"
    if not path.exists():
        path = ANALYSIS / "progress_audit.json"
    audit = read(path)
    if (JOINT_ANALYSIS / "progress_audit.json").exists() or (JOINT_ANALYSIS / "full_audit.json").exists():
        joint_path = JOINT_ANALYSIS / "full_audit.json"
        if not joint_path.exists():
            joint_path = JOINT_ANALYSIS / "progress_audit.json"
        joint_audit = read(joint_path)
        for dataset in ("cub", "dogs", "pets"):
            audit["datasets"][dataset] = joint_audit["datasets"][dataset]
        audit["pairs"] = [p for p in audit["pairs"] if p["dataset"] == "aircraft"] + joint_audit["pairs"]
        refresh_incomplete(audit)
        from internvl35_joint_config_v1 import write_json as atomic_write, digest
        atomic_write(JOINT_ANALYSIS / "combined_results_audit.json", {
            **audit, "basis": {"aircraft_audit_sha256": digest(path),
                               "joint_audit_sha256": digest(joint_path)},
            "note": "Composite of independently audited Aircraft v4 and joint CUB/Dogs/Pets; DTD is separate old single-patch audit"})
        if (JOINT_RUN / "pipeline_logs/status.json").exists():
            status = read(JOINT_RUN / "pipeline_logs/status.json")
        elif status["state"] == "paused" and (JOINT_ANALYSIS / "transition.json").exists():
            status = {**read(JOINT_ANALYSIS / "transition.json"), "version": "joint_transition"}
    dtd_audit = read(ROOT / "analysis/internvl35_full_v1/progress_audit.json")
    document = ROOT / "CURRENT_RESULTS.md"
    original = document.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in original else "\n"
    section = block(status, audit, dtd_audit).replace("\n", newline)
    if (START in original) != (END in original):
        raise ValueError("Malformed managed-results section markers")
    if START in original:
        begin, end = original.index(START), original.index(END) + len(END)
        updated = original[:begin] + section + original[end:]
    else:
        updated = original.rstrip("\r\n") + newline * 2 + section + newline
    document.write_bytes(updated.encode("utf-8"))
    print(f"Updated CURRENT_RESULTS.md: {progress(status)}")


if __name__ == "__main__":
    main()
