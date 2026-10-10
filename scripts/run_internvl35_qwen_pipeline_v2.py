"""Single GPU, durable supervision of the dynamic-image four-dataset rerun."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from internvl35_qwen_config_v2 import (
    ROOT, DATASETS, METHODS, OUTPUT, ANALYSIS, SNAPSHOT, PROTOCOL_PATH,
    digest, write_json, now, ensure_protocol, load_model,
)

LOG_ROOT = OUTPUT / "pipeline_logs"


def cmd(script, *args):
    return [sys.executable, "-u", str(ROOT / "scripts" / script), *args]


def gpu_free():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    )
    if result.stdout.strip():
        raise RuntimeError(f"GPU has another compute process: {result.stdout.strip()}")


def lock_gpu():
    import msvcrt
    OUTPUT.mkdir(parents=True, exist_ok=True)
    handle = (OUTPUT / "supervisor.lock").open("a+b")
    handle.seek(0)
    if not handle.read(1):
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        handle.close()
        raise RuntimeError("Another v2 supervisor or smoke already holds the GPU lock")
    return handle


def prepare():
    from PIL import Image
    from transformers import AutoProcessor
    from multidataset_protocol import load_dataset
    from run_internvl35_qwen_sources_v2 import freeze_manifest
    # Bind the actual files once; later stages verify their sizes and modification times.
    path = ANALYSIS / "model_files.json"
    receipt = {p.name: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns,
                        "sha256": digest(p)}
               for p in sorted(SNAPSHOT.iterdir()) if p.is_file()}
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != receipt:
        raise ValueError("Pinned model receipt changed")
    write_json(path, receipt)
    ensure_protocol()
    processor = AutoProcessor.from_pretrained(SNAPSHOT, local_files_only=True)
    report = {"protocol_sha256": digest(PROTOCOL_PATH), "datasets": {}}
    for dataset in DATASETS:
        bank, query = load_dataset(dataset, ROOT / "datasets")
        folder = OUTPUT / dataset
        frozen = freeze_manifest(dataset, bank, query, folder)
        previous = json.loads((ROOT / "results/internvl35_full_v1" / dataset / "manifest.json").read_text(encoding="utf-8"))
        for split in ("bank", "query"):
            fields = lambda row: (row["sample_id"], row["label"], row["image_sha256"])
            if [fields(x) for x in frozen[split]] != [fields(x) for x in previous[split]]:
                raise ValueError(f"Previous canonical manifest differs: {dataset}/{split}")
        subprocess.run(cmd("run_internvl35_qwen_sources_v2.py", "--dataset", dataset, "--stage", "manifest"), cwd=ROOT, check=True)
        counts = {}
        for sample in bank + query:
            with Image.open(sample.path) as image:
                width, height = image.size
            patches = processor.image_processor.get_number_of_image_patches(height, width, {"crop_to_patches": True, "max_patches": 12})
            counts[sample.sample_id] = {"patches": patches, "visual_tokens": 256 * patches,
                                        "width": width, "height": height}
        write_json(folder / "image_budget.json", {"protocol_sha256": digest(PROTOCOL_PATH), "images": counts})
        report["datasets"][dataset] = {"bank": len(bank), "query": len(query),
            "manifest_sha256": digest(folder / "manifest.json"),
            "source_visual_tokens_min": min(x["visual_tokens"] for x in counts.values()),
            "source_visual_tokens_max": max(x["visual_tokens"] for x in counts.values())}
    write_json(ANALYSIS / "cpu_preflight.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def smoke():
    import torch
    from run_internvl35_qwen_methods_v2 import load_context
    from multimodal_model_adapter import anchor_hidden_states, prepare_batch, zero_shot_messages, icl_messages, generate_legal_labels
    from run_detriever_output_proxy import gold_answer_eos_state
    ensure_protocol()
    gpu_free()
    report_path = ANALYSIS / "gpu_smoke.json"
    report = {"state": "loading", "pid": os.getpid(), "started_utc": now(),
              "protocol_sha256": digest(PROTOCOL_PATH), "cases": {}}
    write_json(report_path, report)
    try:
        model, processor = load_model(ANALYSIS / "smoke_inputs.jsonl")
        for dataset in DATASETS:
            folder, bank, query, _ = load_context(dataset)
            labels = sorted({s.label for s in bank})
            counts = json.loads((folder / "image_budget.json").read_text(encoding="utf-8"))["images"]
            # Predeclared memory stress: the four largest-patch bank images and largest query.
            ranked_bank = sorted(bank, key=lambda s: (-counts[s.sample_id]["patches"], s.sample_id))
            sample = sorted(query, key=lambda s: (-counts[s.sample_id]["patches"], s.sample_id))[0]
            report.update(state="running", dataset=dataset)
            write_json(report_path, report)
            entry = {"query_id": sample.sample_id, "demo_ids": [s.sample_id for s in ranked_bank[:4]],
                     "expected_visual_tokens": sum(counts[s.sample_id]["visual_tokens"] for s in ranked_bank[:4] + [sample])}
            if dataset == "aircraft":
                report["stage"] = "single_image_states_and_proxy"
                write_json(report_path, report)
                inputs = prepare_batch(processor, [zero_shot_messages(bank[0], dataset, labels, None)])
                state = anchor_hidden_states(model, inputs)
                entry["state_shape"] = list(state.shape)
                entry["states_finite"] = bool(torch.isfinite(state).all())
                if state.shape != (1, 36, 2560) or not entry["states_finite"]:
                    raise ValueError("Invalid pre-answer states")
                del inputs, state
                proxy = gold_answer_eos_state(model, processor, bank[0], dataset, labels, None)
                entry["proxy_finite"] = bool(torch.isfinite(proxy).all())
                # Same GPT-MM extraction API used by the scientific runner.
                from run_multidataset_main import extract_gpt_mm_cache
                temp = ANALYSIS / "smoke_gpt_mm"
                temp.mkdir(exist_ok=True)
                extract_gpt_mm_cache(model, processor, bank[:1], dataset, labels, None, 1, 12,
                                     temp / "embeddings.npy", temp / "progress.json", temp / "outputs.jsonl", True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            report["stage"] = "five_image_legal_decode"
            write_json(report_path, report)
            prediction, raw = generate_legal_labels(
                model, processor, [icl_messages(ranked_bank[:4], sample, dataset, labels, None)], labels,
            )[0]
            entry.update(prediction=prediction, raw=raw, legal=prediction in labels,
                         peak_reserved_gb=torch.cuda.max_memory_reserved() / 2**30,
                         peak_allocated_gb=torch.cuda.max_memory_allocated() / 2**30)
            if not entry["legal"]:
                raise ValueError("Illegal smoke prediction")
            report["cases"][dataset] = entry
            write_json(report_path, report)
            print(f"smoke {dataset}: {entry}", flush=True)
        report.update(state="complete", finished_utc=now())
        write_json(report_path, report)
    except Exception as error:
        report.update(state="failed", error=repr(error), finished_utc=now())
        write_json(report_path, report)
        raise


def stages():
    for dataset in DATASETS:
        source = "run_internvl35_qwen_sources_v2.py"
        methods = "run_internvl35_qwen_methods_v2.py"
        base = ("--dataset", dataset)
        yield dataset, "anchor", cmd(source, *base, "--stage", "anchor", "--batch-size", "1")
        yield dataset, "train_proj", cmd(methods, *base, "--stage", "train_proj")
        yield dataset, "train_learn", cmd(methods, *base, "--stage", "train_learn")
        yield dataset, "gpt_mm_source", cmd(source, *base, "--stage", "gpt_mm", "--batch-size", "1")
        yield dataset, "detriever_proxy", cmd(methods, *base, "--stage", "proxy")
        yield dataset, "train_detriever", cmd(methods, *base, "--stage", "train_detriever")
        yield dataset, "clip_source", cmd(source, *base, "--stage", "clip", "--batch-size", "32")
        for method in METHODS:
            yield dataset, f"retrieve_{method}", cmd(methods, *base, "--stage", "retrieve", "--method", method)
        for method in METHODS:
            yield dataset, f"icl_{method}", cmd("run_internvl35_qwen_icl_v2.py", *base, "--method", method, "--batch-size", "1")
        yield dataset, "cpu_audit", cmd("audit_internvl35_qwen_config_v2.py", "--allow-incomplete")
    yield "all", "full_audit", cmd("audit_internvl35_qwen_config_v2.py")


def pipeline(resume=False):
    ensure_protocol()
    smoke_report = json.loads((ANALYSIS / "gpu_smoke.json").read_text(encoding="utf-8"))
    if smoke_report["state"] != "complete" or smoke_report["protocol_sha256"] != digest(PROTOCOL_PATH):
        raise RuntimeError("Exact-protocol GPU smoke must complete before full pipeline")
    gpu_free()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    status_path = LOG_ROOT / "status.json"
    planned = list(stages())
    binding = digest(PROTOCOL_PATH)
    if status_path.exists():
        if not resume:
            raise RuntimeError("Existing pipeline; use --resume for the exact protocol")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status["protocol_sha256"] != binding or status["state"] == "complete":
            raise ValueError("Cannot resume different or complete protocol")
        completed = [r for r in status["stages"] if r["state"] == "complete"]
        for r, (dataset, stage, argv) in zip(completed, planned):
            if (r["dataset"], r["stage"], r["command"], r["return_code"]) != (dataset, stage, argv, 0):
                raise ValueError("Completed stage does not match pinned plan")
        status.setdefault("history", []).extend(r for r in status["stages"] if r["state"] != "complete")
        status["stages"] = completed
    else:
        status = {"version": 2, "started_utc": now(), "protocol_sha256": binding, "stages": []}
    status.update(state="running", supervisor_pid=os.getpid(), checked_utc=now(), planned_stages=len(planned))
    write_json(status_path, status)
    for dataset, stage, argv in planned[len(status["stages"]):]:
        if (LOG_ROOT / "STOP_REQUESTED").exists():
            status.update(state="paused", checked_utc=now())
            write_json(status_path, status)
            return
        record = {"dataset": dataset, "stage": stage, "command": argv, "state": "running", "started_utc": now()}
        status["stages"].append(record)
        status["current"] = {"dataset": dataset, "stage": stage}
        write_json(status_path, status)
        log_path = LOG_ROOT / f"{dataset}_{stage}.log"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n{now()} {argv!r}\n")
            handle.flush()
            worker = subprocess.Popen(argv, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            record["worker_pid"] = worker.pid
            write_json(status_path, status)
            code = worker.wait()
        record.update(state="complete" if code == 0 else "failed", return_code=code, finished_utc=now(), log=str(log_path))
        status["checked_utc"] = now()
        if code != 0:
            status.update(state="failed", finished_utc=now())
        write_json(status_path, status)
        if code != 0:
            raise SystemExit(code)
    status.update(state="complete", finished_utc=now())
    status.pop("current", None)
    write_json(status_path, status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("prepare", "smoke", "pipeline", "plan"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "plan":
        print(json.dumps(list(stages()), indent=2))
    else:
        with lock_gpu():
            if args.stage == "smoke":
                smoke()
            else:
                pipeline(args.resume)


if __name__ == "__main__":
    main()
