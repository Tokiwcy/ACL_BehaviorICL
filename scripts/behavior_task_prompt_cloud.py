"""Independent prompt operational policy. CPU only unless called on exact cloud.

This does not import or invoke any old deployment, scientific queue or watcher.
The frozen scientific implementation is read-only.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import subprocess
import tarfile
import time

import behavior_task_prompt_policy as p

ROOT = p.ROOT
WORK = ROOT / ".tools/paper_task_prompt_cloud_v1"
OUT = ROOT / "results" / p.OPERATION
PARENT = "scripts/bootstrap_behavior_task_prompt.py"
RUNNER = "scripts/run_behavior_task_prompt.py"
WATCHER = "scripts/watch_behavior_task_prompt.py"
ENTRY = "scripts/start_behavior_task_prompt_once.py"
IMPLEMENTATION = "4e5ab5d4389b2b43ed391edcff0ebebb56d1bf2d6ece51a93a6524db9fc1d580"
BASELINE = 5.8960340268
OLD_PODS = ("luqrjwb9j2igm4", "4ho5jbqxrfcke8", "22bt1cs3t74e93")
EXTENSIONS = (".json", ".jsonl", ".log")
AUTH_KEYS = ("experiment", "protocol_hash", "implementation_hash", "pod_id", "launch_id")


def run(command, *, input=None, timeout=90):
    result = subprocess.run(command, input=input, text=True, encoding="utf-8",
                            capture_output=True, timeout=timeout)
    if result.returncode:
        # Never return full CLI/API/env output in error paths.
        raise RuntimeError("Command failed: " + Path(command[0]).name +
                           "; returncode=" + str(result.returncode))
    return result.stdout


def safe_relative(name):
    path = PurePosixPath(name)
    if (not isinstance(name, str) or not name or "\\" in name or path.is_absolute()
            or ":" in name or any(v in ("..", ".") for v in name.split("/"))):
        raise ValueError("Unsafe archive relative path")
    return path


def image_inventory(plan):
    result = {}
    for dataset in p.DATASETS:
        for split in ("bank", "query"):
            for row in plan["datasets"][dataset]["manifest"][split]:
                name = row["path"]
                safe_relative(name)
                if not name.startswith("datasets/") or name in result:
                    raise ValueError("Duplicate/out-of-scope complete image")
                result[name] = row["image_sha256"]
    if len(result) != 17428:
        raise ValueError("Complete image denominator changed")
    return result


def audit_scope(request):
    if (request.get("new_paper_cloud_execution_authorized") is not True
            or request.get("max_concurrent_gpu_jobs") != 1
            or request.get("local_gpu_execution_authorized") is not False
            or request.get("preserve_all_old_disks_sources_locks_checkpoints_predictions") is not True
            or request.get("first_four_confirmed_dataset_allocation", {}).get("task_prompt_vs_generic_prompt") != list(p.DATASETS)
            or request.get("latest_first_four_priority", [None])[0] != "task_prompt_vs_generic_prompt"
            or request.get("critical_prompt_protocol_hash") != p.PROTOCOL_HASH):
        raise ValueError("Wrong direct-human prompt scope/single cloud GPU authority")
    pod = request.get("current_paper_pod_id")
    if not isinstance(pod, str) or not re.fullmatch(r"[a-z0-9]{12,40}", pod):
        raise ValueError("Missing exact existing Pod authority")
    return pod


def available_host_authority(request):
    audit_scope(request)
    if (request.get("task_prompt_available_host_execution_authorized") is not True
            or request.get("task_prompt_new_Pod_creation_limit") != 1
            or not request.get("task_prompt_latest_execution_human_quote")
            or request.get("task_prompt_new_Pod_creation_consumed") is True):
        raise ValueError("No new direct-human once-only available-host authority")


def choose_gpu(catalog):
    for model in ("5090", "4090", "RTX PRO 4500"):
        for item in catalog:
            if (model in item.get("gpuId", "") and item.get("available") is True
                    and item.get("secureCloud") is True):
                return item
    raise ValueError("No available human-authorized single secure GPU; no creation")


def audit():
    request = p.read(ROOT / ".tools/behavior_priority_experiments_request.json")
    pod = audit_scope(request)
    plan = p.validate_plan(p.read(p.PLAN))
    binding = p.validate_binding(plan, p.read(p.BINDING))
    if binding["implementation_hash"] != IMPLEMENTATION:
        raise ValueError("Frozen scientific implementation changed")
    prior = p.read(ROOT / request["current_paper_deployment_receipt"])
    backup = Path(prior["backup_dir"]).resolve()
    if not backup.is_relative_to(ROOT / "results/cloud_trial") or prior["pod_id"] != pod:
        raise ValueError("Wrong previous exact stopped segment")
    stop = p.read(backup / "stop_confirmation.json")
    takeback = p.read(backup / "full_takeback_receipt.json")
    if (stop.get("pod_id") != pod or stop.get("reason") != "all_complete_verified"
            or not stop.get("disks_preserved") or takeback.get("stage") != "all_complete_verified"
            or not re.fullmatch(r"[0-9a-f]{64}", takeback.get("archive_sha256", ""))
            or abs(request.get("new_paper_GPU_estimate_USD", -1) - BASELINE) > 1e-8):
        raise ValueError("Previous full takeback/stop/accounting not verified")
    return request, plan, binding


def validate_package(package, plan, binding):
    if (package.get("experiment") != p.SCIENCE or package.get("operation") != p.OPERATION
            or package.get("implementation_hash") != IMPLEMENTATION or package.get("protocol_hash") != p.PROTOCOL_HASH
            or package.get("job_hashes") != binding["job_hashes"]
            or package.get("images") != image_inventory(plan)
            or package.get("model_inventory") != plan["model_inventory"]
            or package.get("expected_environment") != plan["expected_environment"]
            or package.get("GPU_execution") is not False or package.get("new_Pod_creation") is not False):
        raise ValueError("Independent operational package scope changed")
    for rel, expected in binding["code_sha256"].items():
        if package["code_sha256"].get(rel) != expected:
            raise ValueError("Scientific identity not preserved by operational package")
    for rel, expected in package["code_sha256"].items():
        safe_relative(rel)
        if p.digest(ROOT / rel) != expected:
            raise ValueError("Frozen operational/source bytes changed: " + rel)


def exact_identity(record, auth):
    if any(not auth.get(key) or record.get(key) != auth[key] for key in AUTH_KEYS):
        raise ValueError("Wrong independent execution identity")


def assess_native(fields, expected, *, empty_gpu=True):
    values = [fields.get(k) for k in ("proc1_pod_id", "export_file_pod_id") if fields.get(k) is not None]
    if (not values or any(v != expected for v in values)
            or fields.get("ssh_process_pod_id") not in (None, expected)
            or (fields.get("native_export_exists") and not fields.get("export_root_owned_nonwritable_regular"))):
        raise ValueError("Actual native Pod proof absent/conflicting/untrusted")
    if empty_gpu and fields.get("GPU_compute_pids"):
        raise ValueError("Existing computation; never parallel GPU")
    device = fields.get("devices", [])
    if len(device) != 1 or not any(s in device[0] for s in ("4090", "5090", "RTX PRO 4500")):
        raise ValueError("Wrong actual authorized single GPU")
    return {"actual_api_pod_id": expected, "proc1_pod_id": fields.get("proc1_pod_id"),
            "export_file_pod_id": fields.get("export_file_pod_id"),
            "ssh_process_pod_id": fields.get("ssh_process_pod_id"),
            "export_root_owned_nonwritable_regular": fields.get("export_root_owned_nonwritable_regular"),
            "device": device[0], "utc": p.now(), "expected_Pod_field_injected": False}


# Standalone bounded native read BEFORE any extraction/import from workspace.
# Full environments are parsed locally on cloud and NEVER returned.
NATIVE_REMOTE = r'''
import json,os,pathlib,re,stat,subprocess
def one(values):
    if len(values)>1: raise ValueError('Duplicate native Pod field')
    if values and not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',values[0]): raise ValueError('Malformed Pod field')
    return values[0] if values else None
proc=one([v.decode('ascii') for item in pathlib.Path('/proc/1/environ').read_bytes().split(b'\0')
          for k,sep,v in [item.partition(b'=')] if k==b'RUNPOD_POD_ID' and sep])
export=pathlib.Path('/etc/rp_environment'); exists=export.exists() or export.is_symlink(); trusted=None; native=None
if exists:
    info=export.lstat();trusted=stat.S_ISREG(info.st_mode) and info.st_uid==0 and not info.st_mode&0o022
    if not trusted: raise ValueError('Untrusted native export')
    values=[]
    for line in export.read_text().splitlines():
        if line.startswith('export RUNPOD_POD_ID='):
            match=re.fullmatch(r'export RUNPOD_POD_ID=["\x27]?([A-Za-z0-9_-]{1,80})["\x27]?',line)
            if not match:raise ValueError('Malformed native export')
            values.append(match.group(1))
    native=one(values)
pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
print(json.dumps({'proc1_pod_id':proc,'export_file_pod_id':native,'native_export_exists':exists,
 'export_root_owned_nonwritable_regular':trusted,'ssh_process_pod_id':os.environ.get('RUNPOD_POD_ID'),
 'GPU_compute_pids':sorted({int(v.strip()) for v in pids.splitlines() if v.strip().isdigit()}),
 'devices':subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip().splitlines()}))
'''


def ssh_for(live, expected):
    endpoint = live.get("ssh") or {}
    if (live.get("id") != expected or live.get("gpuCount") != 1
            or live.get("desiredStatus") != "RUNNING" or live.get("runtimeStatus") != "running"
            or not endpoint.get("ip") or not endpoint.get("port")):
        raise ValueError("No fresh running exact-Pod SSH authority")
    opts = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=accept-new",
            "-i", "C:/Users/ASUS/.runpod/ssh/runpodctl-ssh-key"]
    return (["ssh", *opts, "-p", str(endpoint["port"]), "root@" + endpoint["ip"]],
            ["scp", "-q", *opts, "-P", str(endpoint["port"])], "root@" + endpoint["ip"])


def watcher_identity(rows, auth, backup, expected_pid):
    if isinstance(rows, dict):
        rows = [rows]
    exact = []
    for row in rows:
        cmd = row.get("CommandLine") or ""
        if all(v in cmd for v in (WATCHER, auth["pod_id"], auth["launch_id"],
                                   auth["implementation_hash"], str(backup))):
            exact.append(row)
    pids = {r["ProcessId"] for r in exact}
    roots = [r for r in exact if r["ParentProcessId"] not in pids]
    if len(roots) != 1 or roots[0]["ProcessId"] != expected_pid:
        raise ValueError("CIM exact unique watcher identity not verified")
    return roots


def fee(account, end, capacity=False):
    seconds = 0 if capacity else max(0, (datetime.fromisoformat(end) - datetime.fromisoformat(account["start_request_utc"])).total_seconds())
    return {"GPU_seconds_estimate": seconds, "GPU_cost_estimate_USD": seconds * account["cost_per_hour"] / 3600,
            "new_GPU_segment_started": not capacity, "estimate_not_invoice": True, "storage_included": False}


def process_info(pid):
    proc = Path("/proc") / str(pid)
    args = proc.joinpath("cmdline").read_bytes().split(b"\0")
    # stat comm may contain spaces/parentheses; fields after final ')' start at3.
    tail = proc.joinpath("stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "ppid": int(tail[1]), "start_ticks": int(tail[19]),
            "cwd": str(proc.joinpath("cwd").resolve()), "args": [a.decode() for a in args if a]}


def verify_parent(launch):
    info = process_info(launch["pid"])
    if (info["cwd"] != str(ROOT) or PARENT not in info["args"] or info["start_ticks"] != launch["start_ticks"]):
        raise ValueError("Exact parent command/cwd/start-time changed")
    return info


def science_processes(launch):
    result = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            info = process_info(int(proc.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if info["ppid"] == launch["pid"] and RUNNER in info["args"]:
            if info["cwd"] != str(ROOT):
                raise ValueError("Scientific child wrong cwd")
            result.append(info)
    if len(result) > 1:
        raise ValueError("More than one serial scientific child")
    return result


def stop_bound_processes(launch):
    children = science_processes(launch)
    try:
        parent = verify_parent(launch)
    except (FileNotFoundError, ProcessLookupError):
        parent = None
    bound = children + ([parent] if parent else [])
    for info in bound:
        os.kill(info["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and any(Path("/proc", str(i["pid"])).exists() for i in bound):
        time.sleep(.2)
    for info in bound:
        try:
            current = process_info(info["pid"])
            if current["start_ticks"] == info["start_ticks"]:
                os.kill(info["pid"], signal.SIGKILL)
        except (FileNotFoundError, ProcessLookupError):
            pass


def committed_coverage(science, binding):
    """Only committed source rows or complete newline-delimited predictions.

    Preallocated arrays, source file mtimes, ready/config/log chatter are not progress.
    """
    signature, coverage = {}, {}
    for dataset in p.DATASETS:
        coverage[dataset] = {}
        for arm in p.ARMS:
            source = science / dataset / "sources" / arm / "progress.json"
            if source.exists():
                row = p.read(source)
                if (row.get("job_hash") != binding["job_hashes"][f"{dataset}/{arm}/capture"]
                        or type(row.get("completed")) is not int or not 0 <= row["completed"] <= sum(p.COUNTS[dataset].values())):
                    raise ValueError("Wrong committed source progress identity")
                coverage[dataset][arm + "_source"] = row["completed"]
                if row["completed"]:
                    signature[dataset + "/" + arm + "/source"] = row["completed"]
            prediction = science / dataset / "test" / arm / "predictions.jsonl"
            if prediction.exists():
                count = 0
                with prediction.open(encoding="utf-8") as handle:
                    for line in handle:
                        if not line.endswith("\n"):
                            break  # active last write is not committed; never edit it
                        row = json.loads(line)
                        if row.get("job_hash") != binding["job_hashes"][f"{dataset}/{arm}/generate"]:
                            raise ValueError("Wrong actual official prediction job")
                        count += 1
                if count > p.COUNTS[dataset]["query"]:
                    raise ValueError("Too many official predictions")
                coverage[dataset][arm + "_predictions"] = count
                if count:
                    signature[dataset + "/" + arm + "/predictions"] = count
    return signature, coverage


def safe_members(tar, full):
    allowed = EXTENSIONS + ((".npy",) if full else ())
    prefixes = ("results/" + p.SCIENCE + "/", "results/" + p.OPERATION + "/")
    names = set()
    for member in tar.getmembers():
        path = safe_relative(member.name)
        if (member.name in names or not member.name.startswith(prefixes) or not member.isfile()
                or member.issym() or member.islnk() or path.suffix not in allowed):
            raise ValueError("Unsafe/duplicate/out-of-scope backup member")
        names.add(member.name)
    return tar.getmembers()


def remote_snapshot(auth, previous, force_reason=None):
    if str(ROOT) != "/workspace/BehaviorICL":
        raise ValueError("Cloud snapshot only")
    p.native_identity(auth["pod_id"])
    stored, launch = p.read(OUT / "execution_authorization.json"), p.read(OUT / "launch.json")
    exact_identity(stored, auth)
    exact_identity(launch, auth)
    binding = p.read(p.BINDING)
    if binding["implementation_hash"] != auth["implementation_hash"]:
        raise ValueError("Scientific binding changed")
    value = p.read(OUT / "sequence_progress.json") if (OUT / "sequence_progress.json").exists() else {"stage": "waiting_CPU_parent_not_science"}
    request = OUT / "stop_request.json"
    if request.exists():
        stop = p.read(request)
        exact_identity(stop, auth)
        force_reason = force_reason or stop["reason"]
    fault = None
    try:
        if value.get("stage") not in ("failed", "all_complete_verified"):
            verify_parent(launch)
        children = science_processes(launch)
        active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
        pids = sorted({int(v.strip()) for v in active.splitlines() if v.strip().isdigit()})
        if len(pids) > 1 or any(pid not in {c["pid"] for c in children} for pid in pids):
            raise ValueError("Unrelated/parallel actual GPU computation")
    except (ValueError, FileNotFoundError, ProcessLookupError) as error:
        fault = type(error).__name__ + ": " + str(error)
        children, pids = [], []
    if force_reason or fault or value.get("stage") == "failed":
        stop_bound_processes(launch)
        value = {"stage": "failed", "reason": force_reason or value.get("reason", "parent_failure"),
                 "error": fault or value.get("error"), "utc": p.now()}
        p.atomic(OUT / "sequence_progress.json", value)
    full = value.get("stage") in ("failed", "all_complete_verified")
    if full and science_processes(launch):
        raise ValueError("Scientific child still active before full takeback")
    signature, coverage = committed_coverage(ROOT / "results" / p.SCIENCE, binding)
    paths, signatures = [], {}
    for group in (p.SCIENCE, p.OPERATION):
        for path in (ROOT / "results" / group).rglob("*"):
            if path.is_file() and path.suffix in EXTENSIONS + ((".npy",) if full else ()):
                if path.is_symlink() or not path.resolve().is_relative_to(ROOT / "results" / group):
                    raise ValueError("Linked/out-of-scope scientific backup")
                name = path.relative_to(ROOT).as_posix()
                info = path.stat()
                signatures[name] = [info.st_size, info.st_mtime_ns]
                if full or previous.get(name) != signatures[name]:
                    paths.append(path)
    archive = OUT / ("takeback_" + auth["launch_id"] + ".tar.gz")
    members = []
    with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
        for path in paths:
            name = path.relative_to(ROOT).as_posix()
            if path.suffix == ".npy":
                before = path.stat()
                tar.add(path, arcname=name, recursive=False)
                sha = p.digest(path)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError("Quiescent full source array changed during takeback")
                size = before.st_size
            else:
                # Immutable byte snapshot even if an active JSONL append follows.
                blob = path.read_bytes()
                info = tarfile.TarInfo(name)
                info.size = len(blob)
                info.mode = 0o600
                tar.addfile(info, io.BytesIO(blob))
                size, sha = len(blob), hashlib.sha256(blob).hexdigest()
            members.append({"name": name, "bytes": size, "sha256": sha})
    return {"stage": value, "full": full, "archive": str(archive), "archive_sha256": p.digest(archive),
            "members": members, "signatures": signatures, "science_signature": signature,
            "remote_committed_coverage": coverage, "GPU_compute_pids": pids,
            "actual_children": children, "launch": launch, "utc": p.now()}
