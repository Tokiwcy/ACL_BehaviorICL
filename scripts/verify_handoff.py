"""Verify the portable handoff offline, not a full scientific/model replay."""
from __future__ import annotations

import ast
import csv
import io
import json
import math
import re
from collections import Counter
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit

from handoff_artifact_utils import ROOT, manifest, repository_path, sha256, verified_payload

OFFICIAL = {"dtd": 1880, "aircraft": 3333, "cub": 5794, "pets": 3669, "dogs": 8580}
SECRET_PATTERNS = [
    rb"\bgh[oprsu]_[A-Za-z0-9]{20,}\b", rb"\bgithub_pat_[A-Za-z0-9_]{30,}\b",
    rb"\bhf_[A-Za-z0-9]{24,}\b", rb"\bAKIA[A-Z0-9]{16}\b",
    rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: str) -> dict:
    return json.loads(repository_path(path).read_text(encoding="utf-8-sig"))


def rows(path: str) -> list[dict]:
    with repository_path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def near(actual: float, expected: float, message: str) -> None:
    require(math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-9), message)


def check_paired(table: list[dict], first: str, second: str) -> None:
    for row in table:
        n = int(row["queries"])
        require(n > 0, "Empty paired denominator")
        near(row["difference_pp"], 100 * (int(row[first]) - int(row[second])) / n, "Paired count/delta mismatch")
        require(int(row["bootstrap_repetitions"]) == 10000, "Unexpected CI repetition count")
        require(float(row["paired_bootstrap_low_pp"]) <= float(row["paired_bootstrap_high_pp"]), "Reversed CI")


def check_results() -> dict:
    facts = read_json("analysis/current_project_consolidation_v1/summary_facts.json")
    main = rows("analysis/paper_p1_recheck_v1/full_test_counts.csv")
    require(len(main) == 30, "Expected 30 seed73 method/dataset groups")
    require(len({(row["dataset"], row["method"]) for row in main}) == 30, "Duplicate main groups")
    for row in main:
        dataset, method = row["dataset"], row["method"]
        require(int(row["total"]) == OFFICIAL[dataset], "Wrong official test denominator")
        near(row["accuracy_percent"], 100 * int(row["correct"]) / int(row["total"]), "Main accuracy/count mismatch")
        near(row["accuracy_percent"], facts["main_seed73"][method]["accuracy_percent"][dataset], "Summary/main mismatch")
    for method, result in facts["main_seed73"].items():
        selected = [float(row["accuracy_percent"]) for row in main if row["method"] == method]
        require(len(selected) == 5 and result["seed"] == 73 and not result["three_seed_complete"], "Seed73 scope mismatch")
        near(sum(selected) / 5, result["equal_dataset_mean_percent"], "Macro mean mismatch")
    require(sum(OFFICIAL.values()) == facts["main_query_count"] == 23256, "Main query count mismatch")
    task = rows("analysis/paper_task_prompt_complete_results_v1/paired_task_generic.csv")
    require(len(task) == 2 and {r["dataset"] for r in task} == {"dtd", "cub"}, "Task dataset scope mismatch")
    check_paired(task, "task_correct", "generic_correct")
    require(task == facts["task_prompt_paired"], "Task summary copied inconsistently")
    arms = rows("analysis/paper_full_test_labels_complete_results_v1/arm_summary.csv")
    require(len(arms) == 8, "Expected two datasets/four label arms")
    require(sum(int(r["planned_queries"]) for r in arms) == 30696, "Full label planned count mismatch")
    require(sum(int(r["generated_queries"]) for r in arms) == 30526, "Full label generated count mismatch")
    require(sum(int(r["explicit_missing"]) for r in arms) == 170, "Missing own-label rows lost")
    for row in arms:
        require(int(row["generated_queries"]) + int(row["explicit_missing"]) == int(row["planned_queries"]) == OFFICIAL[row["dataset"]], "Label cohort mismatch")
    labels = [r for r in rows("analysis/paper_full_test_labels_complete_results_v1/paired_label_effects.csv") if r["first_arm"] == "gold_demo_labels" and r["second_arm"] == "demo_own_legal_zero_labels" and r["scope"] == "all_available_pairs" and r["metric"] == "accuracy"]
    require(labels == facts["full_label_gold_own_available_pairs"], "Full label paired summary mismatch")
    require({r["dataset"]: int(r["queries"]) for r in labels} == {"dtd": 1809, "cub": 5695}, "Missingness-adjusted pairs mismatch")
    check_paired(labels, "first_positive", "second_positive")
    sample = rows("analysis/paired_label_sampled_results_v1/arm_summary.csv")
    require(len(sample) == 20 and all(int(r["planned_queries"]) == 256 for r in sample), "Sampled labels confused with full test")
    qv = rows("analysis/paper_qwenvision_two_complete_results_v1/full_official_accuracy.csv")
    require(qv == facts["qwenvision_accuracy"] and len(qv) == 2, "QV summary mismatch")
    for row in qv:
        near(row["accuracy_percent"], 100 * int(row["correct"]) / int(row["total"]), "QV accuracy/count mismatch")
    qv_pairs = rows("analysis/paper_qwenvision_two_complete_results_v1/paired_accuracy_CI.csv")
    require(qv_pairs == facts["qwenvision_paired"], "QV paired summary mismatch")
    check_paired(qv_pairs, "first_rescued", "second_rescued")
    match = rows("analysis/paper_qwenvision_match_control_v1/paired_differences.csv")
    require(match == facts["qwenvision_match_paired"], "Match summary mismatch")
    check_paired(match, "first_positive", "second_positive")
    repair = rows("analysis/paper_repair_five_dataset_v1/all_error_repair_summary.csv")
    require(len(repair) == 30, "Repair group count mismatch")
    for row in repair:
        require(int(row["legal_zero_errors"]) + int(row["illegal_zero_errors"]) == int(row["zero_errors"]), "Illegal errors omitted")
        require(sum(int(row[k]) for k in ("strict_present_queries", "strict_absent_queries", "strict_unknown_queries")) == int(row["zero_errors"]), "Unknown match status omitted")
        require(int(row["zero_correct"]) + int(row["rescued"]) - int(row["harmed"]) == int(row["icl_correct"]), "Repair accounting mismatch")
    return {"seed73_groups": 30, "official_queries": 23256, "full_label_records": 30696, "explicit_missing": 170}


def check_tam_and_pause() -> None:
    protocol = read_json("analysis/query_tam_pilot_v1/protocol.json")
    require(protocol["model"] == "Qwen/Qwen3-VL-4B-Instruct", "Wrong TAM target model")
    require(protocol["revision"] == "ebb281ec70b05090aa6165b016eac8ec08e71b17", "Wrong fixed model revision")
    examples = protocol["examples"]
    require(len(examples) == 8 and len({r["bundle_path"] for r in examples}) == 8, "TAM example count mismatch")
    for dataset in ("dtd", "cub"):
        require(Counter(r["historical_group"] for r in examples if r["dataset"] == dataset) == Counter({"task_only": 1, "generic_only": 1, "both_correct": 1, "both_wrong": 1}), "TAM strata mismatch")
    require(all(set(r["queries"]) == {"task", "generic", "focus_a", "focus_b"} for r in examples), "TAM query count mismatch")
    require(sum(len(r["queries"]) * 2 for r in examples) == 64, "TAM run scope mismatch")
    upstream = read_json("handoff/tam_upstream_receipt.json")
    require(protocol["TAM"]["commit"] == upstream["commit"] and protocol["TAM"]["SHA"] == upstream["SHA"], "TAM upstream binding mismatch")
    pause = read_json("analysis/current_project_consolidation_v1/pause_receipt.json")
    require(pause["automation_status"] == "PAUSED" and pause["live_GPU_count"] == 0 and not pause["science_auto_resume_authorized_now"], "Pause snapshot mismatch")
    require(pause["TAM_status"] == "stopped_before_native_or_worker_execution_no_scientific_results", "TAM must not be reported complete")


def check_query_tables() -> dict:
    entries = {entry["source_path"]: entry for entry in manifest()["files"]}
    for folder, expected_rows, expected_groups in (
        ("paper_repair_five_dataset_v1", 139536, 30),
        ("paper_qwenvision_match_control_v1", 53718, 14),
    ):
        path = f"analysis/{folder}/query_level_results.csv"
        identity = set()
        totals = {}
        # Read the verified compressed payload even on a clone without materialized files.
        with io.TextIOWrapper(io.BytesIO(verified_payload(entries[path])), encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                key = (row["dataset"], row["method"], row["query_id"])
                require(key not in identity, "Duplicate query/method identity")
                identity.add(key)
                count = totals.setdefault(key[:2], Counter())
                count["test_queries"] += 1
                zero_correct = row["zero_shot_correct"] == "True"
                initial_error = row["initial_error"] == "True"
                require(initial_error != zero_correct, "Initial error flag mismatch")
                count["zero_correct"] += zero_correct
                count["zero_errors"] += initial_error
                count["illegal_zero_errors"] += initial_error and row["zero_shot_valid"] != "True"
                count["icl_correct"] += row["icl_correct"] == "True"
                count["rescued"] += row["rescued"] == "True"
                count["harmed"] += row["harmed"] == "True"
                if initial_error:
                    status = row["strict_status"]
                    require(status in {"present", "absent", "unknown"}, "Unknown strict status encoding")
                    count[f"strict_{status}_queries"] += 1
        require(len(identity) == expected_rows and len(totals) == expected_groups, "Complete query table scope mismatch")
        for row in rows(f"analysis/{folder}/all_error_repair_summary.csv"):
            for metric, value in totals[(row["dataset"], row["method"])].items():
                require(int(row[metric]) == value, f"Query table/summary count mismatch: {folder}/{metric}")
    for folder, expected_rows in (("paper_full_test_labels_complete_results_v1", 30696), ("paired_label_sampled_results_v1", 5120)):
        identity = set()
        totals = {}
        with repository_path(f"analysis/{folder}/query_arm_outcomes.csv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                key = (row["dataset"], row["query_id"], row["arm"])
                require(key not in identity, "Duplicate label query/arm identity")
                identity.add(key)
                count = totals.setdefault((key[0], key[2]), Counter())
                count["planned_queries"] += 1
                generated = row["status"] == "generated"
                require(generated or row["status"] == "missing_illegal_demo_zero", "Unexpected label generation status")
                count["generated_queries"] += generated
                count["explicit_missing"] += not generated
                count["correct"] += row["correct"] == "True"
                require(generated or row["correct"] == "", "Missing label row has invented result")
        require(len(identity) == expected_rows, "Label query table row count mismatch")
        for row in rows(f"analysis/{folder}/arm_summary.csv"):
            for metric, value in totals[(row["dataset"], row["arm"])].items():
                require(int(row[metric]) == value, f"Label table/summary mismatch: {metric}")
    return {"retrospective_query_method_rows": 139536, "QV_match_query_method_rows": 53718, "sampled_label_query_arm_rows": 5120}


def check_links() -> int:
    count = 0
    documents = sorted(ROOT.glob("*.md")) + sorted((ROOT / "handoff").glob("*.md"))
    for document in documents:
        text = document.read_text(encoding="utf-8-sig")
        for target in re.findall(r"\]\((<[^>]+>|[^)]+)\)", text):
            target = target.strip("<>")
            if target.startswith(("https://", "http://", "mailto:", "#")):
                continue
            require(not re.match(r"(?:[A-Za-z]:[/\\]|/|file:)", target), f"Machine-specific Markdown link: {document.name}")
            relative = unquote(urlsplit(target).path)
            resolved = (document.parent / relative).resolve()
            require(resolved.is_relative_to(ROOT.resolve()) and resolved.exists(), f"Broken Markdown link in {document.relative_to(ROOT)}: {target}")
            count += 1
    return count


def main() -> None:
    report = manifest()
    require(report["cloud_experiments_paused"] and not report["new_GPU_operations"] and not report["scientific_source_bytes_modified"], "Publication scope changed")
    originals = compressed = materialized = 0
    for entry in report["files"]:
        relative = PurePosixPath(entry["path"])
        require(relative.parts[0] not in {".tools", "datasets", "results", ".hf_cache"}, "Private runtime included")
        require(relative.suffix not in {".pt", ".pth", ".npy", ".npz", ".safetensors", ".pem", ".key"}, "Weight/private artifact included")
        payload = verified_payload(entry)
        require(not any(re.search(pattern, payload) for pattern in SECRET_PATTERNS), f"Credential-like payload detected: {entry['path']}")
        originals += 1
        if entry["kind"] == "gzip_exact":
            compressed += 1
            target = repository_path(entry["materialize_path"])
            if target.exists():
                require(target.stat().st_size == entry["source_bytes"] and sha256(target.read_bytes()) == entry["source_sha256"], "Materialized artifact changed")
                materialized += 1
    index = read_json("analysis/current_project_consolidation_v1/current_code_dependency_index.json")
    for parent, children in index["static_local_import_edges"].items():
        require(repository_path(parent).is_file(), "Missing scientific import parent")
        for child in children:
            require(repository_path(child).is_file(), f"Missing local import dependency: {child}")
    python_files = sorted((ROOT / "scripts").glob("*.py")) + sorted((ROOT / "tests").glob("*.py"))
    for source in python_files:
        ast.parse(source.read_bytes(), filename=str(source.relative_to(ROOT)))
    results = {**check_results(), **check_query_tables()}
    check_tam_and_pause()
    links = check_links()
    # New handoff prose/code is also checked, not only immutable scientific files.
    for source in python_files + sorted(ROOT.glob("*.md")) + sorted((ROOT / "handoff").glob("*.md")):
        require(not any(re.search(pattern, source.read_bytes()) for pattern in SECRET_PATTERNS), f"Credential-like content detected: {source.relative_to(ROOT)}")
    print(json.dumps({"status": "portable_handoff_verified", "exact_or_packed_artifacts": originals, "gzip_exact_artifacts": compressed, "materialized_exact_artifacts": materialized, "python_AST_files": len(python_files), "relative_document_links": links, **results, "TAM_planned_runs": 64, "TAM_actual_results": 0, "cloud_paused": True, "scientific_model_replay": False}, indent=2))


if __name__ == "__main__":
    main()
