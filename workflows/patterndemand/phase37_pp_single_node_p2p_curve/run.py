#!/usr/bin/env python3
"""一条命令完成Phase37拓扑选择、测量、有限方差补测和归档。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import append_jsonl, environment_record, load_json, repo_root, utc_now, write_json
from finalize import finalize, read_records
from preflight import normalize_link, run_checks


def gpu_inventory() -> list[dict]:
    query = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"
    ], text=True)
    rows = []
    for line in query.splitlines():
        if not line.strip():
            continue
        index, uuid, name, total, used, utilization = [value.strip() for value in line.split(",", 5)]
        rows.append({"index": int(index), "uuid": uuid, "name": name, "memory_total_mib": int(total), "memory_used_mib": int(used), "utilization_percent": int(utilization)})
    return rows


def resolve_pairs(preflight: dict, override_path: Path | None, reason: str | None, decision_log: Path) -> dict:
    selected = dict(preflight["default_selected_pairs"])
    if override_path is None:
        append_jsonl(decision_log, {"at_utc": utc_now(), "level": "AUTO", "decision": "使用每类拓扑字典序最小GPU对", "selected_pairs": selected})
        return selected
    if not reason:
        raise RuntimeError("使用GPU pair override时必须提供--override-reason")
    overrides = load_json(override_path)
    all_pairs = preflight["topology"]["pairs_by_category"]
    for category, gpus in overrides.items():
        normalized = sorted(int(value) for value in gpus)
        matches = [row for row in all_pairs.get(category, []) if sorted(row["gpus"]) == normalized]
        if not matches:
            raise RuntimeError(f"override GPU对不属于声明的拓扑类别：{category}={gpus}")
        selected[category] = matches[0]
    append_jsonl(decision_log, {"at_utc": utc_now(), "level": "RECORD_AND_CONTINUE", "decision": "同拓扑类别GPU对替换", "reason": reason, "selected_pairs": selected})
    return selected


def run_one(contract: dict, category: str, pair: dict, repeat_id: int, raw_path: Path, log_path: Path, decision_log: Path) -> None:
    command = [
        shutil.which("torchrun") or "torchrun", "--standalone", "--nnodes=1", "--nproc-per-node=2",
        str(HERE / "benchmark_p2p.py"), "--output", str(raw_path), "--repeat-id", str(repeat_id),
        "--topology-category", category, "--raw-link", pair["raw_link"],
        "--physical-gpus", ",".join(str(value) for value in pair["gpus"]),
        "--warmup", str(contract["warmup_iterations"]), "--iterations", str(contract["timed_iterations"]),
        "--payload-bytes", *[str(value) for value in contract["payload_bytes"]],
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in pair["gpus"])
    env["PYTHONPATH"] = str(repo_root() / "python") + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    attempts = int(contract["retry_limit_per_process_run"]) + 1
    for attempt in range(attempts):
        attempt_raw = raw_path.with_name(raw_path.name + f".attempt{attempt + 1}.partial")
        attempt_command = list(command)
        attempt_command[attempt_command.index(str(raw_path))] = str(attempt_raw)
        completed = subprocess.run(attempt_command, cwd=repo_root(), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        with log_path.open("a", encoding="utf-8") as target:
            target.write(f"attempt={attempt + 1} returncode={completed.returncode}\n{completed.stdout}\n")
        if completed.returncode == 0:
            attempt_raw.replace(raw_path)
            return
        append_jsonl(decision_log, {"at_utc": utc_now(), "level": "AUTO" if attempt + 1 < attempts else "BLOCKED", "decision": "P2P测量进程失败", "category": category, "pair": pair, "repeat_id": repeat_id, "attempt": attempt + 1, "returncode": completed.returncode})
    raise RuntimeError(f"P2P测量失败：category={category}, repeat={repeat_id}; 见{log_path}")


def maximum_repeat_cv(raw_dir: Path) -> float:
    by_point_repeat = defaultdict(list)
    for record in read_records(raw_dir):
        by_point_repeat[(record["topology_category"], int(record["payload_bytes"]), int(record["repeat_id"]))].append(float(record["latency_us"]["median"]))
    grouped = defaultdict(list)
    for (category, payload, _repeat), direction_medians in by_point_repeat.items():
        if len(direction_medians) == 2:
            grouped[(category, payload)].append(max(direction_medians))
    values = []
    for medians in grouped.values():
        if len(medians) > 1 and statistics.fmean(medians):
            values.append(statistics.pstdev(medians) / statistics.fmean(medians))
    return max(values, default=0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-workflow-commit", required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--pair-overrides", type=Path)
    parser.add_argument("--override-reason")
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase37_pp_single_node_p2p_curve")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    raw_dir = args.raw_dir.expanduser().resolve()
    preflight = run_checks(args.expected_workflow_commit, output, raw_dir)
    contract = load_json(HERE / "experiment.json")
    for name in ("audit", "analysis", "curves", "figures", "logs", "contracts"):
        (output / name).mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=False)
    decision_log = output / "logs/decision_log.jsonl"
    selected = resolve_pairs(preflight, args.pair_overrides, args.override_reason, decision_log)
    write_json(output / "contracts/experiment.json", contract)
    (output / "audit/nvidia_smi_topo_m.txt").write_text(preflight["topology"]["raw"], encoding="utf-8")
    inventory_before = gpu_inventory()
    write_json(output / "audit/environment_before.json", {**environment_record(), "torch": preflight["torch"], "cuda": preflight["cuda"], "gpu_inventory": inventory_before})

    repeat_target = int(contract["minimum_independent_repeats"])
    completed_repeats = 0
    while completed_repeats < repeat_target:
        repeat_id = completed_repeats
        for category, pair in sorted(selected.items()):
            raw_path = raw_dir / f"{category.lower()}_gpu{pair['gpus'][0]}_gpu{pair['gpus'][1]}_repeat{repeat_id}.jsonl"
            log_path = output / "logs" / f"{category.lower()}_repeat{repeat_id}.log"
            run_one(contract, category, pair, repeat_id, raw_path, log_path, decision_log)
        completed_repeats += 1
        if completed_repeats == repeat_target:
            current_cv = maximum_repeat_cv(raw_dir)
            threshold = float(contract["repeat_median_cv_threshold"])
            maximum = int(contract["maximum_independent_repeats"])
            if current_cv > threshold and repeat_target < maximum:
                previous_target = repeat_target
                repeat_target = min(repeat_target + int(contract["variance_extra_repeats_per_round"]), maximum)
                append_jsonl(decision_log, {"at_utc": utc_now(), "level": "AUTO", "decision": "repeat中位数CV超过阈值，按合同追加独立重复", "max_cv": current_cv, "threshold": threshold, "previous_repeat_target": previous_target, "new_repeat_target": repeat_target})

    inventory_after = gpu_inventory()
    raw_bundle_id = f"phase37-{preflight['workflow_commit'][:12]}-{utc_now().replace(':', '').replace('+00:00', 'Z')}"
    runtime_state = {
        "workflow_commit": preflight["workflow_commit"],
        "input_audit": preflight["pinned_inputs"],
        "source_semantics": preflight["source_semantics"],
        "selected_pairs": selected,
        "repeat_count": completed_repeats,
        "maximum_repeat_cv_before_finalize": maximum_repeat_cv(raw_dir),
        "raw_samples_external_to_git": repo_root().resolve() not in raw_dir.parents and raw_dir != repo_root().resolve(),
        "raw_bundle_id": raw_bundle_id,
        "gpu_inventory": inventory_before,
    }
    write_json(output / "audit/runtime_state.json", runtime_state)
    write_json(output / "audit/environment_after.json", {**environment_record(), "gpu_inventory": inventory_after})
    summary = finalize(output, raw_dir)
    print(json.dumps({"status": summary["status"], "output": str(output), "raw_bundle_id": raw_bundle_id, "topology_categories": sorted(selected), "repeats": completed_repeats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
