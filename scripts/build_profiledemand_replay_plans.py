#!/usr/bin/env python3
"""Build bounded ProfileDemand GPU replay plans from steady service profiles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


STRATEGIES = {
    "latency": {"max_batch_size": 4, "max_prefill_tokens": 8192},
    "balanced": {"max_batch_size": 8, "max_prefill_tokens": 32768},
    "throughput": {"max_batch_size": 16, "max_prefill_tokens": 65536},
}
INPUT_EDGES = np.asarray([0, 128, 512, 2048, np.inf])
OUTPUT_EDGES = np.asarray([0, 16, 32, 64, np.inf])
REQUESTS_PER_PROFILE = 32
SMOKE_PROFILES = (
    "profile_01_burstgpt_1_c0",
    "profile_08_burstgpt_2_c2",
    "profile_16_mooncake_conversation_c0",
)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profiles",
        type=Path,
        default=root / "experiment-results/phase16_service_profiles/service_profiles.csv",
    )
    parser.add_argument(
        "--requests",
        type=Path,
        default=root
        / "experiment-results/phase16_service_profiles/representative_requests.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase16_profiledemand_plans",
    )
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stratified_indices(requests, count):
    inputs = np.asarray([row["input_len_capped"] for row in requests])
    outputs = np.asarray([row["output_len_capped"] for row in requests])
    lbin = np.minimum(np.searchsorted(INPUT_EDGES, inputs, side="right") - 1, 3)
    mbin = np.minimum(np.searchsorted(OUTPUT_EDGES, outputs, side="right") - 1, 3)
    cells = lbin * 4 + mbin
    cell_counts = np.bincount(cells, minlength=16)
    exact = cell_counts * (count / len(requests))
    allocation = np.floor(exact).astype(int)
    remainder = count - int(np.sum(allocation))
    for cell in np.argsort(-(exact - allocation), kind="stable"):
        if remainder <= 0:
            break
        if cell_counts[cell]:
            allocation[cell] += 1
            remainder -= 1
    selected = []
    for cell, take in enumerate(allocation):
        members = np.flatnonzero(cells == cell)
        if take:
            selected.extend(members[np.linspace(0, len(members) - 1, take, dtype=int)].tolist())
    return sorted(selected)


def microbatches(requests, strategy):
    batches = []
    current = []
    current_tokens = 0
    for row in requests:
        length = int(row["input_len_capped"])
        would_exceed = current and (
            len(current) >= strategy["max_batch_size"]
            or current_tokens + length > strategy["max_prefill_tokens"]
        )
        if would_exceed:
            batches.append(current)
            current, current_tokens = [], 0
        current.append(row)
        current_tokens += length
    if current:
        batches.append(current)
    return batches


def joint_distribution(requests):
    inputs = [row["input_len_capped"] for row in requests]
    outputs = [row["output_len_capped"] for row in requests]
    hist, _, _ = np.histogram2d(inputs, outputs, bins=(INPUT_EDGES, OUTPUT_EDGES))
    return (hist / np.sum(hist)).reshape(-1)


def build_plans(profiles, requests_by_profile, selected_profiles, repeats):
    plans = []
    sampling_audit = []
    for profile_id in selected_profiles:
        source = requests_by_profile[profile_id]
        selected = [source[index] for index in stratified_indices(source, REQUESTS_PER_PROFILE)]
        full_joint = joint_distribution(source)
        selected_joint = joint_distribution(selected)
        sampling_audit.append(
            {
                "profile_id": profile_id,
                "source_requests": len(source),
                "selected_requests": len(selected),
                "joint_l1": float(np.sum(np.abs(full_joint - selected_joint))),
            }
        )
        for strategy_name, strategy in STRATEGIES.items():
            batches = microbatches(selected, strategy)
            for repeat in range(repeats):
                for batch_index, batch in enumerate(batches):
                    plans.append(
                        {
                            "workload_id": f"{profile_id}-{strategy_name}-batch{batch_index:02d}-r{repeat}",
                            "profile_id": profile_id,
                            "source": profiles[profile_id]["source"],
                            "segment": profiles[profile_id]["segment"],
                            "split": profiles[profile_id]["split"],
                            "strategy": strategy_name,
                            "strategy_max_batch_size": strategy["max_batch_size"],
                            "strategy_max_prefill_tokens": strategy["max_prefill_tokens"],
                            "repeat": repeat,
                            "batch_index": batch_index,
                            "profile_requests_replayed": REQUESTS_PER_PROFILE,
                            "trace_replay_mode": "profiledemand_draining_microbatch",
                            "input_lens_per_request": [int(row["input_len_capped"]) for row in batch],
                            "output_lens_per_request": [int(row["output_len_capped"]) for row in batch],
                            "arrival_offsets_ms_audit_only": [int(row["arrival_offset_ms_audit_only"]) for row in batch],
                            "chunk_interaction": "not_crossed_in_profile_grid; controlled chunk data reused from phase14c",
                        }
                    )
    return plans, sampling_audit


def write_jsonl(path, rows):
    with path.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.profiles.open(newline="") as source:
        profiles = {row["profile_id"]: row for row in csv.DictReader(source)}
    requests_by_profile = defaultdict(list)
    for line in args.requests.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            requests_by_profile[row["profile_id"]].append(row)
    for rows in requests_by_profile.values():
        rows.sort(key=lambda row: row["request_index"])

    full_plans, full_audit = build_plans(
        profiles, requests_by_profile, sorted(profiles), repeats=1
    )
    smoke_plans, smoke_audit = build_plans(
        profiles, requests_by_profile, SMOKE_PROFILES, repeats=3
    )
    write_jsonl(args.output_dir / "full_replay_plan.jsonl", full_plans)
    write_jsonl(args.output_dir / "smoke_replay_plan.jsonl", smoke_plans)
    with (args.output_dir / "sampling_audit.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(full_audit[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(full_audit)

    def plan_summary(rows):
        return {
            "workloads": len(rows),
            "profiles": len({row["profile_id"] for row in rows}),
            "repeats": sorted({row["repeat"] for row in rows}),
            "requests_executed": sum(len(row["input_lens_per_request"]) for row in rows),
            "max_batch_size": max(len(row["input_lens_per_request"]) for row in rows),
            "max_prefill_tokens": max(sum(row["input_lens_per_request"]) for row in rows),
            "max_actual_output_len": max(max(row["output_lens_per_request"]) for row in rows),
        }

    summary = {
        "schema_version": "profiledemand-replay-plans-v1",
        "requests_per_profile_per_strategy": REQUESTS_PER_PROFILE,
        "strategies": STRATEGIES,
        "full": plan_summary(full_plans),
        "smoke": plan_summary(smoke_plans),
        "max_profile_sampling_joint_l1": max(row["joint_l1"] for row in full_audit),
        "profiles_sha256": sha256(args.profiles),
        "requests_sha256": sha256(args.requests),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    readme = f"""# Phase 16E：ProfileDemand GPU 回放计划

每个服务画像从 128 个固定代表请求中再分层选择 {REQUESTS_PER_PROFILE} 个请求。三种执行
策略均回放同一组请求，只改变 `max_batch_size` 和 `max_prefill_tokens`，按请求原顺序形成
draining microbatches。完整计划包含 {summary['full']['profiles']} 个画像、
{summary['full']['workloads']} 个 GPU workloads；smoke 使用 3 个画像和 3 次重复，共
{summary['smoke']['workloads']} 个 workloads。

为保证 Qwen3-8B 的 Prefill logical payload 不超过已实测 L1 曲线 512 MiB，所有策略的
单 batch token budget 不超过 65,536。当前画像网格不与 heterogeneous chunked prefill
做全交叉；chunk 机理使用 Phase14C 的 108 个受控配置，避免把 one_batch 尚不支持的
异长请求 chunk 回放伪装成真实 online 调度。

到达 offset 被保留审计，但当前仍是离线 draining microbatch，不声称已完成 online
continuous batching。
"""
    (args.output_dir / "README.md").write_text(readme)
    checks = {
        "full_profiles_24": summary["full"]["profiles"] == 24,
        "smoke_profiles_3": summary["smoke"]["profiles"] == 3,
        "same_32_requests_per_profile_strategy": all(
            sum(len(row["input_lens_per_request"]) for row in full_plans if row["profile_id"] == profile and row["strategy"] == strategy)
            == REQUESTS_PER_PROFILE
            for profile in profiles
            for strategy in STRATEGIES
        ),
        "prefill_token_budget_at_most_65536": summary["full"]["max_prefill_tokens"] <= 65536,
        "actual_output_len_at_most_128": summary["full"]["max_actual_output_len"] <= 128,
        "sampling_joint_l1_below_0_30": summary["max_profile_sampling_joint_l1"] < 0.30,
    }
    audit = {
        "schema_version": "profiledemand-replay-plans-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    (args.output_dir / "audit_summary.json").write_text(json.dumps(audit, indent=2) + "\n")
    if audit["status"] != "PASS":
        raise RuntimeError(audit)
    (args.output_dir / "DONE").write_text("PASS\n")
    (args.output_dir / "run.log").write_text(json.dumps({"checks": checks}, indent=2) + "\n")
    files = sorted(
        path for path in args.output_dir.iterdir() if path.is_file() and path.name != "manifest.sha256"
    )
    (args.output_dir / "manifest.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files)
    )
    print(json.dumps({"summary": summary, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
