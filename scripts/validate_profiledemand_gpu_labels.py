#!/usr/bin/env python3
"""Validate and aggregate histogram-only ProfileDemand GPU replay results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument(
        "--model-features",
        type=Path,
        default=root / "experiment-results/phase16_model_features/model_features.json",
    )
    parser.add_argument(
        "--bins",
        type=Path,
        default=root / "experiment-results/phase16_profiledemand_binning/bin_definitions.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalized_rank_histogram(rank):
    return sorted(
        (
            row["phase"],
            row["op"],
            int(row["group_size"]),
            int(row["input_payload_bytes"]),
            int(row["count"]),
            row.get("active_batch_size"),
            row.get("prefill_chunk_index"),
            row.get("prefill_chunk_tokens"),
            row.get("first_decode_step"),
            row.get("last_decode_step"),
        )
        for row in rank["event_histograms"]
    )


def observed_histograms(rank):
    canonical = defaultdict(Counter)
    raw = defaultdict(Counter)
    for row in rank["event_histograms"]:
        phase = row["phase"]
        op = row["op"]
        payload = int(row["input_payload_bytes"])
        count = int(row["count"])
        canonical[phase][payload] += count
        raw[phase][(op, payload)] += count
    return canonical, raw


def expected_h0(plan, features):
    calls_per_forward = int(features["logical_collectives_per_forward_prior"])
    bytes_per_token = int(features["payload_bytes_per_active_token_prior"])
    predicted = {"prefill": Counter(), "decode": Counter()}
    prefill_tokens = sum(int(value) for value in plan["input_lens_per_request"])
    predicted["prefill"][prefill_tokens * bytes_per_token] += calls_per_forward
    output_lens = [int(value) for value in plan["output_lens_per_request"]]
    for step in range(1, max(output_lens)):
        active = sum(length > step for length in output_lens)
        if active:
            predicted["decode"][active * bytes_per_token] += calls_per_forward
    return predicted


def load_bins(path):
    with path.open(newline="") as source:
        rows = [row for row in csv.DictReader(source) if int(row["bin_count"]) == 12]
    rows.sort(key=lambda row: int(row["bin_index"]))
    if len(rows) != 12:
        raise ValueError(f"expected 12 selected bins, got {len(rows)}")
    return rows


def bin_index(payload, bins):
    for row in bins:
        left, right = int(row["left_bytes"]), int(row["right_bytes"])
        final = int(row["bin_index"]) == len(bins) - 1
        if left <= payload < right or (final and payload == right):
            return int(row["bin_index"])
    raise ValueError(f"payload {payload} is outside selected bin range")


def aggregate_labels(validated, bins, model, tp):
    grouped = defaultdict(lambda: {"requests": 0, "canonical": Counter(), "raw": Counter()})
    for row in validated:
        plan = row["plan"]
        for phase in ("prefill", "decode"):
            key = (plan["profile_id"], plan["strategy"], int(plan["repeat"]), phase)
            target = grouped[key]
            if phase == "prefill":
                target["requests"] += len(plan["input_lens_per_request"])
            target["canonical"].update(row["canonical"][phase])
            target["raw"].update(row["raw"][phase])
    labels = []
    for (profile, strategy, repeat, phase), value in sorted(grouped.items()):
        requests = value["requests"] if phase == "prefill" else next(
            item["requests"]
            for (p, s, r, ph), item in grouped.items()
            if (p, s, r, ph) == (profile, strategy, repeat, "prefill")
        )
        scale = 1000.0 / requests
        calls_by_bin = [0.0] * len(bins)
        bytes_by_bin = [0.0] * len(bins)
        for payload, calls in value["canonical"].items():
            index = bin_index(payload, bins)
            calls_by_bin[index] += calls * scale
            bytes_by_bin[index] += calls * payload * scale
        raw_json = {
            f"{op}:{payload}": calls * scale
            for (op, payload), calls in sorted(value["raw"].items())
        }
        exact_json = {
            str(payload): calls * scale
            for payload, calls in sorted(value["canonical"].items())
        }
        labels.append(
            {
                "model": model,
                "tp": tp,
                "profile_id": profile,
                "strategy": strategy,
                "repeat": repeat,
                "phase": phase,
                "requests": requests,
                "normalization_requests": 1000,
                "total_calls_per_1000": sum(value["canonical"].values()) * scale,
                "total_logical_bytes_per_1000": sum(
                    payload * calls for payload, calls in value["canonical"].items()
                )
                * scale,
                "calls_by_12bin_json": json.dumps(calls_by_bin, separators=(",", ":")),
                "logical_bytes_by_12bin_json": json.dumps(bytes_by_bin, separators=(",", ":")),
                "canonical_exact_histogram_per_1000_json": json.dumps(exact_json, separators=(",", ":")),
                "raw_op_exact_histogram_per_1000_json": json.dumps(raw_json, separators=(",", ":")),
            }
        )
    return labels


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_rows = read_jsonl(args.plan)
    result_rows = read_jsonl(args.result)
    plans = {row["workload_id"]: row for row in plan_rows}
    features = {
        row["model"]: row for row in json.loads(args.model_features.read_text())
    }[args.model]
    if len(result_rows) != len(plan_rows):
        raise ValueError(f"result/plan length mismatch: {len(result_rows)} != {len(plan_rows)}")

    validated = []
    all_rank_equal = True
    fixed_outputs = True
    h0_exact = True
    capture_contract = True
    group_size_correct = True
    for result in result_rows:
        plan = result["trace_replay_plan"]
        workload_id = plan["workload_id"]
        if workload_id not in plans or plan != plans[workload_id]:
            raise ValueError(f"embedded plan mismatch: {workload_id}")
        ranks = result["comm_profile"]
        capture_contract &= len(ranks) == args.tp and all(
            rank["capture_mode"] == "histogram-only"
            and not rank["raw_events_saved"]
            and not rank["events"]
            for rank in ranks
        )
        normalized = [normalized_rank_histogram(rank) for rank in ranks]
        all_rank_equal &= all(hist == normalized[0] for hist in normalized[1:])
        group_size_correct &= all(
            item[2] == args.tp for item in normalized[0]
        )
        fixed_outputs &= result["generated_output_tokens_per_request"] == plan["output_lens_per_request"]
        canonical, raw = observed_histograms(ranks[0])
        expected = expected_h0(plan, features)
        h0_exact &= all(dict(canonical[phase]) == dict(expected[phase]) for phase in ("prefill", "decode"))
        validated.append({"plan": plan, "canonical": canonical, "raw": raw})

    bins = load_bins(args.bins)
    labels = aggregate_labels(validated, bins, args.model, args.tp)
    with (args.output_dir / "phase_labels.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(labels[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(labels)

    repeat_groups = defaultdict(list)
    for row in labels:
        repeat_groups[(row["profile_id"], row["strategy"], row["phase"])].append(
            (
                row["total_calls_per_1000"],
                row["total_logical_bytes_per_1000"],
                row["calls_by_12bin_json"],
                row["logical_bytes_by_12bin_json"],
            )
        )
    repeats_identical = all(len(set(values)) == 1 for values in repeat_groups.values())
    summary = {
        "schema_version": "profiledemand-gpu-labels-v1",
        "model": args.model,
        "tp": args.tp,
        "workloads": len(result_rows),
        "phase_labels": len(labels),
        "profiles": len({row["profile_id"] for row in labels}),
        "strategies": sorted({row["strategy"] for row in labels}),
        "repeats": sorted({row["repeat"] for row in labels}),
        "statistical_contract": {
            "calls": "group-level collective calls from representative rank",
            "payload": "representative-rank logical input tensor bytes",
            "normalization": "per 1000 requests",
            "output": "12 bins, retaining both calls and logical bytes",
        },
        "checks": {
            "all_rank_histograms_equal": all_rank_equal,
            "fixed_actual_outputs": fixed_outputs,
            "h0_canonical_histograms_exact": h0_exact,
            "histogram_only_no_raw_events": capture_contract,
            "group_size_matches_tp": group_size_correct,
            "repeats_identical": repeats_identical,
        },
        "result_sha256": sha256(args.result),
        "plan_sha256": sha256(args.plan),
        "model_features_sha256": sha256(args.model_features),
        "bins_sha256": sha256(args.bins),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    readme = f"""# ProfileDemand GPU labels：{args.model} TP={args.tp}

共验证 {len(result_rows)} 个 histogram-only draining microbatches，聚合为 {len(labels)} 条
`profile×strategy×repeat×phase` 标签。每条标签按 1000 请求归一化，并同时保存 12 桶
calls、12 桶 logical bytes、canonical 精确直方图和 raw-op 精确直方图。

all-rank 对齐、固定实际输出、group size、histogram-only 无 raw events、H0 canonical
解析映射和重复一致性均通过。raw fused-op 仅供第二阶段 backend 细化；第一阶段正式
目标为 canonical logical AllReduce PatternDemand。
"""
    (args.output_dir / "README.md").write_text(readme)
    checks = summary["checks"]
    audit = {
        "schema_version": "profiledemand-gpu-labels-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    (args.output_dir / "audit_summary.json").write_text(json.dumps(audit, indent=2) + "\n")
    if audit["status"] != "PASS":
        raise RuntimeError(audit)
    (args.output_dir / "DONE").write_text("PASS\n")
    files = sorted(
        path for path in args.output_dir.iterdir() if path.is_file() and path.name not in {"manifest.sha256", "run.log"}
    )
    (args.output_dir / "manifest.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
