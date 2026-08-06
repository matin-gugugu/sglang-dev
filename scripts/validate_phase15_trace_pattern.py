#!/usr/bin/env python3
"""Validate trace-derived heterogeneous draining-batch PatternDemand results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalized_profile(profile):
    return {
        "capture_mode": profile["capture_mode"],
        "raw_events_saved": profile["raw_events_saved"],
        "stats": profile["stats"],
        "events": profile["events"],
        "event_histograms": profile["event_histograms"],
        "events_total": profile["events_total"],
        "events_truncated": profile["events_truncated"],
    }


def aggregate_phase(histograms, phase, tp):
    by_op_payload = defaultdict(int)
    calls = 0
    logical_payload = 0
    for event in histograms:
        if event["phase"] != phase:
            continue
        if int(event["group_size"]) != tp:
            raise AssertionError(f"unexpected group size: {event}")
        count = int(event["count"])
        payload = int(event["input_payload_bytes"])
        calls += count
        logical_payload += count * payload
        by_op_payload[f"{event['op']}:{payload}"] += count
    alpha = 2 * (tp - 1) / tp
    beta = 2 * (tp - 1)
    return {
        "calls": calls,
        "logical_payload_bytes": logical_payload,
        "equivalent_bytes": alpha * logical_payload,
        "equivalent_rounds": beta * calls,
        "calls_by_op_payload_json": json.dumps(
            dict(sorted(by_op_payload.items())), separators=(",", ":")
        ),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plans = load_jsonl(args.plan)
    results = load_jsonl(args.result)
    plan_by_id = {row["workload_id"]: row for row in plans}
    if len(plan_by_id) != len(plans):
        raise AssertionError("duplicate workload_id in plan")
    if len(results) != len(plans):
        raise AssertionError(f"result rows {len(results)} != plan rows {len(plans)}")

    labels = []
    raw_ops = set()
    for result in results:
        plan = result.get("trace_replay_plan")
        if not plan or plan["workload_id"] not in plan_by_id:
            raise AssertionError("result is missing a known trace_replay_plan")
        expected = plan_by_id[plan["workload_id"]]
        if plan != expected:
            raise AssertionError(f"embedded plan changed for {plan['workload_id']}")
        if result["input_lens_per_request"] != expected["input_lens_per_request"]:
            raise AssertionError("input length vector changed")
        if result["output_lens_per_request"] != expected["output_lens_per_request"]:
            raise AssertionError("output length vector changed")
        if result["generated_output_tokens_per_request"] != expected["output_lens_per_request"]:
            raise AssertionError("actual output lengths do not match the fixed plan")
        profiles = result["comm_profile"]
        if len(profiles) != args.tp:
            raise AssertionError(f"expected {args.tp} rank profiles")
        ranks = sorted(int(profile["tp_rank"]) for profile in profiles)
        if ranks != list(range(args.tp)):
            raise AssertionError(f"rank set mismatch: {ranks}")
        reference = normalized_profile(profiles[0])
        if reference["capture_mode"] != "histogram-only":
            raise AssertionError("capture mode is not histogram-only")
        if reference["raw_events_saved"] or reference["events"]:
            raise AssertionError("raw events were unexpectedly saved")
        if reference["events_truncated"]:
            raise AssertionError("histogram-only profile cannot be truncated")
        for profile in profiles[1:]:
            if normalized_profile(profile) != reference:
                raise AssertionError(f"rank histogram mismatch for {plan['workload_id']}")
        for event in reference["event_histograms"]:
            raw_ops.add(event["op"])
        for phase in ("prefill", "decode"):
            aggregate = aggregate_phase(reference["event_histograms"], phase, args.tp)
            labels.append(
                {
                    "workload_id": plan["workload_id"],
                    "source": plan["source"],
                    "segment": plan["segment"],
                    "split": plan["split"],
                    "tp": args.tp,
                    "phase": phase,
                    "batch_size": result["batch_size"],
                    "input_lens_json": json.dumps(
                        result["input_lens_per_request"], separators=(",", ":")
                    ),
                    "output_lens_json": json.dumps(
                        result["output_lens_per_request"], separators=(",", ":")
                    ),
                    "history_features_json": json.dumps(
                        plan["history_features"], separators=(",", ":")
                    ),
                    **aggregate,
                }
            )

    with (args.output_dir / "pattern_labels.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(labels[0]))
        writer.writeheader()
        writer.writerows(labels)
    summary = {
        "schema_version": "phase15-trace-pattern-validation-v1",
        "status": "PASS",
        "tp": args.tp,
        "workloads": len(results),
        "phase_labels": len(labels),
        "all_rank_histograms_identical": True,
        "fixed_actual_output_lengths": True,
        "histogram_only": True,
        "raw_ops": sorted(raw_ops),
        "trace_replay_mode": "draining_batch_a_i_zero",
    }
    (args.output_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
