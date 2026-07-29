#!/usr/bin/env python3
"""Aggregate rank-aligned collective kernels into a critical communication cost."""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from extract_inference_comm_trace import (
    aggregate_pattern_demand,
    classify_collective_kernel,
    infer_phase,
    phase_latency_us,
    read_json,
    read_result,
    serialize_pattern_demand,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat-id", type=int, required=True)
    parser.add_argument("--phase", choices=("prefill", "decode"))
    parser.add_argument("--profile-start-step", type=int)
    parser.add_argument("--profile-end-step", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def matched_kernels(path):
    trace = read_json(path)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    matched = []
    for event in events:
        if event.get("cat") != "kernel":
            continue
        backend = classify_collective_kernel(event.get("name", ""))
        if backend is None:
            continue
        matched.append(
            {
                "timestamp_us": float(event.get("ts", 0.0)),
                "duration_us": float(event["dur"]),
                "backend": backend,
                "name": event.get("name", ""),
            }
        )
    matched.sort(key=lambda event: (event["timestamp_us"], event["duration_us"]))
    return matched


def demand_signature(pattern):
    return (
        pattern["group_size"],
        pattern["calls"],
        pattern["payload_bytes"],
        tuple(sorted(pattern["calls_by_payload"].items())),
    )


def main():
    args = parse_args()
    phase = args.phase or infer_phase(args.trace[0])
    if (args.profile_start_step is None) != (args.profile_end_step is None):
        raise ValueError("profile start/end step must be specified together")

    result = read_result(args.result, args.trace[0])
    rank_zero_profile = next(
        profile for profile in result["comm_profile"] if profile["tp_rank"] == 0
    )
    profiled_pattern = aggregate_pattern_demand(
        rank_zero_profile,
        phase,
        args.profile_start_step,
        args.profile_end_step,
    )
    full_pattern = aggregate_pattern_demand(
        rank_zero_profile,
        phase,
        None,
        None,
    )
    group_size = profiled_pattern["group_size"]
    if len(args.trace) != group_size:
        raise ValueError(
            f"expected {group_size} rank traces, got {len(args.trace)}"
        )
    comm_profiles = sorted(
        result["comm_profile"], key=lambda profile: profile["tp_rank"]
    )
    if [profile["tp_rank"] for profile in comm_profiles] != list(
        range(group_size)
    ):
        raise ValueError("comm profiles do not contain exactly one entry per rank")
    profiled_patterns_by_rank = [
        aggregate_pattern_demand(
            profile,
            phase,
            args.profile_start_step,
            args.profile_end_step,
        )
        for profile in comm_profiles
    ]
    full_patterns_by_rank = [
        aggregate_pattern_demand(profile, phase, None, None)
        for profile in comm_profiles
    ]
    identical_profiled_demand = all(
        demand_signature(pattern) == demand_signature(profiled_pattern)
        for pattern in profiled_patterns_by_rank
    )
    identical_full_demand = all(
        demand_signature(pattern) == demand_signature(full_pattern)
        for pattern in full_patterns_by_rank
    )
    if not identical_profiled_demand or not identical_full_demand:
        raise ValueError("logical PatternDemand differs across ranks")

    events_by_rank = [matched_kernels(path) for path in args.trace]
    expected_calls = profiled_pattern["calls"]
    counts = [len(events) for events in events_by_rank]
    if any(count != expected_calls for count in counts):
        raise ValueError(
            f"rank kernel counts {counts} do not match expected calls {expected_calls}"
        )
    backend_sequences = [
        [event["backend"] for event in events] for events in events_by_rank
    ]
    if any(sequence != backend_sequences[0] for sequence in backend_sequences[1:]):
        raise ValueError("collective backend sequence differs across ranks")

    durations = np.asarray(
        [
            [event["duration_us"] for event in events]
            for events in events_by_rank
        ],
        dtype=np.float64,
    )
    per_rank_totals = np.sum(durations, axis=1)
    per_collective_max = np.max(durations, axis=0)
    per_collective_min = np.min(durations, axis=0)
    per_collective_skew = per_collective_max - per_collective_min
    full_scale = (
        full_pattern["calls"] / expected_calls if expected_calls else None
    )

    rank_zero_window = float(per_rank_totals[0])
    max_rank_window = float(np.max(per_rank_totals))
    critical_window = float(np.sum(per_collective_max))
    record = {
        "schema_version": "all-rank-comm-critical-v1",
        "run_name": result["run_name"],
        "repeat_id": args.repeat_id,
        "phase": phase,
        "workload": {
            "batch_size": result["batch_size"],
            "input_len": result["input_len"],
            "output_len": result["output_len"],
            "output_lens_per_request": result["output_lens_per_request"],
            "prefill_chunk_size": result["prefill_chunk_size"],
        },
        "profile_window": {
            "start_decode_step": args.profile_start_step,
            "end_decode_step": args.profile_end_step,
        },
        "pattern_demand": serialize_pattern_demand(profiled_pattern),
        "full_phase_pattern_demand": serialize_pattern_demand(full_pattern),
        "all_rank_ground_truth": {
            "rank_count": group_size,
            "trace_files": [str(path) for path in args.trace],
            "kernel_invocations_per_rank": counts,
            "logical_pattern_demand_per_rank": {
                str(rank): {
                    "profiled_calls": pattern["calls"],
                    "profiled_payload_bytes": pattern["payload_bytes"],
                    "full_phase_calls": full_patterns_by_rank[rank]["calls"],
                    "full_phase_payload_bytes": full_patterns_by_rank[rank][
                        "payload_bytes"
                    ],
                }
                for rank, pattern in enumerate(profiled_patterns_by_rank)
            },
            "backend_sequence_signature": "+".join(
                sorted(set(backend_sequences[0]))
            ),
            "profiled_window": {
                "rank_kernel_time_us": {
                    str(rank): float(total)
                    for rank, total in enumerate(per_rank_totals)
                },
                "rank0_kernel_time_us": rank_zero_window,
                "max_rank_total_kernel_time_us": max_rank_window,
                "per_collective_critical_kernel_time_us": critical_window,
                "critical_over_rank0": (
                    critical_window / rank_zero_window
                    if rank_zero_window
                    else None
                ),
                "max_rank_total_over_rank0": (
                    max_rank_window / rank_zero_window
                    if rank_zero_window
                    else None
                ),
                "per_collective_rank_skew_us": {
                    "median": float(statistics.median(per_collective_skew)),
                    "p95": percentile(per_collective_skew, 95),
                    "p99": percentile(per_collective_skew, 99),
                    "max": float(np.max(per_collective_skew)),
                },
            },
            "full_phase_estimate": {
                "profiled_to_full_call_scale": full_scale,
                "rank0_kernel_time_us": rank_zero_window * full_scale,
                "max_rank_total_kernel_time_us": max_rank_window * full_scale,
                "per_collective_critical_kernel_time_us": (
                    critical_window * full_scale
                ),
                "phase_wall_time_us": phase_latency_us(
                    result, phase, None, None
                ),
                "definition": (
                    "sum over aligned group-level collectives of the slowest "
                    "rank kernel duration, then scale a uniform Decode window "
                    "by full-phase calls; Prefill scale is 1"
                ),
            },
        },
        "alignment": {
            "expected_group_level_calls": expected_calls,
            "rank_kernel_invocations": counts,
            "exact_count_on_every_rank": all(
                count == expected_calls for count in counts
            ),
            "identical_backend_sequence": all(
                sequence == backend_sequences[0]
                for sequence in backend_sequences[1:]
            ),
            "identical_profiled_pattern_demand_on_every_rank": (
                identical_profiled_demand
            ),
            "identical_full_phase_pattern_demand_on_every_rank": (
                identical_full_demand
            ),
            "note": (
                "Ordinal alignment is valid only after exact call counts and "
                "identical backend sequences are verified on every rank. "
                "PatternDemand remains group-level and uses one representative "
                "rank; per-rank copies are checked for equality, not summed."
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "a"
    with args.output.open(mode) as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        f"{record['run_name']} {phase}: calls={expected_calls} ranks={group_size} "
        f"rank0_us={rank_zero_window:.3f} critical_us={critical_window:.3f} "
        f"ratio={critical_window / rank_zero_window:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
