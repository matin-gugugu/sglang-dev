#!/usr/bin/env python3
"""Extract collective-kernel ground truth from a PyTorch profiler trace."""

import argparse
import gzip
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ALL_REDUCE_FAMILY_OPS = {
    "all_reduce",
    "fused_allreduce_residual_rmsnorm",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("prefill", "decode"))
    parser.add_argument("--repeat-id", type=int, required=True)
    parser.add_argument("--profile-start-step", type=int)
    parser.add_argument("--profile-end-step", type=int)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output instead of appending one JSONL row.",
    )
    return parser.parse_args()


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def read_json(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as source:
        return json.load(source)


def trace_workload(path):
    match = re.search(
        r"_batch(?P<batch>\d+)_input(?P<input>\d+)_output(?P<output>\d+)_"
        r"(?:prefill|decode)\.trace\.json(?:\.gz)?$",
        path.name,
    )
    if not match:
        return None
    return {key: int(value) for key, value in match.groupdict().items()}


def read_result(path, trace_path):
    with path.open() as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if len(rows) == 1:
        return rows[0]
    workload = trace_workload(trace_path)
    if workload is not None:
        matches = [
            row
            for row in rows
            if int(row["batch_size"]) == workload["batch"]
            and int(row["input_len"]) == workload["input"]
            and int(row["output_len"]) == workload["output"]
        ]
        if len(matches) == 1:
            return matches[0]
    raise ValueError(
        f"could not select one result row in {path} for trace {trace_path.name}"
    )


def infer_phase(path):
    match = re.search(r"_(prefill|decode)\.trace\.json(?:\.gz)?$", path.name)
    if not match:
        raise ValueError("cannot infer phase from trace filename; pass --phase")
    return match.group(1)


def classify_collective_kernel(name):
    lowered = name.lower()
    custom_algorithms = (
        "all_reduce_one_shot_push_kernel",
        "all_reduce_one_shot_pull_kernel",
        "all_reduce_one_shot_kernel",
        "all_reduce_two_shot_push_kernel",
        "all_reduce_two_shot_pull_kernel",
        "all_reduce_two_shot_kernel",
    )
    for algorithm in custom_algorithms:
        if algorithm in lowered:
            return f"sglang_custom:{algorithm.removesuffix('_kernel')}"
    if "trtllm_mnnvl_allreduce" in lowered:
        if "allreducefusionkernel" in lowered:
            return "flashinfer_mnnvl:fused_allreduce_residual_rmsnorm"
        # The two-shot fused path emits one communication kernel followed by
        # a separate RMSNorm kernel. Match only the AllReduce kernel so one
        # logical collective still maps to one timed communication kernel.
        if "twoshotallreducekernel" in lowered:
            return "flashinfer_mnnvl:fused_allreduce_residual_rmsnorm_twoshot"
    if "nccl" in lowered and ("allreduce" in lowered or "all_reduce" in lowered):
        return "nccl:all_reduce"
    return None


def profiled_count(histogram, phase, start_step, end_step):
    count = int(histogram["count"])
    if phase != "decode" or start_step is None:
        return count
    first = histogram.get("first_decode_step")
    last = histogram.get("last_decode_step")
    if first is None or last is None:
        raise ValueError("decode histogram is missing first/last_decode_step")
    total_steps = last - first + 1
    if count % total_steps:
        raise ValueError(
            f"histogram count {count} is not divisible by its {total_steps} steps"
        )
    overlap_first = max(first, start_step)
    overlap_last = min(last, end_step)
    overlap_steps = max(0, overlap_last - overlap_first + 1)
    return (count // total_steps) * overlap_steps


def phase_latency_us(result, phase, start_step, end_step):
    if phase == "prefill":
        return float(result["prefill_latency"]) * 1_000_000
    records = result["decode_step_records"]
    if start_step is not None:
        records = [
            record
            for record in records
            if start_step <= int(record["decode_step"]) <= end_step
        ]
    return sum(float(record["latency"]) for record in records) * 1_000_000


def aggregate_pattern_demand(rank_zero, phase, start_step, end_step):
    calls_by_payload = Counter()
    calls_by_op_payload = Counter()
    calls = 0
    payload_bytes = 0
    group_sizes = set()
    for histogram in rank_zero["event_histograms"]:
        raw_op = histogram["op"]
        if (
            histogram["phase"] != phase
            or raw_op not in ALL_REDUCE_FAMILY_OPS
        ):
            continue
        count = profiled_count(
            histogram,
            phase,
            start_step,
            end_step,
        )
        input_payload_bytes = int(histogram["input_payload_bytes"])
        group_sizes.add(int(histogram["group_size"]))
        calls_by_payload[input_payload_bytes] += count
        calls_by_op_payload[(raw_op, input_payload_bytes)] += count
        calls += count
        payload_bytes += count * input_payload_bytes
    if len(group_sizes) != 1:
        raise ValueError(
            f"expected one AllReduce group size in {phase}, got {sorted(group_sizes)}"
        )
    return {
        "group_size": next(iter(group_sizes)),
        "calls": calls,
        "payload_bytes": payload_bytes,
        "calls_by_payload": calls_by_payload,
        "calls_by_op_payload": calls_by_op_payload,
    }


def serialize_pattern_demand(pattern):
    group_size = pattern["group_size"]
    ring_alpha = 2 * (group_size - 1) / group_size
    ring_beta = 2 * (group_size - 1)
    return {
        "rank_scope": "representative-rank-0",
        "count_scope": "group-level-collective-calls",
        "payload_scope": "representative-rank-logical-input",
        "group_size": group_size,
        "all_reduce_calls": pattern["calls"],
        "input_payload_bytes": pattern["payload_bytes"],
        "ring_equivalent": {
            "alpha_bytes": ring_alpha,
            "beta_rounds": ring_beta,
            "bytes": pattern["payload_bytes"] * ring_alpha,
            "rounds": pattern["calls"] * ring_beta,
            "note": (
                "Normalized ring-style modeling demand, not measured wire traffic "
                "or implementation kernel steps."
            ),
        },
        "calls_by_input_payload_bytes": {
            str(key): value
            for key, value in sorted(pattern["calls_by_payload"].items())
        },
        "calls_by_raw_op_and_input_payload_bytes": [
            {
                "raw_op": raw_op,
                "collective_family": "all_reduce",
                "input_payload_bytes": payload,
                "count": count,
            }
            for (raw_op, payload), count in sorted(
                pattern["calls_by_op_payload"].items()
            )
        ],
    }


def main():
    args = parse_args()
    phase = args.phase or infer_phase(args.trace)
    if (args.profile_start_step is None) != (args.profile_end_step is None):
        raise ValueError("profile start/end step must be specified together")

    trace = read_json(args.trace)
    result = read_result(args.result, args.trace)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace

    matched = []
    backend_counts = Counter()
    name_counts = Counter()
    backend_durations = defaultdict(list)
    for event in events:
        if event.get("cat") != "kernel":
            continue
        name = event.get("name", "")
        backend = classify_collective_kernel(name)
        if backend is None:
            continue
        duration_us = float(event["dur"])
        matched.append((backend, name, duration_us))
        backend_counts[backend] += 1
        name_counts[name] += 1
        backend_durations[backend].append(duration_us)

    rank_zero = next(
        profile for profile in result["comm_profile"] if profile["tp_rank"] == 0
    )
    profiled_pattern = aggregate_pattern_demand(
        rank_zero,
        phase,
        args.profile_start_step,
        args.profile_end_step,
    )
    full_phase_pattern = aggregate_pattern_demand(
        rank_zero,
        phase,
        None,
        None,
    )
    expected_calls = profiled_pattern["calls"]
    group_size = profiled_pattern["group_size"]
    if group_size != full_phase_pattern["group_size"]:
        raise ValueError("profiled and full-phase group sizes differ")

    durations = [duration for _, _, duration in matched]
    measured_kernel_count = len(durations)
    latency_us = phase_latency_us(
        result,
        phase,
        args.profile_start_step,
        args.profile_end_step,
    )
    full_phase_latency = phase_latency_us(result, phase, None, None)
    full_phase_scale = (
        full_phase_pattern["calls"] / expected_calls if expected_calls else None
    )
    full_phase_kernel_time = (
        float(sum(durations)) * full_phase_scale
        if full_phase_scale is not None
        else None
    )
    full_phase_structural_time = (
        float(statistics.median(durations)) * full_phase_pattern["calls"]
        if durations
        else None
    )
    record = {
        "schema_version": "inference-comm-ground-truth-v1",
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
        "full_phase_pattern_demand": serialize_pattern_demand(full_phase_pattern),
        "gpu_ground_truth": {
            "trace_file": str(args.trace),
            "collective_kernel_invocations": measured_kernel_count,
            "collective_kernel_time_us": {
                "total": float(sum(durations)),
                "median_per_invocation": (
                    float(statistics.median(durations)) if durations else None
                ),
                "p95_per_invocation": percentile(durations, 95) if durations else None,
                "p99_per_invocation": percentile(durations, 99) if durations else None,
                "min_per_invocation": min(durations) if durations else None,
                "max_per_invocation": max(durations) if durations else None,
            },
            "phase_wall_time_us": latency_us,
            "full_phase_wall_time_us": full_phase_latency,
            "full_phase_estimate": {
                "method": (
                    "profiled-window mean kernel cost multiplied by full-phase calls"
                ),
                "profiled_to_full_call_scale": full_phase_scale,
                "collective_kernel_time_us": full_phase_kernel_time,
                "structural_median_kernel_time_us": full_phase_structural_time,
                "note": (
                    "For uniform Decode workloads the payload is constant across steps; "
                    "Prefill is fully profiled and has scale 1."
                ),
            },
            "collective_kernel_fraction_of_phase_wall_time": (
                float(sum(durations)) / latency_us if latency_us else None
            ),
            "backend_kernel_counts": dict(sorted(backend_counts.items())),
            "kernel_name_counts": dict(sorted(name_counts.items())),
            "backend_time_us": {
                backend: float(sum(values))
                for backend, values in sorted(backend_durations.items())
            },
        },
        "alignment": {
            "expected_group_level_calls": expected_calls,
            "measured_collective_kernel_invocations": measured_kernel_count,
            "exact_one_kernel_per_call": expected_calls == measured_kernel_count,
            "note": (
                "Exact alignment proves one matched GPU kernel per recorded group-level "
                "collective for this backend/workload. A mismatch can be legitimate for "
                "multi-kernel collective implementations and must be investigated."
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "a"
    with args.output.open(mode) as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        f"{record['run_name']} {phase}: expected_calls={expected_calls} "
        f"kernels={measured_kernel_count} comm_us={sum(durations):.3f} "
        f"fraction={record['gpu_ground_truth']['collective_kernel_fraction_of_phase_wall_time']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
