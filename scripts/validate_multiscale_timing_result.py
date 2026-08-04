#!/usr/bin/env python3
"""Validate all-rank timing labels for the multiscale PatternDemand suite."""

import argparse
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("mixed-decode", "chunked-prefill"), required=True
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--output-lens", type=int, nargs="*")
    parser.add_argument("--chunk-size", type=int)
    return parser.parse_args()


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def workload_key(workload):
    return (
        int(workload["batch_size"]),
        int(workload["input_len"]),
        int(workload["output_len"]),
        tuple(int(value) for value in workload["output_lens_per_request"]),
        workload["prefill_chunk_size"],
    )


def validate_op_aware_pattern(pattern, require_op_aware):
    entries = pattern.get(
        "calls_by_raw_op_and_input_payload_bytes"
    )
    if entries is None:
        assert not require_op_aware
        return
    assert entries
    assert all(
        entry["collective_family"] == "all_reduce"
        for entry in entries
    )
    assert all(
        entry["raw_op"]
        in {
            "all_reduce",
            "fused_allreduce_residual_rmsnorm",
        }
        for entry in entries
    )
    assert sum(int(entry["count"]) for entry in entries) == int(
        pattern["all_reduce_calls"]
    )
    marginal = {}
    for entry in entries:
        payload = str(int(entry["input_payload_bytes"]))
        marginal[payload] = marginal.get(payload, 0) + int(
            entry["count"]
        )
    assert marginal == {
        str(int(payload)): int(count)
        for payload, count in pattern[
            "calls_by_input_payload_bytes"
        ].items()
    }


def main():
    args = parse_args()
    results = read_jsonl(args.result)
    labels = read_jsonl(args.ground_truth)
    assert len(results) == args.expected_rows, (len(results), args.expected_rows)
    assert len(labels) == args.expected_rows, (len(labels), args.expected_rows)

    result_by_key = {
        workload_key(
            {
                "batch_size": row["batch_size"],
                "input_len": row["input_len"],
                "output_len": row["output_len"],
                "output_lens_per_request": row["output_lens_per_request"],
                "prefill_chunk_size": row["prefill_chunk_size"],
            }
        ): row
        for row in results
    }
    assert len(result_by_key) == args.expected_rows

    seen = set()
    for label in labels:
        assert label["schema_version"] == "all-rank-comm-labels-v2"
        expected_phase = "decode" if args.mode == "mixed-decode" else "prefill"
        assert label["phase"] == expected_phase
        key = workload_key(label["workload"])
        assert key in result_by_key
        assert key not in seen
        seen.add(key)

        result = result_by_key[key]
        profiles = sorted(result["comm_profile"], key=lambda row: row["tp_rank"])
        assert [row["tp_rank"] for row in profiles] == list(range(args.tp))
        assert all(profile["capture_mode"] == "histogram-only" for profile in profiles)
        assert all(profile["raw_events_saved"] is False for profile in profiles)
        assert all(profile["events"] == [] for profile in profiles)
        assert all(profile["events_truncated"] is False for profile in profiles)
        assert all(
            profile["stats"] == profiles[0]["stats"]
            and profile["event_histograms"] == profiles[0]["event_histograms"]
            for profile in profiles[1:]
        )

        alignment = label["alignment"]
        assert alignment["exact_count_on_every_rank"]
        assert alignment["identical_backend_sequence"]
        assert alignment["identical_profiled_pattern_demand_on_every_rank"]
        assert alignment["identical_full_phase_pattern_demand_on_every_rank"]
        require_op_aware = args.model == "qwen3-30b-a3b"
        validate_op_aware_pattern(
            label["pattern_demand"], require_op_aware
        )
        validate_op_aware_pattern(
            label["full_phase_pattern_demand"],
            require_op_aware,
        )

        truth = label["all_rank_ground_truth"]
        assert truth["rank_count"] == args.tp
        estimate = truth["full_phase_estimate"]
        assert math.isclose(
            float(estimate["profiled_to_full_call_scale"]), 1.0, abs_tol=1e-12
        )
        intrinsic = float(estimate["skew_free_intrinsic_kernel_time_us"])
        post_rendezvous = float(
            estimate["post_rendezvous_completion_kernel_time_us"]
        )
        sync_inclusive = float(
            estimate["synchronization_inclusive_max_duration_sum_us"]
        )
        assert intrinsic > 0
        assert post_rendezvous > 0
        assert sync_inclusive >= intrinsic

        if args.mode == "mixed-decode":
            assert args.output_lens
            assert list(key[3]) == args.output_lens
            assert int(key[4]) == 0
        else:
            assert args.chunk_size is not None
            assert int(key[4]) == args.chunk_size

    assert len(seen) == args.expected_rows
    print(
        json.dumps(
            {
                "validated": str(args.ground_truth),
                "model": args.model,
                "mode": args.mode,
                "tp": args.tp,
                "rows": len(labels),
                "full_phase_scale": 1.0,
                "all_rank_aligned": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
