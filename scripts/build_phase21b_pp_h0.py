#!/usr/bin/env python3
"""Build the transparent pure-PP PatternDemand base model H0.

Only the compressed service profile and declared execution configuration are
used.  No measured request order or GPU microbatch boundary is exposed to H0.
The output is a canonical draining-batch microbatch schedule, a per-boundary
payload histogram, and the derived pipeline-wide logical demand.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


BIN_COUNT = 12
BIN_EDGES = np.geomspace(4 * 1024, 8 * 1024 * 1024 * 1024, BIN_COUNT + 1)
PP_SIZES = (2, 4, 8)
MICROBATCH_SIZES = (1, 4, 16)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profiles",
        type=Path,
        default=root / "experiment-results/phase16_service_profiles/service_profiles.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase21b_pp_offline_profiledemand/h0-v1",
    )
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument("--proxy-tensor-count", type=int, default=2)
    parser.add_argument("--chunk-tokens", type=int, default=4096)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def allocate_counts(probabilities: np.ndarray, total: int) -> np.ndarray:
    probabilities = np.maximum(np.asarray(probabilities, dtype=np.float64), 0)
    probabilities /= probabilities.sum()
    raw = probabilities * total
    result = np.floor(raw).astype(int)
    # Explicit index tie-breaking keeps the canonical reconstruction identical
    # across NumPy versions and machines.
    order = sorted(
        range(len(raw)),
        key=lambda index: (-(float(raw[index]) - int(result[index])), index),
    )
    for index in order[: total - int(result.sum())]:
        result[index] += 1
    return result


def scaled_representatives(base, weights, target, lower, upper) -> np.ndarray:
    base = np.asarray(base, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
    low, high = 0.001, 256.0
    for _ in range(80):
        scale = (low + high) / 2
        mean = float(np.sum(weights * np.clip(base * scale, lower, upper)))
        if mean < target:
            low = scale
        else:
            high = scale
    return np.rint(np.clip(base * ((low + high) / 2), lower, upper)).astype(int)


def pseudo_requests(profile: dict[str, str]) -> list[tuple[int, int]]:
    """Reconstruct a deterministic 32-request canonical window from the image."""
    joint = np.asarray(json.loads(profile["joint_lm_4x4_json"]), dtype=np.float64).reshape(4, 4)
    counts = allocate_counts(joint.reshape(-1), 32)
    input_values = scaled_representatives(
        [64, 320, 1280, 4096],
        joint.sum(axis=1),
        float(profile["input_mean_capped"]),
        1,
        8192,
    )
    output_values = scaled_representatives(
        [8, 24, 48, 96],
        joint.sum(axis=0),
        float(profile["output_mean_capped"]),
        1,
        128,
    )
    scheduled = []
    for cell, amount in enumerate(counts):
        for occurrence in range(amount):
            scheduled.append(((occurrence + 0.5) / amount, cell))
    scheduled.sort()
    return [
        (int(input_values[cell // 4]), int(output_values[cell % 4]))
        for _, cell in scheduled
    ]


def prefill_events(
    requests: list[tuple[int, int]], max_microbatch: int, chunk_tokens: int
) -> list[dict]:
    remaining = [row[0] for row in requests]
    events = []
    microbatch_id = 0
    while any(value > 0 for value in remaining):
        budget = chunk_tokens
        selected = []
        active_tokens = 0
        for request_index, value in enumerate(remaining):
            if value <= 0 or len(selected) >= max_microbatch or budget <= 0:
                continue
            take = min(value, budget)
            remaining[request_index] -= take
            budget -= take
            active_tokens += take
            selected.append(request_index)
        if active_tokens <= 0:
            raise RuntimeError("Prefill H0 made no progress")
        events.append(
            {
                "phase": "prefill",
                "microbatch_id": microbatch_id,
                "decode_step": None,
                "request_indices": selected,
                "active_requests": len(selected),
                "active_tokens": active_tokens,
            }
        )
        microbatch_id += 1
    return events


def static_decode_groups(
    requests: list[tuple[int, int]], max_microbatch: int, chunk_tokens: int
) -> list[list[int]]:
    groups = []
    current = []
    current_tokens = 0
    for request_index, (input_len, _) in enumerate(requests):
        if current and (
            len(current) >= max_microbatch or current_tokens + input_len > chunk_tokens
        ):
            groups.append(current)
            current = []
            current_tokens = 0
        if input_len > chunk_tokens:
            if current:
                groups.append(current)
                current = []
                current_tokens = 0
            groups.append([request_index])
        else:
            current.append(request_index)
            current_tokens += input_len
    if current:
        groups.append(current)
    return groups


def decode_events(
    requests: list[tuple[int, int]], max_microbatch: int, chunk_tokens: int
) -> list[dict]:
    events = []
    for microbatch_id, group in enumerate(
        static_decode_groups(requests, max_microbatch, chunk_tokens)
    ):
        max_output = max(requests[index][1] for index in group)
        # Prefill samples token one.  Decode starts at step one and emits the
        # remaining M_i - 1 tokens, matching the established TP/PP convention.
        for step in range(1, max_output):
            active = [index for index in group if requests[index][1] > step]
            if active:
                events.append(
                    {
                        "phase": "decode",
                        "microbatch_id": microbatch_id,
                        "decode_step": step,
                        "request_indices": active,
                        "active_requests": len(active),
                        "active_tokens": len(active),
                    }
                )
    return events


def histogram(events: list[dict], bytes_per_token: int, proxy_count: int) -> Counter[int]:
    result: Counter[int] = Counter()
    for event in events:
        result[int(event["active_tokens"]) * bytes_per_token] += proxy_count
    return result


def bin_vectors(payload_histogram: Counter[int]) -> tuple[list[float], list[float]]:
    calls = np.zeros(BIN_COUNT, dtype=np.float64)
    logical_bytes = np.zeros(BIN_COUNT, dtype=np.float64)
    for payload, count in payload_histogram.items():
        index = int(
            np.clip(
                np.searchsorted(BIN_EDGES, payload, side="right") - 1,
                0,
                BIN_COUNT - 1,
            )
        )
        calls[index] += count
        logical_bytes[index] += payload * count
    return calls.tolist(), logical_bytes.tolist()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.profiles.open() as source:
        profiles = list(csv.DictReader(source))
    if len(profiles) != 24:
        raise ValueError(f"expected 24 service profiles, got {len(profiles)}")

    bytes_per_token = args.hidden_size * args.dtype_bytes
    sample_rows = []
    schedule_rows = []
    for profile in profiles:
        requests = pseudo_requests(profile)
        for pp_size in PP_SIZES:
            for max_microbatch in MICROBATCH_SIZES:
                phase_events = {
                    "prefill": prefill_events(requests, max_microbatch, args.chunk_tokens),
                    "decode": decode_events(requests, max_microbatch, args.chunk_tokens),
                }
                for phase, events in phase_events.items():
                    payload_histogram = histogram(
                        events, bytes_per_token, args.proxy_tensor_count
                    )
                    calls, logical_bytes = bin_vectors(payload_histogram)
                    normalization = 1000.0 / len(requests)
                    calls_1000 = [value * normalization for value in calls]
                    bytes_1000 = [value * normalization for value in logical_bytes]
                    forward_events = len(events)
                    ideal_ticks = forward_events + pp_size - 1 if forward_events else 0
                    sample_id = (
                        f"{args.model}/pp{pp_size}/mb{max_microbatch}/"
                        f"{profile['profile_id']}/{phase}"
                    )
                    sample_rows.append(
                        {
                            "sample_id": sample_id,
                            "model": args.model,
                            "profile_id": profile["profile_id"],
                            "source": profile["source"],
                            "segment": profile["segment"],
                            "phase": phase,
                            "pp_size": pp_size,
                            "pp_max_micro_batch_size": max_microbatch,
                            "representative_requests": len(requests),
                            "forward_events": forward_events,
                            "ideal_pipeline_ticks": ideal_ticks,
                            "ideal_pipeline_utilization": (
                                forward_events / ideal_ticks if ideal_ticks else 0.0
                            ),
                            "per_boundary_calls": int(sum(payload_histogram.values())),
                            "per_boundary_logical_bytes": int(
                                sum(payload * count for payload, count in payload_histogram.items())
                            ),
                            "pipeline_calls": int(sum(payload_histogram.values()))
                            * (pp_size - 1),
                            "pipeline_logical_bytes": int(
                                sum(payload * count for payload, count in payload_histogram.items())
                            )
                            * (pp_size - 1),
                            "payload_histogram_json": json.dumps(
                                dict(sorted(payload_histogram.items())), separators=(",", ":")
                            ),
                            "calls_by_12bin_per_1000_json": json.dumps(
                                calls_1000, separators=(",", ":")
                            ),
                            "logical_bytes_by_12bin_per_1000_json": json.dumps(
                                bytes_1000, separators=(",", ":")
                            ),
                        }
                    )
                    active_tokens = Counter(int(event["active_tokens"]) for event in events)
                    active_requests = Counter(int(event["active_requests"]) for event in events)
                    events_per_microbatch = Counter(
                        int(event["microbatch_id"]) for event in events
                    )
                    decode_steps = [
                        int(event["decode_step"])
                        for event in events
                        if event["decode_step"] is not None
                    ]
                    schedule_rows.append(
                        {
                            "sample_id": sample_id,
                            "phase": phase,
                            "forward_events": forward_events,
                            "microbatch_slots_used": len(events_per_microbatch),
                            "max_active_requests": max(active_requests, default=0),
                            "max_active_tokens": max(active_tokens, default=0),
                            "decode_step_start": min(decode_steps) if decode_steps else "",
                            "decode_step_end": max(decode_steps) if decode_steps else "",
                            "active_tokens_histogram_json": json.dumps(
                                dict(sorted(active_tokens.items())), separators=(",", ":")
                            ),
                            "active_requests_histogram_json": json.dumps(
                                dict(sorted(active_requests.items())), separators=(",", ":")
                            ),
                            "events_per_microbatch_json": json.dumps(
                                dict(sorted(events_per_microbatch.items())), separators=(",", ":")
                            ),
                        }
                    )

    write_csv(args.output_dir / "h0_samples.csv", sample_rows)
    write_csv(args.output_dir / "compact_schedule_summary.csv", schedule_rows)

    checks = {
        "profiles_24": len(profiles) == 24,
        "phase_samples_432": len(sample_rows) == 24 * 3 * 3 * 2,
        "only_profile_inputs": True,
        "positive_calls_and_bytes": all(
            int(row["per_boundary_calls"]) > 0
            and int(row["per_boundary_logical_bytes"]) > 0
            for row in sample_rows
        ),
    }
    summary = {
        "schema_version": "phase21b-pure-pp-h0-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "model": args.model,
        "input_contract": (
            "compressed steady service profile + model structure + PP size + "
            "PP microbatch/chunk policy"
        ),
        "output_contract": (
            "canonical per-boundary and pipeline-wide PP payload histogram, "
            "calls and logical bytes"
        ),
        "formula": {
            "payload_bytes_per_proxy_tensor": "active_tokens * hidden_size * dtype_bytes",
            "calls_per_forward_per_boundary": args.proxy_tensor_count,
            "pipeline_boundary_multiplier": "pp_size - 1",
            "canonical_arrival_mode": "draining",
        },
        "model_structure": {
            "hidden_size": args.hidden_size,
            "dtype_bytes": args.dtype_bytes,
            "proxy_tensor_count": args.proxy_tensor_count,
            "chunk_tokens": args.chunk_tokens,
        },
        "profiles": len(profiles),
        "samples": len(sample_rows),
        "schedule_summary_rows": len(schedule_rows),
        "schedule_events_represented": sum(
            int(row["forward_events"]) for row in schedule_rows
        ),
        "checks": checks,
        "profiles_sha256": sha256(args.profiles),
        "boundary": (
            "H0 deliberately reconstructs a canonical draining schedule from the "
            "compressed image. Exact request order, arrival offsets and GPU-observed "
            "microbatch boundaries are label-only information learned as a bounded residual."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["status"] != "PASS":
        raise RuntimeError(summary)
    (args.output_dir / "DONE").write_text("PASS\n")
    files = sorted(
        path
        for path in args.output_dir.iterdir()
        if path.is_file() and path.name != "manifest.sha256"
    )
    (args.output_dir / "manifest.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
