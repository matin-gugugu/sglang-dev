#!/usr/bin/env python3
"""Run arrival-aware pure-PP service-profile PatternDemand experiments.

The source profiles come from Phase 16.  Each profile is replayed twice with
the exact same request lengths: once at a deterministic arrival process
derived from the service profile and once as an all-at-once draining batch.  This paired control makes
arrival/burst effects identifiable without turning the task into a forecast of
the next time window.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SMOKE_PROFILES = (
    "profile_02_burstgpt_1_c1",  # high-RPS, bursty, short requests
    "profile_10_burstgpt_2_c4",  # comparatively steady, short requests
    "profile_24_mooncake_synthetic_c0",  # external, long-input profile
)


def post_json(url: str, payload: dict[str, Any], timeout: float) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def wait_for_server(base_url: str, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "not contacted"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2):
                return
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = str(error)
            time.sleep(2)
    raise TimeoutError(f"server did not become ready: {last_error}")


def terminate_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def synthetic_input_ids(length: int, salt: int) -> list[int]:
    return [1000 + ((salt + index) % 31) for index in range(length)]


def load_profile_requests(plan_path: Path, profile_ids: tuple[str, ...]) -> dict[str, list[dict[str, int]]]:
    """Recover the fixed 32-request subset used by Phase 16 from one strategy.

    The three Phase-16 strategies replayed the same request set but partitioned
    it into different draining microbatches.  The latency plan is used only as
    a lossless container for those requests; PP strategies are applied by the
    server in this experiment.
    """
    rows = [json.loads(line) for line in plan_path.read_text().splitlines() if line]
    result: dict[str, list[dict[str, int]]] = {}
    for profile_id in profile_ids:
        selected = sorted(
            (
                row
                for row in rows
                if row["profile_id"] == profile_id
                and row["strategy"] == "latency"
                and int(row["repeat"]) == 0
            ),
            key=lambda row: int(row["batch_index"]),
        )
        requests = []
        for row in selected:
            lengths = (
                len(row["input_lens_per_request"]),
                len(row["output_lens_per_request"]),
                len(row["arrival_offsets_ms_audit_only"]),
            )
            if len(set(lengths)) != 1:
                raise ValueError(f"{profile_id}: mismatched replay-plan arrays: {lengths}")
            for input_len, output_len, arrival_ms in zip(
                row["input_lens_per_request"],
                row["output_lens_per_request"],
                row["arrival_offsets_ms_audit_only"],
            ):
                requests.append(
                    {
                        "input_len": int(input_len),
                        "output_len": int(output_len),
                        "arrival_offset_ms": int(arrival_ms),
                    }
                )
        if len(requests) != 32:
            raise ValueError(f"{profile_id}: expected 32 fixed requests, got {len(requests)}")
        requests.sort(key=lambda row: row["arrival_offset_ms"])
        for index, request in enumerate(requests):
            request["request_index"] = index
        result[profile_id] = requests
    return result


def load_profile_metadata(path: Path, profile_ids: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    with path.open() as source:
        rows = {row["profile_id"]: row for row in csv.DictReader(source)}
    missing = set(profile_ids) - set(rows)
    if missing:
        raise ValueError(f"missing service profiles: {sorted(missing)}")
    return {profile_id: rows[profile_id] for profile_id in profile_ids}


def coefficient_of_variation(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean <= 0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / mean


def apply_profiled_arrivals(
    requests: list[dict[str, int]], profile: dict[str, Any]
) -> tuple[list[dict[str, int]], dict[str, float]]:
    """Create a deterministic stationary arrival realization from an image.

    The 32 requests are stratified length samples, not a consecutive trace
    slice. Their sparse original timestamps therefore cannot be replayed as-is.
    A gamma renewal process preserves the image's mean RPS, targets its
    inter-arrival CV, and does not claim to forecast a future request sequence.
    """
    target_rps = float(profile["rps"])
    target_cv = max(float(profile["interarrival_cv"]), 0.05)
    seed = int(hashlib.sha256(profile["profile_id"].encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    shape = 1.0 / (target_cv * target_cv)
    raw_intervals = [
        rng.gammavariate(shape, 1.0 / shape) for _ in range(len(requests) - 1)
    ]
    target_span_s = (len(requests) - 1) / max(target_rps, 1e-6)
    scale = target_span_s / max(sum(raw_intervals), 1e-12)
    intervals_s = [value * scale for value in raw_intervals]
    offsets_ms = [0]
    elapsed = 0.0
    for interval in intervals_s:
        elapsed += interval
        offsets_ms.append(round(elapsed * 1000))
    profiled = [
        {**request, "arrival_offset_ms": int(offset_ms)}
        for request, offset_ms in zip(requests, offsets_ms)
    ]
    return profiled, {
        "target_rps": target_rps,
        "target_interarrival_cv": target_cv,
        "planned_rps": (len(requests) - 1) / max(offsets_ms[-1] / 1000.0, 1e-9),
        "planned_interarrival_cv": coefficient_of_variation(intervals_s),
        "planned_span_ms": float(offsets_ms[-1]),
    }


async def replay_profile(
    *,
    base_url: str,
    workload_id: str,
    requests: list[dict[str, int]],
    arrival_mode: str,
    repeat: int,
    profile_index: int,
    timeout: float,
) -> dict[str, Any]:
    started = time.perf_counter()

    async def send_one(request: dict[str, int]) -> dict[str, Any]:
        planned_ms = request["arrival_offset_ms"] if arrival_mode == "profiled" else 0
        delay = started + planned_ms / 1000.0 - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        submit_started = time.perf_counter()
        request_index = request["request_index"]
        payload = {
            "rid": f"{workload_id}::req{request_index}",
            "input_ids": synthetic_input_ids(
                request["input_len"],
                salt=repeat * 10000 + profile_index * 1000 + request_index,
            ),
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": request["output_len"],
                "ignore_eos": True,
            },
        }
        response = await asyncio.to_thread(
            post_json, f"{base_url}/generate", payload, timeout
        )
        completed = time.perf_counter()
        response_item = response[0] if isinstance(response, list) else response
        actual_output_len = int(response_item["meta_info"]["completion_tokens"])
        return {
            **request,
            "planned_submit_offset_ms": planned_ms,
            "actual_submit_offset_ms": (submit_started - started) * 1000.0,
            "completion_offset_ms": (completed - started) * 1000.0,
            "actual_output_len": actual_output_len,
            "output_length_match": actual_output_len == request["output_len"],
        }

    results = await asyncio.gather(*(send_one(request) for request in requests))
    results.sort(key=lambda row: row["request_index"])
    actual_submit_offsets = [row["actual_submit_offset_ms"] for row in results]
    planned_submit_offsets = [row["planned_submit_offset_ms"] for row in results]
    max_arrival_error = max(
        abs(actual - planned)
        for actual, planned in zip(actual_submit_offsets, planned_submit_offsets)
    )
    return {
        "workload_id": workload_id,
        "arrival_mode": arrival_mode,
        "request_count": len(results),
        "planned_arrival_span_ms": max(planned_submit_offsets),
        "actual_arrival_span_ms": max(actual_submit_offsets) - min(actual_submit_offsets),
        "max_arrival_submit_error_ms": max_arrival_error,
        "wall_time_s": time.perf_counter() - started,
        "all_output_lengths_exact": all(row["output_length_match"] for row in results),
        "requests": results,
    }


def sender_signature(snapshot: dict[str, Any], expected: set[str]) -> dict[str, dict[tuple, int]]:
    signatures: dict[str, dict[tuple, int]] = defaultdict(lambda: defaultdict(int))
    for row in snapshot["histograms"]:
        workload_id = row.get("workload_id")
        if row.get("msg_type") != "proxy" or workload_id not in expected:
            continue
        key = (
            row["phase"],
            row["raw_op"],
            int(row["payload_bytes"]),
            row["tensor_name"],
            row.get("active_batch_size"),
            row.get("active_tokens"),
        )
        signatures[workload_id][key] += int(row["count"])
    return {workload: dict(values) for workload, values in signatures.items()}


def payload_signature(signature: dict[tuple, int]) -> dict[tuple, int]:
    """Remove phase and scheduler annotations but retain exact message sizes."""
    result: dict[tuple, int] = defaultdict(int)
    for key, count in signature.items():
        _phase, raw_op, payload, tensor_name, _active_batch, _active_tokens = key
        result[(raw_op, payload, tensor_name)] += count
    return dict(result)


def audit_cell(
    *,
    output_dir: Path,
    pp_size: int,
    microbatch_size: int,
    records: list[dict[str, Any]],
    profiles: tuple[str, ...],
    arrival_modes: tuple[str, ...],
) -> dict[str, Any]:
    snapshots = [
        json.loads(path.read_text())
        for path in sorted((output_dir / "profile").glob("*.json"))
    ]
    expected = {record["workload_id"] for record in records}
    senders = sorted(
        (snapshot for snapshot in snapshots if int(snapshot["pp_rank"]) < pp_size - 1),
        key=lambda snapshot: snapshot["pp_rank"],
    )
    signatures = [sender_signature(snapshot, expected) for snapshot in senders]
    paired_differences = []
    by_key = {
        (record["profile_id"], record["repeat"], record["arrival_mode"]): record["workload_id"]
        for record in records
    }
    reference = signatures[0] if signatures else {}
    paired_controls_applicable = set(arrival_modes) == {"profiled", "draining"}
    if paired_controls_applicable:
        for profile_id in profiles:
            for repeat in sorted({int(record["repeat"]) for record in records}):
                trace_id = by_key[(profile_id, repeat, "profiled")]
                draining_id = by_key[(profile_id, repeat, "draining")]
                profiled_signature = reference.get(trace_id, {})
                draining_signature = reference.get(draining_id, {})
                paired_differences.append(
                    {
                        "profile_id": profile_id,
                        "repeat": repeat,
                        "phase_aware_histogram_changed": profiled_signature
                        != draining_signature,
                        "payload_histogram_changed": payload_signature(profiled_signature)
                        != payload_signature(draining_signature),
                    }
                )

    checks = {
        "all_output_lengths_exact": all(record["all_output_lengths_exact"] for record in records),
        "rank_file_count_matches_pp": len(snapshots) == pp_size,
        "rank_ids_complete": {int(row["pp_rank"]) for row in snapshots} == set(range(pp_size)),
        "histogram_only": all(
            row["capture_mode"] == "histogram-only" and not row["raw_events_saved"]
            for row in snapshots
        ),
        "last_stage_has_no_proxy_send": all(
            histogram["msg_type"] != "proxy"
            for snapshot in snapshots
            if int(snapshot["pp_rank"]) == pp_size - 1
            for histogram in snapshot["histograms"]
        ),
        "all_expected_workloads_labeled": bool(senders)
        and all(set(signature) == expected for signature in signatures),
        "forward_boundaries_identical": bool(signatures)
        and all(signature == signatures[0] for signature in signatures[1:]),
        "arrival_schedule_applied": all(
            record["max_arrival_submit_error_ms"] <= 1000.0
            for record in records
            if record["arrival_mode"] == "profiled"
        ),
        # Whether arrival changes the histogram is the scientific result, not
        # a validity precondition.  A valid negative result must not fail the
        # experiment. Completeness of every paired control is audited here.
        "paired_arrival_controls_complete": not paired_controls_applicable
        or len(paired_differences)
        == len(profiles) * len({int(record["repeat"]) for record in records}),
    }
    audit = {
        "schema_version": "phase21-pp-profile-cell-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "pp_size": pp_size,
        "pp_max_micro_batch_size": microbatch_size,
        "arrival_modes": list(arrival_modes),
        "paired_arrival_controls_applicable": paired_controls_applicable,
        "request_profiles": len(profiles),
        "profile_replays": len(records),
        "logical_requests": sum(record["request_count"] for record in records),
        "phases_observed": sorted(
            {
                row["phase"]
                for snapshot in senders[:1]
                for row in snapshot["histograms"]
                if row.get("workload_id") in expected and row.get("msg_type") == "proxy"
            }
        ),
        "checks": checks,
        "paired_arrival_controls": paired_differences,
        "phase_aware_changed_pairs": sum(
            row["phase_aware_histogram_changed"] for row in paired_differences
        ),
        "payload_changed_pairs": sum(
            row["payload_histogram_changed"] for row in paired_differences
        ),
    }
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if audit["status"] == "PASS":
        (output_dir / "DONE").write_text("PASS\n")
    return audit


def run_cell(
    *,
    repo_root: Path,
    output_dir: Path,
    model_path: str,
    pp_size: int,
    strategy: str,
    microbatch_size: int,
    profile_requests: dict[str, list[dict[str, int]]],
    profile_metadata: dict[str, dict[str, Any]],
    arrival_modes: tuple[str, ...],
    repeats: int,
    port: int,
    startup_timeout: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = output_dir / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"http://127.0.0.1:{port}"
    run_id = f"qwen3-8b-profile-pp{pp_size}-{strategy}"

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in range(pp_size)),
            "PYTHONPATH": str(repo_root / "python"),
            "SGLANG_PP_COMM_PROFILE_DIR": str(profile_dir),
            "SGLANG_PP_COMM_PROFILE_RUN_ID": run_id,
            "SGLANG_PP_COMM_PROFILE_FLUSH_INTERVAL": "1",
        }
    )
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--tp-size",
        "1",
        "--pp-size",
        str(pp_size),
        "--pp-max-micro-batch-size",
        str(microbatch_size),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--disable-cuda-graph",
        "--disable-radix-cache",
        "--skip-server-warmup",
        "--mem-fraction-static",
        "0.75",
        "--chunked-prefill-size",
        "4096",
    ]
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "schema_version": "phase21-pp-profile-cell-v1",
                "model": "qwen3-8b",
                "model_path": model_path,
                "tp_size": 1,
                "pp_size": pp_size,
                "strategy": strategy,
                "pp_max_micro_batch_size": microbatch_size,
                "chunked_prefill_size": 4096,
                "profiles": list(profile_requests),
                "arrival_modes": list(arrival_modes),
                "repeats": repeats,
                "command": command,
            },
            indent=2,
        )
        + "\n"
    )

    server_log = (output_dir / "server.log").open("w")
    process = subprocess.Popen(
        command,
        cwd=repo_root,
        env=env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    (output_dir / "server.pid").write_text(f"{process.pid}\n")
    records = []
    try:
        wait_for_server(base_url, process, startup_timeout)
        with (output_dir / "client_results.jsonl").open("w") as client_log:
            for repeat in range(repeats):
                for profile_index, (profile_id, requests) in enumerate(profile_requests.items()):
                    profiled_requests, arrival_audit = apply_profiled_arrivals(
                        requests, profile_metadata[profile_id]
                    )
                    for arrival_mode in arrival_modes:
                        workload_id = (
                            f"qwen3-8b/pp{pp_size}/{strategy}/{profile_id}/"
                            f"{arrival_mode}/r{repeat}"
                        )
                        replay = asyncio.run(
                            replay_profile(
                                base_url=base_url,
                                workload_id=workload_id,
                                requests=profiled_requests,
                                arrival_mode=arrival_mode,
                                repeat=repeat,
                                profile_index=profile_index,
                                timeout=1800,
                            )
                        )
                        record = {
                            **replay,
                            "model": "qwen3-8b",
                            "tp_size": 1,
                            "pp_size": pp_size,
                            "strategy": strategy,
                            "pp_max_micro_batch_size": microbatch_size,
                            "profile_id": profile_id,
                            "repeat": repeat,
                            "arrival_profile": arrival_audit,
                        }
                        client_log.write(json.dumps(record, sort_keys=True) + "\n")
                        client_log.flush()
                        records.append(record)
                        time.sleep(0.5)
    finally:
        terminate_process_group(process)
        server_log.close()

    audit = audit_cell(
        output_dir=output_dir,
        pp_size=pp_size,
        microbatch_size=microbatch_size,
        records=records,
        profiles=tuple(profile_requests),
        arrival_modes=arrival_modes,
    )
    if audit["status"] != "PASS":
        raise AssertionError(json.dumps(audit, indent=2))
    return audit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="/media/ssd1/Qwen3-8B")
    p.add_argument(
        "--plan",
        type=Path,
        default=Path("experiment-results/phase16_profiledemand_plans/full_replay_plan.jsonl"),
    )
    p.add_argument(
        "--service-profiles",
        type=Path,
        default=Path("experiment-results/phase16_service_profiles/service_profiles.csv"),
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiment-results/phase21_pp_service_profile/qwen3-8b-smoke-v1"),
    )
    p.add_argument("--profiles", nargs="+", default=list(DEFAULT_SMOKE_PROFILES))
    p.add_argument(
        "--all-profiles",
        action="store_true",
        help="Use every profile in service_profiles.csv; overrides --profiles.",
    )
    p.add_argument(
        "--arrival-modes",
        nargs="+",
        choices=("profiled", "draining"),
        default=["profiled", "draining"],
    )
    p.add_argument("--pp-sizes", nargs="+", type=int, default=[2, 4, 8])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--startup-timeout", type=float, default=1200)
    p.add_argument("--port-base", type=int, default=31300)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    plan = args.plan if args.plan.is_absolute() else repo_root / args.plan
    service_profiles = (
        args.service_profiles
        if args.service_profiles.is_absolute()
        else repo_root / args.service_profiles
    )
    output_root = args.output_root if args.output_root.is_absolute() else repo_root / args.output_root
    if args.all_profiles:
        with service_profiles.open() as source:
            profiles = tuple(row["profile_id"] for row in csv.DictReader(source))
    else:
        profiles = tuple(args.profiles)
    arrival_modes = tuple(dict.fromkeys(args.arrival_modes))
    requests = load_profile_requests(plan, profiles)
    profile_metadata = load_profile_metadata(service_profiles, profiles)
    strategies = {"mb1": 1, "mb4": 4, "mb16": 16}
    audits = []
    for pp_index, pp_size in enumerate(args.pp_sizes):
        for strategy_index, (strategy, microbatch_size) in enumerate(strategies.items()):
            cell = output_root / f"pp{pp_size}" / strategy
            if (cell / "DONE").exists():
                audits.append(json.loads((cell / "audit.json").read_text()))
                continue
            if cell.exists() and any(cell.iterdir()):
                client_path = cell / "client_results.jsonl"
                records = (
                    [
                        json.loads(line)
                        for line in client_path.read_text().splitlines()
                        if line
                    ]
                    if client_path.exists()
                    else []
                )
                expected_records = args.repeats * len(profiles) * len(arrival_modes)
                if len(records) == expected_records:
                    recovered = audit_cell(
                        output_dir=cell,
                        pp_size=pp_size,
                        microbatch_size=microbatch_size,
                        records=records,
                        profiles=profiles,
                        arrival_modes=arrival_modes,
                    )
                    if recovered["status"] == "PASS":
                        audits.append(recovered)
                        continue
                raise RuntimeError(
                    f"incomplete cell requires a new attempt: {cell}; "
                    f"records={len(records)}/{expected_records}"
                )
            audits.append(
                run_cell(
                    repo_root=repo_root,
                    output_dir=cell,
                    model_path=args.model_path,
                    pp_size=pp_size,
                    strategy=strategy,
                    microbatch_size=microbatch_size,
                    profile_requests=requests,
                    profile_metadata=profile_metadata,
                    arrival_modes=arrival_modes,
                    repeats=args.repeats,
                    port=args.port_base + pp_index * 10 + strategy_index,
                    startup_timeout=args.startup_timeout,
                )
            )

    summary = {
        "schema_version": "phase21-pp-profile-matrix-v1",
        "status": "PASS" if all(audit["status"] == "PASS" for audit in audits) else "FAIL",
        "model": "qwen3-8b",
        "tp_size": 1,
        "pp_sizes": args.pp_sizes,
        "strategies": strategies,
        "profiles": list(profiles),
        "arrival_modes": list(arrival_modes),
        "repeats": args.repeats,
        "cells": audits,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "matrix_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["status"] == "PASS":
        (output_root / "MATRIX_DONE").write_text("PASS\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
