#!/usr/bin/env python3
"""Run the first pure-PP Qwen3-8B PatternDemand matrix.

This experiment always uses TP=1.  It varies PP size and the maximum PP
microbatch size, submits controlled fixed/draining workloads, and saves only
sender-side payload histograms plus compact client metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Workload:
    name: str
    input_lens: tuple[int, ...]
    output_lens: tuple[int, ...]


def build_workloads() -> list[Workload]:
    workloads = []
    for batch_size in (1, 4, 16):
        for input_len in (128, 512, 2048):
            workloads.append(
                Workload(
                    name=f"fixed_b{batch_size}_l{input_len}_m32",
                    input_lens=(input_len,) * batch_size,
                    output_lens=(32,) * batch_size,
                )
            )
    workloads.extend(
        [
            Workload(
                name="fixed_b8_l512_m128",
                input_lens=(512,) * 8,
                output_lens=(128,) * 8,
            ),
            Workload(
                name="mixed_b8_l512",
                input_lens=(512,) * 8,
                output_lens=(8, 16, 32, 64, 96, 128, 128, 128),
            ),
            Workload(
                name="longtail_b8_l512",
                input_lens=(512,) * 8,
                output_lens=(2, 2, 8, 16, 32, 64, 128, 128),
            ),
            Workload(
                name="chunk_b1_l8192_m8",
                input_lens=(8192,),
                output_lens=(8,),
            ),
        ]
    )
    return workloads


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


def input_ids(length: int, salt: int) -> list[int]:
    # Valid ordinary Qwen vocabulary IDs.  A tiny repeating pattern prevents
    # all requests from sharing identical contents while preserving length.
    return [1000 + ((salt + index) % 31) for index in range(length)]


def audit_server_cell(
    *,
    output_dir: Path,
    pp_size: int,
    strategy: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    profile_dir = output_dir / "profile"
    rank_files = sorted(profile_dir.glob("*.json"))
    snapshots = [json.loads(path.read_text(encoding="utf-8")) for path in rank_files]
    expected_workloads = {record["workload_id"] for record in records}
    checks = {
        "tp_is_one": all(record["tp_size"] == 1 for record in records),
        "all_output_lengths_exact": all(
            record["output_length_match"] for record in records
        ),
        "rank_file_count_matches_pp": len(snapshots) == pp_size,
        "rank_ids_complete": {row["pp_rank"] for row in snapshots}
        == set(range(pp_size)),
        "histogram_only": all(
            row["capture_mode"] == "histogram-only"
            and not row["raw_events_saved"]
            for row in snapshots
        ),
        "last_stage_has_no_proxy_send": all(
            histogram["msg_type"] != "proxy"
            for snapshot in snapshots
            if snapshot["pp_rank"] == pp_size - 1
            for histogram in snapshot["histograms"]
        ),
        "every_forward_boundary_has_proxy": all(
            any(
                histogram["msg_type"] == "proxy"
                for histogram in snapshot["histograms"]
            )
            for snapshot in snapshots
            if snapshot["pp_rank"] < pp_size - 1
        ),
        # The server may execute one unlabeled internal startup forward even
        # with --skip-server-warmup.  It is retained for transparency but is
        # not part of the submitted benchmark workload set.
        "all_workloads_labeled_on_forward_boundaries": all(
            {
                histogram["workload_id"]
                for histogram in snapshot["histograms"]
                if histogram["msg_type"] == "proxy"
                and histogram["workload_id"] is not None
            }
            == expected_workloads
            for snapshot in snapshots
            if snapshot["pp_rank"] < pp_size - 1
        ),
    }
    audit = {
        "schema_version": "phase19-pp-pattern-audit-v1",
        "model": "qwen3-8b",
        "tp_size": 1,
        "pp_size": pp_size,
        "strategy": strategy,
        "request_batches": len(records),
        "logical_requests": sum(len(row["input_lens"]) for row in records),
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if audit["status"] == "PASS":
        (output_dir / "DONE").write_text("PASS\n", encoding="utf-8")
    return audit


def run_server_cell(
    *,
    repo_root: Path,
    output_dir: Path,
    model_path: str,
    pp_size: int,
    strategy: str,
    pp_microbatch_size: int,
    repeats: int,
    port: int,
    startup_timeout: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = output_dir / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"qwen3-8b-pp{pp_size}-{strategy}"
    base_url = f"http://127.0.0.1:{port}"

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in range(pp_size)),
            "PYTHONPATH": str(repo_root / "python"),
            "SGLANG_PP_COMM_PROFILE_DIR": str(profile_dir),
            "SGLANG_PP_COMM_PROFILE_RUN_ID": run_id,
            # SGLang workers are terminated after each server cell and do not
            # necessarily execute Python atexit handlers.  Persist every event
            # so the final short chunk/decode tail cannot remain only in RAM.
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
        str(pp_microbatch_size),
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
                "schema_version": "phase19-pp-pattern-cell-v1",
                "model": "qwen3-8b",
                "model_path": model_path,
                "tp_size": 1,
                "pp_size": pp_size,
                "strategy": strategy,
                "pp_max_micro_batch_size": pp_microbatch_size,
                "chunked_prefill_size": 4096,
                "repeats": repeats,
                "command": command,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    server_log_path = output_dir / "server.log"
    client_log_path = output_dir / "client_results.jsonl"
    server_log = server_log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=repo_root,
        env=env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    (output_dir / "server.pid").write_text(f"{process.pid}\n", encoding="utf-8")

    records = []
    try:
        wait_for_server(base_url, process, startup_timeout)
        with client_log_path.open("w", encoding="utf-8") as client_log:
            for repeat in range(repeats):
                for workload_index, workload in enumerate(build_workloads()):
                    workload_id = (
                        f"qwen3-8b/pp{pp_size}/{strategy}/{workload.name}/r{repeat}"
                    )
                    request_count = len(workload.input_lens)
                    payload = {
                        "rid": [
                            f"{workload_id}::req{request_index}"
                            for request_index in range(request_count)
                        ],
                        "input_ids": [
                            input_ids(
                                length,
                                salt=repeat * 1000
                                + workload_index * 100
                                + request_index,
                            )
                            for request_index, length in enumerate(workload.input_lens)
                        ],
                        "sampling_params": [
                            {
                                "temperature": 0,
                                "max_new_tokens": output_len,
                                "ignore_eos": True,
                            }
                            for output_len in workload.output_lens
                        ],
                    }
                    started = time.perf_counter()
                    response = post_json(f"{base_url}/generate", payload, timeout=1800)
                    elapsed = time.perf_counter() - started
                    response_list = response if isinstance(response, list) else [response]
                    actual_output_lens = tuple(
                        int(item["meta_info"]["completion_tokens"])
                        for item in response_list
                    )
                    record = {
                        "workload_id": workload_id,
                        "model": "qwen3-8b",
                        "tp_size": 1,
                        "pp_size": pp_size,
                        "strategy": strategy,
                        "pp_max_micro_batch_size": pp_microbatch_size,
                        "repeat": repeat,
                        "workload": workload.name,
                        "input_lens": list(workload.input_lens),
                        "requested_output_lens": list(workload.output_lens),
                        "actual_output_lens": list(actual_output_lens),
                        "wall_time_s": elapsed,
                        "output_length_match": actual_output_lens
                        == workload.output_lens,
                    }
                    client_log.write(json.dumps(record, sort_keys=True) + "\n")
                    client_log.flush()
                    records.append(record)
    finally:
        terminate_process_group(process)
        server_log.close()

    audit = audit_server_cell(
        output_dir=output_dir,
        pp_size=pp_size,
        strategy=strategy,
        records=records,
    )
    if audit["status"] != "PASS":
        raise AssertionError(json.dumps(audit, indent=2))
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/media/ssd1/Qwen3-8B")
    parser.add_argument(
        "--output-root",
        default="experiment-results/phase19_pp_pattern/qwen3-8b",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--startup-timeout", type=float, default=1200)
    parser.add_argument("--port-base", type=int, default=31100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_root = (repo_root / args.output_root).resolve()
    strategies = {"mb1": 1, "mb4": 4, "mb16": 16}
    audits = []
    for pp_index, pp_size in enumerate((2, 4, 8)):
        for strategy_index, (strategy, microbatch_size) in enumerate(
            strategies.items()
        ):
            cell_dir = output_root / f"pp{pp_size}" / strategy
            if (cell_dir / "DONE").exists():
                audits.append(
                    json.loads((cell_dir / "audit.json").read_text(encoding="utf-8"))
                )
                continue
            client_results = cell_dir / "client_results.jsonl"
            if client_results.exists():
                records = [
                    json.loads(line)
                    for line in client_results.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if len(records) == args.repeats * len(build_workloads()):
                    recovered_audit = audit_server_cell(
                        output_dir=cell_dir,
                        pp_size=pp_size,
                        strategy=strategy,
                        records=records,
                    )
                    if recovered_audit["status"] == "PASS":
                        audits.append(recovered_audit)
                        continue
            audits.append(
                run_server_cell(
                    repo_root=repo_root,
                    output_dir=cell_dir,
                    model_path=args.model_path,
                    pp_size=pp_size,
                    strategy=strategy,
                    pp_microbatch_size=microbatch_size,
                    repeats=args.repeats,
                    port=args.port_base + pp_index * 10 + strategy_index,
                    startup_timeout=args.startup_timeout,
                )
            )

    summary = {
        "schema_version": "phase19-pp-pattern-matrix-v1",
        "model": "qwen3-8b",
        "tp_size": 1,
        "pp_sizes": [2, 4, 8],
        "strategies": strategies,
        "repeats": args.repeats,
        "workloads_per_repeat": len(build_workloads()),
        "cells": audits,
        "status": "PASS" if all(row["status"] == "PASS" for row in audits) else "FAIL",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "matrix_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if summary["status"] == "PASS":
        (output_root / "MATRIX_DONE").write_text("PASS\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
