#!/usr/bin/env python3
"""Run Phase41 GPU sentinel, then build the compact development dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import (  # noqa: E402
    environment_record,
    load_json,
    repo_root,
    require_clean_before_run,
    require_expected_head,
    sha256,
    utc_now,
    verify_pinned_inputs,
    write_json,
)
from contracts import (  # noqa: E402
    BIN_EDGES_BYTES,
    histogram_vectors,
    profile_example_rows,
    read_bundle,
    read_csv,
    sentinel_workload,
    validate_bundle,
    write_csv,
    write_csv_gz,
)
from preflight import bundle_audit, parse_gpu_pair, phase40_preflight_module  # noqa: E402


def wait_http(url: str, process: subprocess.Popen, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    last_error: str | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"process exited before readiness: rc={process.returncode}, url={url}"
            )
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if 200 <= response.status < 300:
                    return
        except Exception as error:  # readiness polling intentionally broad
            last_error = str(error)
        time.sleep(2)
    raise RuntimeError(f"server readiness timeout: {url}, last_error={last_error}")


def post_json(url: str, payload: dict[str, Any], timeout: int = 1800) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"request failed: status={error.code}, body={detail[:2000]}"
        ) from error
    return json.loads(body)


def terminate_processes(processes: list[subprocess.Popen]) -> None:
    for process in reversed(processes):
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
    deadline = time.monotonic() + 30
    for process in reversed(processes):
        remaining = max(0.1, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def redacted_command(command: list[str], model_path: Path) -> list[str]:
    return ["$MODEL_PATH" if value == str(model_path.resolve()) else value for value in command]


def build_server_commands(
    *,
    contract: dict[str, Any],
    model_path: Path,
    ib_device: str,
    prefill_port: int,
    decode_port: int,
    router_port: int,
    bootstrap_port: int,
) -> dict[str, list[str]]:
    measurement = contract["measurement_contract"]
    common = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model_path.resolve()),
        "--tp",
        "1",
        "--pp-size",
        "1",
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "auto",
        "--attention-backend",
        measurement["attention_backend"],
        "--page-size",
        str(measurement["page_size_tokens"]),
        "--schedule-policy",
        measurement["schedule_policy"],
        "--chunked-prefill-size",
        str(measurement["chunked_prefill_tokens"]),
        "--max-prefill-tokens",
        str(measurement["max_prefill_tokens"]),
        "--max-running-requests",
        str(measurement["max_running_requests"]),
        "--context-length",
        str(measurement["context_length"]),
        "--mem-fraction-static",
        str(measurement["mem_fraction_static"]),
        "--disable-radix-cache",
        "--disable-overlap-schedule",
        "--disaggregation-transfer-backend",
        measurement["transfer_backend"],
        "--disaggregation-ib-device",
        ib_device,
        "--disaggregation-bootstrap-port",
        str(bootstrap_port),
        "--host",
        "127.0.0.1",
    ]
    return {
        "prefill": [
            *common,
            "--disaggregation-mode",
            "prefill",
            "--optimistic-prefill-retries",
            "0",
            "--port",
            str(prefill_port),
        ],
        "decode": [
            *common,
            "--disaggregation-mode",
            "decode",
            "--port",
            str(decode_port),
        ],
        "router": [
            sys.executable,
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--prefill",
            f"http://127.0.0.1:{prefill_port}",
            "--decode",
            f"http://127.0.0.1:{decode_port}",
            "--host",
            "127.0.0.1",
            "--port",
            str(router_port),
        ],
    }


def read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid JSONL: {path}:{line_number}") from error
    return rows


def raw_manifest(raw_dir: Path, event_count: int) -> dict[str, Any]:
    files = []
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "relative_path": str(path.relative_to(raw_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    return {
        "schema_version": "phase41-external-raw-manifest-v1",
        "external_raw_dir": str(raw_dir),
        "raw_committed_to_git": False,
        "complete_requests_committed_to_git": False,
        "file_count": len(files),
        "bytes": sum(row["bytes"] for row in files),
        "profiler_event_count": event_count,
        "files": files,
    }


def compare_gpu_teacher(
    *,
    contract: dict[str, Any],
    model: dict[str, Any],
    requests: list[dict[str, Any]],
    teacher: list[dict[str, Any]],
    raw_events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    request_map = {row["rid"]: row for row in requests}
    expected_rids = set(request_map)
    unexpected = [row for row in raw_events if row.get("rid") not in expected_rids]
    matching = [row for row in raw_events if row.get("rid") in expected_rids]
    matching.sort(key=lambda row: int(row.get("sequence", -1)))
    gpu_by_rid = {rid: [] for rid in expected_rids}
    teacher_by_rid = {rid: [] for rid in expected_rids}
    for row in matching:
        gpu_by_rid[row["rid"]].append(row)
    for row in teacher:
        teacher_by_rid[row["rid"]].append(row)

    fields = ("page_start", "page_end", "kv_page_count", "logical_bytes")
    exact_by_rid = {}
    mismatch_examples = []
    for rid in sorted(expected_rids):
        actual = gpu_by_rid[rid]
        expected = teacher_by_rid[rid]
        exact = len(actual) == len(expected) and all(
            all(int(left.get(field, -1)) == int(right[field]) for field in fields)
            for left, right in zip(actual, expected)
        )
        exact_by_rid[rid] = exact
        if not exact and len(mismatch_examples) < 20:
            mismatch_examples.append(
                {
                    "rid": rid,
                    "request": {
                        key: request_map[rid][key]
                        for key in ("case", "repeat", "wave_index", "request_index", "prompt_tokens")
                    },
                    "gpu": [{field: row.get(field) for field in fields} for row in actual],
                    "teacher": [{field: row.get(field) for field in fields} for row in expected],
                }
            )

    cases = [item["name"] for item in contract["gpu_sentinel_contract"]["synthetic_cases"]]
    cases += [item["name"] for item in contract["gpu_sentinel_contract"]["real_full_window_cases"]]
    alignment = []
    histogram_rows = []
    for case in [*cases, "overall"]:
        rids = [
            rid
            for rid, request in request_map.items()
            if case == "overall" or request["case"] == case
        ]
        gpu = [row for rid in rids for row in gpu_by_rid[rid]]
        expected = [row for rid in rids for row in teacher_by_rid[rid]]
        gpu_calls, teacher_calls = len(gpu), len(expected)
        gpu_bytes = sum(int(row["logical_bytes"]) for row in gpu)
        teacher_bytes = sum(int(row["logical_bytes"]) for row in expected)
        alignment.append(
            {
                "case": case,
                "requests": len(rids),
                "exact_requests": sum(bool(exact_by_rid[rid]) for rid in rids),
                "waves": len(
                    {
                        (
                            request_map[rid]["case"],
                            request_map[rid]["repeat"],
                            request_map[rid]["wave_index"],
                        )
                        for rid in rids
                    }
                ),
                "gpu_calls": gpu_calls,
                "teacher_calls": teacher_calls,
                "calls_absolute_error": abs(gpu_calls - teacher_calls),
                "gpu_logical_bytes": gpu_bytes,
                "teacher_logical_bytes": teacher_bytes,
                "logical_bytes_absolute_error": abs(gpu_bytes - teacher_bytes),
            }
        )
        gpu_call_bins, gpu_byte_bins = histogram_vectors(gpu, len(rids))
        teacher_call_bins, teacher_byte_bins = histogram_vectors(expected, len(rids))
        for index in range(12):
            histogram_rows.append(
                {
                    "case": case,
                    "bin_index": index,
                    "lower_bytes": BIN_EDGES_BYTES[index],
                    "upper_bytes": BIN_EDGES_BYTES[index + 1],
                    "gpu_calls_per_1000": gpu_call_bins[index],
                    "teacher_calls_per_1000": teacher_call_bins[index],
                    "calls_absolute_error": abs(gpu_call_bins[index] - teacher_call_bins[index]),
                    "gpu_logical_bytes_per_1000": gpu_byte_bins[index],
                    "teacher_logical_bytes_per_1000": teacher_byte_bins[index],
                    "logical_bytes_absolute_error": abs(
                        gpu_byte_bins[index] - teacher_byte_bins[index]
                    ),
                }
            )

    repeat_signatures: dict[str, list[list[tuple[int, int, int, int]]]] = defaultdict(list)
    for item in contract["gpu_sentinel_contract"]["synthetic_cases"]:
        case = item["name"]
        for repeat in range(int(item["repeats"])):
            case_requests = sorted(
                [
                    row
                    for row in requests
                    if row["case"] == case and int(row["repeat"]) == repeat
                ],
                key=lambda row: row["request_index"],
            )
            signature = []
            for request in case_requests:
                signature.append(
                    [
                        (
                            int(event.get("page_start", -1)),
                            int(event.get("page_end", -1)),
                            int(event.get("kv_page_count", -1)),
                            int(event.get("logical_bytes", -1)),
                        )
                        for event in gpu_by_rid[request["rid"]]
                    ]
                )
            repeat_signatures[case].append(signature)
    runtime_bytes = {int(row.get("kv_bytes_per_page", -1)) for row in matching}
    expected_bytes = int(model["derived"]["kv_bytes_per_page"])
    checks = {
        "no_unexpected_raw_events": not unexpected,
        "all_expected_rids_seen": all(gpu_by_rid[rid] for rid in expected_rids),
        "request_level_exact": all(exact_by_rid.values()),
        "aggregate_calls_exact": all(int(row["calls_absolute_error"]) == 0 for row in alignment),
        "aggregate_bytes_exact": all(
            int(row["logical_bytes_absolute_error"]) == 0 for row in alignment
        ),
        "histograms_exact": all(
            float(row["calls_absolute_error"]) == 0.0
            and float(row["logical_bytes_absolute_error"]) == 0.0
            for row in histogram_rows
        ),
        "synthetic_repeats_exact": all(
            all(signature == signatures[0] for signature in signatures[1:])
            for signatures in repeat_signatures.values()
        ),
        "runtime_kv_bytes_per_page_exact": runtime_bytes == {expected_bytes},
        "logical_bytes_formula_exact": all(
            int(row.get("logical_bytes", -1))
            == int(row.get("kv_page_count", -2)) * expected_bytes
            for row in matching
        ),
        "page_size_one": all(
            int(row.get("page_size_tokens", -1))
            == int(contract["measurement_contract"]["page_size_tokens"])
            for row in matching
        ),
        "mooncake_sender": all(row.get("backend") == "MooncakeKVSender" for row in matching),
        "state_payload_zero": all(
            int(row.get("state_logical_bytes", -1)) == 0 for row in matching
        ),
        "no_tensor_contents": all(
            row.get("raw_tensor_contents_saved") is False for row in matching
        ),
    }
    return alignment, histogram_rows, {
        "checks": checks,
        "expected_requests": len(requests),
        "exact_requests": sum(exact_by_rid.values()),
        "gpu_events": len(matching),
        "teacher_events": len(teacher),
        "runtime_kv_bytes_per_page": sorted(runtime_bytes),
        "repeat_signatures": {
            case: [
                hashlib.sha256(
                    json.dumps(signature, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                for signature in signatures
            ]
            for case, signatures in repeat_signatures.items()
        },
        "mismatch_examples": mismatch_examples,
        "unexpected_event_count": len(unexpected),
    }


def h0_analysis_row(example: dict[str, Any]) -> dict[str, Any]:
    calls_l1 = sum(abs(float(example[f"residual_calls_bin_{index:02d}"])) for index in range(12))
    bytes_l1 = sum(
        abs(float(example[f"residual_logical_bytes_bin_{index:02d}"]))
        for index in range(12)
    )
    return {
        "profile_id": example["profile_id"],
        "split_role": example["split_role"],
        "source": example["source"],
        "request_count": example["feature_profile_request_count"],
        "h0_total_calls_per_1000": example["h0_total_calls_per_1000"],
        "target_total_calls_per_1000": example["target_total_calls_per_1000"],
        "calls_12bin_l1": calls_l1,
        "h0_total_logical_bytes_per_1000": example["h0_total_logical_bytes_per_1000"],
        "target_total_logical_bytes_per_1000": example[
            "target_total_logical_bytes_per_1000"
        ],
        "logical_bytes_12bin_l1": bytes_l1,
    }


def build_dataset(
    *,
    contract: dict[str, Any],
    feature_contract: dict[str, Any],
    bundle: dict[str, Any],
    model: dict[str, Any],
) -> dict[str, Any]:
    kv_bytes = int(model["derived"]["kv_bytes_per_page"])
    development_examples = []
    target_rows = []
    profile_rows = []
    for item in bundle["development"]:
        profile = item["profile"]
        requests = [tuple(map(int, pair)) for pair in item["requests"]]
        example, target = profile_example_rows(
            profile=profile,
            requests=requests,
            contract=contract,
            feature_contract=feature_contract,
            kv_bytes_per_page=kv_bytes,
        )
        if target is None:
            raise RuntimeError("development target unexpectedly absent")
        profile_rows.append(profile)
        development_examples.append(example)
        target_rows.append(target)
    blind_rows = []
    for profile in bundle["blind_features"]:
        features, target = profile_example_rows(
            profile=profile,
            requests=None,
            contract=contract,
            feature_contract=feature_contract,
            kv_bytes_per_page=kv_bytes,
        )
        if target is not None or any(
            name.startswith(("target_", "residual_")) for name in features
        ):
            raise RuntimeError(f"blind target leakage: {profile['profile_id']}")
        blind_rows.append(features)
    expected = contract["acceptance_gates"]
    checks = {
        "development_examples": len(development_examples)
        == int(expected["development_example_rows"]),
        "development_targets": len(target_rows) == int(expected["development_target_rows"]),
        "blind_features": len(blind_rows) == int(expected["blind_feature_rows"]),
        "blind_targets_zero": int(expected["blind_target_rows"]) == 0
        and all(
            not any(name.startswith(("target_", "residual_")) for name in row)
            for row in blind_rows
        ),
        "roles_75_19": sum(
            row["split_role"] == "development_train" for row in development_examples
        )
        == 75
        and sum(
            row["split_role"] == "development_validation"
            for row in development_examples
        )
        == 19,
        "request_count_35524": sum(int(row["request_count"]) for row in target_rows)
        == int(contract["dataset_contract"]["development_full_requests"]),
        "ids_unique": len({row["profile_id"] for row in development_examples})
        == len(development_examples),
        "development_blind_disjoint": not {
            row["profile_id"] for row in development_examples
        }.intersection({row["profile_id"] for row in blind_rows}),
        "all_h0_present": all(
            all(f"h0_calls_bin_{index:02d}" in row for index in range(12))
            and all(f"h0_logical_bytes_bin_{index:02d}" in row for index in range(12))
            for row in development_examples + blind_rows
        ),
        "all_residuals_exact": all(
            abs(
                float(row[f"residual_calls_bin_{index:02d}"])
                - (
                    float(row[f"target_calls_bin_{index:02d}"])
                    - float(row[f"h0_calls_bin_{index:02d}"])
                )
            )
            < 1e-12
            and abs(
                float(row[f"residual_logical_bytes_bin_{index:02d}"])
                - (
                    float(row[f"target_logical_bytes_bin_{index:02d}"])
                    - float(row[f"h0_logical_bytes_bin_{index:02d}"])
                )
            )
            < 1e-6
            for row in development_examples
            for index in range(12)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError({"dataset_checks": checks})
    inventory = []
    for role in ("development_train", "development_validation", "blind_confirmation"):
        rows = (
            [row for row in development_examples if row["split_role"] == role]
            if role != "blind_confirmation"
            else blind_rows
        )
        inventory.append(
            {
                "split_role": role,
                "profiles": len(rows),
                "requests_from_low_dim_profile": sum(
                    int(row["feature_profile_request_count"]) for row in rows
                ),
                "hfull_target_generated": role != "blind_confirmation",
                "h0_generated": True,
            }
        )
    return {
        "checks": checks,
        "development_profiles": profile_rows,
        "development_examples": development_examples,
        "targets": target_rows,
        "blind_features": blind_rows,
        "h0_analysis": [h0_analysis_row(row) for row in development_examples],
        "inventory": inventory,
    }


def launch_and_run(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_json(HERE / "experiment.json")
    feature_contract = load_json(HERE / "feature_contract.json")
    head = require_expected_head(args.expected_workflow_commit)
    require_clean_before_run()
    verify_pinned_inputs(contract)
    output = (repo_root() / contract["result_dir"]).resolve()
    if output.exists():
        raise RuntimeError(f"formal result directory already exists: {output}")
    raw_dir = args.raw_dir.expanduser().resolve()
    if not raw_dir.is_dir() or any(raw_dir.iterdir()):
        raise RuntimeError(f"formal raw directory must exist and be empty: {raw_dir}")
    preflight = load_json(args.preflight_audit.expanduser().resolve())
    if preflight.get("status") != "PASS" or preflight.get("workflow_commit") != head:
        raise RuntimeError("preflight audit does not match W41")
    if Path(preflight["external_raw_dir"]).resolve() != raw_dir:
        raise RuntimeError("preflight raw directory mismatch")
    bundle, transfer = bundle_audit(contract, args.bundle_dir, head)
    if transfer["manifest"]["bundle_sha256"] != preflight["bundle"]["manifest"][
        "bundle_sha256"
    ]:
        raise RuntimeError("bundle changed after preflight")
    validate_bundle(contract, bundle)
    phase40_contract = load_json(
        repo_root() / contract["reused_phase40_contract"]["path"]
    )
    model = phase40_preflight_module().model_contract(
        args.model_path.resolve(), phase40_contract
    )
    if model["config_sha256"] != preflight["model_contract"]["config_sha256"]:
        raise RuntimeError("model changed after preflight")
    requests, teacher, case_inventory = sentinel_workload(
        contract, bundle, int(model["derived"]["kv_bytes_per_page"])
    )

    base_env = dict(os.environ)
    if base_env.get("HF_HUB_OFFLINE") != "1" or base_env.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("Phase41 formal execution must remain offline")
    for name in (
        "MC_FORCE_TCP",
        "MC_FORCE_MNNVL",
        "MC_INTRANODE_NVLINK",
        "SGLANG_MOONCAKE_CUSTOM_MEM_POOL",
        "SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB",
        "SGLANG_PD_COMM_PROFILE_DIR",
        "SGLANG_PD_COMM_PROFILE_RUN_ID",
        "SGLANG_PP_COMM_PROFILE_DIR",
        "SGLANG_PP_COMM_PROFILE_RUN_ID",
    ):
        base_env.pop(name, None)
    base_env["MOONCAKE_PROTOCOL"] = "rdma"
    base_env["WITH_NVIDIA_PEERMEM"] = "0"
    base_env["SGLANG_DISAGG_STAGING_BUFFER"] = "0"
    base_env["SGLANG_PD_BOOTSTRAP_BATCH_BARRIER"] = "1"

    profile_dir = raw_dir / "profile"
    log_dir = raw_dir / "server_logs"
    profile_dir.mkdir()
    log_dir.mkdir()
    commands = build_server_commands(
        contract=contract,
        model_path=args.model_path,
        ib_device=args.ib_device,
        prefill_port=args.prefill_port,
        decode_port=args.decode_port,
        router_port=args.router_port,
        bootstrap_port=args.bootstrap_port,
    )
    processes: list[subprocess.Popen] = []
    handles = []
    started_at = utc_now()
    wave_responses = 0
    try:
        prefill_env = dict(base_env)
        prefill_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_pair[0])
        prefill_env["SGLANG_PD_COMM_PROFILE_DIR"] = str(profile_dir)
        prefill_env["SGLANG_PD_COMM_PROFILE_RUN_ID"] = "phase41_gpu_sentinel"
        decode_env = dict(base_env)
        decode_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_pair[1])
        for name, env in (("prefill", prefill_env), ("decode", decode_env)):
            handle = (log_dir / f"{name}.log").open("w", encoding="utf-8")
            handles.append(handle)
            process = subprocess.Popen(
                commands[name],
                cwd=repo_root(),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append(process)
        wait_http(
            f"http://127.0.0.1:{args.prefill_port}/health",
            processes[0],
            args.startup_timeout,
        )
        wait_http(
            f"http://127.0.0.1:{args.decode_port}/health",
            processes[1],
            args.startup_timeout,
        )
        router_handle = (log_dir / "router.log").open("w", encoding="utf-8")
        handles.append(router_handle)
        router = subprocess.Popen(
            commands["router"],
            cwd=repo_root(),
            env=base_env,
            stdout=router_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append(router)
        wait_http(f"http://127.0.0.1:{args.router_port}/health", router, 120)

        wave_keys = []
        for row in requests:
            key = (row["case"], row["repeat"], row["wave_index"])
            if key not in wave_keys:
                wave_keys.append(key)
        for case, repeat, wave_index in wave_keys:
            wave = sorted(
                [
                    row
                    for row in requests
                    if row["case"] == case
                    and row["repeat"] == repeat
                    and row["wave_index"] == wave_index
                ],
                key=lambda row: row["wave_request_index"],
            )
            if len(wave) > int(contract["measurement_contract"]["wave_size"]):
                raise RuntimeError("wave exceeds frozen maximum")
            payload = {
                "input_ids": [
                    [int(contract["measurement_contract"]["input_token_id"])]
                    * int(row["prompt_tokens"])
                    for row in wave
                ],
                "rid": wave[0]["rid_prefix"],
                "sampling_params": {
                    "temperature": float(contract["measurement_contract"]["temperature"]),
                    "max_new_tokens": int(
                        contract["measurement_contract"]["max_new_tokens"]
                    ),
                    "ignore_eos": bool(contract["measurement_contract"]["ignore_eos"]),
                },
                "stream": False,
            }
            response = post_json(
                f"http://127.0.0.1:{args.router_port}/generate", payload
            )
            if isinstance(response, dict) and response.get("error"):
                raise RuntimeError(
                    {
                        "case": case,
                        "repeat": repeat,
                        "wave_index": wave_index,
                        "response": response,
                    }
                )
            wave_responses += 1

        expected_rids = {row["rid"] for row in requests}
        deadline = time.monotonic() + 300
        raw_events = []
        while time.monotonic() < deadline:
            raw_events = read_jsonl(sorted(profile_dir.glob("*.jsonl")))
            if expected_rids <= {row.get("rid") for row in raw_events}:
                break
            time.sleep(1)
        seen = {row.get("rid") for row in raw_events}
        if not expected_rids <= seen:
            raise RuntimeError({"missing_profile_rids": sorted(expected_rids - seen)[:100]})
    finally:
        terminate_processes(processes)
        for handle in handles:
            handle.close()

    raw_events = read_jsonl(sorted(profile_dir.glob("*.jsonl")))
    log_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(log_dir.glob("*.log"))
    )
    forbidden_errors = (
        "Failed to send kv chunk",
        "Failed to get kvcache from prefill instance",
        "remote mooncake session",
    )
    alignment, histograms, gpu_evidence = compare_gpu_teacher(
        contract=contract,
        model=model,
        requests=requests,
        teacher=teacher,
        raw_events=raw_events,
    )
    gpu_evidence["checks"].update(
        {
            "all_waves_returned": wave_responses
            == int(contract["gpu_sentinel_contract"]["expected_waves"]),
            "no_transfer_error_in_logs": not any(
                pattern in log_text for pattern in forbidden_errors
            ),
        }
    )
    if not all(gpu_evidence["checks"].values()):
        raise RuntimeError({"Phase41_GateB_GPU_SENTINEL_FAILED": gpu_evidence})

    # Gate C starts only after the exact GPU gate above has passed.
    dataset = build_dataset(
        contract=contract,
        feature_contract=feature_contract,
        bundle=bundle,
        model=model,
    )
    if not all(dataset["checks"].values()):
        raise RuntimeError({"Phase41_GateC_DATASET_FAILED": dataset["checks"]})

    output.mkdir(parents=True, exist_ok=False)
    (output / "contracts").mkdir()
    shutil.copy2(HERE / "experiment.json", output / "contracts/experiment.json")
    shutil.copy2(HERE / "feature_contract.json", output / "contracts/feature_contract.json")
    write_json(output / "contracts/model_contract.json", model)
    write_json(
        output / "contracts/dataset_contract.json",
        {
            "schema_version": "phase41-result-dataset-contract-v1",
            **contract["dataset_contract"],
            "bundle_sha256": transfer["manifest"]["bundle_sha256"],
            "blind_target_generated": False,
            "blind_target_rows": 0,
            "full_request_list_saved_in_result": False,
        },
    )
    write_json(
        output / "audit/input_freeze.json",
        {
            "workflow_commit": head,
            "workflow_parent_result_commit": contract["workflow_parent_result_commit"],
            "bundle_sha256": transfer["manifest"]["bundle_sha256"],
            "bundle_bytes": transfer["manifest"]["bundle_bytes"],
            "bundle_external": True,
            "pinned_inputs": preflight["pinned_inputs"],
            "source_inventory": bundle["source_inventory"],
        },
    )
    environment = environment_record()
    environment.update(preflight["environment"])
    environment["gpu_pair"] = list(args.gpu_pair)
    environment["selected_gpus"] = preflight["gpus"]["selected"]
    environment["gpu_topology_text"] = preflight["gpus"]["topology_text"]
    environment["ib"] = preflight["ib"]
    write_json(output / "audit/environment.json", environment)
    write_json(output / "audit/source_semantics.json", preflight["source_semantics"])
    write_json(
        output / "audit/server_launch.json",
        {
            "started_at_utc": started_at,
            "finished_at_utc": utc_now(),
            "commands": {
                name: redacted_command(command, args.model_path)
                for name, command in commands.items()
            },
            "model_path_redacted": True,
            "prefill_physical_gpu": args.gpu_pair[0],
            "decode_physical_gpu": args.gpu_pair[1],
            "ib_device": args.ib_device,
            "transport_environment": {
                "MOONCAKE_PROTOCOL": base_env.get("MOONCAKE_PROTOCOL"),
                "WITH_NVIDIA_PEERMEM": base_env.get("WITH_NVIDIA_PEERMEM"),
                "MC_FORCE_TCP": base_env.get("MC_FORCE_TCP"),
                "MC_FORCE_MNNVL": base_env.get("MC_FORCE_MNNVL"),
                "MC_INTRANODE_NVLINK": base_env.get("MC_INTRANODE_NVLINK"),
                "SGLANG_MOONCAKE_CUSTOM_MEM_POOL": base_env.get(
                    "SGLANG_MOONCAKE_CUSTOM_MEM_POOL"
                ),
            },
            "admission_environment": {
                "SGLANG_PD_BOOTSTRAP_BATCH_BARRIER": base_env.get(
                    "SGLANG_PD_BOOTSTRAP_BATCH_BARRIER"
                ),
                "SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB": base_env.get(
                    "SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB"
                ),
            },
            "wave_protocol": contract["measurement_contract"]["wave_partition"],
            "wave_size": contract["measurement_contract"]["wave_size"],
            "wave_responses": wave_responses,
        },
    )
    raw = raw_manifest(raw_dir, gpu_evidence["gpu_events"])
    write_json(output / "audit/raw_manifest.json", raw)
    write_json(
        output / "audit/dataset_build.json",
        {
            "schema_version": "phase41-dataset-build-audit-v1",
            "gate_b_gpu_pass_before_build": True,
            "checks": dataset["checks"],
            "development_profile_count": len(dataset["development_examples"]),
            "blind_profile_count": len(dataset["blind_features"]),
            "blind_target_generated": False,
            "full_requests_saved_in_git": False,
            "bundle_manifest_summary": {
                "workflow_commit": transfer["manifest"]["workflow_commit"],
                "bundle_sha256": transfer["manifest"]["bundle_sha256"],
                "bundle_bytes": transfer["manifest"]["bundle_bytes"],
                "development_reconstruction": transfer["manifest"][
                    "development_reconstruction"
                ],
                "blind_selection_audit": transfer["manifest"]["blind_selection_audit"],
            },
        },
    )
    write_csv(output / "analysis/gpu_teacher_alignment.csv", alignment)
    write_csv(output / "analysis/gpu_teacher_histograms_12bin.csv", histograms)
    write_csv(output / "analysis/gpu_case_inventory.csv", case_inventory)
    write_csv(output / "analysis/h0_vs_hfull.csv", dataset["h0_analysis"])
    write_csv(output / "analysis/dataset_inventory.csv", dataset["inventory"])
    write_csv_gz(
        output / "profiles/development_profiles.csv.gz", dataset["development_profiles"]
    )
    write_csv_gz(
        output / "dataset/pd_development_hfull_targets.csv.gz", dataset["targets"]
    )
    write_csv_gz(
        output / "dataset/pd_development_h0_residual_examples.csv.gz",
        dataset["development_examples"],
    )
    write_csv_gz(
        output / "dataset/pd_blind_target_free_features.csv.gz", dataset["blind_features"]
    )
    runtime_state = {
        "schema_version": "phase41-runtime-state-v1",
        "workflow_commit": head,
        "gates": {
            "GateA_CONTROL_BUNDLE": all(transfer["checks"].values()),
            "GateB_GPU_SENTINEL": all(gpu_evidence["checks"].values()),
            "GateC_CPU_DATASET": all(dataset["checks"].values()),
        },
        "checks": {**gpu_evidence["checks"], **dataset["checks"]},
        "counts": {
            "gpu_sentinel_requests": len(requests),
            "gpu_sentinel_waves": wave_responses,
            "gpu_logical_chunks": gpu_evidence["gpu_events"],
            "teacher_logical_chunks": gpu_evidence["teacher_events"],
            "gpu_exact_requests": gpu_evidence["exact_requests"],
            "development_profiles": len(dataset["development_examples"]),
            "development_full_requests": sum(
                int(row["request_count"]) for row in dataset["targets"]
            ),
            "development_target_rows": len(dataset["targets"]),
            "blind_feature_rows": len(dataset["blind_features"]),
            "blind_target_rows": 0,
            "external_raw_files": raw["file_count"],
            "external_raw_bytes": raw["bytes"],
        },
        "runtime_kv_bytes_per_page": gpu_evidence["runtime_kv_bytes_per_page"],
        "mismatch_examples": gpu_evidence["mismatch_examples"],
    }
    write_json(output / "audit/runtime_state.json", runtime_state)
    (output / "logs").mkdir()
    (output / "logs/runtime.log").write_text(
        f"Phase41 started={started_at} finished={utc_now()}\n"
        f"workflow_commit={head}\n"
        f"gateA_bundle_sha256={transfer['manifest']['bundle_sha256']}\n"
        f"gateB_requests={len(requests)} waves={wave_responses} gpu_chunks={gpu_evidence['gpu_events']} teacher_chunks={gpu_evidence['teacher_events']} exact_requests={gpu_evidence['exact_requests']}\n"
        f"gateC_development_profiles={len(dataset['development_examples'])} development_requests={runtime_state['counts']['development_full_requests']} blind_features={len(dataset['blind_features'])} blind_targets=0\n"
        f"raw_dir={raw_dir} raw_files={raw['file_count']} raw_bytes={raw['bytes']} raw_committed_to_git=false\n",
        encoding="utf-8",
    )
    from finalize import finalize

    return finalize(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-workflow-commit", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--gpu-pair", type=parse_gpu_pair, required=True)
    parser.add_argument("--ib-device", required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--preflight-audit", type=Path, required=True)
    parser.add_argument("--prefill-port", type=int, default=40000)
    parser.add_argument("--decode-port", type=int, default=40001)
    parser.add_argument("--router-port", type=int, default=40002)
    parser.add_argument("--bootstrap-port", type=int, default=40003)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    args = parser.parse_args()
    ports = {args.prefill_port, args.decode_port, args.router_port, args.bootstrap_port}
    if len(ports) != 4 or any(port <= 1024 or port > 65535 for port in ports):
        raise RuntimeError("Phase41 requires four distinct valid non-privileged ports")
    print(json.dumps(launch_and_run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
