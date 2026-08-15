#!/usr/bin/env python3
"""Launch frozen P1-D1 servers, execute Phase40 waves and build compact results."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import (
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
from contracts import (
    build_teacher,
    histogram_rows,
    read_jsonl,
    wave_rid_prefix,
    workload_rows,
    write_csv,
)
from preflight import model_contract, parse_gpu_pair


def wait_http(url: str, process: subprocess.Popen, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    last_error: str | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"process exited before readiness: rc={process.returncode}, url={url}")
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if 200 <= response.status < 300:
                    return
        except Exception as error:  # readiness polling intentionally broad
            last_error = str(error)
        time.sleep(2)
    raise RuntimeError(f"server readiness timeout: {url}, last_error={last_error}")


def post_json(url: str, payload: dict[str, Any], timeout: int = 900) -> Any:
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
        raise RuntimeError(f"request failed: status={error.code}, body={detail[:2000]}") from error
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
    return ["$MODEL_PATH" if value == str(model_path) else value for value in command]


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
        "schema_version": "phase40-external-raw-manifest-v1",
        "external_raw_dir": str(raw_dir),
        "raw_committed_to_git": False,
        "file_count": len(files),
        "bytes": sum(row["bytes"] for row in files),
        "profiler_event_count": event_count,
        "files": files,
    }


def compare_events(
    contract: dict[str, Any],
    model: dict[str, Any],
    raw_events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    requests = workload_rows(contract)
    request_map = {row["rid"]: row for row in requests}
    filtered = []
    for event in raw_events:
        rid = event.get("rid")
        if rid not in request_map:
            continue
        row = dict(event)
        row.update({key: request_map[rid][key] for key in ("scenario", "repeat", "request_index", "prompt_tokens")})
        filtered.append(row)
    filtered.sort(key=lambda row: int(row["sequence"]))
    teacher = build_teacher(contract, model)
    gpu_by_rid: dict[str, list[dict[str, Any]]] = {rid: [] for rid in request_map}
    teacher_by_rid: dict[str, list[dict[str, Any]]] = {rid: [] for rid in request_map}
    for row in filtered:
        gpu_by_rid[row["rid"]].append(row)
    for row in teacher:
        teacher_by_rid[row["rid"]].append(row)

    exact_by_rid = {}
    mismatch_examples = []
    fields = ("page_start", "page_end", "kv_page_count", "logical_bytes")
    for rid in sorted(request_map):
        gpu = gpu_by_rid[rid]
        expected = teacher_by_rid[rid]
        exact = len(gpu) == len(expected) and all(
            all(int(actual[field]) == int(wanted[field]) for field in fields)
            for actual, wanted in zip(gpu, expected)
        )
        exact_by_rid[rid] = exact
        if not exact and len(mismatch_examples) < 10:
            mismatch_examples.append(
                {
                    "rid": rid,
                    "gpu": [{field: row.get(field) for field in fields} for row in gpu],
                    "teacher": [{field: row.get(field) for field in fields} for row in expected],
                }
            )

    scenario_names = [row["name"] for row in contract["workload_scenarios"]]
    alignment = []
    for scenario in [*scenario_names, "overall"]:
        rids = [rid for rid, row in request_map.items() if scenario == "overall" or row["scenario"] == scenario]
        gpu = [row for rid in rids for row in gpu_by_rid[rid]]
        expected = [row for rid in rids for row in teacher_by_rid[rid]]
        gpu_calls = len(gpu)
        teacher_calls = len(expected)
        gpu_bytes = sum(int(row["logical_bytes"]) for row in gpu)
        teacher_bytes = sum(int(row["logical_bytes"]) for row in expected)
        alignment.append(
            {
                "scenario": scenario,
                "requests": len(rids),
                "exact_requests": sum(bool(exact_by_rid[rid]) for rid in rids),
                "gpu_calls": gpu_calls,
                "teacher_calls": teacher_calls,
                "calls_absolute_error": abs(gpu_calls - teacher_calls),
                "gpu_logical_bytes": gpu_bytes,
                "teacher_logical_bytes": teacher_bytes,
                "logical_bytes_absolute_error": abs(gpu_bytes - teacher_bytes),
            }
        )

    gpu_hist = histogram_rows(filtered, requests, "gpu")
    teacher_hist = histogram_rows(teacher, requests, "teacher")
    gpu_hist_map = {(row["scenario"], row["bin_index"]): row for row in gpu_hist}
    histogram = []
    for expected in teacher_hist:
        actual = gpu_hist_map[(expected["scenario"], expected["bin_index"])]
        histogram.append(
            {
                "scenario": expected["scenario"],
                "bin_index": expected["bin_index"],
                "lower_bytes": expected["lower_bytes"],
                "upper_bytes": expected["upper_bytes"],
                "gpu_calls_per_1000_requests": actual["calls_per_1000_requests"],
                "teacher_calls_per_1000_requests": expected["calls_per_1000_requests"],
                "calls_absolute_error": abs(actual["calls_per_1000_requests"] - expected["calls_per_1000_requests"]),
                "gpu_logical_bytes_per_1000_requests": actual["logical_bytes_per_1000_requests"],
                "teacher_logical_bytes_per_1000_requests": expected["logical_bytes_per_1000_requests"],
                "logical_bytes_absolute_error": abs(actual["logical_bytes_per_1000_requests"] - expected["logical_bytes_per_1000_requests"]),
            }
        )

    repeat_signatures = {}
    for scenario in scenario_names:
        signatures = []
        for repeat in range(int(contract["measurement_contract"]["independent_repeats"])):
            rows = [row for row in filtered if row["scenario"] == scenario and int(row["repeat"]) == repeat]
            signature: dict[int, tuple[int, int]] = {}
            from contracts import bin_index

            for row in rows:
                index = bin_index(int(row["logical_bytes"]))
                calls, logical_bytes = signature.get(index, (0, 0))
                signature[index] = (calls + 1, logical_bytes + int(row["logical_bytes"]))
            signatures.append(sorted(signature.items()))
        repeat_signatures[scenario] = {
            "exact": all(signature == signatures[0] for signature in signatures[1:]),
            "signatures": signatures,
        }

    runtime_bytes_per_page = sorted({int(row["kv_bytes_per_page"]) for row in filtered})
    checks = {
        "request_count_45": len(requests) == int(contract["acceptance_gates"]["expected_requests"]),
        "all_requests_have_gpu_events": all(gpu_by_rid.values()),
        "all_requests_exact": all(exact_by_rid.values()),
        "alignment_rows_6": len(alignment) == int(contract["expected_alignment_rows"]),
        "aggregate_calls_exact": all(int(row["calls_absolute_error"]) == 0 for row in alignment),
        "aggregate_bytes_exact": all(int(row["logical_bytes_absolute_error"]) == 0 for row in alignment),
        "histogram_rows_72": len(histogram) == int(contract["expected_histogram_rows"]),
        "histogram_calls_exact": all(float(row["calls_absolute_error"]) == 0.0 for row in histogram),
        "histogram_bytes_exact": all(float(row["logical_bytes_absolute_error"]) == 0.0 for row in histogram),
        "repeat_histograms_exact": all(row["exact"] for row in repeat_signatures.values()),
        "backend_mooncake_sender_only": all(row.get("backend") == "MooncakeKVSender" for row in filtered),
        "page_size_one": all(int(row.get("page_size_tokens", 0)) == 1 for row in filtered),
        "runtime_formula_exact": runtime_bytes_per_page == [int(model["derived"]["kv_bytes_per_page"])],
        "state_payload_zero": all(int(row.get("state_logical_bytes", -1)) == 0 for row in filtered),
        "no_tensor_contents": all(row.get("raw_tensor_contents_saved") is False for row in filtered),
    }
    evidence = {
        "checks": checks,
        "requests": len(requests),
        "gpu_events": len(filtered),
        "teacher_events": len(teacher),
        "exact_requests": sum(exact_by_rid.values()),
        "runtime_kv_bytes_per_page": runtime_bytes_per_page,
        "repeat_signatures": repeat_signatures,
        "mismatch_examples": mismatch_examples,
    }
    return alignment, histogram, evidence


def launch_and_run(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_json(HERE / "experiment.json")
    head = require_expected_head(args.expected_workflow_commit)
    require_clean_before_run()
    verify_pinned_inputs(contract)
    output = (repo_root() / contract["result_dir"]).resolve()
    if output.exists():
        raise RuntimeError(f"formal result directory exists: {output}")
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("unset CUDA_VISIBLE_DEVICES before Phase40 run")
    offline_checks = {
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE") == "1",
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE") == "1",
    }
    if not all(offline_checks.values()):
        raise RuntimeError({"formal_execution_must_be_offline": offline_checks})
    expected_repo_python = str((repo_root() / "python").resolve())
    pythonpath_entries = [str(Path(row).resolve()) for row in os.environ.get("PYTHONPATH", "").split(os.pathsep) if row]
    if not pythonpath_entries or pythonpath_entries[0] != expected_repo_python:
        raise RuntimeError(
            {
                "repo_python_must_be_first_on_PYTHONPATH": {
                    "expected": expected_repo_python,
                    "actual": pythonpath_entries,
                }
            }
        )
    raw_dir = args.raw_dir.resolve()
    if not raw_dir.is_dir() or any(raw_dir.iterdir()):
        raise RuntimeError(f"raw directory must exist and be empty at run start: {raw_dir}")
    preflight = load_json(args.preflight_audit.resolve())
    current_model = model_contract(args.model_path.resolve(), contract)
    preflight_checks = {
        "status": preflight.get("status") == "PASS",
        "workflow_commit": preflight.get("workflow_commit") == head,
        "raw_dir": Path(preflight.get("external_raw_dir", "")).resolve() == raw_dir,
        "gpu_pair": preflight.get("gpu_pair") == list(args.gpu_pair),
        "ib_device": preflight.get("ib", {}).get("device") == args.ib_device,
        "model_path": preflight.get("model_contract", {}).get("model_path") == str(args.model_path.resolve()),
        "model_config_sha": preflight.get("model_contract", {}).get("config_sha256") == current_model["config_sha256"],
    }
    if not all(preflight_checks.values()):
        raise RuntimeError({"preflight_mismatch": preflight_checks})

    profile_dir = raw_dir / "profile"
    log_dir = raw_dir / "server_logs"
    profile_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    base_env = dict(os.environ)
    base_env.pop("MC_FORCE_TCP", None)
    base_env["MOONCAKE_PROTOCOL"] = "rdma"
    base_env["SGLANG_DISAGG_STAGING_BUFFER"] = "0"
    base_env.pop("SGLANG_PP_COMM_PROFILE_DIR", None)
    base_env.pop("SGLANG_PP_COMM_PROFILE_RUN_ID", None)

    common_server = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(args.model_path.resolve()),
        "--tp",
        "1",
        "--pp-size",
        "1",
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "auto",
        "--page-size",
        "1",
        "--schedule-policy",
        "fcfs",
        "--chunked-prefill-size",
        "4096",
        "--max-prefill-tokens",
        "4096",
        "--max-running-requests",
        "64",
        "--context-length",
        str(contract["measurement_contract"]["context_length"]),
        "--mem-fraction-static",
        str(contract["measurement_contract"]["mem_fraction_static"]),
        "--disable-radix-cache",
        "--disable-overlap-schedule",
        "--disaggregation-transfer-backend",
        "mooncake",
        "--disaggregation-ib-device",
        args.ib_device,
        "--disaggregation-bootstrap-port",
        str(args.bootstrap_port),
        "--host",
        "127.0.0.1",
    ]
    prefill_command = [*common_server, "--disaggregation-mode", "prefill", "--port", str(args.prefill_port)]
    decode_command = [*common_server, "--disaggregation-mode", "decode", "--port", str(args.decode_port)]
    router_command = [
        sys.executable,
        "-m",
        "sglang_router.launch_router",
        "--pd-disaggregation",
        "--prefill",
        f"http://127.0.0.1:{args.prefill_port}",
        "--decode",
        f"http://127.0.0.1:{args.decode_port}",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.router_port),
    ]
    commands = {
        "prefill": redacted_command(prefill_command, args.model_path.resolve()),
        "decode": redacted_command(decode_command, args.model_path.resolve()),
        "router": router_command,
    }

    processes: list[subprocess.Popen] = []
    handles = []
    started_at = utc_now()
    try:
        prefill_env = dict(base_env)
        prefill_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_pair[0])
        prefill_env["SGLANG_PD_COMM_PROFILE_DIR"] = str(profile_dir)
        prefill_env["SGLANG_PD_COMM_PROFILE_RUN_ID"] = "phase40"
        decode_env = dict(base_env)
        decode_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_pair[1])
        for name, command, env in (
            ("prefill", prefill_command, prefill_env),
            ("decode", decode_command, decode_env),
        ):
            handle = (log_dir / f"{name}.log").open("w", encoding="utf-8")
            handles.append(handle)
            process = subprocess.Popen(command, cwd=repo_root(), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
            processes.append(process)
        wait_http(f"http://127.0.0.1:{args.prefill_port}/health", processes[0], args.startup_timeout)
        wait_http(f"http://127.0.0.1:{args.decode_port}/health", processes[1], args.startup_timeout)
        router_handle = (log_dir / "router.log").open("w", encoding="utf-8")
        handles.append(router_handle)
        router = subprocess.Popen(router_command, cwd=repo_root(), env=base_env, stdout=router_handle, stderr=subprocess.STDOUT, text=True)
        processes.append(router)
        wait_http(f"http://127.0.0.1:{args.router_port}/health", router, 120)

        requests = workload_rows(contract)
        scenario_order = [row["name"] for row in contract["workload_scenarios"]]
        for scenario in scenario_order:
            for repeat in range(int(contract["measurement_contract"]["independent_repeats"])):
                wave = [row for row in requests if row["scenario"] == scenario and row["repeat"] == repeat]
                wave.sort(key=lambda row: row["request_index"])
                payload = {
                    "input_ids": [[int(row["input_token_id"])] * int(row["prompt_tokens"]) for row in wave],
                    # The Rust router accepts one scalar rid. SGLang expands it
                    # deterministically to <prefix>_<batch_index> without
                    # changing the input_ids batch or its admission order.
                    "rid": wave_rid_prefix(scenario, repeat),
                    "sampling_params": {
                        "temperature": 0.0,
                        "max_new_tokens": int(contract["measurement_contract"]["max_new_tokens"]),
                        "ignore_eos": True,
                    },
                    "stream": False,
                }
                response = post_json(f"http://127.0.0.1:{args.router_port}/generate", payload)
                if isinstance(response, dict) and response.get("error"):
                    raise RuntimeError({"scenario": scenario, "repeat": repeat, "response": response})

        expected_rids = {row["rid"] for row in requests}
        deadline = time.monotonic() + 120
        raw_events = []
        while time.monotonic() < deadline:
            paths = sorted(profile_dir.glob("*.jsonl"))
            raw_events = read_jsonl(paths) if paths else []
            seen = {row.get("rid") for row in raw_events}
            if expected_rids <= seen:
                break
            time.sleep(1)
        if not expected_rids <= {row.get("rid") for row in raw_events}:
            raise RuntimeError({"missing_profile_rids": sorted(expected_rids - {row.get("rid") for row in raw_events})})
    finally:
        terminate_processes(processes)
        for handle in handles:
            handle.close()

    all_raw_events = read_jsonl(sorted(profile_dir.glob("*.jsonl")))
    alignment, histograms, evidence = compare_events(contract, current_model, all_raw_events)
    if not all(evidence["checks"].values()):
        raise RuntimeError({"phase40_alignment_failed": evidence})

    output.mkdir(parents=True)
    (output / "contracts").mkdir()
    shutil.copy2(HERE / "experiment.json", output / "contracts/experiment.json")
    write_json(output / "contracts/model_contract.json", current_model)
    write_json(
        output / "contracts/workload_contract.json",
        {
            "schema_version": "phase40-frozen-workload-v1",
            "request_order": workload_rows(contract),
            "full_request_list_usage": "offline teacher and representative GPU audit only",
        },
    )
    write_json(
        output / "audit/input_freeze.json",
        {
            "workflow_commit": head,
            "workflow_parent_result_commit": contract["workflow_parent_result_commit"],
            "pinned_inputs": preflight["pinned_inputs"],
            "model_config_sha256": current_model["config_sha256"],
            "preflight_checks": preflight_checks,
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
            "commands": commands,
            "model_path_redacted": True,
            "prefill_physical_gpu": args.gpu_pair[0],
            "decode_physical_gpu": args.gpu_pair[1],
            "ib_device": args.ib_device,
            "transport": "rdma",
            "batch_rid_protocol": "one scalar wave prefix expanded by SGLang to <prefix>_<batch_index>",
        },
    )
    manifest = raw_manifest(raw_dir, evidence["gpu_events"])
    write_json(output / "audit/raw_manifest.json", manifest)
    write_csv(output / "analysis/gpu_teacher_alignment.csv", alignment)
    write_csv(output / "analysis/histograms_12bin.csv", histograms)
    invariants = [
        {"invariant": "requests", "expected": 45, "actual": evidence["requests"], "exact": evidence["requests"] == 45},
        {"invariant": "request_level_alignment", "expected": 45, "actual": evidence["exact_requests"], "exact": evidence["exact_requests"] == 45},
        {"invariant": "kv_bytes_per_page", "expected": current_model["derived"]["kv_bytes_per_page"], "actual": json.dumps(evidence["runtime_kv_bytes_per_page"]), "exact": evidence["checks"]["runtime_formula_exact"]},
        {"invariant": "state_payload_bytes", "expected": 0, "actual": 0, "exact": evidence["checks"]["state_payload_zero"]},
        {"invariant": "repeat_histograms", "expected": "exact", "actual": "exact", "exact": evidence["checks"]["repeat_histograms_exact"]},
    ]
    write_csv(output / "analysis/formula_invariants.csv", invariants)
    runtime_state = {
        "schema_version": "phase40-runtime-state-v1",
        "workflow_commit": head,
        "checks": evidence["checks"],
        "counts": {
            "scenarios": len(contract["workload_scenarios"]),
            "repeats_per_scenario": contract["measurement_contract"]["independent_repeats"],
            "requests": evidence["requests"],
            "gpu_logical_chunks": evidence["gpu_events"],
            "teacher_logical_chunks": evidence["teacher_events"],
            "exact_requests": evidence["exact_requests"],
            "histogram_rows": len(histograms),
            "external_raw_files": manifest["file_count"],
            "external_raw_bytes": manifest["bytes"],
        },
        "runtime_kv_bytes_per_page": evidence["runtime_kv_bytes_per_page"],
        "repeat_signatures": evidence["repeat_signatures"],
        "mismatch_examples": evidence["mismatch_examples"],
    }
    write_json(output / "audit/runtime_state.json", runtime_state)
    (output / "logs").mkdir(parents=True)
    (output / "logs/runtime.log").write_text(
        f"Phase40 started={started_at} finished={utc_now()}\n"
        f"workflow_commit={head}\nrequests={evidence['requests']} gpu_chunks={evidence['gpu_events']} teacher_chunks={evidence['teacher_events']} exact_requests={evidence['exact_requests']}\n"
        f"raw_dir={raw_dir} raw_files={manifest['file_count']} raw_bytes={manifest['bytes']} raw_committed_to_git=false\n",
        encoding="utf-8",
    )
    from finalize import finalize

    summary = finalize(output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-workflow-commit", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--gpu-pair", type=parse_gpu_pair, required=True)
    parser.add_argument("--ib-device", required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--preflight-audit", type=Path, required=True)
    parser.add_argument("--prefill-port", type=int, default=39000)
    parser.add_argument("--decode-port", type=int, default=39001)
    parser.add_argument("--router-port", type=int, default=39002)
    parser.add_argument("--bootstrap-port", type=int, default=39003)
    parser.add_argument("--startup-timeout", type=int, default=900)
    args = parser.parse_args()
    ports = {args.prefill_port, args.decode_port, args.router_port, args.bootstrap_port}
    if len(ports) != 4 or any(port <= 1024 or port > 65535 for port in ports):
        raise RuntimeError("Phase40 requires four distinct non-privileged valid ports")
    print(json.dumps(launch_and_run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
