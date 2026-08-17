#!/usr/bin/env python3
"""Sequentially run the five Phase47 P1-D1 GPU/teacher validations."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

HERE = Path(__file__).resolve().parent
P40 = HERE.parent / "phase40_pure_pd_semantics_teacher"
sys.path.insert(0, str(HERE.parent))
from common import (  # noqa: E402
    environment_record,
    load_json,
    refresh_manifest,
    repo_root,
    require_clean_before_run,
    require_expected_head,
    utc_now,
    validate_result_tree,
    verify_pinned_inputs,
    write_json,
)
from contracts import (  # noqa: E402
    add_model_id,
    inspect_model,
    load_model_map,
    model_specs,
    runtime_contract,
    write_csv,
)
from preflight import parse_gpu_pair  # noqa: E402


def load_phase40_run():
    """Load the SHA-pinned low-level server runner with its own contracts module."""
    saved_contracts = sys.modules.pop("contracts", None)
    saved_preflight = sys.modules.pop("preflight", None)
    sys.path.insert(0, str(P40))
    try:
        spec = importlib.util.spec_from_file_location("phase47_phase40_run", P40 / "run.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load pinned Phase40 run library")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(P40))
        if saved_contracts is not None:
            sys.modules["contracts"] = saved_contracts
        else:
            sys.modules.pop("contracts", None)
        if saved_preflight is not None:
            sys.modules["preflight"] = saved_preflight
        else:
            sys.modules.pop("preflight", None)
    return module


def pin_bfloat16_kv_cache(p40: Any) -> None:
    """Replace Phase40's safe Qwen auto setting with Phase47's explicit cross-model dtype."""
    original = p40.build_server_commands

    def build_server_commands(**kwargs):
        commands = original(**kwargs)
        for name in ("prefill", "decode"):
            index = commands[name].index("--kv-cache-dtype")
            commands[name][index + 1] = "bfloat16"
        return commands

    p40.build_server_commands = build_server_commands


def validate_smoke_events_generic(
    contract: dict[str, Any], model: dict[str, Any], raw_events: list[dict[str, Any]]
) -> dict[str, Any]:
    smoke = contract["compatibility_smoke_contract"]
    transport = smoke["transport_request"]
    transport_rid = f"{transport['rid_prefix']}_0"
    probe = smoke["admission_probe"]
    page = int(smoke["expected_page_size_tokens"])
    expected_segments: dict[str, list[tuple[int, int]]] = {
        transport_rid: [(0, int(smoke["expected_kv_page_count"]))]
    }
    admission_rids = []
    for repeat in range(int(probe["repeats"])):
        for request_index, segments in enumerate(probe["expected_segments_by_request_index"]):
            rid = f"{probe['rid_prefix_base']}{repeat}_{request_index}"
            admission_rids.append(rid)
            expected_segments[rid] = [(int(segment[0]), int(segment[1])) for segment in segments]
    expected_rids = set(expected_segments)
    events = [row for row in raw_events if row.get("rid") in expected_rids]
    by_rid = {rid: [] for rid in expected_rids}
    for row in sorted(events, key=lambda item: int(item.get("sequence", -1))):
        by_rid[row["rid"]].append(row)
    actual_segments = {
        rid: [(int(row.get("page_start", -1)), int(row.get("page_end", -1))) for row in rows]
        for rid, rows in by_rid.items()
    }
    expected_bytes = int(model["derived"]["kv_bytes_per_page"])
    transport_events = by_rid[transport_rid]
    signatures = [
        [actual_segments[f"{probe['rid_prefix_base']}{repeat}_{index}"] for index in range(len(probe["prompt_tokens"]))]
        for repeat in range(int(probe["repeats"]))
    ]
    checks = {
        "exactly_one_transport_sender_chunk": len(transport_events) == 1,
        "sender_chunks_total_exact": len(events) == int(smoke["expected_sender_chunks_total"]),
        "no_unexpected_profile_records": len(raw_events) == len(events),
        "mooncake_sender": all(row.get("backend") == "MooncakeKVSender" for row in events),
        "page_size_frozen": all(int(row.get("page_size_tokens", -1)) == page for row in events),
        "transport_page_count_exact": all(int(row.get("kv_page_count", -1)) == int(smoke["expected_kv_page_count"]) for row in transport_events),
        "bytes_per_page_exact": all(int(row.get("kv_bytes_per_page", -1)) == expected_bytes for row in events),
        "logical_bytes_formula_exact": all(int(row.get("logical_bytes", -1)) == int(row.get("kv_page_count", -2)) * expected_bytes for row in events),
        "admission_segments_exact": all(actual_segments[rid] == expected_segments[rid] for rid in admission_rids),
        "admission_repeats_exact": all(row == signatures[0] for row in signatures[1:]),
        "state_payload_zero": all(int(row.get("state_logical_bytes", -1)) == 0 for row in events),
        "no_tensor_contents": all(row.get("raw_tensor_contents_saved") is False for row in events),
    }
    return {
        "transport_expected_rid": transport_rid,
        "admission_expected_segments": {rid: [list(segment) for segment in expected_segments[rid]] for rid in admission_rids},
        "profile_records_total": len(raw_events),
        "matching_sender_chunks": len(events),
        "transport_sender_chunks": len(transport_events),
        "admission_sender_chunks": sum(len(by_rid[rid]) for rid in admission_rids),
        "admission_signatures": signatures,
        "checks": checks,
        "observed": [{key: row.get(key) for key in ("rid", "backend", "page_start", "page_end", "page_size_tokens", "kv_page_count", "kv_bytes_per_page", "logical_bytes", "state_logical_bytes")} for row in events],
    }


def formal_model_run(
    *,
    p40: Any,
    args: argparse.Namespace,
    spec: dict[str, Any],
    model_path: Path,
    model: dict[str, Any],
    contract: dict[str, Any],
    raw_dir: Path,
    smoke_dir: Path,
) -> dict[str, Any]:
    runtime = runtime_contract(contract, spec)
    raw_dir.mkdir(parents=True, exist_ok=False)
    smoke_args = SimpleNamespace(
        smoke_dir=smoke_dir,
        model_path=model_path,
        ib_device=args.ib_device,
        gpu_pair=args.gpu_pair,
        smoke_prefill_port=args.smoke_prefill_port,
        smoke_decode_port=args.smoke_decode_port,
        smoke_router_port=args.smoke_router_port,
        smoke_bootstrap_port=args.smoke_bootstrap_port,
        startup_timeout=args.startup_timeout,
    )
    base_env = dict(os.environ)
    for name in ("MC_FORCE_TCP", "MC_FORCE_MNNVL", "MC_INTRANODE_NVLINK", "SGLANG_MOONCAKE_CUSTOM_MEM_POOL", "SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB", "SGLANG_PD_COMM_PROFILE_DIR", "SGLANG_PD_COMM_PROFILE_RUN_ID", "SGLANG_PP_COMM_PROFILE_DIR", "SGLANG_PP_COMM_PROFILE_RUN_ID"):
        base_env.pop(name, None)
    base_env.update(
        {
            "MOONCAKE_PROTOCOL": "rdma",
            "WITH_NVIDIA_PEERMEM": "0",
            "SGLANG_DISAGG_STAGING_BUFFER": "0",
            "SGLANG_PD_BOOTSTRAP_BATCH_BARRIER": "1",
        }
    )
    smoke = p40.run_compatibility_smoke(smoke_args, runtime, model, base_env, raw_dir)
    profile_dir = raw_dir / "profile"
    logs_dir = raw_dir / "server_logs"
    profile_dir.mkdir()
    logs_dir.mkdir()
    commands = p40.build_server_commands(
        contract=runtime,
        model_path=model_path,
        ib_device=args.ib_device,
        prefill_port=args.prefill_port,
        decode_port=args.decode_port,
        router_port=args.router_port,
        bootstrap_port=args.bootstrap_port,
    )
    processes: list[subprocess.Popen] = []
    handles = []
    started = utc_now()
    try:
        prefill_env = dict(base_env)
        prefill_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_pair[0])
        prefill_env["SGLANG_PD_COMM_PROFILE_DIR"] = str(profile_dir)
        prefill_env["SGLANG_PD_COMM_PROFILE_RUN_ID"] = f"phase47_{spec['model_id']}"
        decode_env = dict(base_env)
        decode_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_pair[1])
        for name, env in (("prefill", prefill_env), ("decode", decode_env)):
            handle = (logs_dir / f"{name}.log").open("w", encoding="utf-8")
            handles.append(handle)
            process = subprocess.Popen(commands[name], cwd=repo_root(), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
            processes.append(process)
        p40.wait_http(f"http://127.0.0.1:{args.prefill_port}/health", processes[0], args.startup_timeout)
        p40.wait_http(f"http://127.0.0.1:{args.decode_port}/health", processes[1], args.startup_timeout)
        handle = (logs_dir / "router.log").open("w", encoding="utf-8")
        handles.append(handle)
        router = subprocess.Popen(commands["router"], cwd=repo_root(), env=base_env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        processes.append(router)
        p40.wait_http(f"http://127.0.0.1:{args.router_port}/health", router, 120)
        requests = p40.workload_rows(runtime)
        for scenario in [row["name"] for row in runtime["workload_scenarios"]]:
            for repeat in range(3):
                wave = sorted([row for row in requests if row["scenario"] == scenario and int(row["repeat"]) == repeat], key=lambda row: row["request_index"])
                response = p40.post_json(
                    f"http://127.0.0.1:{args.router_port}/generate",
                    {
                        "input_ids": [[int(row["input_token_id"])] * int(row["prompt_tokens"]) for row in wave],
                        "rid": p40.wave_rid_prefix(scenario, repeat),
                        "sampling_params": {"temperature": 0.0, "max_new_tokens": 2, "ignore_eos": True},
                        "stream": False,
                    },
                )
                if isinstance(response, dict) and response.get("error"):
                    raise RuntimeError({"model_id": spec["model_id"], "scenario": scenario, "repeat": repeat, "response": response})
        expected_rids = {row["rid"] for row in requests}
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            paths = sorted(profile_dir.glob("*.jsonl"))
            events = p40.read_jsonl(paths) if paths else []
            if expected_rids <= {row.get("rid") for row in events}:
                break
            time.sleep(1)
        else:
            raise RuntimeError({"model_id": spec["model_id"], "missing_profile_rids": sorted(expected_rids - {row.get('rid') for row in events})})
    finally:
        p40.terminate_processes(processes)
        for handle in handles:
            handle.close()
    raw_events = p40.read_jsonl(sorted(profile_dir.glob("*.jsonl")))
    alignment, histograms, evidence = p40.compare_events(runtime, model, raw_events)
    evidence["checks"].pop("page_size_one", None)
    expected_page = int(spec["page_size_tokens"])
    formal_events = [row for row in raw_events if row.get("rid") in {item["rid"] for item in p40.workload_rows(runtime)}]
    evidence["checks"]["page_size_frozen"] = all(int(row.get("page_size_tokens", -1)) == expected_page for row in formal_events)
    evidence["checks"]["no_unexpected_profile_records"] = len(formal_events) == len(raw_events)
    log_text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in sorted(logs_dir.glob("*.log")))
    evidence["checks"]["no_transfer_error_in_logs"] = not any(value in log_text for value in ("Failed to send kv chunk", "Failed to get kvcache from prefill instance", "remote mooncake session"))
    if not all(evidence["checks"].values()):
        raise RuntimeError({"phase47_model_alignment_failed": spec["model_id"], "evidence": evidence})
    return {
        "model_id": spec["model_id"],
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "alignment": add_model_id(alignment, spec["model_id"]),
        "histograms": add_model_id(histograms, spec["model_id"]),
        "evidence": evidence,
        "smoke": {"model_id": spec["model_id"], **smoke},
        "raw_manifest": {"model_id": spec["model_id"], **p40.raw_manifest(raw_dir, evidence["gpu_events"])},
        "server": {
            "model_id": spec["model_id"],
            "attention_backend": spec["attention_backend"],
            "page_size_tokens": expected_page,
            "gpu_pair": list(args.gpu_pair),
            "ib_device": args.ib_device,
            "commands": {name: p40.redacted_command(command, model_path) for name, command in commands.items()},
            "transport_environment": {name: base_env.get(name) for name in ("MOONCAKE_PROTOCOL", "WITH_NVIDIA_PEERMEM", "SGLANG_DISAGG_STAGING_BUFFER", "MC_FORCE_TCP", "MC_FORCE_MNNVL", "MC_INTRANODE_NVLINK", "SGLANG_MOONCAKE_CUSTOM_MEM_POOL")},
            "admission_environment": {"SGLANG_PD_BOOTSTRAP_BATCH_BARRIER": base_env.get("SGLANG_PD_BOOTSTRAP_BATCH_BARRIER"), "SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB": base_env.get("SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB")},
        },
    }


def finalize(output: Path, *, head: str, contract: dict[str, Any], preflight: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    (output / "contracts").mkdir()
    shutil.copy2(HERE / "experiment.json", output / "contracts/experiment.json")
    shutil.copy2(HERE / "models.json", output / "contracts/models.json")
    inventories = []
    for audit in preflight["model_audits"]:
        row = dict(audit)
        row.pop("model_path", None)
        inventories.append(row)
    write_json(output / "contracts/model_inventory.json", {"schema_version": "phase47-model-inventory-v1", "models": inventories})
    write_json(output / "audit/input_freeze.json", {"workflow_commit": head, "workflow_parent_result_commit": contract["workflow_parent_result_commit"], "pinned_inputs": preflight["pinned_inputs"]})
    environment = environment_record()
    environment.update(preflight["environment"])
    environment["gpu_pair"] = preflight["gpu_pair"]
    environment["selected_gpus"] = preflight["gpus"]["selected"]
    environment["gpu_topology_text"] = preflight["gpus"]["topology_text"]
    environment["ib"] = preflight["ib"]
    write_json(output / "audit/environment.json", environment)
    write_json(output / "audit/source_semantics.json", preflight["source_semantics"])
    write_json(output / "audit/compatibility_smoke.json", {"schema_version": "phase47-five-model-smoke-v1", "models": [row["smoke"] for row in results]})
    write_json(output / "audit/server_launch.json", {"schema_version": "phase47-five-model-launch-v1", "models_run_sequentially": True, "models": [row["server"] for row in results]})
    write_json(output / "audit/raw_manifests.json", {"schema_version": "phase47-external-raw-manifests-v1", "raw_committed_to_git": False, "models": [row["raw_manifest"] for row in results]})
    alignments = [item for row in results for item in row["alignment"]]
    histograms = [item for row in results for item in row["histograms"]]
    write_csv(output / "analysis/gpu_teacher_alignment.csv", alignments)
    write_csv(output / "analysis/histograms_12bin.csv", histograms)
    invariants = []
    model_summaries = []
    for row, audit in zip(results, inventories):
        evidence = row["evidence"]
        invariants.append({"model_id": row["model_id"], "expected_kv_bytes_per_page": audit["derived"]["kv_bytes_per_page"], "runtime_kv_bytes_per_page": json.dumps(evidence["runtime_kv_bytes_per_page"]), "exact": evidence["checks"]["runtime_formula_exact"]})
        model_summaries.append({"model_id": row["model_id"], "status": "PASS", "attention_backend": audit["attention_backend"], "page_size_tokens": audit["structure"]["page_size_tokens"], "requests": evidence["requests"], "exact_requests": evidence["exact_requests"], "gpu_logical_chunks": evidence["gpu_events"], "teacher_logical_chunks": evidence["teacher_events"], "kv_bytes_per_page": audit["derived"]["kv_bytes_per_page"], "checks": evidence["checks"]})
    write_csv(output / "analysis/formula_invariants.csv", invariants)
    runtime = {"schema_version": "phase47-runtime-v1", "workflow_commit": head, "models": model_summaries, "counts": {"models": len(results), "requests": sum(row["requests"] for row in model_summaries), "exact_requests": sum(row["exact_requests"] for row in model_summaries), "alignment_rows": len(alignments), "histogram_rows": len(histograms)}, "all_checks_pass": all(all(row["checks"].values()) for row in model_summaries)}
    write_json(output / "audit/runtime_state.json", runtime)
    summary = {"schema_version": "phase47-pd-five-model-teacher-validation-result-v1", "status": "PASS", "completed_at_utc": utc_now(), "workflow_commit": head, "objective": contract["objective"], "counts": runtime["counts"], "models": model_summaries, "training_performed": False, "checkpoint_loaded": False, "physical_curve_measured": False, "scheduler_or_placement_evaluated": False, "evidence_boundary": {"proved": "the other five frozen models exactly match the scheduler-faithful pure-PD logical teacher at the registered GPU policy points", "combined_with": "Phase40/41 provide the Qwen3-8B member of the frozen six-model roster", "not_proved": "six-model low-dimensional predictor quality, physical RDMA cost, placement, latency or online scheduling"}}
    write_json(output / "summary.json", summary)
    (output / "README.md").write_text(
        "# Phase47：纯PD其余五模型teacher语义GPU验证\n\n"
        f"最终状态：`PASS`。五个模型顺序复用同一对GPU，共`{runtime['counts']['requests']}`个请求，逐请求精确匹配`{runtime['counts']['exact_requests']}`个；"
        "calls、logical bytes与12-bin直方图误差均为0。DeepSeek-V2-Lite固定为TRTLLM MLA/page64，其余四模型固定为FlashInfer/page1，均使用Mooncake/RDMA和P1-D1。\n\n"
        "本阶段不训练DNN、不测通信时间、不做placement。模型权重、HF凭证、profiler JSONL和完整服务日志均保存在Git外。结合Phase40/41，现已为冻结六模型阵容建立代表性的纯PD Hfull teacher GPU语义证据；下一阶段才生成六模型完整窗口数据并训练。\n",
        encoding="utf-8",
    )
    (output / "logs").mkdir()
    (output / "logs/runtime.log").write_text("\n".join(f"model={row['model_id']} requests={row['evidence']['requests']} exact={row['evidence']['exact_requests']} gpu_chunks={row['evidence']['gpu_events']} teacher_chunks={row['evidence']['teacher_events']}" for row in results) + "\n", encoding="utf-8")
    tree = validate_result_tree(output)
    if not tree["ok"]:
        raise RuntimeError({"forbidden_result_assets": tree["violations"]})
    (output / "DONE").write_text("PASS\n", encoding="utf-8")
    refresh_manifest(output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-workflow-commit", required=True)
    parser.add_argument("--model-map", type=Path, required=True)
    parser.add_argument("--gpu-pair", type=parse_gpu_pair, required=True)
    parser.add_argument("--ib-device", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--preflight-audit", type=Path, required=True)
    parser.add_argument("--prefill-port", type=int, default=43000)
    parser.add_argument("--decode-port", type=int, default=43001)
    parser.add_argument("--router-port", type=int, default=43002)
    parser.add_argument("--bootstrap-port", type=int, default=43003)
    parser.add_argument("--smoke-prefill-port", type=int, default=43100)
    parser.add_argument("--smoke-decode-port", type=int, default=43101)
    parser.add_argument("--smoke-router-port", type=int, default=43102)
    parser.add_argument("--smoke-bootstrap-port", type=int, default=43103)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    args = parser.parse_args()
    ports = {args.prefill_port, args.decode_port, args.router_port, args.bootstrap_port, args.smoke_prefill_port, args.smoke_decode_port, args.smoke_router_port, args.smoke_bootstrap_port}
    if len(ports) != 8 or any(port <= 1024 or port > 65535 for port in ports):
        raise RuntimeError("eight distinct valid ports are required")
    contract = load_json(HERE / "experiment.json")
    head = require_expected_head(args.expected_workflow_commit)
    require_clean_before_run()
    verify_pinned_inputs(contract)
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("unset CUDA_VISIBLE_DEVICES before Phase47 run")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("Phase47 formal run must be offline")
    output = (repo_root() / contract["result_dir"]).resolve()
    if output.exists():
        raise RuntimeError(f"formal result exists: {output}")
    preflight = load_json(args.preflight_audit.expanduser().resolve())
    raw_root = args.raw_root.expanduser().resolve()
    smoke_root = args.smoke_root.expanduser().resolve()
    preflight_checks = {"status": preflight.get("status") == "PASS", "workflow": preflight.get("workflow_commit") == head, "gpu_pair": preflight.get("gpu_pair") == list(args.gpu_pair), "ib": preflight.get("ib", {}).get("device") == args.ib_device, "raw_root": Path(preflight.get("external_raw_root", "")).resolve() == raw_root, "smoke_root": Path(preflight.get("external_smoke_root", "")).resolve() == smoke_root}
    if not all(preflight_checks.values()) or not raw_root.is_dir() or any(raw_root.iterdir()) or not smoke_root.is_dir() or any(smoke_root.iterdir()):
        raise RuntimeError({"preflight_or_external_roots": preflight_checks})
    mapping = load_model_map(args.model_map.expanduser().resolve())
    preflight_by_id = {row["model_id"]: row for row in preflight["model_audits"]}
    models = {}
    for spec in model_specs():
        current = inspect_model(spec["model_id"], mapping[spec["model_id"]], hash_weights=False)
        frozen = preflight_by_id.get(spec["model_id"], {})
        current_files = [(row["name"], row["bytes"]) for row in current["artifact_inventory"]]
        frozen_files = [(row["name"], row["bytes"]) for row in frozen.get("artifact_inventory", [])]
        if current["config_sha256"] != frozen.get("config_sha256") or current["source_marker_sha256"] != frozen.get("source_marker_sha256") or current["artifact_bytes"] != frozen.get("artifact_bytes") or current_files != frozen_files:
            raise RuntimeError({"model_changed_after_preflight": spec["model_id"]})
        models[spec["model_id"]] = current
    p40 = load_phase40_run()
    p40.validate_smoke_events = validate_smoke_events_generic
    pin_bfloat16_kv_cache(p40)
    results = []
    for spec in model_specs():
        model_id = spec["model_id"]
        results.append(formal_model_run(p40=p40, args=args, spec=spec, model_path=mapping[model_id], model=models[model_id], contract=contract, raw_dir=raw_root / model_id, smoke_dir=smoke_root / model_id))
    print(json.dumps(finalize(output, head=head, contract=contract, preflight=preflight, results=results), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
