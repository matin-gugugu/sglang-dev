#!/usr/bin/env python3
"""Phase40 read-only environment, model, GPU and source audit."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import (
    ensure_external_raw_dir,
    load_json,
    repo_root,
    require_clean_before_run,
    require_expected_head,
    sha256,
    utc_now,
    verify_pinned_inputs,
    write_json,
)


def parse_gpu_pair(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):
        raise argparse.ArgumentTypeError("--gpu-pair must be PHYSICAL_ID,PHYSICAL_ID")
    pair = tuple(int(part) for part in parts)
    if pair[0] == pair[1]:
        raise argparse.ArgumentTypeError("P and D must use distinct GPUs")
    return pair  # type: ignore[return-value]


def model_contract(model_path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if not model_path.is_dir():
        raise RuntimeError(f"model path is not a local directory: {model_path}")
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"missing local model config: {config_path}")
    config = load_json(config_path)
    required = contract["model_contract"]
    architectures = config.get("architectures") or []
    checks = {
        "model_type_qwen3": config.get("model_type") == required["required_model_type"],
        "architecture_qwen3_causal_lm": required["required_architecture"] in architectures,
        "non_mla": config.get("kv_lora_rank") is None,
        "torch_dtype_bfloat16": config.get("torch_dtype") == required["dtype"],
        "token_id_valid": int(contract["measurement_contract"]["input_token_id"]) < int(config.get("vocab_size", 0)),
    }
    numeric_names = (
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "num_hidden_layers",
    )
    for name in numeric_names:
        checks[f"config_{name}"] = isinstance(config.get(name), int) and int(config[name]) > 0
    required_structure = required["required_structure"]
    for name, expected in required_structure.items():
        actual = config.get(name)
        if name == "head_dim" and actual is None and config.get("hidden_size") and config.get("num_attention_heads"):
            actual = int(config["hidden_size"]) // int(config["num_attention_heads"])
        checks[f"required_{name}"] = int(actual or 0) == int(expected)
    if not all(checks.values()):
        raise RuntimeError({"model_contract_checks": checks})
    hidden_size = int(config["hidden_size"])
    attention_heads = int(config["num_attention_heads"])
    head_dim = int(config.get("head_dim") or hidden_size // attention_heads)
    kv_heads = int(config["num_key_value_heads"])
    layers = int(config["num_hidden_layers"])
    page_size = int(contract["measurement_contract"]["page_size_tokens"])
    dtype_bytes = 2
    kv_bytes_per_token = layers * 2 * kv_heads * head_dim * dtype_bytes
    weight_candidates = sorted(
        path.name
        for path in model_path.iterdir()
        if path.is_file() and path.suffix.lower() in {".safetensors", ".bin", ".pt", ".pth"}
    )
    index_candidates = sorted(path.name for path in model_path.glob("*.index.json"))
    if not weight_candidates and not index_candidates:
        raise RuntimeError("local model directory has no visible weight or weight-index files")
    official = contract["official_model_download"]
    expected_shards = official["weight_shards"]
    artifact_checks = {
        "config_sha256": sha256(config_path) == official["config_sha256"],
        "weight_index_present": (model_path / official["weight_index_file"]).is_file(),
        "weight_shard_names_exact": weight_candidates == sorted(row["name"] for row in expected_shards),
    }
    index_path = model_path / official["weight_index_file"]
    artifact_checks["weight_index_sha256"] = (
        index_path.is_file() and sha256(index_path) == official["weight_index_sha256"]
    )
    verified_shards = []
    for expected in expected_shards:
        path = model_path / expected["name"]
        actual_bytes = path.stat().st_size if path.is_file() else None
        actual_sha256 = sha256(path) if path.is_file() and actual_bytes == expected["bytes"] else None
        exact = actual_bytes == expected["bytes"] and actual_sha256 == expected["sha256"]
        artifact_checks[f"weight_{expected['name']}"] = exact
        verified_shards.append(
            {
                "name": expected["name"],
                "bytes": actual_bytes,
                "sha256": actual_sha256,
                "exact": exact,
            }
        )
    artifact_checks["weights_total_bytes"] = (
        sum(int(row["bytes"] or 0) for row in verified_shards) == official["weights_total_bytes"]
    )
    if not all(artifact_checks.values()):
        raise RuntimeError({"official_model_artifact_checks": artifact_checks})
    return {
        "schema_version": "phase40-official-model-contract-v2",
        "model_path": str(model_path),
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "model_type": config["model_type"],
        "architectures": architectures,
        "torch_dtype_declared": config.get("torch_dtype"),
        "structure": {
            "hidden_size": hidden_size,
            "num_attention_heads": attention_heads,
            "num_key_value_heads": kv_heads,
            "num_hidden_layers": layers,
            "head_dim": head_dim,
            "page_size_tokens": page_size,
            "dtype": "bfloat16",
            "dtype_bytes": dtype_bytes,
        },
        "derived": {
            "kv_bytes_per_token": kv_bytes_per_token,
            "kv_bytes_per_page": kv_bytes_per_token * page_size,
            "formula": "num_layers * 2(K,V) * num_kv_heads * head_dim * bf16_bytes * page_size",
        },
        "weight_inventory": {
            "weight_file_count": len(weight_candidates),
            "weight_index_count": len(index_candidates),
            "files_hashed_or_copied": True,
        },
        "official_source": {
            "repo_id": official["repo_id"],
            "revision": official["revision"],
            "source_url": official["source_url"],
            "config_sha256": official["config_sha256"],
            "weight_index_sha256": official["weight_index_sha256"],
            "verified_shards": verified_shards,
            "weights_total_bytes": official["weights_total_bytes"],
            "artifact_checks": artifact_checks,
        },
        "checks": checks,
    }


def source_semantics_audit() -> dict[str, bool]:
    root = repo_root()
    prefill = (root / "python/sglang/srt/disaggregation/prefill.py").read_text(encoding="utf-8")
    common = (root / "python/sglang/srt/disaggregation/common/conn.py").read_text(encoding="utf-8")
    mooncake = (root / "python/sglang/srt/disaggregation/mooncake/conn.py").read_text(encoding="utf-8")
    schedule = (root / "python/sglang/srt/managers/schedule_policy.py").read_text(encoding="utf-8")
    profiler = (root / "python/sglang/srt/disaggregation/pd_comm_profile.py").read_text(encoding="utf-8")
    io_struct = (root / "python/sglang/srt/managers/io_struct.py").read_text(encoding="utf-8")
    tokenizer_manager = (root / "python/sglang/srt/managers/tokenizer_manager.py").read_text(encoding="utf-8")
    scheduler = (root / "python/sglang/srt/managers/scheduler.py").read_text(encoding="utf-8")
    server_args = (root / "python/sglang/srt/server_args.py").read_text(encoding="utf-8")
    overrides = (root / "python/sglang/srt/arg_groups/overrides.py").read_text(encoding="utf-8")
    attention_registry = (root / "python/sglang/srt/layers/attention/attention_registry.py").read_text(encoding="utf-8")
    environ = (root / "python/sglang/srt/environ.py").read_text(encoding="utf-8")
    router = (root / "sgl-model-gateway/src/routers/http/pd_router.rs").read_text(encoding="utf-8")
    checks = {
        "prefill_sender_call": "req.disagg_kv_sender.send(page_indices, state_indices)" in prefill,
        "prefill_rid_attribution": "req.disagg_kv_sender.profile_rid = str(req.rid)" in prefill,
        "common_metric_formula": "self._transfer_num_kv_indices * self.kv_mgr.kv_item_lens_sum" in common,
        "common_profile_after_filter": "record_pd_send(" in common and "page_start=self.curr_idx - len(kv_indices)" in common,
        "mooncake_records_indices": "self._record_transfer_indices(kv_indices, state_indices)" in mooncake,
        "scheduler_global_chunk_continuation": "def add_chunked_req(self, req: Req)" in schedule,
        "scheduler_chunk_budget": "trunc_len = self.rem_chunk_tokens" in schedule,
        "profiler_sender_only": "decode receiver is not instrumented" in profiler,
        "profiler_no_tensor_contents": '"raw_tensor_contents_saved": False' in profiler,
        "router_detects_input_id_batch": "if let Some(InputIds::Batch(batches)) = &req.input_ids" in router,
        "router_injects_room_per_batch_item": "for _ in 0..n" in router and "Value::Array(rooms.into_iter().map(Value::from).collect())" in router,
        "generate_request_accepts_scalar_or_list_rid": "rid: Optional[Union[str, List[str]]]" in io_struct,
        "scalar_rid_expands_by_batch_index": 'new_rids = [f"{self.rid}_{i}" for i in range(num)]' in io_struct,
        "pretokenized_batch_uses_batch_dispatch": "(not self._batch_has_text(batch_size, requests))" in tokenizer_manager and "self._send_batch_request(tokenized_objs)" in tokenizer_manager,
        "batch_tokenized_request_single_dispatch": "batch_req = BatchTokenizedGenerateReqInput(batch=tokenized_objs)" in tokenizer_manager and "self._dispatch_to_scheduler(batch_req)" in tokenizer_manager,
        "scheduler_handles_batch_before_admission": "def handle_batch_generate_request(" in scheduler and "for tokenized_req in recv_req:" in scheduler,
        "bootstrap_barrier_env_declared": "SGLANG_PD_BOOTSTRAP_BATCH_BARRIER = EnvBool(False)" in environ,
        "bootstrap_barrier_waits_for_all": "if envs.SGLANG_PD_BOOTSTRAP_BATCH_BARRIER.get():" in prefill and "if any(poll == KVPoll.Bootstrapping for poll in polls):" in prefill,
        "bootstrap_barrier_requires_whole_batch_capacity": "bootstrap batch barrier requires metadata capacity for the whole batch" in prefill,
        "attention_backend_cli_contract": "attention_backend: A[" in server_args and "choices=ATTENTION_BACKEND_CHOICES" in server_args,
        "flashinfer_attention_registered": '@register_attention_backend("flashinfer")' in attention_registry,
        "trtllm_mha_rejects_page_one": "TensorRT-LLM MHA only supports page_size of 16, 32 or 64" in overrides,
        "flashinfer_not_in_page_snap_clause": 'view.attention_backend == "trtllm_mha"' in overrides and 'view.attention_backend == "flashinfer"' not in overrides[overrides.index("def _mla_backend_page_constraints"):overrides.index("def _mla_kv_cache_dtype_checks")],
    }
    if not all(checks.values()):
        raise RuntimeError({"source_semantics_checks": checks})
    return checks


def ib_audit(device: str) -> dict[str, Any]:
    root = Path("/sys/class/infiniband") / device
    if not root.is_dir():
        raise RuntimeError(f"IB device not found: {device}")
    ports = []
    for port_dir in sorted((root / "ports").iterdir()):
        state = (port_dir / "state").read_text(encoding="utf-8").strip()
        rate = (port_dir / "rate").read_text(encoding="utf-8").strip()
        ports.append({"port": port_dir.name, "state": state, "rate": rate})
    if not any("ACTIVE" in row["state"] for row in ports):
        raise RuntimeError({"ib_device_not_active": device, "ports": ports})
    return {"device": device, "ports": ports}


def gpu_audit(pair: tuple[int, int]) -> dict[str, Any]:
    query = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total", "--format=csv,noheader,nounits"],
        text=True,
    )
    rows = []
    for line in query.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5:
            rows.append({"index": int(parts[0]), "uuid": parts[1], "name": parts[2], "driver": parts[3], "memory_mib": int(parts[4])})
    selected = [row for row in rows if row["index"] in pair]
    if len(selected) != 2:
        raise RuntimeError({"gpu_pair_not_available": pair, "inventory": rows})
    topo = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
    return {"inventory": rows, "selected": selected, "topology_text": topo}


def run_checks(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_json(HERE / "experiment.json")
    head = require_expected_head(args.expected_workflow_commit)
    require_clean_before_run()
    expected_output = (repo_root() / contract["result_dir"]).resolve()
    if expected_output.exists():
        raise RuntimeError(f"formal result directory already exists: {expected_output}")
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("unset CUDA_VISIBLE_DEVICES before Phase40 preflight")
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
    raw_dir = ensure_external_raw_dir(args.raw_dir)
    if args.audit_output.resolve() == repo_root() or repo_root() in args.audit_output.resolve().parents:
        raise RuntimeError("preflight audit must remain outside Git")
    if args.audit_output.resolve() == raw_dir or raw_dir in args.audit_output.resolve().parents:
        raise RuntimeError("preflight audit must not be written inside the raw directory")
    pins = verify_pinned_inputs(contract)
    semantics = source_semantics_audit()
    router_spec = importlib.util.find_spec("sglang_router.launch_router")
    if router_spec is None:
        raise RuntimeError("sglang_router.launch_router is unavailable")
    mooncake_engine_spec = importlib.util.find_spec("mooncake.engine")
    if mooncake_engine_spec is None:
        raise RuntimeError("mooncake.engine is unavailable; backend fallback is forbidden")
    sglang_spec = importlib.util.find_spec("sglang")
    if sglang_spec is None or sglang_spec.origin is None:
        raise RuntimeError("sglang import is unavailable")
    flashinfer_spec = importlib.util.find_spec("flashinfer")
    if flashinfer_spec is None or flashinfer_spec.origin is None:
        raise RuntimeError("flashinfer is unavailable; the frozen page-size-1 attention backend cannot be used")
    sglang_origin = str(Path(sglang_spec.origin).resolve())
    if Path(expected_repo_python) not in Path(sglang_origin).parents:
        raise RuntimeError({"sglang_not_loaded_from_repo_python": sglang_origin})
    import mooncake
    import torch
    from sglang.srt.utils import is_flashinfer_available

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Phase40 requires at least two visible CUDA GPUs")
    if not is_flashinfer_available():
        raise RuntimeError("SGLang reports FlashInfer unavailable")
    model = model_contract(args.model_path.resolve(), contract)
    gpus = gpu_audit(args.gpu_pair)
    ib = ib_audit(args.ib_device)
    return {
        "schema_version": "phase40-preflight-v2",
        "status": "PASS",
        "captured_at_utc": utc_now(),
        "workflow_commit": head,
        "expected_output_dir": str(expected_output.relative_to(repo_root())),
        "external_raw_dir": str(raw_dir),
        "gpu_pair": list(args.gpu_pair),
        "ib": ib,
        "model_contract": model,
        "source_semantics": semantics,
        "attention_backend": {
            "required": contract["backend_contract"]["inference_attention_backend"],
            "fallback_permitted": contract["backend_contract"]["inference_attention_backend_fallback_permitted"],
            "module_origin": str(Path(flashinfer_spec.origin).resolve()),
            "sglang_reports_available": True,
            "required_page_size_tokens": contract["measurement_contract"]["page_size_tokens"],
        },
        "admission_contract": {
            "batched_input_ids_single_dispatch": True,
            "bootstrap_batch_barrier_required": True,
            "bootstrap_batch_barrier_env": contract["measurement_contract"]["bootstrap_batch_barrier_env"],
            "optimistic_prefill_retries": contract["measurement_contract"]["optimistic_prefill_retries"],
        },
        "pinned_inputs": pins,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "mooncake_version": getattr(mooncake, "__version__", None),
            "container_image_env": {key: value for key, value in os.environ.items() if re.search(r"(IMAGE|CONTAINER|PYTORCH_VERSION)", key)},
            "formal_execution_offline": offline_checks,
            "python_source": {
                "PYTHONPATH_entries": pythonpath_entries,
                "expected_repo_python": expected_repo_python,
                "sglang_origin": sglang_origin,
                "repo_sglang_loaded": True,
                "sglang_router_launch_origin": router_spec.origin,
                "mooncake_engine_origin": mooncake_engine_spec.origin,
            },
        },
        "gpus": gpus,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-workflow-commit", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--gpu-pair", type=parse_gpu_pair, required=True)
    parser.add_argument("--ib-device", required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    result = run_checks(args)
    args.raw_dir.resolve().mkdir(parents=True, exist_ok=False)
    write_json(args.audit_output.resolve(), result)
    printable = dict(result)
    printable["gpus"] = {"selected": result["gpus"]["selected"]}
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
