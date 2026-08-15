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
    return {
        "schema_version": "phase40-local-model-contract-v1",
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
        "weight_inventory_only": {
            "weight_file_count": len(weight_candidates),
            "weight_index_count": len(index_candidates),
            "files_hashed_or_copied": False,
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
    raw_dir = ensure_external_raw_dir(args.raw_dir)
    if args.audit_output.resolve() == repo_root() or repo_root() in args.audit_output.resolve().parents:
        raise RuntimeError("preflight audit must remain outside Git")
    if args.audit_output.resolve() == raw_dir or raw_dir in args.audit_output.resolve().parents:
        raise RuntimeError("preflight audit must not be written inside the raw directory")
    pins = verify_pinned_inputs(contract)
    semantics = source_semantics_audit()
    if importlib.util.find_spec("sglang_router.launch_router") is None:
        raise RuntimeError("sglang_router.launch_router is unavailable")
    if importlib.util.find_spec("mooncake.engine") is None:
        raise RuntimeError("mooncake.engine is unavailable; backend fallback is forbidden")
    import mooncake
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Phase40 requires at least two visible CUDA GPUs")
    model = model_contract(args.model_path.resolve(), contract)
    gpus = gpu_audit(args.gpu_pair)
    ib = ib_audit(args.ib_device)
    return {
        "schema_version": "phase40-preflight-v1",
        "status": "PASS",
        "captured_at_utc": utc_now(),
        "workflow_commit": head,
        "expected_output_dir": str(expected_output.relative_to(repo_root())),
        "external_raw_dir": str(raw_dir),
        "gpu_pair": list(args.gpu_pair),
        "ib": ib,
        "model_contract": model,
        "source_semantics": semantics,
        "pinned_inputs": pins,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "mooncake_version": getattr(mooncake, "__version__", None),
            "container_image_env": {key: value for key, value in os.environ.items() if re.search(r"(IMAGE|CONTAINER|PYTORCH_VERSION)", key)},
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
