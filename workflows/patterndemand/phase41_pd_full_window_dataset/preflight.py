#!/usr/bin/env python3
"""Phase41 remote bundle, source, model, GPU and transport preflight."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import (  # noqa: E402
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
from contracts import read_bundle, sentinel_workload, validate_bundle  # noqa: E402


def parse_gpu_pair(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):
        raise argparse.ArgumentTypeError("--gpu-pair must be PHYSICAL_ID,PHYSICAL_ID")
    pair = (int(parts[0]), int(parts[1]))
    if pair[0] == pair[1]:
        raise argparse.ArgumentTypeError("P and D must use distinct GPUs")
    return pair


def phase40_preflight_module() -> Any:
    path = repo_root() / "workflows/patterndemand/phase40_pure_pd_semantics_teacher/preflight.py"
    spec = importlib.util.spec_from_file_location("phase40_preflight_pinned", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned Phase40 preflight: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bundle_audit(
    contract: dict[str, Any], bundle_dir: Path, expected_head: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = repo_root().resolve()
    directory = bundle_dir.expanduser().resolve()
    if directory == root or root in directory.parents:
        raise RuntimeError("bundle must remain outside Git")
    manifest_path = directory / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"bundle manifest missing: {manifest_path}")
    manifest = load_json(manifest_path)
    bundle_path = directory / manifest.get("bundle_file", "")
    checks = {
        "manifest_schema": manifest.get("schema_version")
        == "phase41-external-bundle-manifest-v1",
        "manifest_workflow_commit": manifest.get("workflow_commit") == expected_head,
        "bundle_file_present": bundle_path.is_file(),
        "bundle_bytes": bundle_path.is_file()
        and bundle_path.stat().st_size == int(manifest.get("bundle_bytes", -1)),
        "bundle_sha256": bundle_path.is_file()
        and sha256(bundle_path) == manifest.get("bundle_sha256"),
        "bundle_not_in_git": manifest.get("git_policy", {}).get("bundle_committed_to_git")
        is False,
        "blind_requests_not_exported": manifest.get("git_policy", {}).get(
            "blind_complete_requests_exported"
        )
        is False,
    }
    if not all(checks.values()):
        raise RuntimeError({"bundle_manifest_checks": checks})
    bundle = read_bundle(bundle_path)
    validation = validate_bundle(contract, bundle)
    checks.update(
        {
            "bundle_workflow_commit": bundle.get("workflow_commit") == expected_head,
            "bundle_parent_result": bundle.get("workflow_parent_result_commit")
            == contract["workflow_parent_result_commit"],
            "feature_contract_exact": bundle.get("feature_contract")
            == load_json(HERE / "feature_contract.json"),
            "source_hashes_exact": all(
                row.get("exact") is True for row in bundle.get("source_inventory", [])
            )
            and len(bundle.get("source_inventory", []))
            == len(contract["raw_source_contract"]["files"]),
        }
    )
    if not all(checks.values()):
        raise RuntimeError({"bundle_content_checks": checks})
    return bundle, {
        "checks": checks,
        "validation": validation,
        "manifest": manifest,
        "bundle_path": str(bundle_path),
        "manifest_path": str(manifest_path),
    }


def run_checks(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_json(HERE / "experiment.json")
    head = require_expected_head(args.expected_workflow_commit)
    require_clean_before_run()
    expected_output = (repo_root() / contract["result_dir"]).resolve()
    if expected_output.exists():
        raise RuntimeError(f"formal result directory already exists: {expected_output}")
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("unset CUDA_VISIBLE_DEVICES before Phase41 preflight")
    offline_checks = {
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE") == "1",
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE") == "1",
    }
    if not all(offline_checks.values()):
        raise RuntimeError({"formal_execution_must_be_offline": offline_checks})
    expected_repo_python = str((repo_root() / "python").resolve())
    pythonpath_entries = [
        str(Path(row).resolve())
        for row in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if row
    ]
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
    bundle, transfer = bundle_audit(contract, args.bundle_dir, head)
    bundle_path = Path(transfer["bundle_path"])
    if raw_dir == bundle_path.parent or raw_dir in bundle_path.parents or bundle_path.parent in raw_dir.parents:
        raise RuntimeError("formal raw and transfer bundle directories must be disjoint")
    audit_output = args.audit_output.expanduser().resolve()
    root = repo_root().resolve()
    if audit_output == root or root in audit_output.parents:
        raise RuntimeError("preflight audit must remain outside Git")
    if audit_output == raw_dir or raw_dir in audit_output.parents:
        raise RuntimeError("preflight audit must not be written inside formal raw")
    if audit_output == bundle_path.parent or bundle_path.parent in audit_output.parents:
        raise RuntimeError("preflight audit must not overwrite the transfer bundle")
    if audit_output.exists():
        raise RuntimeError(f"preflight audit already exists: {audit_output}")

    pins = verify_pinned_inputs(contract)
    phase40_contract = load_json(
        repo_root() / contract["reused_phase40_contract"]["path"]
    )
    phase40_pins = verify_pinned_inputs(phase40_contract)
    p40 = phase40_preflight_module()
    semantics = p40.source_semantics_audit()
    model = p40.model_contract(args.model_path.resolve(), phase40_contract)
    gpus = p40.gpu_audit(args.gpu_pair)
    ib = p40.ib_audit(args.ib_device)
    gpu_names = [str(row.get("name", "")) for row in gpus["selected"]]
    if not all("B200" in name for name in gpu_names):
        raise RuntimeError({"Phase41_requires_B200_class_pair": gpu_names})

    router_spec = importlib.util.find_spec("sglang_router.launch_router")
    mooncake_spec = importlib.util.find_spec("mooncake.engine")
    sglang_spec = importlib.util.find_spec("sglang")
    flashinfer_spec = importlib.util.find_spec("flashinfer")
    if router_spec is None or mooncake_spec is None:
        raise RuntimeError("SGLang router or Mooncake is unavailable; fallback is forbidden")
    if sglang_spec is None or sglang_spec.origin is None:
        raise RuntimeError("sglang import is unavailable")
    if flashinfer_spec is None or flashinfer_spec.origin is None:
        raise RuntimeError("FlashInfer is unavailable")
    sglang_origin = str(Path(sglang_spec.origin).resolve())
    if Path(expected_repo_python) not in Path(sglang_origin).parents:
        raise RuntimeError({"sglang_not_loaded_from_repo_python": sglang_origin})
    import mooncake
    import torch
    from sglang.srt.utils import is_flashinfer_available

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Phase41 requires at least two visible CUDA GPUs")
    if not is_flashinfer_available():
        raise RuntimeError("SGLang reports FlashInfer unavailable")

    request_rows, teacher_rows, cases = sentinel_workload(
        contract, bundle, int(model["derived"]["kv_bytes_per_page"])
    )
    workload_checks = {
        "requests": len(request_rows)
        == int(contract["gpu_sentinel_contract"]["expected_requests"]),
        "waves": len({(row["case"], row["repeat"], row["wave_index"]) for row in request_rows})
        == int(contract["gpu_sentinel_contract"]["expected_waves"]),
        "request_ids_unique": len({row["rid"] for row in request_rows}) == len(request_rows),
        "teacher_nonempty": bool(teacher_rows),
        "wave_size_bounded": max(row["wave_request_index"] for row in request_rows) < int(
            contract["measurement_contract"]["wave_size"]
        ),
    }
    if not all(workload_checks.values()):
        raise RuntimeError({"sentinel_workload_checks": workload_checks})
    phase40_measurement = phase40_contract["measurement_contract"]
    semantic_match = {
        "chunk_tokens": contract["measurement_contract"]["chunked_prefill_tokens"]
        == phase40_measurement["chunked_prefill_tokens"],
        "page_size": contract["measurement_contract"]["page_size_tokens"]
        == phase40_measurement["page_size_tokens"],
        "max_running": contract["measurement_contract"]["max_running_requests"]
        == phase40_measurement["max_running_requests"],
        "attention": contract["measurement_contract"]["attention_backend"]
        == phase40_contract["backend_contract"]["inference_attention_backend"],
        "transfer": contract["measurement_contract"]["transfer_backend"]
        == phase40_contract["backend_contract"]["sglang_transfer_backend"],
        "transport": contract["measurement_contract"]["transport"]
        == phase40_contract["backend_contract"]["transport"],
    }
    if not all(semantic_match.values()):
        raise RuntimeError({"phase40_semantic_reuse_mismatch": semantic_match})
    return {
        "schema_version": "phase41-preflight-v1",
        "status": "PASS",
        "captured_at_utc": utc_now(),
        "workflow_commit": head,
        "expected_output_dir": str(expected_output.relative_to(repo_root())),
        "external_raw_dir": str(raw_dir),
        "bundle": transfer,
        "bundle_workload": {
            "checks": workload_checks,
            "case_inventory": cases,
            "requests": len(request_rows),
            "teacher_chunks": len(teacher_rows),
        },
        "model_contract": model,
        "source_semantics": semantics,
        "semantic_match_to_phase40": semantic_match,
        "pinned_inputs": pins,
        "phase40_pinned_inputs": phase40_pins,
        "gpu_pair": list(args.gpu_pair),
        "gpus": gpus,
        "ib": ib,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "mooncake_version": getattr(mooncake, "__version__", None),
            "formal_execution_offline": offline_checks,
            "container_image_env": {
                key: value
                for key, value in os.environ.items()
                if re.search(r"(IMAGE|CONTAINER|PYTORCH_VERSION)", key)
            },
            "python_source": {
                "PYTHONPATH_entries": pythonpath_entries,
                "expected_repo_python": expected_repo_python,
                "sglang_origin": sglang_origin,
                "repo_sglang_loaded": True,
                "sglang_router_launch_origin": router_spec.origin,
                "mooncake_engine_origin": mooncake_spec.origin,
                "flashinfer_origin": flashinfer_spec.origin,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-workflow-commit", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--gpu-pair", type=parse_gpu_pair, required=True)
    parser.add_argument("--ib-device", required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    result = run_checks(args)
    args.raw_dir.expanduser().resolve().mkdir(parents=True, exist_ok=False)
    write_json(args.audit_output.expanduser().resolve(), result)
    printable = dict(result)
    printable["gpus"] = {"selected": result["gpus"]["selected"]}
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
