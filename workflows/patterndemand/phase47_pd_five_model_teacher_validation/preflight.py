#!/usr/bin/env python3
"""Read-only Phase47 source/model/GPU/IB audit; creates only external empty roots."""

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
P40 = HERE.parent / "phase40_pure_pd_semantics_teacher"
sys.path.insert(0, str(HERE.parent))
from common import (  # noqa: E402
    ensure_external_raw_dir,
    load_json,
    repo_root,
    require_clean_before_run,
    require_expected_head,
    utc_now,
    verify_pinned_inputs,
    write_json,
)
from contracts import inspect_model, load_model_map, model_specs  # noqa: E402


def load_phase40_preflight():
    spec = importlib.util.spec_from_file_location("phase47_phase40_preflight", P40 / "preflight.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned Phase40 preflight library")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_gpu_pair(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):
        raise argparse.ArgumentTypeError("--gpu-pair must be PHYSICAL_ID,PHYSICAL_ID")
    pair = (int(parts[0]), int(parts[1]))
    if pair[0] == pair[1]:
        raise argparse.ArgumentTypeError("P and D GPUs must be distinct")
    return pair


def _external_file(path: Path, label: str) -> Path:
    value = path.expanduser().resolve()
    root = repo_root().resolve()
    if value == root or root in value.parents:
        raise RuntimeError(f"{label} must remain outside Git")
    return value


def source_additions() -> dict[str, bool]:
    registry = (repo_root() / "python/sglang/srt/layers/attention/attention_registry.py").read_text()
    overrides = (repo_root() / "python/sglang/srt/arg_groups/overrides.py").read_text()
    checks = {
        "flashmla_registered": '@register_attention_backend("flashmla")' in registry,
        "flashmla_page64_enforced": "FlashMLA only supports a page_size of 64" in overrides,
        "flashmla_sets_page64": 'view.attention_backend == "flashmla"' in overrides
        and "page_size = 64" in overrides,
    }
    if not all(checks.values()):
        raise RuntimeError({"phase47_source_additions": checks})
    return checks


def run_checks(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_json(HERE / "experiment.json")
    head = require_expected_head(args.expected_workflow_commit)
    require_clean_before_run()
    output = (repo_root() / contract["result_dir"]).resolve()
    if output.exists():
        raise RuntimeError(f"formal result already exists: {output}")
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("unset CUDA_VISIBLE_DEVICES before Phase47 preflight")
    offline = {
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE") == "1",
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE") == "1",
    }
    if not all(offline.values()):
        raise RuntimeError({"formal_execution_must_be_offline": offline})
    expected_python = str((repo_root() / "python").resolve())
    pythonpath = [str(Path(row).resolve()) for row in os.environ.get("PYTHONPATH", "").split(os.pathsep) if row]
    if not pythonpath or pythonpath[0] != expected_python:
        raise RuntimeError({"repo_python_must_be_first": {"expected": expected_python, "actual": pythonpath}})
    raw_root = ensure_external_raw_dir(args.raw_root)
    smoke_root = ensure_external_raw_dir(args.smoke_root)
    if raw_root == smoke_root or raw_root in smoke_root.parents or smoke_root in raw_root.parents:
        raise RuntimeError("raw and smoke roots must be disjoint")
    audit_output = _external_file(args.audit_output, "preflight audit")
    model_map_file = _external_file(args.model_map, "model map")
    if audit_output in {raw_root, smoke_root} or raw_root in audit_output.parents or smoke_root in audit_output.parents:
        raise RuntimeError("audit output must be outside raw/smoke roots")
    if raw_root.exists() or smoke_root.exists():
        raise RuntimeError("raw and smoke roots must not exist before preflight")
    pins = verify_pinned_inputs(contract)
    p40 = load_phase40_preflight()
    semantics = p40.source_semantics_audit()
    semantics.update(source_additions())
    model_map = load_model_map(model_map_file)
    model_audits = [inspect_model(spec["model_id"], model_map[spec["model_id"]], hash_weights=True) for spec in model_specs()]
    module_checks = {
        "sglang_router": importlib.util.find_spec("sglang_router.launch_router") is not None,
        "mooncake": importlib.util.find_spec("mooncake.engine") is not None,
        "flashinfer": importlib.util.find_spec("flashinfer") is not None,
        "sgl_kernel_flash_mla": importlib.util.find_spec("sgl_kernel.flash_mla") is not None,
        "repo_flashmla_backend": importlib.util.find_spec("sglang.srt.layers.attention.flashmla_backend") is not None,
    }
    if not all(module_checks.values()):
        raise RuntimeError({"required_modules": module_checks})
    import mooncake
    import torch
    import sglang
    from sglang.srt.utils import is_flashinfer_available

    sglang_origin = str(Path(sglang.__file__).resolve())
    runtime_checks = {
        "repo_sglang": Path(expected_python) in Path(sglang_origin).parents,
        "cuda": torch.cuda.is_available(),
        "two_visible_gpus": torch.cuda.device_count() >= 2,
        "flashinfer_reported": is_flashinfer_available(),
    }
    if not all(runtime_checks.values()):
        raise RuntimeError({"runtime_checks": runtime_checks})
    gpus = p40.gpu_audit(args.gpu_pair)
    ib = p40.ib_audit(args.ib_device)
    raw_root.mkdir(parents=True, exist_ok=False)
    smoke_root.mkdir(parents=True, exist_ok=False)
    result = {
        "schema_version": "phase47-preflight-v1",
        "status": "PASS",
        "captured_at_utc": utc_now(),
        "workflow_commit": head,
        "expected_result_dir": contract["result_dir"],
        "gpu_pair": list(args.gpu_pair),
        "ib": ib,
        "external_raw_root": str(raw_root),
        "external_smoke_root": str(smoke_root),
        "model_map_sha256_recorded_externally": True,
        "model_audits": model_audits,
        "pinned_inputs": pins,
        "source_semantics": semantics,
        "required_modules": module_checks,
        "runtime_checks": runtime_checks,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "mooncake_version": getattr(mooncake, "__version__", None),
            "sglang_origin": sglang_origin,
            "pythonpath": pythonpath,
            "formal_execution_offline": offline,
            "container_image_env": {key: value for key, value in os.environ.items() if re.search(r"(IMAGE|CONTAINER|PYTORCH_VERSION)", key)},
        },
        "gpus": gpus,
    }
    write_json(audit_output, result)
    printable = dict(result)
    printable["model_audits"] = [{"model_id": row["model_id"], "artifact_bytes": row["artifact_bytes"], "checks": row["checks"]} for row in model_audits]
    printable["gpus"] = {"selected": gpus["selected"]}
    return printable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-workflow-commit", required=True)
    parser.add_argument("--model-map", type=Path, required=True)
    parser.add_argument("--gpu-pair", type=parse_gpu_pair, required=True)
    parser.add_argument("--ib-device", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_checks(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
