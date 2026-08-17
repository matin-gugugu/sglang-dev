#!/usr/bin/env python3
"""Phase47 five-model identity, workload and teacher contracts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_specs() -> list[dict[str, Any]]:
    return load_json(HERE / "models.json")["models"]


def model_spec(model_id: str) -> dict[str, Any]:
    matches = [row for row in model_specs() if row["model_id"] == model_id]
    if len(matches) != 1:
        raise RuntimeError(f"unknown or duplicated model_id: {model_id}")
    return matches[0]


def load_model_map(path: Path) -> dict[str, Path]:
    raw = load_json(path)
    if raw.get("schema_version") != "phase47-external-model-map-v1":
        raise RuntimeError("invalid model-map schema")
    mapping = raw.get("models")
    if not isinstance(mapping, dict):
        raise RuntimeError("model-map models must be an object")
    expected = {row["model_id"] for row in model_specs()}
    if set(mapping) != expected:
        raise RuntimeError({"model_map_ids": sorted(mapping), "expected": sorted(expected)})
    result = {name: Path(value).expanduser().resolve() for name, value in mapping.items()}
    if len(set(result.values())) != len(result):
        raise RuntimeError("every Phase47 model must use a distinct local directory")
    return result


def _head_dim(config: dict[str, Any]) -> int:
    return int(
        config.get("head_dim")
        or int(config["hidden_size"]) // int(config["num_attention_heads"])
    )


def _weight_files(model_path: Path) -> list[Path]:
    suffixes = {".safetensors", ".bin", ".pt", ".pth"}
    return sorted(
        path for path in model_path.iterdir() if path.is_file() and path.suffix.lower() in suffixes
    )


def inspect_model(model_id: str, model_path: Path, *, hash_weights: bool) -> dict[str, Any]:
    """Validate structure/source marker and optionally hash every weight shard."""
    spec = model_spec(model_id)
    if not model_path.is_dir():
        raise RuntimeError(f"missing model directory for {model_id}: {model_path}")
    config_path = model_path / "config.json"
    marker_path = model_path / ".phase47_source.json"
    if not config_path.is_file() or not marker_path.is_file():
        raise RuntimeError(f"{model_id} requires config.json and .phase47_source.json")
    config = load_json(config_path)
    marker = load_json(marker_path)
    architectures = config.get("architectures") or []
    checks: dict[str, bool] = {
        "source_repo": marker.get("repo_id") == spec["repo_id"],
        "source_revision": marker.get("revision") == spec["revision"],
        "source_model_id": marker.get("model_id") == model_id,
        "source_config_sha": marker.get("config_sha256") == sha256(config_path),
        "model_type": config.get("model_type") == spec["model_type"],
        "architecture": spec["architecture"] in architectures,
        "dtype_bfloat16": config.get("torch_dtype") in {"bfloat16", None},
        "config_sha": spec.get("official_config_sha256") is None
        or sha256(config_path) == spec["official_config_sha256"],
    }
    actual_structure: dict[str, int] = {}
    for name, expected in spec["structure"].items():
        actual = _head_dim(config) if name == "head_dim" else int(config.get(name) or 0)
        actual_structure[name] = actual
        checks[f"structure_{name}"] = actual == int(expected)
    weights = _weight_files(model_path)
    indexes = sorted(model_path.glob("*.index.json"))
    checks["weight_artifacts_present"] = bool(weights or indexes)
    if not all(checks.values()):
        raise RuntimeError({"model_id": model_id, "model_checks": checks})
    inventory = []
    for path in [*indexes, *weights]:
        inventory.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path) if hash_weights else None,
            }
        )
    if spec["is_mla"]:
        per_token = (
            int(config["num_hidden_layers"])
            * (int(config["kv_lora_rank"]) + int(config["qk_rope_head_dim"]))
            * 2
        )
    else:
        per_token = (
            int(config["num_hidden_layers"])
            * 2
            * int(config["num_key_value_heads"])
            * _head_dim(config)
            * 2
        )
    page_size = int(spec["page_size_tokens"])
    derived_checks = {
        "kv_bytes_per_token": per_token == int(spec["expected_kv_bytes_per_token"]),
        "kv_bytes_per_page": per_token * page_size
        == int(spec["expected_kv_bytes_per_page"]),
    }
    if not all(derived_checks.values()):
        raise RuntimeError({"model_id": model_id, "derived_checks": derived_checks})
    return {
        "model_id": model_id,
        "model_path": str(model_path),
        "model_path_redacted_in_git": True,
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "config_sha256": sha256(config_path),
        "source_marker_sha256": sha256(marker_path),
        "structure": {**actual_structure, "page_size_tokens": page_size, "dtype_bytes": 2},
        "derived": {
            "kv_bytes_per_token": per_token,
            "kv_bytes_per_page": per_token * page_size,
            "formula": spec["kv_formula"],
        },
        "attention_backend": spec["attention_backend"],
        "is_mla": bool(spec["is_mla"]),
        "artifact_inventory": inventory,
        "artifact_inventory_hashed": hash_weights,
        "artifact_bytes": sum(row["bytes"] for row in inventory),
        "checks": {**checks, **derived_checks},
    }


def runtime_contract(base: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Translate Phase47 into the proven Phase40 single-model runner interface."""
    page = int(spec["page_size_tokens"])
    chunk = int(base["measurement_contract"]["chunked_prefill_tokens"])

    def page_segment(start_token: int, end_token: int) -> list[int]:
        return [start_token // page, math.ceil(end_token / page)]

    # For the 1000/1000/1000/2000 FCFS wave, the last request is split at
    # floor((4096-3000)/page)*page so a non-final chunk is page aligned.
    split = ((chunk - 3000) // page) * page
    expected_segments = [
        [page_segment(0, 1000)],
        [page_segment(0, 1000)],
        [page_segment(0, 1000)],
        [page_segment(0, split), page_segment(split, 2000)],
    ]
    result = dict(base)
    result["backend_contract"] = {
        "inference_attention_backend": spec["attention_backend"],
        "inference_attention_backend_fallback_permitted": False,
        "sglang_transfer_backend": "mooncake",
        "transport": "rdma",
    }
    result["measurement_contract"] = dict(base["measurement_contract"])
    result["measurement_contract"].update(
        {
            "page_size_tokens": page,
            "bootstrap_batch_barrier_env": "SGLANG_PD_BOOTSTRAP_BATCH_BARRIER=1",
        }
    )
    result["compatibility_smoke_contract"] = {
        "transport_request": {
            "rid_prefix": f"p47::{spec['model_id']}::compat_smoke",
            "prompt_tokens": 64,
            "max_new_tokens": 2,
        },
        "admission_probe": {
            "rid_prefix_base": f"p47::{spec['model_id']}::admission_smoke::rep",
            "prompt_tokens": [1000, 1000, 1000, 2000],
            "repeats": 2,
            "expected_segments_by_request_index": expected_segments,
        },
        "expected_transport_sender_chunks": 1,
        "expected_sender_chunks_total": 11,
        "expected_page_size_tokens": page,
        "expected_kv_page_count": math.ceil(64 / page),
        "expected_transfer_backend": "MooncakeKVSender",
    }
    result["acceptance_gates"] = {
        "expected_requests": 45,
    }
    result["expected_alignment_rows"] = 6
    result["expected_histogram_rows"] = 72
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refuse empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def read_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"invalid JSONL: {path}:{line_number}") from error
    return rows


def add_model_id(rows: list[dict[str, Any]], model_id: str) -> list[dict[str, Any]]:
    return [{"model_id": model_id, **row} for row in rows]


def repeat_histograms_exact(events: list[dict[str, Any]], scenarios: list[str]) -> bool:
    """Independent compact determinism check used by the Phase47 verifier."""
    signatures: dict[tuple[str, int], dict[int, tuple[int, int]]] = defaultdict(dict)
    edges = load_json(HERE / "experiment.json")["phase34_bin_edges_bytes"]
    for row in events:
        value = int(row["logical_bytes"])
        index = next((i for i in range(12) if value < edges[i + 1]), 11)
        key = (str(row["scenario"]), int(row["repeat"]))
        calls, logical = signatures[key].get(index, (0, 0))
        signatures[key][index] = (calls + 1, logical + value)
    for scenario in scenarios:
        rows = [sorted(signatures[(scenario, repeat)].items()) for repeat in range(3)]
        if not rows or any(row != rows[0] for row in rows[1:]):
            return False
    return True
