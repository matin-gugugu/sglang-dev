#!/usr/bin/env python3
"""Phase48 six-model low-dimensional feature and page-aware teacher contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
P41 = HERE.parent / "phase41_pd_full_window_dataset"
P47 = HERE.parent / "phase47_pd_five_model_teacher_validation"
import importlib.util
_SPEC41 = importlib.util.spec_from_file_location("phase41_contracts", P41 / "contracts.py")
if _SPEC41 is None or _SPEC41.loader is None: raise RuntimeError("cannot load Phase41 contracts")
_P41 = importlib.util.module_from_spec(_SPEC41); _SPEC41.loader.exec_module(_P41)

# Import Phase47 under an unambiguous module name: both source directories use contracts.py.
_SPEC = importlib.util.spec_from_file_location("phase47_contracts", P47 / "contracts.py")
if _SPEC is None or _SPEC.loader is None: raise RuntimeError("cannot load Phase47 contracts")
_P47 = importlib.util.module_from_spec(_SPEC); _SPEC.loader.exec_module(_P47)


def load_models() -> list[dict[str, Any]]:
    return json.loads((HERE / "models.json").read_text(encoding="utf-8"))["models"]


def validate_models(models: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [row["model_id"] for row in models]
    checks = {
        "six_unique_models": len(ids) == 6 and len(set(ids)) == 6,
        "qwen_present": "qwen3-8b" in ids,
        "five_phase47_models_exact": set(ids) - {"qwen3-8b"} == {
            "deepseek-v2-lite", "qwen3-30b-a3b", "llama-3.2-3b-instruct",
            "qwen2.5-14b-instruct", "mixtral-8x7b-instruct-v0.1",
        },
        "positive_sizes": all(int(row["kv_bytes_per_page"]) > 0 and int(row["page_size_tokens"]) > 0 for row in models),
        "mla_page64_only": all((row["model_id"] == "deepseek-v2-lite") == (int(row["page_size_tokens"]) == 64) for row in models),
    }
    if not all(checks.values()): raise RuntimeError({"model_contract_checks": checks})
    return checks


def _events(requests: list[tuple[int, int]], model: dict[str, Any], case: str, wave_size: int, chunk_tokens: int) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for wave_index, wave in enumerate(_P41.partition_requests(requests, wave_size)):
        rows = [
            {"scenario": case, "repeat": 0, "wave_index": wave_index,
             "rid": f"p48::{case}::w{wave_index}::r{index}", "prompt_tokens": int(pair[0])}
            for index, pair in enumerate(wave)
        ]
        events.extend(_P47.teacher_chunks_for_wave_page_aware(
            rows, chunk_tokens=chunk_tokens,
            page_size_tokens=int(model["page_size_tokens"]),
            kv_bytes_per_page=int(model["kv_bytes_per_page"]),
        ))
    return events


def model_features(model: dict[str, Any]) -> dict[str, int]:
    backend = str(model["attention_backend"])
    return {
        "feature_model_num_hidden_layers": int(model["num_hidden_layers"]),
        "feature_model_hidden_size": int(model["hidden_size"]),
        "feature_model_num_attention_heads": int(model["num_attention_heads"]),
        "feature_model_num_key_value_heads": int(model["num_key_value_heads"]),
        "feature_model_head_dim": int(model["head_dim"]),
        "feature_model_kv_lora_rank": int(model["kv_lora_rank"]),
        "feature_model_qk_rope_head_dim": int(model["qk_rope_head_dim"]),
        "feature_model_dtype_bytes": int(model["dtype_bytes"]),
        "feature_model_kv_bytes_per_token": int(model["kv_bytes_per_token"]),
        "feature_model_kv_bytes_per_page": int(model["kv_bytes_per_page"]),
        "feature_model_is_mla": int(model["is_mla"]),
        "feature_model_backend_flashinfer": int(backend == "flashinfer"),
        "feature_model_backend_trtllm_mla": int(backend == "trtllm_mla"),
    }


def six_model_example_rows(*, profile: dict[str, Any], requests: list[tuple[int, int]] | None,
                           phase41_contract: dict[str, Any], feature_contract: dict[str, Any],
                           models: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = _P41.predictor_features(profile, feature_contract)
    for key in [name for name in base if name.startswith("feature_model_") or name.startswith("feature_p_") or name.startswith("feature_d_") or name.startswith("feature_pd_")]:
        del base[key]
    measurement = phase41_contract["measurement_contract"]
    wave_size = int(measurement["wave_size"]); chunk_tokens = int(measurement["chunked_prefill_tokens"])
    identifiers = {name: profile[name] for name in ("profile_id", "split_role", "source", "segment", "source_split", "window_id", "cutoff_ms")}
    execution = {
        "feature_p_tp_size": 1, "feature_p_pp_size": 1, "feature_d_tp_size": 1, "feature_d_pp_size": 1,
        "feature_pd_wave_size": wave_size, "feature_pd_chunk_tokens": chunk_tokens,
        "feature_pd_fcfs": 1, "feature_pd_fixed_draining": 1, "feature_pd_atomic_wave_barrier": 1,
        "feature_pd_radix_cache": 0, "feature_pd_overlap_schedule": 0,
    }
    h0_requests = _P41.pseudo_requests(profile); examples=[]; targets=[]
    for model in models:
        model_id = str(model["model_id"]); ids = {**identifiers, "model": model_id}
        h0_events = _events(h0_requests, model, f"h0::{profile['profile_id']}::{model_id}", wave_size, chunk_tokens)
        h0 = _P41.histogram_fields("h0", h0_events, len(h0_requests))
        features = {**base, **model_features(model), **execution, "feature_pd_page_size_tokens": int(model["page_size_tokens"])}
        if requests is None:
            examples.append({**ids, **features, **h0}); continue
        target_events = _events(requests, model, f"hfull::{profile['profile_id']}::{model_id}", wave_size, chunk_tokens)
        target = _P41.histogram_fields("target", target_events, len(requests))
        residual = {
            f"residual_{kind}_bin_{index:02d}": target[f"target_{kind}_bin_{index:02d}"] - h0[f"h0_{kind}_bin_{index:02d}"]
            for kind in ("calls", "logical_bytes") for index in range(12)
        }
        teacher = "six_model_pure_pd_page_aware_bounded_wave_hfull_v1"
        examples.append({**ids, **features, **h0, **target, **residual, "teacher_kind": teacher})
        targets.append({**ids, "request_count": len(requests), **target, "teacher_kind": teacher})
    return examples, targets


def contract_self_check() -> dict[str, Any]:
    models = load_models(); checks = validate_models(models)
    # At page size 1, the Phase47 page-aware debit equals Phase41's original token debit.
    qwen = next(row for row in models if row["model_id"] == "qwen3-8b")
    events = _events([(4097, 1), (63, 1)], qwen, "selfcheck", 64, 4096)
    checks["qwen_4097_plus_63_chunks"] = len(events) == 3
    deepseek = next(row for row in models if row["model_id"] == "deepseek-v2-lite")
    ds_events = _events([(65, 1)], deepseek, "selfcheck", 64, 4096)
    checks["deepseek_65_tokens_two_pages"] = len(ds_events) == 1 and int(ds_events[0]["kv_page_count"]) == 2
    checks["chunk_page_aligned"] = all(4096 % int(row["page_size_tokens"]) == 0 for row in models)
    if not all(checks.values()): raise RuntimeError(checks)
    return checks
