#!/usr/bin/env python3
"""Phase54 source-schema and target-gate contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def load_contract() -> dict[str, Any]:
    return json.loads((HERE / "experiment.json").read_text(encoding="utf-8"))


def required_histogram_fields(prefix: str) -> set[str]:
    return {f"{prefix}_{kind}_bin_{i:02d}" for kind in ("calls", "logical_bytes") for i in range(12)} | {f"{prefix}_total_calls_per_1000", f"{prefix}_total_logical_bytes_per_1000"}


def validate_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("empty Phase48 examples")
    fields = set(rows[0])
    forbidden = {"requests", "full_request_list", "input_lens", "output_lens", "timestamp", "arrival_time", "prompt_text", "completion_text"}
    checks = {
        "rows_7200": len(rows) == 7200,
        "profiles_1200": len({row["profile_id"] for row in rows}) == 1200,
        "models_6": len({row["model"] for row in rows}) == 6,
        "models_per_profile": all(len({item["model"] for item in rows if item["profile_id"] == pid}) == 6 for pid in {row["profile_id"] for row in rows}),
        "train_rows_5760": sum(row["split_role"] == "expanded_train" for row in rows) == 5760,
        "validation_rows_1440": sum(row["split_role"] == "expanded_validation" for row in rows) == 1440,
        "target_free_feature_schema": not any(name.startswith("target_") for name in fields if name.startswith("feature_")),
        "required_h0_target": required_histogram_fields("h0").issubset(fields) and required_histogram_fields("target").issubset(fields),
        "no_forbidden_full_requests": not fields.intersection(forbidden),
        "finite_numeric_histograms": all(float(row[name]) >= 0.0 for row in rows for name in required_histogram_fields("h0") | required_histogram_fields("target")),
    }
    if not all(checks.values()):
        raise RuntimeError({"phase54_source_checks": checks})
    return checks


def loss_contract_self_check() -> dict[str, Any]:
    from model import loss_weights
    uniform = loss_weights("uniform"); shape = loss_weights("shape_focus"); tail = loss_weights("tail_shape_focus")
    checks = {
        "encoded_size_26": len(uniform) == 26,
        "shape_weights_totals_lower": float(shape[0]) < float(shape[1]) and float(shape[13]) < float(shape[14]),
        "tail_non_decreasing": bool((tail[1:13][1:] >= tail[1:13][:-1]).all()),
        "tail_bytes_non_decreasing": bool((tail[14:26][1:] >= tail[14:26][:-1]).all()),
        "positive_weights": bool((uniform > 0).all() and (shape > 0).all() and (tail > 0).all()),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    return checks
