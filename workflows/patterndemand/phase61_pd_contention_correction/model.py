#!/usr/bin/env python3
"""Deterministic lightweight contention models for Phase61."""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


CANDIDATES = [
    {"candidate_id": "global_scale", "complexity_rank": 1, "family": "scale", "scope": []},
    {"candidate_id": "global_affine_max_min", "complexity_rank": 2, "family": "affine_max_min", "scope": []},
    {"candidate_id": "configuration_affine_max_min", "complexity_rank": 3, "family": "affine_max_min", "scope": ["configuration"]},
    {"candidate_id": "model_affine_max_min", "complexity_rank": 4, "family": "affine_max_min", "scope": ["model_id"]},
    {"candidate_id": "model_configuration_affine_max_min", "complexity_rank": 5, "family": "affine_max_min", "scope": ["model_id", "configuration"]},
    {"candidate_id": "model_configuration_topology_affine_max_min", "complexity_rank": 6, "family": "affine_max_min", "scope": ["model_id", "configuration", "topology_level"]},
]

NUMERIC_FIELDS = (
    "page_count0",
    "page_count1",
    "payload_bytes0",
    "payload_bytes1",
    "phase51_flow0_us",
    "phase51_flow1_us",
    "phase51_ideal_us",
    "matched_solo_ideal_us",
    "actual_concurrent_wave_us",
)


def read_points(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for field in NUMERIC_FIELDS:
            row[field] = float(row[field])
        row["curve_max_us"] = max(row["phase51_flow0_us"], row["phase51_flow1_us"])
        row["curve_min_us"] = min(row["phase51_flow0_us"], row["phase51_flow1_us"])
    return rows


def _scope_key(row: dict[str, Any], scope: list[str]) -> str:
    return "__global__" if not scope else "|".join(str(row[field]) for field in scope)


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [list(matrix[index]) + [float(vector[index])] for index in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda index: abs(augmented[index][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise RuntimeError("singular Phase61 normal equation")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for index in range(size):
            if index == column:
                continue
            factor = augmented[index][column]
            augmented[index] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[index], augmented[column])
            ]
    return [augmented[index][-1] for index in range(size)]


def _fit_affine(rows: list[dict[str, Any]]) -> dict[str, float]:
    max_values = [float(row["curve_max_us"]) for row in rows]
    min_values = [float(row["curve_min_us"]) for row in rows]
    targets = [float(row["actual_concurrent_wave_us"]) for row in rows]
    max_mean = math.fsum(max_values) / len(max_values)
    min_mean = math.fsum(min_values) / len(min_values)
    max_scale = math.sqrt(math.fsum((value - max_mean) ** 2 for value in max_values) / len(max_values)) or 1.0
    min_scale = math.sqrt(math.fsum((value - min_mean) ** 2 for value in min_values) / len(min_values)) or 1.0
    features = [
        [1.0, (maximum - max_mean) / max_scale, (minimum - min_mean) / min_scale]
        for maximum, minimum in zip(max_values, min_values)
    ]
    normal = [
        [math.fsum(row[left] * row[right] for row in features) for right in range(3)]
        for left in range(3)
    ]
    target = [math.fsum(row[index] * value for row, value in zip(features, targets)) for index in range(3)]
    centered = _solve(normal, target)
    beta_max = centered[1] / max_scale
    beta_min = centered[2] / min_scale
    intercept = centered[0] - beta_max * max_mean - beta_min * min_mean
    return {
        "intercept_us": intercept,
        "beta_max": beta_max,
        "beta_min": beta_min,
        "training_rows": len(rows),
    }


def fit_model(rows: list[dict[str, Any]], candidate: dict[str, Any], floor_us: float = 1.0) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_scope_key(row, candidate["scope"])].append(row)
    groups: dict[str, dict[str, float]] = {}
    for key, values in sorted(grouped.items()):
        if candidate["family"] == "scale":
            numerator = math.fsum(float(row["curve_max_us"]) * float(row["actual_concurrent_wave_us"]) for row in values)
            denominator = math.fsum(float(row["curve_max_us"]) ** 2 for row in values)
            if denominator <= 0:
                raise RuntimeError("invalid Phase61 scale denominator")
            groups[key] = {"scale": numerator / denominator, "training_rows": len(values)}
        else:
            groups[key] = _fit_affine(values)
    return {
        "schema_version": "phase61-contention-model-v1",
        "candidate_id": candidate["candidate_id"],
        "complexity_rank": candidate["complexity_rank"],
        "family": candidate["family"],
        "scope": candidate["scope"],
        "prediction_floor_us": float(floor_us),
        "required_runtime_inputs": ["phase51_flow0_us", "phase51_flow1_us"],
        "groups": groups,
    }


def predict(model: dict[str, Any], row: dict[str, Any]) -> float:
    key = _scope_key(row, list(model["scope"]))
    coefficients = model["groups"][key]
    maximum = float(row["curve_max_us"])
    minimum = float(row["curve_min_us"])
    if model["family"] == "scale":
        value = float(coefficients["scale"]) * maximum
    else:
        value = (
            float(coefficients["intercept_us"])
            + float(coefficients["beta_max"]) * maximum
            + float(coefficients["beta_min"]) * minimum
        )
    return max(float(model["prediction_floor_us"]), value)


def prediction_rows(rows: list[dict[str, Any]], candidate_id: str, values: list[float], fold: str) -> list[dict[str, Any]]:
    output = []
    for row, value in zip(rows, values):
        actual = float(row["actual_concurrent_wave_us"])
        output.append({
            "candidate_id": candidate_id,
            "fold_pair_id": fold,
            "model_id": row["model_id"],
            "configuration": row["configuration"],
            "topology_level": row["topology_level"],
            "pair_id": row["pair_id"],
            "payload_bytes0": int(row["payload_bytes0"]),
            "payload_bytes1": int(row["payload_bytes1"]),
            "phase51_flow0_us": row["phase51_flow0_us"],
            "phase51_flow1_us": row["phase51_flow1_us"],
            "predicted_concurrent_wave_us": value,
            "actual_concurrent_wave_us": actual,
            "absolute_error_us": abs(value - actual),
            "signed_error_us": value - actual,
        })
    return output


def slice_metrics(predictions: list[dict[str, Any]], candidate_id: str, evidence: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        keys = [
            ("overall", "all"),
            ("configuration", row["configuration"]),
            ("topology", row["topology_level"]),
            ("model", row["model_id"]),
            ("configuration_topology", f"{row['configuration']}/{row['topology_level']}"),
        ]
        for key in keys:
            groups[key].append(row)
    output = []
    for (slice_type, slice_value), rows in sorted(groups.items()):
        actual = math.fsum(float(row["actual_concurrent_wave_us"]) for row in rows)
        predicted = math.fsum(float(row["predicted_concurrent_wave_us"]) for row in rows)
        absolute = math.fsum(float(row["absolute_error_us"]) for row in rows)
        output.append({
            "candidate_id": candidate_id,
            "evidence": evidence,
            "slice_type": slice_type,
            "slice_value": slice_value,
            "points": len(rows),
            "wape": absolute / actual,
            "signed_bias": (predicted - actual) / actual,
        })
    return output


def gate(candidate: dict[str, Any], slices: list[dict[str, Any]], baseline_wape: float, contract: dict[str, Any]) -> dict[str, Any]:
    acceptance = contract["acceptance_gate"]
    overall = next(row for row in slices if row["slice_type"] == "overall")
    config_topology = [row for row in slices if row["slice_type"] == "configuration_topology"]
    max_wape = max(float(row["wape"]) for row in config_topology)
    max_bias = max(abs(float(row["signed_bias"])) for row in config_topology)
    checks = {
        "overall_wape": float(overall["wape"]) <= float(acceptance["oof_overall_wape_max"]),
        "configuration_topology_wape": max_wape <= float(acceptance["oof_each_configuration_topology_wape_max"]),
        "overall_signed_bias": abs(float(overall["signed_bias"])) <= float(acceptance["oof_overall_absolute_signed_bias_max"]),
        "configuration_topology_signed_bias": max_bias <= float(acceptance["oof_each_configuration_topology_absolute_signed_bias_max"]),
        "positive_predictions": candidate["all_predictions_positive"],
        "improves_baseline": float(overall["wape"]) < baseline_wape,
    }
    return {
        "target_guard": all(checks.values()),
        "checks": checks,
        "overall_wape": float(overall["wape"]),
        "overall_signed_bias": float(overall["signed_bias"]),
        "max_configuration_topology_wape": max_wape,
        "max_configuration_topology_absolute_signed_bias": max_bias,
    }


def evaluate_candidates(rows: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    baseline_values = [float(row["curve_max_us"]) for row in rows]
    baseline_predictions = prediction_rows(rows, "phase51_max", baseline_values, "not_applicable")
    baseline_slices = slice_metrics(baseline_predictions, "phase51_max", "development_baseline")
    baseline_wape = float(next(row for row in baseline_slices if row["slice_type"] == "overall")["wape"])
    all_predictions = list(baseline_predictions)
    all_slices = list(baseline_slices)
    candidates = []
    pair_ids = sorted({str(row["pair_id"]) for row in rows})
    for specification in CANDIDATES:
        oof_rows = []
        for pair_id in pair_ids:
            training = [row for row in rows if row["pair_id"] != pair_id]
            held_out = [row for row in rows if row["pair_id"] == pair_id]
            fitted = fit_model(training, specification)
            values = [predict(fitted, row) for row in held_out]
            oof_rows.extend(prediction_rows(held_out, specification["candidate_id"], values, pair_id))
        oof_rows.sort(key=lambda row: (row["model_id"], row["configuration"], row["topology_level"], row["pair_id"]))
        slices = slice_metrics(oof_rows, specification["candidate_id"], "leave_one_payload_pair_out")
        candidate = {
            **specification,
            "oof_points": len(oof_rows),
            "all_predictions_positive": all(float(row["predicted_concurrent_wave_us"]) > 0 for row in oof_rows),
        }
        candidate["gate"] = gate(candidate, slices, baseline_wape, contract)
        candidates.append(candidate)
        all_predictions.extend(oof_rows)
        all_slices.extend(slices)
    selected = next((candidate for candidate in candidates if candidate["gate"]["target_guard"]), None)
    return {
        "baseline_wape": baseline_wape,
        "pair_folds": len(pair_ids),
        "candidates": candidates,
        "selected": selected,
        "oof_predictions": all_predictions,
        "oof_slices": all_slices,
    }


def candidate_metric_rows(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    baseline = next(row for row in evaluation["oof_slices"] if row["candidate_id"] == "phase51_max" and row["slice_type"] == "overall")
    rows.append({
        "candidate_id": "phase51_max",
        "complexity_rank": 0,
        "family": "baseline",
        "scope": "none",
        "oof_points": 120,
        "overall_wape": baseline["wape"],
        "overall_signed_bias": baseline["signed_bias"],
        "max_configuration_topology_wape": max(row["wape"] for row in evaluation["oof_slices"] if row["candidate_id"] == "phase51_max" and row["slice_type"] == "configuration_topology"),
        "max_configuration_topology_absolute_signed_bias": max(abs(row["signed_bias"]) for row in evaluation["oof_slices"] if row["candidate_id"] == "phase51_max" and row["slice_type"] == "configuration_topology"),
        "all_predictions_positive": True,
        "target_guard": False,
        "selected": False,
    })
    selected_id = None if evaluation["selected"] is None else evaluation["selected"]["candidate_id"]
    for candidate in evaluation["candidates"]:
        gate_result = candidate["gate"]
        rows.append({
            "candidate_id": candidate["candidate_id"],
            "complexity_rank": candidate["complexity_rank"],
            "family": candidate["family"],
            "scope": "+".join(candidate["scope"]) or "global",
            "oof_points": candidate["oof_points"],
            "overall_wape": gate_result["overall_wape"],
            "overall_signed_bias": gate_result["overall_signed_bias"],
            "max_configuration_topology_wape": gate_result["max_configuration_topology_wape"],
            "max_configuration_topology_absolute_signed_bias": gate_result["max_configuration_topology_absolute_signed_bias"],
            "all_predictions_positive": candidate["all_predictions_positive"],
            "target_guard": gate_result["target_guard"],
            "selected": candidate["candidate_id"] == selected_id,
        })
    return rows
