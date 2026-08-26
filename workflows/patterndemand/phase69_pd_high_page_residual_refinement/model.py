#!/usr/bin/env python3
"""Deterministic Phase69 high-page residual selection."""
from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

CANDIDATES = [
    {"candidate_id": "r67_high_page_linear", "rank": 1, "feature_family": "high_linear"},
    {"candidate_id": "r67_high_page_sqrt", "rank": 2, "feature_family": "high_sqrt"},
    {"candidate_id": "r67_high_page_linear_sqrt", "rank": 3, "feature_family": "high_linear_sqrt"},
]
BASELINES = ("max_edge", "r61_graph", "r65_frozen", "r67_frozen")
SCHEMES = ("payload_cohort", "topology", "tail64", "source_blocked")
PRIMARY_SCHEMES = ("payload_cohort", "topology", "tail64")


def _legacy_r65(model: dict[str, Any], row: dict[str, Any]) -> float:
    group = model["groups"][f"{row['model_id']}|{row['configuration']}"]
    m, b, s = (float(row[key]) for key in ("max_edge_baseline_us", "busiest_endpoint_sum_us", "sum_edge_baseline_us"))
    value = float(group["intercept_us"]) + float(group["beta_M"]) * m
    value += float(group["beta_busy"]) * max(0.0, b - m)
    value += float(group["beta_nonbusy"]) * max(0.0, s - b)
    return max(float(model["prediction_floor_us"]), value)


def _r67_features(row: dict[str, Any], family: str) -> list[float]:
    m, b, s = (float(row[key]) for key in ("max_edge_baseline_us", "busiest_endpoint_sum_us", "sum_edge_baseline_us"))
    values = [1.0, m, max(0.0, b - m), max(0.0, s - b)]
    pages = row["pages_list"]
    page_max = float(max(pages))
    page_rest = float(sum(pages) - max(pages))
    if family in ("page_linear", "page_sqrt"):
        values += [page_max, page_rest]
    if family == "page_sqrt":
        values += [math.sqrt(page_max), math.sqrt(page_rest)]
    return values


def predict_r67(model: dict[str, Any], row: dict[str, Any]) -> float:
    group = model["groups"][f"{row['model_id']}|{row['configuration']}"]
    value = math.fsum(a * b for a, b in zip(group["coefficients"], _r67_features(row, model["feature_family"])))
    return max(float(model["prediction_floor_us"]), value)


def read_development(
    phase64: Path,
    phase66: Path,
    phase68: Path,
    r65_model: dict[str, Any],
    r67_model: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, path in (("phase64", phase64), ("phase66", phase66), ("phase68", phase68)):
        with path.open(encoding="utf-8", newline="") as stream:
            values = list(csv.DictReader(stream))
        for value in values:
            row: dict[str, Any] = dict(value)
            row["source_phase"] = source
            row["pages_list"] = [int(part) for part in row["pages"].split("|")]
            match = re.search(r"v(\d+)$", row["vector_id"])
            if match is None:
                raise RuntimeError({"bad_vector": row["vector_id"]})
            row["vector_index"] = int(match.group(1))
            for key in ("max_edge_baseline_us", "busiest_endpoint_sum_us", "sum_edge_baseline_us", "actual_concurrent_wave_us"):
                row[key] = float(row[key])
            if source == "phase64":
                row["r61_prediction_us"] = float(row["graph_prediction_us"])
                row["r65_prediction_us"] = _legacy_r65(r65_model, row)
            else:
                row["r61_prediction_us"] = float(row["r61_graph_prediction_us"])
                row["r65_prediction_us"] = float(row["phase65_prediction_us"])
            row["r67_prediction_us"] = predict_r67(r67_model, row)
            if source == "phase68" and not math.isclose(row["r67_prediction_us"], float(row["phase67_prediction_us"]), rel_tol=1e-12, abs_tol=1e-8):
                raise RuntimeError({"phase68_r67_recompute_mismatch": row["vector_id"]})
            rows.append(row)
    return rows


def feature_names(family: str) -> list[str]:
    if family == "high_linear":
        return ["page_max_excess32", "mean_other_pages_excess32"]
    if family == "high_sqrt":
        return ["sqrt_page_max_excess32", "sqrt_mean_other_pages_excess32"]
    if family == "high_linear_sqrt":
        return ["page_max_excess32", "mean_other_pages_excess32", "sqrt_page_max_excess32", "sqrt_mean_other_pages_excess32"]
    raise RuntimeError(family)


def features(row: dict[str, Any], family: str) -> list[float]:
    if row["configuration"] == "P2D2_MATCHING":
        return [0.0] * len(feature_names(family))
    pages = row["pages_list"]
    page_max = float(max(pages))
    mean_other = float(sum(pages) - max(pages)) / max(1, len(pages) - 1)
    high_max = max(0.0, page_max - 32.0)
    high_other = max(0.0, mean_other - 32.0)
    if family == "high_linear":
        return [high_max, high_other]
    if family == "high_sqrt":
        return [math.sqrt(high_max), math.sqrt(high_other)]
    if family == "high_linear_sqrt":
        return [high_max, high_other, math.sqrt(high_max), math.sqrt(high_other)]
    raise RuntimeError(family)


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    augmented = [list(matrix[index]) + [float(vector[index])] for index in range(n)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda index: abs(augmented[index][column]))
        if abs(augmented[pivot][column]) < 1e-15:
            raise RuntimeError("singular Phase69 normal equation")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for index in range(n):
            if index == column:
                continue
            factor = augmented[index][column]
            augmented[index] = [value - factor * pivot_value for value, pivot_value in zip(augmented[index], augmented[column])]
    return [augmented[index][-1] for index in range(n)]


def _fit_group(rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    raw = [features(row, family) for row in rows]
    dimensions = len(raw[0])
    scales = [max(1e-12, math.sqrt(math.fsum(values[index] ** 2 for values in raw) / len(raw))) for index in range(dimensions)]
    matrix = [[value / scales[index] for index, value in enumerate(values)] for values in raw]
    target = [float(row["actual_concurrent_wave_us"]) - float(row["r67_prediction_us"]) for row in rows]
    normal = [[math.fsum(values[left] * values[right] for values in matrix) for right in range(dimensions)] for left in range(dimensions)]
    projected = [math.fsum(values[index] * residual for values, residual in zip(matrix, target)) for index in range(dimensions)]
    ridge = max(1e-8, math.fsum(normal[index][index] for index in range(dimensions)) * 1e-6 / dimensions)
    for index in range(dimensions):
        normal[index][index] += ridge
    scaled = _solve(normal, projected)
    coefficients = [scaled[index] / scales[index] for index in range(dimensions)]
    return {"training_rows": len(rows), "feature_scales": scales, "coefficients": coefficients, "ridge_absolute": ridge}


def fit_model(rows: list[dict[str, Any]], candidate: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[f"{row['model_id']}|{row['configuration']}"].append(row)
    return {
        "schema_version": "phase69-multiflow-high-page-residual-v1",
        "candidate_id": candidate["candidate_id"],
        "complexity_rank": candidate["rank"],
        "scope": ["model_id", "configuration"],
        "anchor_model": "frozen_phase67_graph_page_shape",
        "anchor_coefficient": 1.0,
        "activation_page_threshold": 32,
        "feature_family": candidate["feature_family"],
        "feature_names": feature_names(candidate["feature_family"]),
        "prediction_floor_us": 1.0,
        "groups": {key: _fit_group(group, candidate["feature_family"]) for key, group in sorted(groups.items())},
    }


def predict(model: dict[str, Any], row: dict[str, Any]) -> float:
    group = model["groups"][f"{row['model_id']}|{row['configuration']}"]
    residual = math.fsum(a * b for a, b in zip(group["coefficients"], features(row, model["feature_family"])))
    return max(float(model["prediction_floor_us"]), float(row["r67_prediction_us"]) + residual)


def baseline_value(name: str, row: dict[str, Any]) -> float:
    if name == "max_edge":
        return float(row["max_edge_baseline_us"])
    if name == "r61_graph":
        return float(row["r61_prediction_us"])
    if name == "r65_frozen":
        return float(row["r65_prediction_us"])
    if name == "r67_frozen":
        return float(row["r67_prediction_us"])
    raise RuntimeError(name)


def split_folds(rows: list[dict[str, Any]], scheme: str) -> list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]]:
    if scheme == "payload_cohort":
        keys = [(source, index) for source in ("phase64", "phase66", "phase68") for index in range(10)]
        return [(f"{source}_v{index:02d}", [row for row in rows if (row["source_phase"], row["vector_index"]) != key], [row for row in rows if (row["source_phase"], row["vector_index"]) == key]) for source, index in keys for key in [(source, index)]]
    if scheme == "topology":
        return [(level, [row for row in rows if row["topology_level"] != level], [row for row in rows if row["topology_level"] == level]) for level in ("L1", "L2", "L3")]
    if scheme == "tail64":
        return [("contains_page64", [row for row in rows if 64 not in row["pages_list"]], [row for row in rows if 64 in row["pages_list"]])]
    if scheme == "source_blocked":
        return [(source, [row for row in rows if row["source_phase"] != source], [row for row in rows if row["source_phase"] == source]) for source in ("phase64", "phase66", "phase68")]
    raise RuntimeError(scheme)


def prediction_rows(candidate: str, scheme: str, fold: str, rows: list[dict[str, Any]], values: list[float]) -> list[dict[str, Any]]:
    output = []
    for row, prediction in zip(rows, values):
        actual = float(row["actual_concurrent_wave_us"])
        output.append({
            "candidate_id": candidate, "oof_scheme": scheme, "fold": fold, "source_phase": row["source_phase"],
            "model_id": row["model_id"], "configuration": row["configuration"], "topology_level": row["topology_level"],
            "vector_id": row["vector_id"], "vector_index": row["vector_index"], "pages": row["pages"],
            "predicted_us": prediction, "actual_us": actual, "absolute_error_us": abs(prediction - actual),
            "signed_error_us": prediction - actual,
        })
    return output


def slice_metrics(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        keys = (
            ("overall", "all"),
            ("model", row["model_id"]),
            ("configuration", row["configuration"]),
            ("model_configuration", f"{row['model_id']}/{row['configuration']}"),
            ("configuration_topology", f"{row['configuration']}/{row['topology_level']}"),
            ("source_model_configuration", f"{row['source_phase']}/{row['model_id']}/{row['configuration']}"),
        )
        for key in keys:
            groups[key].append(row)
    output = []
    for (kind, value), rows in sorted(groups.items()):
        actual = math.fsum(float(row["actual_us"]) for row in rows)
        predicted = math.fsum(float(row["predicted_us"]) for row in rows)
        absolute = math.fsum(float(row["absolute_error_us"]) for row in rows)
        output.append({
            "candidate_id": rows[0]["candidate_id"], "oof_scheme": rows[0]["oof_scheme"],
            "slice_type": kind, "slice_value": value, "points": len(rows),
            "wape": absolute / actual, "signed_bias": (predicted - actual) / actual,
        })
    return output


def _slices(rows: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["slice_type"] == kind]


def evaluate(rows: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    all_predictions: list[dict[str, Any]] = []
    all_slices: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    cached_baselines: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for baseline in BASELINES:
        for scheme in SCHEMES:
            held = [row for _fold, _training, test in split_folds(rows, scheme) for row in test]
            predictions = prediction_rows(baseline, scheme, "fixed", held, [baseline_value(baseline, row) for row in held])
            slices = slice_metrics(predictions)
            cached_baselines[(baseline, scheme)] = slices
            all_predictions += predictions
            all_slices += slices
    gates = contract["acceptance_gate"]
    for candidate in CANDIDATES:
        scheme_values: dict[str, Any] = {}
        candidate_all_pass = True
        for scheme in SCHEMES:
            predictions = []
            for fold, training, held in split_folds(rows, scheme):
                model = fit_model(training, candidate)
                predictions += prediction_rows(candidate["candidate_id"], scheme, fold, held, [predict(model, row) for row in held])
            predictions.sort(key=lambda row: (row["source_phase"], row["model_id"], row["configuration"], row["topology_level"], int(row["vector_index"])))
            slices = slice_metrics(predictions)
            all_predictions += predictions
            all_slices += slices
            overall = _slices(slices, "overall")[0]
            models = _slices(slices, "model")
            configurations = _slices(slices, "configuration")
            model_configurations = _slices(slices, "model_configuration")
            configuration_topologies = _slices(slices, "configuration_topology")
            source_model_configurations = _slices(slices, "source_model_configuration")
            baseline_overall = {name: _slices(cached_baselines[(name, scheme)], "overall")[0] for name in BASELINES}
            baseline_configurations = {
                name: {row["slice_value"]: row for row in _slices(cached_baselines[(name, scheme)], "configuration")}
                for name in BASELINES
            }
            checks = {
                "overall_wape": overall["wape"] <= gates["overall_wape_max"],
                "each_model_wape": all(row["wape"] <= gates["model_wape_max"] for row in models),
                "each_configuration_wape": all(row["wape"] <= gates["configuration_wape_max"] for row in configurations),
                "each_model_configuration_wape": all(row["wape"] <= gates["model_configuration_wape_max"] for row in model_configurations),
                "each_configuration_topology_wape": all(row["wape"] <= gates["configuration_topology_wape_max"] for row in configuration_topologies),
                "each_source_model_configuration_wape": all(row["wape"] <= gates["source_model_configuration_wape_max"] for row in source_model_configurations),
                "overall_bias": abs(overall["signed_bias"]) <= gates["overall_absolute_bias_max"],
                "each_model_bias": all(abs(row["signed_bias"]) <= gates["model_absolute_bias_max"] for row in models),
                "each_configuration_bias": all(abs(row["signed_bias"]) <= gates["configuration_absolute_bias_max"] for row in configurations),
                "each_model_configuration_bias": all(abs(row["signed_bias"]) <= gates["model_configuration_absolute_bias_max"] for row in model_configurations),
                "each_configuration_topology_bias": all(abs(row["signed_bias"]) <= gates["configuration_topology_absolute_bias_max"] for row in configuration_topologies),
                "each_source_model_configuration_bias": all(abs(row["signed_bias"]) <= gates["source_model_configuration_absolute_bias_max"] for row in source_model_configurations),
                "positive": all(row["predicted_us"] > 0 for row in predictions),
            }
            if scheme in PRIMARY_SCHEMES:
                checks["improves_all_baselines_overall"] = all(overall["wape"] < baseline_overall[name]["wape"] for name in BASELINES)
                checks["improves_best_baseline_each_shared_endpoint_configuration"] = all(
                    row["wape"] < min(baseline_configurations[name][row["slice_value"]]["wape"] for name in BASELINES)
                    for row in configurations
                    if row["slice_value"] != "P2D2_MATCHING"
                )
                matching = next(row for row in configurations if row["slice_value"] == "P2D2_MATCHING")
                checks["preserves_p2d2_matching"] = math.isclose(
                    matching["wape"], baseline_configurations["r67_frozen"]["P2D2_MATCHING"]["wape"],
                    rel_tol=0.0, abs_tol=1e-15,
                )
            else:
                checks["does_not_degrade_r67_overall"] = overall["wape"] <= baseline_overall["r67_frozen"]["wape"] + 1e-12
            passed = all(checks.values())
            candidate_all_pass &= passed
            scheme_values[scheme] = {
                "pass": passed, "checks": checks, "points": len(predictions),
                "overall_wape": overall["wape"], "overall_signed_bias": overall["signed_bias"],
                "max_model_wape": max(row["wape"] for row in models),
                "max_configuration_wape": max(row["wape"] for row in configurations),
                "max_model_configuration_wape": max(row["wape"] for row in model_configurations),
                "max_configuration_topology_wape": max(row["wape"] for row in configuration_topologies),
                "max_source_model_configuration_wape": max(row["wape"] for row in source_model_configurations),
                "max_model_configuration_absolute_bias": max(abs(row["signed_bias"]) for row in model_configurations),
            }
        summaries.append({
            "candidate_id": candidate["candidate_id"], "complexity_rank": candidate["rank"],
            "feature_family": candidate["feature_family"], "target_guard": candidate_all_pass, "schemes": scheme_values,
        })
    selected = next((candidate for candidate in summaries if candidate["target_guard"]), None)
    return {"candidates": summaries, "selected": selected, "predictions": all_predictions, "slices": all_slices}
