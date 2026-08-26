#!/usr/bin/env python3
"""Deterministic Phase67 graph+page correction selection."""
from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

CANDIDATES = [
    {"candidate_id": "model_configuration_affine_graph", "rank": 1, "feature_family": "graph"},
    {"candidate_id": "model_configuration_graph_page_linear", "rank": 2, "feature_family": "page_linear"},
    {"candidate_id": "model_configuration_graph_page_sqrt", "rank": 3, "feature_family": "page_sqrt"},
]
BASELINES = ("max_edge", "r61_graph", "r65_frozen")
SCHEMES = ("payload_cohort", "topology", "source_blocked", "tail32")


def _legacy_r65(model: dict[str, Any], row: dict[str, Any]) -> float:
    key = f"{row['model_id']}|{row['configuration']}"
    co = model["groups"][key]
    m, b, s = (float(row[k]) for k in ("max_edge_baseline_us", "busiest_endpoint_sum_us", "sum_edge_baseline_us"))
    value = float(co["intercept_us"]) + float(co["beta_M"]) * m + float(co["beta_busy"]) * max(0.0, b-m) + float(co["beta_nonbusy"]) * max(0.0, s-b)
    return max(float(model["prediction_floor_us"]), value)


def read_development(phase64: Path, phase66: Path, r65_model: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, path in (("phase64", phase64), ("phase66", phase66)):
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
            row["r61_prediction_us"] = float(row["graph_prediction_us"] if source == "phase64" else row["r61_graph_prediction_us"])
            row["r65_prediction_us"] = _legacy_r65(r65_model, row) if source == "phase64" else float(row["phase65_prediction_us"])
            rows.append(row)
    return rows


def feature_names(family: str) -> list[str]:
    names = ["intercept", "M", "B_minus_M", "S_minus_B"]
    if family in ("page_linear", "page_sqrt"):
        names += ["page_max", "page_rest"]
    if family == "page_sqrt":
        names += ["sqrt_page_max", "sqrt_page_rest"]
    return names


def features(row: dict[str, Any], family: str) -> list[float]:
    m = float(row["max_edge_baseline_us"]); b = float(row["busiest_endpoint_sum_us"]); s = float(row["sum_edge_baseline_us"])
    values = [1.0, m, max(0.0, b-m), max(0.0, s-b)]
    pages = row["pages_list"]; page_max = float(max(pages)); page_rest = float(sum(pages)-max(pages))
    if family in ("page_linear", "page_sqrt"):
        values += [page_max, page_rest]
    if family == "page_sqrt":
        values += [math.sqrt(page_max), math.sqrt(page_rest)]
    return values


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector); aug = [list(matrix[i])+[float(vector[i])] for i in range(n)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda i: abs(aug[i][column]))
        if abs(aug[pivot][column]) < 1e-15:
            raise RuntimeError("singular Phase67 normal equation")
        aug[column], aug[pivot] = aug[pivot], aug[column]
        scale = aug[column][column]; aug[column] = [v/scale for v in aug[column]]
        for i in range(n):
            if i == column:
                continue
            factor = aug[i][column]; aug[i] = [v-factor*p for v, p in zip(aug[i], aug[column])]
    return [aug[i][-1] for i in range(n)]


def _fit_group(rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    raw = [features(row, family) for row in rows]; n = len(raw[0])
    scales = [1.0] + [max(1e-12, math.sqrt(math.fsum(x[j]*x[j] for x in raw)/len(raw))) for j in range(1, n)]
    x = [[value/scales[j] for j, value in enumerate(values)] for values in raw]
    y = [float(row["actual_concurrent_wave_us"]) for row in rows]
    normal = [[math.fsum(a[i]*a[j] for a in x) for j in range(n)] for i in range(n)]
    target = [math.fsum(a[i]*v for a, v in zip(x, y)) for i in range(n)]
    ridge = math.fsum(normal[i][i] for i in range(n))*1e-10/n
    for i in range(n): normal[i][i] += ridge
    scaled = _solve(normal, target); coefficients = [scaled[j]/scales[j] for j in range(n)]
    return {"training_rows": len(rows), "feature_scales": scales, "coefficients": coefficients, "ridge_absolute": ridge}


def fit_model(rows: list[dict[str, Any]], candidate: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: groups[f"{row['model_id']}|{row['configuration']}"].append(row)
    return {
        "schema_version": "phase67-multiflow-graph-page-correction-v1",
        "candidate_id": candidate["candidate_id"], "complexity_rank": candidate["rank"],
        "scope": ["model_id", "configuration"], "feature_family": candidate["feature_family"],
        "feature_names": feature_names(candidate["feature_family"]), "prediction_floor_us": 1.0,
        "groups": {key: _fit_group(group, candidate["feature_family"]) for key, group in sorted(groups.items())},
    }


def predict(model: dict[str, Any], row: dict[str, Any]) -> float:
    group = model["groups"][f"{row['model_id']}|{row['configuration']}"]
    value = math.fsum(a*b for a, b in zip(group["coefficients"], features(row, model["feature_family"])))
    return max(float(model["prediction_floor_us"]), value)


def baseline_value(name: str, row: dict[str, Any]) -> float:
    if name == "max_edge": return float(row["max_edge_baseline_us"])
    if name == "r61_graph": return float(row["r61_prediction_us"])
    if name == "r65_frozen": return float(row["r65_prediction_us"])
    raise RuntimeError(name)


def split_folds(rows: list[dict[str, Any]], scheme: str) -> list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]]:
    if scheme == "payload_cohort":
        keys = [(source, index) for source in ("phase64", "phase66") for index in range(10)]
        return [(f"{source}_v{index:02d}", [r for r in rows if (r["source_phase"], r["vector_index"]) != key], [r for r in rows if (r["source_phase"], r["vector_index"]) == key]) for source, index in keys for key in [(source, index)]]
    if scheme == "topology":
        return [(level, [r for r in rows if r["topology_level"] != level], [r for r in rows if r["topology_level"] == level]) for level in ("L1", "L2", "L3")]
    if scheme == "source_blocked":
        return [(source, [r for r in rows if r["source_phase"] != source], [r for r in rows if r["source_phase"] == source]) for source in ("phase64", "phase66")]
    if scheme == "tail32":
        return [("contains_page32", [r for r in rows if 32 not in r["pages_list"]], [r for r in rows if 32 in r["pages_list"]])]
    raise RuntimeError(scheme)


def prediction_rows(candidate: str, scheme: str, fold: str, rows: list[dict[str, Any]], values: list[float]) -> list[dict[str, Any]]:
    output = []
    for row, prediction in zip(rows, values):
        actual = float(row["actual_concurrent_wave_us"])
        output.append({"candidate_id": candidate, "oof_scheme": scheme, "fold": fold, "source_phase": row["source_phase"], "model_id": row["model_id"], "configuration": row["configuration"], "topology_level": row["topology_level"], "vector_id": row["vector_id"], "vector_index": row["vector_index"], "pages": row["pages"], "predicted_us": prediction, "actual_us": actual, "absolute_error_us": abs(prediction-actual), "signed_error_us": prediction-actual})
    return output


def slice_metrics(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        for key in (("overall", "all"), ("model", row["model_id"]), ("configuration", row["configuration"]), ("configuration_topology", f"{row['configuration']}/{row['topology_level']}")):
            groups[key].append(row)
    output = []
    for (kind, value), rows in sorted(groups.items()):
        actual = math.fsum(float(r["actual_us"]) for r in rows); predicted = math.fsum(float(r["predicted_us"]) for r in rows); absolute = math.fsum(float(r["absolute_error_us"]) for r in rows)
        output.append({"candidate_id": rows[0]["candidate_id"], "oof_scheme": rows[0]["oof_scheme"], "slice_type": kind, "slice_value": value, "points": len(rows), "wape": absolute/actual, "signed_bias": (predicted-actual)/actual})
    return output


def evaluate(rows: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    all_predictions: list[dict[str, Any]] = []; all_slices: list[dict[str, Any]] = []; summaries: list[dict[str, Any]] = []
    cached_baselines: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for baseline in BASELINES:
        for scheme in SCHEMES:
            tests = [r for _, _, held in split_folds(rows, scheme) for r in held]
            predictions = prediction_rows(baseline, scheme, "fixed", tests, [baseline_value(baseline, r) for r in tests]); slices = slice_metrics(predictions)
            cached_baselines[(baseline, scheme)] = slices; all_predictions += predictions; all_slices += slices
    gates = contract["acceptance_gate"]
    for candidate in CANDIDATES:
        scheme_values: dict[str, Any] = {}; candidate_all_pass = True
        for scheme in SCHEMES:
            predictions = []
            for fold, training, held in split_folds(rows, scheme):
                model = fit_model(training, candidate)
                predictions += prediction_rows(candidate["candidate_id"], scheme, fold, held, [predict(model, r) for r in held])
            predictions.sort(key=lambda r: (r["source_phase"], r["model_id"], r["configuration"], r["topology_level"], int(r["vector_index"])))
            slices = slice_metrics(predictions); all_predictions += predictions; all_slices += slices
            overall = next(r for r in slices if r["slice_type"] == "overall"); models = [r for r in slices if r["slice_type"] == "model"]; configs = [r for r in slices if r["slice_type"] == "configuration"]; fine = [r for r in slices if r["slice_type"] == "configuration_topology"]
            base_overall = {b: next(r for r in cached_baselines[(b, scheme)] if r["slice_type"] == "overall") for b in BASELINES}
            base_configs = {b: {r["slice_value"]: r for r in cached_baselines[(b, scheme)] if r["slice_type"] == "configuration"} for b in BASELINES}
            checks = {
                "overall_wape": overall["wape"] <= gates["overall_wape_max"],
                "each_model_wape": all(r["wape"] <= gates["model_wape_max"] for r in models),
                "each_configuration_wape": all(r["wape"] <= gates["configuration_wape_max"] for r in configs),
                "each_configuration_topology_wape": all(r["wape"] <= gates["configuration_topology_wape_max"] for r in fine),
                "overall_bias": abs(overall["signed_bias"]) <= gates["overall_absolute_bias_max"],
                "each_model_bias": all(abs(r["signed_bias"]) <= gates["model_absolute_bias_max"] for r in models),
                "each_configuration_bias": all(abs(r["signed_bias"]) <= gates["configuration_absolute_bias_max"] for r in configs),
                "each_configuration_topology_bias": all(abs(r["signed_bias"]) <= gates["configuration_topology_absolute_bias_max"] for r in fine),
                "positive": all(r["predicted_us"] > 0 for r in predictions),
                "improves_all_baselines_overall": all(overall["wape"] < base_overall[b]["wape"] for b in BASELINES),
                "improves_best_baseline_each_configuration": all(r["wape"] < min(base_configs[b][r["slice_value"]]["wape"] for b in BASELINES) for r in configs),
            }
            passed = all(checks.values()); candidate_all_pass &= passed
            scheme_values[scheme] = {"pass": passed, "checks": checks, "points": len(predictions), "overall_wape": overall["wape"], "overall_signed_bias": overall["signed_bias"], "max_model_wape": max(r["wape"] for r in models), "max_configuration_wape": max(r["wape"] for r in configs), "max_configuration_topology_wape": max(r["wape"] for r in fine), "max_model_absolute_bias": max(abs(r["signed_bias"]) for r in models)}
        summaries.append({"candidate_id": candidate["candidate_id"], "complexity_rank": candidate["rank"], "feature_family": candidate["feature_family"], "target_guard": candidate_all_pass, "schemes": scheme_values})
    selected = next((r for r in summaries if r["target_guard"]), None)
    return {"candidates": summaries, "selected": selected, "predictions": all_predictions, "slices": all_slices}
