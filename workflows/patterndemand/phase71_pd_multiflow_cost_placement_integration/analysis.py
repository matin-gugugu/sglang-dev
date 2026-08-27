#!/usr/bin/env python3
"""Deterministic Phase71 histogram-to-multiflow-cost integration."""
from __future__ import annotations

import bisect
import csv
import gzip
import io
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

METHODS = ("h0", "h0_plus_dnn_residual")
LEVELS = ("L1", "L2", "L3")
POLICIES = ("bin_aligned", "cyclic_staggered", "opposed_extremes")
SIX_MODELS = (
    "qwen3-8b", "deepseek-v2-lite", "qwen3-30b-a3b", "llama-3.2-3b-instruct",
    "qwen2.5-14b-instruct", "mixtral-8x7b-instruct-v0.1",
)
R69_MODELS = ("qwen3-8b", "deepseek-v2-lite")
CONFIGURATIONS = {
    "P1D1": {"flow_count": 1, "physical_model": "phase51_single_link", "models": SIX_MODELS, "edges": ((0, 1),)},
    "P1D2": {"flow_count": 2, "physical_model": "frozen_r61_two_flow", "models": SIX_MODELS, "edges": ((0, 1), (0, 2))},
    "P2D1": {"flow_count": 2, "physical_model": "frozen_r61_two_flow", "models": SIX_MODELS, "edges": ((0, 2), (1, 2))},
    "P1D4": {"flow_count": 4, "physical_model": "frozen_r69_multiflow", "models": R69_MODELS, "edges": ((0, 1), (0, 2), (0, 3), (0, 4))},
    "P4D1": {"flow_count": 4, "physical_model": "frozen_r69_multiflow", "models": R69_MODELS, "edges": ((0, 4), (1, 4), (2, 4), (3, 4))},
    "P2D2_MATCHING": {"flow_count": 2, "physical_model": "frozen_r69_multiflow", "models": R69_MODELS, "edges": ((0, 2), (1, 3))},
    "P2D2_ALL_TO_ALL": {"flow_count": 4, "physical_model": "frozen_r69_multiflow", "models": R69_MODELS, "edges": ((0, 2), (0, 3), (1, 2), (1, 3))},
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refuse empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as target:
                target.write(buffer.getvalue().encode())
    else:
        path.write_text(buffer.getvalue(), encoding="utf-8")


def vector(row: dict[str, str], prefix: str) -> list[float]:
    values = [float(row[f"{prefix}_bin_{index:02d}"]) for index in range(12)]
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise RuntimeError({"invalid_vector": prefix, "profile": row.get("profile_id"), "model": row.get("model")})
    return values


def supported_configurations(model: str) -> tuple[str, ...]:
    return tuple(configuration for configuration, value in CONFIGURATIONS.items() if model in value["models"])


def validate_inputs(predictions: list[dict[str, str]], targets: list[dict[str, str]], curves: list[dict], layouts: list[dict], spec: dict) -> dict:
    errors = []
    if len(predictions) != 3600 or len(targets) != 1800:
        errors.append("row_counts")
    target_keys = {(row["profile_id"], row["model"]) for row in targets}
    prediction_keys = {(row["profile_id"], row["model"], row["method"]) for row in predictions}
    if len(target_keys) != len(targets) or prediction_keys != {(profile, model, method) for profile, model in target_keys for method in METHODS}:
        errors.append("prediction_target_keys")
    models = tuple(sorted({row["model"] for row in targets}))
    if set(models) != set(SIX_MODELS) or len({row["profile_id"] for row in targets}) != 300:
        errors.append("roster")
    curve_keys = {(row["model_id"], row["topology_level"]) for row in curves}
    if curve_keys != {(model, level) for model in SIX_MODELS for level in LEVELS} or len(curves) != 18:
        errors.append("curve_matrix")
    layout_keys = {row["model_id"] for row in layouts}
    if layout_keys != set(SIX_MODELS):
        errors.append("layout_matrix")
    scalar_max = 0.0
    for row in predictions:
        calls, logical = vector(row, "predicted_calls"), vector(row, "predicted_logical_bytes")
        scalar_max = max(scalar_max, abs(sum(calls) - float(row["predicted_total_calls_per_1000"])) / max(1.0, abs(float(row["predicted_total_calls_per_1000"]))), abs(sum(logical) - float(row["predicted_total_logical_bytes_per_1000"])) / max(1.0, abs(float(row["predicted_total_logical_bytes_per_1000"]))))
    for row in targets:
        calls, logical = vector(row, "target_calls"), vector(row, "target_logical_bytes")
        scalar_max = max(scalar_max, abs(sum(calls) - float(row["target_total_calls_per_1000"])) / max(1.0, abs(float(row["target_total_calls_per_1000"]))), abs(sum(logical) - float(row["target_total_logical_bytes_per_1000"])) / max(1.0, abs(float(row["target_total_logical_bytes_per_1000"]))))
    if scalar_max > 1e-12:
        errors.append("scalar_vector_mismatch")
    expected_units = sum(len(supported_configurations(row["model"])) for row in predictions)
    if expected_units != spec["expected_counts"]["supported_profile_model_method_configurations"]:
        errors.append("supported_units")
    if errors:
        raise RuntimeError({"phase71_inputs": errors})
    return {"prediction_rows": len(predictions), "target_rows": len(targets), "profiles": 300, "models": list(SIX_MODELS), "supported_units": expected_units, "scalar_max_relative_difference": scalar_max}


def _curve_map(curves: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["model_id"], row["topology_level"]): row for row in curves}


def _layout_map(layouts: list[dict]) -> dict[str, dict]:
    return {row["model_id"]: row for row in layouts}


def _interpolate(curve: dict, payload: float) -> float:
    knots = curve["knots"]
    xs = [float(row["payload_bytes"]) for row in knots]
    ys = [float(row["official_latency_us"]) for row in knots]
    if payload < xs[0] - 1e-7 or payload > xs[-1] + 1e-7:
        raise RuntimeError({"payload_outside_curve": payload, "support": [xs[0], xs[-1]]})
    payload = min(max(payload, xs[0]), xs[-1])
    if payload in xs:
        return ys[xs.index(payload)]
    right = bisect.bisect_right(xs, payload)
    left = right - 1
    fraction = (math.log2(payload) - math.log2(xs[left])) / (math.log2(xs[right]) - math.log2(xs[left]))
    return ys[left] + fraction * (ys[right] - ys[left])


def _distribution(calls: list[float], logical: list[float], bytes_per_page: float, audit: dict[str, float]) -> tuple[list[tuple[float, float]], float]:
    total = math.fsum(calls)
    if total <= 0:
        raise RuntimeError("histogram has no calls")
    merged: dict[float, float] = defaultdict(float)
    for count, byte_count in zip(calls, logical):
        if count <= 1e-12:
            if byte_count > 1e-6:
                raise RuntimeError("positive bytes with zero calls")
            continue
        if byte_count <= 0:
            raise RuntimeError("positive calls with zero bytes")
        raw_page = byte_count / count / bytes_per_page
        page = min(64.0, max(1.0, raw_page))
        if raw_page < 1.0:
            audit["low_clamped_calls"] += count
        if raw_page > 64.0:
            audit["high_clamped_calls"] += count
        audit["processed_calls"] += count
        merged[page] += count
    entries = [(page, weight / total) for page, weight in sorted(merged.items())]
    if not math.isclose(math.fsum(weight for _page, weight in entries), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("distribution mass")
    return entries, total


def _cdf_boundaries(entries: list[tuple[float, float]]) -> list[float]:
    cumulative = 0.0
    output = []
    for _page, weight in entries[:-1]:
        cumulative += weight
        if 1e-14 < cumulative < 1.0 - 1e-14:
            output.append(cumulative)
    return output


def _quantile(entries: list[tuple[float, float]], q: float) -> float:
    q = q % 1.0
    cumulative = 0.0
    for page, weight in entries:
        cumulative += weight
        if q < cumulative + 1e-14:
            return page
    return entries[-1][0]


def _mapped_quantile(policy: str, edge: int, flows: int, q: float) -> float:
    if policy == "bin_aligned":
        return q
    if policy == "cyclic_staggered":
        return (q + edge / flows) % 1.0
    if policy == "opposed_extremes":
        return q if edge % 2 == 0 else 1.0 - q
    raise RuntimeError(f"unknown policy: {policy}")


def _integration_intervals(entries: list[tuple[float, float]], flows: int, policy: str) -> list[tuple[float, float]]:
    points = {0.0, 1.0}
    for edge in range(flows):
        for boundary in _cdf_boundaries(entries):
            if policy == "bin_aligned":
                value = boundary
            elif policy == "cyclic_staggered":
                value = (boundary - edge / flows) % 1.0
            elif policy == "opposed_extremes":
                value = boundary if edge % 2 == 0 else 1.0 - boundary
            else:
                raise RuntimeError(policy)
            if 1e-14 < value < 1.0 - 1e-14:
                points.add(value)
    ordered = sorted(points)
    return [(left, right) for left, right in zip(ordered, ordered[1:]) if right - left > 1e-14]


def _graph_features(costs: list[float], edges: tuple[tuple[int, int], ...]) -> tuple[float, float, float]:
    outbound: dict[int, float] = defaultdict(float)
    inbound: dict[int, float] = defaultdict(float)
    for cost, (sender, receiver) in zip(costs, edges):
        outbound[sender] += cost
        inbound[receiver] += cost
    return max(costs), max([*outbound.values(), *inbound.values()]), math.fsum(costs)


def _r61(costs: list[float], model: dict) -> float:
    group = model["groups"]["__global__"]
    high, low = max(costs), min(costs)
    return max(float(model["prediction_floor_us"]), float(group["intercept_us"]) + float(group["beta_max"]) * high + float(group["beta_min"]) * low)


def _r67(model_id: str, configuration: str, costs: list[float], pages: list[float], edges: tuple[tuple[int, int], ...], model: dict) -> float:
    m, busy, total = _graph_features(costs, edges)
    pmax = max(pages)
    prest = math.fsum(pages) - pmax
    features = (1.0, m, max(0.0, busy - m), max(0.0, total - busy), pmax, prest, math.sqrt(pmax), math.sqrt(prest))
    group = model["groups"][f"{model_id}|{configuration}"]
    return max(float(model["prediction_floor_us"]), math.fsum(float(coefficient) * feature for coefficient, feature in zip(group["coefficients"], features)))


def _r69(model_id: str, configuration: str, anchor: float, pages: list[float], model: dict) -> float:
    if configuration == "P2D2_MATCHING":
        features = (0.0, 0.0)
    else:
        pmax = max(pages)
        mean_other = (math.fsum(pages) - pmax) / max(1, len(pages) - 1)
        features = (max(0.0, pmax - 32.0), max(0.0, mean_other - 32.0))
    group = model["groups"][f"{model_id}|{configuration}"]
    return max(float(model["prediction_floor_us"]), anchor + math.fsum(float(coefficient) * feature for coefficient, feature in zip(group["coefficients"], features)))


def _wave_latency(model_id: str, configuration: str, level: str, pages: list[float], curves: dict, layouts: dict, r61: dict, r67: dict, r69: dict) -> float:
    info = CONFIGURATIONS[configuration]
    per_page = float(layouts[model_id]["knots"][0]["payload_bytes"])
    costs = [_interpolate(curves[(model_id, level)], per_page * page) for page in pages]
    if configuration == "P1D1":
        return costs[0]
    if info["physical_model"] == "frozen_r61_two_flow":
        return _r61(costs, r61)
    anchor = _r67(model_id, configuration, costs, pages, info["edges"], r67)
    return _r69(model_id, configuration, anchor, pages, r69)


def histogram_costs(calls: list[float], logical: list[float], model_id: str, configuration: str, level: str, assets: dict, audit: dict[str, float]) -> dict[str, float]:
    info = CONFIGURATIONS[configuration]
    flows = int(info["flow_count"])
    per_page = float(assets["layouts"][model_id]["knots"][0]["payload_bytes"])
    entries, total_calls = _distribution(calls, logical, per_page, audit)
    wave_mass = total_calls / flows
    output = {}
    for policy in POLICIES:
        average = 0.0
        for left, right in _integration_intervals(entries, flows, policy):
            q = (left + right) / 2.0
            pages = [_quantile(entries, _mapped_quantile(policy, edge, flows, q)) for edge in range(flows)]
            average += (right - left) * _wave_latency(model_id, configuration, level, pages, assets["curves"], assets["layouts"], assets["r61"], assets["r67"], assets["r69"])
        output[policy] = wave_mass * average
    if any(not math.isfinite(value) or value <= 0 for value in output.values()):
        raise RuntimeError({"nonpositive_cost": output})
    return output


def cost_rows(predictions: list[dict[str, str]], targets: list[dict[str, str]], curves: list[dict], layouts: list[dict], r61: dict, r67: dict, r69: dict) -> tuple[list[dict], dict]:
    target_map = {(row["profile_id"], row["model"]): row for row in targets}
    assets = {"curves": _curve_map(curves), "layouts": _layout_map(layouts), "r61": r61, "r67": r67, "r69": r69}
    audits: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    teacher_cache: dict[tuple[str, str, str, str], dict[str, float]] = {}
    output = []
    for prediction in predictions:
        model = prediction["model"]
        target = target_map[(prediction["profile_id"], model)]
        pc, pb = vector(prediction, "predicted_calls"), vector(prediction, "predicted_logical_bytes")
        tc, tb = vector(target, "target_calls"), vector(target, "target_logical_bytes")
        for configuration in supported_configurations(model):
            info = CONFIGURATIONS[configuration]
            for level in LEVELS:
                predicted = histogram_costs(pc, pb, model, configuration, level, assets, audits[f"prediction/{prediction['method']}/{configuration}/{level}"])
                teacher_key = (prediction["profile_id"], model, configuration, level)
                if teacher_key not in teacher_cache:
                    teacher_cache[teacher_key] = histogram_costs(tc, tb, model, configuration, level, assets, audits[f"teacher/{configuration}/{level}"])
                teacher = teacher_cache[teacher_key]
                row = {
                    "profile_id": prediction["profile_id"], "source": prediction["source"], "segment": prediction["segment"],
                    "model": model, "method": prediction["method"], "configuration": configuration,
                    "flow_count": info["flow_count"], "physical_model": info["physical_model"], "topology_level": level,
                    "request_count": target["request_count"],
                }
                for policy in POLICIES:
                    row[f"predicted_cost_{policy}_us_per_1000"] = predicted[policy]
                    row[f"teacher_cost_{policy}_us_per_1000"] = teacher[policy]
                official_p, official_t = predicted["bin_aligned"], teacher["bin_aligned"]
                row.update({"absolute_error_us_per_1000": abs(official_p - official_t), "absolute_percentage_error": abs(official_p - official_t) / max(official_t, 1e-12), "signed_error_us_per_1000": official_p - official_t})
                output.append(row)
    output.sort(key=lambda row: (row["profile_id"], row["model"], row["method"], row["configuration"], LEVELS.index(row["topology_level"])))
    interpolation = {"schema_version": "phase71-interpolation-audit-v1", "page_support": [1, 64], "roles": {key: dict(value) for key, value in sorted(audits.items())}, "teacher_cache_entries": len(teacher_cache), "exact_piecewise_cdf_integration": True}
    return output, interpolation


def _slices(row: dict) -> tuple[tuple[str, str], ...]:
    return (("overall", "all"), ("model", row["model"]), ("segment", row["segment"]))


def aggregate_cost(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        for kind, value in _slices(row):
            groups[(row["method"], row["configuration"], row["topology_level"], kind, value)].append(row)
    output = []
    for (method, configuration, level, kind, value), items in sorted(groups.items()):
        teacher = math.fsum(float(row["teacher_cost_bin_aligned_us_per_1000"]) for row in items)
        predicted = math.fsum(float(row["predicted_cost_bin_aligned_us_per_1000"]) for row in items)
        output.append({"method": method, "configuration": configuration, "topology_level": level, "slice_type": kind, "slice_value": value, "cases": len(items), "cost_mape": statistics.fmean(float(row["absolute_percentage_error"]) for row in items), "cost_wape": math.fsum(float(row["absolute_error_us_per_1000"]) for row in items) / teacher, "signed_bias": (predicted - teacher) / teacher, "predicted_cost_us_per_1000_sum": predicted, "teacher_cost_us_per_1000_sum": teacher})
    return output


def compare_cost(metrics: list[dict]) -> list[dict]:
    groups: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in metrics:
        groups[(row["configuration"], row["topology_level"], row["slice_type"], row["slice_value"])][row["method"]] = row
    output = []
    for (configuration, level, kind, value), methods in sorted(groups.items()):
        h0, dnn = methods["h0"], methods["h0_plus_dnn_residual"]
        output.append({"configuration": configuration, "topology_level": level, "slice_type": kind, "slice_value": value, "cases": h0["cases"], "h0_cost_mape": h0["cost_mape"], "dnn_cost_mape": dnn["cost_mape"], "cost_mape_ratio": dnn["cost_mape"] / max(h0["cost_mape"], 1e-12), "h0_cost_wape": h0["cost_wape"], "dnn_cost_wape": dnn["cost_wape"], "cost_wape_ratio": dnn["cost_wape"] / max(h0["cost_wape"], 1e-12), "h0_signed_bias": h0["signed_bias"], "dnn_signed_bias": dnn["signed_bias"], "strict_mape_and_wape_improvement": dnn["cost_mape"] < h0["cost_mape"] and dnn["cost_wape"] < h0["cost_wape"]})
    return output


def placement(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["profile_id"], row["model"], row["method"], row["configuration"])].append(row)
    output = []
    for key, items in sorted(groups.items()):
        if len(items) != 3 or {row["topology_level"] for row in items} != set(LEVELS):
            raise RuntimeError({"placement_candidates": key})
        base = {name: items[0][name] for name in ("profile_id", "source", "segment", "model", "method", "configuration", "flow_count", "physical_model", "request_count")}
        row = {**base, "ranking_scope": "communication_only_fixed_configuration"}
        predicted_choices, teacher_choices = [], []
        for policy in POLICIES:
            pkey = f"predicted_cost_{policy}_us_per_1000"
            tkey = f"teacher_cost_{policy}_us_per_1000"
            predicted = sorted(items, key=lambda item: (float(item[pkey]), LEVELS.index(item["topology_level"])))
            teacher = sorted(items, key=lambda item: (float(item[tkey]), LEVELS.index(item["topology_level"])))
            selected, oracle = predicted[0], teacher[0]
            predicted_choices.append(selected["topology_level"])
            teacher_choices.append(oracle["topology_level"])
            oracle_cost = float(oracle[tkey])
            row.update({f"selected_topology_{policy}": selected["topology_level"], f"oracle_topology_{policy}": oracle["topology_level"], f"agreement_{policy}": selected["topology_level"] == oracle["topology_level"], f"teacher_regret_{policy}": (float(selected[tkey]) - oracle_cost) / max(oracle_cost, 1e-12), f"predicted_margin_{policy}": (float(predicted[1][pkey]) - float(selected[pkey])) / max(float(selected[pkey]), 1e-12)})
        row["predicted_choice_stable_across_wave_policies"] = len(set(predicted_choices)) == 1
        row["teacher_choice_stable_across_wave_policies"] = len(set(teacher_choices)) == 1
        output.append(row)
    return output


def aggregate_placement(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        for kind, value in _slices(row):
            groups[(row["method"], row["configuration"], kind, value)].append(row)
    output = []
    for (method, configuration, kind, value), items in sorted(groups.items()):
        output.append({"method": method, "configuration": configuration, "slice_type": kind, "slice_value": value, "cases": len(items), "agreement_rate": statistics.fmean(float(row["agreement_bin_aligned"]) for row in items), "mean_teacher_regret": statistics.fmean(float(row["teacher_regret_bin_aligned"]) for row in items), "max_teacher_regret": max(float(row["teacher_regret_bin_aligned"]) for row in items), "predicted_choice_wave_stability_rate": statistics.fmean(float(row["predicted_choice_stable_across_wave_policies"]) for row in items), "teacher_choice_wave_stability_rate": statistics.fmean(float(row["teacher_choice_stable_across_wave_policies"]) for row in items)})
    return output


def compare_placement(metrics: list[dict]) -> list[dict]:
    groups: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in metrics:
        groups[(row["configuration"], row["slice_type"], row["slice_value"])][row["method"]] = row
    output = []
    for (configuration, kind, value), methods in sorted(groups.items()):
        h0, dnn = methods["h0"], methods["h0_plus_dnn_residual"]
        output.append({"configuration": configuration, "slice_type": kind, "slice_value": value, "cases": h0["cases"], "h0_agreement_rate": h0["agreement_rate"], "dnn_agreement_rate": dnn["agreement_rate"], "agreement_delta": dnn["agreement_rate"] - h0["agreement_rate"], "h0_mean_teacher_regret": h0["mean_teacher_regret"], "dnn_mean_teacher_regret": dnn["mean_teacher_regret"], "regret_delta": dnn["mean_teacher_regret"] - h0["mean_teacher_regret"], "weak_agreement_and_regret_improvement": dnn["agreement_rate"] >= h0["agreement_rate"] and dnn["mean_teacher_regret"] <= h0["mean_teacher_regret"] + 1e-15})
    return output


def wave_sensitivity(costs: list[dict], decisions: list[dict]) -> list[dict]:
    decision_map = {(row["profile_id"], row["model"], row["method"], row["configuration"]): row for row in decisions}
    output = []
    for configuration in CONFIGURATIONS:
        eligible = [row for row in costs if row["configuration"] == configuration]
        for role_method in ("h0", "h0_plus_dnn_residual", "teacher"):
            if role_method == "teacher":
                items = [row for row in eligible if row["method"] == "h0"]
                prefix = "teacher_cost"
                selected_key = "teacher_choice_stable_across_wave_policies"
            else:
                items = [row for row in eligible if row["method"] == role_method]
                prefix = "predicted_cost"
                selected_key = "predicted_choice_stable_across_wave_policies"
            ranges = []
            for row in items:
                values = [float(row[f"{prefix}_{policy}_us_per_1000"]) for policy in POLICIES]
                ranges.append((max(values) - min(values)) / max(values[0], 1e-12))
            unique_decisions = {key: value for key, value in decision_map.items() if key[3] == configuration and (role_method == "teacher" and key[2] == "h0" or key[2] == role_method)}
            output.append({"configuration": configuration, "role_method": role_method, "cost_rows": len(items), "placement_cases": len(unique_decisions), "mean_relative_cost_range": statistics.fmean(ranges), "max_relative_cost_range": max(ranges), "placement_stability_rate": statistics.fmean(float(row[selected_key]) for row in unique_decisions.values()), "official_policy": "bin_aligned", "diagnostic_policies_selected": False})
    return output


def build_analysis(predictions: list[dict[str, str]], targets: list[dict[str, str]], curves: list[dict], layouts: list[dict], r61: dict, r67: dict, r69: dict, spec: dict) -> dict[str, Any]:
    input_audit = validate_inputs(predictions, targets, curves, layouts, spec)
    costs, interpolation = cost_rows(predictions, targets, curves, layouts, r61, r67, r69)
    cost_metrics = aggregate_cost(costs)
    cost_comparison = compare_cost(cost_metrics)
    decisions = placement(costs)
    placement_metrics = aggregate_placement(decisions)
    placement_comparison = compare_placement(placement_metrics)
    sensitivity = wave_sensitivity(costs, decisions)
    counts = spec["expected_counts"]
    actual = {"unit_configuration_topology_cost_rows": len(costs), "placement_decision_rows": len(decisions), "cost_metric_rows": len(cost_metrics), "cost_comparison_rows": len(cost_comparison), "placement_metric_rows": len(placement_metrics), "placement_comparison_rows": len(placement_comparison), "wave_sensitivity_rows": len(sensitivity)}
    if any(actual[key] != counts[key] for key in actual):
        raise RuntimeError({"phase71_cardinality": actual, "expected": counts})
    cost_overall = [row for row in cost_comparison if row["slice_type"] == "overall"]
    placement_overall = [row for row in placement_comparison if row["slice_type"] == "overall"]
    checks = {"dnn_strict_cost_improvement_every_configuration_topology": all(row["strict_mape_and_wape_improvement"] for row in cost_overall), "dnn_weak_placement_improvement_every_configuration": all(row["weak_agreement_and_regret_improvement"] for row in placement_overall), "official_wave_policy_pre_registered": True, "diagnostic_wave_policy_not_selected": True}
    confirmed = checks["dnn_strict_cost_improvement_every_configuration_topology"] and checks["dnn_weak_placement_improvement_every_configuration"]
    outcome = "MULTIFLOW_COST_PLACEMENT_INTEGRATION_CONFIRMED" if confirmed else "MULTIFLOW_COST_PLACEMENT_INTEGRATION_MIXED_RETAIN_AS_EVIDENCE"
    return {"input_audit": input_audit, "costs": costs, "cost_metrics": cost_metrics, "cost_comparison": cost_comparison, "decisions": decisions, "placement_metrics": placement_metrics, "placement_comparison": placement_comparison, "wave_sensitivity": sensitivity, "interpolation": interpolation, "decision": {"scientific_outcome": outcome, "checks": checks, "execution_pass_independent_of_outcome": True}}
