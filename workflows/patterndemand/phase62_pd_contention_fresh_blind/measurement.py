#!/usr/bin/env python3
"""Reuse Phase60 raw validation and evaluate the frozen R61 model on blind points."""
from __future__ import annotations

import importlib.util
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
P60 = HERE.parent / "phase60_pd_multi_endpoint_composability"
sys.path.insert(0, str(HERE))

import contracts as phase62_contracts  # noqa: E402
from contracts import contract, load_json, payload_pairs  # noqa: E402


def _load_phase60_measurement():
    sys.modules["contracts"] = phase62_contracts
    spec = importlib.util.spec_from_file_location("phase62_pinned_phase60_measurement", P60 / "measurement.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned Phase60 raw validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P60_MEASUREMENT = _load_phase60_measurement()
validate_raw = P60_MEASUREMENT.validate_raw


def _frozen_model() -> dict[str, Any]:
    value = load_json(ROOT / "experiment-results/phase61_pd_contention_correction/model/contention_correction.json")
    expected = contract()["frozen_correction_contract"]
    coefficients = value.get("groups", {}).get("__global__", {})
    actual = {key: float(coefficients[key]) for key in ("intercept_us", "beta_max", "beta_min")}
    if value.get("candidate_id") != expected["candidate_id"] or actual != expected["coefficients"]:
        raise RuntimeError({"frozen_model_mismatch": actual, "expected": expected})
    return value


def _corrected(model: dict[str, Any], c0: float, c1: float) -> float:
    coefficients = model["groups"]["__global__"]
    value = (
        float(coefficients["intercept_us"])
        + float(coefficients["beta_max"]) * max(c0, c1)
        + float(coefficients["beta_min"]) * min(c0, c1)
    )
    return max(float(model["prediction_floor_us"]), value)


def _cv(values: list[float]) -> float:
    return statistics.stdev(values) / statistics.mean(values) if len(values) > 1 and statistics.mean(values) else 0.0


def build_blind_analysis(plan: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    if not raw["complete"]:
        raise RuntimeError("cannot aggregate incomplete Phase62 raw")
    spec = contract()
    model = _frozen_model()
    curves = P60_MEASUREMENT._phase51_curve_map()
    points: list[dict[str, Any]] = []
    replica_points: list[dict[str, Any]] = []
    spreads: list[dict[str, Any]] = []
    for model_id in spec["selected_models"]:
        for configuration in spec["research_scope"]["fixed_configurations"]:
            for level in ("L1", "L2", "L3"):
                curve = curves[(model_id, level)]
                for pair in payload_pairs(model_id):
                    replicas = []
                    for replica in (0, 1):
                        measurement_id = f"{model_id}__{configuration.lower()}__{level.lower()}__r{replica}"
                        selected = [
                            next(row for row in rows if row["pair_id"] == pair["pair_id"])
                            for rows in raw["records"][measurement_id].values()
                        ]
                        concurrent = [float(row["concurrent_wave"]["wave_latency_us"]["median"]) for row in selected]
                        solo0 = [float(row["solo_flow0"]["wave_latency_us"]["median"]) for row in selected]
                        solo1 = [float(row["solo_flow1"]["wave_latency_us"]["median"]) for row in selected]
                        item = {
                            "model_id": model_id,
                            "configuration": configuration,
                            "topology_level": level,
                            "pair_id": pair["pair_id"],
                            "replica_id": replica,
                            "measurement_id": measurement_id,
                            "payload_bytes0": pair["payload_bytes0"],
                            "payload_bytes1": pair["payload_bytes1"],
                            "repeat_count": len(selected),
                            "concurrent_wave_us": statistics.median(concurrent),
                            "solo0_us": statistics.median(solo0),
                            "solo1_us": statistics.median(solo1),
                            "matched_solo_ideal_us": max(statistics.median(solo0), statistics.median(solo1)),
                            "concurrent_repeat_cv": _cv(concurrent),
                        }
                        replicas.append(item)
                        replica_points.append(item)
                    actual = max(row["concurrent_wave_us"] for row in replicas)
                    matched = max(row["matched_solo_ideal_us"] for row in replicas)
                    c0 = P60_MEASUREMENT._interpolate(curve, pair["payload_bytes0"])
                    c1 = P60_MEASUREMENT._interpolate(curve, pair["payload_bytes1"])
                    baseline = max(c0, c1)
                    corrected = _corrected(model, c0, c1)
                    spread = (
                        max(row["concurrent_wave_us"] for row in replicas)
                        - min(row["concurrent_wave_us"] for row in replicas)
                    ) / actual if actual else 0.0
                    points.append({
                        "model_id": model_id,
                        "configuration": configuration,
                        "topology_level": level,
                        "pair_id": pair["pair_id"],
                        "page_count0": pair["page_count0"],
                        "page_count1": pair["page_count1"],
                        "payload_bytes0": pair["payload_bytes0"],
                        "payload_bytes1": pair["payload_bytes1"],
                        "phase51_flow0_us": c0,
                        "phase51_flow1_us": c1,
                        "phase51_max_us": baseline,
                        "frozen_corrected_us": corrected,
                        "matched_solo_ideal_us": matched,
                        "actual_concurrent_wave_us": actual,
                        "phase51_absolute_error_us": abs(baseline - actual),
                        "corrected_absolute_error_us": abs(corrected - actual),
                        "matched_solo_absolute_error_us": abs(matched - actual),
                        "phase51_signed_error_us": baseline - actual,
                        "corrected_signed_error_us": corrected - actual,
                        "cross_replica_relative_spread": spread,
                    })
                    spreads.append({
                        "model_id": model_id,
                        "configuration": configuration,
                        "topology_level": level,
                        "pair_id": pair["pair_id"],
                        "replica0_us": replicas[0]["concurrent_wave_us"],
                        "replica1_us": replicas[1]["concurrent_wave_us"],
                        "official_us": actual,
                        "relative_spread": spread,
                        "above_threshold": spread > float(spec["measurement_contract"]["cross_replica_relative_spread_diagnostic_threshold"]),
                    })
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in points:
        keys = [
            ("overall", "all"),
            ("configuration", row["configuration"]),
            ("topology", row["topology_level"]),
            ("model", row["model_id"]),
            ("configuration_topology", f"{row['configuration']}/{row['topology_level']}"),
        ]
        for key in keys:
            groups[key].append(row)
    metrics = []
    for (slice_type, slice_value), rows in sorted(groups.items()):
        actual = math.fsum(float(row["actual_concurrent_wave_us"]) for row in rows)
        baseline_sum = math.fsum(float(row["phase51_max_us"]) for row in rows)
        corrected_sum = math.fsum(float(row["frozen_corrected_us"]) for row in rows)
        metrics.append({
            "slice_type": slice_type,
            "slice_value": slice_value,
            "points": len(rows),
            "phase51_wape": math.fsum(float(row["phase51_absolute_error_us"]) for row in rows) / actual,
            "corrected_wape": math.fsum(float(row["corrected_absolute_error_us"]) for row in rows) / actual,
            "phase51_signed_bias": (baseline_sum - actual) / actual,
            "corrected_signed_bias": (corrected_sum - actual) / actual,
        })
    gate = spec["fresh_blind_acceptance_gate"]
    overall = next(row for row in metrics if row["slice_type"] == "overall")
    slices = [row for row in metrics if row["slice_type"] == "configuration_topology"]
    checks = {
        "corrected_overall_wape": overall["corrected_wape"] <= float(gate["corrected_overall_wape_max"]),
        "corrected_configuration_topology_wape": all(row["corrected_wape"] <= float(gate["corrected_each_configuration_topology_wape_max"]) for row in slices),
        "corrected_overall_signed_bias": abs(overall["corrected_signed_bias"]) <= float(gate["corrected_overall_absolute_signed_bias_max"]),
        "corrected_configuration_topology_signed_bias": all(abs(row["corrected_signed_bias"]) <= float(gate["corrected_each_configuration_topology_absolute_signed_bias_max"]) for row in slices),
        "all_predictions_positive": all(float(row["frozen_corrected_us"]) > 0 for row in points),
        "strictly_improves_uncorrected_phase51_max": overall["corrected_wape"] < overall["phase51_wape"],
    }
    passed = all(checks.values())
    return {
        "points": points,
        "replica_points": replica_points,
        "spreads": spreads,
        "metrics": metrics,
        "decision": {
            "scientific_outcome": "FROZEN_CONTENTION_CORRECTION_FRESH_BLIND_PASS" if passed else "FROZEN_CONTENTION_CORRECTION_FRESH_BLIND_FAIL",
            "fresh_blind_gate_pass": passed,
            "checks": checks,
            "thresholds": gate,
            "training_performed": False,
            "recalibration_performed": False,
            "blind_labels_used_for_fitting": False,
        },
    }
