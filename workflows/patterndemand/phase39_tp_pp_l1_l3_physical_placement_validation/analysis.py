#!/usr/bin/env python3
"""Deterministic Phase39 histogram convolution and placement metrics."""

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

NUMERIC_HISTOGRAM_FIELDS = (
    "calls_mape", "calls_wape", "bytes_mape", "bytes_wape",
    "mean_histogram_l1", "mean_histogram_tv", "mean_normalized_log_payload_emd",
    "common_reference_cost_mape", "common_reference_cost_wape",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def deterministic_gzip(path: Path, text: str) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as output:
            output.write(text.encode("utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"拒绝写空CSV：{path}")
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(name for name in row if name not in fields)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    if path.suffix == ".gz":
        deterministic_gzip(path, buffer.getvalue())
    else:
        path.write_text(buffer.getvalue(), encoding="utf-8")


def vector(row: dict, field: str) -> list[float]:
    values = [float(value) for value in json.loads(row[field])]
    if len(values) != 12 or not all(math.isfinite(value) and value >= 0 for value in values):
        raise RuntimeError(f"非法12-bin向量：example={row.get('example_id')}, field={field}")
    return values


def vector_sum(vectors: list[list[float]]) -> list[float]:
    return [sum(values) for values in zip(*vectors)]


def input_rows(root: Path, spec: dict) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    paths = {item["name"]: root / item["path"] for item in spec["pinned_inputs"]}
    predictions = []
    for parallelism, name in (("tp", "phase34c_tp_frozen_predictions"), ("pp", "phase34c_pp_frozen_predictions")):
        predictions.extend(
            row for row in read_csv(paths[name])
            if row["prediction_set"] == spec["prediction_filter"]["prediction_set"]
            and row["method"] == spec["prediction_filter"]["method"]
            and row["parallelism"] == parallelism
        )
    targets = read_csv(paths["phase34d_hfull_targets"])
    predictions.sort(key=lambda row: row["example_id"])
    targets.sort(key=lambda row: row["example_id"])
    return predictions, targets


def interpolate(curve: dict, payload_bytes: float, field: str) -> tuple[float, str]:
    knots = curve["knots"]
    payloads = [float(row["payload_bytes"]) for row in knots]
    latencies = [float(row[field]) for row in knots]
    if payloads != sorted(payloads) or len(payloads) != len(set(payloads)):
        raise RuntimeError(f"curve payload不严格递增：{curve['curve_id']}")
    payload = max(float(payload_bytes), 1.0)
    if payload <= payloads[0]:
        return latencies[0], "low" if payload < payloads[0] else "inside"
    if payload >= payloads[-1]:
        return latencies[-1], "high" if payload > payloads[-1] else "inside"
    right = bisect.bisect_right(payloads, payload)
    left = right - 1
    fraction = (math.log2(payload) - math.log2(payloads[left])) / (math.log2(payloads[right]) - math.log2(payloads[left]))
    return latencies[left] + fraction * (latencies[right] - latencies[left]), "inside"


def histogram_cost(calls: list[float], logical_bytes: list[float], curve: dict, field: str, audit: dict) -> float:
    total = 0.0
    for count, byte_count in zip(calls, logical_bytes):
        if count <= 1e-12:
            continue
        latency, position = interpolate(curve, byte_count / count, field)
        total += count * latency
        audit["nonempty_bins"] += 1
        audit["logical_calls"] += count
        if position != "inside":
            audit[f"{position}_clamped_bins"] += 1
            audit[f"{position}_clamped_calls"] += count
    return total


def relevant_curves(prediction: dict, curves: list[dict]) -> list[dict]:
    parallelism = prediction["parallelism"]
    size = int(prediction["parallel_size"])
    return [
        curve for curve in curves
        if curve["parallelism"] == parallelism
        and (parallelism == "pp" or int(curve["group_size"]) == size)
    ]


def cost_phase_rows(predictions: list[dict], targets: list[dict], curves: list[dict]) -> tuple[list[dict], dict]:
    target_by_id = {row["example_id"]: row for row in targets}
    if len(target_by_id) != len(targets) or {row["example_id"] for row in predictions} != set(target_by_id):
        raise RuntimeError("Phase39 prediction/target example_id集合不完全一致")
    audits = defaultdict(lambda: defaultdict(float))
    output = []
    for prediction in predictions:
        target = target_by_id[prediction["example_id"]]
        predicted_calls = vector(prediction, "predicted_calls_by_12bin_json")
        predicted_bytes = vector(prediction, "predicted_logical_bytes_by_12bin_json")
        teacher_calls = vector(target, "target_calls_by_12bin_json")
        teacher_bytes = vector(target, "target_logical_bytes_by_12bin_json")
        candidates = relevant_curves(prediction, curves)
        if {curve["topology_level"] for curve in candidates} != {"L1", "L2", "L3"}:
            raise RuntimeError({"missing_candidate_curves": prediction["example_id"]})
        for curve in candidates:
            values = {}
            for label, field in (("official", "official_latency_us"), ("lower", "lower_latency_us"), ("upper", "upper_latency_us")):
                values[f"predicted_{label}"] = histogram_cost(predicted_calls, predicted_bytes, curve, field, audits[(curve["curve_id"], f"prediction_{label}")])
                values[f"teacher_{label}"] = histogram_cost(teacher_calls, teacher_bytes, curve, field, audits[(curve["curve_id"], f"teacher_{label}")])
            predicted = values["predicted_official"]
            teacher = values["teacher_official"]
            output.append({
                "example_id": prediction["example_id"],
                "profile_id": prediction["profile_id"],
                "source": prediction["source"],
                "segment": prediction["segment"],
                "model": prediction["model"],
                "parallelism": prediction["parallelism"],
                "parallel_size": prediction["parallel_size"],
                "policy": prediction["policy"],
                "phase": prediction["phase"],
                "topology_level": curve["topology_level"],
                "placement_id": f"phase39_{prediction['parallelism']}_{curve['topology_level'].lower()}_physical",
                "curve_id": curve["curve_id"],
                "curve_evidence": "physical_measurement",
                "predicted_cost_us_per_1000": predicted,
                "teacher_cost_us_per_1000": teacher,
                "predicted_cost_lower_us_per_1000": values["predicted_lower"],
                "predicted_cost_upper_us_per_1000": values["predicted_upper"],
                "teacher_cost_lower_us_per_1000": values["teacher_lower"],
                "teacher_cost_upper_us_per_1000": values["teacher_upper"],
                "absolute_error_us_per_1000": abs(predicted - teacher),
                "absolute_percentage_error": abs(predicted - teacher) / max(teacher, 1e-12),
                "signed_error_us_per_1000": predicted - teacher,
                "teacher_kind": target["teacher_kind"],
                "evidence_set": "phase34_open_target_repeated_engineering_with_phase39_physical_curves",
            })
    compact = {f"{curve_id}/{role}": dict(values) for (curve_id, role), values in audits.items()}
    return output, compact


def combined_cost_rows(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[name] for name in ("profile_id", "model", "parallelism", "parallel_size", "policy", "topology_level"))
        groups[key].append(row)
    output = []
    for values in groups.values():
        if len(values) != 2 or {row["phase"] for row in values} != {"prefill", "decode"}:
            raise RuntimeError("每个placement配置必须恰好有prefill/decode")
        source = values[0]
        totals = {}
        for prefix in ("predicted_cost", "teacher_cost"):
            for suffix in ("us_per_1000", "lower_us_per_1000", "upper_us_per_1000"):
                field = f"{prefix}_{suffix}"
                totals[field] = sum(float(row[field]) for row in values)
        predicted = totals["predicted_cost_us_per_1000"]
        teacher = totals["teacher_cost_us_per_1000"]
        output.append({
            **{name: source[name] for name in (
                "profile_id", "source", "segment", "model", "parallelism", "parallel_size", "policy",
                "topology_level", "placement_id", "curve_id", "curve_evidence", "teacher_kind", "evidence_set",
            )},
            "example_id": source["example_id"].rsplit("/", 1)[0] + "/total",
            "phase": "total",
            **totals,
            "absolute_error_us_per_1000": abs(predicted - teacher),
            "absolute_percentage_error": abs(predicted - teacher) / max(teacher, 1e-12),
            "signed_error_us_per_1000": predicted - teacher,
        })
    return sorted(output, key=lambda row: (row["parallelism"], row["example_id"], row["topology_level"]))


def aggregate_cost_metrics(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        slices = (
            ("overall", "all"), ("model", row["model"]), ("policy", row["policy"]),
            ("parallel_size", str(row["parallel_size"])),
        )
        for slice_type, slice_value in slices:
            groups[(row["parallelism"], row["topology_level"], row["phase"], slice_type, str(slice_value))].append(row)
    output = []
    for (parallelism, level, phase, slice_type, slice_value), values in sorted(groups.items()):
        teacher = sum(float(row["teacher_cost_us_per_1000"]) for row in values)
        predicted = sum(float(row["predicted_cost_us_per_1000"]) for row in values)
        output.append({
            "parallelism": parallelism,
            "topology_level": level,
            "placement_id": f"phase39_{parallelism}_{level.lower()}_physical",
            "curve_ids_json": json.dumps(sorted({row["curve_id"] for row in values}), separators=(",", ":")),
            "curve_evidence": "physical_measurement",
            "phase": phase,
            "slice_type": slice_type,
            "slice_value": slice_value,
            "cases": len(values),
            "cost_mape": statistics.fmean(float(row["absolute_percentage_error"]) for row in values),
            "cost_wape": sum(float(row["absolute_error_us_per_1000"]) for row in values) / max(teacher, 1e-12),
            "signed_bias": (predicted - teacher) / max(teacher, 1e-12),
            "evidence_set": values[0]["evidence_set"],
        })
    return output


def histogram_tv(predicted: list[float], actual: list[float]) -> float:
    predicted_total = max(sum(predicted), 1e-12)
    actual_total = max(sum(actual), 1e-12)
    return sum(abs(p / predicted_total - a / actual_total) for p, a in zip(predicted, actual)) / 2


def normalized_log_emd(predicted: list[float], actual: list[float], edges: list[float]) -> float:
    predicted_total = max(sum(predicted), 1e-12)
    actual_total = max(sum(actual), 1e-12)
    p_cdf, a_cdf, p_run, a_run = [], [], 0.0, 0.0
    for p_value, a_value in zip(predicted, actual):
        p_run += p_value / predicted_total
        a_run += a_value / actual_total
        p_cdf.append(p_run); a_cdf.append(a_run)
    centers = [(math.log2(left) + math.log2(right)) / 2 for left, right in zip(edges[:-1], edges[1:])]
    area = sum(abs(p_cdf[index] - a_cdf[index]) * (centers[index + 1] - centers[index]) for index in range(len(centers) - 1))
    return area / (math.log2(edges[-1]) - math.log2(edges[0]))


def histogram_record(prediction: dict, target: dict, phase: str, p_calls: list[float], p_bytes: list[float], a_calls: list[float], a_bytes: list[float], edges: list[float]) -> dict:
    p_calls_total, a_calls_total = sum(p_calls), sum(a_calls)
    p_bytes_total, a_bytes_total = sum(p_bytes), sum(a_bytes)
    p_reference = 5.0 * p_calls_total + p_bytes_total / 100e9 * 1e6
    a_reference = 5.0 * a_calls_total + a_bytes_total / 100e9 * 1e6
    tv = histogram_tv(p_calls, a_calls)
    return {
        "profile_id": prediction["profile_id"], "source": prediction["source"], "model": prediction["model"],
        "parallelism": prediction["parallelism"], "parallel_size": prediction["parallel_size"], "policy": prediction["policy"],
        "method": "h0_plus_dnn_residual", "phase": phase,
        "actual_total_calls": a_calls_total, "predicted_total_calls": p_calls_total,
        "calls_absolute_error": abs(p_calls_total - a_calls_total), "calls_ape": abs(p_calls_total - a_calls_total) / max(a_calls_total, 1e-12),
        "actual_total_logical_bytes": a_bytes_total, "predicted_total_logical_bytes": p_bytes_total,
        "bytes_absolute_error": abs(p_bytes_total - a_bytes_total), "bytes_ape": abs(p_bytes_total - a_bytes_total) / max(a_bytes_total, 1e-12),
        "histogram_l1": 2 * tv, "histogram_tv": tv,
        "normalized_log_payload_emd": normalized_log_emd(p_calls, a_calls, edges),
        "actual_common_reference_cost_us": a_reference, "predicted_common_reference_cost_us": p_reference,
        "cost_absolute_error": abs(p_reference - a_reference), "cost_ape": abs(p_reference - a_reference) / max(a_reference, 1e-12),
    }


def frozen_histogram_metrics(predictions: list[dict], targets: list[dict], edges: list[float]) -> list[dict]:
    target_by_id = {row["example_id"]: row for row in targets}
    records = []
    grouped = defaultdict(list)
    for prediction in predictions:
        target = target_by_id[prediction["example_id"]]
        values = (
            vector(prediction, "predicted_calls_by_12bin_json"), vector(prediction, "predicted_logical_bytes_by_12bin_json"),
            vector(target, "target_calls_by_12bin_json"), vector(target, "target_logical_bytes_by_12bin_json"),
        )
        records.append(histogram_record(prediction, target, prediction["phase"], *values, edges))
        key = tuple(prediction[name] for name in ("profile_id", "model", "parallelism", "parallel_size", "policy"))
        grouped[key].append((prediction, target, *values))
    for values in grouped.values():
        if len(values) != 2 or {value[0]["phase"] for value in values} != {"prefill", "decode"}:
            raise RuntimeError("冻结直方图配置缺少两个phase")
        prediction, target = values[0][0], values[0][1]
        record = histogram_record(
            prediction, target, "total",
            vector_sum([value[2] for value in values]), vector_sum([value[3] for value in values]),
            vector_sum([value[4] for value in values]), vector_sum([value[5] for value in values]), edges,
        )
        p_phase = [item for value in values for item in value[2]]
        a_phase = [item for value in values for item in value[4]]
        record["histogram_tv"] = histogram_tv(p_phase, a_phase)
        record["histogram_l1"] = 2 * record["histogram_tv"]
        records.append(record)
    groups = defaultdict(list)
    for row in records:
        for slice_type, slice_value in (
            ("overall", "all"), ("model", row["model"]), ("policy", row["policy"]),
            ("parallel_size", row["parallel_size"]), ("source", row["source"]),
        ):
            groups[(row["parallelism"], row["phase"], slice_type, str(slice_value))].append(row)
    output = []
    for (parallelism, phase, slice_type, slice_value), values in sorted(groups.items()):
        actual_calls = sum(float(row["actual_total_calls"]) for row in values)
        actual_bytes = sum(float(row["actual_total_logical_bytes"]) for row in values)
        actual_reference = sum(float(row["actual_common_reference_cost_us"]) for row in values)
        output.append({
            "parallelism": parallelism, "method": "h0_plus_dnn_residual", "phase": phase,
            "slice_type": slice_type, "slice_value": slice_value, "cases": len(values),
            "calls_mape": statistics.fmean(float(row["calls_ape"]) for row in values),
            "calls_wape": sum(float(row["calls_absolute_error"]) for row in values) / max(actual_calls, 1e-12),
            "bytes_mape": statistics.fmean(float(row["bytes_ape"]) for row in values),
            "bytes_wape": sum(float(row["bytes_absolute_error"]) for row in values) / max(actual_bytes, 1e-12),
            "mean_histogram_l1": statistics.fmean(float(row["histogram_l1"]) for row in values),
            "mean_histogram_tv": statistics.fmean(float(row["histogram_tv"]) for row in values),
            "mean_normalized_log_payload_emd": statistics.fmean(float(row["normalized_log_payload_emd"]) for row in values),
            "common_reference_cost_mape": statistics.fmean(float(row["cost_ape"]) for row in values),
            "common_reference_cost_wape": sum(float(row["cost_absolute_error"]) for row in values) / max(actual_reference, 1e-12),
            "evidence_set": "phase34_blind_six_model",
        })
    return output


def compare_histogram_metrics(recomputed: list[dict], official_rows: list[dict]) -> dict:
    official = [row for row in official_rows if row["method"] == "h0_plus_dnn_residual" and row["evidence_set"] == "phase34_blind_six_model"]
    key = lambda row: (row["parallelism"], row["phase"], row["slice_type"], str(row["slice_value"]))
    actual_by_key, expected_by_key = {key(row): row for row in recomputed}, {key(row): row for row in official}
    maxima = {name: 0.0 for name in NUMERIC_HISTOGRAM_FIELDS}
    case_mismatches = []
    for metric_key in sorted(set(actual_by_key) & set(expected_by_key)):
        actual, expected = actual_by_key[metric_key], expected_by_key[metric_key]
        if int(actual["cases"]) != int(expected["cases"]):
            case_mismatches.append(metric_key)
        for name in NUMERIC_HISTOGRAM_FIELDS:
            maxima[name] = max(maxima[name], abs(float(actual[name]) - float(expected[name])))
    maximum = max(maxima.values(), default=float("inf"))
    return {
        "recomputed_rows": len(recomputed), "official_rows": len(official),
        "missing": sorted(set(expected_by_key) - set(actual_by_key)), "extra": sorted(set(actual_by_key) - set(expected_by_key)),
        "case_mismatches": case_mismatches, "max_absolute_difference_by_metric": maxima,
        "max_absolute_difference": maximum, "tolerance": 1e-12,
        "ok": len(recomputed) == 84 and len(official) == 84 and set(actual_by_key) == set(expected_by_key) and not case_mismatches and maximum <= 1e-12,
    }


def placement_decisions(combined: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    groups = defaultdict(list)
    for row in combined:
        key = tuple(row[name] for name in ("profile_id", "model", "parallelism", "parallel_size", "policy"))
        groups[key].append(row)
    decisions, rankings = [], []
    for values in groups.values():
        if {row["topology_level"] for row in values} != {"L1", "L2", "L3"} or len(values) != 3:
            raise RuntimeError("placement候选必须严格为L1/L2/L3")
        predicted = sorted(values, key=lambda row: float(row["predicted_cost_us_per_1000"]))
        teacher = sorted(values, key=lambda row: float(row["teacher_cost_us_per_1000"]))
        predicted_rank = {row["topology_level"]: index + 1 for index, row in enumerate(predicted)}
        teacher_rank = {row["topology_level"]: index + 1 for index, row in enumerate(teacher)}
        selected, oracle = predicted[0], teacher[0]
        regret = (float(selected["teacher_cost_us_per_1000"]) - float(oracle["teacher_cost_us_per_1000"])) / max(float(oracle["teacher_cost_us_per_1000"]), 1e-12)
        spearman = 1.0 - 6.0 * sum((predicted_rank[level] - teacher_rank[level]) ** 2 for level in ("L1", "L2", "L3")) / 24.0
        margin = (float(predicted[1]["predicted_cost_us_per_1000"]) - float(selected["predicted_cost_us_per_1000"])) / max(float(selected["predicted_cost_us_per_1000"]), 1e-12)
        robust_predicted = float(selected["predicted_cost_upper_us_per_1000"]) <= min(float(row["predicted_cost_lower_us_per_1000"]) for row in values if row is not selected)
        robust_teacher = float(oracle["teacher_cost_upper_us_per_1000"]) <= min(float(row["teacher_cost_lower_us_per_1000"]) for row in values if row is not oracle)
        base = {name: selected[name] for name in ("profile_id", "source", "segment", "model", "parallelism", "parallel_size", "policy")}
        decisions.append({
            **base,
            "predicted_selected_topology": selected["topology_level"], "teacher_optimal_topology": oracle["topology_level"],
            "top1_match": selected["topology_level"] == oracle["topology_level"],
            "teacher_optimal_in_predicted_top2": oracle["topology_level"] in {row["topology_level"] for row in predicted[:2]},
            "teacher_regret": regret, "spearman_rank_correlation": spearman,
            "predicted_decision_margin": margin,
            "predicted_selection_stable_under_replica_envelope": robust_predicted,
            "teacher_optimum_stable_under_replica_envelope": robust_teacher,
            "ranking_scope": "communication_only_fixed_parallel_configuration",
        })
        for row in values:
            rankings.append({
                **base, "topology_level": row["topology_level"], "placement_id": row["placement_id"],
                "predicted_rank": predicted_rank[row["topology_level"]], "teacher_rank": teacher_rank[row["topology_level"]],
                "predicted_cost_us_per_1000": row["predicted_cost_us_per_1000"], "teacher_cost_us_per_1000": row["teacher_cost_us_per_1000"],
                "predicted_selected": row is selected, "teacher_optimal": row is oracle,
            })
    metric_groups = defaultdict(list)
    for row in decisions:
        slices = (
            ("overall", "all"), ("parallelism", row["parallelism"]), ("model", row["model"]),
            ("policy", row["policy"]), ("parallel_size", str(row["parallel_size"])),
            ("parallelism_size", f"{row['parallelism']}{row['parallel_size']}"),
        )
        for slice_type, slice_value in slices:
            metric_groups[(slice_type, slice_value)].append(row)
    metrics = []
    for (slice_type, slice_value), values in sorted(metric_groups.items()):
        regrets = sorted(float(row["teacher_regret"]) for row in values)
        p95_index = max(0, math.ceil(0.95 * len(regrets)) - 1)
        metrics.append({
            "slice_type": slice_type, "slice_value": slice_value, "cases": len(values),
            "top1_agreement": statistics.fmean(float(bool(row["top1_match"])) for row in values),
            "top2_coverage": statistics.fmean(float(bool(row["teacher_optimal_in_predicted_top2"])) for row in values),
            "mean_teacher_regret": statistics.fmean(regrets), "p95_teacher_regret": regrets[p95_index],
            "mean_spearman_rank_correlation": statistics.fmean(float(row["spearman_rank_correlation"]) for row in values),
            "mean_predicted_decision_margin": statistics.fmean(float(row["predicted_decision_margin"]) for row in values),
            "predicted_selection_envelope_stability_rate": statistics.fmean(float(bool(row["predicted_selection_stable_under_replica_envelope"])) for row in values),
            "teacher_optimum_envelope_stability_rate": statistics.fmean(float(bool(row["teacher_optimum_stable_under_replica_envelope"])) for row in values),
        })
    return rankings, decisions, metrics


def compare_phase35(physical_metrics: list[dict], phase35_rows: list[dict]) -> list[dict]:
    mapping = {
        ("tp", "L1"): "tp_l1_single_node_b200_nvlink_measured",
        ("tp", "L2"): "tp_l2_same_rack_nominal_proxy",
        ("tp", "L3"): "tp_l3_cross_rack_nominal_proxy",
        ("pp", "L1"): "pp_l1_single_node_nominal_proxy",
        ("pp", "L2"): "pp_l2_same_rack_nominal_proxy",
        ("pp", "L3"): "pp_l3_cross_rack_nominal_proxy",
    }
    proxy = {(row["parallelism"], row["placement_id"], row["phase"], row["slice_type"], str(row["slice_value"])): row for row in phase35_rows}
    output = []
    for row in physical_metrics:
        if row["slice_type"] not in {"overall", "model", "policy"}:
            continue
        placement = mapping[(row["parallelism"], row["topology_level"])]
        baseline = proxy[(row["parallelism"], placement, row["phase"], row["slice_type"], str(row["slice_value"]))]
        output.append({
            "parallelism": row["parallelism"], "topology_level": row["topology_level"], "phase": row["phase"],
            "slice_type": row["slice_type"], "slice_value": row["slice_value"], "cases": row["cases"],
            "physical_cost_mape": row["cost_mape"], "phase35_cost_mape": baseline["cost_mape"],
            "physical_minus_phase35_cost_mape": float(row["cost_mape"]) - float(baseline["cost_mape"]),
            "physical_cost_wape": row["cost_wape"], "phase35_cost_wape": baseline["cost_wape"],
            "physical_minus_phase35_cost_wape": float(row["cost_wape"]) - float(baseline["cost_wape"]),
            "physical_signed_bias": row["signed_bias"], "phase35_signed_bias": baseline["signed_bias"],
            "phase35_placement_id": placement,
            "physical_evidence": "phase39_physical_measurement",
            "phase35_evidence": "physical_only_for_tp_l1_otherwise_parameterized_proxy",
        })
    return output


def make_figure(path: Path, decision_metrics: list[dict]) -> None:
    rows = [row for row in decision_metrics if row["slice_type"] in {"overall", "parallelism"}]
    width, height, margin = 900, 480, 80
    maximum = max([0.01, *(float(row["mean_teacher_regret"]) for row in rows)]) * 1.2
    bar_width = (width - 2 * margin) / len(rows)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="80" y="32" font-family="sans-serif" font-size="20">Phase39 communication-only placement mean teacher regret</text>',
    ]
    for index, row in enumerate(rows):
        value = float(row["mean_teacher_regret"])
        x = margin + index * bar_width
        bar_height = value / maximum * 310
        y = 390 - bar_height
        label = f"{row['slice_type']}:{row['slice_value']}"
        svg.extend([
            f'<rect x="{x + 16:.1f}" y="{y:.1f}" width="{bar_width - 32:.1f}" height="{bar_height:.1f}" fill="#2563eb"/>',
            f'<text x="{x + bar_width/2:.1f}" y="{y - 7:.1f}" text-anchor="middle" font-family="sans-serif" font-size="12">{value:.2%}</text>',
            f'<text x="{x + bar_width/2:.1f}" y="420" text-anchor="middle" font-family="sans-serif" font-size="11">{label}</text>',
        ])
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")
