#!/usr/bin/env python3
"""一条CPU命令确定性运行Phase38物理曲线cost重算。"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import io
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATTERN_WORKFLOWS = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(PATTERN_WORKFLOWS))

from common import environment_record, load_json, sha256, utc_now, write_json
from finalize import finalize
from preflight import run_checks


NUMERIC_HISTOGRAM_FIELDS = (
    "calls_mape",
    "calls_wape",
    "bytes_mape",
    "bytes_wape",
    "mean_histogram_l1",
    "mean_histogram_tv",
    "mean_normalized_log_payload_emd",
    "common_reference_cost_mape",
    "common_reference_cost_wape",
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
    if path.suffix == ".gz":
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        deterministic_gzip(path, buffer.getvalue())
    else:
        with path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def vector(row: dict[str, str], name: str) -> list[float]:
    values = [float(value) for value in json.loads(row[name])]
    if len(values) != 12 or not all(math.isfinite(value) and value >= 0 for value in values):
        raise RuntimeError(f"非法12-bin向量：example={row.get('example_id')}, field={name}")
    return values


def vector_sum(vectors: list[list[float]]) -> list[float]:
    return [sum(values) for values in zip(*vectors)]


def histogram_sha(row: dict[str, str], prefix: str) -> str:
    payload = row[f"{prefix}_calls_by_12bin_json"] + "|" + row[f"{prefix}_logical_bytes_by_12bin_json"]
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def interpolate_log2(knots: list[dict], payload_bytes: float) -> tuple[float, str]:
    if not knots:
        raise RuntimeError("物理曲线没有knots")
    payloads = [float(knot["payload_bytes"]) for knot in knots]
    latencies = [float(knot["median_latency_us"]) for knot in knots]
    if payloads != sorted(payloads) or len(set(payloads)) != len(payloads):
        raise RuntimeError("物理曲线payload必须严格递增")
    payload = max(float(payload_bytes), 1.0)
    if payload <= payloads[0]:
        return latencies[0], "low" if payload < payloads[0] else "inside"
    if payload >= payloads[-1]:
        return latencies[-1], "high" if payload > payloads[-1] else "inside"
    right = bisect.bisect_right(payloads, payload)
    left = right - 1
    x0 = math.log2(payloads[left])
    x1 = math.log2(payloads[right])
    fraction = (math.log2(payload) - x0) / (x1 - x0)
    return latencies[left] + fraction * (latencies[right] - latencies[left]), "inside"


def histogram_cost(
    calls: list[float],
    logical_bytes: list[float],
    curve: dict,
    audit: dict[str, float],
) -> float:
    total = 0.0
    for count, byte_count in zip(calls, logical_bytes):
        if count <= 1e-12:
            continue
        payload = byte_count / count
        latency, position = interpolate_log2(curve["knots"], payload)
        total += count * latency
        audit["nonempty_bins"] += 1
        audit["logical_calls"] += count
        if position != "inside":
            audit[f"{position}_clamped_bins"] += 1
            audit[f"{position}_clamped_calls"] += count
    return total


def cost_phase_rows(
    predictions: list[dict[str, str]],
    targets: list[dict[str, str]],
    curves: list[dict],
) -> tuple[list[dict], dict]:
    target_by_id = {row["example_id"]: row for row in targets}
    prediction_ids = {row["example_id"] for row in predictions}
    if len(target_by_id) != len(targets) or len(prediction_ids) != len(predictions) or set(target_by_id) != prediction_ids:
        raise RuntimeError("Phase38 prediction/target example_id集合不完全一致")
    interpolation = {
        curve["curve_id"]: {
            role: defaultdict(float)
            for role in ("prediction", "teacher")
        }
        for curve in curves
    }
    output = []
    for prediction in sorted(predictions, key=lambda row: row["example_id"]):
        target = target_by_id[prediction["example_id"]]
        predicted_calls = vector(prediction, "predicted_calls_by_12bin_json")
        predicted_bytes = vector(prediction, "predicted_logical_bytes_by_12bin_json")
        teacher_calls = vector(target, "target_calls_by_12bin_json")
        teacher_bytes = vector(target, "target_logical_bytes_by_12bin_json")
        for curve in curves:
            predicted_cost = histogram_cost(
                predicted_calls,
                predicted_bytes,
                curve,
                interpolation[curve["curve_id"]]["prediction"],
            )
            teacher_cost = histogram_cost(
                teacher_calls,
                teacher_bytes,
                curve,
                interpolation[curve["curve_id"]]["teacher"],
            )
            output.append({
                "example_id": prediction["example_id"],
                "profile_id": prediction["profile_id"],
                "source": prediction["source"],
                "segment": prediction["segment"],
                "model": prediction["model"],
                "parallelism": "pp",
                "parallel_size": prediction["parallel_size"],
                "policy": prediction["policy"],
                "phase": prediction["phase"],
                "curve_id": curve["curve_id"],
                "topology_scope": curve["topology_scope"],
                "topology_category": curve["topology_category"],
                "raw_link": curve["raw_link"],
                "physical_gpu_pair_json": json.dumps(curve["physical_gpu_pair"], separators=(",", ":")),
                "gpu_models_json": json.dumps(curve["gpu_models"], separators=(",", ":")),
                "curve_evidence": "physical_measurement",
                "backend": curve["backend"],
                "measurement_scope": curve["measurement_scope"],
                "direction_policy": curve["direction_policy"],
                "interpolation": curve["interpolation"],
                "predicted_cost_us_per_1000": predicted_cost,
                "teacher_cost_us_per_1000": teacher_cost,
                "absolute_error_us_per_1000": abs(predicted_cost - teacher_cost),
                "absolute_percentage_error": abs(predicted_cost - teacher_cost) / max(teacher_cost, 1e-12),
                "signed_error_us_per_1000": predicted_cost - teacher_cost,
                "predicted_histogram_sha256": histogram_sha(prediction, "predicted"),
                "teacher_histogram_sha256": histogram_sha(target, "target"),
                "teacher_kind": target["teacher_kind"],
                "evidence_set": "phase34_open_target_repeated_engineering_with_phase37_physical_curve",
            })
    compact_audit = {
        curve_id: {role: dict(values) for role, values in roles.items()}
        for curve_id, roles in interpolation.items()
    }
    return output, compact_audit


def combined_cost_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row[name] for name in (
            "profile_id", "model", "parallel_size", "policy", "curve_id"
        ))
        grouped[key].append(row)
    output = []
    for values in grouped.values():
        if len(values) != 2 or {row["phase"] for row in values} != {"prefill", "decode"}:
            raise RuntimeError("PP cost配置必须恰好包含prefill和decode")
        values.sort(key=lambda row: row["phase"])
        source = values[0]
        predicted = sum(float(row["predicted_cost_us_per_1000"]) for row in values)
        teacher = sum(float(row["teacher_cost_us_per_1000"]) for row in values)
        output.append({
            **{name: source[name] for name in (
                "profile_id", "source", "segment", "model", "parallelism", "parallel_size", "policy",
                "curve_id", "topology_scope", "topology_category", "raw_link", "physical_gpu_pair_json",
                "gpu_models_json", "curve_evidence", "backend", "measurement_scope", "direction_policy",
                "interpolation", "teacher_kind", "evidence_set",
            )},
            "example_id": source["example_id"].rsplit("/", 1)[0] + "/total",
            "phase": "total",
            "predicted_cost_us_per_1000": predicted,
            "teacher_cost_us_per_1000": teacher,
            "absolute_error_us_per_1000": abs(predicted - teacher),
            "absolute_percentage_error": abs(predicted - teacher) / max(teacher, 1e-12),
            "signed_error_us_per_1000": predicted - teacher,
            "predicted_phase_histogram_sha256_json": json.dumps(
                {row["phase"]: row["predicted_histogram_sha256"] for row in values},
                sort_keys=True,
                separators=(",", ":"),
            ),
            "teacher_phase_histogram_sha256_json": json.dumps(
                {row["phase"]: row["teacher_histogram_sha256"] for row in values},
                sort_keys=True,
                separators=(",", ":"),
            ),
        })
    return sorted(output, key=lambda row: (row["curve_id"], row["example_id"]))


def aggregate_cost_metrics(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        for slice_type, slice_value in (
            ("overall", "all"),
            ("model", row["model"]),
            ("policy", row["policy"]),
        ):
            groups[(row["curve_id"], row["phase"], slice_type, str(slice_value))].append(row)
    output = []
    for (curve_id, phase, slice_type, slice_value), values in sorted(groups.items()):
        teacher = sum(float(row["teacher_cost_us_per_1000"]) for row in values)
        predicted = sum(float(row["predicted_cost_us_per_1000"]) for row in values)
        output.append({
            "parallelism": "pp",
            "curve_id": curve_id,
            "topology_scope": values[0]["topology_scope"],
            "topology_category": values[0]["topology_category"],
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
    predicted_cdf = []
    actual_cdf = []
    p_running = 0.0
    a_running = 0.0
    for p_value, a_value in zip(predicted, actual):
        p_running += p_value / predicted_total
        a_running += a_value / actual_total
        predicted_cdf.append(p_running)
        actual_cdf.append(a_running)
    centers = [(math.log2(left) + math.log2(right)) / 2 for left, right in zip(edges[:-1], edges[1:])]
    area = sum(
        abs(predicted_cdf[index] - actual_cdf[index]) * (centers[index + 1] - centers[index])
        for index in range(len(centers) - 1)
    )
    return area / (math.log2(edges[-1]) - math.log2(edges[0]))


def histogram_case_record(
    prediction: dict,
    target: dict,
    phase: str,
    predicted_calls: list[float],
    predicted_bytes: list[float],
    actual_calls: list[float],
    actual_bytes: list[float],
    edges: list[float],
) -> dict:
    predicted_calls_total = sum(predicted_calls)
    actual_calls_total = sum(actual_calls)
    predicted_bytes_total = sum(predicted_bytes)
    actual_bytes_total = sum(actual_bytes)
    predicted_reference = 5.0 * predicted_calls_total + predicted_bytes_total / 100e9 * 1e6
    actual_reference = 5.0 * actual_calls_total + actual_bytes_total / 100e9 * 1e6
    tv = histogram_tv(predicted_calls, actual_calls)
    return {
        "profile_id": prediction["profile_id"],
        "source": prediction["source"],
        "model": prediction["model"],
        "parallelism": "pp",
        "parallel_size": prediction["parallel_size"],
        "policy": prediction["policy"],
        "method": "h0_plus_dnn_residual",
        "phase": phase,
        "actual_total_calls": actual_calls_total,
        "predicted_total_calls": predicted_calls_total,
        "calls_absolute_error": abs(predicted_calls_total - actual_calls_total),
        "calls_ape": abs(predicted_calls_total - actual_calls_total) / max(actual_calls_total, 1e-12),
        "actual_total_logical_bytes": actual_bytes_total,
        "predicted_total_logical_bytes": predicted_bytes_total,
        "bytes_absolute_error": abs(predicted_bytes_total - actual_bytes_total),
        "bytes_ape": abs(predicted_bytes_total - actual_bytes_total) / max(actual_bytes_total, 1e-12),
        "histogram_l1": 2 * tv,
        "histogram_tv": tv,
        "normalized_log_payload_emd": normalized_log_emd(predicted_calls, actual_calls, edges),
        "actual_common_reference_cost_us": actual_reference,
        "predicted_common_reference_cost_us": predicted_reference,
        "cost_absolute_error": abs(predicted_reference - actual_reference),
        "cost_ape": abs(predicted_reference - actual_reference) / max(actual_reference, 1e-12),
    }


def frozen_histogram_records(
    predictions: list[dict[str, str]],
    targets: list[dict[str, str]],
    edges: list[float],
) -> list[dict]:
    target_by_id = {row["example_id"]: row for row in targets}
    records = []
    grouped = defaultdict(list)
    for prediction in predictions:
        target = target_by_id[prediction["example_id"]]
        values = (
            vector(prediction, "predicted_calls_by_12bin_json"),
            vector(prediction, "predicted_logical_bytes_by_12bin_json"),
            vector(target, "target_calls_by_12bin_json"),
            vector(target, "target_logical_bytes_by_12bin_json"),
        )
        record = histogram_case_record(
            prediction,
            target,
            prediction["phase"],
            values[0],
            values[1],
            values[2],
            values[3],
            edges,
        )
        records.append(record)
        key = tuple(prediction[name] for name in ("profile_id", "model", "parallel_size", "policy"))
        grouped[key].append((prediction, target, *values))
    for values in grouped.values():
        if len(values) != 2 or {value[0]["phase"] for value in values} != {"prefill", "decode"}:
            raise RuntimeError("冻结直方图配置缺少两个phase")
        values.sort(key=lambda value: value[0]["phase"])
        prediction, target = values[0][0], values[0][1]
        predicted_calls = vector_sum([value[2] for value in values])
        predicted_bytes = vector_sum([value[3] for value in values])
        actual_calls = vector_sum([value[4] for value in values])
        actual_bytes = vector_sum([value[5] for value in values])
        total = histogram_case_record(
            prediction,
            target,
            "total",
            predicted_calls,
            predicted_bytes,
            actual_calls,
            actual_bytes,
            edges,
        )
        predicted_phase_calls = [item for value in values for item in value[2]]
        actual_phase_calls = [item for value in values for item in value[4]]
        total["histogram_tv"] = histogram_tv(predicted_phase_calls, actual_phase_calls)
        total["histogram_l1"] = 2 * total["histogram_tv"]
        records.append(total)
    return records


def aggregate_histogram_metrics(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in records:
        for slice_type, slice_value in (
            ("overall", "all"),
            ("model", row["model"]),
            ("policy", row["policy"]),
            ("parallel_size", row["parallel_size"]),
            ("source", row["source"]),
        ):
            groups[(row["phase"], slice_type, str(slice_value))].append(row)
    output = []
    for (phase, slice_type, slice_value), values in sorted(groups.items()):
        actual_calls = sum(float(row["actual_total_calls"]) for row in values)
        actual_bytes = sum(float(row["actual_total_logical_bytes"]) for row in values)
        actual_reference = sum(float(row["actual_common_reference_cost_us"]) for row in values)
        output.append({
            "parallelism": "pp",
            "method": "h0_plus_dnn_residual",
            "phase": phase,
            "slice_type": slice_type,
            "slice_value": slice_value,
            "cases": len(values),
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


def compare_official_histogram_metrics(recomputed: list[dict], official_rows: list[dict]) -> dict:
    official = [
        row
        for row in official_rows
        if row["parallelism"] == "pp"
        and row["method"] == "h0_plus_dnn_residual"
        and row["evidence_set"] == "phase34_blind_six_model"
    ]
    key = lambda row: (row["phase"], row["slice_type"], str(row["slice_value"]))
    recomputed_by_key = {key(row): row for row in recomputed}
    official_by_key = {key(row): row for row in official}
    maxima = {name: 0.0 for name in NUMERIC_HISTOGRAM_FIELDS}
    case_mismatches = []
    for metric_key in sorted(set(recomputed_by_key) & set(official_by_key)):
        actual = recomputed_by_key[metric_key]
        expected = official_by_key[metric_key]
        if int(actual["cases"]) != int(expected["cases"]):
            case_mismatches.append({"key": metric_key, "actual": actual["cases"], "expected": expected["cases"]})
        for name in NUMERIC_HISTOGRAM_FIELDS:
            maxima[name] = max(maxima[name], abs(float(actual[name]) - float(expected[name])))
    max_absolute = max(maxima.values(), default=float("inf"))
    return {
        "recomputed_rows": len(recomputed),
        "official_rows": len(official),
        "missing_from_recomputed": sorted(set(official_by_key) - set(recomputed_by_key)),
        "extra_in_recomputed": sorted(set(recomputed_by_key) - set(official_by_key)),
        "case_mismatches": case_mismatches,
        "max_absolute_difference_by_metric": maxima,
        "max_absolute_difference": max_absolute,
        "tolerance": 1e-12,
        "ok": len(recomputed) == 42
        and len(official) == 42
        and set(recomputed_by_key) == set(official_by_key)
        and not case_mismatches
        and max_absolute <= 1e-12,
    }


def compare_phase35_proxy(physical_metrics: list[dict], phase35_rows: list[dict], placement_id: str) -> list[dict]:
    proxy = {
        (row["phase"], row["slice_type"], str(row["slice_value"])): row
        for row in phase35_rows
        if row["parallelism"] == "pp" and row["placement_id"] == placement_id
    }
    if len(proxy) != 30:
        raise RuntimeError(f"Phase35 PP L1 proxy指标应为30行，实际{len(proxy)}")
    output = []
    for row in physical_metrics:
        baseline = proxy[(row["phase"], row["slice_type"], str(row["slice_value"]))]
        if int(row["cases"]) != int(baseline["cases"]):
            raise RuntimeError({"physical_proxy_case_count_mismatch": {"physical": row, "proxy": baseline}})
        output.append({
            "curve_id": row["curve_id"],
            "topology_category": row["topology_category"],
            "phase": row["phase"],
            "slice_type": row["slice_type"],
            "slice_value": row["slice_value"],
            "cases": row["cases"],
            "physical_cost_mape": row["cost_mape"],
            "phase35_proxy_cost_mape": baseline["cost_mape"],
            "physical_minus_proxy_cost_mape": float(row["cost_mape"]) - float(baseline["cost_mape"]),
            "physical_cost_wape": row["cost_wape"],
            "phase35_proxy_cost_wape": baseline["cost_wape"],
            "physical_minus_proxy_cost_wape": float(row["cost_wape"]) - float(baseline["cost_wape"]),
            "physical_signed_bias": row["signed_bias"],
            "phase35_proxy_signed_bias": baseline["signed_bias"],
            "physical_minus_proxy_signed_bias": float(row["signed_bias"]) - float(baseline["signed_bias"]),
            "physical_evidence": "phase37_physical_measurement",
            "proxy_evidence": "phase35_parameterized_p2p_sensitivity_not_physical_measurement",
        })
    return output


def make_figure(path: Path, metrics: list[dict], reference: float) -> None:
    rows = [
        row for row in metrics
        if row["phase"] == "total" and row["slice_type"] == "overall"
    ]
    width, height, margin = 1000, 500, 85
    maximum = max([reference, *(float(row["cost_wape"]) for row in rows)]) * 1.25
    bar_width = (width - 2 * margin) / max(len(rows), 1)
    reference_y = 390 - reference / max(maximum, 1e-12) * 310
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="85" y="32" font-family="sans-serif" font-size="20">Phase38 PP物理P2P曲线cost WAPE</text>',
        f'<line x1="{margin}" y1="{reference_y:.1f}" x2="{width-margin}" y2="{reference_y:.1f}" stroke="#dc2626" stroke-dasharray="6 5"/>',
        f'<text x="{width-margin-90}" y="{reference_y-6:.1f}" font-family="sans-serif" font-size="12" fill="#dc2626">5%诊断线</text>',
    ]
    for index, row in enumerate(rows):
        value = float(row["cost_wape"])
        x = margin + index * bar_width
        bar_height = value / max(maximum, 1e-12) * 310
        y = 390 - bar_height
        label = row["topology_category"]
        svg.extend([
            f'<rect x="{x+14:.1f}" y="{y:.1f}" width="{bar_width-28:.1f}" height="{bar_height:.1f}" fill="#2563eb"/>',
            f'<text x="{x+bar_width/2:.1f}" y="{y-8:.1f}" text-anchor="middle" font-family="sans-serif" font-size="13">{value:.2%}</text>',
            f'<text x="{x+bar_width/2:.1f}" y="420" text-anchor="middle" font-family="sans-serif" font-size="12">{label}</text>',
        ])
    svg.append('</svg>')
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def input_rows(contract: dict) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    by_name = {item["name"]: ROOT / item["path"] for item in contract["pinned_inputs"]}
    prediction_filter = contract["prediction_filter"]
    predictions = [
        row for row in read_csv(by_name["phase34c_frozen_pp_predictions"])
        if all(row[name] == value for name, value in prediction_filter.items())
    ]
    target_filter = contract["target_filter"]
    targets = [
        row for row in read_csv(by_name["phase34d_opened_hfull_targets"])
        if all(row[name] == value for name, value in target_filter.items())
    ]
    return predictions, targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-workflow-commit", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiment-results/phase38_pp_physical_curve_cost_recompute",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    preflight = run_checks(args.expected_workflow_commit, output)
    contract = load_json(HERE / "experiment.json")
    for name in ("audit", "analysis", "contracts", "figures", "logs"):
        (output / name).mkdir(parents=True, exist_ok=True)

    predictions, targets = input_rows(contract)
    prediction_ids = {row["example_id"] for row in predictions}
    target_ids = {row["example_id"] for row in targets}
    curves = preflight["curve_payload"]["curves"]
    phase_rows, interpolation_audit = cost_phase_rows(predictions, targets, curves)
    total_rows = combined_cost_rows(phase_rows)
    cost_metrics = aggregate_cost_metrics([*phase_rows, *total_rows])

    by_name = {item["name"]: ROOT / item["path"] for item in contract["pinned_inputs"]}
    official_histogram_rows = read_csv(by_name["phase34d_official_histogram_metrics"])
    histogram_records = frozen_histogram_records(predictions, targets, contract["phase34_pp_bin_edges_bytes"])
    histogram_metrics = aggregate_histogram_metrics(histogram_records)
    histogram_invariance = compare_official_histogram_metrics(histogram_metrics, official_histogram_rows)
    proxy_comparison = compare_phase35_proxy(
        cost_metrics,
        read_csv(by_name["phase35_cost_metrics"]),
        contract["phase35_proxy_baseline_placement_id"],
    )
    phase35_registry = load_json(by_name["phase35_curve_registry"])
    proxy_registry_rows = [
        row for row in phase35_registry["curves"]
        if row["placement_id"] == contract["phase35_proxy_baseline_placement_id"]
    ]
    proxy_registry_contract_ok = len(proxy_registry_rows) == 1 and (
        proxy_registry_rows[0]["parallelism"] == "pp"
        and proxy_registry_rows[0]["curve_kind"] == contract["phase35_proxy_required_curve_kind"]
        and proxy_registry_rows[0]["evidence"] == contract["phase35_proxy_required_evidence"]
    )

    scalar_max_relative = 0.0
    for prediction in predictions:
        for vector_name, total_name in (
            ("predicted_calls_by_12bin_json", "predicted_total_calls_per_1000"),
            ("predicted_logical_bytes_by_12bin_json", "predicted_total_logical_bytes_per_1000"),
        ):
            computed = sum(vector(prediction, vector_name))
            saved = float(prediction[total_name])
            scalar_max_relative = max(scalar_max_relative, abs(computed - saved) / max(abs(saved), 1.0))
    for target in targets:
        for vector_name, total_name in (
            ("target_calls_by_12bin_json", "target_total_calls_per_1000"),
            ("target_logical_bytes_by_12bin_json", "target_total_logical_bytes_per_1000"),
        ):
            computed = sum(vector(target, vector_name))
            saved = float(target[total_name])
            scalar_max_relative = max(scalar_max_relative, abs(computed - saved) / max(abs(saved), 1.0))

    write_csv(output / "analysis/phase_costs.csv.gz", phase_rows)
    write_csv(output / "analysis/combined_costs.csv.gz", total_rows)
    write_csv(output / "analysis/cost_metrics.csv", cost_metrics)
    write_csv(output / "analysis/frozen_histogram_metrics.csv", histogram_metrics)
    write_csv(output / "analysis/physical_vs_phase35_proxy.csv", proxy_comparison)
    write_json(output / "analysis/histogram_invariance.json", {
        **histogram_invariance,
        "prediction_source_sha256": preflight["pinned_inputs"]["phase34c_frozen_pp_predictions"]["actual_sha256"],
        "target_source_sha256": preflight["pinned_inputs"]["phase34d_opened_hfull_targets"]["actual_sha256"],
        "prediction_recomputation_performed": False,
        "checkpoint_loaded": False,
        "interpretation": "calls/bytes/TV/EMD逐slice复现Phase34D正式值；Phase38只替换cost curve",
    })
    write_json(output / "audit/interpolation_audit.json", interpolation_audit)
    write_json(output / "contracts/experiment.json", contract)
    write_json(output / "contracts/phase37_curve_snapshot.json", preflight["curve_payload"])
    snapshot_sha = sha256(output / "contracts/phase37_curve_snapshot.json")
    input_freeze = {
        "schema_version": "phase38-input-freeze-v1",
        "created_at_utc": utc_now(),
        "workflow_commit": preflight["workflow_commit"],
        "static_pinned_inputs": preflight["pinned_inputs"],
        "phase37": preflight["phase37"],
        "phase37_curve_snapshot_sha256": snapshot_sha,
        "phase37_source_and_snapshot_sha_match": snapshot_sha == preflight["phase37"]["curve_sha256"],
        "prediction_filter": contract["prediction_filter"],
        "target_filter": contract["target_filter"],
    }
    write_json(output / "audit/input_freeze.json", input_freeze)
    write_json(output / "audit/environment.json", {
        **environment_record(),
        "gpu_required": False,
        "gpu_used": False,
        "training_performed": False,
        "checkpoint_loaded": False,
    })
    write_json(output / "logs/runtime.log", {
        "event": "phase38_pp_physical_curve_cost_recompute_complete",
        "completed_at_utc": utc_now(),
        "workflow_commit": preflight["workflow_commit"],
        "phase37_result_commit": preflight["phase37"]["result_commit"],
        "curve_count": len(curves),
        "training_performed": False,
        "gpu_used": False,
    })
    reference = float(contract["diagnostic_overall_total_cost_wape_reference"])
    make_figure(output / "figures/pp_physical_cost_wape.svg", cost_metrics, reference)

    expected_phase_rows = int(contract["expected_prediction_phase_rows"]) * len(curves)
    expected_total_rows = int(contract["expected_total_cases"]) * len(curves)
    expected_cost_metrics = 30 * len(curves)
    headline = [
        {
            "curve_id": row["curve_id"],
            "topology_category": row["topology_category"],
            "cost_mape": row["cost_mape"],
            "cost_wape": row["cost_wape"],
            "signed_bias": row["signed_bias"],
            "above_5pct_diagnostic_reference": float(row["cost_wape"]) > reference,
        }
        for row in cost_metrics
        if row["phase"] == "total" and row["slice_type"] == "overall"
    ]
    checks = {
        "preflight_pass": preflight["status"] == "PASS",
        "phase37_curve_source_and_snapshot_sha_match": input_freeze["phase37_source_and_snapshot_sha_match"],
        "prediction_phase_rows_1296": len(predictions) == int(contract["expected_prediction_phase_rows"]),
        "target_phase_rows_1296": len(targets) == int(contract["expected_target_phase_rows"]),
        "prediction_ids_unique": len(prediction_ids) == len(predictions),
        "target_ids_unique": len(target_ids) == len(targets),
        "prediction_target_ids_exact": prediction_ids == target_ids,
        "profile_inventory_12": len({row["profile_id"] for row in predictions}) == int(contract["expected_profiles"]),
        "model_inventory_exact": {row["model"] for row in predictions} == set(contract["expected_models"]),
        "parallel_size_inventory_exact": {int(row["parallel_size"]) for row in predictions} == set(contract["expected_parallel_sizes"]),
        "policy_inventory_exact": {row["policy"] for row in predictions} == set(contract["expected_policies"]),
        "phase_inventory_exact": {row["phase"] for row in predictions} == set(contract["expected_phases"]),
        "all_predictions_are_frozen_pp_residual": all(
            row["parallelism"] == "pp"
            and row["prediction_set"] == "phase34_blind_new"
            and row["method"] == "h0_plus_dnn_residual"
            for row in predictions
        ),
        "histogram_metrics_exactly_reproduced": histogram_invariance["ok"],
        "histogram_metric_rows_42": len(histogram_metrics) == int(contract["expected_phase34_histogram_metric_rows"]),
        "saved_totals_match_12bin_vectors": scalar_max_relative <= 1e-12,
        "phase_cost_row_count": len(phase_rows) == expected_phase_rows,
        "total_cost_row_count": len(total_rows) == expected_total_rows,
        "cost_metric_row_count": len(cost_metrics) == expected_cost_metrics,
        "proxy_comparison_row_count": len(proxy_comparison) == expected_cost_metrics,
        "phase35_proxy_registry_preserves_nonphysical_label": proxy_registry_contract_ok,
        "only_physical_curves_used": all(row["curve_evidence"] == "physical_measurement" for row in phase_rows),
        "all_cost_metrics_finite": all(
            math.isfinite(float(row[name]))
            for row in cost_metrics
            for name in ("cost_mape", "cost_wape", "signed_bias")
        ),
        "no_training_gpu_checkpoint_or_prediction_recompute": True,
    }
    write_json(output / "audit/runtime_state.json", {
        "schema_version": "phase38-runtime-state-v1",
        "workflow_commit": preflight["workflow_commit"],
        "phase37": preflight["phase37"],
        "counts": {
            "prediction_phase_rows": len(predictions),
            "target_phase_rows": len(targets),
            "curves": len(curves),
            "phase_cost_rows": len(phase_rows),
            "total_cost_rows": len(total_rows),
            "cost_metric_rows": len(cost_metrics),
            "frozen_histogram_metric_rows": len(histogram_metrics),
            "proxy_comparison_rows": len(proxy_comparison),
        },
        "scalar_max_relative_difference_vs_saved_totals": scalar_max_relative,
        "histogram_invariance": histogram_invariance,
        "headline": headline,
        "checks": checks,
    })
    summary = finalize(output)
    print(json.dumps({
        "status": summary["status"],
        "output": str(output),
        "phase37_result_commit": preflight["phase37"]["result_commit"],
        "curve_count": len(curves),
        "headline": headline,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
