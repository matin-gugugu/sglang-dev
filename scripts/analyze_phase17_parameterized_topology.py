#!/usr/bin/env python3
"""Parameterize L2/L3 collective curves and propagate ProfileDemand through them.

This is a sensitivity analysis, not a physical L2/L3 benchmark.  Exact GPU
histograms are the demand oracle; measured B200 L1 curves and explicitly declared
L2/L3 parameter scenarios are independent cost profiles.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


MIN_PAYLOAD = 4 * 1024
MAX_PAYLOAD = 512 * 1024 * 1024
TPS = (2, 4, 8)
BIN_COUNTS = (1, 3, 12)
EVALUATIONS = ("traffic_segment_holdout", "model_holdout")
PREDICTED_METHODS = ("h0", "h0_residual")

# These are deliberately broad sensitivity anchors, not measured hardware facts.
# GB/s is decimal bytes/s and round_us is one ring communication-step latency.
PARAMETER_SCENARIOS = (
    {
        "curve_id": "l2_optimistic",
        "topology": "L2_same_rack_two_node_proxy",
        "scenario": "optimistic",
        "launch_us": 3.0,
        "round_us": 1.5,
        "bandwidth_gbps": 200.0,
        "saturation_mib": 0.25,
    },
    {
        "curve_id": "l2_nominal",
        "topology": "L2_same_rack_two_node_proxy",
        "scenario": "nominal",
        "launch_us": 5.0,
        "round_us": 4.0,
        "bandwidth_gbps": 100.0,
        "saturation_mib": 1.0,
    },
    {
        "curve_id": "l2_pessimistic",
        "topology": "L2_same_rack_two_node_proxy",
        "scenario": "pessimistic",
        "launch_us": 10.0,
        "round_us": 12.0,
        "bandwidth_gbps": 50.0,
        "saturation_mib": 4.0,
    },
    {
        "curve_id": "l3_optimistic",
        "topology": "L3_cross_rack_two_node_proxy",
        "scenario": "optimistic",
        "launch_us": 5.0,
        "round_us": 5.0,
        "bandwidth_gbps": 100.0,
        "saturation_mib": 1.0,
    },
    {
        "curve_id": "l3_nominal",
        "topology": "L3_cross_rack_two_node_proxy",
        "scenario": "nominal",
        "launch_us": 10.0,
        "round_us": 15.0,
        "bandwidth_gbps": 50.0,
        "saturation_mib": 4.0,
    },
    {
        "curve_id": "l3_pessimistic",
        "topology": "L3_cross_rack_two_node_proxy",
        "scenario": "pessimistic",
        "launch_us": 20.0,
        "round_us": 40.0,
        "bandwidth_gbps": 25.0,
        "saturation_mib": 8.0,
    },
    {
        "curve_id": "l2_protocol_transition_stress",
        "topology": "L2_same_rack_two_node_proxy",
        "scenario": "protocol_transition_stress",
        "launch_us": 3.0,
        "round_us": 2.0,
        "bandwidth_gbps": 25.0,
        "saturation_mib": 0.25,
        "alternate_regime": {
            "launch_us": 20.0,
            "round_us": 5.0,
            "bandwidth_gbps": 200.0,
            "saturation_mib": 4.0,
        },
    },
    {
        "curve_id": "l3_protocol_transition_stress",
        "topology": "L3_cross_rack_two_node_proxy",
        "scenario": "protocol_transition_stress",
        "launch_us": 10.0,
        "round_us": 8.0,
        "bandwidth_gbps": 12.5,
        "saturation_mib": 0.5,
        "alternate_regime": {
            "launch_us": 50.0,
            "round_us": 15.0,
            "bandwidth_gbps": 100.0,
            "saturation_mib": 8.0,
        },
    },
)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=Path,
        default=root / "experiment-results/phase16_profiledemand_dataset/phase_labels.csv",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=root / "experiment-results/phase16_profiledemand_predictor/predictions.csv.gz",
    )
    parser.add_argument(
        "--l1-curve-root",
        type=Path,
        default=root / "experiment-results/phase14f_post_rendezvous/curve",
    )
    parser.add_argument(
        "--l1-curve-extension",
        type=Path,
        default=root / "experiment-results/phase15_l1_curve_extension/curve_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase17_parameterized_topology",
    )
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"empty rows: {path}")
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_gzip_csv(path, rows):
    if not rows:
        raise ValueError(f"empty rows: {path}")
    with gzip.open(path, "wt", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def identity(row):
    return (
        row["model"],
        int(row["tp"]),
        row["profile_id"],
        row["strategy"],
        row["phase"],
    )


def ring_alpha(tp):
    return 2.0 * (tp - 1) / tp


def ring_beta(tp):
    return 2 * (tp - 1)


def make_edges(count):
    return np.power(
        2.0,
        np.linspace(math.log2(MIN_PAYLOAD), math.log2(MAX_PAYLOAD), count + 1),
    )


def bin_index(payload, edges):
    payload = float(np.clip(payload, MIN_PAYLOAD, MAX_PAYLOAD))
    return min(int(np.searchsorted(edges, payload, side="right") - 1), len(edges) - 2)


def histogram_to_moments(histogram, count):
    edges = make_edges(count)
    calls = np.zeros(count, dtype=np.float64)
    logical_bytes = np.zeros(count, dtype=np.float64)
    for payload, amount in histogram.items():
        index = bin_index(payload, edges)
        calls[index] += amount
        logical_bytes[index] += amount * payload
    return calls, logical_bytes


def exact_histogram(row):
    return {
        int(payload): float(calls)
        for payload, calls in json.loads(row["canonical_exact_histogram_per_1000_json"]).items()
    }


def load_l1_curves(root, extension):
    samples = defaultdict(list)
    for path in sorted(root.glob("tp*/all_reduce/r*/curve.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                samples[(int(row["group_size"]), int(row["payload_bytes"]))].extend(
                    float(value) for value in row["post_rendezvous_samples_us"]
                )
    points = defaultdict(dict)
    for (tp, payload), values in samples.items():
        points[tp][payload] = float(np.median(values))
    for row in read_csv(extension):
        points[int(row["tp"])][int(row["payload_bytes"])] = float(
            row["median_post_rendezvous_us"]
        )
    result = {tp: sorted(values.items()) for tp, values in points.items()}
    for tp in TPS:
        if result[tp][0][0] > MIN_PAYLOAD or result[tp][-1][0] < MAX_PAYLOAD:
            raise ValueError(f"incomplete L1 curve TP={tp}")
    return result


def measured_interpolate(points, payload):
    payload = float(np.clip(payload, points[0][0], points[-1][0]))
    xs = np.log2(np.asarray([point[0] for point in points], dtype=np.float64))
    ys = np.asarray([point[1] for point in points], dtype=np.float64)
    return max(float(np.interp(math.log2(payload), xs, ys)), 1e-9)


def single_regime_cost_us(payload, tp, scenario):
    payload = float(np.clip(payload, MIN_PAYLOAD, MAX_PAYLOAD))
    saturation_bytes = scenario["saturation_mib"] * 1024 * 1024
    utilization = max(1.0 - math.exp(-payload / saturation_bytes), 1e-6)
    effective_bandwidth = scenario["bandwidth_gbps"] * 1e9 * utilization
    data_us = ring_alpha(tp) * payload / effective_bandwidth * 1e6
    return scenario["launch_us"] + ring_beta(tp) * scenario["round_us"] + data_us


def parameterized_cost_us(payload, tp, scenario):
    primary = single_regime_cost_us(payload, tp, scenario)
    if "alternate_regime" not in scenario:
        return primary
    # The faster of a low-latency and a high-bandwidth regime creates a
    # continuous curve with an algorithm crossover. This is only a stress model
    # for nonlinearity, not a claim about a particular NCCL threshold.
    alternate = single_regime_cost_us(payload, tp, scenario["alternate_regime"])
    return min(primary, alternate)


def curve_profiles(l1_points):
    profiles = [
        {
            "curve_id": "l1_measured",
            "topology": "L1_single_node_B200_NVLink_measured",
            "scenario": "measured",
            "curve_kind": "physical_measurement",
        }
    ]
    profiles.extend({**row, "curve_kind": "parameterized_sensitivity"} for row in PARAMETER_SCENARIOS)
    return profiles


def curve_cost(curve, tp, payload, l1_points):
    if curve["curve_id"] == "l1_measured":
        return measured_interpolate(l1_points[tp], payload)
    return parameterized_cost_us(payload, tp, curve)


def asymptotic_bandwidth(curve, tp, l1_points):
    if curve["curve_id"] != "l1_measured":
        candidates = [curve["bandwidth_gbps"]]
        if "alternate_regime" in curve:
            candidates.append(curve["alternate_regime"]["bandwidth_gbps"])
        return max(candidates) * 1e9
    payload = float(l1_points[tp][-1][0])
    latency_s = float(l1_points[tp][-1][1]) * 1e-6
    return ring_alpha(tp) * payload / latency_s


def cost_from_moments(calls, logical_bytes, curve, tp, l1_points):
    total = 0.0
    for amount, byte_count in zip(calls, logical_bytes):
        if amount > 1e-12:
            total += amount * curve_cost(curve, tp, byte_count / amount, l1_points)
    return total


def exact_cost(histogram, curve, tp, l1_points):
    return sum(
        amount * curve_cost(curve, tp, payload, l1_points)
        for payload, amount in histogram.items()
    )


def data_only_cost(total_bytes, curve, tp, l1_points):
    bandwidth = asymptotic_bandwidth(curve, tp, l1_points)
    return ring_alpha(tp) * total_bytes / bandwidth * 1e6


def prediction_lookup(rows):
    result = {}
    required = {
        "predicted_calls_by_12bin_json",
        "predicted_bytes_by_12bin_json",
    }
    for row in rows:
        if row["evaluation"] not in EVALUATIONS or row["method"] not in PREDICTED_METHODS:
            continue
        if not required.issubset(row):
            raise ValueError("Phase 16 predictions do not contain replayable 12-bin vectors")
        key = (row["evaluation"], row["method"], *identity(row))
        if key in result:
            raise ValueError(f"duplicate prediction {key}")
        result[key] = (
            np.asarray(json.loads(row["predicted_calls_by_12bin_json"]), dtype=np.float64),
            np.asarray(json.loads(row["predicted_bytes_by_12bin_json"]), dtype=np.float64),
        )
    expected = len(EVALUATIONS) * len(PREDICTED_METHODS) * 1296
    if len(result) != expected:
        raise ValueError(f"expected {expected} selected prediction vectors, got {len(result)}")
    return result


def build_curve_support(curves, l1_points):
    rows = []
    payloads = np.rint(
        np.power(2.0, np.linspace(math.log2(MIN_PAYLOAD), math.log2(MAX_PAYLOAD), 65))
    ).astype(int)
    for curve in curves:
        for tp in TPS:
            for payload in payloads:
                rows.append(
                    {
                        "curve_id": curve["curve_id"],
                        "topology": curve["topology"],
                        "scenario": curve["scenario"],
                        "curve_kind": curve["curve_kind"],
                        "tp": tp,
                        "payload_bytes": payload,
                        "latency_us": curve_cost(curve, tp, payload, l1_points),
                        "launch_us_parameter": curve.get("launch_us", "measured"),
                        "round_us_parameter": curve.get("round_us", "measured"),
                        "bandwidth_gbps_parameter": curve.get("bandwidth_gbps", "measured"),
                        "saturation_mib_parameter": curve.get("saturation_mib", "measured"),
                        "alternate_regime_json": json.dumps(
                            curve.get("alternate_regime", {}), separators=(",", ":")
                        ),
                        "rank_mapping": "single_node_all_ranks" if curve["curve_id"] == "l1_measured" else "two_nodes_even_split_network_proxy",
                    }
                )
    return rows


def build_cost_rows(labels, predictions, curves, l1_points):
    rows = []
    for label in labels:
        tp = int(label["tp"])
        histogram = exact_histogram(label)
        exact_1_calls, exact_1_bytes = histogram_to_moments(histogram, 1)
        exact_3_calls, exact_3_bytes = histogram_to_moments(histogram, 3)
        exact_12_calls = np.asarray(json.loads(label["calls_by_12bin_json"]), dtype=np.float64)
        exact_12_bytes = np.asarray(json.loads(label["logical_bytes_by_12bin_json"]), dtype=np.float64)
        total_calls = float(sum(histogram.values()))
        total_bytes = float(sum(payload * amount for payload, amount in histogram.items()))
        for evaluation in EVALUATIONS:
            predicted = {
                method: predictions[(evaluation, method, *identity(label))]
                for method in PREDICTED_METHODS
            }
            for curve in curves:
                oracle = exact_cost(histogram, curve, tp, l1_points)
                representations = {
                    "total_bytes_data_only": data_only_cost(total_bytes, curve, tp, l1_points),
                    "onebin_calls_bytes": cost_from_moments(exact_1_calls, exact_1_bytes, curve, tp, l1_points),
                    "threebin_calls_bytes": cost_from_moments(exact_3_calls, exact_3_bytes, curve, tp, l1_points),
                    "twelvebin_exact": cost_from_moments(exact_12_calls, exact_12_bytes, curve, tp, l1_points),
                    "h0_predicted_12bin": cost_from_moments(*predicted["h0"], curve, tp, l1_points),
                    "residual_predicted_12bin": cost_from_moments(*predicted["h0_residual"], curve, tp, l1_points),
                    "exact_payload_oracle": oracle,
                }
                for representation, predicted_cost in representations.items():
                    rows.append(
                        {
                            "evaluation": evaluation,
                            "model": label["model"],
                            "tp": tp,
                            "profile_id": label["profile_id"],
                            "strategy": label["strategy"],
                            "phase": label["phase"],
                            "curve_id": curve["curve_id"],
                            "topology": curve["topology"],
                            "scenario": curve["scenario"],
                            "curve_kind": curve["curve_kind"],
                            "representation": representation,
                            "actual_total_calls_per_1000": total_calls,
                            "actual_total_bytes_per_1000": total_bytes,
                            "oracle_cost_us_per_1000": oracle,
                            "estimated_cost_us_per_1000": predicted_cost,
                            "absolute_percentage_error": abs(predicted_cost - oracle) / max(oracle, 1e-12),
                        }
                    )
    return rows


def add_combined_rows(cost_rows):
    grouped = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    metadata = {}
    for row in cost_rows:
        key = (
            row["evaluation"],
            row["model"],
            int(row["tp"]),
            row["profile_id"],
            row["strategy"],
            row["curve_id"],
            row["representation"],
        )
        values = grouped[key]
        values[0] += float(row["actual_total_calls_per_1000"])
        values[1] += float(row["actual_total_bytes_per_1000"])
        values[2] += float(row["oracle_cost_us_per_1000"])
        values[3] += float(row["estimated_cost_us_per_1000"])
        metadata[key] = row
    combined = []
    for key, values in grouped.items():
        source = metadata[key]
        combined.append(
            {
                **{name: source[name] for name in ("evaluation", "model", "tp", "profile_id", "strategy", "curve_id", "topology", "scenario", "curve_kind", "representation")},
                "phase": "combined",
                "actual_total_calls_per_1000": values[0],
                "actual_total_bytes_per_1000": values[1],
                "oracle_cost_us_per_1000": values[2],
                "estimated_cost_us_per_1000": values[3],
                "absolute_percentage_error": abs(values[3] - values[2]) / max(values[2], 1e-12),
            }
        )
    return combined


def metrics(cost_rows):
    grouped = defaultdict(list)
    for row in cost_rows:
        grouped[(row["evaluation"], row["curve_id"], row["phase"], row["representation"])].append(row)
    result = []
    for (evaluation, curve_id, phase, representation), rows in sorted(grouped.items()):
        errors = np.asarray([float(row["absolute_percentage_error"]) for row in rows])
        actual = np.asarray([float(row["oracle_cost_us_per_1000"]) for row in rows])
        predicted = np.asarray([float(row["estimated_cost_us_per_1000"]) for row in rows])
        source = rows[0]
        result.append(
            {
                "evaluation": evaluation,
                "curve_id": curve_id,
                "topology": source["topology"],
                "scenario": source["scenario"],
                "phase": phase,
                "representation": representation,
                "samples": len(rows),
                "mape": float(np.mean(errors)),
                "median_ape": float(np.median(errors)),
                "p95_ape": float(np.percentile(errors, 95)),
                "max_ape": float(np.max(errors)),
                "signed_bias": float(np.sum(predicted - actual) / np.sum(actual)),
            }
        )
    return result


def strategy_contrasts(combined):
    selected = [
        row
        for row in combined
        if row["evaluation"] == "traffic_segment_holdout"
        and row["representation"] in {"exact_payload_oracle", "residual_predicted_12bin"}
    ]
    grouped = defaultdict(dict)
    for row in selected:
        key = (
            row["model"],
            int(row["tp"]),
            row["profile_id"],
            row["curve_id"],
            row["representation"],
        )
        grouped[key][row["strategy"]] = row
    result = []
    for key, strategies in sorted(grouped.items()):
        if set(strategies) != {"latency", "balanced", "throughput"}:
            raise ValueError(f"incomplete strategy group {key}")
        latency, throughput = strategies["latency"], strategies["throughput"]
        result.append(
            {
                "model": key[0],
                "tp": key[1],
                "profile_id": key[2],
                "curve_id": key[3],
                "representation": key[4],
                "latency_total_calls": latency["actual_total_calls_per_1000"],
                "throughput_total_calls": throughput["actual_total_calls_per_1000"],
                "latency_total_bytes": latency["actual_total_bytes_per_1000"],
                "throughput_total_bytes": throughput["actual_total_bytes_per_1000"],
                "latency_estimated_cost_us": latency["estimated_cost_us_per_1000"],
                "throughput_estimated_cost_us": throughput["estimated_cost_us_per_1000"],
                "calls_ratio_latency_over_throughput": float(latency["actual_total_calls_per_1000"]) / float(throughput["actual_total_calls_per_1000"]),
                "bytes_ratio_latency_over_throughput": float(latency["actual_total_bytes_per_1000"]) / float(throughput["actual_total_bytes_per_1000"]),
                "cost_ratio_latency_over_throughput": float(latency["estimated_cost_us_per_1000"]) / float(throughput["estimated_cost_us_per_1000"]),
            }
        )
    return result


def contrast_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["curve_id"], row["representation"])].append(row)
    result = []
    for (curve_id, representation), values in sorted(grouped.items()):
        result.append(
            {
                "curve_id": curve_id,
                "representation": representation,
                "samples": len(values),
                "median_calls_ratio_latency_over_throughput": float(np.median([float(row["calls_ratio_latency_over_throughput"]) for row in values])),
                "median_bytes_ratio_latency_over_throughput": float(np.median([float(row["bytes_ratio_latency_over_throughput"]) for row in values])),
                "median_cost_ratio_latency_over_throughput": float(np.median([float(row["cost_ratio_latency_over_throughput"]) for row in values])),
                "p95_cost_ratio_latency_over_throughput": float(np.percentile([float(row["cost_ratio_latency_over_throughput"]) for row in values], 95)),
            }
        )
    return result


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = read_csv(args.labels)
    if len(labels) != 1296:
        raise ValueError(f"expected 1296 labels, got {len(labels)}")
    predictions = prediction_lookup(read_csv(args.predictions))
    l1_points = load_l1_curves(args.l1_curve_root, args.l1_curve_extension)
    curves = curve_profiles(l1_points)
    support = build_curve_support(curves, l1_points)
    phase_costs = build_cost_rows(labels, predictions, curves, l1_points)
    combined = add_combined_rows(phase_costs)
    all_costs = phase_costs + combined
    metric_rows = metrics(all_costs)
    contrast_rows = strategy_contrasts(combined)
    contrast_summary_rows = contrast_summary(contrast_rows)

    write_csv(
        args.output_dir / "curve_scenarios.csv",
        [
            {
                "curve_id": curve["curve_id"],
                "topology": curve["topology"],
                "scenario": curve["scenario"],
                "curve_kind": curve["curve_kind"],
                "launch_us": curve.get("launch_us", "measured_curve"),
                "round_us": curve.get("round_us", "measured_curve"),
                "bandwidth_gbps": curve.get("bandwidth_gbps", "measured_curve"),
                "saturation_mib": curve.get("saturation_mib", "measured_curve"),
                "alternate_regime_json": json.dumps(
                    curve.get("alternate_regime", {}), separators=(",", ":")
                ),
                "rank_mapping": "single_node_all_ranks" if curve["curve_id"] == "l1_measured" else "two_nodes_even_split_network_proxy",
                "evidence_boundary": "measured" if curve["curve_id"] == "l1_measured" else "hypothetical_parameter_sensitivity_not_physical_truth",
            }
            for curve in curves
        ],
    )
    write_csv(args.output_dir / "curve_support.csv", support)
    write_gzip_csv(args.output_dir / "cost_predictions.csv.gz", all_costs)
    write_csv(args.output_dir / "cost_metrics.csv", metric_rows)
    write_gzip_csv(args.output_dir / "strategy_contrasts.csv.gz", contrast_rows)
    write_csv(args.output_dir / "strategy_contrast_summary.csv", contrast_summary_rows)

    metric_lookup = {
        (row["evaluation"], row["curve_id"], row["phase"], row["representation"]): row
        for row in metric_rows
    }
    headline_curves = (
        "l1_measured",
        "l2_nominal",
        "l3_nominal",
        "l2_protocol_transition_stress",
        "l3_protocol_transition_stress",
    )
    headline_representations = (
        "total_bytes_data_only",
        "onebin_calls_bytes",
        "threebin_calls_bytes",
        "twelvebin_exact",
        "h0_predicted_12bin",
        "residual_predicted_12bin",
    )
    headline = {}
    for curve_id in headline_curves:
        headline[curve_id] = {}
        for representation in headline_representations:
            row = metric_lookup[("traffic_segment_holdout", curve_id, "combined", representation)]
            headline[curve_id][representation] = {
                "mape": row["mape"],
                "p95_ape": row["p95_ape"],
                "signed_bias": row["signed_bias"],
            }
    contrast_lookup = {
        (row["curve_id"], row["representation"]): row for row in contrast_summary_rows
    }
    summary = {
        "schema_version": "phase17-parameterized-topology-v1",
        "status": "PASS",
        "physical_truth_boundary": {
            "L1": "measured B200 single-node post-rendezvous curve",
            "L2_L3": "parameterized sensitivity only; no physical timing truth and no L2/L3 MAPE claim",
        },
        "demand_labels": len(labels),
        "service_cases": len({(row["model"], row["profile_id"], row["strategy"]) for row in labels}),
        "curve_profiles": len(curves),
        "curve_support_rows": len(support),
        "phase_cost_rows": len(phase_costs),
        "combined_cost_rows": len(combined),
        "rank_mapping_proxy": {
            "L1": "all ranks on one node",
            "L2_L3": "two nodes, ranks evenly split; network-level ring proxy, not hierarchical NCCL reconstruction",
        },
        "formula": "launch_us + 2*(p-1)*round_us + [2*(p-1)/p]*payload/BW_eff(payload); transition stress uses min(low-latency regime, high-bandwidth regime)",
        "effective_bandwidth": "BW_max * (1-exp(-payload/saturation_bytes))",
        "headline_traffic_segment_holdout_combined": headline,
        "latency_vs_throughput_exact_oracle": {
            curve_id: contrast_lookup[(curve_id, "exact_payload_oracle")]
            for curve_id in headline_curves
        },
        "source_hashes": {
            "labels": sha256(args.labels),
            "predictions": sha256(args.predictions),
            "l1_extension": sha256(args.l1_curve_extension),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    table = [
        "| curve | representation | MAPE | P95 APE | bias |",
        "|---|---|---:|---:|---:|",
    ]
    for curve_id in headline_curves:
        for representation in headline_representations:
            row = headline[curve_id][representation]
            table.append(
                f"| {curve_id} | {representation} | {100 * row['mape']:.2f}% | "
                f"{100 * row['p95_ape']:.2f}% | {100 * row['signed_bias']:.2f}% |"
            )
    contrast_table = [
        "| curve | calls ratio L/T | bytes ratio L/T | cost ratio L/T |",
        "|---|---:|---:|---:|",
    ]
    for curve_id in headline_curves:
        row = contrast_lookup[(curve_id, "exact_payload_oracle")]
        contrast_table.append(
            f"| {curve_id} | {row['median_calls_ratio_latency_over_throughput']:.3f} | "
            f"{row['median_bytes_ratio_latency_over_throughput']:.3f} | "
            f"{row['median_cost_ratio_latency_over_throughput']:.3f} |"
        )
    readme = f"""# Phase 17：L2/L3 参数化连续代价与 ProfileDemand 传播

本实验严格分离需求与代价：Phase 16 的 1296 条标签/留出预测提供消息货物清单；L1
使用 B200 单节点实测曲线；L2/L3 各使用 optimistic/nominal/pessimistic 三组显式假设
参数生成连续曲线。L2/L3 没有物理时间真值，不能报告真实 MAPE，也不能把场景参数写成
实际集群规格。

参数化单次 AllReduce 采用：

`launch + 2(p-1)×round + [2(p-1)/p]×payload/BW_eff(payload)`，其中
`BW_eff=BW_max×(1-exp(-payload/saturation))`。L2/L3 rank mapping 简化为两节点均分
rank 的 network-level ring proxy，尚未重建 hierarchical NCCL。额外的
`protocol_transition_stress` 使用低时延/低带宽与高启动/高带宽两条子曲线的较小值，
只用于检验算法切换非线性何时使消息尺度分布不可省略。

## 成本表征误差：traffic-segment holdout，Prefill+Decode

{chr(10).join(table)}

`total_bytes_data_only` 是乐观的纯带宽基线，故意忽略 calls/RTT；`onebin` 保留总 calls
和总 bytes；三桶、12桶进一步保留尺度分布；`exact_payload_oracle` 为需求真值。

## latency配置相对throughput配置的中位比值

{chr(10).join(contrast_table)}

两种策略处理同一组请求。若 bytes ratio 接近 1 而 calls/cost ratio 显著大于 1，说明
高 RTT 拓扑会放大“小 batch、多次启动”的代价，这正是消息直方图对拓扑感知调度的
价值。

## 证据边界

- 这里可以报告不同参数场景下的结构成本、表征误差和决策敏感性；
- 若 nominal alpha–beta 下单桶已足够而 transition stress 下直方图才有优势，应如实报告
  “直方图价值取决于代价曲线非线性”，不能只保留有利场景；
- 不可以报告真实 L2/L3 通信时间准确率；
- placement 前还需加入显存可行性、计算时间和资源可用性，否则只最小化通信可能得到
  平凡选择；
- 未来获得两节点资源后，只需用相同 `op×payload×group_size×rank_mapping` 微基准替换
  参数曲线，不需要重跑全部模型 PatternDemand 网格。
"""
    (args.output_dir / "README.md").write_text(readme)

    checks = {
        "labels_1296": len(labels) == 1296,
        "selected_prediction_vectors_5184": len(predictions) == 5184,
        "nine_curve_profiles": len(curves) == 9,
        "curve_support_1755": len(support) == 9 * 3 * 65,
        "all_costs_finite": all(math.isfinite(float(row["estimated_cost_us_per_1000"])) for row in all_costs),
        "oracle_zero_error": all(float(row["absolute_percentage_error"]) == 0.0 for row in all_costs if row["representation"] == "exact_payload_oracle"),
        "l2_l3_marked_parameterized": all(curve["curve_kind"] == "parameterized_sensitivity" for curve in curves if curve["curve_id"] != "l1_measured"),
    }
    audit = {
        "schema_version": "phase17-parameterized-topology-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    (args.output_dir / "audit_summary.json").write_text(json.dumps(audit, indent=2) + "\n")
    if audit["status"] != "PASS":
        raise RuntimeError(audit)
    (args.output_dir / "DONE").write_text("PASS\n")
    (args.output_dir / "run.log").write_text(json.dumps({"summary": summary, "checks": checks}, indent=2) + "\n")
    files = sorted(path for path in args.output_dir.iterdir() if path.is_file() and path.name != "manifest.sha256")
    (args.output_dir / "manifest.sha256").write_text("".join(f"{sha256(path)}  {path.name}\n" for path in files))
    print(json.dumps({"summary": summary, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
