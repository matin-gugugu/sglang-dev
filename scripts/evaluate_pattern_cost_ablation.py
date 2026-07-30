#!/usr/bin/env python3
"""Evaluate bytes-only, three-bin, and continuous PatternDemand cost models."""

import argparse
import csv
import glob
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODELS = ("total_bytes_only", "three_hard_bins", "continuous_histogram")
BIN_DEFINITIONS = (
    ("small", 0, 64 * 1024),
    ("medium", 64 * 1024, 4 * 1024 * 1024),
    ("large", 4 * 1024 * 1024, math.inf),
)


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase1-dir",
        type=Path,
        default=repo_root / "experiment-results" / "phase1",
    )
    parser.add_argument(
        "--custom-curve",
        type=Path,
        default=repo_root
        / "experiment-results"
        / "phase2"
        / "summary_l1_custom_kernel_curve"
        / "custom_kernel_curve_summary.csv",
    )
    parser.add_argument(
        "--nccl-curve",
        type=Path,
        default=repo_root
        / "experiment-results"
        / "phase2"
        / "summary_l1_curve"
        / "collective_curve_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "experiment-results" / "phase3" / "pattern_cost_ablation",
    )
    return parser.parse_args()


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def read_csv(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def decode_shape(output_lens):
    if len(set(output_lens)) == 1:
        return "uniform"
    counts = defaultdict(int)
    for length in output_lens:
        counts[length] += 1
    if max(counts.values()) >= 6:
        return "longtail"
    return "mixed"


def workload_key(record):
    workload = record["workload"]
    return (
        record["phase"],
        int(record["pattern_demand"]["group_size"]),
        int(workload["batch_size"]),
        int(workload["input_len"]),
        tuple(int(value) for value in workload["output_lens_per_request"]),
    )


def load_ground_truth(phase1_dir):
    grouped = defaultdict(list)
    for group_size in (2, 4, 8):
        pattern = (
            phase1_dir
            / f"qwen3_8b_tp{group_size}_inference_comm"
            / "representative"
            / "**"
            / "comm_ground_truth.jsonl"
        )
        for path_string in glob.glob(str(pattern), recursive=True):
            path = Path(path_string)
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                record["_source"] = str(path)
                grouped[workload_key(record)].append(record)

    rows = []
    for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        phase, group_size, batch_size, input_len, output_lens = key
        pattern = values[0]["pattern_demand"]
        calls_by_size = {
            int(payload): int(count)
            for payload, count in pattern["calls_by_input_payload_bytes"].items()
        }
        for record in values[1:]:
            candidate = {
                int(payload): int(count)
                for payload, count in record["pattern_demand"][
                    "calls_by_input_payload_bytes"
                ].items()
            }
            if candidate != calls_by_size:
                raise ValueError(f"PatternDemand changed across repeats for {key}")

        measured_totals = []
        structural_targets = []
        backend_signatures = set()
        for record in values:
            ground_truth = record["gpu_ground_truth"]
            measured_total = float(ground_truth["collective_kernel_time_us"]["total"])
            median_per_call = float(
                ground_truth["collective_kernel_time_us"]["median_per_invocation"]
            )
            calls = int(pattern["all_reduce_calls"])
            measured_totals.append(measured_total)
            structural_targets.append(calls * median_per_call)
            backend_signatures.add(
                "+".join(sorted(ground_truth["backend_kernel_counts"]))
            )
        if len(backend_signatures) != 1:
            raise ValueError(f"backend changed across repeats for {key}")

        shape = decode_shape(output_lens) if phase == "decode" else f"L{input_len}"
        measured_median = float(statistics.median(measured_totals))
        structural_median = float(statistics.median(structural_targets))
        rows.append(
            {
                "phase": phase,
                "group_size": group_size,
                "shape": shape,
                "batch_size": batch_size,
                "input_len": input_len,
                "output_lens_per_request": json.dumps(output_lens),
                "repeat_count": len(values),
                "calls": int(pattern["all_reduce_calls"]),
                "logical_payload_bytes": int(pattern["input_payload_bytes"]),
                "ring_equivalent_bytes": float(pattern["ring_equivalent"]["bytes"]),
                "ring_equivalent_rounds": int(pattern["ring_equivalent"]["rounds"]),
                "calls_by_payload_json": json.dumps(calls_by_size, sort_keys=True),
                "backend_signature": next(iter(backend_signatures)),
                "measured_total_us_median": measured_median,
                "measured_total_us_p25": percentile(measured_totals, 25),
                "measured_total_us_p75": percentile(measured_totals, 75),
                "structural_target_us_median": structural_median,
                "structural_target_us_p25": percentile(structural_targets, 25),
                "structural_target_us_p75": percentile(structural_targets, 75),
                "wait_residual_us_median": measured_median - structural_median,
                "wait_residual_fraction": (
                    (measured_median - structural_median) / measured_median
                    if measured_median
                    else 0.0
                ),
                "_calls_by_size": calls_by_size,
            }
        )
    return rows


class BackendAwareCostCurve:
    def __init__(
        self,
        custom_path,
        nccl_path,
        custom_latency_column="intrinsic_median_latency_us",
        nccl_latency_column="median_latency_us",
    ):
        custom_rows = read_csv(custom_path)
        nccl_rows = [row for row in read_csv(nccl_path) if row["op"] == "all_reduce"]
        self.custom = defaultdict(list)
        self.nccl = defaultdict(list)
        for row in custom_rows:
            if custom_latency_column not in row:
                raise ValueError(
                    f"{custom_path} has no {custom_latency_column} column"
                )
            self.custom[int(row["group_size"])].append(
                (
                    int(row["payload_bytes"]),
                    float(row[custom_latency_column]),
                    row["algorithm"],
                )
            )
        for row in nccl_rows:
            if nccl_latency_column not in row:
                raise ValueError(
                    f"{nccl_path} has no {nccl_latency_column} column"
                )
            self.nccl[int(row["group_size"])].append(
                (
                    int(row["payload_bytes"]),
                    float(row[nccl_latency_column]),
                    "NCCL",
                )
            )
        for values in (*self.custom.values(), *self.nccl.values()):
            values.sort()
        self.custom_max = {
            group_size: max(payload for payload, _, _ in values)
            for group_size, values in self.custom.items()
        }

    @staticmethod
    def interpolate(points, payload_bytes):
        payloads = np.asarray([point[0] for point in points], dtype=np.float64)
        costs = np.asarray([point[1] for point in points], dtype=np.float64)
        x = math.log2(payload_bytes)
        return float(
            np.interp(
                x,
                np.log2(payloads),
                costs,
                left=costs[0],
                right=costs[-1],
            )
        )

    def lookup(self, group_size, payload_bytes):
        if payload_bytes <= self.custom_max[group_size]:
            return self.interpolate(self.custom[group_size], payload_bytes)
        nccl_points = [
            point
            for point in self.nccl[group_size]
            if point[0] >= self.custom_max[group_size]
        ]
        return self.interpolate(nccl_points, payload_bytes)

    def production_points(self, group_size):
        custom_max = self.custom_max[group_size]
        points = [(payload, cost) for payload, cost, _ in self.custom[group_size]]
        points.extend(
            (payload, cost)
            for payload, cost, _ in self.nccl[group_size]
            if payload > custom_max
        )
        return sorted(points)


class NCCLOnlyCostCurve:
    def __init__(self, nccl_path, latency_column="median_latency_us"):
        rows = [row for row in read_csv(nccl_path) if row["op"] == "all_reduce"]
        self.nccl = defaultdict(list)
        for row in rows:
            if latency_column not in row:
                raise ValueError(f"{nccl_path} has no {latency_column} column")
            self.nccl[int(row["group_size"])].append(
                (
                    int(row["payload_bytes"]),
                    float(row[latency_column]),
                )
            )
        for values in self.nccl.values():
            values.sort()
        missing = sorted({2, 4, 8} - set(self.nccl))
        if missing:
            raise ValueError(f"{nccl_path} has no AllReduce curves for TP={missing}")

    def lookup(self, group_size, payload_bytes):
        points = [
            (payload, cost, "NCCL")
            for payload, cost in self.nccl[group_size]
        ]
        return BackendAwareCostCurve.interpolate(points, payload_bytes)

    def production_points(self, group_size):
        return list(self.nccl[group_size])


def bucket_name(payload_bytes):
    for name, lower, upper in BIN_DEFINITIONS:
        if lower < payload_bytes <= upper:
            return name
    raise ValueError(f"payload outside bins: {payload_bytes}")


def nonnegative_affine_fit(points):
    x = np.asarray([point[0] for point in points], dtype=np.float64)
    y = np.asarray([point[1] for point in points], dtype=np.float64)
    if len(points) == 1:
        return float(y[0]), 0.0
    design = np.column_stack((np.ones_like(x), x))
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    if slope < 0:
        return float(np.mean(y)), 0.0
    if intercept < 0:
        slope = float(np.dot(x, y) / np.dot(x, x))
        return 0.0, max(0.0, slope)
    return float(intercept), float(slope)


def fit_bucket_models(curve):
    models = {}
    for group_size in (2, 4, 8):
        grouped = defaultdict(list)
        for payload, cost in curve.production_points(group_size):
            grouped[bucket_name(payload)].append((payload, cost))
        for name, _, _ in BIN_DEFINITIONS:
            if not grouped[name]:
                raise ValueError(f"no cost points for TP={group_size}, bucket={name}")
            intercept, slope = nonnegative_affine_fit(grouped[name])
            models[(group_size, name)] = {
                "startup_us_per_call": intercept,
                "transfer_us_per_byte": slope,
                "effective_bandwidth_GBps": (1e-3 / slope if slope > 0 else None),
                "fit_points": len(grouped[name]),
            }
    return models


def raw_predictions(row, curve, bucket_models):
    calls_by_size = row["_calls_by_size"]
    continuous = sum(
        count * curve.lookup(row["group_size"], payload)
        for payload, count in calls_by_size.items()
    )
    three_bins = 0.0
    for payload, count in calls_by_size.items():
        model = bucket_models[(row["group_size"], bucket_name(payload))]
        three_bins += count * (
            model["startup_us_per_call"] + model["transfer_us_per_byte"] * payload
        )
    return {
        "total_bytes_only": float(row["logical_payload_bytes"]),
        "three_hard_bins": three_bins,
        "continuous_histogram": continuous,
    }


def calibration_anchors(rows):
    anchors = {}
    for group_size in (2, 4, 8):
        matches = [
            row
            for row in rows
            if row["phase"] == "decode"
            and row["group_size"] == group_size
            and row["shape"] == "uniform"
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one decode anchor for TP={group_size}")
        anchors[("decode", group_size)] = matches[0]
    prefill = [
        row
        for row in rows
        if row["phase"] == "prefill"
        and row["group_size"] == 2
        and row["input_len"] == 128
    ]
    if len(prefill) != 1:
        raise ValueError("expected one TP=2 L=128 prefill anchor")
    anchors[("prefill", 2)] = prefill[0]
    return anchors


def apply_calibration(rows, curve, bucket_models):
    for row in rows:
        row["_raw_predictions"] = raw_predictions(row, curve, bucket_models)
    anchors = calibration_anchors(rows)
    scales = {}
    for anchor_key, anchor in anchors.items():
        target = anchor["structural_target_us_median"]
        for model in MODELS:
            raw = anchor["_raw_predictions"][model]
            scales[(anchor_key, model)] = target / raw

    prediction_rows = []
    for row in rows:
        anchor_key = (row["phase"], row["group_size"])
        if anchor_key not in anchors:
            continue
        output = {key: value for key, value in row.items() if not key.startswith("_")}
        output["calibration_anchor"] = anchors[anchor_key]["shape"]
        target = row["structural_target_us_median"]
        for model in MODELS:
            predicted = row["_raw_predictions"][model] * scales[(anchor_key, model)]
            output[f"{model}_predicted_us"] = predicted
            output[f"{model}_ape"] = abs(predicted - target) / target
        prediction_rows.append(output)
    return prediction_rows, scales


def metric_row(scope, model, rows):
    errors = [float(row[f"{model}_ape"]) for row in rows]
    targets = np.asarray([float(row["structural_target_us_median"]) for row in rows])
    predictions = np.asarray([float(row[f"{model}_predicted_us"]) for row in rows])
    residual = targets - predictions
    denominator = float(np.sum((targets - np.mean(targets)) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / denominator if denominator else None
    return {
        "scope": scope,
        "model": model,
        "samples": len(rows),
        "mape": float(np.mean(errors)),
        "median_ape": float(np.median(errors)),
        "p95_ape": percentile(errors, 95),
        "mae_us": float(np.mean(np.abs(residual))),
        "r2": r2,
    }


def evaluate_metrics(prediction_rows):
    decode_holdout = [
        row
        for row in prediction_rows
        if row["phase"] == "decode" and row["shape"] != "uniform"
    ]
    prefill_holdout = [
        row
        for row in prediction_rows
        if row["phase"] == "prefill" and row["input_len"] != 128
    ]
    metrics = []
    for scope, rows in (
        ("decode_equal_payload_holdout", decode_holdout),
        ("prefill_payload_holdout", prefill_holdout),
    ):
        for model in MODELS:
            metrics.append(metric_row(scope, model, rows))
    for group_size in (2, 4, 8):
        subset = [row for row in decode_holdout if row["group_size"] == group_size]
        for model in MODELS:
            metrics.append(metric_row(f"decode_tp{group_size}", model, subset))
    return metrics


def plot_results(path, prediction_rows, metrics):
    colors = {
        "total_bytes_only": "#9D755D",
        "three_hard_bins": "#F58518",
        "continuous_histogram": "#4C78A8",
    }
    labels = {
        "total_bytes_only": "Total bytes only",
        "three_hard_bins": "Three hard bins",
        "continuous_histogram": "Continuous histogram",
    }
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))

    decode = [row for row in prediction_rows if row["phase"] == "decode"]
    shapes = ("uniform", "mixed", "longtail")
    x = np.arange(len(shapes))
    for group_size, marker in ((2, "o"), (4, "s"), (8, "^")):
        values = {
            row["shape"]: row for row in decode if row["group_size"] == group_size
        }
        target_anchor = values["uniform"]["structural_target_us_median"]
        axes[0, 0].plot(
            x,
            [
                values[shape]["structural_target_us_median"] / target_anchor
                for shape in shapes
            ],
            color="black",
            marker=marker,
            linestyle="--",
            alpha=0.55,
            label=f"Measured structural TP={group_size}",
        )
        for model in MODELS:
            axes[0, 0].plot(
                x,
                [
                    values[shape][f"{model}_predicted_us"] / target_anchor
                    for shape in shapes
                ],
                color=colors[model],
                marker=marker,
                alpha=0.8,
                label=(f"{labels[model]} TP={group_size}" if group_size == 2 else None),
            )
    axes[0, 0].set_xticks(x, shapes)
    axes[0, 0].set_ylabel("Time relative to uniform")
    axes[0, 0].set_title("Equal total payload: shape discrimination")
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=8, ncol=2)

    decode_metrics = [
        row for row in metrics if row["scope"] == "decode_equal_payload_holdout"
    ]
    axes[0, 1].bar(
        [labels[row["model"]] for row in decode_metrics],
        [100 * row["mape"] for row in decode_metrics],
        color=[colors[row["model"]] for row in decode_metrics],
    )
    axes[0, 1].set_ylabel("MAPE (%)")
    axes[0, 1].set_title("Held-out mixed/longtail prediction error")
    axes[0, 1].tick_params(axis="x", rotation=15)
    axes[0, 1].grid(True, axis="y", alpha=0.25)

    prefill = sorted(
        [row for row in prediction_rows if row["phase"] == "prefill"],
        key=lambda row: row["input_len"],
    )
    payload_mib = [
        min(
            int(payload)
            for payload in json.loads(row["calls_by_payload_json"])
        )
        / (1024**2)
        for row in prefill
    ]
    axes[1, 0].plot(
        payload_mib,
        [row["structural_target_us_median"] for row in prefill],
        color="black",
        marker="o",
        linewidth=2,
        label="Measured structural",
    )
    for model in MODELS:
        axes[1, 0].plot(
            payload_mib,
            [row[f"{model}_predicted_us"] for row in prefill],
            color=colors[model],
            marker="o",
            label=labels[model],
        )
    axes[1, 0].set_xscale("log", base=2)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("Single-call Prefill payload (MiB)")
    axes[1, 0].set_ylabel("73-call structural communication time (μs)")
    axes[1, 0].set_title("Prefill curve and backend transition")
    axes[1, 0].grid(True, which="both", alpha=0.25)
    axes[1, 0].legend(fontsize=8)

    residual_rows = sorted(
        decode, key=lambda row: (row["group_size"], shapes.index(row["shape"]))
    )
    positions = np.arange(len(residual_rows))
    structural = np.asarray(
        [row["structural_target_us_median"] for row in residual_rows]
    )
    wait = np.asarray(
        [max(0.0, row["wait_residual_us_median"]) for row in residual_rows]
    )
    axes[1, 1].bar(positions, structural, color="#4C78A8", label="Structural component")
    axes[1, 1].bar(
        positions,
        wait,
        bottom=structural,
        color="#E45756",
        alpha=0.75,
        label="Rank-skew/runtime residual",
    )
    axes[1, 1].set_xticks(
        positions,
        [f"TP{row['group_size']}\n{row['shape']}" for row in residual_rows],
    )
    axes[1, 1].set_ylabel("Median profiler kernel total (μs)")
    axes[1, 1].set_title("Why the neural residual is still needed")
    axes[1, 1].grid(True, axis="y", alpha=0.25)
    axes[1, 1].legend(fontsize=8)

    figure.suptitle("Qwen3-8B PatternDemand → backend-aware L1 cost ablation")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def serialize_scales(scales):
    return {
        f"{phase}_tp{group_size}_{model}": value
        for ((phase, group_size), model), value in sorted(scales.items())
    }


def main():
    args = parse_args()
    curve = BackendAwareCostCurve(args.custom_curve, args.nccl_curve)
    bucket_models = fit_bucket_models(curve)
    workloads = load_ground_truth(args.phase1_dir)
    prediction_rows, scales = apply_calibration(workloads, curve, bucket_models)
    metrics = evaluate_metrics(prediction_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "workload_predictions.csv", prediction_rows)
    write_csv(args.output_dir / "ablation_metrics.csv", metrics)
    plot_results(
        args.output_dir / "phase3_pattern_cost_ablation.png",
        prediction_rows,
        metrics,
    )
    summary = {
        "schema_version": "pattern-cost-ablation-v1",
        "workload_count": len(prediction_rows),
        "ground_truth_definition": (
            "median across repeats of calls × median collective kernel "
            "duration; profiler total minus this term is rank-skew/runtime residual"
        ),
        "calibration": (
            "one anchor per TP for decode (uniform) and TP2 L=128 for prefill; "
            "mixed/longtail and longer Prefill workloads are held out"
        ),
        "bins": [
            {
                "name": name,
                "lower_exclusive_bytes": lower,
                "upper_inclusive_bytes": (upper if math.isfinite(upper) else None),
            }
            for name, lower, upper in BIN_DEFINITIONS
        ],
        "bucket_models": {
            f"tp{group_size}_{name}": values
            for (group_size, name), values in sorted(bucket_models.items())
        },
        "calibration_scales": serialize_scales(scales),
        "metrics": metrics,
    }
    (args.output_dir / "phase3_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(
        f"Wrote {len(prediction_rows)} workload predictions and "
        f"{len(metrics)} metric rows to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
