#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OPS = ("all_reduce", "all_gather")
GROUP_SIZES = (2, 4, 8)


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repo_root
        / "experiment-results"
        / "phase2"
        / "b200_l1_collective_curve",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "experiment-results" / "phase2" / "summary_l1_curve",
    )
    parser.add_argument(
        "--expected-repeats",
        type=int,
        default=5,
        help="Number of independent process-level repeats expected per curve.",
    )
    parser.add_argument(
        "--plot-name",
        default="b200_l1_collective_curve.png",
        help="Output filename for the four-panel curve figure.",
    )
    return parser.parse_args()


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def coefficient_of_variation(values):
    mean = statistics.mean(values)
    return statistics.pstdev(values) / mean if mean else 0.0


def read_records(input_dir):
    rows = []
    for path in sorted(input_dir.glob("tp*/*/r*/curve.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            record["_source"] = str(path)
            rows.append(record)
    return rows


def validate_records(records, expected_repeats):
    expected_sizes = {1 << exponent for exponent in range(10, 28)}
    expected_sizes.add(48 * 1024)
    grouped = defaultdict(list)
    topologies = {record["topology"] for record in records}
    timing_modes = {record.get("timing_mode", "steady_state") for record in records}
    if len(topologies) != 1:
        raise ValueError(f"expected one topology, found {sorted(topologies)}")
    if len(timing_modes) != 1:
        raise ValueError(f"expected one timing mode, found {sorted(timing_modes)}")
    for record in records:
        key = (
            record["op"],
            int(record["group_size"]),
            int(record["repeat_id"]),
        )
        grouped[key].append(record)
        expected_scope = (
            "representative-rank-logical-input"
            if record["op"] == "all_reduce"
            else "logical-gathered-output"
        )
        if record["payload_scope"] != expected_scope:
            raise ValueError(f"payload scope mismatch in {record['_source']}")
        if len(record["samples_us"]) != 100:
            raise ValueError(f"sample count mismatch in {record['_source']}")
        if len(record["rank_samples_us"]) != int(record["group_size"]):
            raise ValueError(f"rank sample count mismatch in {record['_source']}")
        if any(len(samples) != 100 for samples in record["rank_samples_us"]):
            raise ValueError(f"per-rank sample count mismatch in {record['_source']}")

    expected_keys = {
        (op, group_size, repeat)
        for op in OPS
        for group_size in GROUP_SIZES
        for repeat in range(expected_repeats)
    }
    if set(grouped) != expected_keys:
        missing = sorted(expected_keys - set(grouped))
        extra = sorted(set(grouped) - expected_keys)
        raise ValueError(f"incomplete suite: missing={missing}, extra={extra}")
    for key, values in grouped.items():
        sizes = {int(record["payload_bytes"]) for record in values}
        if sizes != expected_sizes:
            raise ValueError(f"payload sweep mismatch for {key}")


def flatten_records(records):
    rows = []
    for record in records:
        latency = record["latency_us"]
        rank_samples = np.asarray(record["rank_samples_us"], dtype=np.float64)
        intrinsic_samples = np.min(rank_samples, axis=0)
        rows.append(
            {
                "op": record["op"],
                "topology": record["topology"],
                "transport_label": record.get("transport_label", "unspecified"),
                "timing_mode": record.get("timing_mode", "steady_state"),
                "node_count": int(record.get("node_count", 1)),
                "group_size": int(record["group_size"]),
                "repeat_id": int(record["repeat_id"]),
                "payload_scope": record["payload_scope"],
                "payload_bytes": int(record["payload_bytes"]),
                "input_payload_bytes_per_rank": int(
                    record["input_payload_bytes_per_rank"]
                ),
                "ring_equivalent_factor": float(record["ring_equivalent_factor"]),
                "median_latency_us": float(latency["median"]),
                "p95_latency_us": float(latency["p95"]),
                "p99_latency_us": float(latency["p99"]),
                "max_latency_us": float(latency["max"]),
                "intrinsic_min_median_latency_us": float(
                    np.median(intrinsic_samples)
                ),
                "intrinsic_min_p95_latency_us": percentile(
                    intrinsic_samples, 95
                ),
                "algorithmic_bandwidth_GBps": float(
                    record["algorithmic_bandwidth_GBps"]
                ),
                "ring_equivalent_bus_bandwidth_GBps": float(
                    record["ring_equivalent_bus_bandwidth_GBps"]
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["op"],
            row["group_size"],
            row["payload_bytes"],
            row["repeat_id"],
        ),
    )


def aggregate_records(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[
            (
                record["op"],
                int(record["group_size"]),
                int(record["payload_bytes"]),
            )
        ].append(record)

    rows = []
    for (op, group_size, payload_bytes), group in sorted(grouped.items()):
        samples = [float(sample) for record in group for sample in record["samples_us"]]
        intrinsic_samples = [
            float(sample)
            for record in group
            for sample in np.min(
                np.asarray(record["rank_samples_us"], dtype=np.float64),
                axis=0,
            )
        ]
        repeat_medians = [float(record["latency_us"]["median"]) for record in group]
        intrinsic_repeat_medians = [
            float(
                np.median(
                    np.min(
                        np.asarray(record["rank_samples_us"], dtype=np.float64),
                        axis=0,
                    )
                )
            )
            for record in group
        ]
        median_us = float(statistics.median(samples))
        intrinsic_median_us = float(statistics.median(intrinsic_samples))
        seconds = median_us / 1_000_000.0
        intrinsic_seconds = intrinsic_median_us / 1_000_000.0
        algorithmic_bandwidth = payload_bytes / seconds / 1e9
        intrinsic_algorithmic_bandwidth = (
            payload_bytes / intrinsic_seconds / 1e9
        )
        ring_factor = (
            2 * (group_size - 1) / group_size
            if op == "all_reduce"
            else (group_size - 1) / group_size
        )
        rows.append(
            {
                "topology": group[0]["topology"],
                "transport_label": group[0].get(
                    "transport_label", "unspecified"
                ),
                "timing_mode": group[0].get("timing_mode", "steady_state"),
                "node_count": int(group[0].get("node_count", 1)),
                "backend": "nccl",
                "op": op,
                "group_size": group_size,
                "payload_scope": group[0]["payload_scope"],
                "payload_bytes": payload_bytes,
                "input_payload_bytes_per_rank": int(
                    group[0]["input_payload_bytes_per_rank"]
                ),
                "repeats": len(group),
                "pooled_samples": len(samples),
                "median_latency_us": median_us,
                "mean_latency_us": float(statistics.mean(samples)),
                "p95_latency_us": percentile(samples, 95),
                "p99_latency_us": percentile(samples, 99),
                "min_latency_us": float(min(samples)),
                "max_latency_us": float(max(samples)),
                "latency_cv": coefficient_of_variation(samples),
                "repeat_median_cv": coefficient_of_variation(repeat_medians),
                "intrinsic_min_median_latency_us": intrinsic_median_us,
                "intrinsic_min_mean_latency_us": float(
                    statistics.mean(intrinsic_samples)
                ),
                "intrinsic_min_p95_latency_us": percentile(
                    intrinsic_samples, 95
                ),
                "intrinsic_min_p99_latency_us": percentile(
                    intrinsic_samples, 99
                ),
                "intrinsic_min_latency_cv": coefficient_of_variation(
                    intrinsic_samples
                ),
                "intrinsic_min_repeat_median_cv": coefficient_of_variation(
                    intrinsic_repeat_medians
                ),
                "ring_equivalent_factor": ring_factor,
                "ring_equivalent_bytes": payload_bytes * ring_factor,
                "algorithmic_bandwidth_GBps": algorithmic_bandwidth,
                "ring_equivalent_bus_bandwidth_GBps": (
                    algorithmic_bandwidth * ring_factor
                ),
                "intrinsic_min_algorithmic_bandwidth_GBps": (
                    intrinsic_algorithmic_bandwidth
                ),
                "intrinsic_min_ring_equivalent_bus_bandwidth_GBps": (
                    intrinsic_algorithmic_bandwidth * ring_factor
                ),
            }
        )
    return rows


def first_payload_at_fraction(rows, fraction):
    maximum = max(row["ring_equivalent_bus_bandwidth_GBps"] for row in rows)
    target = maximum * fraction
    envelope = 0.0
    for row in sorted(rows, key=lambda item: item["payload_bytes"]):
        envelope = max(envelope, row["ring_equivalent_bus_bandwidth_GBps"])
        if envelope >= target:
            return int(row["payload_bytes"])
    return int(rows[-1]["payload_bytes"])


def build_cost_model(summary):
    curves = []
    for op in OPS:
        for group_size in GROUP_SIZES:
            rows = [
                row
                for row in summary
                if row["op"] == op and row["group_size"] == group_size
            ]
            rows.sort(key=lambda row: row["payload_bytes"])
            curves.append(
                {
                    "op": op,
                    "group_size": group_size,
                    "topology": rows[0]["topology"],
                    "transport_label": rows[0]["transport_label"],
                    "timing_mode": rows[0]["timing_mode"],
                    "node_count": rows[0]["node_count"],
                    "backend": "nccl",
                    "payload_scope": rows[0]["payload_scope"],
                    "ring_equivalent_factor": rows[0]["ring_equivalent_factor"],
                    "throughput_regime_knots": {
                        "25pct_max_bus_bw_payload_bytes": (
                            first_payload_at_fraction(rows, 0.25)
                        ),
                        "75pct_max_bus_bw_payload_bytes": (
                            first_payload_at_fraction(rows, 0.75)
                        ),
                        "90pct_max_bus_bw_payload_bytes": (
                            first_payload_at_fraction(rows, 0.90)
                        ),
                        "note": (
                            "Heuristic knots from the monotonic bandwidth "
                            "envelope; use continuous interpolation for prediction."
                        ),
                    },
                    "knots": [
                        {
                            "payload_bytes": row["payload_bytes"],
                            "median_latency_us": row["median_latency_us"],
                            "p95_latency_us": row["p95_latency_us"],
                            "intrinsic_min_median_latency_us": row[
                                "intrinsic_min_median_latency_us"
                            ],
                            "intrinsic_min_p95_latency_us": row[
                                "intrinsic_min_p95_latency_us"
                            ],
                            "ring_equivalent_bus_bandwidth_GBps": row[
                                "ring_equivalent_bus_bandwidth_GBps"
                            ],
                        }
                        for row in rows
                    ],
                }
            )
    return {
        "schema_version": "continuous-collective-cost-v1",
        "interpolation": "linear-in-log2-payload",
        "topology": summary[0]["topology"],
        "transport_label": summary[0]["transport_label"],
        "timing_mode": summary[0]["timing_mode"],
        "node_count": summary[0]["node_count"],
        "latency_scopes": {
            "median_latency_us": (
                "max-rank-local-envelope-after-rendezvous"
                if summary[0]["timing_mode"] == "rendezvous"
                else "max-completion-across-ranks"
            ),
            "intrinsic_min_median_latency_us": (
                "skew-free-minimum-duration-across-ranks"
            ),
        },
        "curves": curves,
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def format_payload_axis(axis):
    ticks = [1 << exponent for exponent in range(10, 28, 2)]
    labels = []
    for value in ticks:
        if value < 1024**2:
            labels.append(f"{value // 1024}K")
        else:
            labels.append(f"{value // 1024**2}M")
    axis.set_xticks(ticks, labels)


def plot_curves(path, summary):
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    colors = {2: "#4C78A8", 4: "#F58518", 8: "#E45756"}
    panels = {
        "all_reduce": (axes[0, 0], axes[0, 1], "AllReduce"),
        "all_gather": (axes[1, 0], axes[1, 1], "AllGather"),
    }
    for op, (latency_axis, bandwidth_axis, title) in panels.items():
        for group_size in GROUP_SIZES:
            rows = [
                row
                for row in summary
                if row["op"] == op and row["group_size"] == group_size
            ]
            rows.sort(key=lambda row: row["payload_bytes"])
            payloads = [row["payload_bytes"] for row in rows]
            medians = [row["median_latency_us"] for row in rows]
            p95s = [row["p95_latency_us"] for row in rows]
            bandwidth = [row["ring_equivalent_bus_bandwidth_GBps"] for row in rows]
            latency_axis.plot(
                payloads,
                medians,
                marker="o",
                markersize=3,
                color=colors[group_size],
                label=f"TP={group_size}",
            )
            latency_axis.fill_between(
                payloads,
                medians,
                p95s,
                color=colors[group_size],
                alpha=0.12,
            )
            bandwidth_axis.plot(
                payloads,
                bandwidth,
                marker="o",
                markersize=3,
                color=colors[group_size],
                label=f"TP={group_size}",
            )
        latency_axis.set(
            xscale="log",
            yscale="log",
            xlabel="Logical payload bytes",
            ylabel="Latency (μs)",
            title=f"{title}: median and p95 envelope",
        )
        bandwidth_axis.set(
            xscale="log",
            xlabel="Logical payload bytes",
            ylabel="Ring-equivalent bus bandwidth (GB/s)",
            title=f"{title}: effective bandwidth",
        )
        format_payload_axis(latency_axis)
        format_payload_axis(bandwidth_axis)
        latency_axis.legend()
        bandwidth_axis.legend()

    for axis in axes.flat:
        axis.grid(True, which="both", alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = read_records(args.input_dir)
    validate_records(records, args.expected_repeats)
    repeat_rows = flatten_records(records)
    summary = aggregate_records(records)
    cost_model = build_cost_model(summary)

    write_csv(args.output_dir / "collective_curve_repeat_records.csv", repeat_rows)
    write_csv(args.output_dir / "collective_curve_summary.csv", summary)
    with (args.output_dir / "collective_cost_knots.json").open("w") as output:
        json.dump(cost_model, output, indent=2)
        output.write("\n")
    plot_curves(args.output_dir / args.plot_name, summary)
    print(
        f"Wrote {len(summary)} aggregate points from {len(records)} records "
        f"and {sum(len(record['samples_us']) for record in records)} samples"
    )


if __name__ == "__main__":
    main()
