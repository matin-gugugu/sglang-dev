#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

GROUP_SIZES = (2, 4, 8)
REPEATS = 5


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repo_root
        / "experiment-results"
        / "phase2"
        / "b200_l1_custom_kernel_curve",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "experiment-results"
        / "phase2"
        / "summary_l1_custom_kernel_curve",
    )
    return parser.parse_args()


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def coefficient_of_variation(values):
    mean = statistics.mean(values)
    return statistics.pstdev(values) / mean if mean else 0.0


def read_records(input_dir):
    rows = []
    for path in sorted(input_dir.glob("tp*/r*/curve.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            record["_source"] = str(path)
            rows.append(record)
    return rows


def validate_records(records):
    expected_sizes = {1 << exponent for exponent in range(13, 25)}
    expected_sizes.add(48 * 1024)
    grouped = defaultdict(list)
    for record in records:
        key = (int(record["group_size"]), int(record["repeat_id"]))
        grouped[key].append(record)
        if record["schema_version"] != "collective-kernel-cost-v1":
            raise ValueError(f"schema mismatch in {record['_source']}")
        if record["backend"] != "sglang_custom_all_reduce_v2":
            raise ValueError(f"backend mismatch in {record['_source']}")
        if record["latency_scope"] != "skew-free-intrinsic-lower-envelope-across-ranks":
            raise ValueError(f"latency scope mismatch in {record['_source']}")
        if len(record["samples_us"]) != 100:
            raise ValueError(f"sample count mismatch in {record['_source']}")

    expected_keys = {
        (group_size, repeat) for group_size in GROUP_SIZES for repeat in range(REPEATS)
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
        rows.append(
            {
                "topology": record["topology"],
                "backend": record["backend"],
                "op": record["op"],
                "algorithm": record["algorithm"],
                "group_size": int(record["group_size"]),
                "repeat_id": int(record["repeat_id"]),
                "payload_scope": record["payload_scope"],
                "payload_bytes": int(record["payload_bytes"]),
                "intrinsic_median_latency_us": float(record["latency_us"]["median"]),
                "intrinsic_p95_latency_us": float(record["latency_us"]["p95"]),
                "completion_median_latency_us": float(
                    record["completion_latency_us"]["median"]
                ),
                "completion_p95_latency_us": float(
                    record["completion_latency_us"]["p95"]
                ),
                "rank_skew_median_us": float(record["rank_skew_us"]["median"]),
                "rank_skew_p95_us": float(record["rank_skew_us"]["p95"]),
                "ring_equivalent_factor": float(record["ring_equivalent_factor"]),
                "ring_equivalent_bus_bandwidth_GBps": float(
                    record["ring_equivalent_bus_bandwidth_GBps"]
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["group_size"],
            row["payload_bytes"],
            row["repeat_id"],
        ),
    )


def aggregate_records(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(int(record["group_size"]), int(record["payload_bytes"]))].append(
            record
        )

    rows = []
    for (group_size, payload_bytes), values in sorted(grouped.items()):
        algorithms = {record["algorithm"] for record in values}
        if len(algorithms) != 1:
            raise ValueError(
                f"algorithm changed across repeats for TP={group_size}, "
                f"payload={payload_bytes}: {sorted(algorithms)}"
            )
        intrinsic_samples = [
            float(sample) for record in values for sample in record["samples_us"]
        ]
        completion_samples = [
            float(sample)
            for record in values
            for sample in record["completion_samples_us"]
        ]
        skew_samples = [
            float(sample)
            for record in values
            for sample in record["rank_skew_samples_us"]
        ]
        repeat_medians = [float(record["latency_us"]["median"]) for record in values]
        median_us = float(np.median(intrinsic_samples))
        ring_factor = float(values[0]["ring_equivalent_factor"])
        rows.append(
            {
                "topology": values[0]["topology"],
                "backend": values[0]["backend"],
                "op": values[0]["op"],
                "algorithm": next(iter(algorithms)),
                "group_size": group_size,
                "payload_scope": values[0]["payload_scope"],
                "payload_bytes": payload_bytes,
                "repeats": len(values),
                "pooled_samples": len(intrinsic_samples),
                "intrinsic_median_latency_us": median_us,
                "intrinsic_p95_latency_us": percentile(intrinsic_samples, 95),
                "intrinsic_p99_latency_us": percentile(intrinsic_samples, 99),
                "intrinsic_latency_cv": coefficient_of_variation(intrinsic_samples),
                "repeat_median_cv": coefficient_of_variation(repeat_medians),
                "completion_median_latency_us": float(np.median(completion_samples)),
                "completion_p95_latency_us": percentile(completion_samples, 95),
                "rank_skew_median_us": float(np.median(skew_samples)),
                "rank_skew_p95_us": percentile(skew_samples, 95),
                "ring_equivalent_factor": ring_factor,
                "ring_equivalent_bytes": payload_bytes * ring_factor,
                "algorithmic_bandwidth_GBps": (
                    payload_bytes / (median_us / 1_000_000.0) / 1e9
                ),
                "ring_equivalent_bus_bandwidth_GBps": (
                    payload_bytes * ring_factor / (median_us / 1_000_000.0) / 1e9
                ),
            }
        )
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_knots(path, rows):
    curves = []
    for group_size in GROUP_SIZES:
        values = [row for row in rows if row["group_size"] == group_size]
        curves.append(
            {
                "op": "all_reduce",
                "backend": "sglang_custom_all_reduce_v2",
                "group_size": group_size,
                "topology": "single-node-nvlink",
                "payload_scope": "representative-rank-logical-input",
                "latency_scope": ("skew-free-intrinsic-lower-envelope-across-ranks"),
                "knots": [
                    {
                        "payload_bytes": row["payload_bytes"],
                        "algorithm": row["algorithm"],
                        "intrinsic_median_latency_us": row[
                            "intrinsic_median_latency_us"
                        ],
                        "intrinsic_p95_latency_us": row["intrinsic_p95_latency_us"],
                        "completion_median_latency_us": row[
                            "completion_median_latency_us"
                        ],
                        "rank_skew_median_us": row["rank_skew_median_us"],
                    }
                    for row in values
                ],
            }
        )
    document = {
        "schema_version": "collective-kernel-cost-knots-v1",
        "interpolation": "linear-in-log2-payload-within-one-backend",
        "note": (
            "Use intrinsic median for the structural base term. Completion and "
            "rank-skew fields quantify launch/synchronization uncertainty."
        ),
        "curves": curves,
    }
    path.write_text(json.dumps(document, indent=2) + "\n")


def plot_summary(path, rows):
    colors = {2: "#4C78A8", 4: "#F58518", 8: "#E45756"}
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for group_size in GROUP_SIZES:
        values = [row for row in rows if row["group_size"] == group_size]
        payloads = np.asarray([row["payload_bytes"] for row in values])
        medians = np.asarray([row["intrinsic_median_latency_us"] for row in values])
        p95 = np.asarray([row["intrinsic_p95_latency_us"] for row in values])
        bandwidth = np.asarray(
            [row["ring_equivalent_bus_bandwidth_GBps"] for row in values]
        )
        skew_median = np.asarray([row["rank_skew_median_us"] for row in values])
        skew_p95 = np.asarray([row["rank_skew_p95_us"] for row in values])
        color = colors[group_size]
        label = f"TP={group_size}"

        axes[0].plot(payloads, medians, marker="o", color=color, label=label)
        axes[0].fill_between(payloads, medians, p95, color=color, alpha=0.14)
        axes[1].plot(payloads, bandwidth, marker="o", color=color, label=label)
        axes[2].plot(payloads, skew_median, marker="o", color=color, label=label)
        axes[2].fill_between(payloads, skew_median, skew_p95, color=color, alpha=0.14)

    axes[0].set_title("CustomAllReduce intrinsic kernel cost")
    axes[0].set_ylabel("GPU kernel latency (μs)")
    axes[1].set_title("Intrinsic ring-equivalent bandwidth")
    axes[1].set_ylabel("GB/s")
    axes[2].set_title("Cross-rank launch/skew envelope")
    axes[2].set_ylabel("max(kernel) - min(kernel) (μs)")

    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Representative-rank logical payload bytes")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    axes[0].set_yscale("log")
    axes[2].set_yscale("symlog", linthresh=0.1)

    figure.suptitle("B200 single-node NVLink SGLang CustomAllReduceV2 (5 repeats)")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    records = read_records(args.input_dir)
    validate_records(records)
    repeat_rows = flatten_records(records)
    summary_rows = aggregate_records(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "custom_kernel_curve_repeat_records.csv", repeat_rows)
    write_csv(args.output_dir / "custom_kernel_curve_summary.csv", summary_rows)
    write_knots(args.output_dir / "custom_kernel_cost_knots.json", summary_rows)
    plot_summary(args.output_dir / "b200_l1_custom_kernel_curve.png", summary_rows)
    print(
        f"Wrote {len(summary_rows)} aggregate points from {len(records)} records "
        f"and {sum(len(record['samples_us']) for record in records)} samples"
    )


if __name__ == "__main__":
    main()
