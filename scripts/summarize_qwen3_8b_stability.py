#!/usr/bin/env python3
"""Compare three-repeat and ten-repeat stability for high-IQR workloads."""

import argparse
import csv
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=repo_root
        / "experiment-results"
        / "phase4"
        / "qwen3_8b_expanded",
    )
    parser.add_argument(
        "--stability-dir",
        type=Path,
        default=repo_root
        / "experiment-results"
        / "phase5"
        / "qwen3_8b_stability",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "experiment-results"
        / "phase5"
        / "qwen3_8b_stability_summary",
    )
    parser.add_argument("--high-iqr-threshold", type=float, default=0.20)
    return parser.parse_args()


def workload_key(record):
    workload = record["workload"]
    demand = record["full_phase_pattern_demand"]
    return (
        record["phase"],
        int(demand["group_size"]),
        int(workload["batch_size"]),
        int(workload["input_len"]),
        int(workload["output_len"]),
    )


def workload_id(key):
    phase, tp, batch_size, input_len, output_len = key
    return (
        f"{phase}-tp{tp}-b{batch_size}-l{input_len}-m{output_len}"
    )


def load_records(directory):
    grouped = defaultdict(list)
    pattern = directory / "tp*" / "r*" / "**" / "comm_ground_truth.jsonl"
    for path_string in glob.glob(str(pattern), recursive=True):
        path = Path(path_string)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            record["_source"] = str(path)
            grouped[workload_key(record)].append(record)
    return grouped


def target_us(record):
    return float(
        record["gpu_ground_truth"]["full_phase_estimate"][
            "collective_kernel_time_us"
        ]
    )


def iqr_fraction(values):
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    if median == 0:
        return 0.0
    return float(
        (np.percentile(values, 75) - np.percentile(values, 25)) / median
    )


def pattern_signature(record):
    demand = record["full_phase_pattern_demand"]
    return (
        int(demand["all_reduce_calls"]),
        int(demand["input_payload_bytes"]),
        json.dumps(
            demand["calls_by_input_payload_bytes"],
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def write_csv(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    baseline = load_records(args.baseline_dir)
    added = load_records(args.stability_dir)
    if not baseline:
        raise ValueError(f"no baseline records below {args.baseline_dir}")
    if not added:
        raise ValueError(f"no stability records below {args.stability_dir}")

    high_iqr_keys = []
    for key, records in baseline.items():
        if len(records) != 3:
            raise ValueError(
                f"{workload_id(key)}: expected 3 baseline repeats, "
                f"got {len(records)}"
            )
        if iqr_fraction([target_us(record) for record in records]) > (
            args.high_iqr_threshold
        ):
            high_iqr_keys.append(key)

    rows = []
    for key in sorted(high_iqr_keys):
        before_records = baseline[key]
        new_records = added.get(key, [])
        combined_records = before_records + new_records
        before = [target_us(record) for record in before_records]
        combined = [target_us(record) for record in combined_records]
        signatures = {
            pattern_signature(record) for record in combined_records
        }
        phase, tp, batch_size, input_len, output_len = key
        rows.append(
            {
                "workload_id": workload_id(key),
                "phase": phase,
                "group_size": tp,
                "batch_size": batch_size,
                "input_len": input_len,
                "output_len": output_len,
                "baseline_repeat_count": len(before_records),
                "added_repeat_count": len(new_records),
                "combined_repeat_count": len(combined_records),
                "pattern_signature_count": len(signatures),
                "baseline_target_us_median": statistics.median(before),
                "combined_target_us_median": statistics.median(combined),
                "baseline_iqr_fraction": iqr_fraction(before),
                "combined_iqr_fraction": iqr_fraction(combined),
                "iqr_fraction_reduction": (
                    iqr_fraction(before) - iqr_fraction(combined)
                ),
                "combined_min_us": min(combined),
                "combined_max_us": max(combined),
            }
        )

    if len(rows) != 25:
        raise ValueError(
            f"expected 25 baseline high-IQR workloads, got {len(rows)}"
        )
    if any(row["combined_repeat_count"] != 10 for row in rows):
        incomplete = [
            (row["workload_id"], row["combined_repeat_count"])
            for row in rows
            if row["combined_repeat_count"] != 10
        ]
        raise ValueError(f"incomplete ten-repeat workloads: {incomplete}")
    if any(row["pattern_signature_count"] != 1 for row in rows):
        changed = [
            row["workload_id"]
            for row in rows
            if row["pattern_signature_count"] != 1
        ]
        raise ValueError(f"PatternDemand changed across repeats: {changed}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "stability_comparison.csv", rows)

    before = np.asarray(
        [row["baseline_iqr_fraction"] for row in rows], dtype=np.float64
    )
    after = np.asarray(
        [row["combined_iqr_fraction"] for row in rows], dtype=np.float64
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    axes[0].scatter(before * 100, after * 100, color="#4C78A8", s=52)
    upper = max(float(np.max(before)), float(np.max(after))) * 100
    axes[0].plot([0, upper], [0, upper], "--", color="black", linewidth=1)
    axes[0].axhline(
        args.high_iqr_threshold * 100,
        color="#E45756",
        linestyle=":",
        linewidth=1.3,
    )
    axes[0].set_xlabel("3-repeat IQR / median (%)")
    axes[0].set_ylabel("10-repeat IQR / median (%)")
    axes[0].set_title("High-IQR workloads before and after rerun")
    axes[0].grid(True, alpha=0.25)

    order = np.argsort(after)
    axes[1].plot(
        np.arange(1, len(after) + 1),
        after[order] * 100,
        marker="o",
        color="#54A24B",
        label="10 repeats",
    )
    axes[1].plot(
        np.arange(1, len(before) + 1),
        before[order] * 100,
        marker=".",
        color="#9D755D",
        alpha=0.75,
        label="Original 3 repeats",
    )
    axes[1].axhline(
        args.high_iqr_threshold * 100,
        color="#E45756",
        linestyle=":",
        linewidth=1.3,
        label="20% threshold",
    )
    axes[1].set_xlabel("Targeted workload (sorted by 10-repeat IQR)")
    axes[1].set_ylabel("IQR / median (%)")
    axes[1].set_title("Tail stability after increasing repeats")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(
        args.output_dir / "qwen3_8b_stability_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    summary = {
        "schema_version": "qwen3-8b-stability-summary-v1",
        "selection": {
            "baseline_repeat_count": 3,
            "high_iqr_threshold": args.high_iqr_threshold,
            "selected_workloads": len(rows),
        },
        "rerun": {
            "added_repeat_count": 7,
            "combined_repeat_count": 10,
            "pattern_demand_unchanged_workloads": sum(
                row["pattern_signature_count"] == 1 for row in rows
            ),
        },
        "repeat_iqr_fraction": {
            "baseline_median": float(np.median(before)),
            "combined_median": float(np.median(after)),
            "improved_workloads": int(np.sum(after < before)),
            "remaining_above_threshold": int(
                np.sum(after > args.high_iqr_threshold)
            ),
            "combined_p95": float(np.percentile(after, 95)),
            "combined_max": float(np.max(after)),
        },
        "interpretation": (
            "PatternDemand is deterministic when the pattern signature is "
            "unchanged; the IQR comparison quantifies noise in the measured "
            "GPU communication-time label."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        f"selected={len(rows)} improved={summary['repeat_iqr_fraction']['improved_workloads']} "
        f"remaining_above_20pct="
        f"{summary['repeat_iqr_fraction']['remaining_above_threshold']}"
    )


if __name__ == "__main__":
    main()
