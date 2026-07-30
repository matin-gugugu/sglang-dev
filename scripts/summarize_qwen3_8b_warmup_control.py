#!/usr/bin/env python3
"""Compare historical cold-shape and same-shape-warmed all-rank Prefill labels."""

import argparse
import csv
import glob
import json
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
        / "phase5"
        / "qwen3_8b_all_rank",
    )
    parser.add_argument(
        "--warmed-dir",
        type=Path,
        default=repo_root
        / "experiment-results"
        / "phase6"
        / "qwen3_8b_warmup_control",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "experiment-results"
        / "phase6"
        / "qwen3_8b_warmup_control_summary",
    )
    return parser.parse_args()


def read_records(root):
    records = []
    pattern = root / "tp*" / "r*" / "**" / "all_rank_ground_truth.jsonl"
    for path_string in glob.glob(str(pattern), recursive=True):
        for line in Path(path_string).read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def workload_key(record):
    workload = record["workload"]
    return (
        record["phase"],
        record["all_rank_ground_truth"]["rank_count"],
        workload["batch_size"],
        workload["input_len"],
        workload["output_len"],
    )


def demand_signature(record):
    demand = record["full_phase_pattern_demand"]
    return json.dumps(
        {
            "calls": demand["all_reduce_calls"],
            "payload": demand["input_payload_bytes"],
            "histogram": demand["calls_by_input_payload_bytes"],
            "ring": demand["ring_equivalent"],
        },
        sort_keys=True,
    )


def metric_values(records, field):
    return np.asarray(
        [
            record["all_rank_ground_truth"]["full_phase_estimate"][field]
            for record in records
        ],
        dtype=np.float64,
    )


def iqr_fraction(values):
    median = float(np.median(values))
    if median == 0:
        return 0.0
    return float(
        (np.percentile(values, 75) - np.percentile(values, 25)) / median
    )


def tail_fraction(values):
    median = float(np.median(values))
    if median == 0:
        return 0.0
    return float(np.percentile(values, 95) / median)


def write_csv(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    baseline = defaultdict(list)
    warmed = defaultdict(list)
    for record in read_records(args.baseline_dir):
        if record["phase"] == "prefill":
            baseline[workload_key(record)].append(record)
    for record in read_records(args.warmed_dir):
        warmed[workload_key(record)].append(record)

    expected_keys = {
        ("prefill", tp, 1, input_len, 8)
        for tp in (2, 4, 8)
        for input_len in (128, 2048, 8192)
    }
    expected_keys.add(("prefill", 2, 8, 128, 8))
    if set(warmed) != expected_keys:
        raise ValueError(
            f"unexpected warmed workload set: missing={expected_keys - set(warmed)}, "
            f"extra={set(warmed) - expected_keys}"
        )

    rows = []
    for key in sorted(expected_keys):
        baseline_records = baseline[key]
        warmed_records = warmed[key]
        if len(baseline_records) != 3:
            raise ValueError(f"{key}: expected 3 baseline repeats")
        if len(warmed_records) != 10:
            raise ValueError(f"{key}: expected 10 warmed repeats")
        signatures = {
            demand_signature(record)
            for record in baseline_records + warmed_records
        }
        if len(signatures) != 1:
            raise ValueError(f"{key}: PatternDemand changed across repeats")

        _, tp, batch_size, input_len, output_len = key
        baseline_rank0 = metric_values(
            baseline_records, "rank0_kernel_time_us"
        )
        warmed_rank0 = metric_values(warmed_records, "rank0_kernel_time_us")
        baseline_intrinsic = metric_values(
            baseline_records, "skew_free_intrinsic_kernel_time_us"
        )
        warmed_intrinsic = metric_values(
            warmed_records, "skew_free_intrinsic_kernel_time_us"
        )
        baseline_sync_inclusive = metric_values(
            baseline_records,
            "synchronization_inclusive_max_duration_sum_us",
        )
        warmed_sync_inclusive = metric_values(
            warmed_records,
            "synchronization_inclusive_max_duration_sum_us",
        )
        demand = warmed_records[0]["full_phase_pattern_demand"]
        rows.append(
            {
                "workload_id": f"prefill-tp{tp}-b{batch_size}-l{input_len}-m{output_len}",
                "group_size": tp,
                "batch_size": batch_size,
                "input_len": input_len,
                "output_len": output_len,
                "calls": demand["all_reduce_calls"],
                "logical_payload_bytes": demand["input_payload_bytes"],
                "message_payload_bytes": int(
                    next(iter(demand["calls_by_input_payload_bytes"]))
                ),
                "baseline_repeat_count": len(baseline_records),
                "warmed_repeat_count": len(warmed_records),
                "baseline_rank0_median_us": float(np.median(baseline_rank0)),
                "warmed_rank0_median_us": float(np.median(warmed_rank0)),
                "baseline_rank0_iqr_fraction": iqr_fraction(baseline_rank0),
                "warmed_rank0_iqr_fraction": iqr_fraction(warmed_rank0),
                "baseline_intrinsic_median_us": float(
                    np.median(baseline_intrinsic)
                ),
                "warmed_intrinsic_median_us": float(
                    np.median(warmed_intrinsic)
                ),
                "warmed_over_baseline_intrinsic_median": float(
                    np.median(warmed_intrinsic) / np.median(baseline_intrinsic)
                ),
                "baseline_intrinsic_iqr_fraction": iqr_fraction(
                    baseline_intrinsic
                ),
                "warmed_intrinsic_iqr_fraction": iqr_fraction(
                    warmed_intrinsic
                ),
                "baseline_intrinsic_p95_over_median": tail_fraction(
                    baseline_intrinsic
                ),
                "warmed_intrinsic_p95_over_median": tail_fraction(
                    warmed_intrinsic
                ),
                "baseline_sync_inclusive_iqr_fraction": iqr_fraction(
                    baseline_sync_inclusive
                ),
                "warmed_sync_inclusive_iqr_fraction": iqr_fraction(
                    warmed_sync_inclusive
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "warmup_control.csv", rows)

    colors = {2: "#4C78A8", 4: "#F58518", 8: "#E45756"}
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for row in rows:
        axes[0].scatter(
            100 * row["baseline_intrinsic_iqr_fraction"],
            100 * row["warmed_intrinsic_iqr_fraction"],
            color=colors[row["group_size"]],
            s=65,
        )
    max_iqr = 100 * max(
        max(row["baseline_intrinsic_iqr_fraction"] for row in rows),
        max(row["warmed_intrinsic_iqr_fraction"] for row in rows),
    )
    axes[0].plot([0, max_iqr], [0, max_iqr], "--", color="black", linewidth=1)
    axes[0].set_xlabel("Historical control IQR / median (%)")
    axes[0].set_ylabel("Same-shape warmup IQR / median (%)")
    axes[0].set_title("Skew-free intrinsic label stability")
    axes[0].grid(True, alpha=0.25)

    labels = [
        f"TP{row['group_size']}\nB{row['batch_size']} L{row['input_len']}"
        for row in rows
    ]
    positions = np.arange(len(rows))
    width = 0.38
    axes[1].bar(
        positions - width / 2,
        [100 * row["baseline_intrinsic_iqr_fraction"] for row in rows],
        width,
        color="#9ECAE1",
        label="Historical control (3 reps)",
    )
    axes[1].bar(
        positions + width / 2,
        [100 * row["warmed_intrinsic_iqr_fraction"] for row in rows],
        width,
        color=[colors[row["group_size"]] for row in rows],
        label="Same-shape warmup (10 reps)",
    )
    axes[1].set_xticks(positions, labels, rotation=55, ha="right", fontsize=8)
    axes[1].set_ylabel("IQR / median (%)")
    axes[1].set_title("Per-workload repeat dispersion")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)

    axes[2].bar(
        positions,
        [row["warmed_over_baseline_intrinsic_median"] for row in rows],
        color=[colors[row["group_size"]] for row in rows],
    )
    axes[2].axhline(1, color="black", linewidth=1, linestyle="--")
    axes[2].set_xticks(positions, labels, rotation=55, ha="right", fontsize=8)
    axes[2].set_ylabel("Warmed median / historical median")
    axes[2].set_title("Warmup shift in measured target")
    axes[2].grid(True, axis="y", alpha=0.25)

    figure.suptitle(
        "Qwen3-8B Prefill: same-shape warmup control for all-rank labels",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(
        args.output_dir / "qwen3_8b_warmup_control.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    baseline_iqr = [
        row["baseline_intrinsic_iqr_fraction"] for row in rows
    ]
    warmed_iqr = [row["warmed_intrinsic_iqr_fraction"] for row in rows]
    target_ratios = [
        row["warmed_over_baseline_intrinsic_median"] for row in rows
    ]
    summary = {
        "schema_version": "qwen3-8b-prefill-warmup-control-v1",
        "workload_count": len(rows),
        "baseline_repeat_count": 3,
        "same_shape_warmup_repeat_count": 10,
        "pattern_demand_unchanged_workloads": len(rows),
        "skew_free_intrinsic_iqr_fraction": {
            "baseline_median": float(np.median(baseline_iqr)),
            "same_shape_warmup_median": float(np.median(warmed_iqr)),
            "improved_workloads": sum(
                after < before
                for before, after in zip(baseline_iqr, warmed_iqr)
            ),
        },
        "same_shape_warmup_intrinsic_target_shift": {
            "median_ratio": float(np.median(target_ratios)),
            "min_ratio": min(target_ratios),
            "max_ratio": max(target_ratios),
        },
        "comparison_note": (
            "The historical control has three repeats and the warmed arm has ten; "
            "use this as a protocol gate, not as a randomized causal estimate."
        ),
        "synchronization_inclusive_diagnostic_iqr_fraction": {
            "baseline_median": float(
                np.median(
                    [
                        row["baseline_sync_inclusive_iqr_fraction"]
                        for row in rows
                    ]
                )
            ),
            "same_shape_warmup_median": float(
                np.median(
                    [
                        row["warmed_sync_inclusive_iqr_fraction"]
                        for row in rows
                    ]
                )
            ),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        f"summarized {len(rows)} workloads; median intrinsic IQR/median "
        f"{np.median(baseline_iqr):.4f} -> {np.median(warmed_iqr):.4f}"
    )


if __name__ == "__main__":
    main()
