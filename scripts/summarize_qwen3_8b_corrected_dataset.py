#!/usr/bin/env python3
"""Audit the complete Qwen3-8B all-rank intrinsic dataset."""

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABEL_FIELDS = {
    "rank0": "rank0_kernel_time_us",
    "intrinsic": "skew_free_intrinsic_kernel_time_us",
    "post_rendezvous": "post_rendezvous_completion_kernel_time_us",
    "sync_inclusive": "synchronization_inclusive_max_duration_sum_us",
}


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repo_root
        / "experiment-results"
        / "phase6"
        / "qwen3_8b_corrected_all_rank",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "experiment-results"
        / "phase6"
        / "qwen3_8b_corrected_summary",
    )
    return parser.parse_args()


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
            "group_size": demand["group_size"],
            "calls": demand["all_reduce_calls"],
            "payload": demand["input_payload_bytes"],
            "histogram": demand["calls_by_input_payload_bytes"],
            "ring": demand["ring_equivalent"],
        },
        sort_keys=True,
    )


def iqr_fraction(values):
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    if median == 0:
        return 0.0
    return float(
        (np.percentile(values, 75) - np.percentile(values, 25)) / median
    )


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def write_csv(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    grouped = defaultdict(list)
    pattern = args.input_dir / "tp*" / "r*" / "**" / "all_rank_ground_truth.jsonl"
    paths = glob.glob(str(pattern), recursive=True)
    if not paths:
        raise ValueError(f"no all-rank records below {args.input_dir}")
    record_count = 0
    for path_string in paths:
        for line in Path(path_string).read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record["schema_version"] != "all-rank-comm-labels-v2":
                raise ValueError(f"old schema in {path_string}")
            grouped[workload_key(record)].append(record)
            record_count += 1

    if len(grouped) != 195 or record_count != 585:
        raise ValueError(
            f"expected 195 workloads and 585 records, got "
            f"{len(grouped)} and {record_count}"
        )

    rows = []
    for key, records in sorted(grouped.items()):
        if len(records) != 3:
            raise ValueError(f"{key}: expected three repeats")
        if len({demand_signature(record) for record in records}) != 1:
            raise ValueError(f"{key}: PatternDemand changed across repeats")
        if any(
            not record["alignment"]["exact_count_on_every_rank"]
            or not record["alignment"]["identical_backend_sequence"]
            or not record["alignment"][
                "identical_full_phase_pattern_demand_on_every_rank"
            ]
            for record in records
        ):
            raise ValueError(f"{key}: all-rank alignment failed")

        phase, tp, batch_size, input_len, output_len = key
        full_estimates = [
            record["all_rank_ground_truth"]["full_phase_estimate"]
            for record in records
        ]
        values = {
            label: [
                float(estimate[field]) for estimate in full_estimates
            ]
            for label, field in LABEL_FIELDS.items()
        }
        demand = records[0]["full_phase_pattern_demand"]
        row = {
            "workload_id": (
                f"{phase}-tp{tp}-b{batch_size}-l{input_len}-m{output_len}"
            ),
            "phase": phase,
            "group_size": tp,
            "batch_size": batch_size,
            "input_len": input_len,
            "output_len": output_len,
            "repeat_count": len(records),
            "calls": demand["all_reduce_calls"],
            "logical_payload_bytes": demand["input_payload_bytes"],
            "message_payload_bytes": int(
                next(iter(demand["calls_by_input_payload_bytes"]))
            ),
            "ring_equivalent_bytes": demand["ring_equivalent"]["bytes"],
            "ring_equivalent_rounds": demand["ring_equivalent"]["rounds"],
        }
        for label in LABEL_FIELDS:
            row[f"{label}_median_us"] = float(np.median(values[label]))
            row[f"{label}_iqr_fraction"] = iqr_fraction(values[label])
        row["post_rendezvous_over_intrinsic"] = (
            row["post_rendezvous_median_us"]
            / row["intrinsic_median_us"]
        )
        row["sync_inclusive_over_intrinsic"] = (
            row["sync_inclusive_median_us"]
            / row["intrinsic_median_us"]
        )
        rows.append(row)

    phase_counts = {
        phase: sum(row["phase"] == phase for row in rows)
        for phase in ("prefill", "decode")
    }
    tp_counts = {
        str(tp): sum(row["group_size"] == tp for row in rows)
        for tp in (2, 4, 8)
    }
    if phase_counts != {"prefill": 60, "decode": 135}:
        raise ValueError(f"phase counts mismatch: {phase_counts}")
    if tp_counts != {"2": 65, "4": 65, "8": 65}:
        raise ValueError(f"TP counts mismatch: {tp_counts}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "corrected_dataset_summary.csv", rows)

    colors = {2: "#4C78A8", 4: "#F58518", 8: "#E45756"}
    figure, axes = plt.subplots(2, 2, figsize=(15, 11))
    labels = ("rank0", "intrinsic", "post_rendezvous", "sync_inclusive")
    axes[0, 0].boxplot(
        [
            [100 * row[f"{label}_iqr_fraction"] for row in rows]
            for label in labels
        ],
        tick_labels=(
            "Rank 0",
            "Intrinsic",
            "Post-rendezvous",
            "Sync-inclusive",
        ),
        showfliers=False,
    )
    axes[0, 0].set_ylabel("Repeat IQR / median (%)")
    axes[0, 0].set_title("Repeat stability across 195 workloads")
    axes[0, 0].grid(True, axis="y", alpha=0.25)

    for tp in (2, 4, 8):
        selected = [row for row in rows if row["group_size"] == tp]
        axes[0, 1].scatter(
            [row["intrinsic_median_us"] for row in selected],
            [row["post_rendezvous_median_us"] for row in selected],
            color=colors[tp],
            s=28,
            alpha=0.7,
            label=f"TP={tp}",
        )
    lower = min(row["intrinsic_median_us"] for row in rows)
    upper = max(row["post_rendezvous_median_us"] for row in rows)
    axes[0, 1].plot([lower, upper], [lower, upper], "--", color="black")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("Intrinsic median (μs)")
    axes[0, 1].set_ylabel("Post-rendezvous median (μs)")
    axes[0, 1].set_title("Portable target versus timestamp diagnostic")
    axes[0, 1].grid(True, which="both", alpha=0.25)
    axes[0, 1].legend()

    axes[1, 0].boxplot(
        [
            [
                row["sync_inclusive_over_intrinsic"]
                for row in rows
                if row["group_size"] == tp
            ]
            for tp in (2, 4, 8)
        ],
        tick_labels=("TP=2", "TP=4", "TP=8"),
        showfliers=False,
    )
    axes[1, 0].set_ylabel("Sync-inclusive / intrinsic")
    axes[1, 0].set_title("Inflation from rank arrival skew")
    axes[1, 0].grid(True, axis="y", alpha=0.25)

    markers = {"prefill": "o", "decode": "s"}
    for phase in ("prefill", "decode"):
        for tp in (2, 4, 8):
            selected = [
                row
                for row in rows
                if row["phase"] == phase and row["group_size"] == tp
            ]
            axes[1, 1].scatter(
                [row["message_payload_bytes"] for row in selected],
                [
                    row["intrinsic_median_us"] / row["calls"]
                    for row in selected
                ],
                color=colors[tp],
                marker=markers[phase],
                s=32,
                alpha=0.7,
                label=f"{phase} TP={tp}",
            )
    axes[1, 1].set_xscale("log", base=2)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Logical payload per collective (bytes)")
    axes[1, 1].set_ylabel("Intrinsic cost per collective (μs)")
    axes[1, 1].set_title("Continuous message-size cost structure")
    axes[1, 1].grid(True, which="both", alpha=0.25)
    axes[1, 1].legend(fontsize=8)

    figure.suptitle("Qwen3-8B corrected all-rank dataset audit")
    figure.tight_layout()
    figure.savefig(
        args.output_dir / "qwen3_8b_corrected_dataset_audit.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    stability = {}
    for label in labels:
        fractions = [row[f"{label}_iqr_fraction"] for row in rows]
        stability[label] = {
            "median_iqr_fraction": float(np.median(fractions)),
            "p95_iqr_fraction": percentile(fractions, 95),
            "workloads_above_20pct": sum(value > 0.20 for value in fractions),
        }
    post_ratios = [
        row["post_rendezvous_over_intrinsic"] for row in rows
    ]
    sync_ratios = [
        row["sync_inclusive_over_intrinsic"] for row in rows
    ]
    equal_payload = {}
    for tp in (2, 4, 8):
        small_message = next(
            row
            for row in rows
            if row["phase"] == "decode"
            and row["group_size"] == tp
            and row["input_len"] == 2048
            and row["batch_size"] == 1
            and row["output_len"] == 512
        )
        large_message = next(
            row
            for row in rows
            if row["phase"] == "decode"
            and row["group_size"] == tp
            and row["input_len"] == 2048
            and row["batch_size"] == 16
            and row["output_len"] == 32
        )
        equal_payload[str(tp)] = {
            "logical_payload_ratio": (
                small_message["logical_payload_bytes"]
                / large_message["logical_payload_bytes"]
            ),
            "intrinsic_time_ratio_small_over_large_message": (
                small_message["intrinsic_median_us"]
                / large_message["intrinsic_median_us"]
            ),
            "post_rendezvous_time_ratio_small_over_large_message": (
                small_message["post_rendezvous_median_us"]
                / large_message["post_rendezvous_median_us"]
            ),
        }
    summary = {
        "schema_version": "qwen3-8b-corrected-dataset-audit-v1",
        "record_count": record_count,
        "unique_workloads": len(rows),
        "repeat_count": 3,
        "phase_counts": phase_counts,
        "tp_counts": tp_counts,
        "pattern_demand_unchanged_workloads": len(rows),
        "label_repeat_stability": stability,
        "post_rendezvous_over_intrinsic": {
            "median": float(np.median(post_ratios)),
            "p95": percentile(post_ratios, 95),
            "max": max(post_ratios),
        },
        "sync_inclusive_over_intrinsic": {
            "median": float(np.median(sync_ratios)),
            "p95": percentile(sync_ratios, 95),
            "max": max(sync_ratios),
        },
        "near_equal_payload": equal_payload,
        "recommended_same_node_training_target": (
            "post_rendezvous_completion_kernel_time_us"
        ),
        "portable_lower_bound": "skew_free_intrinsic_kernel_time_us",
        "target_note": (
            "Post-rendezvous is the stable completion label after the final "
            "rank enters, but requires comparable cross-rank timestamps. "
            "Intrinsic uses durations only and remains the portable lower bound "
            "when timestamp synchronization is unavailable."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        f"audited {record_count} records / {len(rows)} workloads; "
        f"intrinsic median IQR={stability['intrinsic']['median_iqr_fraction']:.4f}"
    )


if __name__ == "__main__":
    main()
