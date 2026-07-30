#!/usr/bin/env python3
"""Summarize representative-rank, intrinsic, and synchronization-inclusive labels."""

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
        "--input-dir",
        type=Path,
        default=repo_root / "experiment-results" / "phase5" / "qwen3_8b_all_rank",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "experiment-results"
        / "phase5"
        / "qwen3_8b_all_rank_summary",
    )
    return parser.parse_args()


def write_csv(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def workload_key(record):
    workload = record["workload"]
    return (
        record["phase"],
        record["all_rank_ground_truth"]["rank_count"],
        workload["batch_size"],
        workload["input_len"],
        workload["output_len"],
    )


def iqr_fraction(values):
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    if median == 0:
        return 0.0
    return float(
        (np.percentile(values, 75) - np.percentile(values, 25)) / median
    )


def main():
    args = parse_args()
    grouped = defaultdict(list)
    pattern = args.input_dir / "tp*" / "r*" / "**" / "all_rank_ground_truth.jsonl"
    for path_string in glob.glob(str(pattern), recursive=True):
        for line in Path(path_string).read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                grouped[workload_key(record)].append(record)

    rows = []
    for key, records in sorted(grouped.items()):
        if len(records) != 3:
            raise ValueError(f"{key}: expected 3 repeats, got {len(records)}")
        phase, tp, batch_size, input_len, output_len = key
        full = [
            record["all_rank_ground_truth"]["full_phase_estimate"]
            for record in records
        ]
        windows = [
            record["all_rank_ground_truth"]["profiled_window"]
            for record in records
        ]
        demand = records[0]["full_phase_pattern_demand"]
        rank0 = [item["rank0_kernel_time_us"] for item in full]
        max_rank = [item["max_rank_total_kernel_time_us"] for item in full]
        intrinsic = [
            item["skew_free_intrinsic_kernel_time_us"] for item in full
        ]
        post_rendezvous = [
            item["post_rendezvous_completion_kernel_time_us"] for item in full
        ]
        sync_inclusive = [
            item["synchronization_inclusive_max_duration_sum_us"]
            for item in full
        ]
        rows.append(
            {
                "workload_id": (
                    f"{phase}-tp{tp}-b{batch_size}-l{input_len}-m{output_len}"
                ),
                "phase": phase,
                "group_size": tp,
                "batch_size": batch_size,
                "input_len": input_len,
                "output_len": output_len,
                "repeat_count": len(records),
                "calls": int(demand["all_reduce_calls"]),
                "logical_payload_bytes": int(demand["input_payload_bytes"]),
                "message_payload_bytes": int(
                    next(
                        iter(demand["calls_by_input_payload_bytes"])
                    )
                ),
                "ring_equivalent_bytes": float(
                    demand["ring_equivalent"]["bytes"]
                ),
                "ring_equivalent_rounds": int(
                    demand["ring_equivalent"]["rounds"]
                ),
                "rank0_us_median": statistics.median(rank0),
                "max_rank_total_us_median": statistics.median(max_rank),
                "intrinsic_us_median": statistics.median(intrinsic),
                "post_rendezvous_us_median": statistics.median(
                    post_rendezvous
                ),
                "sync_inclusive_us_median": statistics.median(sync_inclusive),
                "sync_inclusive_over_intrinsic_median": statistics.median(
                    sync_value / intrinsic_value
                    for sync_value, intrinsic_value in zip(
                        sync_inclusive, intrinsic
                    )
                ),
                "rank0_repeat_iqr_fraction": iqr_fraction(rank0),
                "intrinsic_repeat_iqr_fraction": iqr_fraction(intrinsic),
                "post_rendezvous_repeat_iqr_fraction": iqr_fraction(
                    post_rendezvous
                ),
                "sync_inclusive_repeat_iqr_fraction": iqr_fraction(
                    sync_inclusive
                ),
                "max_rank_over_rank0_median": statistics.median(
                    max_value / rank0_value
                    for max_value, rank0_value in zip(max_rank, rank0)
                ),
                "per_call_skew_us_median": statistics.median(
                    item["per_collective_rank_skew_us"]["median"]
                    for item in windows
                ),
                "per_call_skew_us_p95_median": statistics.median(
                    item["per_collective_rank_skew_us"]["p95"]
                    for item in windows
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "all_rank_summary.csv", rows)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    markers = {"prefill": "o", "decode": "s"}
    colors = {2: "#4C78A8", 4: "#F58518", 8: "#E45756"}
    for phase in ("prefill", "decode"):
        for tp in (2, 4, 8):
            selected = [
                row
                for row in rows
                if row["phase"] == phase and row["group_size"] == tp
            ]
            axes[0].scatter(
                [row["rank0_us_median"] for row in selected],
                [row["intrinsic_us_median"] for row in selected],
                marker=markers[phase],
                color=colors[tp],
                s=55,
                alpha=0.8,
                label=f"{phase} TP={tp}",
            )
    bounds = [
        min(row["rank0_us_median"] for row in rows),
        max(row["rank0_us_median"] for row in rows),
    ]
    axes[0].plot(bounds, bounds, color="black", linestyle="--", linewidth=1)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Rank 0 kernel envelope (μs)")
    axes[0].set_ylabel("All-rank skew-free intrinsic cost (μs)")
    axes[0].set_title("Representative rank versus intrinsic label")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8)

    positions = np.arange(len(rows))
    axes[1].bar(
        positions,
        [
            row["sync_inclusive_over_intrinsic_median"]
            for row in rows
        ],
        color=[colors[row["group_size"]] for row in rows],
    )
    axes[1].axhline(1, color="black", linewidth=1)
    axes[1].set_xticks(
        positions,
        [
            f"{row['phase'][0].upper()} TP{row['group_size']}\n"
            f"B{row['batch_size']} L{row['input_len']} M{row['output_len']}"
            for row in rows
        ],
        rotation=70,
        ha="right",
        fontsize=7,
    )
    axes[1].set_ylabel("Synchronization-inclusive / intrinsic")
    axes[1].set_title("Inflation from cross-rank entry skew")
    axes[1].grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        args.output_dir / "qwen3_8b_all_rank_critical.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    equal_payload_shapes = ((1, 512), (4, 128), (16, 32))
    equal_payload_rows = {
        tp: [
            next(
                row
                for row in rows
                if row["phase"] == "decode"
                and row["group_size"] == tp
                and row["input_len"] == 2048
                and row["batch_size"] == batch_size
                and row["output_len"] == output_len
            )
            for batch_size, output_len in equal_payload_shapes
        ]
        for tp in (2, 4, 8)
    }
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 5.2), sharey=True)
    shape_colors = ("#4C78A8", "#F58518", "#54A24B")
    for axis, tp in zip(axes, (2, 4, 8)):
        selected = equal_payload_rows[tp]
        positions = np.arange(len(selected))
        seconds = [
            row["intrinsic_us_median"] / 1_000_000 for row in selected
        ]
        bars = axis.bar(positions, seconds, color=shape_colors)
        axis.set_xticks(
            positions,
            [
                f"B{row['batch_size']} / M{row['output_len']}"
                for row in selected
            ],
        )
        for bar, row in zip(bars, selected):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                (
                    f"{row['calls']:,} calls\n"
                    f"{row['message_payload_bytes'] / 1024:.0f} KiB/msg"
                ),
                ha="center",
                va="bottom",
                fontsize=8,
            )
        payload_mib = [
            row["logical_payload_bytes"] / (1024**2) for row in selected
        ]
        axis.set_title(
            f"TP={tp}\nlogical payload {min(payload_mib):.0f}–"
            f"{max(payload_mib):.0f} MiB"
        )
        axis.set_xlabel("Batch size / output length")
        axis.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("Skew-free intrinsic communication time (s)")
    figure.suptitle(
        "Near-equal total payload, different message shapes and call counts",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(
        args.output_dir / "qwen3_8b_equal_payload_all_rank.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 6))
    for phase in ("prefill", "decode"):
        for tp in (2, 4, 8):
            selected = [
                row
                for row in rows
                if row["phase"] == phase and row["group_size"] == tp
            ]
            axis.scatter(
                [
                    100 * row["rank0_repeat_iqr_fraction"]
                    for row in selected
                ],
                [
                    100 * row["intrinsic_repeat_iqr_fraction"]
                    for row in selected
                ],
                marker=markers[phase],
                color=colors[tp],
                s=58,
                alpha=0.8,
                label=f"{phase} TP={tp}",
            )
    max_iqr = 100 * max(
        max(row["rank0_repeat_iqr_fraction"] for row in rows),
        max(row["intrinsic_repeat_iqr_fraction"] for row in rows),
    )
    axis.plot([0, max_iqr], [0, max_iqr], "--", color="black", linewidth=1)
    axis.set_xlabel("Representative rank-0 IQR / median (%)")
    axis.set_ylabel("All-rank intrinsic IQR / median (%)")
    axis.set_title("Repeat stability of the communication-time label")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(
        args.output_dir / "qwen3_8b_all_rank_stability.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    ratios = [
        row["sync_inclusive_over_intrinsic_median"] for row in rows
    ]
    rank0_iqr = [row["rank0_repeat_iqr_fraction"] for row in rows]
    intrinsic_iqr = [row["intrinsic_repeat_iqr_fraction"] for row in rows]
    equal_payload_summary = {}
    for tp, selected in equal_payload_rows.items():
        small_message = selected[0]
        large_message = selected[-1]
        equal_payload_summary[str(tp)] = {
            "small_message_workload": small_message["workload_id"],
            "large_message_workload": large_message["workload_id"],
            "logical_payload_ratio": (
                small_message["logical_payload_bytes"]
                / large_message["logical_payload_bytes"]
            ),
            "intrinsic_time_ratio_small_over_large_message": (
                small_message["intrinsic_us_median"]
                / large_message["intrinsic_us_median"]
            ),
        }
    summary = {
        "schema_version": "all-rank-intrinsic-summary-v2",
        "workload_count": len(rows),
        "repeat_count": 3,
        "synchronization_inclusive_over_intrinsic": {
            "median": float(np.median(ratios)),
            "p95": float(np.percentile(ratios, 95)),
            "max": max(ratios),
        },
        "repeat_iqr_fraction": {
            "rank0_median": float(np.median(rank0_iqr)),
            "all_rank_intrinsic_median": float(np.median(intrinsic_iqr)),
            "intrinsic_more_stable_workloads": sum(
                intrinsic < rank0
                for intrinsic, rank0 in zip(intrinsic_iqr, rank0_iqr)
            ),
            "workload_count": len(rows),
        },
        "near_equal_payload": equal_payload_summary,
        "definition": (
            "Per workload, sum the minimum kernel duration across aligned ranks "
            "for every group-level collective to remove pre-entry synchronization "
            "wait; report the median across three repeats."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        f"summarized {len(rows)} workloads; sync-inclusive/intrinsic "
        f"median={np.median(ratios):.4f} p95={np.percentile(ratios, 95):.4f}"
    )


if __name__ == "__main__":
    main()
