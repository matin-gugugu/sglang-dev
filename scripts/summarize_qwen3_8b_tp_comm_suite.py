#!/usr/bin/env python3
"""Summarize Qwen3-8B equal-payload communication experiments across TP sizes."""

import argparse
import csv
import glob
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SHAPE_ORDER = {"uniform": 0, "mixed": 1, "longtail": 2}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase1-root", type=Path, default=Path("experiment-results/phase1")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment-results/phase1/summary_cross_tp"),
    )
    return parser.parse_args()


def read_jsonl(path):
    with open(path) as source:
        return [json.loads(line) for line in source if line.strip()]


def cv(values):
    mean = statistics.mean(values)
    return statistics.stdev(values) / mean if len(values) > 1 and mean else 0.0


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def collect_decode_records(root):
    pattern = (
        root
        / "qwen3_8b_tp*_inference_comm"
        / "representative"
        / "decode_equal_payload"
        / "*"
        / "r*"
        / "comm_ground_truth.jsonl"
    )
    rows = []
    for filename in sorted(glob.glob(str(pattern))):
        path = Path(filename)
        match = re.search(r"qwen3_8b_tp(\d+)_inference_comm", filename)
        if not match:
            continue
        tp = int(match.group(1))
        shape = path.parents[1].name
        for record in read_jsonl(filename):
            demand = record["pattern_demand"]
            equivalent = demand["ring_equivalent"]
            kernel = record["gpu_ground_truth"]["collective_kernel_time_us"]
            if not record["alignment"]["exact_one_kernel_per_call"]:
                raise ValueError(f"call/kernel alignment failed in {filename}")
            rows.append(
                {
                    "tp": tp,
                    "shape": shape,
                    "repeat_id": int(record["repeat_id"]),
                    "group_size": demand["group_size"],
                    "logical_payload_bytes": demand["input_payload_bytes"],
                    "group_level_calls": demand["all_reduce_calls"],
                    "ring_alpha_bytes": equivalent["alpha_bytes"],
                    "ring_beta_rounds": equivalent["beta_rounds"],
                    "ring_equivalent_bytes": equivalent["bytes"],
                    "ring_equivalent_rounds": equivalent["rounds"],
                    "kernel_time_total_us": kernel["total"],
                    "kernel_time_median_us": kernel["median_per_invocation"],
                    "kernel_time_p95_us": kernel["p95_per_invocation"],
                    "kernel_time_p99_us": kernel["p99_per_invocation"],
                    "kernel_time_max_us": kernel["max_per_invocation"],
                }
            )
    return rows


def aggregate_decode(rows, common_repeats_only=False):
    grouped = defaultdict(list)
    for row in rows:
        if common_repeats_only and row["repeat_id"] > 2:
            continue
        grouped[(row["tp"], row["shape"])].append(row)

    summary = []
    for (tp, shape), records in sorted(
        grouped.items(), key=lambda item: (item[0][0], SHAPE_ORDER[item[0][1]])
    ):
        totals = [row["kernel_time_total_us"] for row in records]
        per_kernel = [row["kernel_time_median_us"] for row in records]
        first = records[0]
        summary.append(
            {
                "tp": tp,
                "shape": shape,
                "repeats": len(records),
                "group_size": first["group_size"],
                "logical_payload_bytes": first["logical_payload_bytes"],
                "group_level_calls": first["group_level_calls"],
                "ring_alpha_bytes": first["ring_alpha_bytes"],
                "ring_beta_rounds": first["ring_beta_rounds"],
                "ring_equivalent_bytes": first["ring_equivalent_bytes"],
                "ring_equivalent_rounds": first["ring_equivalent_rounds"],
                "median_kernel_time_total_us": statistics.median(totals),
                "mean_kernel_time_total_us": statistics.mean(totals),
                "p95_kernel_time_total_us": float(np.percentile(totals, 95)),
                "min_kernel_time_total_us": min(totals),
                "max_kernel_time_total_us": max(totals),
                "kernel_time_total_cv": cv(totals),
                "median_per_kernel_us": statistics.median(per_kernel),
                "calls_x_median_kernel_us": (
                    first["group_level_calls"] * statistics.median(per_kernel)
                ),
            }
        )
    return summary


def collect_prefill_8mib(root):
    pattern = (
        root
        / "qwen3_8b_tp2_inference_comm"
        / "representative"
        / "prefill_payload_curve"
        / "r*"
        / "comm_ground_truth.jsonl"
    )
    rows = []
    for filename in sorted(glob.glob(str(pattern))):
        for record in read_jsonl(filename):
            if (
                record["phase"] != "prefill"
                or record["workload"]["input_len"] != 1024
            ):
                continue
            kernel = record["gpu_ground_truth"]["collective_kernel_time_us"]
            rows.append(
                {
                    "repeat_id": int(record["repeat_id"]),
                    "payload_bytes_per_call": int(
                        next(
                            iter(
                                record["pattern_demand"][
                                    "calls_by_input_payload_bytes"
                                ]
                            )
                        )
                    ),
                    "group_level_calls": record["pattern_demand"][
                        "all_reduce_calls"
                    ],
                    "kernel_time_total_us": kernel["total"],
                    "kernel_time_median_us": kernel["median_per_invocation"],
                    "kernel_time_p95_us": kernel["p95_per_invocation"],
                    "kernel_time_p99_us": kernel["p99_per_invocation"],
                    "kernel_time_max_us": kernel["max_per_invocation"],
                }
            )
    return rows


def summarize_prefill(rows):
    if not rows:
        return []
    totals = [row["kernel_time_total_us"] for row in rows]
    medians = [row["kernel_time_median_us"] for row in rows]
    return [
        {
            "repeats": len(rows),
            "payload_bytes_per_call": rows[0]["payload_bytes_per_call"],
            "group_level_calls": rows[0]["group_level_calls"],
            "median_kernel_time_total_us": statistics.median(totals),
            "mean_kernel_time_total_us": statistics.mean(totals),
            "p95_kernel_time_total_us": float(np.percentile(totals, 95)),
            "min_kernel_time_total_us": min(totals),
            "max_kernel_time_total_us": max(totals),
            "kernel_time_total_cv": cv(totals),
            "median_per_kernel_us": statistics.median(medians),
        }
    ]


def verify_equal_payload(rows):
    grouped = defaultdict(set)
    for row in rows:
        grouped[row["tp"]].add(row["logical_payload_bytes"])
    invalid = {tp: values for tp, values in grouped.items() if len(values) != 1}
    if invalid:
        raise ValueError(f"equal-payload invariant failed: {invalid}")


def plot_results(path, summary, decode_rows, prefill_rows):
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    tps = sorted({row["tp"] for row in summary})
    colors = {"uniform": "#4C78A8", "mixed": "#F58518", "longtail": "#E45756"}

    for shape in SHAPE_ORDER:
        records = [row for row in summary if row["shape"] == shape]
        axes[0, 0].plot(
            [row["tp"] for row in records],
            [row["ring_equivalent_bytes"] / 1024**2 for row in records],
            marker="o",
            color=colors[shape],
            label=shape,
        )
        axes[0, 1].plot(
            [row["tp"] for row in records],
            [row["ring_equivalent_rounds"] for row in records],
            marker="o",
            color=colors[shape],
            label=shape,
        )
        axes[1, 0].errorbar(
            [row["tp"] for row in records],
            [row["median_kernel_time_total_us"] / 1000 for row in records],
            yerr=[
                [
                    (
                        row["median_kernel_time_total_us"]
                        - row["min_kernel_time_total_us"]
                    )
                    / 1000
                    for row in records
                ],
                [
                    (
                        row["max_kernel_time_total_us"]
                        - row["median_kernel_time_total_us"]
                    )
                    / 1000
                    for row in records
                ],
            ],
            marker="o",
            capsize=4,
            color=colors[shape],
            label=shape,
        )

    axes[0, 0].set(
        xticks=tps,
        xlabel="TP / group size",
        ylabel="Ring-equivalent bytes (MiB)",
        title="A. Equivalent bytes grow with TP",
    )
    axes[0, 1].set(
        xticks=tps,
        xlabel="TP / group size",
        ylabel="Ring-equivalent rounds",
        title="B. Calls and group size jointly determine rounds",
    )
    axes[1, 0].set(
        xticks=tps,
        xlabel="TP / group size",
        ylabel="Measured communication kernel time (ms)",
        title="C. Equal logical payload: measured cost across TP",
    )

    stability = {
        "prefill 8 MiB": [row["kernel_time_total_us"] / 1000 for row in prefill_rows],
        "decode mixed": [
            row["kernel_time_total_us"] / 1000
            for row in decode_rows
            if row["tp"] == 2 and row["shape"] == "mixed"
        ],
        "decode longtail": [
            row["kernel_time_total_us"] / 1000
            for row in decode_rows
            if row["tp"] == 2 and row["shape"] == "longtail"
        ],
    }
    labels = [label for label, values in stability.items() if values]
    values = [stability[label] for label in labels]
    axes[1, 1].boxplot(values, tick_labels=labels, showmeans=True)
    axes[1, 1].set(
        ylabel="Communication kernel time (ms)",
        title="D. TP=2 tail stability after repeated runs",
    )

    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
    for axis in list(axes.flat)[:3]:
        axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    decode_rows = collect_decode_records(args.phase1_root)
    verify_equal_payload(decode_rows)
    summary = aggregate_decode(decode_rows)
    common_summary = aggregate_decode(decode_rows, common_repeats_only=True)
    prefill_rows = collect_prefill_8mib(args.phase1_root)
    prefill_summary = summarize_prefill(prefill_rows)

    write_csv(args.output_dir / "decode_cross_tp_records.csv", decode_rows)
    write_csv(args.output_dir / "decode_cross_tp_summary.csv", summary)
    write_csv(
        args.output_dir / "decode_cross_tp_common_r0_r2_summary.csv",
        common_summary,
    )
    write_csv(args.output_dir / "prefill_8mib_records.csv", prefill_rows)
    write_csv(args.output_dir / "prefill_8mib_summary.csv", prefill_summary)
    with (args.output_dir / "cross_tp_summary.json").open("w") as output:
        json.dump(
            {
                "decode_summary_all_repeats": summary,
                "decode_summary_common_r0_r2": common_summary,
                "prefill_8mib_summary": prefill_summary,
            },
            output,
            indent=2,
        )
        output.write("\n")
    plot_results(
        args.output_dir / "qwen3_8b_cross_tp_results.png",
        common_summary,
        decode_rows,
        prefill_rows,
    )
    print(
        f"Wrote {len(decode_rows)} decode records across "
        f"{sorted({row['tp'] for row in decode_rows})}; "
        f"Prefill 8 MiB repeats={len(prefill_rows)}"
    )


if __name__ == "__main__":
    main()
