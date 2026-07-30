#!/usr/bin/env python3
"""Diagnose the gap between representative-rank and all-rank targets."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODELS = (
    "total_bytes_only",
    "three_hard_bins",
    "continuous_histogram",
    "continuous_histogram_dnn_residual",
)
MODEL_LABELS = {
    "total_bytes_only": "Total bytes",
    "three_hard_bins": "Three bins",
    "continuous_histogram": "Continuous histogram",
    "continuous_histogram_dnn_residual": "Continuous + DNN residual",
}
MODEL_COLORS = {
    "total_bytes_only": "#9D755D",
    "three_hard_bins": "#F58518",
    "continuous_histogram": "#4C78A8",
    "continuous_histogram_dnn_residual": "#54A24B",
}


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-csv",
        type=Path,
        default=repo_root
        / "experiment-results"
        / "phase5"
        / "qwen3_8b_prediction_eval_stabilized"
        / "aggregated_workloads.csv",
    )
    parser.add_argument(
        "--all-rank-csv",
        type=Path,
        default=repo_root
        / "experiment-results"
        / "phase5"
        / "qwen3_8b_all_rank_summary"
        / "all_rank_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "experiment-results"
        / "phase5"
        / "qwen3_8b_all_rank_target_gap",
    )
    return parser.parse_args()


def read_csv(path):
    with path.open() as source:
        return list(csv.DictReader(source))


def write_csv(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    ape = np.abs(predicted - actual) / actual
    return {
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
        "p95_ape": float(np.percentile(ape, 95)),
        "mae_us": float(np.mean(np.abs(predicted - actual))),
    }


def main():
    args = parse_args()
    predictions = {
        row["workload_id"]: row for row in read_csv(args.prediction_csv)
    }
    all_rank = read_csv(args.all_rank_csv)
    rows = []
    for rank_row in all_rank:
        workload_id = rank_row["workload_id"]
        if workload_id not in predictions:
            raise ValueError(f"missing prediction for {workload_id}")
        prediction = predictions[workload_id]
        critical_us = float(rank_row["critical_us_median"])
        representative_us = float(prediction["target_comm_us"])
        row = {
            "workload_id": workload_id,
            "phase": rank_row["phase"],
            "group_size": int(rank_row["group_size"]),
            "batch_size": int(rank_row["batch_size"]),
            "input_len": int(rank_row["input_len"]),
            "output_len": int(rank_row["output_len"]),
            "representative_rank_target_us": representative_us,
            "all_rank_critical_target_us": critical_us,
            "critical_over_representative_target": (
                critical_us / representative_us
            ),
        }
        for model in MODELS:
            predicted_us = float(prediction[f"{model}_predicted_us"])
            row[f"{model}_predicted_us"] = predicted_us
            row[f"{model}_all_rank_ape"] = (
                abs(predicted_us - critical_us) / critical_us
            )
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "all_rank_target_gap.csv", rows)

    actual = [row["all_rank_critical_target_us"] for row in rows]
    model_metrics = {
        model: metrics(
            actual,
            [row[f"{model}_predicted_us"] for row in rows],
        )
        for model in MODELS
    }
    target_ratios = np.asarray(
        [row["critical_over_representative_target"] for row in rows],
        dtype=np.float64,
    )
    summary = {
        "schema_version": "all-rank-target-gap-v1",
        "workload_count": len(rows),
        "critical_over_representative_rank_target": {
            "median": float(np.median(target_ratios)),
            "p95": float(np.percentile(target_ratios, 95)),
            "max": float(np.max(target_ratios)),
        },
        "existing_model_error_against_all_rank_critical": model_metrics,
        "interpretation": (
            "Diagnostic only: the four existing models were trained on the "
            "representative-rank target and are not retrained here. Large error "
            "quantifies target-definition mismatch; it is not a fair final "
            "benchmark of histogram features against corrected labels."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    colors = {2: "#4C78A8", 4: "#F58518", 8: "#E45756"}
    markers = {"prefill": "o", "decode": "s"}
    for phase in ("prefill", "decode"):
        for tp in (2, 4, 8):
            selected = [
                row
                for row in rows
                if row["phase"] == phase and row["group_size"] == tp
            ]
            axes[0].scatter(
                [
                    row["representative_rank_target_us"]
                    for row in selected
                ],
                [
                    row["all_rank_critical_target_us"]
                    for row in selected
                ],
                marker=markers[phase],
                color=colors[tp],
                s=58,
                alpha=0.8,
                label=f"{phase} TP={tp}",
            )
    lower = min(
        min(row["representative_rank_target_us"] for row in rows),
        min(row["all_rank_critical_target_us"] for row in rows),
    )
    upper = max(
        max(row["representative_rank_target_us"] for row in rows),
        max(row["all_rank_critical_target_us"] for row in rows),
    )
    axes[0].plot([lower, upper], [lower, upper], "--", color="black")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Representative-rank target (μs)")
    axes[0].set_ylabel("All-rank critical target (μs)")
    axes[0].set_title("Ground-truth definition gap")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].bar(
        np.arange(len(MODELS)),
        [100 * model_metrics[model]["mape"] for model in MODELS],
        color=[MODEL_COLORS[model] for model in MODELS],
    )
    axes[1].set_xticks(
        np.arange(len(MODELS)),
        [MODEL_LABELS[model] for model in MODELS],
        rotation=18,
        ha="right",
    )
    axes[1].set_ylabel("MAPE against all-rank critical target (%)")
    axes[1].set_title("Existing proxy-trained models without retraining")
    axes[1].grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        args.output_dir / "qwen3_8b_all_rank_target_gap.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(
        f"matched={len(rows)} target_gap_median={np.median(target_ratios):.3f}x"
    )
    for model in MODELS:
        print(f"{model}: MAPE={100 * model_metrics[model]['mape']:.3f}%")


if __name__ == "__main__":
    main()
