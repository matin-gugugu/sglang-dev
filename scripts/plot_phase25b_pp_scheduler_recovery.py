#!/usr/bin/env python3
"""Plot the Phase 25A static-formula error against Phase 25B labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=root / "experiment-results/phase25b_pp_scheduler_teacher",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.result_dir / "analysis/old_vs_scheduler_aggregate.csv"
    with source.open(newline="") as handle:
        rows = {row["policy"]: row for row in csv.DictReader(handle)}

    import matplotlib.pyplot as plt

    policies = ("mb1", "mb4", "mb16")
    panels = (
        ("calls_wape", "Calls WAPE", 100.0, "%"),
        ("mean_calls_histogram_tv", "Histogram TV", 1.0, ""),
        (
            "mean_normalized_log_payload_emd",
            "Normalized log-payload EMD",
            1.0,
            "",
        ),
        ("common_reference_cost_mape", "Reference-cost MAPE", 100.0, "%"),
    )
    colors = ("#4C78A8", "#F58518", "#E45756")
    figure, axes = plt.subplots(2, 2, figsize=(9.4, 6.5), constrained_layout=True)
    for axis, (field, title, scale, suffix) in zip(axes.flat, panels):
        values = [float(rows[policy][field]) * scale for policy in policies]
        bars = axis.bar([policy.upper() for policy in policies], values, color=colors)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}{suffix}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        axis.margins(y=0.20)
    figure.suptitle(
        "Phase 25A static PP formula error vs scheduler-faithful full-window labels"
    )
    figure.savefig(
        args.result_dir / "analysis/old_vs_scheduler_metrics.png",
        dpi=180,
        metadata={"Software": "matplotlib"},
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
