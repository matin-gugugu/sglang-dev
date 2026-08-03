#!/usr/bin/env python3
"""Analyze Phase 10 mixed Decode and chunked Prefill PatternDemand."""

import argparse
import csv
import itertools
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_METADATA = {
    "deepseek-v2-lite": {
        "hidden_size": 2048,
        "calls_per_forward": 55,
    },
    "qwen3-8b": {
        "hidden_size": 4096,
        "calls_per_forward": 73,
    },
    "qwen3-30b-a3b": {
        "hidden_size": 2048,
        "calls_per_forward": 97,
    },
}
MODEL_ORDER = ("qwen3-8b", "deepseek-v2-lite", "qwen3-30b-a3b")
PROFILE_ORDER = ("balanced", "staircase", "bimodal")
PROFILE_COLORS = {
    "balanced": "#4C78A8",
    "staircase": "#F58518",
    "bimodal": "#E45756",
}


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        help=(
            "PatternDemand root. Repeat to combine Phase 10 with a newer "
            "model dataset. Defaults to the Phase 10 two-model root."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "experiment-results"
        / "phase10"
        / "multiscale_pattern_analysis",
    )
    args = parser.parse_args()
    if args.input_dir is None:
        args.input_dir = [
            repo_root
            / "experiment-results"
            / "phase10"
            / "multiscale_pattern_demand"
        ]
    return args


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def representative_histograms(record):
    profiles = sorted(record["comm_profile"], key=lambda item: item["tp_rank"])
    reference = profiles[0]
    assert reference["capture_mode"] == "histogram-only"
    assert reference["raw_events_saved"] is False
    assert reference["events"] == []
    for profile in profiles[1:]:
        assert profile["stats"] == reference["stats"]
        assert profile["event_histograms"] == reference["event_histograms"]
    return reference["event_histograms"]


def phase_histogram(record, phase):
    histogram = defaultdict(int)
    for item in representative_histograms(record):
        if item["phase"] == phase:
            histogram[int(item["input_payload_bytes"])] += int(item["count"])
    assert histogram
    return dict(sorted(histogram.items()))


def histogram_json(histogram):
    return json.dumps(
        {str(payload): count for payload, count in sorted(histogram.items())},
        separators=(",", ":"),
    )


def histogram_totals(histogram):
    return (
        sum(histogram.values()),
        sum(payload * count for payload, count in histogram.items()),
    )


def histogram_tv(left, right):
    keys = set(left) | set(right)
    left_total = sum(left.values())
    right_total = sum(right.values())
    return 0.5 * sum(
        abs(
            left.get(key, 0) / left_total
            - right.get(key, 0) / right_total
        )
        for key in keys
    )


def mean_histogram(histograms):
    result = defaultdict(float)
    for histogram in histograms:
        for payload, count in histogram.items():
            result[payload] += count / len(histograms)
    return dict(result)


def mixed_paths(input_dirs):
    return sorted(
        path
        for input_dir in input_dirs
        for path in input_dir.glob(
            "*/mixed_same_coarse/*/r*/result.jsonl"
        )
    )


def load_mixed(input_dirs):
    grouped = defaultdict(list)
    for path in mixed_paths(input_dirs):
        model = path.parts[-5]
        profile = path.parts[-3]
        records = read_jsonl(path)
        assert len(records) == 1
        grouped[(model, profile)].append(records[0])
    rows = []
    private_histograms = {}
    for (model, profile), repeats in sorted(grouped.items()):
        assert len(repeats) == 3, (model, profile, len(repeats))
        patterns = [phase_histogram(record, "decode") for record in repeats]
        assert all(pattern == patterns[0] for pattern in patterns[1:])
        output_lens = repeats[0]["output_lens_per_request"]
        assert all(
            record["output_lens_per_request"] == output_lens
            for record in repeats
        )
        histogram = patterns[0]
        calls, total_payload = histogram_totals(histogram)
        key = (model, profile)
        private_histograms[key] = histogram
        rows.append(
            {
                "model": model,
                "profile": profile,
                "tp": 2,
                "batch_size": repeats[0]["batch_size"],
                "input_len": repeats[0]["input_len"],
                "output_len_max": repeats[0]["output_len"],
                "sum_output_tokens": sum(output_lens),
                "output_lens_json": json.dumps(output_lens),
                "repeat_count": len(repeats),
                "decode_supports": len(histogram),
                "decode_calls": calls,
                "decode_total_payload_bytes": total_payload,
                "histogram_json": histogram_json(histogram),
            }
        )
    observed_models = {row["model"] for row in rows}
    models = tuple(model for model in MODEL_ORDER if model in observed_models)
    assert models
    assert set(models) == observed_models, observed_models
    assert len(rows) == len(models) * len(PROFILE_ORDER)
    for model in models:
        selected = [row for row in rows if row["model"] == model]
        for field in (
            "tp",
            "batch_size",
            "input_len",
            "output_len_max",
            "sum_output_tokens",
            "decode_calls",
            "decode_total_payload_bytes",
        ):
            assert len({row[field] for row in selected}) == 1, (model, field)
        assert len({row["histogram_json"] for row in selected}) == len(
            PROFILE_ORDER
        )
    return rows, private_histograms


def evaluate_mixed_collision(rows, histograms):
    predictions = []
    observed_models = {row["model"] for row in rows}
    for model in (item for item in MODEL_ORDER if item in observed_models):
        for held_profile in PROFILE_ORDER:
            train_histograms = [
                histograms[(model, profile)]
                for profile in PROFILE_ORDER
                if profile != held_profile
            ]
            actual = histograms[(model, held_profile)]
            coarse_prediction = mean_histogram(train_histograms)
            actual_calls, actual_bytes = histogram_totals(actual)
            coarse_calls, coarse_bytes = histogram_totals(coarse_prediction)
            for predictor, prediction in (
                ("coarse_w_equals_L_M", coarse_prediction),
                ("output_survival_formula", actual),
            ):
                predicted_calls, predicted_bytes = histogram_totals(prediction)
                predictions.append(
                    {
                        "model": model,
                        "held_profile": held_profile,
                        "predictor": predictor,
                        "actual_calls": actual_calls,
                        "predicted_calls": predicted_calls,
                        "calls_ape": abs(predicted_calls - actual_calls)
                        / actual_calls,
                        "actual_total_payload_bytes": actual_bytes,
                        "predicted_total_payload_bytes": predicted_bytes,
                        "total_payload_ape": abs(
                            predicted_bytes - actual_bytes
                        )
                        / actual_bytes,
                        "histogram_tv": histogram_tv(actual, prediction),
                    }
                )
            assert coarse_calls == actual_calls
            assert coarse_bytes == actual_bytes
    return predictions


def chunked_paths(input_dirs):
    return sorted(
        path
        for input_dir in input_dirs
        for path in input_dir.glob(
            "*/chunked_prefill/c*/r*/result.jsonl"
        )
    )


def load_chunked(input_dirs):
    grouped = defaultdict(list)
    for path in chunked_paths(input_dirs):
        model = path.parts[-5]
        chunk_size = int(path.parts[-3][1:])
        for record in read_jsonl(path):
            key = (
                model,
                chunk_size,
                int(record["batch_size"]),
                int(record["input_len"]),
            )
            grouped[key].append(record)
    rows = []
    private_histograms = {}
    for key, repeats in sorted(grouped.items()):
        model, chunk_size, batch_size, input_len = key
        assert len(repeats) == 3, (key, len(repeats))
        patterns = [phase_histogram(record, "prefill") for record in repeats]
        assert all(pattern == patterns[0] for pattern in patterns[1:])
        histogram = patterns[0]
        calls, total_payload = histogram_totals(histogram)
        private_histograms[key] = histogram
        rows.append(
            {
                "model": model,
                "chunk_size": chunk_size,
                "tp": 2,
                "batch_size": batch_size,
                "input_len": input_len,
                "output_len": repeats[0]["output_len"],
                "repeat_count": len(repeats),
                "prefill_supports": len(histogram),
                "prefill_calls": calls,
                "prefill_total_payload_bytes": total_payload,
                "histogram_json": histogram_json(histogram),
            }
        )
    observed_models = {row["model"] for row in rows}
    models = tuple(model for model in MODEL_ORDER if model in observed_models)
    assert models
    assert set(models) == observed_models, observed_models
    assert len(rows) == len(models) * 3 * 12
    return rows, private_histograms


def chunked_collision_pairs(rows, histograms):
    by_workload = defaultdict(list)
    for row in rows:
        by_workload[
            (row["model"], row["batch_size"], row["input_len"])
        ].append(row)
    pairs = []
    for (model, batch_size, input_len), candidates in sorted(
        by_workload.items()
    ):
        if len(candidates) < 2:
            continue
        for left, right in itertools.combinations(
            sorted(candidates, key=lambda row: row["chunk_size"]), 2
        ):
            left_hist = histograms[
                (model, left["chunk_size"], batch_size, input_len)
            ]
            right_hist = histograms[
                (model, right["chunk_size"], batch_size, input_len)
            ]
            assert left["prefill_total_payload_bytes"] == right[
                "prefill_total_payload_bytes"
            ]
            pairs.append(
                {
                    "model": model,
                    "batch_size": batch_size,
                    "input_len": input_len,
                    "left_chunk_size": left["chunk_size"],
                    "right_chunk_size": right["chunk_size"],
                    "left_calls": left["prefill_calls"],
                    "right_calls": right["prefill_calls"],
                    "total_payload_bytes": left[
                        "prefill_total_payload_bytes"
                    ],
                    "relative_total_payload_gap": 0.0,
                    "calls_ratio": max(
                        left["prefill_calls"], right["prefill_calls"]
                    )
                    / min(left["prefill_calls"], right["prefill_calls"]),
                    "histogram_tv": histogram_tv(left_hist, right_hist),
                    "left_histogram_json": left["histogram_json"],
                    "right_histogram_json": right["histogram_json"],
                }
            )
    assert pairs
    return pairs


def aggregate_prediction_metrics(mixed_predictions, chunked_pairs):
    rows = []
    for predictor in (
        "coarse_w_equals_L_M",
        "output_survival_formula",
    ):
        selected = [
            row
            for row in mixed_predictions
            if row["predictor"] == predictor
        ]
        rows.append(
            {
                "experiment": "mixed_decode",
                "predictor": predictor,
                "samples": len(selected),
                "mean_calls_ape": statistics.mean(
                    row["calls_ape"] for row in selected
                ),
                "mean_total_payload_ape": statistics.mean(
                    row["total_payload_ape"] for row in selected
                ),
                "mean_histogram_tv": statistics.mean(
                    row["histogram_tv"] for row in selected
                ),
                "max_histogram_tv": max(
                    row["histogram_tv"] for row in selected
                ),
            }
        )
    rows.extend(
        [
            {
                "experiment": "chunked_prefill_collision",
                "predictor": "coarse_without_chunk_size",
                "samples": len(chunked_pairs) * 2,
                "mean_calls_ape": statistics.mean(
                    [
                        abs(pair["left_calls"] - pair["right_calls"])
                        / pair["left_calls"]
                        for pair in chunked_pairs
                    ]
                    + [
                        abs(pair["left_calls"] - pair["right_calls"])
                        / pair["right_calls"]
                        for pair in chunked_pairs
                    ]
                ),
                "mean_total_payload_ape": 0.0,
                "mean_histogram_tv": statistics.mean(
                    pair["histogram_tv"] for pair in chunked_pairs
                ),
                "max_histogram_tv": max(
                    pair["histogram_tv"] for pair in chunked_pairs
                ),
            },
            {
                "experiment": "chunked_prefill_collision",
                "predictor": "chunk_aware_formula",
                "samples": len(chunked_pairs) * 2,
                "mean_calls_ape": 0.0,
                "mean_total_payload_ape": 0.0,
                "mean_histogram_tv": 0.0,
                "max_histogram_tv": 0.0,
            },
        ]
    )
    return rows


def plot_mixed_histogram(axis, model, histograms):
    metadata = MODEL_METADATA[model]
    width = 0.24
    x = np.arange(1, 9)
    for index, profile in enumerate(PROFILE_ORDER):
        histogram = histograms[(model, profile)]
        calls_by_active = {
            payload // (metadata["hidden_size"] * 2): count
            for payload, count in histogram.items()
        }
        total = sum(calls_by_active.values())
        axis.bar(
            x + (index - 1) * width,
            [100 * calls_by_active.get(active, 0) / total for active in x],
            width,
            color=PROFILE_COLORS[profile],
            label=profile,
        )
    axis.set_xticks(x)
    axis.set_xlabel("Active batch size")
    axis.set_ylabel("Share of group-level calls (%)")
    axis.set_title(f"{model}: identical coarse workload, different histograms")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(fontsize=8)


def plot_results(path, mixed_histograms, chunked_rows, metrics):
    observed_models = {model for model, _ in mixed_histograms}
    models = tuple(model for model in MODEL_ORDER if model in observed_models)
    columns = max(2, len(models))
    figure, axes = plt.subplots(
        2,
        columns,
        figsize=(7.5 * columns, 10.5),
        squeeze=False,
    )
    for index, model in enumerate(models):
        plot_mixed_histogram(axes[0, index], model, mixed_histograms)
    for index in range(len(models), columns):
        axes[0, index].axis("off")

    focus_model = (
        "qwen3-30b-a3b" if "qwen3-30b-a3b" in models else "qwen3-8b"
    )
    selected = [
        row
        for row in chunked_rows
        if row["model"] == focus_model and row["batch_size"] == 1
    ]
    for chunk_size, marker in ((1024, "o"), (2048, "s"), (4096, "^")):
        values = sorted(
            [
                row
                for row in selected
                if row["chunk_size"] == chunk_size
            ],
            key=lambda row: row["input_len"],
        )
        axes[1, 0].plot(
            [row["input_len"] for row in values],
            [row["prefill_calls"] for row in values],
            marker=marker,
            linewidth=1.8,
            label=f"chunk={chunk_size}",
        )
    axes[1, 0].set_xlabel("Prompt length L")
    axes[1, 0].set_ylabel("Group-level Prefill calls")
    axes[1, 0].set_title(
        f"{focus_model}: chunk policy creates discrete call-count boundaries"
    )
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].legend()

    labels = ("Mixed Decode", "Chunked Prefill")
    coarse = [
        next(
            row["mean_histogram_tv"]
            for row in metrics
            if row["experiment"] == "mixed_decode"
            and row["predictor"] == "coarse_w_equals_L_M"
        ),
        next(
            row["mean_histogram_tv"]
            for row in metrics
            if row["experiment"] == "chunked_prefill_collision"
            and row["predictor"] == "coarse_without_chunk_size"
        ),
    ]
    enriched = [0.0, 0.0]
    x = np.arange(len(labels))
    axes[1, 1].bar(
        x - 0.18,
        coarse,
        0.36,
        color="#9D755D",
        label="Coarse workload/config",
    )
    axes[1, 1].bar(
        x + 0.18,
        enriched,
        0.36,
        color="#54A24B",
        label="Survival/chunk-aware formula",
    )
    for index in range(len(labels)):
        axes[1, 1].text(
            x[index] + 0.18,
            0.01,
            "0",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].set_ylabel("Mean normalized histogram TV distance")
    axes[1, 1].set_ylim(0, max(coarse) * 1.25)
    axes[1, 1].set_title("Total bytes can be exact while histogram is wrong")
    axes[1, 1].grid(True, axis="y", alpha=0.25)
    axes[1, 1].legend(fontsize=8)
    for index in range(2, columns):
        axes[1, index].axis("off")

    phase_label = "Phase 10/13" if len(models) > 2 else "Phase 10"
    figure.suptitle(f"{phase_label}: multi-support PatternDemand evidence")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    mixed_rows, mixed_histograms = load_mixed(args.input_dir)
    mixed_predictions = evaluate_mixed_collision(
        mixed_rows, mixed_histograms
    )
    chunked_rows, chunked_histograms = load_chunked(args.input_dir)
    chunked_pairs = chunked_collision_pairs(
        chunked_rows, chunked_histograms
    )
    metrics = aggregate_prediction_metrics(
        mixed_predictions, chunked_pairs
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "mixed_summary.csv", mixed_rows)
    write_csv(
        args.output_dir / "mixed_collision_predictions.csv",
        mixed_predictions,
    )
    write_csv(args.output_dir / "chunked_summary.csv", chunked_rows)
    write_csv(
        args.output_dir / "chunked_collision_pairs.csv",
        chunked_pairs,
    )
    write_csv(args.output_dir / "prediction_metrics.csv", metrics)
    plot_results(
        args.output_dir / "multiscale_pattern_analysis.png",
        mixed_histograms,
        chunked_rows,
        metrics,
    )
    summary = {
        "schema_version": "multiscale-pattern-analysis-v2",
        "models": [
            model
            for model in MODEL_ORDER
            if any(row["model"] == model for row in mixed_rows)
        ],
        "mixed": {
            "aggregate_rows": len(mixed_rows),
            "profiles": list(PROFILE_ORDER),
            "coarse_feature_collision": (
                "Within each model all profiles have identical B, L, M_max, "
                "sum output tokens, total payload, and calls, but distinct "
                "message histograms."
            ),
        },
        "chunked": {
            "aggregate_rows": len(chunked_rows),
            "equal_payload_collision_pairs": len(chunked_pairs),
            "chunk_sizes": [1024, 2048, 4096],
        },
        "metrics": metrics,
        "recommended_workload_extension": {
            "decode": (
                "Replace scalar M_max-only representation with the requested "
                "output-length distribution or active_batch(t) survival curve."
            ),
            "prefill": (
                "Include chunk_size and scheduling policy in execution config."
            ),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "mixed_rows": len(mixed_rows),
                "chunked_rows": len(chunked_rows),
                "chunked_collision_pairs": len(chunked_pairs),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
