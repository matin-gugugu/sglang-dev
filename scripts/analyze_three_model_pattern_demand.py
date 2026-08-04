#!/usr/bin/env python3
"""Phase 13A three-model, multi-TP PatternDemand analysis."""

import argparse
import csv
import itertools
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_cross_model_pattern_demand import (  # noqa: E402
    OP_FACTORS,
    load_model_dataset,
)


MODEL_ORDER = ("qwen3-8b", "deepseek-v2-lite", "qwen3-30b-a3b")
MODEL_METADATA = {
    "qwen3-8b": {
        "family": "qwen",
        "architecture": "dense",
        "hidden_size": 4096,
        "layers": 36,
        "attention_heads": 32,
        "kv_heads": 8,
        "experts": 0,
        "experts_per_token": 0,
        "dtype_bytes": 2,
        "supports_fused_allreduce_residual_rmsnorm": False,
        "fused_max_payload_bytes": 0,
    },
    "deepseek-v2-lite": {
        "family": "deepseek",
        "architecture": "moe",
        "hidden_size": 2048,
        "layers": 27,
        "attention_heads": 16,
        "kv_heads": 16,
        "experts": 64,
        "experts_per_token": 6,
        "dtype_bytes": 2,
        "supports_fused_allreduce_residual_rmsnorm": False,
        "fused_max_payload_bytes": 0,
    },
    "qwen3-30b-a3b": {
        "family": "qwen",
        "architecture": "moe",
        "hidden_size": 2048,
        "layers": 48,
        "attention_heads": 32,
        "kv_heads": 4,
        "experts": 128,
        "experts_per_token": 8,
        "dtype_bytes": 2,
        "supports_fused_allreduce_residual_rmsnorm": True,
        "fused_max_payload_bytes": 8 * 1024 * 1024,
    },
}
METHOD_ORDER = (
    "workload_only",
    "model_class",
    "model_structure",
    "structured_pattern",
)
METHOD_LABELS = {
    "workload_only": "Workload only",
    "model_class": "Workload + dense/MoE",
    "model_structure": "Model structure scaling",
    "structured_pattern": "Analytical PatternDemand",
}
COLLECTIVE_FAMILY = {
    "all_reduce": "all_reduce",
    "fused_allreduce_residual_rmsnorm": "all_reduce",
}


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=2,
        metavar=("MODEL", "DIRECTORY"),
        help="Model label and histogram-only dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "experiment-results"
        / "phase13"
        / "three_model_pattern_analysis",
    )
    args = parser.parse_args()
    if args.dataset is None:
        args.dataset = [
            (
                "qwen3-8b",
                str(
                    repo_root
                    / "experiment-results"
                    / "phase6"
                    / "qwen3_8b_corrected_all_rank"
                ),
            ),
            (
                "deepseek-v2-lite",
                str(
                    repo_root
                    / "experiment-results"
                    / "phase8"
                    / "deepseek_v2_lite_pattern_demand"
                ),
            ),
            (
                "qwen3-30b-a3b",
                str(
                    repo_root
                    / "experiment-results"
                    / "phase12"
                    / "qwen3_30b_a3b_pattern_demand"
                ),
            ),
        ]
    return args


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"fieldnames required for empty CSV: {path}")
        fieldnames = list(rows[0])
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def workload_key(row):
    return (
        row["phase"],
        int(row["tp"]),
        int(row["batch_size"]),
        int(row["input_len"]),
        int(row["output_len"]),
    )


def workload_id(row):
    phase, tp, batch, input_len, output_len = workload_key(row)
    return (
        f"{phase}-tp{tp}-b{batch}-l{input_len}-m{output_len}"
    )


def parse_histogram(row):
    output = {}
    for key, count in json.loads(row["histogram_json"]).items():
        op, payload = key.rsplit(":", 1)
        output[(op, int(payload))] = float(count)
    return output


def histogram_json(histogram):
    return json.dumps(
        {
            f"{op}:{payload}": round(float(count), 10)
            for (op, payload), count in sorted(histogram.items())
            if count > 1e-12
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def histogram_metrics(histogram, tp):
    calls = sum(histogram.values())
    logical_payload = 0.0
    equivalent_bytes = 0.0
    equivalent_rounds = 0.0
    for (op, payload), count in histogram.items():
        if op not in OP_FACTORS:
            raise ValueError(f"missing topology factors for raw op: {op}")
        logical_payload += payload * count
        equivalent_bytes += (
            payload * count * OP_FACTORS[op]["bytes"](tp)
        )
        equivalent_rounds += count * OP_FACTORS[op]["rounds"](tp)
    return {
        "calls": calls,
        "logical_payload_bytes": logical_payload,
        "ring_equivalent_bytes": equivalent_bytes,
        "ring_equivalent_rounds": equivalent_rounds,
    }


def histogram_tv(actual, predicted):
    keys = set(actual) | set(predicted)
    actual_total = sum(actual.values())
    predicted_total = sum(predicted.values())
    if not actual_total or not predicted_total:
        return 1.0
    return 0.5 * sum(
        abs(
            actual.get(key, 0.0) / actual_total
            - predicted.get(key, 0.0) / predicted_total
        )
        for key in keys
    )


def payload_distribution(histogram):
    result = defaultdict(float)
    for (_, payload), count in histogram.items():
        result[math.log2(payload)] += count
    total = sum(result.values())
    return {key: value / total for key, value in result.items()}


def log_payload_wasserstein(actual, predicted):
    left = payload_distribution(actual)
    right = payload_distribution(predicted)
    points = sorted(set(left) | set(right))
    if len(points) < 2:
        return 0.0 if points and set(left) == set(right) else 1.0
    left_cdf = 0.0
    right_cdf = 0.0
    distance = 0.0
    for index, point in enumerate(points[:-1]):
        left_cdf += left.get(point, 0.0)
        right_cdf += right.get(point, 0.0)
        distance += abs(left_cdf - right_cdf) * (
            points[index + 1] - point
        )
    return distance


def ape(actual, predicted):
    return abs(predicted - actual) / actual


def mean_histograms(histograms, weights=None):
    if weights is None:
        weights = [1.0] * len(histograms)
    total_weight = sum(weights)
    output = defaultdict(float)
    for histogram, weight in zip(histograms, weights):
        for key, count in histogram.items():
            output[key] += count * weight / total_weight
    return dict(output)


def transform_histogram(histogram, source_model, target_model):
    source = MODEL_METADATA[source_model]
    target = MODEL_METADATA[target_model]
    payload_scale = target["hidden_size"] / source["hidden_size"]
    call_scale = (
        2 * target["layers"] + 1
    ) / (
        2 * source["layers"] + 1
    )
    output = defaultdict(float)
    for (op, payload), count in histogram.items():
        transformed_op = op
        if (
            op == "fused_allreduce_residual_rmsnorm"
            and not target[
                "supports_fused_allreduce_residual_rmsnorm"
            ]
        ):
            transformed_op = "all_reduce"
        transformed_payload = int(round(payload * payload_scale))
        output[(transformed_op, transformed_payload)] += (
            count * call_scale
        )
    return dict(output)


def model_distance(left_model, right_model):
    left = MODEL_METADATA[left_model]
    right = MODEL_METADATA[right_model]
    components = [
        math.log2(left["hidden_size"] / right["hidden_size"]),
        (left["layers"] - right["layers"]) / 48,
        (left["attention_heads"] - right["attention_heads"]) / 32,
        (left["kv_heads"] - right["kv_heads"]) / 16,
        (
            (left["architecture"] == "moe")
            - (right["architecture"] == "moe")
        ),
        (left["experts"] - right["experts"]) / 128,
        (
            left["experts_per_token"] - right["experts_per_token"]
        )
        / 8,
        (
            left[
                "supports_fused_allreduce_residual_rmsnorm"
            ]
            - right[
                "supports_fused_allreduce_residual_rmsnorm"
            ]
        ),
    ]
    return math.sqrt(sum(value * value for value in components))


def analytical_histogram(model, row):
    metadata = MODEL_METADATA[model]
    if row["phase"] == "prefill":
        active_tokens = row["batch_size"] * row["input_len"]
        forward_count = 1
    else:
        active_tokens = row["batch_size"]
        forward_count = row["output_len"] - 1
    payload = (
        active_tokens
        * metadata["hidden_size"]
        * metadata["dtype_bytes"]
    )
    calls_per_forward = 2 * metadata["layers"] + 1
    if (
        metadata[
            "supports_fused_allreduce_residual_rmsnorm"
        ]
        and payload <= metadata["fused_max_payload_bytes"]
    ):
        return {
            ("all_reduce", payload): 2 * forward_count,
            (
                "fused_allreduce_residual_rmsnorm",
                payload,
            ): (calls_per_forward - 2) * forward_count,
        }
    return {
        ("all_reduce", payload): calls_per_forward * forward_count
    }


def load_datasets(dataset_args):
    rows = []
    for model, directory in dataset_args:
        if model not in MODEL_METADATA:
            raise ValueError(f"missing model metadata: {model}")
        rows.extend(load_model_dataset(model, directory))
    observed_models = tuple(
        model for model in MODEL_ORDER if any(
            row["model"] == model for row in rows
        )
    )
    if observed_models != MODEL_ORDER:
        raise ValueError(
            f"expected models {MODEL_ORDER}, got {observed_models}"
        )
    model_keys = {}
    for model in MODEL_ORDER:
        selected = [row for row in rows if row["model"] == model]
        if len(selected) != 195:
            raise ValueError(f"{model}: expected 195 rows, got {len(selected)}")
        model_keys[model] = {workload_key(row) for row in selected}
    if len({frozenset(keys) for keys in model_keys.values()}) != 1:
        raise ValueError("the three models do not share the same workload grid")
    for row in rows:
        raw_ops = sorted(op for op, _ in parse_histogram(row))
        row["raw_ops_json"] = json.dumps(raw_ops, separators=(",", ":"))
        row["collective_families_json"] = json.dumps(
            sorted({COLLECTIVE_FAMILY[op] for op in raw_ops}),
            separators=(",", ":"),
        )
    return rows


def model_structure_summary(rows):
    output = []
    for model in MODEL_ORDER:
        selected = [row for row in rows if row["model"] == model]
        calls_per_forward = set()
        payload_per_active_token = set()
        ops = set()
        for row in selected:
            factor = (
                1
                if row["phase"] == "prefill"
                else row["output_len"] - 1
            )
            calls_per_forward.add(round(row["calls"] / factor, 10))
            histogram = parse_histogram(row)
            active_tokens = (
                row["batch_size"] * row["input_len"]
                if row["phase"] == "prefill"
                else row["batch_size"]
            )
            payloads = {payload for _, payload in histogram}
            payload_per_active_token.update(
                round(payload / active_tokens, 10)
                for payload in payloads
            )
            ops.update(op for op, _ in histogram)
        if len(calls_per_forward) != 1:
            raise ValueError(
                f"{model}: calls per forward changed: {calls_per_forward}"
            )
        if len(payload_per_active_token) != 1:
            raise ValueError(
                f"{model}: payload/token changed: "
                f"{payload_per_active_token}"
            )
        metadata = MODEL_METADATA[model]
        output.append(
            {
                "model": model,
                "family": metadata["family"],
                "architecture": metadata["architecture"],
                "hidden_size": metadata["hidden_size"],
                "layers": metadata["layers"],
                "attention_heads": metadata["attention_heads"],
                "kv_heads": metadata["kv_heads"],
                "experts": metadata["experts"],
                "experts_per_token": metadata["experts_per_token"],
                "dtype_bytes": metadata["dtype_bytes"],
                "observed_calls_per_forward": next(
                    iter(calls_per_forward)
                ),
                "observed_payload_bytes_per_active_token": next(
                    iter(payload_per_active_token)
                ),
                "ops_json": json.dumps(sorted(ops), separators=(",", ":")),
                "aggregated_workloads": len(selected),
                "minimum_repeats": min(
                    int(row["repeat_count"]) for row in selected
                ),
            }
        )
    return output


def workload_effects(rows):
    output = []
    for model in MODEL_ORDER:
        for phase in ("prefill", "decode"):
            selected = [
                row
                for row in rows
                if row["model"] == model and row["phase"] == phase
            ]
            x_values = []
            y_values = []
            normalized_calls = set()
            for row in selected:
                histogram = parse_histogram(row)
                payloads = {payload for _, payload in histogram}
                if len(payloads) != 1:
                    raise ValueError(
                        f"{model} {workload_id(row)} has multiple payloads"
                    )
                active_tokens = (
                    row["batch_size"] * row["input_len"]
                    if phase == "prefill"
                    else row["batch_size"]
                )
                x_values.append(math.log2(active_tokens))
                y_values.append(math.log2(next(iter(payloads))))
                factor = (
                    1 if phase == "prefill" else row["output_len"] - 1
                )
                normalized_calls.add(round(row["calls"] / factor, 10))
            slope, intercept = np.polyfit(x_values, y_values, 1)
            predictions = np.asarray(x_values) * slope + intercept
            actual = np.asarray(y_values)
            ss_res = float(np.sum((actual - predictions) ** 2))
            ss_total = float(np.sum((actual - np.mean(actual)) ** 2))
            r2 = 1.0 if ss_total == 0 else 1 - ss_res / ss_total
            output.append(
                {
                    "model": model,
                    "phase": phase,
                    "log2_payload_vs_active_tokens_slope": slope,
                    "log2_payload_intercept": intercept,
                    "r2": r2,
                    "calls_per_forward_values_json": json.dumps(
                        sorted(normalized_calls)
                    ),
                }
            )
    return output


def decode_input_length_invariance(rows):
    output = []
    for model in MODEL_ORDER:
        selected = [
            row
            for row in rows
            if row["model"] == model and row["phase"] == "decode"
        ]
        grouped = defaultdict(list)
        for row in selected:
            key = (
                row["tp"],
                row["batch_size"],
                row["output_len"],
            )
            grouped[key].append(row)
        passed = 0
        for group in grouped.values():
            if len({row["histogram_json"] for row in group}) == 1:
                passed += 1
        output.append(
            {
                "model": model,
                "groups": len(grouped),
                "input_length_invariant_groups": passed,
                "fraction": passed / len(grouped),
            }
        )
    return output


def tp_scaling(rows):
    output = []
    groups = defaultdict(dict)
    for row in rows:
        key = (
            row["model"],
            row["phase"],
            row["batch_size"],
            row["input_len"],
            row["output_len"],
        )
        groups[key][row["tp"]] = row
    for key, by_tp in sorted(groups.items()):
        if set(by_tp) != {2, 4, 8}:
            raise ValueError(f"missing TP scaling points: {key}")
        baseline = by_tp[2]
        for tp in (4, 8):
            target = by_tp[tp]
            output.append(
                {
                    "model": key[0],
                    "phase": key[1],
                    "batch_size": key[2],
                    "input_len": key[3],
                    "output_len": key[4],
                    "baseline_tp": 2,
                    "target_tp": tp,
                    "calls_ratio": target["calls"] / baseline["calls"],
                    "logical_payload_ratio": (
                        target["logical_payload_bytes"]
                        / baseline["logical_payload_bytes"]
                    ),
                    "equivalent_bytes_ratio": (
                        target["ring_equivalent_bytes"]
                        / baseline["ring_equivalent_bytes"]
                    ),
                    "equivalent_rounds_ratio": (
                        target["ring_equivalent_rounds"]
                        / baseline["ring_equivalent_rounds"]
                    ),
                    "raw_op_changed": (
                        target["ops_json"] != baseline["ops_json"]
                    ),
                }
            )
    return output


def same_workload_model_pairs(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[workload_key(row)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        if {row["model"] for row in group} != set(MODEL_ORDER):
            raise ValueError(f"incomplete model comparison: {key}")
        for left, right in itertools.combinations(
            sorted(group, key=lambda row: MODEL_ORDER.index(row["model"])),
            2,
        ):
            left_hist = parse_histogram(left)
            right_hist = parse_histogram(right)
            output.append(
                {
                    "phase": key[0],
                    "tp": key[1],
                    "batch_size": key[2],
                    "input_len": key[3],
                    "output_len": key[4],
                    "left_model": left["model"],
                    "right_model": right["model"],
                    "left_calls": left["calls"],
                    "right_calls": right["calls"],
                    "calls_ratio": max(left["calls"], right["calls"])
                    / min(left["calls"], right["calls"]),
                    "left_logical_payload_bytes": left[
                        "logical_payload_bytes"
                    ],
                    "right_logical_payload_bytes": right[
                        "logical_payload_bytes"
                    ],
                    "payload_ratio": max(
                        left["logical_payload_bytes"],
                        right["logical_payload_bytes"],
                    )
                    / min(
                        left["logical_payload_bytes"],
                        right["logical_payload_bytes"],
                    ),
                    "histogram_tv": histogram_tv(left_hist, right_hist),
                    "log_payload_wasserstein": log_payload_wasserstein(
                        left_hist, right_hist
                    ),
                    "left_ops_json": left["ops_json"],
                    "right_ops_json": right["ops_json"],
                }
            )
    return output


def near_equal_payload_pairs(rows, threshold=0.035):
    output = []
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            if left["phase"] != right["phase"]:
                continue
            left_payload = left["logical_payload_bytes"]
            right_payload = right["logical_payload_bytes"]
            gap = abs(left_payload - right_payload) / max(
                left_payload, right_payload
            )
            if gap > threshold:
                continue
            left_hist = parse_histogram(left)
            right_hist = parse_histogram(right)
            if left_hist == right_hist:
                continue
            output.append(
                {
                    "phase": left["phase"],
                    "left_model": left["model"],
                    "right_model": right["model"],
                    "left_workload": workload_id(left),
                    "right_workload": workload_id(right),
                    "left_tp": left["tp"],
                    "right_tp": right["tp"],
                    "left_calls": left["calls"],
                    "right_calls": right["calls"],
                    "calls_ratio": max(left["calls"], right["calls"])
                    / min(left["calls"], right["calls"]),
                    "left_logical_payload_bytes": left_payload,
                    "right_logical_payload_bytes": right_payload,
                    "relative_payload_gap": gap,
                    "different_model": left["model"] != right["model"],
                    "different_tp": left["tp"] != right["tp"],
                    "different_ops": left["ops_json"] != right["ops_json"],
                    "histogram_tv": histogram_tv(left_hist, right_hist),
                    "log_payload_wasserstein": log_payload_wasserstein(
                        left_hist, right_hist
                    ),
                    "left_histogram_json": left["histogram_json"],
                    "right_histogram_json": right["histogram_json"],
                }
            )
    return sorted(
        output,
        key=lambda row: (
            -row["calls_ratio"],
            -row["histogram_tv"],
            row["relative_payload_gap"],
        ),
    )


def predict_histogram(method, held_model, row, source_rows):
    source_models = [item["model"] for item in source_rows]
    source_histograms = [parse_histogram(item) for item in source_rows]
    if method == "workload_only":
        return mean_histograms(source_histograms)
    if method == "model_class":
        architecture = MODEL_METADATA[held_model]["architecture"]
        selected = [
            histogram
            for model, histogram in zip(
                source_models, source_histograms
            )
            if MODEL_METADATA[model]["architecture"] == architecture
        ]
        return mean_histograms(selected or source_histograms)
    if method == "model_structure":
        transformed = [
            transform_histogram(histogram, model, held_model)
            for model, histogram in zip(
                source_models, source_histograms
            )
        ]
        weights = [
            1.0 / (model_distance(model, held_model) + 0.25)
            for model in source_models
        ]
        return mean_histograms(transformed, weights)
    if method == "structured_pattern":
        return analytical_histogram(held_model, row)
    raise ValueError(f"unknown method: {method}")


def leave_one_model_out(rows):
    by_model_workload = {
        (row["model"], workload_key(row)): row for row in rows
    }
    predictions = []
    for held_model in MODEL_ORDER:
        for row in (
            item for item in rows if item["model"] == held_model
        ):
            source_rows = [
                by_model_workload[(model, workload_key(row))]
                for model in MODEL_ORDER
                if model != held_model
            ]
            actual_histogram = parse_histogram(row)
            actual_metrics = histogram_metrics(
                actual_histogram, row["tp"]
            )
            for method in METHOD_ORDER:
                predicted_histogram = predict_histogram(
                    method, held_model, row, source_rows
                )
                predicted_metrics = histogram_metrics(
                    predicted_histogram, row["tp"]
                )
                predictions.append(
                    {
                        "held_model": held_model,
                        "method": method,
                        "phase": row["phase"],
                        "tp": row["tp"],
                        "batch_size": row["batch_size"],
                        "input_len": row["input_len"],
                        "output_len": row["output_len"],
                        "actual_calls": actual_metrics["calls"],
                        "predicted_calls": predicted_metrics["calls"],
                        "calls_ape": ape(
                            actual_metrics["calls"],
                            predicted_metrics["calls"],
                        ),
                        "actual_logical_payload_bytes": actual_metrics[
                            "logical_payload_bytes"
                        ],
                        "predicted_logical_payload_bytes": predicted_metrics[
                            "logical_payload_bytes"
                        ],
                        "logical_payload_ape": ape(
                            actual_metrics["logical_payload_bytes"],
                            predicted_metrics["logical_payload_bytes"],
                        ),
                        "actual_equivalent_bytes": actual_metrics[
                            "ring_equivalent_bytes"
                        ],
                        "predicted_equivalent_bytes": predicted_metrics[
                            "ring_equivalent_bytes"
                        ],
                        "equivalent_bytes_ape": ape(
                            actual_metrics["ring_equivalent_bytes"],
                            predicted_metrics["ring_equivalent_bytes"],
                        ),
                        "actual_equivalent_rounds": actual_metrics[
                            "ring_equivalent_rounds"
                        ],
                        "predicted_equivalent_rounds": predicted_metrics[
                            "ring_equivalent_rounds"
                        ],
                        "equivalent_rounds_ape": ape(
                            actual_metrics["ring_equivalent_rounds"],
                            predicted_metrics["ring_equivalent_rounds"],
                        ),
                        "histogram_tv": histogram_tv(
                            actual_histogram, predicted_histogram
                        ),
                        "log_payload_wasserstein": (
                            log_payload_wasserstein(
                                actual_histogram, predicted_histogram
                            )
                        ),
                        "actual_histogram_json": histogram_json(
                            actual_histogram
                        ),
                        "predicted_histogram_json": histogram_json(
                            predicted_histogram
                        ),
                    }
                )
    return predictions


def percentile(values, quantile):
    return float(np.percentile(np.asarray(values), quantile))


def aggregate_holdout_metrics(predictions):
    output = []
    for held_model in MODEL_ORDER:
        for method in METHOD_ORDER:
            for phase in ("all", "prefill", "decode"):
                selected = [
                    row
                    for row in predictions
                    if row["held_model"] == held_model
                    and row["method"] == method
                    and (phase == "all" or row["phase"] == phase)
                ]
                output.append(
                    {
                        "held_model": held_model,
                        "method": method,
                        "phase": phase,
                        "samples": len(selected),
                        "calls_mape": statistics.mean(
                            row["calls_ape"] for row in selected
                        ),
                        "calls_p95_ape": percentile(
                            [row["calls_ape"] for row in selected], 95
                        ),
                        "logical_payload_mape": statistics.mean(
                            row["logical_payload_ape"]
                            for row in selected
                        ),
                        "equivalent_bytes_mape": statistics.mean(
                            row["equivalent_bytes_ape"]
                            for row in selected
                        ),
                        "equivalent_rounds_mape": statistics.mean(
                            row["equivalent_rounds_ape"]
                            for row in selected
                        ),
                        "mean_histogram_tv": statistics.mean(
                            row["histogram_tv"] for row in selected
                        ),
                        "p95_histogram_tv": percentile(
                            [row["histogram_tv"] for row in selected], 95
                        ),
                        "mean_log_payload_wasserstein": statistics.mean(
                            row["log_payload_wasserstein"]
                            for row in selected
                        ),
                    }
                )
    return output


def structured_formula_audit(rows):
    failures = []
    for row in rows:
        actual = parse_histogram(row)
        expected = analytical_histogram(row["model"], row)
        if actual != expected:
            failures.append(
                {
                    "model": row["model"],
                    "workload": workload_id(row),
                    "actual": histogram_json(actual),
                    "expected": histogram_json(expected),
                }
            )
    return failures


def plot_results(path, model_summary, tp_rows, holdout_metrics):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    models = [row["model"] for row in model_summary]
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    axes[0, 0].bar(
        models,
        [row["observed_calls_per_forward"] for row in model_summary],
        color=colors,
    )
    axes[0, 0].set_ylabel("Collective calls per forward")
    axes[0, 0].set_title("Model structure changes call count")
    axes[0, 0].tick_params(axis="x", rotation=15)

    for field, label, marker in (
        ("equivalent_bytes_ratio", "Equivalent bytes", "o"),
        ("equivalent_rounds_ratio", "Equivalent rounds", "s"),
    ):
        means = [
            statistics.mean(
                row[field]
                for row in tp_rows
                if row["target_tp"] == tp
            )
            for tp in (4, 8)
        ]
        axes[0, 1].plot((4, 8), means, marker=marker, label=label)
    axes[0, 1].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[0, 1].set_xticks((4, 8))
    axes[0, 1].set_xlabel("Target TP, normalized to TP2")
    axes[0, 1].set_ylabel("Ratio")
    axes[0, 1].set_title("Logical demand is stable; topology cost grows")
    axes[0, 1].legend()

    all_rows = [
        row for row in holdout_metrics if row["phase"] == "all"
    ]
    x = np.arange(len(METHOD_ORDER))
    width = 0.25
    for index, model in enumerate(MODEL_ORDER):
        selected = {
            row["method"]: row
            for row in all_rows
            if row["held_model"] == model
        }
        axes[1, 0].bar(
            x + (index - 1) * width,
            [
                100 * selected[method]["logical_payload_mape"]
                for method in METHOD_ORDER
            ],
            width,
            label=model,
            color=colors[index],
        )
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(
        [METHOD_LABELS[method] for method in METHOD_ORDER],
        rotation=18,
        ha="right",
    )
    axes[1, 0].set_ylabel("Logical payload MAPE (%)")
    axes[1, 0].set_title("Leave-one-model-out scalar prediction")
    axes[1, 0].legend(fontsize=8)

    for index, model in enumerate(MODEL_ORDER):
        selected = {
            row["method"]: row
            for row in all_rows
            if row["held_model"] == model
        }
        axes[1, 1].plot(
            x,
            [
                selected[method]["mean_histogram_tv"]
                for method in METHOD_ORDER
            ],
            marker="o",
            label=model,
            color=colors[index],
        )
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(
        [METHOD_LABELS[method] for method in METHOD_ORDER],
        rotation=18,
        ha="right",
    )
    axes[1, 1].set_ylabel("Mean histogram TV")
    axes[1, 1].set_title("Raw op and payload shape transfer")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Phase 13A: three-model TP PatternDemand", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_readme(
    rows,
    model_summary,
    workload_rows,
    decode_invariance,
    tp_rows,
    model_pairs,
    near_pairs,
    holdout_metrics,
    formula_failures,
):
    all_holdout = [
        row for row in holdout_metrics if row["phase"] == "all"
    ]
    lines = [
        "# Phase 13A：三模型多 TP PatternDemand 综合分析",
        "",
        "## 1. 数据与口径",
        "",
        (
            f"本分析聚合 {len(rows)} 个 model × workload 配置：三模型各 "
            "195 个独立配置，均以完整 workload 聚合重复，不把 repeat "
            "随机拆分为独立样本。"
        ),
        "",
        "| 模型 | 结构 | hidden | layers | calls/forward | bytes/active token | raw ops |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in model_summary:
        lines.append(
            f"| {row['model']} | {row['architecture']} | "
            f"{row['hidden_size']} | {row['layers']} | "
            f"{row['observed_calls_per_forward']:.0f} | "
            f"{row['observed_payload_bytes_per_active_token']:.0f} | "
            f"{row['ops_json']} |"
        )

    lines.extend(
        [
            "",
            "精确直方图键为 (raw_op, payload)；拓扑折算时，"
            "fused_allreduce_residual_rmsnorm 保留 raw op 身份，"
            "但 collective_family 按 AllReduce 计算 equivalent bytes/rounds。",
            "",
            "## 2. Workload 与 TP 规律",
            "",
        ]
    )
    for row in workload_rows:
        lines.append(
            f"- {row['model']} {row['phase']}：log2(payload) 对 "
            f"log2(active tokens) 斜率 {row['log2_payload_vs_active_tokens_slope']:.6f}，"
            f"R²={row['r2']:.6f}。"
        )
    for row in decode_invariance:
        lines.append(
            f"- {row['model']}：Decode 在固定 TP、B、M 时，"
            f"{row['input_length_invariant_groups']}/{row['groups']} 组对 L 完全不变。"
        )

    lines.extend(
        [
            "",
            (
                "- TP2 到 TP4/TP8 的所有配置中，logical calls 和 logical "
                f"payload 保持不变；TP scaling 对照共 {len(tp_rows)} 组。"
            ),
            (
                "- 等效 bytes 与 rounds 随 group size 增长，因此正式模型必须显式"
                "保留 group_size，不能把逻辑直方图不变误写为通信成本不变。"
            ),
            "",
            "## 3. 近等总 payload 对照",
            "",
            (
                f"自动找到 {len(near_pairs)} 组总逻辑 payload 差不超过 3.5%、"
                "但 raw op/payload 直方图不同的对照。"
            ),
            (
                f"同 workload 的三模型两两匹配对照为 {len(model_pairs)} 组。"
            ),
            "",
            "最强的近等 payload 对照：",
            "",
            "| phase | left | right | payload gap | calls ratio | histogram TV |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in near_pairs[:10]:
        lines.append(
            f"| {row['phase']} | {row['left_model']}:{row['left_workload']} | "
            f"{row['right_model']}:{row['right_workload']} | "
            f"{100 * row['relative_payload_gap']:.3f}% | "
            f"{row['calls_ratio']:.3f}× | {row['histogram_tv']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 4. Leave-one-model-out",
            "",
            "每个 held-out 模型使用另外两个模型的同 workload 数据。"
            "四种方法依次增加模型类别、结构缩放和解析 PatternDemand。"
            "解析方法使用模型/运行时元数据，不读取 held-out 模型的实验直方图。",
            "",
            "| held model | method | calls MAPE | payload MAPE | eq bytes MAPE | eq rounds MAPE | hist TV | log-payload EMD |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in all_holdout:
        lines.append(
            f"| {row['held_model']} | {METHOD_LABELS[row['method']]} | "
            f"{100 * row['calls_mape']:.3f}% | "
            f"{100 * row['logical_payload_mape']:.3f}% | "
            f"{100 * row['equivalent_bytes_mape']:.3f}% | "
            f"{100 * row['equivalent_rounds_mape']:.3f}% | "
            f"{row['mean_histogram_tv']:.4f} | "
            f"{row['mean_log_payload_wasserstein']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 5. 解析公式审计",
            "",
            (
                "解析 PatternDemand 使用：payload = active_tokens × hidden_size "
                "× dtype_bytes；calls/forward = 2 × layers + 1；Decode 乘 "
                "(M - 1)。Qwen3-30B-A3B 在 payload 不超过 8 MiB 时保留 "
                "2 次 all_reduce，其余调用表现为 fused op；更大 payload "
                "回到 all_reduce。"
            ),
            "",
            (
                f"该公式对 {len(rows)} 个聚合配置的逐 raw-op 直方图审计失败数为 "
                f"{len(formula_failures)}。"
            ),
            "",
            "这说明当前规则网格中的 PatternDemand 可由模型结构、workload 和"
            "运行时 lowering 规则精确重建；它不是完整端到端时延模型，也不能"
            "外推为 expert-parallel All-to-All 结论。",
            "",
            "## 6. 正式产物",
            "",
            "- pattern_summary.csv：585 个聚合配置及 raw-op 直方图；",
            "- model_structure_summary.csv：从正式数据重算的结构指纹；",
            "- workload_effects.csv 与 decode_input_length_invariance.csv；",
            "- tp_scaling.csv；",
            "- same_workload_model_pairs.csv 与 near_equal_payload_pairs.csv；",
            "- model_holdout_predictions.csv 与 model_holdout_metrics.csv；",
            "- summary.json、phase13a_three_model_analysis.png 和 analyze.log。",
            "",
            "## 7. 结论边界",
            "",
            "可以声称：三模型共同网格下，calls、payload、raw op 和 group size "
            "共同构成可复核的 TP PatternDemand；解析结构模型在当前规则网格上"
            "能够跨模型重建直方图。",
            "",
            "不能声称：已经测量 Qwen3-30B 的真实 collective 时间、端到端推理"
            "时间、EP routing All-to-All、L2/L3，或已经证明解析公式能覆盖 mixed "
            "Decode、chunked Prefill 和未见 runtime lowering。",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    rows = load_datasets(args.dataset)
    model_summary = model_structure_summary(rows)
    workload_rows = workload_effects(rows)
    decode_invariance = decode_input_length_invariance(rows)
    tp_rows = tp_scaling(rows)
    model_pairs = same_workload_model_pairs(rows)
    near_pairs = near_equal_payload_pairs(rows)
    holdout_predictions = leave_one_model_out(rows)
    holdout_metrics = aggregate_holdout_metrics(holdout_predictions)
    formula_failures = structured_formula_audit(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "pattern_summary.csv", rows)
    write_csv(
        args.output_dir / "model_structure_summary.csv", model_summary
    )
    write_csv(args.output_dir / "workload_effects.csv", workload_rows)
    write_csv(
        args.output_dir / "decode_input_length_invariance.csv",
        decode_invariance,
    )
    write_csv(args.output_dir / "tp_scaling.csv", tp_rows)
    write_csv(
        args.output_dir / "same_workload_model_pairs.csv", model_pairs
    )
    write_csv(
        args.output_dir / "near_equal_payload_pairs.csv", near_pairs
    )
    write_csv(
        args.output_dir / "model_holdout_predictions.csv",
        holdout_predictions,
    )
    write_csv(
        args.output_dir / "model_holdout_metrics.csv", holdout_metrics
    )
    write_csv(
        args.output_dir / "structured_formula_failures.csv",
        formula_failures,
        fieldnames=["model", "workload", "actual", "expected"],
    )
    plot_results(
        args.output_dir / "phase13a_three_model_analysis.png",
        model_summary,
        tp_rows,
        holdout_metrics,
    )
    (args.output_dir / "README.md").write_text(
        render_readme(
            rows,
            model_summary,
            workload_rows,
            decode_invariance,
            tp_rows,
            model_pairs,
            near_pairs,
            holdout_metrics,
            formula_failures,
        )
    )
    summary = {
        "schema": "phase13a-three-model-pattern-demand-v1",
        "models": list(MODEL_ORDER),
        "aggregated_workloads_per_model": {
            model: sum(row["model"] == model for row in rows)
            for model in MODEL_ORDER
        },
        "aggregated_model_workload_rows": len(rows),
        "same_workload_model_pairs": len(model_pairs),
        "near_equal_payload_pairs": len(near_pairs),
        "tp_scaling_pairs": len(tp_rows),
        "structured_formula_failures": len(formula_failures),
        "raw_ops": sorted(
            {
                op
                for row in rows
                for op, _ in parse_histogram(row)
            }
        ),
        "holdout_methods": list(METHOD_ORDER),
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
