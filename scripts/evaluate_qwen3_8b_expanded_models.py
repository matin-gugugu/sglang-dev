#!/usr/bin/env python3
"""Train and evaluate four communication-cost models on the Phase 4 dataset."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from evaluate_pattern_cost_ablation import BackendAwareCostCurve


MODEL_NAMES = (
    "total_bytes_only",
    "three_hard_bins",
    "continuous_histogram",
    "continuous_histogram_dnn_residual",
)
MODEL_LABELS = {
    "total_bytes_only": "Total bytes only",
    "three_hard_bins": "Three hard bins",
    "continuous_histogram": "Continuous histogram",
    "continuous_histogram_dnn_residual": "Continuous + DNN residual",
}
MODEL_COLORS = {
    "total_bytes_only": "#9D755D",
    "three_hard_bins": "#F58518",
    "continuous_histogram": "#4C78A8",
    "continuous_histogram_dnn_residual": "#54A24B",
}
BIN_DEFINITIONS = (
    ("small", 0, 64 * 1024),
    ("medium", 64 * 1024, 4 * 1024 * 1024),
    ("large", 4 * 1024 * 1024, math.inf),
)
PHASE_TP_GROUPS = tuple(
    (phase, tp) for phase in ("prefill", "decode") for tp in (2, 4, 8)
)


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repo_root / "experiment-results" / "phase4" / "qwen3_8b_expanded",
    )
    parser.add_argument(
        "--custom-curve",
        type=Path,
        default=repo_root
        / "experiment-results"
        / "phase2"
        / "summary_l1_custom_kernel_curve"
        / "custom_kernel_curve_summary.csv",
    )
    parser.add_argument(
        "--nccl-curve",
        type=Path,
        default=repo_root
        / "experiment-results"
        / "phase2"
        / "summary_l1_curve"
        / "collective_curve_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "experiment-results"
        / "phase4"
        / "qwen3_8b_prediction_eval",
    )
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--max-epochs", type=int, default=3000)
    parser.add_argument("--patience", type=int, default=250)
    return parser.parse_args()


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


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def bucket_name(payload_bytes):
    for name, lower, upper in BIN_DEFINITIONS:
        if lower < payload_bytes <= upper:
            return name
    raise ValueError(f"payload outside bins: {payload_bytes}")


def workload_key(record):
    workload = record["workload"]
    return (
        record["phase"],
        int(record["full_phase_pattern_demand"]["group_size"]),
        int(workload["batch_size"]),
        int(workload["input_len"]),
        int(workload["output_len"]),
    )


def stable_hash(value, seed):
    payload = f"{seed}:{value}".encode()
    return hashlib.sha256(payload).hexdigest()


def load_dataset(input_dir):
    grouped = defaultdict(list)
    paths = sorted(input_dir.glob("tp*/r*/**/comm_ground_truth.jsonl"))
    if not paths:
        raise ValueError(f"no ground-truth files found below {input_dir}")
    for path in paths:
        for record in read_jsonl(path):
            record["_source"] = str(path)
            grouped[workload_key(record)].append(record)

    rows = []
    for key, repeats in sorted(grouped.items()):
        phase, tp, batch_size, input_len, output_len = key
        if len(repeats) != 3:
            raise ValueError(f"{key}: expected 3 repeats, got {len(repeats)}")
        patterns = [
            record["full_phase_pattern_demand"] for record in repeats
        ]
        reference = patterns[0]
        for candidate in patterns[1:]:
            if candidate != reference:
                raise ValueError(f"{key}: PatternDemand changed across repeats")
        if any(
            not record["alignment"]["exact_one_kernel_per_call"]
            for record in repeats
        ):
            raise ValueError(f"{key}: profiler kernel/call alignment failed")

        calls_by_payload = {
            int(payload): int(count)
            for payload, count in reference[
                "calls_by_input_payload_bytes"
            ].items()
        }
        measured = [
            float(
                record["gpu_ground_truth"]["full_phase_estimate"][
                    "collective_kernel_time_us"
                ]
            )
            for record in repeats
        ]
        structural = [
            float(
                record["gpu_ground_truth"]["full_phase_estimate"][
                    "structural_median_kernel_time_us"
                ]
            )
            for record in repeats
        ]
        wall = [
            float(record["gpu_ground_truth"]["full_phase_wall_time_us"])
            for record in repeats
        ]
        backends = sorted(
            {
                "+".join(
                    sorted(record["gpu_ground_truth"]["backend_kernel_counts"])
                )
                for record in repeats
            }
        )
        if len(backends) != 1:
            raise ValueError(f"{key}: backend changed across repeats: {backends}")
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
                "repeat_count": len(repeats),
                "calls": int(reference["all_reduce_calls"]),
                "logical_payload_bytes": int(reference["input_payload_bytes"]),
                "ring_equivalent_bytes": float(
                    reference["ring_equivalent"]["bytes"]
                ),
                "ring_equivalent_rounds": int(
                    reference["ring_equivalent"]["rounds"]
                ),
                "calls_by_payload_json": json.dumps(
                    calls_by_payload, sort_keys=True
                ),
                "backend_signature": backends[0],
                "target_comm_us": float(statistics.median(measured)),
                "target_comm_us_p25": percentile(measured, 25),
                "target_comm_us_p75": percentile(measured, 75),
                "structural_kernel_us": float(statistics.median(structural)),
                "phase_wall_us": float(statistics.median(wall)),
                "comm_fraction_of_wall": (
                    float(statistics.median(measured))
                    / float(statistics.median(wall))
                ),
                "_calls_by_payload": calls_by_payload,
            }
        )
    return rows


def assign_splits(rows, seed):
    strata = defaultdict(list)
    for row in rows:
        strata[(row["phase"], row["group_size"])].append(row)
    for stratum, values in strata.items():
        values.sort(
            key=lambda row: stable_hash(row["workload_id"], seed)
        )
        count = len(values)
        test_count = max(3, round(count * 0.15))
        validation_count = max(3, round(count * 0.15))
        if test_count + validation_count >= count:
            raise ValueError(f"stratum {stratum} is too small for grouped split")
        for index, row in enumerate(values):
            if index < test_count:
                row["split"] = "test"
            elif index < test_count + validation_count:
                row["split"] = "validation"
            else:
                row["split"] = "train"
    return rows


def group_one_hot(row):
    return np.asarray(
        [
            1.0
            if (row["phase"], row["group_size"]) == group
            else 0.0
            for group in PHASE_TP_GROUPS
        ],
        dtype=np.float64,
    )


def total_bytes_features(row):
    group = group_one_hot(row)
    log_bytes = math.log1p(row["logical_payload_bytes"])
    return np.concatenate((group, group * log_bytes))


def bin_statistics(row):
    stats = {
        name: {"calls": 0, "bytes": 0}
        for name, _, _ in BIN_DEFINITIONS
    }
    for payload, count in row["_calls_by_payload"].items():
        name = bucket_name(payload)
        stats[name]["calls"] += count
        stats[name]["bytes"] += count * payload
    return stats


def three_bin_features(row):
    values = list(group_one_hot(row))
    stats = bin_statistics(row)
    for name, _, _ in BIN_DEFINITIONS:
        values.extend(
            (
                math.log1p(stats[name]["calls"]),
                math.log1p(stats[name]["bytes"]),
            )
        )
    return np.asarray(values, dtype=np.float64)


class StandardizedRidge:
    def __init__(self, alpha=1e-3):
        self.alpha = alpha
        self.mean = None
        self.scale = None
        self.coef = None

    def fit(self, features, targets):
        features = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        self.mean = features.mean(axis=0)
        self.scale = features.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.0
        standardized = (features - self.mean) / self.scale
        design = np.column_stack((np.ones(len(standardized)), standardized))
        regularizer = np.eye(design.shape[1], dtype=np.float64)
        regularizer[0, 0] = 0.0
        self.coef = np.linalg.solve(
            design.T @ design + self.alpha * regularizer,
            design.T @ targets,
        )
        return self

    def predict(self, features):
        features = np.asarray(features, dtype=np.float64)
        standardized = (features - self.mean) / self.scale
        design = np.column_stack((np.ones(len(standardized)), standardized))
        return design @ self.coef

    def serialize(self):
        return {
            "alpha": self.alpha,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficient_with_intercept": self.coef.tolist(),
        }


def continuous_cost(row, curve):
    return sum(
        count * curve.lookup(row["group_size"], payload)
        for payload, count in row["_calls_by_payload"].items()
    )


def fit_continuous_calibration(train_rows):
    ratios = defaultdict(list)
    for row in train_rows:
        ratios[(row["phase"], row["group_size"])].append(
            math.log(row["target_comm_us"] / row["continuous_raw_us"])
        )
    return {
        group: math.exp(statistics.median(values))
        for group, values in ratios.items()
    }


def dnn_features(row):
    stats = bin_statistics(row)
    calls = row["calls"]
    log_payloads = np.asarray(
        [math.log2(payload) for payload in row["_calls_by_payload"]],
        dtype=np.float64,
    )
    weights = np.asarray(
        list(row["_calls_by_payload"].values()), dtype=np.float64
    )
    weighted_mean = float(np.average(log_payloads, weights=weights))
    weighted_variance = float(
        np.average((log_payloads - weighted_mean) ** 2, weights=weights)
    )
    values = [
        *group_one_hot(row),
        math.log1p(row["batch_size"]),
        math.log1p(row["input_len"]),
        math.log1p(row["output_len"]),
        math.log1p(calls),
        math.log1p(row["logical_payload_bytes"]),
        math.log1p(row["ring_equivalent_rounds"]),
        math.log1p(row["continuous_calibrated_us"]),
        weighted_mean,
        math.sqrt(weighted_variance),
        float(np.min(log_payloads)),
        float(np.max(log_payloads)),
    ]
    for name, _, _ in BIN_DEFINITIONS:
        values.extend(
            (
                math.log1p(stats[name]["calls"]),
                math.log1p(stats[name]["bytes"]),
            )
        )
    return np.asarray(values, dtype=np.float64)


class ResidualMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, features):
        return self.network(features).squeeze(-1)


def train_residual_mlp(train_rows, validation_rows, seed, max_epochs, patience):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_x = np.asarray([dnn_features(row) for row in train_rows])
    validation_x = np.asarray(
        [dnn_features(row) for row in validation_rows]
    )
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    train_x = (train_x - mean) / scale
    validation_x = (validation_x - mean) / scale
    train_y = np.asarray(
        [
            math.log(
                row["target_comm_us"] / row["continuous_calibrated_us"]
            )
            for row in train_rows
        ],
        dtype=np.float32,
    )
    validation_y = np.asarray(
        [
            math.log(
                row["target_comm_us"] / row["continuous_calibrated_us"]
            )
            for row in validation_rows
        ],
        dtype=np.float32,
    )
    train_tensor = torch.tensor(train_x, dtype=torch.float32)
    train_target = torch.tensor(train_y, dtype=torch.float32)
    validation_tensor = torch.tensor(validation_x, dtype=torch.float32)
    validation_target = torch.tensor(validation_y, dtype=torch.float32)

    model = ResidualMLP(train_tensor.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-3, weight_decay=3e-3
    )
    loss_function = nn.SmoothL1Loss(beta=0.1)
    best_state = None
    best_epoch = None
    best_validation = math.inf
    stale = 0
    history = []
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        train_prediction = model(train_tensor)
        train_loss = loss_function(train_prediction, train_target)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_prediction = model(validation_tensor)
            validation_loss = loss_function(
                validation_prediction, validation_target
            )
        train_value = float(train_loss.detach())
        validation_value = float(validation_loss.detach())
        if epoch % 25 == 0:
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_value,
                    "validation_loss": validation_value,
                }
            )
        if validation_value < best_validation - 1e-7:
            best_validation = validation_value
            best_epoch = epoch
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, mean, scale, {
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "epochs_ran": epoch + 1,
        "history": history,
    }


def predict_residual_mlp(model, mean, scale, rows):
    features = np.asarray([dnn_features(row) for row in rows])
    features = (features - mean) / scale
    with torch.no_grad():
        log_ratios = model(
            torch.tensor(features, dtype=torch.float32)
        ).numpy()
    return np.asarray(
        [
            row["continuous_calibrated_us"] * math.exp(float(log_ratio))
            for row, log_ratio in zip(rows, log_ratios)
        ],
        dtype=np.float64,
    )


def metric_row(scope, model_name, rows):
    targets = np.asarray([row["target_comm_us"] for row in rows])
    predictions = np.asarray(
        [row[f"{model_name}_predicted_us"] for row in rows]
    )
    absolute = np.abs(predictions - targets)
    ape = absolute / targets
    denominator = float(np.sum((targets - targets.mean()) ** 2))
    r2 = (
        1.0 - float(np.sum((targets - predictions) ** 2)) / denominator
        if denominator
        else None
    )
    return {
        "scope": scope,
        "model": model_name,
        "samples": len(rows),
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
        "p95_ape": percentile(ape, 95),
        "mae_us": float(np.mean(absolute)),
        "rmse_us": float(np.sqrt(np.mean((predictions - targets) ** 2))),
        "r2": r2,
    }


def evaluate_metrics(rows):
    test = [row for row in rows if row["split"] == "test"]
    scopes = {
        "test_all": test,
        "test_prefill": [row for row in test if row["phase"] == "prefill"],
        "test_decode": [row for row in test if row["phase"] == "decode"],
    }
    for tp in (2, 4, 8):
        scopes[f"test_tp{tp}"] = [
            row for row in test if row["group_size"] == tp
        ]
    metrics = []
    for scope, scoped_rows in scopes.items():
        for model_name in MODEL_NAMES:
            metrics.append(metric_row(scope, model_name, scoped_rows))
    return metrics


def plot_results(path, rows, metrics):
    test = [row for row in rows if row["split"] == "test"]
    figure, axes = plt.subplots(2, 2, figsize=(15, 11))

    scope_names = ("test_all", "test_prefill", "test_decode")
    x = np.arange(len(scope_names))
    width = 0.19
    for index, model_name in enumerate(MODEL_NAMES):
        values = []
        for scope in scope_names:
            row = next(
                item
                for item in metrics
                if item["scope"] == scope and item["model"] == model_name
            )
            values.append(100 * row["mape"])
        axes[0, 0].bar(
            x + (index - 1.5) * width,
            values,
            width,
            label=MODEL_LABELS[model_name],
            color=MODEL_COLORS[model_name],
        )
    axes[0, 0].set_xticks(x, ("All", "Prefill", "Decode"))
    axes[0, 0].set_ylabel("MAPE (%)")
    axes[0, 0].set_title("Workload-held-out prediction error")
    axes[0, 0].grid(True, axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    targets = np.asarray([row["target_comm_us"] for row in test])
    minimum = min(targets)
    maximum = max(targets)
    axes[0, 1].plot(
        [minimum, maximum],
        [minimum, maximum],
        color="black",
        linestyle="--",
        linewidth=1,
        label="Ideal",
    )
    for model_name in MODEL_NAMES:
        axes[0, 1].scatter(
            targets,
            [row[f"{model_name}_predicted_us"] for row in test],
            s=28,
            alpha=0.7,
            color=MODEL_COLORS[model_name],
            label=MODEL_LABELS[model_name],
        )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("Measured communication envelope (μs)")
    axes[0, 1].set_ylabel("Predicted communication envelope (μs)")
    axes[0, 1].set_title("Prediction versus measured target")
    axes[0, 1].grid(True, which="both", alpha=0.25)
    axes[0, 1].legend(fontsize=8)

    ape_values = [
        [
            100
            * abs(row[f"{model_name}_predicted_us"] - row["target_comm_us"])
            / row["target_comm_us"]
            for row in test
        ]
        for model_name in MODEL_NAMES
    ]
    axes[1, 0].boxplot(
        ape_values,
        labels=[MODEL_LABELS[name] for name in MODEL_NAMES],
        showfliers=True,
    )
    axes[1, 0].set_ylabel("Absolute percentage error (%)")
    axes[1, 0].set_title("Held-out error distribution")
    axes[1, 0].tick_params(axis="x", rotation=15)
    axes[1, 0].grid(True, axis="y", alpha=0.25)

    residual_target = np.asarray(
        [
            math.log(row["target_comm_us"] / row["continuous_calibrated_us"])
            for row in test
        ]
    )
    residual_prediction = np.asarray(
        [
            math.log(
                row["continuous_histogram_dnn_residual_predicted_us"]
                / row["continuous_calibrated_us"]
            )
            for row in test
        ]
    )
    for tp, marker in ((2, "o"), (4, "s"), (8, "^")):
        indices = [
            index for index, row in enumerate(test) if row["group_size"] == tp
        ]
        axes[1, 1].scatter(
            residual_target[indices],
            residual_prediction[indices],
            marker=marker,
            s=45,
            alpha=0.75,
            label=f"TP={tp}",
        )
    bound = max(
        abs(residual_target).max(),
        abs(residual_prediction).max(),
    )
    axes[1, 1].plot(
        [-bound, bound],
        [-bound, bound],
        color="black",
        linestyle="--",
        linewidth=1,
    )
    axes[1, 1].set_xlabel("Measured log(target / structural)")
    axes[1, 1].set_ylabel("Predicted log residual correction")
    axes[1, 1].set_title("DNN learns only the structural residual")
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].legend()

    figure.suptitle(
        "Qwen3-8B expanded PatternDemand prediction evaluation"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def public_row(row):
    return {
        key: value for key, value in row.items() if not key.startswith("_")
    }


def main():
    args = parse_args()
    torch.set_num_threads(1)
    rows = assign_splits(load_dataset(args.input_dir), args.seed)
    curve = BackendAwareCostCurve(args.custom_curve, args.nccl_curve)
    for row in rows:
        row["continuous_raw_us"] = continuous_cost(row, curve)

    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    test = [row for row in rows if row["split"] == "test"]

    total_model = StandardizedRidge(alpha=1e-3).fit(
        [total_bytes_features(row) for row in train],
        [math.log(row["target_comm_us"]) for row in train],
    )
    bin_model = StandardizedRidge(alpha=3e-3).fit(
        [three_bin_features(row) for row in train],
        [math.log(row["target_comm_us"]) for row in train],
    )
    all_rows = train + validation + test
    total_predictions = np.exp(
        total_model.predict([total_bytes_features(row) for row in all_rows])
    )
    bin_predictions = np.exp(
        bin_model.predict([three_bin_features(row) for row in all_rows])
    )
    for row, total_prediction, bin_prediction in zip(
        all_rows, total_predictions, bin_predictions
    ):
        row["total_bytes_only_predicted_us"] = float(total_prediction)
        row["three_hard_bins_predicted_us"] = float(bin_prediction)

    calibration = fit_continuous_calibration(train)
    for row in rows:
        group = (row["phase"], row["group_size"])
        row["continuous_calibration_scale"] = calibration[group]
        row["continuous_calibrated_us"] = (
            row["continuous_raw_us"] * calibration[group]
        )
        row["continuous_histogram_predicted_us"] = row[
            "continuous_calibrated_us"
        ]

    dnn_model, dnn_mean, dnn_scale, training = train_residual_mlp(
        train,
        validation,
        args.seed,
        args.max_epochs,
        args.patience,
    )
    dnn_predictions = predict_residual_mlp(
        dnn_model, dnn_mean, dnn_scale, rows
    )
    for row, prediction in zip(rows, dnn_predictions):
        row["continuous_histogram_dnn_residual_predicted_us"] = float(
            prediction
        )
        for model_name in MODEL_NAMES:
            row[f"{model_name}_ape"] = abs(
                row[f"{model_name}_predicted_us"] - row["target_comm_us"]
            ) / row["target_comm_us"]

    metrics = evaluate_metrics(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "aggregated_workloads.csv",
        [public_row(row) for row in sorted(rows, key=lambda item: item["workload_id"])],
    )
    write_csv(
        args.output_dir / "split_assignments.csv",
        [
            {
                "workload_id": row["workload_id"],
                "phase": row["phase"],
                "group_size": row["group_size"],
                "batch_size": row["batch_size"],
                "input_len": row["input_len"],
                "output_len": row["output_len"],
                "split": row["split"],
            }
            for row in sorted(rows, key=lambda item: item["workload_id"])
        ],
    )
    write_csv(
        args.output_dir / "predictions.csv",
        [public_row(row) for row in sorted(rows, key=lambda item: item["workload_id"])],
    )
    write_csv(args.output_dir / "metrics.csv", metrics)
    plot_results(
        args.output_dir / "qwen3_8b_expanded_prediction_eval.png",
        rows,
        metrics,
    )
    torch.save(
        {
            "state_dict": dnn_model.state_dict(),
            "feature_mean": dnn_mean,
            "feature_scale": dnn_scale,
            "seed": args.seed,
        },
        args.output_dir / "dnn_residual_model.pt",
    )
    summary = {
        "schema_version": "qwen3-8b-expanded-prediction-v1",
        "dataset": {
            "unique_workloads": len(rows),
            "repeat_count_per_workload": 3,
            "phase_counts": {
                phase: sum(row["phase"] == phase for row in rows)
                for phase in ("prefill", "decode")
            },
            "tp_counts": {
                str(tp): sum(row["group_size"] == tp for row in rows)
                for tp in (2, 4, 8)
            },
            "split_counts": {
                split: sum(row["split"] == split for row in rows)
                for split in ("train", "validation", "test")
            },
            "split_unit": "complete workload; repeats are aggregated before split",
            "target": (
                "median across three repeats of the representative-rank GPU "
                "collective-kernel envelope, scaled from an 8-step Decode window "
                "to full-phase calls; Prefill is fully profiled"
            ),
        },
        "models": {
            "total_bytes_only": total_model.serialize(),
            "three_hard_bins": bin_model.serialize(),
            "continuous_histogram_calibration": {
                f"{phase}_tp{tp}": value
                for (phase, tp), value in sorted(calibration.items())
            },
            "dnn_residual": {
                "architecture": "MLP(input,32,16,1), ReLU",
                "target": "log(measured / calibrated_continuous_histogram)",
                "training": training,
            },
        },
        "metrics": metrics,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(
        f"evaluated {len(rows)} workloads: "
        f"train={len(train)} validation={len(validation)} test={len(test)}"
    )
    for row in metrics:
        if row["scope"] == "test_all":
            print(
                f"{row['model']}: MAPE={100 * row['mape']:.3f}% "
                f"P95={100 * row['p95_ape']:.3f}% R2={row['r2']:.5f}"
            )


if __name__ == "__main__":
    main()
