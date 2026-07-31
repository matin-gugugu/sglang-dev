#!/usr/bin/env python3
"""Evaluate first-stage PatternDemand prediction across workloads, TP, and models.

The Phase 8 formal grid currently contains one AllReduce payload support point
per workload.  The predicted histogram is therefore represented by its call
count and per-call payload.  Total logical bytes and ring-equivalent rounds are
derived from those two predictions rather than learned as independent labels.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


MODEL_METADATA = {
    "deepseek-v2-lite": {
        "hidden_size": 2048,
        "num_hidden_layers": 27,
        "dtype_bytes": 2,
        "tp_collectives_per_layer": 2,
        "fixed_tp_collectives": 1,
        "is_moe": 1,
    },
    "qwen3-8b": {
        "hidden_size": 4096,
        "num_hidden_layers": 36,
        "dtype_bytes": 2,
        "tp_collectives_per_layer": 2,
        "fixed_tp_collectives": 1,
        "is_moe": 0,
    },
}
PREDICTORS = (
    "categorical_ridge",
    "structure_ridge",
    "analytic_pattern",
    "analytic_residual_mlp",
)
PREDICTOR_LABELS = {
    "categorical_ridge": "Categorical ridge",
    "structure_ridge": "Structure-aware ridge",
    "analytic_pattern": "Analytic PatternDemand",
    "analytic_residual_mlp": "Analytic + residual MLP",
}
PREDICTOR_COLORS = {
    "categorical_ridge": "#9D755D",
    "structure_ridge": "#4C78A8",
    "analytic_pattern": "#F58518",
    "analytic_residual_mlp": "#54A24B",
}
TARGETS = (
    "calls",
    "payload_bytes",
    "total_payload_bytes",
    "ring_equivalent_rounds",
)


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root
        / "experiment-results"
        / "phase8"
        / "cross_model_pattern_analysis"
        / "pattern_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "experiment-results"
        / "phase9"
        / "cross_model_pattern_prediction",
    )
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=40)
    return parser.parse_args()


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


def stable_hash(value, seed):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def row_id(row):
    return (
        f"{row['model']}-{row['phase']}-tp{row['tp']}"
        f"-b{row['batch_size']}-l{row['input_len']}-m{row['output_len']}"
    )


def load_rows(path):
    with path.open() as source:
        raw_rows = list(csv.DictReader(source))
    rows = []
    for raw in raw_rows:
        model = raw["model"]
        if model not in MODEL_METADATA:
            raise ValueError(f"missing structural metadata for {model}")
        histogram = json.loads(raw["histogram_json"])
        if len(histogram) != 1:
            raise ValueError(
                "Phase 9 evaluator expects the current single-support formal "
                f"grid; got {len(histogram)} supports for {raw}"
            )
        (histogram_key, histogram_calls), = histogram.items()
        op, histogram_payload = histogram_key.split(":")
        if op != "all_reduce":
            raise ValueError(f"unsupported op in current evaluator: {op}")
        calls = int(raw["calls"])
        total_payload = int(raw["logical_payload_bytes"])
        payload = int(histogram_payload)
        if int(histogram_calls) != calls or payload * calls != total_payload:
            raise ValueError(f"inconsistent histogram totals for {raw}")
        row = {
            "model": model,
            "phase": raw["phase"],
            "tp": int(raw["tp"]),
            "batch_size": int(raw["batch_size"]),
            "input_len": int(raw["input_len"]),
            "output_len": int(raw["output_len"]),
            "repeat_count": int(raw["repeat_count"]),
            "op": op,
            "calls": float(calls),
            "payload_bytes": float(payload),
            "total_payload_bytes": float(total_payload),
            "ring_equivalent_rounds": float(
                int(raw["ring_equivalent_rounds"])
            ),
        }
        row["row_id"] = row_id(row)
        rows.append(row)
    if len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate model/workload rows")
    return sorted(rows, key=lambda row: row["row_id"])


class StandardizedRidge:
    def __init__(self, alpha):
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


def phase_tp_features(row):
    return [
        float(row["phase"] == "prefill"),
        float(row["phase"] == "decode"),
        float(row["tp"] == 2),
        float(row["tp"] == 4),
        float(row["tp"] == 8),
    ]


def workload_drivers(row):
    prefill = float(row["phase"] == "prefill")
    decode = 1.0 - prefill
    log_batch = math.log(row["batch_size"])
    log_input = math.log(row["input_len"])
    log_output = math.log(row["output_len"])
    log_decode_steps = math.log(max(row["output_len"] - 1, 1))
    return [
        log_batch,
        log_input,
        log_output,
        log_decode_steps,
        prefill * (log_batch + log_input),
        decode * log_batch,
        decode * log_decode_steps,
    ]


def categorical_features(row):
    model_one_hot = [
        float(row["model"] == model) for model in sorted(MODEL_METADATA)
    ]
    return np.asarray(
        phase_tp_features(row) + model_one_hot + workload_drivers(row),
        dtype=np.float64,
    )


def structural_baseline(row):
    metadata = MODEL_METADATA[row["model"]]
    calls_per_forward = (
        metadata["tp_collectives_per_layer"]
        * metadata["num_hidden_layers"]
        + metadata["fixed_tp_collectives"]
    )
    decode_steps = row["output_len"] - 1 if row["phase"] == "decode" else 1
    calls = float(calls_per_forward * decode_steps)
    token_count = row["batch_size"]
    if row["phase"] == "prefill":
        token_count *= row["input_len"]
    payload = float(
        token_count * metadata["hidden_size"] * metadata["dtype_bytes"]
    )
    return calls, payload


def structure_features(row):
    metadata = MODEL_METADATA[row["model"]]
    baseline_calls, baseline_payload = structural_baseline(row)
    return np.asarray(
        phase_tp_features(row)
        + workload_drivers(row)
        + [
            math.log(metadata["hidden_size"]),
            math.log(metadata["num_hidden_layers"]),
            float(metadata["is_moe"]),
            math.log(baseline_calls),
            math.log(baseline_payload),
        ],
        dtype=np.float64,
    )


def fit_ridge(rows, feature_function, alpha):
    model = StandardizedRidge(alpha=alpha)
    features = [feature_function(row) for row in rows]
    targets = [
        [math.log(row["calls"]), math.log(row["payload_bytes"])]
        for row in rows
    ]
    return model.fit(features, targets)


def predict_ridge(model, rows, feature_function):
    log_predictions = model.predict(
        [feature_function(row) for row in rows]
    )
    return np.exp(log_predictions)


class ResidualMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features):
        return self.network(features)


def residual_features(row):
    calls, payload = structural_baseline(row)
    return np.concatenate(
        (
            structure_features(row),
            np.asarray(
                [
                    math.log(calls),
                    math.log(payload),
                    math.log(calls * payload),
                    math.log(calls * 2 * (row["tp"] - 1)),
                ],
                dtype=np.float64,
            ),
        )
    )


def train_residual_mlp(
    train_rows,
    validation_rows,
    seed,
    max_epochs,
    patience,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_x = np.asarray([residual_features(row) for row in train_rows])
    validation_x = np.asarray(
        [residual_features(row) for row in validation_rows]
    )
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    train_x = (train_x - mean) / scale
    validation_x = (validation_x - mean) / scale

    def target(rows):
        values = []
        for row in rows:
            calls, payload = structural_baseline(row)
            values.append(
                [
                    math.log(row["calls"] / calls),
                    math.log(row["payload_bytes"] / payload),
                ]
            )
        return np.asarray(values, dtype=np.float32)

    train_tensor = torch.tensor(train_x, dtype=torch.float32)
    validation_tensor = torch.tensor(validation_x, dtype=torch.float32)
    train_target = torch.tensor(target(train_rows), dtype=torch.float32)
    validation_target = torch.tensor(
        target(validation_rows), dtype=torch.float32
    )
    model = ResidualMLP(train_tensor.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-3, weight_decay=3e-3
    )
    loss_function = nn.SmoothL1Loss(beta=0.05)
    best_state = None
    best_epoch = None
    best_validation = math.inf
    stale = 0
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        train_loss = loss_function(model(train_tensor), train_target)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = loss_function(
                model(validation_tensor), validation_target
            )
        validation_value = float(validation_loss)
        if validation_value < best_validation - 1e-10:
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
    }


def predict_residual_mlp(model, mean, scale, rows):
    features = np.asarray([residual_features(row) for row in rows])
    features = (features - mean) / scale
    with torch.no_grad():
        log_residuals = model(
            torch.tensor(features, dtype=torch.float32)
        ).numpy()
    predictions = []
    for row, residual in zip(rows, log_residuals):
        calls, payload = structural_baseline(row)
        predictions.append(
            [calls * math.exp(float(residual[0])), payload * math.exp(float(residual[1]))]
        )
    return np.asarray(predictions, dtype=np.float64)


def select_validation_keys(keys, count, seed, label):
    ordered = sorted(
        keys, key=lambda key: stable_hash(f"{label}:{key}", seed)
    )
    return set(ordered[:count])


def build_workload_holdout(rows, seed):
    unique_keys = {
        (
            row["phase"],
            row["batch_size"],
            row["input_len"],
            row["output_len"],
        )
        for row in rows
    }
    forced_test = {
        ("prefill", 1, 8192, 8),
        ("prefill", 2, 2048, 8),
        ("prefill", 8, 128, 8),
        ("prefill", 16, 512, 8),
    }
    forced_test.update(
        {
            ("decode", batch, input_len, output)
            for batch, output in ((1, 512), (16, 32))
            for input_len in (128, 2048, 8192)
        }
    )
    if not forced_test <= unique_keys:
        raise ValueError("forced workload holdout is not contained in dataset")
    test = set(forced_test)
    for phase, target_count in (("prefill", 4), ("decode", 9)):
        current = {key for key in test if key[0] == phase}
        candidates = {
            key
            for key in unique_keys
            if key[0] == phase and key not in test
        }
        extra = select_validation_keys(
            candidates,
            max(0, target_count - len(current)),
            seed,
            f"workload-test-{phase}",
        )
        test.update(extra)
    remaining = unique_keys - test
    validation = set()
    for phase, count in (("prefill", 4), ("decode", 9)):
        candidates = {key for key in remaining if key[0] == phase}
        validation.update(
            select_validation_keys(
                candidates, count, seed, f"workload-validation-{phase}"
            )
        )

    def split(row):
        key = (
            row["phase"],
            row["batch_size"],
            row["input_len"],
            row["output_len"],
        )
        if key in test:
            return "test"
        if key in validation:
            return "validation"
        return "train"

    return [{"protocol": "workload_holdout", "fold": "grouped", "split": split}]


def build_tp_holdouts(rows, seed):
    folds = []
    for held_tp in (2, 4, 8):
        candidate_keys = {
            (
                row["model"],
                row["phase"],
                row["batch_size"],
                row["input_len"],
                row["output_len"],
            )
            for row in rows
            if row["tp"] != held_tp
        }
        validation = select_validation_keys(
            candidate_keys,
            max(1, round(0.18 * len(candidate_keys))),
            seed,
            f"tp{held_tp}-validation",
        )

        def split(row, held_tp=held_tp, validation=validation):
            if row["tp"] == held_tp:
                return "test"
            key = (
                row["model"],
                row["phase"],
                row["batch_size"],
                row["input_len"],
                row["output_len"],
            )
            return "validation" if key in validation else "train"

        folds.append(
            {
                "protocol": "tp_holdout",
                "fold": f"tp{held_tp}",
                "split": split,
            }
        )
    return folds


def build_model_holdouts(rows, seed):
    folds = []
    for held_model in sorted(MODEL_METADATA):
        candidate_keys = {
            (
                row["phase"],
                row["batch_size"],
                row["input_len"],
                row["output_len"],
            )
            for row in rows
            if row["model"] != held_model
        }
        validation = select_validation_keys(
            candidate_keys,
            max(1, round(0.18 * len(candidate_keys))),
            seed,
            f"{held_model}-validation",
        )

        def split(row, held_model=held_model, validation=validation):
            if row["model"] == held_model:
                return "test"
            key = (
                row["phase"],
                row["batch_size"],
                row["input_len"],
                row["output_len"],
            )
            return "validation" if key in validation else "train"

        folds.append(
            {
                "protocol": "model_holdout",
                "fold": held_model,
                "split": split,
            }
        )
    return folds


def derived_prediction(row, calls, payload):
    return {
        "calls": float(calls),
        "payload_bytes": float(payload),
        "total_payload_bytes": float(calls * payload),
        "ring_equivalent_rounds": float(calls * 2 * (row["tp"] - 1)),
    }


def prediction_record(protocol, fold, predictor, row, prediction):
    record = {
        "protocol": protocol,
        "fold": fold,
        "predictor": predictor,
        "row_id": row["row_id"],
        "model": row["model"],
        "phase": row["phase"],
        "tp": row["tp"],
        "batch_size": row["batch_size"],
        "input_len": row["input_len"],
        "output_len": row["output_len"],
    }
    for target in TARGETS:
        actual = row[target]
        predicted = prediction[target]
        record[f"actual_{target}"] = actual
        record[f"predicted_{target}"] = predicted
        record[f"{target}_ape"] = abs(predicted - actual) / actual
    record["payload_log2_distance"] = abs(
        math.log2(prediction["payload_bytes"] / row["payload_bytes"])
    )
    record["joint_pattern_within_1pct"] = int(
        record["calls_ape"] <= 0.01
        and record["payload_bytes_ape"] <= 0.01
    )
    return record


def metric_rows(protocol, fold, predictions):
    rows = []
    for predictor in PREDICTORS:
        selected = [
            row for row in predictions if row["predictor"] == predictor
        ]
        for target in TARGETS:
            errors = np.asarray(
                [row[f"{target}_ape"] for row in selected],
                dtype=np.float64,
            )
            rows.append(
                {
                    "protocol": protocol,
                    "fold": fold,
                    "predictor": predictor,
                    "target": target,
                    "samples": len(selected),
                    "mape": float(errors.mean()),
                    "median_ape": float(np.median(errors)),
                    "p95_ape": float(np.percentile(errors, 95)),
                    "max_ape": float(errors.max()),
                    "within_1pct_rate": float((errors <= 0.01).mean()),
                }
            )
        rows.append(
            {
                "protocol": protocol,
                "fold": fold,
                "predictor": predictor,
                "target": "joint_pattern",
                "samples": len(selected),
                "mape": "",
                "median_ape": "",
                "p95_ape": "",
                "max_ape": "",
                "within_1pct_rate": float(
                    np.mean(
                        [row["joint_pattern_within_1pct"] for row in selected]
                    )
                ),
            }
        )
    return rows


def plot_results(path, metrics):
    aggregate = [
        row for row in metrics if row["fold"] == "all"
    ]
    protocols = (
        "workload_holdout",
        "tp_holdout",
        "model_holdout",
    )
    targets = (
        "calls",
        "payload_bytes",
        "total_payload_bytes",
    )
    figure, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    x = np.arange(len(protocols))
    width = 0.19
    for axis, target in zip(axes.flat[:3], targets):
        for index, predictor in enumerate(PREDICTORS):
            values = []
            for protocol in protocols:
                row = next(
                    item
                    for item in aggregate
                    if item["protocol"] == protocol
                    and item["predictor"] == predictor
                    and item["target"] == target
                )
                values.append(100 * float(row["mape"]))
            bars = axis.bar(
                x + (index - 1.5) * width,
                values,
                width,
                color=PREDICTOR_COLORS[predictor],
                label=PREDICTOR_LABELS[predictor],
            )
            for bar, value in zip(bars, values):
                if value < 0.005:
                    axis.text(
                        bar.get_x() + bar.get_width() / 2,
                        0.08,
                        "≈0",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=90,
                    )
        axis.set_xticks(x, ("Unseen\nworkload", "Unseen\nTP", "Unseen\nmodel"))
        axis.set_ylabel("MAPE (%)")
        axis.set_title(
            {
                "calls": "Collective call-count prediction",
                "payload_bytes": "Per-call payload prediction",
                "total_payload_bytes": "Total logical-payload prediction",
            }[target]
        )
        axis.grid(True, axis="y", alpha=0.25)

    axis = axes[1, 1]
    for index, predictor in enumerate(PREDICTORS):
        values = []
        for protocol in protocols:
            row = next(
                item
                for item in aggregate
                if item["protocol"] == protocol
                and item["predictor"] == predictor
                and item["target"] == "joint_pattern"
            )
            values.append(100 * float(row["within_1pct_rate"]))
        axis.bar(
            x + (index - 1.5) * width,
            values,
            width,
            color=PREDICTOR_COLORS[predictor],
            label=PREDICTOR_LABELS[predictor],
        )
    axis.set_xticks(x, ("Unseen\nworkload", "Unseen\nTP", "Unseen\nmodel"))
    axis.set_ylim(0, 105)
    axis.set_ylabel("Rows with calls and payload both within 1% (%)")
    axis.set_title("Joint PatternDemand accuracy")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(fontsize=8, ncol=2, loc="lower left")

    figure.suptitle(
        "Cross-model first-stage PatternDemand prediction"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    torch.set_num_threads(1)
    rows = load_rows(args.input)
    folds = (
        build_workload_holdout(rows, args.seed)
        + build_tp_holdouts(rows, args.seed)
        + build_model_holdouts(rows, args.seed)
    )
    all_predictions = []
    split_assignments = []
    fold_metrics = []
    ridge_models = {}
    residual_models = {}
    split_counts = {}

    for fold_index, fold in enumerate(folds):
        protocol = fold["protocol"]
        fold_name = fold["fold"]
        assigned = [(row, fold["split"](row)) for row in rows]
        train = [row for row, split in assigned if split == "train"]
        validation = [
            row for row, split in assigned if split == "validation"
        ]
        test = [row for row, split in assigned if split == "test"]
        if not train or not validation or not test:
            raise ValueError(
                f"empty split in {protocol}/{fold_name}: "
                f"{len(train)}/{len(validation)}/{len(test)}"
            )
        split_counts[f"{protocol}/{fold_name}"] = {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        }
        for row, split in assigned:
            split_assignments.append(
                {
                    "protocol": protocol,
                    "fold": fold_name,
                    "row_id": row["row_id"],
                    "model": row["model"],
                    "phase": row["phase"],
                    "tp": row["tp"],
                    "batch_size": row["batch_size"],
                    "input_len": row["input_len"],
                    "output_len": row["output_len"],
                    "split": split,
                }
            )

        categorical_model = fit_ridge(
            train, categorical_features, alpha=1e-3
        )
        structure_model = fit_ridge(
            train, structure_features, alpha=1e-6
        )
        categorical_predictions = predict_ridge(
            categorical_model, test, categorical_features
        )
        structure_predictions = predict_ridge(
            structure_model, test, structure_features
        )
        residual_model, residual_mean, residual_scale, training = (
            train_residual_mlp(
                train,
                validation,
                args.seed + fold_index,
                args.max_epochs,
                args.patience,
            )
        )
        residual_predictions = predict_residual_mlp(
            residual_model,
            residual_mean,
            residual_scale,
            test,
        )
        ridge_models[f"{protocol}/{fold_name}"] = {
            "categorical": categorical_model.serialize(),
            "structure": structure_model.serialize(),
        }
        residual_models[f"{protocol}/{fold_name}"] = {
            "state_dict": residual_model.state_dict(),
            "feature_mean": residual_mean,
            "feature_scale": residual_scale,
            "training": training,
        }

        current_predictions = []
        for index, row in enumerate(test):
            analytic_calls, analytic_payload = structural_baseline(row)
            predictor_values = {
                "categorical_ridge": categorical_predictions[index],
                "structure_ridge": structure_predictions[index],
                "analytic_pattern": np.asarray(
                    [analytic_calls, analytic_payload]
                ),
                "analytic_residual_mlp": residual_predictions[index],
            }
            for predictor, (calls, payload) in predictor_values.items():
                record = prediction_record(
                    protocol,
                    fold_name,
                    predictor,
                    row,
                    derived_prediction(row, calls, payload),
                )
                current_predictions.append(record)
                all_predictions.append(record)
        fold_metrics.extend(
            metric_rows(protocol, fold_name, current_predictions)
        )

    aggregate_metrics = []
    for protocol in (
        "workload_holdout",
        "tp_holdout",
        "model_holdout",
    ):
        selected = [
            row for row in all_predictions if row["protocol"] == protocol
        ]
        aggregate_metrics.extend(metric_rows(protocol, "all", selected))
    metrics = fold_metrics + aggregate_metrics

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "split_assignments.csv", split_assignments)
    write_csv(args.output_dir / "predictions.csv", all_predictions)
    write_csv(args.output_dir / "metrics.csv", metrics)
    plot_results(
        args.output_dir / "cross_model_pattern_prediction.png", metrics
    )
    torch.save(
        residual_models,
        args.output_dir / "residual_mlp_models.pt",
    )
    summary = {
        "schema_version": "cross-model-pattern-prediction-v1",
        "dataset": {
            "rows": len(rows),
            "models": sorted({row["model"] for row in rows}),
            "phases": sorted({row["phase"] for row in rows}),
            "tp": sorted({row["tp"] for row in rows}),
            "single_support_histogram_rows": sum(
                1 for _ in rows
            ),
            "structural_metadata": MODEL_METADATA,
        },
        "prediction_target": {
            "direct": ["calls", "payload_bytes"],
            "derived": [
                "total_payload_bytes",
                "ring_equivalent_rounds",
            ],
            "op": "all_reduce",
            "histogram_note": (
                "Every current formal-grid row contains one payload support. "
                "Calls plus payload therefore reconstruct the complete "
                "continuous histogram for this dataset."
            ),
        },
        "split_counts": split_counts,
        "predictors": {
            "categorical_ridge": (
                "Workload, phase, TP, and categorical model identity."
            ),
            "structure_ridge": (
                "Workload, phase, TP, hidden size, layer count, MoE flag, "
                "and architecture-derived call/payload drivers."
            ),
            "analytic_pattern": (
                "Known TP operator template: calls=(2*layers+1) per forward; "
                "payload=B*tokens*hidden_size*dtype_bytes."
            ),
            "analytic_residual_mlp": (
                "Residual MLP predicts log corrections to analytic calls and "
                "payload; it never predicts PatternDemand from scratch."
            ),
        },
        "ridge_models": ridge_models,
        "metrics": aggregate_metrics,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(
        f"evaluated {len(rows)} rows across {len(folds)} leakage-free folds"
    )
    for protocol in (
        "workload_holdout",
        "tp_holdout",
        "model_holdout",
    ):
        print(protocol)
        for predictor in PREDICTORS:
            values = {
                row["target"]: row
                for row in aggregate_metrics
                if row["protocol"] == protocol
                and row["predictor"] == predictor
            }
            print(
                f"  {predictor}: calls={100 * values['calls']['mape']:.4f}% "
                f"payload={100 * values['payload_bytes']['mape']:.4f}% "
                f"total={100 * values['total_payload_bytes']['mape']:.4f}% "
                f"joint@1%={100 * values['joint_pattern']['within_1pct_rate']:.1f}%"
            )


if __name__ == "__main__":
    main()
