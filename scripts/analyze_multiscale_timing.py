#!/usr/bin/env python3
"""Analyze Phase 11 all-rank timing labels for multiscale PatternDemand."""

import argparse
import csv
import hashlib
import itertools
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
SUPPORTED_MODEL_ORDER = (
    "qwen3-8b",
    "deepseek-v2-lite",
    "qwen3-30b-a3b",
)
MODEL_ORDER = SUPPORTED_MODEL_ORDER[:2]
MODEL_STYLES = {
    "qwen3-8b": ("o", "#4C78A8"),
    "deepseek-v2-lite": ("s", "#F58518"),
    "qwen3-30b-a3b": ("^", "#54A24B"),
}
BIN_DEFINITIONS = (
    ("small", 0, 64 * 1024),
    ("medium", 64 * 1024, 4 * 1024 * 1024),
    ("large", 4 * 1024 * 1024, math.inf),
)
TARGET_FIELD = "post_rendezvous_completion_kernel_time_us"


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        help=(
            "All-rank timing root. Repeat to combine Phase 11 with a newer "
            "model dataset. Defaults to the Phase 11 two-model root."
        ),
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
        / "phase11"
        / "multiscale_timing_analysis",
    )
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--max-epochs", type=int, default=2500)
    parser.add_argument("--patience", type=int, default=250)
    args = parser.parse_args()
    if args.input_dir is None:
        args.input_dir = [
            repo_root
            / "experiment-results"
            / "phase11"
            / "multiscale_timing_ground_truth"
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
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
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


def stable_hash(value, seed):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def histogram_json(histogram):
    return json.dumps(
        {str(payload): count for payload, count in sorted(histogram.items())},
        separators=(",", ":"),
    )


def op_histogram_json(histogram):
    return json.dumps(
        {
            f"{op}:{payload}": count
            for (op, payload), count in sorted(histogram.items())
        },
        separators=(",", ":"),
    )


def public_row(row):
    return {
        key: value for key, value in row.items() if not key.startswith("_")
    }


def workload_key(record, model, mode, case_label):
    workload = record["workload"]
    return (
        model,
        mode,
        case_label,
        int(workload["batch_size"]),
        int(workload["input_len"]),
        int(workload["output_len"]),
        tuple(int(value) for value in workload["output_lens_per_request"]),
        int(workload["prefill_chunk_size"]),
    )


def workload_id(key):
    (
        model,
        mode,
        case_label,
        batch_size,
        input_len,
        output_len,
        output_lens,
        chunk_size,
    ) = key
    if mode == "mixed_same_coarse":
        return f"{model}-mixed-{case_label}"
    return (
        f"{model}-chunk-c{chunk_size}-b{batch_size}-l{input_len}"
        f"-m{output_len}-{hashlib.sha1(str(output_lens).encode()).hexdigest()[:6]}"
    )


def load_dataset(input_dirs):
    grouped = defaultdict(list)
    sources = sorted(
        (input_dir, path)
        for input_dir in input_dirs
        for path in input_dir.glob("*/*/*/r*/all_rank_ground_truth.jsonl")
    )
    observed_models = {
        path.relative_to(input_dir).parts[0]
        for input_dir, path in sources
    }
    model_order = tuple(
        model for model in SUPPORTED_MODEL_ORDER if model in observed_models
    )
    if set(model_order) != observed_models:
        raise ValueError(f"unexpected model directories: {sorted(observed_models)}")
    expected_files = 18 * len(model_order)
    if len(sources) != expected_files:
        raise ValueError(
            f"expected {expected_files} ground-truth files for "
            f"{len(model_order)} models, found {len(sources)}"
        )
    for input_dir, path in sources:
        relative = path.relative_to(input_dir)
        model, mode, case_label, repeat_dir, _ = relative.parts
        if model not in model_order:
            raise ValueError(f"unexpected model path: {path}")
        for record in read_jsonl(path):
            record["_source"] = str(path)
            record["_repeat_dir"] = repeat_dir
            grouped[workload_key(record, model, mode, case_label)].append(record)

    rows = []
    for key, repeats in sorted(grouped.items(), key=lambda item: str(item[0])):
        if len(repeats) != 3:
            raise ValueError(f"{key}: expected three repeats, got {len(repeats)}")
        repeat_ids = sorted(int(record["repeat_id"]) for record in repeats)
        if repeat_ids != [0, 1, 2]:
            raise ValueError(f"{key}: unexpected repeat ids {repeat_ids}")
        patterns = [record["full_phase_pattern_demand"] for record in repeats]
        if any(pattern != patterns[0] for pattern in patterns[1:]):
            raise ValueError(f"{key}: PatternDemand changed across repeats")
        for record in repeats:
            alignment = record["alignment"]
            required = (
                "exact_count_on_every_rank",
                "identical_backend_sequence",
                "identical_profiled_pattern_demand_on_every_rank",
                "identical_full_phase_pattern_demand_on_every_rank",
            )
            if not all(alignment[field] for field in required):
                raise ValueError(f"{record['_source']}: all-rank alignment failed")

        reference = patterns[0]
        op_payload_entries = reference.get(
            "calls_by_raw_op_and_input_payload_bytes"
        )
        if op_payload_entries is None:
            calls_by_op_payload = {
                ("all_reduce", int(payload)): int(count)
                for payload, count in reference[
                    "calls_by_input_payload_bytes"
                ].items()
            }
        else:
            calls_by_op_payload = {
                (
                    entry["raw_op"],
                    int(entry["input_payload_bytes"]),
                ): int(entry["count"])
                for entry in op_payload_entries
            }
            if len(calls_by_op_payload) != len(op_payload_entries):
                raise ValueError(f"{key}: duplicate raw-op/payload entries")
            if any(
                entry["collective_family"] != "all_reduce"
                for entry in op_payload_entries
            ):
                raise ValueError(f"{key}: unsupported collective family")
        calls_by_payload = defaultdict(int)
        for (_, payload), count in calls_by_op_payload.items():
            calls_by_payload[payload] += count
        calls_by_payload = dict(sorted(calls_by_payload.items()))
        serialized_payloads = {
            int(payload): int(count)
            for payload, count in reference[
                "calls_by_input_payload_bytes"
            ].items()
        }
        if calls_by_payload != serialized_payloads:
            raise ValueError(f"{key}: raw-op histogram marginal disagrees")
        estimates = [
            record["all_rank_ground_truth"]["full_phase_estimate"]
            for record in repeats
        ]
        post = [float(estimate[TARGET_FIELD]) for estimate in estimates]
        intrinsic = [
            float(estimate["skew_free_intrinsic_kernel_time_us"])
            for estimate in estimates
        ]
        sync = [
            float(estimate["synchronization_inclusive_max_duration_sum_us"])
            for estimate in estimates
        ]
        wall = [float(estimate["phase_wall_time_us"]) for estimate in estimates]
        backends = sorted(
            {
                record["all_rank_ground_truth"]["backend_sequence_signature"]
                for record in repeats
            }
        )
        if len(backends) != 1:
            raise ValueError(f"{key}: backend changed across repeats: {backends}")

        (
            model,
            mode,
            case_label,
            batch_size,
            input_len,
            output_len,
            output_lens,
            chunk_size,
        ) = key
        phase = "decode" if mode == "mixed_same_coarse" else "prefill"
        target = float(statistics.median(post))
        calls = sum(calls_by_payload.values())
        logical_payload_bytes = sum(
            payload * count for payload, count in calls_by_payload.items()
        )
        if calls != int(reference["all_reduce_calls"]):
            raise ValueError(f"{key}: histogram calls disagree with aggregate")
        if logical_payload_bytes != int(reference["input_payload_bytes"]):
            raise ValueError(f"{key}: histogram bytes disagree with aggregate")
        rows.append(
            {
                "workload_id": workload_id(key),
                "model": model,
                "mode": mode,
                "case_label": case_label,
                "phase": phase,
                "group_size": int(reference["group_size"]),
                "batch_size": batch_size,
                "input_len": input_len,
                "output_len": output_len,
                "output_lens_json": json.dumps(output_lens),
                "sum_output_tokens": sum(output_lens),
                "prefill_chunk_size": chunk_size,
                "repeat_count": len(repeats),
                "calls": calls,
                "logical_payload_bytes": logical_payload_bytes,
                "ring_equivalent_bytes": float(
                    reference["ring_equivalent"]["bytes"]
                ),
                "ring_equivalent_rounds": int(
                    reference["ring_equivalent"]["rounds"]
                ),
                "payload_supports": len(calls_by_payload),
                "op_payload_supports": len(calls_by_op_payload),
                "calls_by_payload_json": histogram_json(calls_by_payload),
                "calls_by_op_payload_json": op_histogram_json(
                    calls_by_op_payload
                ),
                "raw_ops_json": json.dumps(
                    sorted({op for op, _ in calls_by_op_payload}),
                    separators=(",", ":"),
                ),
                "collective_families_json": '["all_reduce"]',
                "raw_all_reduce_calls": sum(
                    count
                    for (op, _), count in calls_by_op_payload.items()
                    if op == "all_reduce"
                ),
                "fused_allreduce_residual_rmsnorm_calls": sum(
                    count
                    for (op, _), count in calls_by_op_payload.items()
                    if op == "fused_allreduce_residual_rmsnorm"
                ),
                "backend_signature": backends[0],
                "target_post_us": target,
                "target_post_p25_us": percentile(post, 25),
                "target_post_p75_us": percentile(post, 75),
                "post_repeat_iqr_fraction": (
                    (percentile(post, 75) - percentile(post, 25)) / target
                ),
                "post_repeat_values_json": json.dumps(post),
                "intrinsic_us": float(statistics.median(intrinsic)),
                "intrinsic_repeat_iqr_fraction": (
                    (percentile(intrinsic, 75) - percentile(intrinsic, 25))
                    / float(statistics.median(intrinsic))
                ),
                "sync_inclusive_us": float(statistics.median(sync)),
                "sync_repeat_iqr_fraction": (
                    (percentile(sync, 75) - percentile(sync, 25))
                    / float(statistics.median(sync))
                ),
                "phase_wall_us": float(statistics.median(wall)),
                "comm_fraction_of_wall": target / float(statistics.median(wall)),
                "controlled_equal_payload": False,
                "_calls_by_payload": calls_by_payload,
                "_calls_by_op_payload": calls_by_op_payload,
                "_output_lens": output_lens,
            }
        )

    expected_rows = 39 * len(model_order)
    if len(rows) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} aggregated configurations, got {len(rows)}"
        )
    mark_equal_payload_rows(rows)
    return rows, model_order


def mark_equal_payload_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["mode"] == "mixed_same_coarse":
            key = ("mixed", row["model"])
        else:
            key = (
                "chunked",
                row["model"],
                row["batch_size"],
                row["input_len"],
            )
        groups[key].append(row)
    for candidates in groups.values():
        for left, right in itertools.combinations(candidates, 2):
            if (
                left["logical_payload_bytes"]
                == right["logical_payload_bytes"]
                and left["calls_by_op_payload_json"]
                != right["calls_by_op_payload_json"]
            ):
                left["controlled_equal_payload"] = True
                right["controlled_equal_payload"] = True


def assign_splits(rows, seed):
    strata = defaultdict(list)
    for row in rows:
        strata[(row["model"], row["mode"])].append(row)
    for stratum, values in sorted(strata.items()):
        values.sort(key=lambda row: stable_hash(row["workload_id"], seed))
        if stratum[1] == "mixed_same_coarse":
            if len(values) != 3:
                raise ValueError(f"{stratum}: expected three profiles")
            counts = {"test": 1, "validation": 1}
        else:
            if len(values) != 36:
                raise ValueError(f"{stratum}: expected 36 chunked configurations")
            counts = {"test": 6, "validation": 6}
        test_ids = {row["workload_id"] for row in values[: counts["test"]]}
        validation_ids = {
            row["workload_id"]
            for row in values[
                counts["test"] : counts["test"] + counts["validation"]
            ]
        }
        for row in values:
            if row["workload_id"] in test_ids:
                row["split"] = "test"
            elif row["workload_id"] in validation_ids:
                row["split"] = "validation"
            else:
                row["split"] = "train"
    return rows


def phase_one_hot(row):
    return np.asarray(
        [float(row["phase"] == phase) for phase in ("prefill", "decode")],
        dtype=np.float64,
    )


def total_bytes_features(row):
    group = phase_one_hot(row)
    return np.concatenate(
        (group, group * math.log1p(row["logical_payload_bytes"]))
    )


def bucket_name(payload_bytes):
    for name, lower, upper in BIN_DEFINITIONS:
        if lower < payload_bytes <= upper:
            return name
    raise ValueError(f"payload outside bins: {payload_bytes}")


def bin_statistics(row):
    stats = {
        name: {"calls": 0, "bytes": 0}
        for name, _, _ in BIN_DEFINITIONS
    }
    for payload, count in row["_calls_by_payload"].items():
        name = bucket_name(payload)
        stats[name]["calls"] += count
        stats[name]["bytes"] += payload * count
    return stats


def three_bin_features(row):
    values = list(phase_one_hot(row))
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


def continuous_cost(row, curve):
    return sum(
        count * curve.lookup(row["group_size"], payload)
        for payload, count in row["_calls_by_payload"].items()
    )


def fit_continuous_calibration(train_rows):
    ratios = defaultdict(list)
    for row in train_rows:
        ratios[row["phase"]].append(
            math.log(row["target_post_us"] / row["continuous_raw_us"])
        )
    missing = {"prefill", "decode"} - set(ratios)
    if missing:
        raise ValueError(f"training split lacks phases: {sorted(missing)}")
    return {
        phase: math.exp(statistics.median(values))
        for phase, values in ratios.items()
    }


def dnn_features(row):
    stats = bin_statistics(row)
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
    output_lens = np.asarray(row["_output_lens"], dtype=np.float64)
    values = [
        *phase_one_hot(row),
        math.log1p(row["batch_size"]),
        math.log1p(row["input_len"]),
        math.log1p(row["output_len"]),
        math.log1p(row["prefill_chunk_size"]),
        math.log1p(row["sum_output_tokens"]),
        math.log1p(row["calls"]),
        math.log1p(row["logical_payload_bytes"]),
        math.log1p(row["ring_equivalent_rounds"]),
        math.log1p(row["continuous_calibrated_us"]),
        math.log1p(row["payload_supports"]),
        math.log1p(row["op_payload_supports"]),
        math.log1p(row["raw_all_reduce_calls"]),
        math.log1p(row["fused_allreduce_residual_rmsnorm_calls"]),
        (
            row["fused_allreduce_residual_rmsnorm_calls"]
            / row["calls"]
        ),
        weighted_mean,
        math.sqrt(weighted_variance),
        float(np.min(log_payloads)),
        float(np.max(log_payloads)),
        float(np.mean(output_lens)),
        float(np.std(output_lens)),
        float(np.min(output_lens)),
        float(np.max(output_lens)),
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
                row["target_post_us"] / row["continuous_calibrated_us"]
            )
            for row in train_rows
        ],
        dtype=np.float32,
    )
    validation_y = np.asarray(
        [
            math.log(
                row["target_post_us"] / row["continuous_calibrated_us"]
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
    if best_state is None:
        raise RuntimeError("residual MLP did not produce a checkpoint")
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


def fit_and_predict(rows, train_ids, validation_ids, seed, max_epochs, patience):
    working = [dict(row) for row in rows]
    train = [row for row in working if row["workload_id"] in train_ids]
    validation = [
        row for row in working if row["workload_id"] in validation_ids
    ]
    if not train or not validation:
        raise ValueError("empty train or validation split")
    total_model = StandardizedRidge(alpha=1e-3).fit(
        [total_bytes_features(row) for row in train],
        [math.log(row["target_post_us"]) for row in train],
    )
    bin_model = StandardizedRidge(alpha=3e-3).fit(
        [three_bin_features(row) for row in train],
        [math.log(row["target_post_us"]) for row in train],
    )
    total_predictions = np.exp(
        total_model.predict([total_bytes_features(row) for row in working])
    )
    bin_predictions = np.exp(
        bin_model.predict([three_bin_features(row) for row in working])
    )
    calibration = fit_continuous_calibration(train)
    for row, total_prediction, bin_prediction in zip(
        working, total_predictions, bin_predictions
    ):
        row["total_bytes_only_predicted_us"] = float(total_prediction)
        row["three_hard_bins_predicted_us"] = float(bin_prediction)
        row["continuous_calibration_scale"] = calibration[row["phase"]]
        row["continuous_calibrated_us"] = (
            row["continuous_raw_us"] * calibration[row["phase"]]
        )
        row["continuous_histogram_predicted_us"] = row[
            "continuous_calibrated_us"
        ]
    train = [row for row in working if row["workload_id"] in train_ids]
    validation = [
        row for row in working if row["workload_id"] in validation_ids
    ]
    dnn_model, dnn_mean, dnn_scale, training = train_residual_mlp(
        train, validation, seed, max_epochs, patience
    )
    dnn_predictions = predict_residual_mlp(
        dnn_model, dnn_mean, dnn_scale, working
    )
    predictions = {}
    for row, prediction in zip(working, dnn_predictions):
        row["continuous_histogram_dnn_residual_predicted_us"] = float(
            prediction
        )
        predictions[row["workload_id"]] = {
            model_name: row[f"{model_name}_predicted_us"]
            for model_name in MODEL_NAMES
        }
        predictions[row["workload_id"]]["continuous_calibration_scale"] = row[
            "continuous_calibration_scale"
        ]
    return {
        "predictions": predictions,
        "total_model": total_model,
        "bin_model": bin_model,
        "calibration": calibration,
        "dnn_model": dnn_model,
        "dnn_mean": dnn_mean,
        "dnn_scale": dnn_scale,
        "training": training,
    }


def apply_prediction_bundle(rows, bundle):
    for row in rows:
        values = bundle["predictions"][row["workload_id"]]
        for model_name in MODEL_NAMES:
            row[f"{model_name}_predicted_us"] = float(values[model_name])
            row[f"{model_name}_ape"] = abs(
                float(values[model_name]) - row["target_post_us"]
            ) / row["target_post_us"]
        row["continuous_calibration_scale"] = values[
            "continuous_calibration_scale"
        ]
        row["continuous_calibrated_us"] = row[
            "continuous_histogram_predicted_us"
        ]


def metric_row(scope, model_name, rows):
    if not rows:
        return {
            "scope": scope,
            "model": model_name,
            "samples": 0,
            "mape": None,
            "median_ape": None,
            "p95_ape": None,
            "mae_us": None,
            "rmse_us": None,
            "r2": None,
        }
    targets = np.asarray([row["target_post_us"] for row in rows])
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
        "test_stable_iqr_le_20pct": [
            row for row in test if row["post_repeat_iqr_fraction"] <= 0.20
        ],
        "test_prefill": [row for row in test if row["phase"] == "prefill"],
        "test_decode": [row for row in test if row["phase"] == "decode"],
        "test_equal_payload": [
            row for row in test if row["controlled_equal_payload"]
        ],
    }
    for model in MODEL_ORDER:
        scopes[f"test_{model}"] = [row for row in test if row["model"] == model]
        for phase in ("prefill", "decode"):
            scopes[f"test_{model}_{phase}"] = [
                row
                for row in test
                if row["model"] == model and row["phase"] == phase
            ]
    return [
        metric_row(scope, model_name, scoped_rows)
        for scope, scoped_rows in scopes.items()
        for model_name in MODEL_NAMES
    ]


def equal_payload_pairs(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["mode"] == "mixed_same_coarse":
            key = ("mixed_decode", row["model"])
        else:
            key = (
                "chunked_prefill",
                row["model"],
                row["batch_size"],
                row["input_len"],
            )
        groups[key].append(row)
    output = []
    for key, candidates in sorted(groups.items(), key=lambda item: str(item[0])):
        for left, right in itertools.combinations(
            sorted(candidates, key=lambda row: row["workload_id"]), 2
        ):
            if left["logical_payload_bytes"] != right["logical_payload_bytes"]:
                continue
            if (
                left["calls_by_op_payload_json"]
                == right["calls_by_op_payload_json"]
            ):
                continue
            row = {
                "comparison": key[0],
                "model": left["model"],
                "batch_size": left["batch_size"],
                "input_len": left["input_len"],
                "logical_payload_bytes": left["logical_payload_bytes"],
                "left_workload_id": left["workload_id"],
                "right_workload_id": right["workload_id"],
                "left_split": left["split"],
                "right_split": right["split"],
                "both_test": left["split"] == right["split"] == "test",
                "left_case": left["case_label"],
                "right_case": right["case_label"],
                "left_calls": left["calls"],
                "right_calls": right["calls"],
                "left_payload_supports": left["payload_supports"],
                "right_payload_supports": right["payload_supports"],
                "left_op_payload_histogram_json": left[
                    "calls_by_op_payload_json"
                ],
                "right_op_payload_histogram_json": right[
                    "calls_by_op_payload_json"
                ],
                "left_post_us": left["target_post_us"],
                "right_post_us": right["target_post_us"],
                "measured_time_ratio_max_over_min": max(
                    left["target_post_us"], right["target_post_us"]
                )
                / min(left["target_post_us"], right["target_post_us"]),
                "measured_absolute_difference_us": abs(
                    left["target_post_us"] - right["target_post_us"]
                ),
            }
            for model_name in MODEL_NAMES:
                left_prediction = left[f"{model_name}_predicted_us"]
                right_prediction = right[f"{model_name}_predicted_us"]
                row[f"{model_name}_left_us"] = left_prediction
                row[f"{model_name}_right_us"] = right_prediction
                row[f"{model_name}_time_ratio_max_over_min"] = max(
                    left_prediction, right_prediction
                ) / min(left_prediction, right_prediction)
            output.append(row)
    expected = 15 * len(MODEL_ORDER)
    if len(output) != expected:
        raise ValueError(
            f"expected {expected} equal-payload pairs, got {len(output)}"
        )
    return output


def boundary_comparisons(rows):
    by_key = {
        (
            row["model"],
            row["prefill_chunk_size"],
            row["batch_size"],
            row["input_len"],
        ): row
        for row in rows
        if row["mode"] == "chunked_prefill"
    }
    output = []
    for model in MODEL_ORDER:
        for chunk_size in (1024, 2048, 4096):
            for batch_size in (1, 4):
                for boundary in (chunk_size, 2 * chunk_size):
                    triplet = [
                        by_key[(model, chunk_size, batch_size, input_len)]
                        for input_len in (boundary - 1, boundary, boundary + 1)
                    ]
                    below, at, above = triplet
                    output.append(
                        {
                            "model": model,
                            "chunk_size": chunk_size,
                            "batch_size": batch_size,
                            "boundary_input_len": boundary,
                            "below_calls": below["calls"],
                            "at_calls": at["calls"],
                            "above_calls": above["calls"],
                            "above_over_at_calls": above["calls"] / at["calls"],
                            "below_post_us": below["target_post_us"],
                            "at_post_us": at["target_post_us"],
                            "above_post_us": above["target_post_us"],
                            "above_over_at_post_time": (
                                above["target_post_us"] / at["target_post_us"]
                            ),
                            "above_minus_at_post_us": (
                                above["target_post_us"] - at["target_post_us"]
                            ),
                            "below_workload_id": below["workload_id"],
                            "at_workload_id": at["workload_id"],
                            "above_workload_id": above["workload_id"],
                        }
                    )
    expected = 12 * len(MODEL_ORDER)
    if len(output) != expected:
        raise ValueError(
            f"expected {expected} boundary comparisons, got {len(output)}"
        )
    return output


def holdout_source_split(rows, held_model, seed):
    source = [row for row in rows if row["model"] != held_model]
    strata = defaultdict(list)
    for row in source:
        strata[(row["model"], row["mode"])].append(row)
    train_ids = set()
    validation_ids = set()
    for (_, mode), values in strata.items():
        values.sort(
            key=lambda row: stable_hash(
                (held_model, row["workload_id"]), seed
            )
        )
        validation_count = 1 if mode == "mixed_same_coarse" else 7
        validation_ids.update(
            row["workload_id"] for row in values[:validation_count]
        )
        train_ids.update(
            row["workload_id"] for row in values[validation_count:]
        )
    return train_ids, validation_ids


def evaluate_model_holdout(rows, args):
    metric_rows = []
    prediction_rows = []
    summaries = {}
    for held_index, held_model in enumerate(MODEL_ORDER):
        train_ids, validation_ids = holdout_source_split(
            rows, held_model, args.seed
        )
        bundle = fit_and_predict(
            rows,
            train_ids,
            validation_ids,
            args.seed + 100 + held_index,
            args.max_epochs,
            args.patience,
        )
        held_rows = [row for row in rows if row["model"] == held_model]
        predicted_rows = []
        for row in held_rows:
            candidate = dict(row)
            values = bundle["predictions"][row["workload_id"]]
            for model_name in MODEL_NAMES:
                candidate[f"{model_name}_predicted_us"] = values[model_name]
            predicted_rows.append(candidate)
            prediction_rows.append(
                {
                    "held_out_model": held_model,
                    "workload_id": row["workload_id"],
                    "phase": row["phase"],
                    "controlled_equal_payload": row[
                        "controlled_equal_payload"
                    ],
                    "target_post_us": row["target_post_us"],
                    **{
                        f"{model_name}_predicted_us": values[model_name]
                        for model_name in MODEL_NAMES
                    },
                }
            )
        scopes = {
            f"holdout_{held_model}_all": predicted_rows,
            f"holdout_{held_model}_prefill": [
                row for row in predicted_rows if row["phase"] == "prefill"
            ],
            f"holdout_{held_model}_decode": [
                row for row in predicted_rows if row["phase"] == "decode"
            ],
            f"holdout_{held_model}_equal_payload": [
                row
                for row in predicted_rows
                if row["controlled_equal_payload"]
            ],
        }
        metric_rows.extend(
            metric_row(scope, model_name, scoped_rows)
            for scope, scoped_rows in scopes.items()
            for model_name in MODEL_NAMES
        )
        summaries[held_model] = {
            "source_models": [
                model for model in MODEL_ORDER if model != held_model
            ],
            "train_configurations": len(train_ids),
            "validation_configurations": len(validation_ids),
            "test_configurations": len(held_rows),
            "continuous_calibration": bundle["calibration"],
            "dnn_training": bundle["training"],
        }
    return metric_rows, prediction_rows, summaries


def reliability_summary(rows, field):
    values = [row[field] for row in rows]
    return {
        "median_iqr_fraction": float(np.median(values)),
        "p95_iqr_fraction": percentile(values, 95),
        "configurations_above_20pct_iqr": sum(value > 0.20 for value in values),
    }


def plot_results(path, rows, metrics, pairs, boundaries):
    test = [row for row in rows if row["split"] == "test"]
    figure, axes = plt.subplots(2, 2, figsize=(15, 11))

    scope_names = ("test_all", "test_prefill", "test_decode")
    x = np.arange(len(scope_names))
    width = 0.19
    for index, model_name in enumerate(MODEL_NAMES):
        values = [
            100
            * next(
                row["mape"]
                for row in metrics
                if row["scope"] == scope and row["model"] == model_name
            )
            for scope in scope_names
        ]
        axes[0, 0].bar(
            x + (index - 1.5) * width,
            values,
            width,
            label=MODEL_LABELS[model_name],
            color=MODEL_COLORS[model_name],
        )
    axes[0, 0].set_xticks(x, ("All", "Prefill", "Decode"))
    axes[0, 0].set_ylabel("MAPE (%)")
    axes[0, 0].set_title("Configuration-held-out prediction error")
    axes[0, 0].grid(True, axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    targets = np.asarray([row["target_post_us"] for row in test])
    minimum = float(np.min(targets))
    maximum = float(np.max(targets))
    axes[0, 1].plot(
        [minimum, maximum],
        [minimum, maximum],
        color="black",
        linestyle="--",
        linewidth=1,
    )
    for model_name in MODEL_NAMES:
        axes[0, 1].scatter(
            targets,
            [row[f"{model_name}_predicted_us"] for row in test],
            s=32,
            alpha=0.72,
            color=MODEL_COLORS[model_name],
            label=MODEL_LABELS[model_name],
        )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("Measured post-rendezvous time (μs)")
    axes[0, 1].set_ylabel("Predicted time (μs)")
    axes[0, 1].set_title("Prediction versus all-rank timing target")
    axes[0, 1].grid(True, which="both", alpha=0.25)
    axes[0, 1].legend(fontsize=8)

    for comparison, marker, color in (
        ("mixed_decode", "o", "#E45756"),
        ("chunked_prefill", "s", "#4C78A8"),
    ):
        selected = [row for row in pairs if row["comparison"] == comparison]
        axes[1, 0].scatter(
            [
                max(row["left_calls"], row["right_calls"])
                / min(row["left_calls"], row["right_calls"])
                for row in selected
            ],
            [row["measured_time_ratio_max_over_min"] for row in selected],
            marker=marker,
            s=48,
            alpha=0.75,
            color=color,
            label=comparison.replace("_", " "),
        )
    axes[1, 0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_xlabel("Calls ratio for equal total payload")
    axes[1, 0].set_ylabel("Measured time ratio (max/min)")
    axes[1, 0].set_title("Equal bytes, different message structure")
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].legend()

    for model in MODEL_ORDER:
        marker, color = MODEL_STYLES[model]
        selected = [row for row in boundaries if row["model"] == model]
        axes[1, 1].scatter(
            [row["above_over_at_calls"] for row in selected],
            [row["above_over_at_post_time"] for row in selected],
            marker=marker,
            s=48,
            alpha=0.75,
            color=color,
            label=model,
        )
    axes[1, 1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_xlabel("Calls ratio immediately above chunk boundary")
    axes[1, 1].set_ylabel("Post-rendezvous time ratio")
    axes[1, 1].set_title("Discrete chunk-boundary timing response")
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].legend()

    phase_label = "Phase 11/13" if len(MODEL_ORDER) > 2 else "Phase 11"
    figure.suptitle(
        f"{phase_label}: multiscale all-rank communication timing"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    global MODEL_ORDER
    args = parse_args()
    torch.set_num_threads(1)
    loaded_rows, MODEL_ORDER = load_dataset(args.input_dir)
    rows = assign_splits(loaded_rows, args.seed)
    curve = BackendAwareCostCurve(
        args.custom_curve,
        args.nccl_curve,
        custom_latency_column="completion_median_latency_us",
        nccl_latency_column="median_latency_us",
    )
    for row in rows:
        row["continuous_raw_us"] = continuous_cost(row, curve)

    train_ids = {
        row["workload_id"] for row in rows if row["split"] == "train"
    }
    validation_ids = {
        row["workload_id"] for row in rows if row["split"] == "validation"
    }
    bundle = fit_and_predict(
        rows,
        train_ids,
        validation_ids,
        args.seed,
        args.max_epochs,
        args.patience,
    )
    apply_prediction_bundle(rows, bundle)
    metrics = evaluate_metrics(rows)
    pairs = equal_payload_pairs(rows)
    boundaries = boundary_comparisons(rows)
    holdout_metrics, holdout_predictions, holdout_summary = evaluate_model_holdout(
        rows, args
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "predictions.csv",
        [public_row(row) for row in sorted(rows, key=lambda row: row["workload_id"])],
    )
    write_csv(
        args.output_dir / "split_assignments.csv",
        [
            {
                "workload_id": row["workload_id"],
                "model": row["model"],
                "mode": row["mode"],
                "case_label": row["case_label"],
                "batch_size": row["batch_size"],
                "input_len": row["input_len"],
                "prefill_chunk_size": row["prefill_chunk_size"],
                "split": row["split"],
            }
            for row in sorted(rows, key=lambda row: row["workload_id"])
        ],
    )
    write_csv(args.output_dir / "metrics.csv", metrics)
    write_csv(args.output_dir / "equal_payload_comparison.csv", pairs)
    write_csv(args.output_dir / "boundary_comparison.csv", boundaries)
    write_csv(args.output_dir / "model_holdout_metrics.csv", holdout_metrics)
    write_csv(
        args.output_dir / "model_holdout_predictions.csv", holdout_predictions
    )
    plot_results(
        args.output_dir / "multiscale_timing_analysis.png",
        rows,
        metrics,
        pairs,
        boundaries,
    )
    torch.save(
        {
            "state_dict": bundle["dnn_model"].state_dict(),
            "feature_mean": bundle["dnn_mean"],
            "feature_scale": bundle["dnn_scale"],
            "seed": args.seed,
        },
        args.output_dir / "dnn_residual_model.pt",
    )

    summary = {
        "schema_version": (
            "multiscale-timing-analysis-v2"
            if len(MODEL_ORDER) > 2
            else "multiscale-timing-analysis-v1"
        ),
        "dataset": {
            "raw_label_rows": len(rows) * 3,
            "aggregated_configurations": len(rows),
            "models": {
                model: sum(row["model"] == model for row in rows)
                for model in MODEL_ORDER
            },
            "modes": {
                mode: sum(row["mode"] == mode for row in rows)
                for mode in ("mixed_same_coarse", "chunked_prefill")
            },
            "split_counts": {
                split: sum(row["split"] == split for row in rows)
                for split in ("train", "validation", "test")
            },
            "split_unit": (
                "complete workload/profile/chunk configuration; three repeats "
                "are aggregated before deterministic stratified splitting"
            ),
            "target": (
                "median across three repeats of all-rank post-rendezvous "
                "completion time: sum over aligned collectives of "
                "max(end)-max(start) across TP ranks"
            ),
            "reliability": {
                "post_rendezvous": reliability_summary(
                    rows, "post_repeat_iqr_fraction"
                ),
                "intrinsic": reliability_summary(
                    rows, "intrinsic_repeat_iqr_fraction"
                ),
                "sync_inclusive": reliability_summary(
                    rows, "sync_repeat_iqr_fraction"
                ),
            },
        },
        "models": {
            "total_bytes_only": bundle["total_model"].serialize(),
            "three_hard_bins": bundle["bin_model"].serialize(),
            "continuous_histogram": {
                "curve": "Phase 2 B200 L1 backend-aware continuous curve",
                "custom_latency_column": "completion_median_latency_us",
                "nccl_latency_column": "median_latency_us",
                "calibration_by_phase": bundle["calibration"],
            },
            "dnn_residual": {
                "architecture": "MLP(input,32,16,1), ReLU",
                "features": (
                    "model-neutral workload, exact histogram statistics, bin "
                    "statistics, and calibrated structural estimate"
                ),
                "target": "log(measured_post / calibrated_continuous_histogram)",
                "training": bundle["training"],
            },
        },
        "metrics": metrics,
        "controlled_comparisons": {
            "equal_payload_pair_count": len(pairs),
            "mixed_equal_payload_pairs": sum(
                row["comparison"] == "mixed_decode" for row in pairs
            ),
            "chunked_equal_payload_pairs": sum(
                row["comparison"] == "chunked_prefill" for row in pairs
            ),
            "chunk_boundary_comparisons": len(boundaries),
        },
        "model_holdout": {
            "metrics": holdout_metrics,
            "training": holdout_summary,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )

    print(
        f"analyzed {len(rows)} configurations: "
        f"train={len(train_ids)} validation={len(validation_ids)} "
        f"test={sum(row['split'] == 'test' for row in rows)}"
    )
    for row in metrics:
        if row["scope"] == "test_all":
            print(
                f"{row['model']}: MAPE={100 * row['mape']:.3f}% "
                f"P95={100 * row['p95_ape']:.3f}% R2={row['r2']:.5f}"
            )
    for model in MODEL_ORDER:
        selected = [
            row
            for row in rows
            if row["model"] == model and row["mode"] == "mixed_same_coarse"
        ]
        print(
            f"{model} mixed post-rendezvous medians: "
            + ", ".join(
                f"{row['case_label']}={row['target_post_us']:.3f}us"
                for row in sorted(selected, key=lambda row: row["case_label"])
            )
        )


if __name__ == "__main__":
    main()
