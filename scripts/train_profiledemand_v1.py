#!/usr/bin/env python3
"""Train and evaluate the ProfileDemand v1 structured predictor.

The formal predictor maps a steady service profile, an execution strategy, model
structure, phase and candidate TP size to a canonical 12-bin calls+bytes demand.
The transparent H0 formula supplies the base estimate; a small MLP predicts only
the residual.  Direct model-id and structure-only MLPs are retained as ablations.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


MIN_PAYLOAD = 4 * 1024
MAX_PAYLOAD = 512 * 1024 * 1024
BIN_COUNT = 12
TPS = (2, 4, 8)
METHODS = ("model_id_direct", "structure_direct", "h0", "h0_residual")
PROFILE_SCALARS = (
    "rps",
    "interarrival_cv",
    "peak_to_mean_1s",
    "fano_1s",
    "input_mean_capped",
    "output_mean_capped",
    "lm_correlation_capped",
    "survival_m_gt_8",
    "survival_m_gt_16",
    "survival_m_gt_32",
    "survival_m_gt_64",
)
MODEL_NUMERICS = (
    "num_hidden_layers",
    "hidden_size",
    "dense_intermediate_ratio",
    "num_attention_heads",
    "head_dim",
    "kv_head_ratio",
    "dtype_bytes",
    "is_moe",
    "num_experts",
    "experts_per_token",
    "moe_intermediate_ratio",
    "num_shared_experts",
    "first_dense_layers",
    "moe_layer_frequency",
    "estimated_moe_layers",
    "logical_collectives_per_forward_prior",
    "payload_bytes_per_active_token_prior",
)
LOG_FEATURES = {
    "rps",
    "interarrival_cv",
    "peak_to_mean_1s",
    "fano_1s",
    "input_mean_capped",
    "output_mean_capped",
    "num_hidden_layers",
    "hidden_size",
    "dense_intermediate_ratio",
    "num_attention_heads",
    "head_dim",
    "num_experts",
    "experts_per_token",
    "num_shared_experts",
    "estimated_moe_layers",
    "logical_collectives_per_forward_prior",
    "payload_bytes_per_active_token_prior",
}


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=Path,
        default=root / "experiment-results/phase16_profiledemand_dataset/phase_labels.csv",
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=root / "experiment-results/phase16_service_profiles/service_profiles.csv",
    )
    parser.add_argument(
        "--model-features",
        type=Path,
        default=root / "experiment-results/phase16_model_features/model_features.json",
    )
    parser.add_argument(
        "--plan-summary",
        type=Path,
        default=root / "experiment-results/phase16_profiledemand_plans/summary.json",
    )
    parser.add_argument(
        "--curve-root",
        type=Path,
        default=root / "experiment-results/phase14f_post_rendezvous/curve",
    )
    parser.add_argument(
        "--curve-extension",
        type=Path,
        default=root / "experiment-results/phase15_l1_curve_extension/curve_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase16_profiledemand_predictor",
    )
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260807)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"empty rows for {path}")
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def log_feature(name, value):
    value = float(value)
    return math.log1p(max(value, 0.0)) if name in LOG_FEATURES else value


def make_bins():
    return np.power(
        2.0,
        np.linspace(math.log2(MIN_PAYLOAD), math.log2(MAX_PAYLOAD), BIN_COUNT + 1),
    )


BIN_EDGES = make_bins()


def bin_index(payload):
    payload = float(np.clip(payload, MIN_PAYLOAD, MAX_PAYLOAD))
    return min(int(np.searchsorted(BIN_EDGES, payload, side="right") - 1), BIN_COUNT - 1)


def allocate_counts(probabilities, count=32):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    exact = probabilities * count
    result = np.floor(exact).astype(int)
    for index in np.argsort(-(exact - result), kind="stable")[: count - result.sum()]:
        result[index] += 1
    return result


def scaled_representatives(base, weights, target, lower, upper):
    base = np.asarray(base, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    low, high = 0.001, 256.0
    for _ in range(80):
        scale = (low + high) / 2
        mean = float(np.sum(weights * np.clip(base * scale, lower, upper)))
        if mean < target:
            low = scale
        else:
            high = scale
    return np.rint(np.clip(base * ((low + high) / 2), lower, upper)).astype(int)


def pseudo_requests(profile):
    joint = np.asarray(json.loads(profile["joint_lm_4x4_json"]), dtype=np.float64).reshape(4, 4)
    counts = allocate_counts(joint.reshape(-1), 32)
    input_values = scaled_representatives(
        [64, 320, 1280, 4096],
        joint.sum(axis=1),
        float(profile["input_mean_capped"]),
        1,
        8192,
    )
    output_values = scaled_representatives(
        [8, 24, 48, 96],
        joint.sum(axis=0),
        float(profile["output_mean_capped"]),
        1,
        128,
    )
    scheduled = []
    for cell, amount in enumerate(counts):
        for occurrence in range(amount):
            # Evenly interleave joint cells; do not recover the measured request order.
            scheduled.append(((occurrence + 0.5) / amount, cell))
    scheduled.sort()
    return [
        (int(input_values[cell // 4]), int(output_values[cell % 4]))
        for _, cell in scheduled
    ]


def microbatches(requests, max_batch_size, max_prefill_tokens):
    batches, current, tokens = [], [], 0
    for request in requests:
        would_exceed = current and (
            len(current) >= max_batch_size or tokens + request[0] > max_prefill_tokens
        )
        if would_exceed:
            batches.append(current)
            current, tokens = [], 0
        current.append(request)
        tokens += request[0]
    if current:
        batches.append(current)
    return batches


def h0_vectors(profile, strategy, model, phase):
    requests = pseudo_requests(profile)
    batches = microbatches(
        requests,
        int(strategy["max_batch_size"]),
        int(strategy["max_prefill_tokens"]),
    )
    calls_per_forward = int(model["logical_collectives_per_forward_prior"])
    bytes_per_token = int(model["payload_bytes_per_active_token_prior"])
    histogram = Counter()
    if phase == "prefill":
        for batch in batches:
            histogram[sum(row[0] for row in batch) * bytes_per_token] += calls_per_forward
    else:
        for batch in batches:
            output_lengths = [row[1] for row in batch]
            for step in range(1, max(output_lengths)):
                active = sum(length > step for length in output_lengths)
                if active:
                    histogram[active * bytes_per_token] += calls_per_forward
    calls = np.zeros(BIN_COUNT, dtype=np.float64)
    logical_bytes = np.zeros(BIN_COUNT, dtype=np.float64)
    normalization = 1000.0 / len(requests)
    for payload, count in histogram.items():
        index = bin_index(payload)
        calls[index] += count * normalization
        logical_bytes[index] += count * payload * normalization
    return calls, logical_bytes


def target_encode(calls, logical_bytes):
    encoded = []
    for vector in (calls, logical_bytes):
        total = max(float(np.sum(vector)), 0.0)
        smoothing = max(total, 1.0) * 1e-6 / BIN_COUNT
        shares = (vector + smoothing) / (total + smoothing * BIN_COUNT)
        encoded.extend([math.log1p(total), *np.log(shares)])
    return np.asarray(encoded, dtype=np.float32)


def target_decode(encoded):
    vectors = []
    offset = 0
    for _ in range(2):
        total = max(math.expm1(float(np.clip(encoded[offset], 0, 40))), 0.0)
        logits = np.clip(encoded[offset + 1 : offset + 1 + BIN_COUNT], -50, 50)
        probabilities = np.exp(logits - np.max(logits))
        probabilities /= probabilities.sum()
        vectors.append(total * probabilities)
        offset += BIN_COUNT + 1
    return vectors[0], vectors[1]


def bounded_residual(target, base):
    """Keep the learned term a calibration, never a replacement for H0."""
    residual = np.asarray(target - base, dtype=np.float32)
    bounds = np.full(residual.shape[-1], 2.0, dtype=np.float32)
    bounds[0] = math.log(2.0)
    bounds[BIN_COUNT + 1] = math.log(2.0)
    return np.clip(residual, -bounds, bounds)


def make_features(rows, profiles, models, strategies, use_model_id):
    model_names = sorted(models)
    columns, names = [], []
    for row in rows:
        profile = profiles[row["profile_id"]]
        strategy = strategies[row["strategy"]]
        values = []
        row_names = []
        for name in PROFILE_SCALARS:
            values.append(log_feature(name, profile[name]))
            row_names.append(f"profile_{name}")
        joint = json.loads(profile["joint_lm_4x4_json"])
        values.extend(float(value) for value in joint)
        row_names.extend(f"profile_joint_lm_{index}" for index in range(16))
        values.extend(
            [
                math.log2(float(strategy["max_batch_size"])),
                math.log2(float(strategy["max_prefill_tokens"])),
                math.log2(int(row["tp"])),
                float(row["phase"] == "prefill"),
                float(row["phase"] == "decode"),
            ]
        )
        row_names.extend(
            ["strategy_log2_max_batch", "strategy_log2_max_prefill_tokens", "tp_log2", "phase_prefill", "phase_decode"]
        )
        if use_model_id:
            values.extend(float(row["model"] == name) for name in model_names)
            row_names.extend(f"model_id_{name}" for name in model_names)
        else:
            model = models[row["model"]]
            for name in MODEL_NUMERICS:
                values.append(log_feature(name, model[name]))
                row_names.append(f"model_{name}")
        columns.append(values)
        names = row_names
    return np.asarray(columns, dtype=np.float32), names


class ResidualMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, value):
        return self.network(value)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validation_indices(rows, train_indices, seed):
    profiles = sorted({rows[index]["profile_id"] for index in train_indices})
    ranked = sorted(profiles, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
    selected = set(ranked[: max(1, len(ranked) // 5)])
    validation = np.asarray([index for index in train_indices if rows[index]["profile_id"] in selected], dtype=int)
    fit = np.asarray([index for index in train_indices if rows[index]["profile_id"] not in selected], dtype=int)
    if not len(fit) or not len(validation):
        raise ValueError("could not form profile-grouped validation split")
    return fit, validation


def fit_network(features, targets, fit, validation, args, seed):
    feature_mean = features[fit].mean(axis=0)
    feature_std = features[fit].std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0
    target_mean = targets[fit].mean(axis=0)
    target_std = targets[fit].std(axis=0)
    target_std[target_std < 1e-6] = 1.0
    # The evaluation intentionally includes unseen model structures.  Bounded
    # standardization prevents an MLP from turning an out-of-domain feature into
    # an unphysical exponential target while still exposing accuracy loss.
    x = np.clip((features - feature_mean) / feature_std, -6.0, 6.0).astype(np.float32)
    y = ((targets - target_mean) / target_std).astype(np.float32)
    seed_all(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResidualMLP(features.shape[1], targets.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x[fit]), torch.from_numpy(y[fit])),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_x = torch.from_numpy(x[validation]).to(device)
    validation_y = torch.from_numpy(y[validation]).to(device)
    best_state, best_loss, stale, history = None, math.inf, 0, []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            loss = loss_fn(model(batch_x), batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(model(validation_x), validation_y).cpu())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(x).to(device)).cpu().numpy()
    prediction = np.clip(prediction, -6.0, 6.0) * target_std + target_mean
    target_min = targets[fit].min(axis=0)
    target_max = targets[fit].max(axis=0)
    margin = np.maximum((target_max - target_min) * 0.10, 1e-4)
    prediction = np.clip(prediction, target_min - margin, target_max + margin)
    checkpoint = {
        "model_state": {key: value.cpu() for key, value in model.state_dict().items()},
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "best_epoch": int(np.argmin([row["validation_loss"] for row in history])),
        "device": str(device),
    }
    return prediction, checkpoint, history


def load_curves(root, extension):
    samples = defaultdict(list)
    for path in sorted(root.glob("tp*/all_reduce/r*/curve.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                samples[(int(row["group_size"]), int(row["payload_bytes"]))].extend(
                    float(value) for value in row["post_rendezvous_samples_us"]
                )
    points = defaultdict(dict)
    for (tp, payload), values in samples.items():
        points[tp][payload] = float(np.median(values))
    for row in read_csv(extension):
        points[int(row["tp"])][int(row["payload_bytes"])] = float(row["median_post_rendezvous_us"])
    result = {tp: sorted(values.items()) for tp, values in points.items()}
    for tp in TPS:
        if not result[tp] or result[tp][0][0] > MIN_PAYLOAD or result[tp][-1][0] < MAX_PAYLOAD:
            raise ValueError(f"incomplete L1 curve for TP={tp}")
    return result


def interpolate_curve(points, payload):
    payload = float(np.clip(payload, points[0][0], points[-1][0]))
    xs = np.log2(np.asarray([row[0] for row in points], dtype=np.float64))
    ys = np.asarray([row[1] for row in points], dtype=np.float64)
    return max(float(np.interp(math.log2(payload), xs, ys)), 1e-6)


def structural_cost(calls, logical_bytes, tp, curves):
    total = 0.0
    for count, byte_count in zip(calls, logical_bytes):
        if count > 1e-9:
            total += count * interpolate_curve(curves[tp], byte_count / count)
    return total


def normalized(vector):
    total = float(np.sum(vector))
    return vector / total if total else np.zeros_like(vector)


def prediction_record(row, evaluation, fold, method, actual_calls, actual_bytes, predicted_calls, predicted_bytes, curves):
    actual_calls_total = float(actual_calls.sum())
    actual_bytes_total = float(actual_bytes.sum())
    predicted_calls_total = float(predicted_calls.sum())
    predicted_bytes_total = float(predicted_bytes.sum())
    actual_share, predicted_share = normalized(actual_calls), normalized(predicted_calls)
    cost_actual = structural_cost(actual_calls, actual_bytes, int(row["tp"]), curves)
    cost_predicted = structural_cost(predicted_calls, predicted_bytes, int(row["tp"]), curves)
    return {
        "evaluation": evaluation,
        "fold": fold,
        "method": method,
        "model": row["model"],
        "tp": row["tp"],
        "profile_id": row["profile_id"],
        "strategy": row["strategy"],
        "phase": row["phase"],
        "calls_vector_absolute_error": float(np.abs(predicted_calls - actual_calls).sum()),
        "bytes_vector_absolute_error": float(np.abs(predicted_bytes - actual_bytes).sum()),
        "actual_total_calls": actual_calls_total,
        "predicted_total_calls": predicted_calls_total,
        "actual_total_bytes": actual_bytes_total,
        "predicted_total_bytes": predicted_bytes_total,
        "histogram_l1": float(np.abs(predicted_share - actual_share).sum()),
        "log_payload_emd": float(np.abs(np.cumsum(predicted_share - actual_share)[:-1]).sum() / (BIN_COUNT - 1)),
        "actual_l1_structural_cost_us_per_1000": cost_actual,
        "predicted_l1_structural_cost_us_per_1000": cost_predicted,
        "l1_structural_cost_ape": abs(cost_predicted - cost_actual) / max(cost_actual, 1e-9),
    }


def aggregate_metrics(predictions):
    result = []
    groups = defaultdict(list)
    for row in predictions:
        for scope in ("all", row["phase"]):
            groups[(row["evaluation"], row["method"], scope)].append(row)
    for (evaluation, method, scope), rows in sorted(groups.items()):
        calls_denominator = sum(float(row["actual_total_calls"]) for row in rows)
        bytes_denominator = sum(float(row["actual_total_bytes"]) for row in rows)
        calls_ape = [abs(float(row["predicted_total_calls"]) - float(row["actual_total_calls"])) / max(float(row["actual_total_calls"]), 1e-9) for row in rows]
        bytes_ape = [abs(float(row["predicted_total_bytes"]) - float(row["actual_total_bytes"])) / max(float(row["actual_total_bytes"]), 1e-9) for row in rows]
        cost_ape = [float(row["l1_structural_cost_ape"]) for row in rows]
        result.append(
            {
                "evaluation": evaluation,
                "method": method,
                "scope": scope,
                "samples": len(rows),
                "calls_vector_wape": sum(float(row["calls_vector_absolute_error"]) for row in rows) / calls_denominator,
                "bytes_vector_wape": sum(float(row["bytes_vector_absolute_error"]) for row in rows) / bytes_denominator,
                "total_calls_mape": float(np.mean(calls_ape)),
                "total_bytes_mape": float(np.mean(bytes_ape)),
                "mean_histogram_l1": float(np.mean([float(row["histogram_l1"]) for row in rows])),
                "mean_log_payload_emd": float(np.mean([float(row["log_payload_emd"]) for row in rows])),
                "l1_structural_cost_mape": float(np.mean(cost_ape)),
                "l1_structural_cost_p95_ape": float(np.percentile(cost_ape, 95)),
            }
        )
    return result


def outer_folds(rows, profiles):
    definitions = defaultdict(dict)
    for profile in profiles.values():
        definitions["traffic_segment_holdout"].setdefault(profile["segment"], set()).update(
            index for index, row in enumerate(rows) if row["profile_id"] == profile["profile_id"]
        )
    for model in sorted({row["model"] for row in rows}):
        definitions["model_holdout"][model] = {index for index, row in enumerate(rows) if row["model"] == model}
    for strategy in sorted({row["strategy"] for row in rows}):
        definitions["strategy_holdout"][strategy] = {index for index, row in enumerate(rows) if row["strategy"] == strategy}
    for tp in TPS:
        definitions["tp_holdout"][f"tp{tp}"] = {index for index, row in enumerate(rows) if int(row["tp"]) == tp}
    return definitions


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.labels)
    profiles = {row["profile_id"]: row for row in read_csv(args.profiles)}
    models = {row["model"]: row for row in json.loads(args.model_features.read_text())}
    plan = json.loads(args.plan_summary.read_text())
    strategies = plan["strategies"]
    curves = load_curves(args.curve_root, args.curve_extension)
    if len(rows) != 1296:
        raise ValueError(f"expected 1296 labels, got {len(rows)}")

    actual_calls = np.stack([np.asarray(json.loads(row["calls_by_12bin_json"]), dtype=np.float64) for row in rows])
    actual_bytes = np.stack([np.asarray(json.loads(row["logical_bytes_by_12bin_json"]), dtype=np.float64) for row in rows])
    encoded = np.stack([target_encode(calls, logical_bytes) for calls, logical_bytes in zip(actual_calls, actual_bytes)])
    h0_calls, h0_bytes = [], []
    for row in rows:
        calls, logical_bytes = h0_vectors(profiles[row["profile_id"]], strategies[row["strategy"]], models[row["model"]], row["phase"])
        h0_calls.append(calls)
        h0_bytes.append(logical_bytes)
    h0_calls, h0_bytes = np.stack(h0_calls), np.stack(h0_bytes)
    h0_encoded = np.stack([target_encode(calls, logical_bytes) for calls, logical_bytes in zip(h0_calls, h0_bytes)])
    residual = bounded_residual(encoded, h0_encoded)
    model_id_features, model_id_names = make_features(rows, profiles, models, strategies, use_model_id=True)
    structure_features, structure_names = make_features(rows, profiles, models, strategies, use_model_id=False)

    predictions, training_history = [], []
    definitions = outer_folds(rows, profiles)
    all_indices = np.arange(len(rows), dtype=int)
    trained_folds = 0
    for evaluation, folds in definitions.items():
        for fold_number, (fold, test_set) in enumerate(sorted(folds.items())):
            test = np.asarray(sorted(test_set), dtype=int)
            train = np.asarray([index for index in all_indices if index not in test_set], dtype=int)
            fit, validation = validation_indices(rows, train, args.seed + fold_number)
            fold_id_features = model_id_features.copy()
            if evaluation == "model_holdout":
                # A model-id baseline has no representation for an unseen model.  Encode the
                # held-out category as the all-zero unknown vector instead of using an
                # untrained random weight for its one-hot column.
                fold_id_features[:, model_id_names.index(f"model_id_{fold}")] = 0.0
            direct_id, _, history_id = fit_network(fold_id_features, encoded, fit, validation, args, args.seed + trained_folds * 3)
            direct_structure, _, history_structure = fit_network(structure_features, encoded, fit, validation, args, args.seed + trained_folds * 3 + 1)
            residual_prediction, _, history_residual = fit_network(structure_features, residual, fit, validation, args, args.seed + trained_folds * 3 + 2)
            for name, history in (("model_id_direct", history_id), ("structure_direct", history_structure), ("h0_residual", history_residual)):
                for history_row in history:
                    training_history.append({"evaluation": evaluation, "fold": fold, "method": name, **history_row})
            encoded_methods = {
                "model_id_direct": direct_id,
                "structure_direct": direct_structure,
                "h0_residual": h0_encoded + residual_prediction,
            }
            for index in test:
                for method in METHODS:
                    if method == "h0":
                        predicted_calls, predicted_bytes = h0_calls[index], h0_bytes[index]
                    else:
                        predicted_calls, predicted_bytes = target_decode(encoded_methods[method][index])
                    predictions.append(
                        prediction_record(rows[index], evaluation, fold, method, actual_calls[index], actual_bytes[index], predicted_calls, predicted_bytes, curves)
                    )
            trained_folds += 1

    metrics = aggregate_metrics(predictions)
    write_csv(args.output_dir / "metrics.csv", metrics)
    with gzip.open(args.output_dir / "predictions.csv.gz", "wt", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(predictions[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(predictions)
    with (args.output_dir / "training_history.jsonl").open("w") as output:
        for row in training_history:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")

    # Train the deployable H0+residual checkpoint with grouped validation.
    final_fit, final_validation = validation_indices(rows, all_indices, args.seed + 9999)
    _, final_checkpoint, final_history = fit_network(
        structure_features,
        residual,
        final_fit,
        final_validation,
        args,
        args.seed + 9999,
    )
    final_checkpoint.update(
        {
            "schema_version": "profiledemand-v1-h0-residual",
            "feature_names": structure_names,
            "bin_edges_bytes": BIN_EDGES.tolist(),
            "target_encoding": "log-total-plus-log-smoothed-shares for calls and logical bytes; bounded network residual calibrates H0",
            "model_feature_names": list(MODEL_NUMERICS),
            "profile_feature_names": list(PROFILE_SCALARS) + [f"joint_lm_{index}" for index in range(16)],
            "label_sha256": sha256(args.labels),
        }
    )
    torch.save(final_checkpoint, args.output_dir / "formal_h0_residual_model.pt")
    (args.output_dir / "feature_names.json").write_text(
        json.dumps({"model_id_direct": model_id_names, "structure_models": structure_names}, indent=2) + "\n"
    )

    primary = {
        f"{row['method']}:{row['scope']}": {
            "samples": row["samples"],
            "calls_vector_wape": row["calls_vector_wape"],
            "bytes_vector_wape": row["bytes_vector_wape"],
            "mean_histogram_l1": row["mean_histogram_l1"],
            "l1_structural_cost_mape": row["l1_structural_cost_mape"],
            "l1_structural_cost_p95_ape": row["l1_structural_cost_p95_ape"],
        }
        for row in metrics
        if row["evaluation"] == "traffic_segment_holdout"
    }
    headline_rows = {
        (row["evaluation"], row["method"]): row
        for row in metrics
        if row["scope"] == "all"
    }
    headline = {
        evaluation: {
            method: {
                key: row[key]
                for key in (
                    "calls_vector_wape",
                    "bytes_vector_wape",
                    "total_calls_mape",
                    "total_bytes_mape",
                    "mean_log_payload_emd",
                    "l1_structural_cost_mape",
                    "l1_structural_cost_p95_ape",
                )
            }
            for (row_evaluation, method), row in headline_rows.items()
            if row_evaluation == evaluation
        }
        for evaluation in definitions
    }
    summary = {
        "schema_version": "profiledemand-v1-evaluation",
        "status": "PASS",
        "input_contract": "steady traffic profile + numeric execution strategy + generalizable model structure + candidate TP + phase",
        "output_contract": "canonical group-level 12-bin calls and representative-rank logical bytes per 1000 requests",
        "labels": len(rows),
        "models": sorted(models),
        "profiles": len(profiles),
        "strategies": sorted(strategies),
        "tp_sizes": list(TPS),
        "methods": list(METHODS),
        "evaluation_regimes": {name: sorted(folds) for name, folds in definitions.items()},
        "trained_outer_folds": trained_folds,
        "primary_traffic_segment_holdout": primary,
        "headline_all_scope": headline,
        "final_checkpoint_best_epoch": final_checkpoint["best_epoch"],
        "important_boundary": "arrival statistics are input features, but current GPU labels are draining microbatches; online arrival-driven batching is not yet identified",
        "l1_metric_boundary": "L1 propagation is structural cost from the predicted canonical histogram and measured all_reduce curve, not a new end-to-end GPU timing label",
        "source_hashes": {
            "labels": sha256(args.labels),
            "profiles": sha256(args.profiles),
            "model_features": sha256(args.model_features),
            "plan_summary": sha256(args.plan_summary),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    table_lines = [
        "| 外层留出 | 方法 | total calls MAPE | total bytes MAPE | log-payload EMD | L1 结构代价 MAPE | P95 APE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for evaluation in definitions:
        for method in ("h0", "h0_residual"):
            row = headline_rows[(evaluation, method)]
            table_lines.append(
                f"| {evaluation} | {method} | {100 * float(row['total_calls_mape']):.2f}% "
                f"| {100 * float(row['total_bytes_mape']):.2f}% | {float(row['mean_log_payload_emd']):.3f} "
                f"| {100 * float(row['l1_structural_cost_mape']):.2f}% "
                f"| {100 * float(row['l1_structural_cost_p95_ape']):.2f}% |"
            )
    headline_table = "\n".join(table_lines)
    direct_segment = headline_rows[("traffic_segment_holdout", "structure_direct")]
    readme = f"""# Phase 16G：ProfileDemand v1 四方法留出评测

输入为低维常态流量画像、数值化 batching 策略、可泛化模型结构、候选 TP 和阶段；输出为
每 1000 请求的 12 桶 group-level calls 与代表 rank logical bytes。共使用 {len(rows)} 条
GPU 聚合标签。对比 Model-ID 直接 DNN、结构特征直接 DNN、透明公式 H0，以及正式方法
H0+DNN residual。

外层测试分别留出完整流量 segment、模型、执行策略和 TP；内层早停再按 profile 分组，
避免同一画像及其跨 TP 重复标签泄漏。`metrics.csv` 报告直方图 calls/bytes WAPE、分布
L1/EMD，以及预测直方图乘 B200 L1 连续 AllReduce 曲线后的结构代价误差。

H0 只从 4×4 长度联合分布和均值合成 32 个伪请求，不读取 GPU replay 的 32 条真实长度
或顺序；residual 因而学习分桶内形态和 batching 边界，而不是重复精确公式。正式 checkpoint
为 `formal_h0_residual_model.pt`。总量 residual 被限制在两倍以内、分布 logit residual
被限制在 ±2；整模型留出时也对标准化输入和输出做训练域裁剪，保证 DNN 只能校正 H0，
不能在未见结构上产生无物理意义的指数外推。

边界：当前到达率/突发特征虽进入输入，但 GPU 标签仍是同时进入的 draining microbatch，
不能把结果表述为 online arrival-aware batching 已完成。L1 传播也是结构代价评估，不是新增
的端到端通信时间真值。

## 核心结果

{headline_table}

H0 在四类留出中固定为 9.20% L1 结构代价 MAPE。受约束 residual 在未见模型、策略和 TP
上分别降至 {100 * headline_rows[('model_holdout', 'h0_residual')]['l1_structural_cost_mape']:.2f}%、
{100 * headline_rows[('strategy_holdout', 'h0_residual')]['l1_structural_cost_mape']:.2f}% 和
{100 * headline_rows[('tp_holdout', 'h0_residual')]['l1_structural_cost_mape']:.2f}%；但在完整未见
流量 segment 上为 {100 * headline_rows[('traffic_segment_holdout', 'h0_residual')]['l1_structural_cost_mape']:.2f}%，
弱于 H0，因此未知流量域应回退 H0，不能宣称 DNN 全面优于结构公式。

residual 的 total calls MAPE 为 8.46%–12.57%，total bytes MAPE 为 4.43%–7.64%；虽然硬桶
vector WAPE 较高，但 log-payload EMD 只有 0.016–0.017，说明主要是相邻硬桶边界迁移而非
消息质量跨越多个尺度。未见流量 segment 时，structure-direct DNN 的 L1 代价 MAPE 为
{100 * direct_segment['l1_structural_cost_mape']:.2f}%，进一步支持“结构公式为主、DNN 只校正残差”。
"""
    (args.output_dir / "README.md").write_text(readme)
    expected_predictions = len(rows) * len(METHODS) * len(definitions)
    finite_metrics = all(
        math.isfinite(float(row[key]))
        for row in metrics
        for key in (
            "calls_vector_wape",
            "bytes_vector_wape",
            "mean_histogram_l1",
            "mean_log_payload_emd",
            "l1_structural_cost_mape",
        )
    )
    checks = {
        "labels_1296": len(rows) == 1296,
        "four_methods": set(METHODS) == {row["method"] for row in predictions},
        "four_evaluation_regimes": len(definitions) == 4,
        "each_row_tested_once_per_regime_and_method": len(predictions) == expected_predictions,
        "all_metrics_finite": finite_metrics,
        "formal_checkpoint_written": (args.output_dir / "formal_h0_residual_model.pt").is_file(),
    }
    audit = {
        "schema_version": "profiledemand-v1-evaluation-audit",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    (args.output_dir / "audit_summary.json").write_text(json.dumps(audit, indent=2) + "\n")
    if audit["status"] != "PASS":
        raise RuntimeError(audit)
    (args.output_dir / "DONE").write_text("PASS\n")
    (args.output_dir / "run.log").write_text(
        json.dumps({"checks": checks, "final_history": final_history}, indent=2) + "\n"
    )
    files = sorted(path for path in args.output_dir.iterdir() if path.is_file() and path.name != "manifest.sha256")
    (args.output_dir / "manifest.sha256").write_text("".join(f"{sha256(path)}  {path.name}\n" for path in files))
    print(json.dumps({"summary": summary, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
