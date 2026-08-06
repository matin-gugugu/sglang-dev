#!/usr/bin/env python3
"""Train a history-profile -> PatternDemand pilot and evaluate L1 propagation."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


CALLS_PER_FORWARD = 73
BYTES_PER_TOKEN = 8192
MAX_BATCH = 8
MAX_DECODE_STEPS = 127
TPS = (2, 4, 8)
SCOPES = ("validation", "test", "temporal_test", "external_test")

SCALAR_FEATURES = (
    "history_count",
    "history_rps",
    "history_interarrival_cv",
    "history_peak_to_mean_1s",
    "history_fano_1s",
    "history_input_mean",
    "history_input_p50",
    "history_input_p90",
    "history_input_p99",
    "history_output_mean",
    "history_output_p50",
    "history_output_p90",
    "history_output_p99",
    "history_lm_correlation",
)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root
        / "experiment-results/phase15_pattern_training_data/analytic_pattern_windows.csv.gz",
    )
    parser.add_argument(
        "--base-curve-root",
        type=Path,
        default=root / "experiment-results/phase14f_post_rendezvous/curve",
    )
    parser.add_argument(
        "--extension-curve-root",
        type=Path,
        default=root / "experiment-results/phase15_l1_curve_extension/curve",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase15_pattern_predictor",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260806)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_histogram(text):
    values = np.asarray(json.loads(text), dtype=np.float32)
    total = float(np.sum(values))
    return values / total if total else values


def make_features(frame):
    columns = []
    names = []
    for name in SCALAR_FEATURES:
        values = frame[name].to_numpy(dtype=np.float32)
        if name != "history_lm_correlation":
            values = np.log1p(np.maximum(values, 0))
        columns.append(values[:, None])
        names.append(name)
    for prefix in ("history_input", "history_output"):
        matrix = np.stack(
            [normalized_histogram(text) for text in frame[f"{prefix}_log2_hist"]]
        )
        columns.append(matrix)
        names.extend(f"{prefix}_log2_bin_{index}" for index in range(matrix.shape[1]))
    return np.concatenate(columns, axis=1).astype(np.float32), names


def make_targets(frame):
    active = (frame["selected_batch_size"].to_numpy(dtype=np.int64) > 0).astype(
        np.float32
    )
    prefill_tokens = (
        frame["prefill_payload_per_call_bytes"].to_numpy(dtype=np.float32)
        / BYTES_PER_TOKEN
    )
    steps = np.stack(
        [
            frame[f"decode_steps_active_{active_count}"].to_numpy(dtype=np.float32)
            for active_count in range(1, MAX_BATCH + 1)
        ],
        axis=1,
    )
    return active, np.concatenate([prefill_tokens[:, None], steps], axis=1)


class PatternMLP(nn.Module):
    def __init__(self, feature_count):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_count, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, features):
        return self.network(features)


def f1_at_threshold(actual, probability, threshold):
    predicted = probability >= threshold
    actual = actual.astype(bool)
    true_positive = int(np.sum(predicted & actual))
    false_positive = int(np.sum(predicted & ~actual))
    false_negative = int(np.sum(~predicted & actual))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return f1, precision, recall


def train_model(features, active, targets, splits, args):
    train_index = np.flatnonzero(splits == "train")
    validation_index = np.flatnonzero(splits == "validation")
    feature_mean = features[train_index].mean(axis=0)
    feature_std = features[train_index].std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0
    normalized_features = (features - feature_mean) / feature_std

    log_targets = np.log1p(targets)
    active_train = train_index[active[train_index] > 0]
    target_mean = log_targets[active_train].mean(axis=0)
    target_std = log_targets[active_train].std(axis=0)
    target_std[target_std < 1e-6] = 1.0
    normalized_targets = (log_targets - target_mean) / target_std

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = PatternMLP(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    positive = float(np.sum(active[train_index]))
    negative = float(len(train_index) - positive)
    activity_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negative / max(positive, 1)], device=device)
    )
    regression_loss = nn.SmoothL1Loss()
    dataset = TensorDataset(
        torch.from_numpy(normalized_features[train_index]),
        torch.from_numpy(active[train_index, None]),
        torch.from_numpy(normalized_targets[train_index].astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_x = torch.from_numpy(normalized_features[validation_index]).to(device)
    validation_active = torch.from_numpy(active[validation_index, None]).to(device)
    validation_target = torch.from_numpy(
        normalized_targets[validation_index].astype(np.float32)
    ).to(device)

    best_state = None
    best_loss = math.inf
    stale = 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for batch_x, batch_active, batch_target in loader:
            batch_x = batch_x.to(device)
            batch_active = batch_active.to(device)
            batch_target = batch_target.to(device)
            output = model(batch_x)
            mask = batch_active[:, 0] > 0
            loss_activity = activity_loss(output[:, :1], batch_active)
            loss_regression = (
                regression_loss(output[mask, 1:], batch_target[mask])
                if torch.any(mask)
                else output[:, 1:].sum() * 0
            )
            loss = loss_activity + loss_regression
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            output = model(validation_x)
            mask = validation_active[:, 0] > 0
            val_activity = activity_loss(output[:, :1], validation_active)
            val_regression = regression_loss(output[mask, 1:], validation_target[mask])
            val_loss = float((val_activity + val_regression).cpu())
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "validation_loss": val_loss,
            }
        )
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(normalized_features).to(device)).cpu().numpy()
    probability = 1 / (1 + np.exp(-np.clip(output[:, 0], -30, 30)))
    predicted_targets = np.expm1(output[:, 1:] * target_std + target_mean)
    predicted_targets = np.maximum(predicted_targets, 0)
    predicted_targets[:, 0] = np.clip(predicted_targets[:, 0], 16, 65536)
    predicted_targets[:, 1:] = np.clip(
        predicted_targets[:, 1:], 0, MAX_DECODE_STEPS
    )
    step_total = predicted_targets[:, 1:].sum(axis=1)
    over = step_total > MAX_DECODE_STEPS
    predicted_targets[over, 1:] *= (
        MAX_DECODE_STEPS / step_total[over, None]
    )
    candidates = np.linspace(0.05, 0.95, 91)
    threshold = max(
        candidates,
        key=lambda value: f1_at_threshold(
            active[validation_index], probability[validation_index], value
        )[0],
    )
    predicted_active = probability >= threshold
    predicted_targets[~predicted_active] = 0
    checkpoint = {
        "model_state": {key: value.cpu() for key, value in model.state_dict().items()},
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "activity_threshold": float(threshold),
        "seed": args.seed,
        "feature_count": features.shape[1],
    }
    return probability, predicted_targets, history, checkpoint, str(device)


def persistence_predictions(frame):
    expected_count = frame["history_rps"].to_numpy(dtype=np.float64) * 60.0
    predicted_active = expected_count >= 0.5
    batch = np.clip(np.rint(expected_count), 1, MAX_BATCH).astype(int)
    input_mean = np.clip(
        frame["history_input_mean"].to_numpy(dtype=np.float64), 16, 8192
    )
    output_mean = np.clip(
        np.rint(frame["history_output_mean"].to_numpy(dtype=np.float64)), 2, 128
    ).astype(int)
    targets = np.zeros((len(frame), 9), dtype=np.float64)
    targets[:, 0] = batch * input_mean
    for index in range(len(frame)):
        targets[index, batch[index]] = output_mean[index] - 1
    targets[~predicted_active] = 0
    return predicted_active.astype(float), targets


def load_curve(*roots):
    grouped = defaultdict(list)
    for root in roots:
        for path in sorted(root.glob("tp*/all_reduce/r*/curve.jsonl")):
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    grouped[(int(row["group_size"]), int(row["payload_bytes"]))].extend(
                        float(value) for value in row["post_rendezvous_samples_us"]
                    )
    points = defaultdict(list)
    for (tp, payload), values in grouped.items():
        points[tp].append((payload, float(np.median(values))))
    for tp in TPS:
        points[tp].sort()
        if not points[tp] or points[tp][0][0] > 8192 or points[tp][-1][0] < 536870912:
            raise AssertionError(f"incomplete L1 curve support for TP={tp}")
    return points


def interpolate(points, payload):
    payload = float(np.clip(payload, points[0][0], points[-1][0]))
    xs = np.log2(np.asarray([point[0] for point in points], dtype=np.float64))
    ys = np.asarray([point[1] for point in points], dtype=np.float64)
    return float(np.interp(math.log2(payload), xs, ys))


def pattern_quantities(targets):
    steps = targets[:, 1:]
    active_weight = np.arange(1, MAX_BATCH + 1, dtype=np.float64)
    return {
        "prefill_bytes": targets[:, 0] * BYTES_PER_TOKEN * CALLS_PER_FORWARD,
        "decode_calls": np.sum(steps, axis=1) * CALLS_PER_FORWARD,
        "decode_bytes": (steps @ active_weight)
        * BYTES_PER_TOKEN
        * CALLS_PER_FORWARD,
    }


def curve_costs(targets, points, tp):
    result = np.zeros(len(targets), dtype=np.float64)
    for index, target in enumerate(targets):
        if target[0] <= 0:
            continue
        result[index] += CALLS_PER_FORWARD * interpolate(
            points[tp], target[0] * BYTES_PER_TOKEN
        )
        for active_count, step_count in enumerate(target[1:], start=1):
            if step_count > 0:
                result[index] += (
                    step_count
                    * CALLS_PER_FORWARD
                    * interpolate(points[tp], active_count * BYTES_PER_TOKEN)
                )
    return result


def safe_ape(actual, predicted):
    return np.abs(predicted - actual) / np.maximum(actual, 1e-12)


def evaluate_method(method, frame, actual_active, actual_targets, probability, predicted, curves):
    rows = []
    prediction_rows = []
    actual_quantities = pattern_quantities(actual_targets)
    predicted_quantities = pattern_quantities(predicted)
    for scope in SCOPES:
        selected = frame["split"].to_numpy() == scope
        scope_active = selected & (actual_active > 0)
        f1, precision, recall = f1_at_threshold(
            actual_active[selected], probability[selected], 0.5
        )
        base = {
            "method": method,
            "scope": scope,
            "windows": int(np.sum(selected)),
            "active_windows": int(np.sum(scope_active)),
            "activity_accuracy": float(
                np.mean((probability[selected] >= 0.5) == (actual_active[selected] > 0))
            ),
            "activity_precision": precision,
            "activity_recall": recall,
            "activity_f1": f1,
        }
        for field in ("prefill_bytes", "decode_calls", "decode_bytes"):
            ape = safe_ape(
                actual_quantities[field][scope_active],
                predicted_quantities[field][scope_active],
            )
            base[f"{field}_mape"] = float(np.mean(ape))
            base[f"{field}_p95_ape"] = float(np.percentile(ape, 95))
        histogram_error = np.sum(
            np.abs(predicted[scope_active, 1:] - actual_targets[scope_active, 1:])
        )
        histogram_total = np.sum(actual_targets[scope_active, 1:])
        base["decode_histogram_wape"] = float(
            histogram_error / max(histogram_total, 1e-12)
        )
        for tp in TPS:
            actual_cost = curve_costs(actual_targets[scope_active], curves, tp)
            predicted_cost = curve_costs(predicted[scope_active], curves, tp)
            ape = safe_ape(actual_cost, predicted_cost)
            base[f"l1_tp{tp}_structural_mape"] = float(np.mean(ape))
            base[f"l1_tp{tp}_structural_p95_ape"] = float(np.percentile(ape, 95))
        rows.append(base)
    for index in range(len(frame)):
        prediction_rows.append(
            {
                "method": method,
                "window_id": frame.iloc[index]["window_id"],
                "source": frame.iloc[index]["source"],
                "segment": frame.iloc[index]["segment"],
                "split": frame.iloc[index]["split"],
                "actual_active": int(actual_active[index]),
                "predicted_active_probability": float(probability[index]),
                "actual_prefill_tokens": float(actual_targets[index, 0]),
                "predicted_prefill_tokens": float(predicted[index, 0]),
                "actual_decode_steps_json": json.dumps(
                    actual_targets[index, 1:].astype(float).tolist(), separators=(",", ":")
                ),
                "predicted_decode_steps_json": json.dumps(
                    predicted[index, 1:].astype(float).tolist(), separators=(",", ":")
                ),
            }
        )
    return rows, prediction_rows


def write_csv(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.dataset, compression="gzip")
    features, feature_names = make_features(frame)
    actual_active, actual_targets = make_targets(frame)
    probability, mlp_targets, history, checkpoint, device = train_model(
        features,
        actual_active,
        actual_targets,
        frame["split"].to_numpy(),
        args,
    )
    threshold = checkpoint["activity_threshold"]
    # Evaluation uses the validation-selected threshold, encoded as 0.5 probability.
    mlp_decision_probability = (probability >= threshold).astype(float)
    persistence_probability, persistence_targets = persistence_predictions(frame)
    curves = load_curve(args.base_curve_root, args.extension_curve_root)

    metrics = []
    predictions = []
    for method, method_probability, method_targets in (
        ("history_persistence", persistence_probability, persistence_targets),
        ("history_profile_mlp", mlp_decision_probability, mlp_targets),
        ("scheduled_batch_formula_oracle", actual_active, actual_targets),
    ):
        method_metrics, method_predictions = evaluate_method(
            method,
            frame,
            actual_active,
            actual_targets,
            method_probability,
            method_targets,
            curves,
        )
        metrics.extend(method_metrics)
        predictions.extend(method_predictions)
    write_csv(args.output_dir / "metrics.csv", metrics)
    write_csv(args.output_dir / "training_history.csv", history)
    with gzip.open(args.output_dir / "predictions.csv.gz", "wt", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    torch.save(checkpoint, args.output_dir / "model.pt")
    (args.output_dir / "feature_names.json").write_text(
        json.dumps(feature_names, indent=2) + "\n"
    )

    selected = {
        (row["method"], row["scope"]): row for row in metrics
    }
    summary = {
        "schema_version": "phase15-history-pattern-predictor-v1",
        "status": "PASS",
        "model": "Qwen3-8B",
        "training_windows": int(np.sum(frame["split"].to_numpy() == "train")),
        "validation_windows": int(
            np.sum(frame["split"].to_numpy() == "validation")
        ),
        "test_scopes": {
            scope: int(np.sum(frame["split"].to_numpy() == scope))
            for scope in SCOPES[1:]
        },
        "input_contract": (
            "300-second historical arrival, prompt-length and output-length profile; "
            "fixed 60-second horizon and max-batch-8 draining policy"
        ),
        "output_contract": (
            "next-window representative draining-batch Prefill payload plus Decode "
            "active-batch step histogram"
        ),
        "device": device,
        "epochs_completed": len(history),
        "activity_threshold": threshold,
        "selected_metrics": {
            f"{method}:{scope}": selected[(method, scope)]
            for method in ("history_persistence", "history_profile_mlp")
            for scope in ("test", "temporal_test", "external_test")
        },
        "l1_evaluation_contract": (
            "structural propagation error: predicted histogram times measured L1 "
            "curve versus oracle histogram times the same curve"
        ),
        "independent_phase14f_absolute_time_mape": 0.044251,
        "important_boundary": (
            "The formula oracle uses future scheduled request lengths and is not a "
            "forecast. The history MLP predicts one representative simultaneous "
            "draining batch, not full online continuous batching. Phase 15 has no new "
            "all-rank real communication-time labels, so its L1 metric isolates Stage-1 "
            "histogram error; absolute curve accuracy is inherited from Phase 14F."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(
        """# Phase 15：历史画像到 PatternDemand 的首版预测器

该阶段以 300 秒历史流量画像为输入，预测下一 60 秒窗口中一个代表性、最多 8 请求的
simultaneous draining batch 的 PatternDemand。输入包含到达率/突发度、历史 prompt 与
output 长度统计及 log2 直方图；输出为 Prefill 单次 payload 和 Decode 中
`active_batch=1..8` 的持续步数，随后恢复精确消息直方图。

对照包括历史均值 persistence、两层 MLP，以及使用未来已调度请求长度的解析公式上界。
L1 评测把预测直方图和真实直方图分别乘同一条实测连续代价曲线，因此衡量的是第一阶段
误差向通信代价的传播；它不是 Phase 15 新测的绝对通信时间误差。Phase14F 已独立验证
“真实直方图 × L1 曲线”对 all-rank 通信时间的 MAPE 为 4.43%。

边界：当前标签仍是代表性 draining batch，不是完整在线 continuous batching。历史画像
无法决定下一窗口的精确请求集合，因此该模型是流量预测 pilot；调度器若已掌握待调度请求
的实际长度，应优先使用解析 PatternDemand 路径。
"""
    )
    manifest = args.output_dir / "manifest.sha256"
    files = sorted(path for path in args.output_dir.iterdir() if path.is_file() and path != manifest)
    manifest.write_text("".join(f"{sha256(path)}  {path.name}\n" for path in files))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
