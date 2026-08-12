#!/usr/bin/env python3
"""Train Phase 27 PP predictors without reading independent confirmation targets."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import io
import json
import math
import platform
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BIN_COUNT = 12
ENCODED_SIZE = 2 * (BIN_COUNT + 1)
METHODS = (
    "h0",
    "legacy_bounded_residual",
    "enhanced_bounded_residual",
    "enhanced_direct",
)
LEARNED_METHODS = METHODS[1:]
FIT_ROLE = "development_train"
VALIDATION_ROLE = "development_validation"
CONFIRMATION_ROLE = "independent_confirmation"
COMMON_REFERENCE_LAUNCH_US = 5.0
COMMON_REFERENCE_BANDWIDTH_GBPS = 100.0
LEGACY_PROFILE_SCALARS = {
    "feature_profile_rps",
    "feature_profile_interarrival_cv",
    "feature_profile_peak_to_mean_1s",
    "feature_profile_fano_1s",
    "feature_profile_input_mean_capped",
    "feature_profile_output_mean_capped",
    "feature_profile_lm_correlation_capped",
    "feature_profile_survival_m_gt_8",
    "feature_profile_survival_m_gt_16",
    "feature_profile_survival_m_gt_32",
    "feature_profile_survival_m_gt_64",
}
LEGACY_DEPLOYMENT = {
    "feature_parallelism_pp",
    "feature_parallel_size_log2",
    "feature_phase_prefill",
    "feature_phase_decode",
    "feature_pp_max_microbatch_size",
    "feature_pp_chunk_tokens",
    "feature_pp_page_size",
    "feature_pp_proxy_tensor_count",
}
MODEL_FEATURE_EXCLUSIONS = {"feature_model_canonical_op_mask_all_reduce"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--development-dataset",
        type=Path,
        default=root
        / "experiment-results/phase27b_pp_hfull_dataset/dataset/development_examples.csv.gz",
    )
    parser.add_argument(
        "--confirmation-features",
        type=Path,
        default=root
        / "experiment-results/phase27b_pp_hfull_dataset/dataset/independent_confirmation_features.csv.gz",
    )
    parser.add_argument(
        "--dataset-summary",
        type=Path,
        default=root / "experiment-results/phase27b_pp_hfull_dataset/summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase27c_pp_scheduler_feature_training",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_gzip(path: Path, text: str) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as output:
            output.write(text.encode())


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_csv_gz(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    deterministic_gzip(path, buffer.getvalue())


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def load_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as source:
        return list(csv.DictReader(source))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def is_log_feature(name: str) -> bool:
    if name.startswith("feature_model_"):
        return name not in {
            "feature_model_is_moe",
            "feature_model_kv_head_ratio",
            "feature_model_moe_intermediate_ratio",
            "feature_model_moe_layer_frequency",
            "feature_model_canonical_op_mask_all_reduce",
        }
    exact = {
        "feature_profile_request_count",
        "feature_profile_rps",
        "feature_profile_interarrival_cv",
        "feature_profile_peak_to_mean_1s",
        "feature_profile_fano_1s",
        "feature_profile_multichunk_run_length_mean",
        "feature_profile_multichunk_run_length_p90",
        "feature_profile_multichunk_run_length_max",
        "feature_pp_max_microbatch_size",
        "feature_pp_chunk_tokens",
        "feature_pp_page_size",
        "feature_pp_proxy_tensor_count",
    }
    if name in exact:
        return True
    return any(
        token in name
        for token in (
            "_input_mean_",
            "_input_p50_",
            "_input_p90_",
            "_input_p99_",
            "_output_mean_",
            "_output_p50_",
            "_output_p90_",
            "_output_p99_",
            "_chunk_count_",
            "_chunk_output_work_",
        )
    )


def transform_feature(name: str, value: str) -> float:
    numeric = float(value)
    return math.log1p(max(numeric, 0.0)) if is_log_feature(name) else numeric


def legacy_feature_names(all_features: list[str]) -> list[str]:
    result = []
    for name in all_features:
        keep = (
            name in LEGACY_PROFILE_SCALARS
            or name.startswith("feature_profile_joint_lm_")
            or (name.startswith("feature_model_") and name not in MODEL_FEATURE_EXCLUSIONS)
            or name in LEGACY_DEPLOYMENT
        )
        if keep:
            result.append(name)
    return result


def target_encode(calls: np.ndarray, logical_bytes: np.ndarray) -> np.ndarray:
    encoded: list[float] = []
    for vector in (calls, logical_bytes):
        total = max(float(np.sum(vector)), 0.0)
        smoothing = max(total, 1.0) * 1e-6 / BIN_COUNT
        shares = (vector + smoothing) / (total + smoothing * BIN_COUNT)
        encoded.extend([math.log1p(total), *np.log(shares)])
    return np.asarray(encoded, dtype=np.float32)


def target_decode(encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vectors = []
    offset = 0
    for _ in range(2):
        total = max(math.expm1(float(np.clip(encoded[offset], 0, 40))), 0.0)
        logits = np.clip(encoded[offset + 1 : offset + BIN_COUNT + 1], -50, 50)
        probabilities = np.exp(logits - np.max(logits))
        probabilities /= probabilities.sum()
        vectors.append(total * probabilities)
        offset += BIN_COUNT + 1
    return vectors[0].astype(np.float64), vectors[1].astype(np.float64)


def residual_bounds() -> np.ndarray:
    bounds = np.full(ENCODED_SIZE, 2.0, dtype=np.float32)
    bounds[0] = math.log(2.0)
    bounds[BIN_COUNT + 1] = math.log(2.0)
    return bounds


class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, bounded: bool):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )
        self.bounded = bounded

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.network(value)
        return torch.tanh(output) if self.bounded else output


def parse_histograms(rows: list[dict[str, str]], prefix: str) -> tuple[np.ndarray, np.ndarray]:
    calls = np.stack(
        [np.asarray(json.loads(row[f"{prefix}_calls_by_12bin_json"]), dtype=np.float64) for row in rows]
    )
    logical_bytes = np.stack(
        [
            np.asarray(
                json.loads(row[f"{prefix}_logical_bytes_by_12bin_json"]),
                dtype=np.float64,
            )
            for row in rows
        ]
    )
    return calls, logical_bytes


def prepare_development(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    target_calls, target_bytes = parse_histograms(rows, "target")
    h0_calls, h0_bytes = parse_histograms(rows, "h0")
    target_encoded = np.stack(
        [target_encode(calls, byte_values) for calls, byte_values in zip(target_calls, target_bytes)]
    )
    h0_encoded = np.stack(
        [target_encode(calls, byte_values) for calls, byte_values in zip(h0_calls, h0_bytes)]
    )
    bounds = residual_bounds()
    return {
        "target_calls": target_calls,
        "target_bytes": target_bytes,
        "h0_calls": h0_calls,
        "h0_bytes": h0_bytes,
        "target_encoded": target_encoded,
        "h0_encoded": h0_encoded,
        "bounded_residual": np.clip(target_encoded - h0_encoded, -bounds, bounds),
    }


def feature_matrix(rows: list[dict[str, str]], feature_names: list[str]) -> np.ndarray:
    return np.asarray(
        [[transform_feature(name, row[name]) for name in feature_names] for row in rows],
        dtype=np.float32,
    )


def fit_model(
    *,
    method: str,
    rows: list[dict[str, str]],
    arrays: dict[str, np.ndarray],
    feature_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[dict, list[dict]]:
    train_indices = np.asarray(
        [index for index, row in enumerate(rows) if row["phase27_role"] == FIT_ROLE], dtype=int
    )
    validation_indices = np.asarray(
        [index for index, row in enumerate(rows) if row["phase27_role"] == VALIDATION_ROLE], dtype=int
    )
    features = feature_matrix(rows, feature_names)
    feature_mean = features[train_indices].mean(axis=0)
    feature_std = features[train_indices].std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0
    scaled = np.clip((features - feature_mean) / feature_std, -6.0, 6.0).astype(np.float32)

    bounded = method.endswith("bounded_residual")
    if bounded:
        target_mean = np.zeros(ENCODED_SIZE, dtype=np.float32)
        target_std = residual_bounds()
        targets = (arrays["bounded_residual"] / target_std).astype(np.float32)
    else:
        raw_targets = arrays["target_encoded"]
        target_mean = raw_targets[train_indices].mean(axis=0)
        target_std = raw_targets[train_indices].std(axis=0)
        target_std[target_std < 1e-6] = 1.0
        targets = ((raw_targets - target_mean) / target_std).astype(np.float32)

    seed_all(seed)
    model = MLP(len(feature_names), ENCODED_SIZE, bounded=bounded).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(scaled[train_indices]),
            torch.from_numpy(targets[train_indices]),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_x = torch.from_numpy(scaled[validation_indices]).to(device)
    validation_y = torch.from_numpy(targets[validation_indices]).to(device)
    best_state = None
    best_loss = math.inf
    stale = 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            loss = loss_fn(model(batch_x), batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(model(validation_x), validation_y).cpu())
        history.append(
            {
                "method": method,
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError(f"no checkpoint for {method}")
    checkpoint = {
        "schema_version": "phase27c-pp-predictor-checkpoint-v1",
        "method": method,
        "bin_schema_id": "pp_native_12bin_4k_8g_v1",
        "feature_names": feature_names,
        "log_feature_names": [name for name in feature_names if is_log_feature(name)],
        "feature_mean": torch.from_numpy(feature_mean),
        "feature_std": torch.from_numpy(feature_std),
        "target_mean": torch.from_numpy(target_mean),
        "target_std_or_residual_bounds": torch.from_numpy(target_std),
        "model_state": {name: value.detach().cpu() for name, value in best_state.items()},
        "architecture": {
            "hidden_sizes": [64, 64],
            "activation": "relu",
            "bounded_tanh": bounded,
        },
        "best_epoch": int(np.argmin([row["validation_loss"] for row in history])),
        "best_validation_loss": best_loss,
        "fit_role": FIT_ROLE,
        "validation_role": VALIDATION_ROLE,
        "forbidden_role": CONFIRMATION_ROLE,
        "seed": seed,
    }
    return checkpoint, history


def predict(
    rows: list[dict[str, str]], checkpoint: dict, h0_encoded: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    names = checkpoint["feature_names"]
    features = feature_matrix(rows, names)
    mean = checkpoint["feature_mean"].numpy()
    std = checkpoint["feature_std"].numpy()
    scaled = np.clip((features - mean) / std, -6.0, 6.0).astype(np.float32)
    bounded = checkpoint["architecture"]["bounded_tanh"]
    model = MLP(len(names), ENCODED_SIZE, bounded=bounded).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        raw = model(torch.from_numpy(scaled).to(device)).cpu().numpy()
    if bounded:
        encoded = h0_encoded + raw * checkpoint["target_std_or_residual_bounds"].numpy()
    else:
        raw = np.clip(raw, -6.0, 6.0)
        encoded = raw * checkpoint["target_std_or_residual_bounds"].numpy() + checkpoint[
            "target_mean"
        ].numpy()
    calls, logical_bytes = zip(*(target_decode(row) for row in encoded))
    return np.stack(calls), np.stack(logical_bytes)


def bin_log_centers(edges: list[float]) -> np.ndarray:
    values = np.asarray(edges, dtype=np.float64)
    return (np.log2(values[:-1]) + np.log2(values[1:])) / 2


def normalized_log_emd(predicted: np.ndarray, actual: np.ndarray, edges: list[float]) -> float:
    predicted_total = max(float(predicted.sum()), 1e-12)
    actual_total = max(float(actual.sum()), 1e-12)
    predicted_cdf = np.cumsum(predicted / predicted_total)
    actual_cdf = np.cumsum(actual / actual_total)
    centers = bin_log_centers(edges)
    area = float(
        np.sum(np.abs(predicted_cdf[:-1] - actual_cdf[:-1]) * np.diff(centers))
    )
    return area / (math.log2(edges[-1]) - math.log2(edges[0]))


def histogram_tv(predicted: np.ndarray, actual: np.ndarray) -> float:
    predicted_total = max(float(predicted.sum()), 1e-12)
    actual_total = max(float(actual.sum()), 1e-12)
    return float(np.abs(predicted / predicted_total - actual / actual_total).sum() / 2)


def common_reference_cost(calls: np.ndarray, logical_bytes: np.ndarray) -> float:
    return float(
        COMMON_REFERENCE_LAUNCH_US * calls.sum()
        + logical_bytes.sum() / (COMMON_REFERENCE_BANDWIDTH_GBPS * 1e9) * 1e6
    )


def case_record(
    row: dict[str, str],
    method: str,
    phase: str,
    actual_calls: np.ndarray,
    actual_bytes: np.ndarray,
    predicted_calls: np.ndarray,
    predicted_bytes: np.ndarray,
    edges: list[float],
) -> dict:
    actual_calls_total = float(actual_calls.sum())
    predicted_calls_total = float(predicted_calls.sum())
    actual_bytes_total = float(actual_bytes.sum())
    predicted_bytes_total = float(predicted_bytes.sum())
    actual_cost = common_reference_cost(actual_calls, actual_bytes)
    predicted_cost = common_reference_cost(predicted_calls, predicted_bytes)
    return {
        "profile_id": row["profile_id"],
        "segment": row["segment"],
        "parallel_size": row["parallel_size"],
        "policy": row["policy"],
        "method": method,
        "phase": phase,
        "actual_total_calls": actual_calls_total,
        "predicted_total_calls": predicted_calls_total,
        "calls_absolute_error": abs(predicted_calls_total - actual_calls_total),
        "calls_ape": abs(predicted_calls_total - actual_calls_total) / max(actual_calls_total, 1e-12),
        "actual_total_logical_bytes": actual_bytes_total,
        "predicted_total_logical_bytes": predicted_bytes_total,
        "bytes_absolute_error": abs(predicted_bytes_total - actual_bytes_total),
        "bytes_ape": abs(predicted_bytes_total - actual_bytes_total) / max(actual_bytes_total, 1e-12),
        "histogram_l1": 2 * histogram_tv(predicted_calls, actual_calls),
        "histogram_tv": histogram_tv(predicted_calls, actual_calls),
        "normalized_log_payload_emd": normalized_log_emd(predicted_calls, actual_calls, edges),
        "actual_common_reference_cost_us": actual_cost,
        "predicted_common_reference_cost_us": predicted_cost,
        "cost_absolute_error": abs(predicted_cost - actual_cost),
        "cost_ape": abs(predicted_cost - actual_cost) / max(actual_cost, 1e-12),
    }


def validation_records(
    rows: list[dict[str, str]],
    arrays: dict[str, np.ndarray],
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    edges: list[float],
) -> list[dict]:
    validation_indices = [
        index for index, row in enumerate(rows) if row["phase27_role"] == VALIDATION_ROLE
    ]
    grouped: dict[tuple, list[int]] = defaultdict(list)
    for index in validation_indices:
        row = rows[index]
        grouped[(row["profile_id"], row["parallel_size"], row["policy"])].append(index)
    records = []
    for method in METHODS:
        predicted_calls, predicted_bytes = predictions[method]
        for indices in grouped.values():
            if len(indices) != 2 or {rows[index]["phase"] for index in indices} != set(PHASES):
                raise ValueError("validation configuration lacks two phases")
            indices = sorted(indices, key=lambda index: rows[index]["phase"])
            for index in indices:
                records.append(
                    case_record(
                        rows[index],
                        method,
                        rows[index]["phase"],
                        arrays["target_calls"][index],
                        arrays["target_bytes"][index],
                        predicted_calls[index],
                        predicted_bytes[index],
                        edges,
                    )
                )
            representative = rows[indices[0]]
            pooled_actual_calls = sum((arrays["target_calls"][index] for index in indices))
            pooled_actual_bytes = sum((arrays["target_bytes"][index] for index in indices))
            pooled_predicted_calls = sum((predicted_calls[index] for index in indices))
            pooled_predicted_bytes = sum((predicted_bytes[index] for index in indices))
            total = case_record(
                representative,
                method,
                "total",
                pooled_actual_calls,
                pooled_actual_bytes,
                pooled_predicted_calls,
                pooled_predicted_bytes,
                edges,
            )
            actual_phase_aware = np.concatenate([arrays["target_calls"][index] for index in indices])
            predicted_phase_aware = np.concatenate([predicted_calls[index] for index in indices])
            total["histogram_tv"] = histogram_tv(predicted_phase_aware, actual_phase_aware)
            total["histogram_l1"] = 2 * total["histogram_tv"]
            records.append(total)
    return records


PHASES = ("prefill", "decode")


def aggregate_records(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in records:
        for segment in ("all", row["segment"]):
            for policy in ("all", row["policy"]):
                groups[(row["method"], row["phase"], policy, segment)].append(row)
    result = []
    for (method, phase, policy, segment), values in sorted(groups.items()):
        actual_calls = sum(float(row["actual_total_calls"]) for row in values)
        actual_bytes = sum(float(row["actual_total_logical_bytes"]) for row in values)
        actual_cost = sum(float(row["actual_common_reference_cost_us"]) for row in values)
        result.append(
            {
                "method": method,
                "phase": phase,
                "policy": policy,
                "segment": segment,
                "cases": len(values),
                "calls_mape": float(np.mean([float(row["calls_ape"]) for row in values])),
                "calls_wape": sum(float(row["calls_absolute_error"]) for row in values) / actual_calls,
                "bytes_mape": float(np.mean([float(row["bytes_ape"]) for row in values])),
                "bytes_wape": sum(float(row["bytes_absolute_error"]) for row in values) / actual_bytes,
                "mean_histogram_l1": float(np.mean([float(row["histogram_l1"]) for row in values])),
                "mean_histogram_tv": float(np.mean([float(row["histogram_tv"]) for row in values])),
                "mean_normalized_log_payload_emd": float(
                    np.mean([float(row["normalized_log_payload_emd"]) for row in values])
                ),
                "common_reference_cost_mape": float(np.mean([float(row["cost_ape"]) for row in values])),
                "common_reference_cost_wape": sum(float(row["cost_absolute_error"]) for row in values)
                / actual_cost,
            }
        )
    return result


def select_candidates(metrics: list[dict]) -> list[dict]:
    rows = []
    residuals = ("legacy_bounded_residual", "enhanced_bounded_residual")
    for policy in ("mb1", "mb4", "mb16"):
        lookup = {
            row["method"]: row
            for row in metrics
            if row["phase"] == "total" and row["policy"] == policy and row["segment"] == "all"
        }
        h0 = lookup["h0"]
        scored = []
        for method in residuals:
            value = lookup[method]
            wins = sum(
                float(value[field]) < float(h0[field])
                for field in ("calls_mape", "mean_histogram_tv", "common_reference_cost_mape")
            )
            cost_guard = float(value["common_reference_cost_mape"]) <= 1.10 * float(
                h0["common_reference_cost_mape"]
            )
            ratio_sum = sum(
                float(value[field]) / max(float(h0[field]), 1e-12)
                for field in ("calls_mape", "mean_histogram_tv", "common_reference_cost_mape")
            )
            scored.append((method, wins, cost_guard, ratio_sum))
        eligible = [row for row in scored if row[1] >= 2 and row[2]]
        selected = min(eligible, key=lambda row: (-row[1], row[3], row[0]))[0] if eligible else "h0"
        rows.append(
            {
                "policy": policy,
                "selected_method": selected,
                "selection_source": "development_validation_only",
                "rule": "at_least_2_of_calls_tv_cost_wins_and_cost_mape_within_110pct_of_h0",
                "h0_calls_mape": h0["calls_mape"],
                "h0_histogram_tv": h0["mean_histogram_tv"],
                "h0_cost_mape": h0["common_reference_cost_mape"],
                "legacy_wins": scored[0][1],
                "legacy_cost_guard": scored[0][2],
                "enhanced_wins": scored[1][1],
                "enhanced_cost_guard": scored[1][2],
            }
        )
    return rows


def confirmation_prediction_rows(
    rows: list[dict[str, str]], predictions: dict[str, tuple[np.ndarray, np.ndarray]]
) -> list[dict]:
    output = []
    for method in METHODS:
        calls, logical_bytes = predictions[method]
        for index, row in enumerate(rows):
            output.append(
                {
                    "training_id": row["training_id"],
                    "profile_id": row["profile_id"],
                    "segment": row["segment"],
                    "parallel_size": row["parallel_size"],
                    "policy": row["policy"],
                    "phase": row["phase"],
                    "method": method,
                    "predicted_total_calls_per_1000": float(calls[index].sum()),
                    "predicted_total_logical_bytes_per_1000": float(logical_bytes[index].sum()),
                    "predicted_common_reference_cost_us_per_1000": common_reference_cost(
                        calls[index], logical_bytes[index]
                    ),
                    "predicted_calls_by_12bin_json": json.dumps(calls[index].tolist(), separators=(",", ":")),
                    "predicted_logical_bytes_by_12bin_json": json.dumps(
                        logical_bytes[index].tolist(), separators=(",", ":")
                    ),
                }
            )
    return output


def plot_validation(path: Path, headline: dict[str, dict]) -> None:
    import matplotlib.pyplot as plt

    labels = ("H0", "Legacy residual", "Enhanced residual", "Direct")
    colors = ("#4C78A8", "#A0A0A0", "#F58518", "#B8B8B8")
    metrics = (
        ("calls_mape", "Total calls MAPE", 100.0, "%"),
        ("mean_histogram_tv", "Histogram TV", 1.0, ""),
        ("common_reference_cost_mape", "Common cost MAPE", 100.0, "%"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for axis, (metric, title, scale, suffix) in zip(axes, metrics):
        values = [headline[method][metric] * scale for method in METHODS]
        bars = axis.bar(labels, values, color=colors, width=0.68)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", rotation=18)
        axis.spines[["top", "right"]].set_visible(False)
        upper = max(values) * 1.18 if max(values) > 0 else 1.0
        axis.set_ylim(0, upper)
        for bar, value in zip(bars, values):
            label = f"{value:.1f}{suffix}" if suffix else f"{value:.3f}"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + upper * 0.025,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )
    figure.suptitle("Phase 27C development validation: PP feature comparison", fontsize=14)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def readme(summary: dict) -> str:
    headline = summary["validation_headline"]
    table = [
        "| 方法 | calls MAPE/WAPE | bytes MAPE/WAPE | TV | norm EMD | common cost MAPE/WAPE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = headline[method]
        table.append(
            "| {method} | {cm:.2%} / {cw:.2%} | {bm:.2%} / {bw:.2%} | {tv:.4f} | {emd:.4f} | {cost:.2%} / {costw:.2%} |".format(
                method=method,
                cm=row["calls_mape"],
                cw=row["calls_wape"],
                bm=row["bytes_mape"],
                bw=row["bytes_wape"],
                tv=row["mean_histogram_tv"],
                emd=row["mean_normalized_log_payload_emd"],
                cost=row["common_reference_cost_mape"],
                costw=row["common_reference_cost_wape"],
            )
        )
    decisions = "\n".join(
        f"- {row['policy']}：`{row['selected_method']}`" for row in summary["candidate_decisions"]
    )
    return f"""# Phase 27C：PP 调度敏感低维特征训练

状态：**{summary['status']}**。本阶段只读取 Phase 27B 的 30 个开发训练画像、12 个开发
验证画像和不含 target 的确认集特征文件；独立确认真值文件没有作为脚本参数，也没有被读取。

## 开发验证集 total 结果

{chr(10).join(table)}

`legacy_bounded_residual`只使用Phase 26同口径的长度联合分布、均值、生存率、模型与固定
PP配置；`enhanced_bounded_residual`在相同样本与训练流程下额外使用4096-token chunk、
chunk×输出、顺序转移、连续段和局部拥塞摘要。因此两者差异主要回答“调度敏感画像是否
提供额外信息”。`enhanced_direct`仅作为控制组，不参与候选规则。

## 已冻结的确认集候选

{decisions}

候选规则只看开发验证集：相对H0至少赢得calls MAPE、TV、cost MAPE中的两项，且cost
MAPE不超过H0的110%；多个residual合格时选择胜项更多、三项相对比值之和更小者。

`analysis/independent_confirmation_predictions.csv.gz`已经在不读确认真值时写出四种方法的
1,296行预测。下一步评测脚本只做hash核验和真值join，不能再训练、早停或改变候选映射。
当前可以确认训练隔离和候选冻结成立；不能把这里的validation结果当作独立泛化结论。
"""


def main() -> None:
    args = parse_args()
    for name in ("checkpoints", "analysis", "figures", "logs"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    dataset_summary = json.loads(args.dataset_summary.read_text())
    if dataset_summary["status"] != "PASS":
        raise ValueError("Phase 27B dataset is not PASS")
    development = load_rows(args.development_dataset)
    confirmation = load_rows(args.confirmation_features)
    if len(development) != 756 or len(confirmation) != 324:
        raise ValueError(f"unexpected row counts: {len(development)}, {len(confirmation)}")
    if any(name.startswith("target_") for name in confirmation[0]):
        raise ValueError("confirmation feature artifact contains target columns")

    all_features = [name for name in development[0] if name.startswith("feature_")]
    legacy_features = legacy_feature_names(all_features)
    if len(all_features) != 108:
        raise ValueError(f"expected 108 enhanced features, got {len(all_features)}")
    device = choose_device(args.device)
    arrays = prepare_development(development)
    development_profiles = {
        row["profile_id"]: row["phase27_role"] for row in development
    }
    profile_role_counts = Counter(development_profiles.values())
    if profile_role_counts != Counter({FIT_ROLE: 30, VALIDATION_ROLE: 12}):
        raise ValueError(profile_role_counts)

    checkpoints = {}
    checkpoint_inventory = []
    history_rows = []
    training_runs = []
    for method_index, method in enumerate(LEARNED_METHODS):
        feature_names = legacy_features if method.startswith("legacy_") else all_features
        checkpoint, history = fit_model(
            method=method,
            rows=development,
            arrays=arrays,
            feature_names=feature_names,
            args=args,
            device=device,
            seed=args.seed + method_index,
        )
        path = args.output_dir / "checkpoints" / f"pp_{method}.pt"
        torch.save(checkpoint, path)
        checkpoints[method] = checkpoint
        history_rows.extend(history)
        inventory = {
            "method": method,
            "path": str(path.relative_to(args.output_dir)),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "feature_columns": len(feature_names),
            "best_epoch": checkpoint["best_epoch"],
            "best_validation_loss": checkpoint["best_validation_loss"],
        }
        checkpoint_inventory.append(inventory)
        training_runs.append(inventory)

    predictions = {"h0": (arrays["h0_calls"], arrays["h0_bytes"])}
    for method, checkpoint in checkpoints.items():
        predictions[method] = predict(
            development, checkpoint, arrays["h0_encoded"], device
        )
    edges = json.loads(development[0]["h0_calls_by_12bin_json"])
    del edges
    bin_edges = np.geomspace(4 * 1024, 8 * 1024 * 1024 * 1024, 13).tolist()
    validation = validation_records(development, arrays, predictions, bin_edges)
    metrics = aggregate_records(validation)
    headline = {
        method: next(
            row
            for row in metrics
            if row["method"] == method
            and row["phase"] == "total"
            and row["policy"] == "all"
            and row["segment"] == "all"
        )
        for method in METHODS
    }
    candidates = select_candidates(metrics)

    confirmation_h0_calls, confirmation_h0_bytes = parse_histograms(confirmation, "h0")
    confirmation_h0_encoded = np.stack(
        [
            target_encode(calls, byte_values)
            for calls, byte_values in zip(confirmation_h0_calls, confirmation_h0_bytes)
        ]
    )
    confirmation_predictions = {"h0": (confirmation_h0_calls, confirmation_h0_bytes)}
    for method, checkpoint in checkpoints.items():
        confirmation_predictions[method] = predict(
            confirmation, checkpoint, confirmation_h0_encoded, device
        )
    frozen_prediction_rows = confirmation_prediction_rows(
        confirmation, confirmation_predictions
    )

    write_csv_gz(args.output_dir / "analysis/training_history.csv.gz", history_rows)
    write_csv_gz(args.output_dir / "analysis/validation_predictions.csv.gz", validation)
    write_csv(args.output_dir / "analysis/validation_metrics.csv", metrics)
    write_csv(args.output_dir / "analysis/checkpoint_inventory.csv", checkpoint_inventory)
    write_csv(args.output_dir / "analysis/candidate_decisions.csv", candidates)
    write_csv_gz(
        args.output_dir / "analysis/independent_confirmation_predictions.csv.gz",
        frozen_prediction_rows,
    )
    plot_validation(args.output_dir / "figures/validation_feature_comparison.png", headline)

    checks = {
        "phase27b_status_pass": dataset_summary["status"] == "PASS",
        "development_rows_756": len(development) == 756,
        "fit_validation_profiles_30_12": profile_role_counts
        == Counter({FIT_ROLE: 30, VALIDATION_ROLE: 12}),
        "confirmation_feature_rows_324": len(confirmation) == 324,
        "confirmation_features_have_no_target_columns": not any(
            name.startswith("target_") for name in confirmation[0]
        ),
        "enhanced_feature_columns_108": len(all_features) == 108,
        "legacy_feature_subset_strict": 0 < len(legacy_features) < len(all_features),
        "three_frozen_checkpoints": len(checkpoint_inventory) == 3,
        "validation_records_1296": len(validation) == 12 * 3 * 3 * 3 * 4,
        "confirmation_predictions_1296": len(frozen_prediction_rows) == 324 * 4,
        "candidate_mapping_three_policies": len(candidates) == 3,
        "all_metrics_finite": all(
            math.isfinite(float(row[field]))
            for row in metrics
            for field in (
                "calls_mape",
                "calls_wape",
                "bytes_mape",
                "bytes_wape",
                "mean_histogram_tv",
                "mean_normalized_log_payload_emd",
                "common_reference_cost_mape",
            )
        ),
        "confirmation_targets_not_a_script_input": not hasattr(args, "confirmation_targets"),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    if status != "PASS":
        raise RuntimeError(checks)

    summary = {
        "schema_version": "phase27c-pp-scheduler-feature-training-v1",
        "status": status,
        "objective": "compare legacy and PP scheduler-sensitive low-dimensional features under Hfull supervision while freezing confirmation predictions before target access",
        "device": str(device),
        "counts": {
            "development_phase_rows": len(development),
            "fit_profiles": 30,
            "validation_profiles": 12,
            "confirmation_feature_rows": len(confirmation),
            "legacy_feature_columns": len(legacy_features),
            "enhanced_feature_columns": len(all_features),
            "checkpoints": len(checkpoint_inventory),
            "validation_prediction_records": len(validation),
            "confirmation_prediction_rows": len(frozen_prediction_rows),
        },
        "inputs": {
            "development_dataset_sha256": sha256(args.development_dataset),
            "confirmation_features_sha256": sha256(args.confirmation_features),
            "dataset_summary_sha256": sha256(args.dataset_summary),
        },
        "split_contract": {
            "fit": FIT_ROLE,
            "early_stopping": VALIDATION_ROLE,
            "confirmation_features_only": CONFIRMATION_ROLE,
            "confirmation_targets_read": False,
        },
        "validation_headline": headline,
        "candidate_decisions": candidates,
        "checkpoints": checkpoint_inventory,
        "checks": checks,
        "next_step": "freeze this commit, then join the already-written confirmation predictions with Phase27B confirmation targets in a separate evaluation script",
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(
        args.output_dir / "feature_contract.json",
        {
            "schema_version": "phase27c-pp-training-feature-contract-v1",
            "legacy_feature_columns": legacy_features,
            "enhanced_feature_columns": all_features,
            "methods": list(METHODS),
            "candidate_rule": candidates[0]["rule"],
            "target_encoding": "log1p total plus log smoothed 12-bin shares for calls and logical bytes",
            "bounded_residual": {
                "total_log_bound": math.log(2.0),
                "share_logit_bound": 2.0,
                "network_output": "tanh",
            },
        },
    )
    write_json(
        args.output_dir / "audit_summary.json",
        {
            "schema_version": "phase27c-pp-scheduler-feature-training-audit-v1",
            "status": status,
            "checks": checks,
            "checkpoint_sha256": {
                row["method"]: row["sha256"] for row in checkpoint_inventory
            },
            "confirmation_predictions_sha256": sha256(
                args.output_dir / "analysis/independent_confirmation_predictions.csv.gz"
            ),
        },
    )
    (args.output_dir / "README.md").write_text(readme(summary))
    (args.output_dir / "DONE").write_text("PASS\n")
    try:
        repository_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        repository_head = "unknown"
    write_json(
        args.output_dir / "logs/training.log",
        {
            "schema_version": "phase27c-training-log-v1",
            "status": status,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository_head_at_training": repository_head,
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "args": {
                **vars(args),
                "development_dataset": str(args.development_dataset),
                "confirmation_features": str(args.confirmation_features),
                "dataset_summary": str(args.dataset_summary),
                "output_dir": str(args.output_dir),
            },
            "training_runs": training_runs,
            "confirmation_targets_read": False,
        },
    )
    manifest = []
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.sha256":
            manifest.append(f"{sha256(path)}  {path.relative_to(args.output_dir)}")
    (args.output_dir / "manifest.sha256").write_text("\n".join(manifest) + "\n")
    print(
        json.dumps(
            {
                "status": status,
                "device": str(device),
                "legacy_features": len(legacy_features),
                "enhanced_features": len(all_features),
                "candidate_decisions": {
                    row["policy"]: row["selected_method"] for row in candidates
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
