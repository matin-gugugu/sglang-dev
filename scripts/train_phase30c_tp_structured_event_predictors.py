#!/usr/bin/env python3
"""Train TP structured-event DNNs without reading either confirmation target."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
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

from build_phase25_full_window_teacher import PHASES, TP_BIN_EDGES
from build_phase29b_tp_hfull_dataset import MODELS, STRATEGIES, TP_SIZES, all_model_features
from build_phase30b_tp_structured_event_dataset import event_names, reconstruct_message_vectors
from train_phase27c_pp_scheduler_feature_predictors import (
    case_record,
    choose_device,
    is_log_feature,
    predict as predict_phase29,
    sha256,
    target_encode,
)


FIT_ROLE = "development_train"
VALIDATION_ROLE = "development_validation"
FIRST_ROLE = "independent_confirmation"
SECOND_ROLE = "second_independent_confirmation"
POLICIES = tuple(STRATEGIES)
METHODS = (
    "h0",
    "phase29_enhanced_bounded_residual_diagnostic",
    "structured_event_bounded_residual",
    "structured_event_direct_control",
)
LEARNED_METHODS = METHODS[2:]
EVENT_COUNT = 62
PREFILL_CATEGORIES = 23
DECODE_LANES = 16
BIN_COUNT = 12
RESIDUAL_BOUND_FLOOR = math.log(2.0)
LOSS_WEIGHTS = {
    "normalized_event_smooth_l1": 1.0,
    "log_total_calls": 0.25,
    "log_total_logical_bytes": 0.25,
    "phase_aware_histogram_tv": 0.50,
    "log_common_reference_cost": 0.25,
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    base = root / "experiment-results/phase30b_tp_structured_event_dataset"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--development-dataset",
        type=Path,
        default=base / "dataset/development_examples.csv.gz",
    )
    parser.add_argument(
        "--first-confirmation-features",
        type=Path,
        default=base / "dataset/first_confirmation_features.csv.gz",
    )
    parser.add_argument(
        "--second-confirmation-features",
        type=Path,
        default=base / "dataset/second_confirmation_features.csv.gz",
    )
    parser.add_argument("--dataset-summary", type=Path, default=base / "summary.json")
    parser.add_argument(
        "--feature-contract", type=Path, default=base / "feature_columns.json"
    )
    parser.add_argument(
        "--event-contract",
        type=Path,
        default=root
        / "experiment-results/phase30a_tp_structured_event_contract/event_contract.json",
    )
    parser.add_argument(
        "--modeling-contract",
        type=Path,
        default=root
        / "experiment-results/phase30a_tp_structured_event_contract/modeling_contract.json",
    )
    parser.add_argument(
        "--model-features",
        type=Path,
        default=root / "experiment-results/phase16_model_features/model_features.json",
    )
    parser.add_argument(
        "--phase29-checkpoint",
        type=Path,
        default=root
        / "experiment-results/phase29c_tp_aligned_training/checkpoints/tp_enhanced_bounded_residual.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase30c_tp_structured_event_training",
    )
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as source:
        return list(csv.DictReader(source))


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


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def transform_feature(name: str, value: str) -> float:
    numeric = float(value)
    return math.log1p(max(numeric, 0.0)) if is_log_feature(name) else numeric


def feature_matrix(rows: list[dict[str, str]], names: list[str]) -> np.ndarray:
    return np.asarray(
        [[transform_feature(name, row[name]) for name in names] for row in rows],
        dtype=np.float32,
    )


def event_matrix(
    rows: list[dict[str, str]], prefix: str, names: list[str]
) -> np.ndarray:
    return np.asarray(
        [[float(row[f"{prefix}{name}"]) for name in names] for row in rows],
        dtype=np.float32,
    )


class EventMLP(nn.Module):
    def __init__(self, input_size: int, bounded: bool):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, EVENT_COUNT),
        )
        self.bounded = bounded

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.network(value)
        return torch.tanh(output) if self.bounded else output


def model_constants(models: dict[str, tuple[dict, dict]]) -> list[tuple[int, int]]:
    return sorted(
        {
            (
                int(raw["logical_collectives_per_forward_prior"]),
                int(raw["payload_bytes_per_active_token_prior"]),
            )
            for raw, _ in models.values()
        }
    )


def histogram_projection(
    contract: dict, bytes_per_token: int, device: torch.device
) -> torch.Tensor:
    projection = torch.zeros(EVENT_COUNT, 2 * BIN_COUNT, device=device)
    byte_key = f"tp_bin_for_{bytes_per_token}_bytes_per_token"
    for category in contract["prefill_joint_categories"]:
        projection[int(category["category"]), int(category[byte_key])] = 1.0
    for lanes in range(1, DECODE_LANES + 1):
        target_bin = int(
            np.clip(
                np.searchsorted(TP_BIN_EDGES, lanes * bytes_per_token, side="right")
                - 1,
                0,
                BIN_COUNT - 1,
            )
        )
        projection[2 * PREFILL_CATEGORIES + lanes - 1, BIN_COUNT + target_bin] = 1.0
    return projection


def multiobjective_loss(
    predicted_log: torch.Tensor,
    target_log: torch.Tensor,
    target_scale: torch.Tensor,
    constants: list[tuple[int, int]],
    projections: dict[int, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    predicted = torch.expm1(torch.clamp(predicted_log, 0.0, 30.0))
    target = torch.expm1(torch.clamp(target_log, 0.0, 30.0))
    event_loss = nn.functional.smooth_l1_loss(
        (predicted_log - target_log) / target_scale,
        torch.zeros_like(predicted_log),
    )
    calls_losses = []
    bytes_losses = []
    tv_losses = []
    cost_losses = []
    for calls_per_forward, bytes_per_token in constants:
        predicted_prefill_calls = predicted[:, :PREFILL_CATEGORIES].sum(1) * calls_per_forward
        target_prefill_calls = target[:, :PREFILL_CATEGORIES].sum(1) * calls_per_forward
        predicted_decode_calls = predicted[:, 2 * PREFILL_CATEGORIES :].sum(1) * calls_per_forward
        target_decode_calls = target[:, 2 * PREFILL_CATEGORIES :].sum(1) * calls_per_forward
        predicted_prefill_bytes = (
            predicted[:, PREFILL_CATEGORIES : 2 * PREFILL_CATEGORIES].sum(1)
            * bytes_per_token
            * calls_per_forward
        )
        target_prefill_bytes = (
            target[:, PREFILL_CATEGORIES : 2 * PREFILL_CATEGORIES].sum(1)
            * bytes_per_token
            * calls_per_forward
        )
        lanes = torch.arange(1, DECODE_LANES + 1, device=predicted.device)
        predicted_decode_bytes = (
            (predicted[:, 2 * PREFILL_CATEGORIES :] * lanes).sum(1)
            * bytes_per_token
            * calls_per_forward
        )
        target_decode_bytes = (
            (target[:, 2 * PREFILL_CATEGORIES :] * lanes).sum(1)
            * bytes_per_token
            * calls_per_forward
        )
        predicted_calls = torch.stack(
            (predicted_prefill_calls, predicted_decode_calls), dim=1
        )
        target_calls = torch.stack((target_prefill_calls, target_decode_calls), dim=1)
        predicted_bytes = torch.stack(
            (predicted_prefill_bytes, predicted_decode_bytes), dim=1
        )
        target_bytes = torch.stack((target_prefill_bytes, target_decode_bytes), dim=1)
        calls_losses.append(
            nn.functional.smooth_l1_loss(
                torch.log1p(predicted_calls), torch.log1p(target_calls)
            )
        )
        bytes_losses.append(
            nn.functional.smooth_l1_loss(
                torch.log1p(predicted_bytes), torch.log1p(target_bytes)
            )
        )
        predicted_histogram = predicted @ projections[bytes_per_token]
        target_histogram = target @ projections[bytes_per_token]
        predicted_share = predicted_histogram / torch.clamp(
            predicted_histogram.sum(1, keepdim=True), min=1e-12
        )
        target_share = target_histogram / torch.clamp(
            target_histogram.sum(1, keepdim=True), min=1e-12
        )
        tv_losses.append(0.5 * torch.abs(predicted_share - target_share).sum(1).mean())
        predicted_cost = (
            5.0 * predicted_calls.sum(1) + predicted_bytes.sum(1) / 100000.0
        )
        target_cost = 5.0 * target_calls.sum(1) + target_bytes.sum(1) / 100000.0
        cost_losses.append(
            nn.functional.smooth_l1_loss(
                torch.log1p(predicted_cost), torch.log1p(target_cost)
            )
        )
    components = {
        "normalized_event_smooth_l1": event_loss,
        "log_total_calls": torch.stack(calls_losses).mean(),
        "log_total_logical_bytes": torch.stack(bytes_losses).mean(),
        "phase_aware_histogram_tv": torch.stack(tv_losses).mean(),
        "log_common_reference_cost": torch.stack(cost_losses).mean(),
    }
    total = sum(LOSS_WEIGHTS[name] * value for name, value in components.items())
    return total, {name: float(value.detach().cpu()) for name, value in components.items()}


def decode_model_output(
    raw: torch.Tensor,
    method: str,
    h0_log: torch.Tensor,
    residual_bound: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    if method == "structured_event_bounded_residual":
        return torch.clamp(h0_log + raw * residual_bound, min=0.0, max=30.0)
    return torch.clamp(raw * target_std + target_mean, min=0.0, max=30.0)


def fit_model(
    *,
    method: str,
    rows: list[dict[str, str]],
    feature_names: list[str],
    event_names_list: list[str],
    contract: dict,
    models: dict[str, tuple[dict, dict]],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[dict, list[dict]]:
    train_indices = np.asarray(
        [index for index, row in enumerate(rows) if row["role"] == FIT_ROLE], dtype=int
    )
    validation_indices = np.asarray(
        [index for index, row in enumerate(rows) if row["role"] == VALIDATION_ROLE],
        dtype=int,
    )
    features = feature_matrix(rows, feature_names)
    feature_mean = features[train_indices].mean(0)
    feature_std = features[train_indices].std(0)
    feature_std[feature_std < 1e-6] = 1.0
    scaled = np.clip((features - feature_mean) / feature_std, -6.0, 6.0).astype(
        np.float32
    )
    h0 = event_matrix(rows, "h0_event_", event_names_list)
    target = event_matrix(rows, "target_event_", event_names_list)
    h0_log = np.log1p(h0).astype(np.float32)
    target_log = np.log1p(target).astype(np.float32)
    residual_bound = np.maximum(
        np.max(np.abs(target_log[train_indices] - h0_log[train_indices]), axis=0),
        RESIDUAL_BOUND_FLOOR,
    ).astype(np.float32)
    target_mean = target_log[train_indices].mean(0).astype(np.float32)
    target_std = target_log[train_indices].std(0).astype(np.float32)
    target_std[target_std < 0.1] = 0.1
    target_scale = target_std.copy()

    seed_all(seed)
    bounded = method == "structured_event_bounded_residual"
    model = EventMLP(len(feature_names), bounded=bounded).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(scaled[train_indices]),
            torch.from_numpy(h0_log[train_indices]),
            torch.from_numpy(target_log[train_indices]),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_x = torch.from_numpy(scaled[validation_indices]).to(device)
    validation_h0 = torch.from_numpy(h0_log[validation_indices]).to(device)
    validation_target = torch.from_numpy(target_log[validation_indices]).to(device)
    bound_tensor = torch.from_numpy(residual_bound).to(device)
    mean_tensor = torch.from_numpy(target_mean).to(device)
    std_tensor = torch.from_numpy(target_std).to(device)
    scale_tensor = torch.from_numpy(target_scale).to(device)
    constants = model_constants(models)
    projections = {
        bytes_per_token: histogram_projection(contract, bytes_per_token, device)
        for _, bytes_per_token in constants
    }

    best_state = None
    best_loss = math.inf
    stale = 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_values = []
        for batch_x, batch_h0, batch_target in loader:
            batch_x = batch_x.to(device)
            batch_h0 = batch_h0.to(device)
            batch_target = batch_target.to(device)
            raw = model(batch_x)
            predicted_log = decode_model_output(
                raw,
                method,
                batch_h0,
                bound_tensor,
                mean_tensor,
                std_tensor,
            )
            loss, _ = multiobjective_loss(
                predicted_log,
                batch_target,
                scale_tensor,
                constants,
                projections,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_values.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            raw = model(validation_x)
            predicted_log = decode_model_output(
                raw,
                method,
                validation_h0,
                bound_tensor,
                mean_tensor,
                std_tensor,
            )
            validation_loss, components = multiobjective_loss(
                predicted_log,
                validation_target,
                scale_tensor,
                constants,
                projections,
            )
        validation_value = float(validation_loss.cpu())
        history.append(
            {
                "method": method,
                "epoch": epoch,
                "train_loss": float(np.mean(train_values)),
                "validation_loss": validation_value,
                **{f"validation_{name}": value for name, value in components.items()},
            }
        )
        if validation_value < best_loss - 1e-6:
            best_loss = validation_value
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError(f"no checkpoint for {method}")
    checkpoint = {
        "schema_version": "phase30c-tp-structured-event-checkpoint-v1",
        "method": method,
        "feature_names": feature_names,
        "event_names": event_names_list,
        "feature_mean": torch.from_numpy(feature_mean),
        "feature_std": torch.from_numpy(feature_std),
        "residual_bound": torch.from_numpy(residual_bound),
        "target_mean": torch.from_numpy(target_mean),
        "target_std": torch.from_numpy(target_std),
        "target_scale": torch.from_numpy(target_scale),
        "model_state": {name: value.detach().cpu() for name, value in best_state.items()},
        "architecture": {
            "hidden_sizes": [64, 64],
            "activation": "relu",
            "bounded_tanh": bounded,
        },
        "loss_weights": LOSS_WEIGHTS,
        "residual_bound_floor": RESIDUAL_BOUND_FLOOR,
        "best_epoch": int(np.argmin([row["validation_loss"] for row in history])),
        "best_validation_loss": best_loss,
        "fit_role": FIT_ROLE,
        "validation_role": VALIDATION_ROLE,
        "forbidden_roles": [FIRST_ROLE, SECOND_ROLE],
        "seed": seed,
    }
    return checkpoint, history


def predict_events(
    rows: list[dict[str, str]], checkpoint: dict, device: torch.device
) -> np.ndarray:
    features = feature_matrix(rows, checkpoint["feature_names"])
    mean = checkpoint["feature_mean"].numpy()
    std = checkpoint["feature_std"].numpy()
    scaled = np.clip((features - mean) / std, -6.0, 6.0).astype(np.float32)
    h0 = event_matrix(rows, "h0_event_", checkpoint["event_names"])
    h0_log = torch.from_numpy(np.log1p(h0).astype(np.float32)).to(device)
    bounded = checkpoint["architecture"]["bounded_tanh"]
    model = EventMLP(len(checkpoint["feature_names"]), bounded=bounded).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        raw = model(torch.from_numpy(scaled).to(device))
        encoded = decode_model_output(
            raw,
            checkpoint["method"],
            h0_log,
            checkpoint["residual_bound"].to(device),
            checkpoint["target_mean"].to(device),
            checkpoint["target_std"].to(device),
        )
    return torch.expm1(encoded).cpu().numpy().astype(np.float64)


def event_dict(vector: np.ndarray, names: list[str]) -> dict[str, float]:
    return {name: float(value) for name, value in zip(names, vector)}


def expanded_rows_and_vectors(
    units: list[dict[str, str]],
    events_by_method: dict[str, np.ndarray],
    target_events: np.ndarray | None,
    names: list[str],
    contract: dict,
    models: dict[str, tuple[dict, dict]],
) -> tuple[
    list[dict[str, str]],
    dict[str, tuple[np.ndarray, np.ndarray]],
    tuple[np.ndarray, np.ndarray] | None,
]:
    expanded = []
    method_calls: dict[str, list[np.ndarray]] = defaultdict(list)
    method_bytes: dict[str, list[np.ndarray]] = defaultdict(list)
    target_calls = []
    target_bytes = []
    for unit_index, unit in enumerate(units):
        for model_name in MODELS:
            raw_model, model_values = models[model_name]
            predicted_vectors = {
                method: reconstruct_message_vectors(
                    event_dict(values[unit_index], names), raw_model, contract
                )
                for method, values in events_by_method.items()
            }
            actual_vectors = (
                reconstruct_message_vectors(
                    event_dict(target_events[unit_index], names), raw_model, contract
                )
                if target_events is not None
                else None
            )
            for tp_size in TP_SIZES:
                for phase in PHASES:
                    row = {
                        **unit,
                        **model_values,
                        "model": model_name,
                        "parallelism": "tp",
                        "parallel_size": str(tp_size),
                        "phase": phase,
                        "feature_parallelism_tp": "1",
                        "feature_parallel_size_log2": str(math.log2(tp_size)),
                        "feature_phase_prefill": str(int(phase == "prefill")),
                        "feature_phase_decode": str(int(phase == "decode")),
                    }
                    expanded.append(row)
                    for method, vectors in predicted_vectors.items():
                        calls, logical_bytes = vectors[phase]
                        method_calls[method].append(calls)
                        method_bytes[method].append(logical_bytes)
                    if actual_vectors is not None:
                        calls, logical_bytes = actual_vectors[phase]
                        target_calls.append(calls)
                        target_bytes.append(logical_bytes)
    predictions = {
        method: (np.stack(method_calls[method]), np.stack(method_bytes[method]))
        for method in events_by_method
    }
    targets = (
        (np.stack(target_calls), np.stack(target_bytes))
        if target_events is not None
        else None
    )
    return expanded, predictions, targets


def phase29_diagnostic(
    rows: list[dict[str, str]],
    h0_vectors: tuple[np.ndarray, np.ndarray],
    checkpoint: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    h0_calls, h0_bytes = h0_vectors
    h0_encoded = np.stack(
        [target_encode(calls, values) for calls, values in zip(h0_calls, h0_bytes)]
    )
    return predict_phase29(rows, checkpoint, h0_encoded, device)


def validation_records(
    rows: list[dict[str, str]],
    targets: tuple[np.ndarray, np.ndarray],
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[dict]:
    actual_calls, actual_bytes = targets
    groups: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row["profile_id"], row["model"], row["parallel_size"], row["policy"])].append(index)
    records = []
    for method in METHODS:
        predicted_calls, predicted_bytes = predictions[method]
        for indices in groups.values():
            if len(indices) != 2 or {rows[index]["phase"] for index in indices} != set(PHASES):
                raise ValueError("validation configuration lacks two phases")
            indices = sorted(indices, key=lambda index: rows[index]["phase"])
            for index in indices:
                record = case_record(
                    rows[index],
                    method,
                    rows[index]["phase"],
                    actual_calls[index],
                    actual_bytes[index],
                    predicted_calls[index],
                    predicted_bytes[index],
                    TP_BIN_EDGES.tolist(),
                )
                record["model"] = rows[index]["model"]
                record["role"] = rows[index]["role"]
                records.append(record)
            representative = rows[indices[0]]
            pooled_actual_calls = sum((actual_calls[index] for index in indices))
            pooled_actual_bytes = sum((actual_bytes[index] for index in indices))
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
                TP_BIN_EDGES.tolist(),
            )
            actual_phase_aware = np.concatenate([actual_calls[index] for index in indices])
            predicted_phase_aware = np.concatenate([predicted_calls[index] for index in indices])
            actual_share = actual_phase_aware / max(float(actual_phase_aware.sum()), 1e-12)
            predicted_share = predicted_phase_aware / max(float(predicted_phase_aware.sum()), 1e-12)
            total["histogram_l1"] = float(np.abs(predicted_share - actual_share).sum())
            total["histogram_tv"] = total["histogram_l1"] / 2
            total["model"] = representative["model"]
            total["role"] = representative["role"]
            records.append(total)
    return records


def aggregate_records(records: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str, str, str, str], list[dict]] = defaultdict(list)
    for row in records:
        for model in ("all", row["model"]):
            for parallel_size in ("all", row["parallel_size"]):
                for policy in ("all", row["policy"]):
                    for segment in ("all", row["segment"]):
                        groups[(row["method"], row["phase"], model, parallel_size, policy, segment)].append(row)
    output = []
    for key, values in sorted(groups.items()):
        method, phase, model, parallel_size, policy, segment = key
        actual_calls = sum(float(row["actual_total_calls"]) for row in values)
        actual_bytes = sum(float(row["actual_total_logical_bytes"]) for row in values)
        actual_cost = sum(float(row["actual_common_reference_cost_us"]) for row in values)
        output.append(
            {
                "method": method,
                "phase": phase,
                "model": model,
                "parallel_size": parallel_size,
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
                "common_reference_cost_wape": sum(float(row["cost_absolute_error"]) for row in values) / actual_cost,
            }
        )
    return output


def select_candidates(metrics: list[dict]) -> list[dict]:
    decisions = []
    for policy in POLICIES:
        lookup = {
            row["method"]: row
            for row in metrics
            if row["phase"] == "total"
            and row["model"] == "all"
            and row["parallel_size"] == "all"
            and row["policy"] == policy
            and row["segment"] == "all"
        }
        h0 = lookup["h0"]
        residual = lookup["structured_event_bounded_residual"]
        fields = ("calls_mape", "mean_histogram_tv", "common_reference_cost_mape")
        wins = sum(float(residual[field]) < float(h0[field]) for field in fields)
        cost_guard = float(residual["common_reference_cost_mape"]) <= 1.10 * float(
            h0["common_reference_cost_mape"]
        )
        selected = "structured_event_bounded_residual" if wins >= 2 and cost_guard else "h0"
        decisions.append(
            {
                "policy": policy,
                "selected_method": selected,
                "selection_source": "development_validation_only",
                "rule": "residual_beats_h0_on_2_of_calls_tv_cost_and_cost_within_110pct",
                "residual_wins": wins,
                "residual_cost_guard": cost_guard,
                "h0_calls_mape": h0["calls_mape"],
                "residual_calls_mape": residual["calls_mape"],
                "h0_histogram_tv": h0["mean_histogram_tv"],
                "residual_histogram_tv": residual["mean_histogram_tv"],
                "h0_cost_mape": h0["common_reference_cost_mape"],
                "residual_cost_mape": residual["common_reference_cost_mape"],
            }
        )
    return decisions


def frozen_prediction_rows(
    rows: list[dict[str, str]], predictions: dict[str, tuple[np.ndarray, np.ndarray]]
) -> list[dict]:
    output = []
    for method in METHODS:
        calls, logical_bytes = predictions[method]
        for index, row in enumerate(rows):
            output.append(
                {
                    "prediction_id": f"{row['profile_id']}/{row['model']}/tp{row['parallel_size']}/{row['policy']}/{row['phase']}/{method}",
                    "profile_id": row["profile_id"],
                    "role": row["role"],
                    "source": row["source"],
                    "segment": row["segment"],
                    "window_id": row["window_id"],
                    "model": row["model"],
                    "parallelism": "tp",
                    "parallel_size": row["parallel_size"],
                    "policy": row["policy"],
                    "phase": row["phase"],
                    "method": method,
                    "predicted_total_calls_per_1000": float(calls[index].sum()),
                    "predicted_total_logical_bytes_per_1000": float(logical_bytes[index].sum()),
                    "predicted_common_reference_cost_us_per_1000": float(
                        5.0 * calls[index].sum() + logical_bytes[index].sum() / 100000.0
                    ),
                    "predicted_calls_by_12bin_json": json.dumps(calls[index].tolist(), separators=(",", ":")),
                    "predicted_logical_bytes_by_12bin_json": json.dumps(
                        logical_bytes[index].tolist(), separators=(",", ":")
                    ),
                    "prediction_frozen_before_first_confirmation_target_access": True,
                }
            )
    return output


def predict_expanded(
    units: list[dict[str, str]],
    names: list[str],
    contract: dict,
    models: dict[str, tuple[dict, dict]],
    checkpoints: dict[str, dict],
    phase29_checkpoint: dict,
    device: torch.device,
    target_events: np.ndarray | None = None,
) -> tuple[list[dict[str, str]], dict[str, tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray] | None]:
    h0 = event_matrix(units, "h0_event_", names).astype(np.float64)
    events = {"h0": h0}
    for method, checkpoint in checkpoints.items():
        events[method] = predict_events(units, checkpoint, device)
    expanded, predictions, targets = expanded_rows_and_vectors(
        units, events, target_events, names, contract, models
    )
    predictions["phase29_enhanced_bounded_residual_diagnostic"] = phase29_diagnostic(
        expanded, predictions["h0"], phase29_checkpoint, device
    )
    return expanded, predictions, targets


def plot_validation(path: Path, headline: dict[str, dict]) -> None:
    import matplotlib.pyplot as plt

    labels = ("H0", "Phase29 residual", "Structured residual", "Direct control")
    colors = ("#4C78A8", "#A0A0A0", "#F58518", "#B8B8B8")
    specs = (
        ("calls_mape", "Total calls MAPE", 100.0, "%"),
        ("mean_histogram_tv", "Histogram TV", 1.0, ""),
        ("common_reference_cost_mape", "Common cost MAPE", 100.0, "%"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for axis, (metric, title, scale, suffix) in zip(axes, specs):
        values = [headline[method][metric] * scale for method in METHODS]
        bars = axis.bar(labels, values, color=colors, width=0.68)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22, linewidth=0.8)
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
    figure.suptitle("Phase 30C development validation: structured TP events")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def readme(summary: dict) -> str:
    table = [
        "| 方法 | calls MAPE/WAPE | bytes MAPE/WAPE | TV | norm EMD | cost MAPE/WAPE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = summary["validation_headline"][method]
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
        f"- {row['policy']}：`{row['selected_method']}`"
        for row in summary["candidate_decisions"]
    )
    return f"""# Phase 30C：TP结构事件残差DNN训练与双确认预测冻结

状态：**{summary['status']}**。本阶段以75个训练画像×3个固定策略作为225个独立训练单位，
以27个验证画像×3个策略作为81个早停单位。输入是91列低维历史画像与固定策略特征；输出是
62维fixed-draining调度事件。主模型为compact32 H0事件先验加有界残差DNN，direct DNN只作
负对照，Phase29模型只作旧目标空间诊断，不参与候选选择。

## 开发验证集结果

{chr(10).join(table)}

按预注册规则冻结的第一确认候选为：

{decisions}

模型结构与TP size不进入事件DNN，而由确定性适配器把事件展开成三模型、TP2/4/8、prefill/decode
的12桶calls和logical bytes。TP size仍保留在预测合同与审计字段中；在当前拓扑无关结构teacher
下，它不改变事件到消息直方图的映射。

第一、第二新确认集各15个画像、45个画像×策略单元。本脚本只读取两套无target特征，并在读取
第一确认真值前同时冻结每套3,240条四方法预测。开发验证只能决定待确认候选，不能代替两级独立
泛化结论，也不能把共同参考代价解释为真实物理拓扑时延。
"""


def main() -> None:
    args = parse_args()
    for name in ("checkpoints", "analysis", "figures", "logs"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    dataset_summary = json.loads(args.dataset_summary.read_text())
    feature_contract = json.loads(args.feature_contract.read_text())
    contract = json.loads(args.event_contract.read_text())
    modeling_contract = json.loads(args.modeling_contract.read_text())
    development = load_rows(args.development_dataset)
    first = load_rows(args.first_confirmation_features)
    second = load_rows(args.second_confirmation_features)
    if dataset_summary["status"] != "PASS" or modeling_contract["primary_architecture"] != "compact32_H0_events_plus_bounded_event_residual_DNN":
        raise ValueError("Phase30 contracts are not PASS")
    if (len(development), len(first), len(second)) != (306, 45, 45):
        raise ValueError("unexpected Phase30C input row counts")
    if any(
        name.startswith("target_event_")
        for rows in (first, second)
        for name in rows[0]
    ):
        raise ValueError("confirmation feature artifact contains targets")
    feature_names = feature_contract["feature_columns"]
    names = event_names(contract)
    if len(feature_names) != 91 or len(names) != EVENT_COUNT:
        raise ValueError("feature/event contract mismatch")
    role_profiles = {row["profile_id"]: row["role"] for row in development}
    role_counts = Counter(role_profiles.values())
    if role_counts != Counter({FIT_ROLE: 75, VALIDATION_ROLE: 27}):
        raise ValueError(role_counts)
    models = all_model_features(args.model_features)
    if set(models) != set(MODELS):
        raise ValueError("model feature mismatch")
    device = choose_device(args.device)
    phase29_checkpoint = torch.load(
        args.phase29_checkpoint, map_location="cpu", weights_only=False
    )

    checkpoints = {}
    checkpoint_inventory = []
    histories = []
    for method_index, method in enumerate(LEARNED_METHODS):
        checkpoint, history = fit_model(
            method=method,
            rows=development,
            feature_names=feature_names,
            event_names_list=names,
            contract=contract,
            models=models,
            args=args,
            device=device,
            seed=args.seed + method_index,
        )
        path = args.output_dir / "checkpoints" / f"tp_{method}.pt"
        torch.save(checkpoint, path)
        checkpoints[method] = checkpoint
        histories.extend(history)
        checkpoint_inventory.append(
            {
                "method": method,
                "path": str(path.relative_to(args.output_dir)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "feature_columns": len(feature_names),
                "event_targets": len(names),
                "best_epoch": checkpoint["best_epoch"],
                "best_validation_loss": checkpoint["best_validation_loss"],
            }
        )

    development_targets = event_matrix(development, "target_event_", names).astype(np.float64)
    validation_units = [row for row in development if row["role"] == VALIDATION_ROLE]
    validation_target_events = event_matrix(
        validation_units, "target_event_", names
    ).astype(np.float64)
    validation_rows, validation_predictions, validation_targets = predict_expanded(
        validation_units,
        names,
        contract,
        models,
        checkpoints,
        phase29_checkpoint,
        device,
        target_events=validation_target_events,
    )
    if validation_targets is None:
        raise RuntimeError("validation targets missing")
    validation_cases = validation_records(
        validation_rows, validation_targets, validation_predictions
    )
    metrics = aggregate_records(validation_cases)
    headline = {
        method: next(
            row
            for row in metrics
            if row["method"] == method
            and row["phase"] == "total"
            and row["model"] == "all"
            and row["parallel_size"] == "all"
            and row["policy"] == "all"
            and row["segment"] == "all"
        )
        for method in METHODS
    }
    decisions = select_candidates(metrics)

    first_rows, first_predictions, _ = predict_expanded(
        first, names, contract, models, checkpoints, phase29_checkpoint, device
    )
    second_rows, second_predictions, _ = predict_expanded(
        second, names, contract, models, checkpoints, phase29_checkpoint, device
    )
    frozen_first = frozen_prediction_rows(first_rows, first_predictions)
    frozen_second = frozen_prediction_rows(second_rows, second_predictions)

    bounds = checkpoints["structured_event_bounded_residual"]["residual_bound"].numpy()
    write_csv_gz(args.output_dir / "analysis/training_history.csv.gz", histories)
    write_csv_gz(
        args.output_dir / "analysis/validation_case_metrics.csv.gz", validation_cases
    )
    write_csv(args.output_dir / "analysis/validation_metrics.csv", metrics)
    write_csv(args.output_dir / "analysis/checkpoint_inventory.csv", checkpoint_inventory)
    write_csv(args.output_dir / "analysis/candidate_decisions.csv", decisions)
    write_csv(
        args.output_dir / "analysis/event_residual_bounds.csv",
        [
            {"event_name": name, "log1p_residual_bound": float(value)}
            for name, value in zip(names, bounds)
        ],
    )
    write_csv_gz(
        args.output_dir / "analysis/first_confirmation_predictions.csv.gz",
        frozen_first,
    )
    write_csv_gz(
        args.output_dir / "analysis/second_confirmation_predictions.csv.gz",
        frozen_second,
    )
    plot_validation(args.output_dir / "figures/validation_tp_comparison.png", headline)

    checks = {
        "phase30b_status_pass": dataset_summary["status"] == "PASS",
        "development_units_306_fit_225_validation_81": len(development) == 306
        and sum(row["role"] == FIT_ROLE for row in development) == 225
        and sum(row["role"] == VALIDATION_ROLE for row in development) == 81,
        "profiles_fit_75_validation_27": role_counts
        == Counter({FIT_ROLE: 75, VALIDATION_ROLE: 27}),
        "features_91_events_62": len(feature_names) == 91 and len(names) == 62,
        "first_features_45_no_targets": len(first) == 45
        and not any(name.startswith("target_event_") for name in first[0]),
        "second_features_45_no_targets": len(second) == 45
        and not any(name.startswith("target_event_") for name in second[0]),
        "two_new_checkpoints": len(checkpoint_inventory) == 2,
        "validation_expanded_rows_1458": len(validation_rows) == 1458,
        "validation_case_metrics_8748": len(validation_cases) == 8748,
        "first_predictions_3240": len(frozen_first) == 3240,
        "second_predictions_3240": len(frozen_second) == 3240,
        "candidate_mapping_three_policies": len(decisions) == 3,
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
        "confirmation_targets_not_script_inputs": not any(
            "confirmation_target" in name for name in vars(args)
        ),
        "phase29_checkpoint_diagnostic_only": "phase29_enhanced_bounded_residual_diagnostic"
        not in {row["selected_method"] for row in decisions},
        "direct_control_not_selectable": "structured_event_direct_control"
        not in {row["selected_method"] for row in decisions},
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    if status != "PASS":
        raise RuntimeError(checks)
    summary = {
        "schema_version": "phase30c-tp-structured-event-training-v1",
        "status": status,
        "objective": "train compact32 H0 plus bounded structured-event residual DNN and freeze both new confirmation predictions before target access",
        "device": str(device),
        "counts": {
            "development_event_units": len(development),
            "fit_event_units": 225,
            "validation_event_units": 81,
            "fit_profiles": 75,
            "validation_profiles": 27,
            "validation_expanded_phase_rows": len(validation_rows),
            "validation_case_metric_rows": len(validation_cases),
            "first_confirmation_event_units": len(first),
            "second_confirmation_event_units": len(second),
            "first_confirmation_prediction_rows": len(frozen_first),
            "second_confirmation_prediction_rows": len(frozen_second),
            "feature_columns": len(feature_names),
            "event_targets": len(names),
            "new_checkpoints": len(checkpoint_inventory),
        },
        "inputs": {
            "development_dataset_sha256": sha256(args.development_dataset),
            "first_confirmation_features_sha256": sha256(args.first_confirmation_features),
            "second_confirmation_features_sha256": sha256(args.second_confirmation_features),
            "dataset_summary_sha256": sha256(args.dataset_summary),
            "feature_contract_sha256": sha256(args.feature_contract),
            "event_contract_sha256": sha256(args.event_contract),
            "modeling_contract_sha256": sha256(args.modeling_contract),
            "model_features_sha256": sha256(args.model_features),
            "phase29_checkpoint_sha256": sha256(args.phase29_checkpoint),
        },
        "split_contract": {
            "fit": FIT_ROLE,
            "early_stopping": VALIDATION_ROLE,
            "first_confirmation_features_only": FIRST_ROLE,
            "second_confirmation_features_only": SECOND_ROLE,
            "confirmation_targets_read": False,
        },
        "primary_architecture": "compact32_H0_events_plus_bounded_event_residual_DNN",
        "loss_weights": LOSS_WEIGHTS,
        "residual_bound": {
            "derivation": "per-target maximum absolute log1p residual on development_train only",
            "floor": RESIDUAL_BOUND_FLOOR,
            "minimum": float(bounds.min()),
            "maximum": float(bounds.max()),
        },
        "validation_headline": headline,
        "candidate_decisions": decisions,
        "checkpoints": checkpoint_inventory,
        "checks": checks,
        "can_conclude": [
            "structured event DNNs were trained with one profile-policy statistical unit",
            "development validation selected guarded candidates for first confirmation",
            "both new confirmation prediction artifacts were frozen without target access",
        ],
        "cannot_conclude": [
            "development validation improvement proves independent-window generalization",
            "the residual DNN is accepted before both confirmation evaluations",
            "the common reference cost is measured physical topology time",
        ],
        "next_step": "archive this run, then join only frozen first-confirmation predictions with isolated first targets",
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(
        args.output_dir / "feature_contract.json",
        {
            "schema_version": "phase30c-tp-structured-event-feature-contract-v1",
            "feature_columns": feature_names,
            "event_names": names,
            "methods": list(METHODS),
            "primary_architecture": summary["primary_architecture"],
            "h0_role": "structural_event_prior_baseline_and_guarded_fallback",
            "phase29_role": "frozen_old_target_space_diagnostic_only",
            "direct_role": "negative_control_only",
            "candidate_rule": decisions[0]["rule"],
            "event_transform": "log1p",
            "loss_weights": LOSS_WEIGHTS,
        },
    )
    write_json(
        args.output_dir / "audit_summary.json",
        {
            "schema_version": "phase30c-tp-structured-event-training-audit-v1",
            "status": status,
            "checks": checks,
            "checkpoint_sha256": {
                row["method"]: row["sha256"] for row in checkpoint_inventory
            },
            "phase29_diagnostic_checkpoint_sha256": sha256(args.phase29_checkpoint),
            "first_confirmation_predictions_sha256": sha256(
                args.output_dir / "analysis/first_confirmation_predictions.csv.gz"
            ),
            "second_confirmation_predictions_sha256": sha256(
                args.output_dir / "analysis/second_confirmation_predictions.csv.gz"
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
            "schema_version": "phase30c-training-log-v1",
            "status": status,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository_head_at_training": repository_head,
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "CPU",
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "training_runs": checkpoint_inventory,
            "loss_weights": LOSS_WEIGHTS,
            "confirmation_targets_read": False,
        },
    )
    files = sorted(
        path
        for path in args.output_dir.rglob("*")
        if path.is_file() and path.name != "manifest.sha256"
    )
    (args.output_dir / "manifest.sha256").write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(args.output_dir)}\n" for path in files
        )
    )
    print(
        json.dumps(
            {
                "status": status,
                "device": str(device),
                "candidate_decisions": {
                    row["policy"]: row["selected_method"] for row in decisions
                },
                "validation_headline": {
                    method: {
                        "calls_mape": headline[method]["calls_mape"],
                        "histogram_tv": headline[method]["mean_histogram_tv"],
                        "cost_mape": headline[method]["common_reference_cost_mape"],
                    }
                    for method in METHODS
                },
                "first_prediction_sha256": sha256(
                    args.output_dir / "analysis/first_confirmation_predictions.csv.gz"
                ),
                "second_prediction_sha256": sha256(
                    args.output_dir / "analysis/second_confirmation_predictions.csv.gz"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
