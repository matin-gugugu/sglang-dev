#!/usr/bin/env python3
"""Small deterministic NumPy models for Phase54 development refinement."""

from __future__ import annotations

import copy
import csv
import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


BIN_COUNT = 12
ENCODED_SIZE = 2 * (BIN_COUNT + 1)
ARRIVAL_TOKENS = ("_rps", "interarrival", "peak_to_mean", "fano")


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_csv_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty rows: {path}")
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows[1:]):
        raise ValueError("inconsistent output schema")
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as output:
            output.write(buffer.getvalue().encode("utf-8"))


def write_json_gz(path: Path, value: Any) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as output:
            output.write(encoded)


def read_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return json.load(source)


def histogram_arrays(rows: list[dict[str, str]], prefix: str) -> tuple[np.ndarray, np.ndarray]:
    calls = np.asarray([[float(row[f"{prefix}_calls_bin_{i:02d}"]) for i in range(BIN_COUNT)] for row in rows], dtype=np.float64)
    logical_bytes = np.asarray([[float(row[f"{prefix}_logical_bytes_bin_{i:02d}"]) for i in range(BIN_COUNT)] for row in rows], dtype=np.float64)
    return calls, logical_bytes


def encode_histograms(calls: np.ndarray, logical_bytes: np.ndarray) -> np.ndarray:
    encoded = []
    for vectors in (calls, logical_bytes):
        totals = np.maximum(vectors.sum(axis=1), 0.0)
        smoothing = np.maximum(totals, 1.0)[:, None] * 1e-6 / BIN_COUNT
        shares = (vectors + smoothing) / (totals[:, None] + smoothing * BIN_COUNT)
        encoded.append(np.concatenate([np.log1p(totals)[:, None], np.log(shares)], axis=1))
    return np.concatenate(encoded, axis=1)


def decode_histograms(encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vectors = []
    offset = 0
    for _ in range(2):
        totals = np.expm1(np.clip(encoded[:, offset], 0.0, 40.0))
        logits = np.clip(encoded[:, offset + 1:offset + BIN_COUNT + 1], -50.0, 50.0)
        logits -= logits.max(axis=1, keepdims=True)
        shares = np.exp(logits); shares /= np.maximum(shares.sum(axis=1, keepdims=True), 1e-12)
        vectors.append(totals[:, None] * shares)
        offset += BIN_COUNT + 1
    return vectors[0], vectors[1]


def residual_bounds() -> np.ndarray:
    bounds = np.full(ENCODED_SIZE, 2.0, dtype=np.float64)
    bounds[0] = math.log(2.0); bounds[BIN_COUNT + 1] = math.log(2.0)
    return bounds


def raw_input_matrix(rows: list[dict[str, str]], names: list[str]) -> np.ndarray:
    values = np.asarray([[float(row[name]) for name in names] for row in rows], dtype=np.float64)
    for column, name in enumerate(names):
        if name.startswith("h0_"):
            values[:, column] = np.log1p(np.maximum(values[:, column], 0.0))
    if not np.isfinite(values).all():
        raise ValueError("non-finite predictor input")
    return values


def fit_transform(rows: list[dict[str, str]], feature_mode: str) -> dict[str, Any]:
    names = sorted(name for name in rows[0] if name.startswith("feature_") or name.startswith("h0_"))
    if feature_mode == "fixed_draining_causal":
        names = [name for name in names if not any(token in name for token in ARRIVAL_TOKENS)]
    elif feature_mode != "full_target_free":
        raise ValueError(feature_mode)
    raw = raw_input_matrix(rows, names)
    std = raw.std(axis=0); keep = std > 1e-12
    selected = [name for name, include in zip(names, keep) if include]
    raw = raw[:, keep]
    mean = raw.mean(axis=0); scale = raw.std(axis=0); scale[scale < 1e-12] = 1.0
    return {"input_names": selected, "input_mean": mean.tolist(), "input_scale": scale.tolist(), "residual_bounds": residual_bounds().tolist(), "feature_mode": feature_mode}


def transform_inputs(rows: list[dict[str, str]], transform: dict[str, Any]) -> np.ndarray:
    raw = raw_input_matrix(rows, list(transform["input_names"]))
    value = (raw - np.asarray(transform["input_mean"])) / np.asarray(transform["input_scale"])
    return np.clip(value, -8.0, 8.0)


def transform_targets(rows: list[dict[str, str]], transform: dict[str, Any]) -> np.ndarray:
    h0_calls, h0_bytes = histogram_arrays(rows, "h0")
    target_calls, target_bytes = histogram_arrays(rows, "target")
    residual = encode_histograms(target_calls, target_bytes) - encode_histograms(h0_calls, h0_bytes)
    return np.clip(residual / np.asarray(transform["residual_bounds"]), -1.0, 1.0)


def loss_weights(mode: str) -> np.ndarray:
    weights = np.ones(ENCODED_SIZE, dtype=np.float64)
    for offset in (0, BIN_COUNT + 1):
        if mode == "uniform":
            continue
        weights[offset] = 0.5
        if mode == "shape_focus":
            weights[offset + 1:offset + BIN_COUNT + 1] = 1.5
        elif mode == "tail_shape_focus":
            weights[offset + 1:offset + BIN_COUNT + 1] = 1.0 + 0.08 * np.arange(BIN_COUNT, dtype=np.float64)
        else:
            raise ValueError(mode)
    return weights


def init_model(input_dim: int, width: int, depth: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    dims = [input_dim] + [width] * depth + [ENCODED_SIZE]
    weights = []; biases = []
    for left, right in zip(dims[:-1], dims[1:]):
        limit = math.sqrt(6.0 / (left + right))
        weights.append(rng.uniform(-limit, limit, size=(left, right))); biases.append(np.zeros(right, dtype=np.float64))
    return {"weights": weights, "biases": biases}


def forward(model: dict[str, Any], x: np.ndarray, *, cache: bool = False) -> Any:
    activations = [x]; value = x
    for weight, bias in zip(model["weights"], model["biases"]):
        value = np.tanh(value @ weight + bias); activations.append(value)
    return (value, activations) if cache else value


def fit_model(x: np.ndarray, y: np.ndarray, config: dict[str, Any], seed: int, *, validation: tuple[np.ndarray, np.ndarray] | None = None, fixed_epochs: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    model = init_model(x.shape[1], int(config["width"]), int(config["depth"]), seed)
    weights = loss_weights(str(config["loss_mode"]))
    m_w = [np.zeros_like(value) for value in model["weights"]]; v_w = [np.zeros_like(value) for value in model["weights"]]
    m_b = [np.zeros_like(value) for value in model["biases"]]; v_b = [np.zeros_like(value) for value in model["biases"]]
    lr = float(config["learning_rate"]); wd = float(config["weight_decay"])
    maximum = int(fixed_epochs if fixed_epochs is not None else config["max_epochs"]); patience = int(config["patience"])
    best_model = copy.deepcopy(model); best_loss = math.inf; best_epoch = 0; stale = 0
    for epoch in range(1, maximum + 1):
        prediction, activations = forward(model, x, cache=True)
        delta = (2.0 * (prediction - y) * weights[None, :] / (x.shape[0] * y.shape[1])) * (1.0 - prediction ** 2)
        grad_w: list[np.ndarray] = [np.empty(0)] * len(model["weights"]); grad_b: list[np.ndarray] = [np.empty(0)] * len(model["biases"])
        for layer in range(len(model["weights"]) - 1, -1, -1):
            grad_w[layer] = activations[layer].T @ delta + wd * model["weights"][layer]
            grad_b[layer] = delta.sum(axis=0)
            if layer:
                delta = (delta @ model["weights"][layer].T) * (1.0 - activations[layer] ** 2)
        for layer in range(len(model["weights"])):
            m_w[layer] = 0.9 * m_w[layer] + 0.1 * grad_w[layer]; v_w[layer] = 0.999 * v_w[layer] + 0.001 * grad_w[layer] ** 2
            m_b[layer] = 0.9 * m_b[layer] + 0.1 * grad_b[layer]; v_b[layer] = 0.999 * v_b[layer] + 0.001 * grad_b[layer] ** 2
            correction1 = 1.0 - 0.9 ** epoch; correction2 = 1.0 - 0.999 ** epoch
            model["weights"][layer] -= lr * (m_w[layer] / correction1) / (np.sqrt(v_w[layer] / correction2) + 1e-8)
            model["biases"][layer] -= lr * (m_b[layer] / correction1) / (np.sqrt(v_b[layer] / correction2) + 1e-8)
        if validation is None:
            best_model = copy.deepcopy(model); best_epoch = epoch; continue
        val_prediction = forward(model, validation[0]); val_loss = float(np.mean(weights[None, :] * (val_prediction - validation[1]) ** 2))
        if val_loss < best_loss - 1e-8:
            best_loss = val_loss; best_epoch = epoch; best_model = copy.deepcopy(model); stale = 0
        else:
            stale += 1
            if stale >= patience: break
    training_loss = float(np.mean(weights[None, :] * (forward(best_model, x) - y) ** 2))
    return best_model, {"best_epoch": best_epoch, "best_validation_loss": None if validation is None else best_loss, "training_loss": training_loss, "epochs_ran": epoch}


def model_to_json(model: dict[str, Any]) -> dict[str, Any]:
    return {"weights": [value.tolist() for value in model["weights"]], "biases": [value.tolist() for value in model["biases"]]}


def predict_histograms(rows: list[dict[str, str]], transform: dict[str, Any], models: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    x = transform_inputs(rows, transform)
    normalized = np.mean([forward(model, x) for model in models], axis=0)
    residual = np.clip(normalized, -1.0, 1.0) * np.asarray(transform["residual_bounds"])
    h0_calls, h0_bytes = histogram_arrays(rows, "h0")
    return decode_histograms(encode_histograms(h0_calls, h0_bytes) + residual)
