#!/usr/bin/env python3
"""Small deterministic NumPy MLP used by the frozen Phase42 predictor."""

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


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_csv_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty rows: {path}")
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows[1:]):
        raise ValueError(f"inconsistent schema: {path}")
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
    calls = np.asarray([[float(row[f"{prefix}_calls_bin_{index:02d}"]) for index in range(BIN_COUNT)] for row in rows], dtype=np.float64)
    logical_bytes = np.asarray([[float(row[f"{prefix}_logical_bytes_bin_{index:02d}"]) for index in range(BIN_COUNT)] for row in rows], dtype=np.float64)
    return calls, logical_bytes


def raw_input_matrix(rows: list[dict[str, str]], input_names: list[str]) -> np.ndarray:
    values = np.asarray([[float(row[name]) for name in input_names] for row in rows], dtype=np.float64)
    for column, name in enumerate(input_names):
        if name.startswith("h0_"):
            values[:, column] = np.log1p(np.maximum(values[:, column], 0.0))
    if not np.isfinite(values).all():
        raise ValueError("non-finite predictor input")
    return values


def fit_transform(rows: list[dict[str, str]]) -> dict[str, Any]:
    names = sorted(name for name in rows[0] if name.startswith("feature_") or name.startswith("h0_"))
    raw = raw_input_matrix(rows, names)
    std = raw.std(axis=0)
    keep = std > 1e-12
    selected = [name for name, include in zip(names, keep) if include]
    raw = raw[:, keep]
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0)
    scale[scale < 1e-12] = 1.0
    h0_calls, h0_bytes = histogram_arrays(rows, "h0")
    target_calls, target_bytes = histogram_arrays(rows, "target")
    residual = np.log1p(np.concatenate([target_calls, target_bytes], axis=1)) - np.log1p(np.concatenate([h0_calls, h0_bytes], axis=1))
    output_scale = residual.std(axis=0)
    output_scale[output_scale < 0.05] = 0.05
    return {
        "input_names": selected,
        "input_mean": mean.tolist(),
        "input_scale": scale.tolist(),
        "output_scale": output_scale.tolist(),
    }


def transform_inputs(rows: list[dict[str, str]], transform: dict[str, Any]) -> np.ndarray:
    raw = raw_input_matrix(rows, list(transform["input_names"]))
    value = (raw - np.asarray(transform["input_mean"])) / np.asarray(transform["input_scale"])
    return np.clip(value, -8.0, 8.0)


def transform_targets(rows: list[dict[str, str]], transform: dict[str, Any]) -> np.ndarray:
    h0_calls, h0_bytes = histogram_arrays(rows, "h0")
    target_calls, target_bytes = histogram_arrays(rows, "target")
    residual = np.log1p(np.concatenate([target_calls, target_bytes], axis=1)) - np.log1p(np.concatenate([h0_calls, h0_bytes], axis=1))
    return residual / np.asarray(transform["output_scale"])


def init_model(input_dim: int, width: int, depth: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    dims = [input_dim] + [width] * depth + [2 * BIN_COUNT]
    weights = []
    biases = []
    for left, right in zip(dims[:-1], dims[1:]):
        limit = math.sqrt(6.0 / (left + right))
        weights.append(rng.uniform(-limit, limit, size=(left, right)))
        biases.append(np.zeros(right, dtype=np.float64))
    return {"weights": weights, "biases": biases}


def forward(model: dict[str, Any], x: np.ndarray, *, cache: bool = False) -> Any:
    activations = [x]
    value = x
    for index, (weight, bias) in enumerate(zip(model["weights"], model["biases"])):
        value = value @ weight + bias
        if index + 1 < len(model["weights"]):
            value = np.tanh(value)
        activations.append(value)
    return (value, activations) if cache else value


def fit_model(x: np.ndarray, y: np.ndarray, config: dict[str, Any], seed: int, *, validation: tuple[np.ndarray, np.ndarray] | None = None, fixed_epochs: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    model = init_model(x.shape[1], int(config["width"]), int(config["depth"]), seed)
    m_w = [np.zeros_like(value) for value in model["weights"]]; v_w = [np.zeros_like(value) for value in model["weights"]]
    m_b = [np.zeros_like(value) for value in model["biases"]]; v_b = [np.zeros_like(value) for value in model["biases"]]
    lr = float(config["learning_rate"]); wd = float(config["weight_decay"])
    maximum = int(fixed_epochs if fixed_epochs is not None else config["max_epochs"])
    patience = int(config["patience"])
    best_model = copy.deepcopy(model); best_loss = math.inf; best_epoch = 0; stale = 0
    for epoch in range(1, maximum + 1):
        prediction, activations = forward(model, x, cache=True)
        delta = 2.0 * (prediction - y) / (x.shape[0] * y.shape[1])
        grad_w: list[np.ndarray] = [np.empty(0)] * len(model["weights"])
        grad_b: list[np.ndarray] = [np.empty(0)] * len(model["biases"])
        for layer in range(len(model["weights"]) - 1, -1, -1):
            grad_w[layer] = activations[layer].T @ delta + wd * model["weights"][layer]
            grad_b[layer] = delta.sum(axis=0)
            if layer:
                delta = (delta @ model["weights"][layer].T) * (1.0 - activations[layer] ** 2)
        for layer in range(len(model["weights"])):
            m_w[layer] = 0.9 * m_w[layer] + 0.1 * grad_w[layer]
            v_w[layer] = 0.999 * v_w[layer] + 0.001 * grad_w[layer] ** 2
            m_b[layer] = 0.9 * m_b[layer] + 0.1 * grad_b[layer]
            v_b[layer] = 0.999 * v_b[layer] + 0.001 * grad_b[layer] ** 2
            correction1 = 1.0 - 0.9 ** epoch; correction2 = 1.0 - 0.999 ** epoch
            model["weights"][layer] -= lr * (m_w[layer] / correction1) / (np.sqrt(v_w[layer] / correction2) + 1e-8)
            model["biases"][layer] -= lr * (m_b[layer] / correction1) / (np.sqrt(v_b[layer] / correction2) + 1e-8)
        if validation is None:
            best_model = copy.deepcopy(model); best_epoch = epoch
            continue
        val_prediction = forward(model, validation[0])
        val_loss = float(np.mean((val_prediction - validation[1]) ** 2))
        if val_loss < best_loss - 1e-8:
            best_loss = val_loss; best_epoch = epoch; best_model = copy.deepcopy(model); stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    training_loss = float(np.mean((forward(best_model, x) - y) ** 2))
    return best_model, {"best_epoch": best_epoch, "best_validation_loss": None if validation is None else best_loss, "training_loss": training_loss, "epochs_ran": epoch}


def model_to_json(model: dict[str, Any]) -> dict[str, Any]:
    return {"weights": [value.tolist() for value in model["weights"]], "biases": [value.tolist() for value in model["biases"]]}


def model_from_json(value: dict[str, Any]) -> dict[str, Any]:
    return {"weights": [np.asarray(item, dtype=np.float64) for item in value["weights"]], "biases": [np.asarray(item, dtype=np.float64) for item in value["biases"]]}


def predict_histograms(rows: list[dict[str, str]], transform: dict[str, Any], models: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    x = transform_inputs(rows, transform)
    mean_normalized_residual = np.mean([forward(model, x) for model in models], axis=0)
    residual = mean_normalized_residual * np.asarray(transform["output_scale"])
    h0_calls, h0_bytes = histogram_arrays(rows, "h0")
    h0_log = np.log1p(np.concatenate([h0_calls, h0_bytes], axis=1))
    prediction = np.expm1(np.clip(h0_log + residual, 0.0, 40.0))
    return prediction[:, :BIN_COUNT], prediction[:, BIN_COUNT:]
