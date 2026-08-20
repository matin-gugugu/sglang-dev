#!/usr/bin/env python3
"""NumPy metric-aligned, total-preserving shape DNN for Phase59."""
from __future__ import annotations

import copy
import math
from typing import Any, Callable

import numpy as np

BIN_COUNT = 12
OUTPUT_SIZE = 24


def init_model(input_dim: int, width: int, depth: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed); dims = [input_dim] + [width] * depth + [OUTPUT_SIZE]; weights = []; biases = []
    for left, right in zip(dims[:-1], dims[1:]):
        limit = math.sqrt(6.0 / (left + right)); weights.append(rng.uniform(-limit, limit, size=(left, right))); biases.append(np.zeros(right, dtype=np.float64))
    return {"weights": weights, "biases": biases}


def forward(model: dict[str, Any], x: np.ndarray, *, cache: bool = False):
    activations = [x]; value = x
    for weight, bias in zip(model["weights"], model["biases"]):
        value = np.tanh(value @ weight + bias); activations.append(value)
    return (value, activations) if cache else value


def model_to_json(model: dict[str, Any]) -> dict[str, Any]:
    return {"weights": [value.tolist() for value in model["weights"]], "biases": [value.tolist() for value in model["biases"]]}


def model_from_json(value: dict[str, Any]) -> dict[str, Any]:
    return {"weights": [np.asarray(item, dtype=np.float64) for item in value["weights"]], "biases": [np.asarray(item, dtype=np.float64) for item in value["biases"]]}


def softmax(logits: np.ndarray) -> np.ndarray:
    stable = logits - logits.max(axis=1, keepdims=True); value = np.exp(np.clip(stable, -50.0, 0.0)); return value / np.maximum(value.sum(axis=1, keepdims=True), 1e-12)


def decode_shape(rows: list[dict[str, str]], normalized: np.ndarray, residual_bound: float, histogram_arrays: Callable) -> tuple[np.ndarray, np.ndarray]:
    h0_calls, h0_bytes = histogram_arrays(rows, "h0"); outputs = []
    for offset, h0 in ((0, h0_calls), (BIN_COUNT, h0_bytes)):
        total = np.maximum(h0.sum(axis=1), 0.0); smooth = np.maximum(total, 1.0)[:, None] * 1e-6 / BIN_COUNT
        base_share = (h0 + smooth) / np.maximum(total[:, None] + smooth * BIN_COUNT, 1e-12)
        share = softmax(np.log(np.maximum(base_share, 1e-12)) + float(residual_bound) * normalized[:, offset:offset + BIN_COUNT])
        outputs.append(np.maximum(total[:, None] * share, 0.0))
    return outputs[0], outputs[1]


def _soft_l1(value: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray]:
    scale = np.sqrt(value * value + delta * delta); return scale - delta, value / np.maximum(scale, 1e-12)


def objective_and_gradient(rows: list[dict[str, str]], normalized: np.ndarray, config: dict[str, Any], histogram_arrays: Callable, *, gradient: bool) -> tuple[float, np.ndarray | None, dict[str, float]]:
    bound = float(config["residual_bound"]); calls, bytes_ = decode_shape(rows, normalized, bound, histogram_arrays); target_calls, target_bytes = histogram_arrays(rows, "target")
    h0_calls, h0_bytes = histogram_arrays(rows, "h0"); row_weight = np.ones(len(rows), dtype=np.float64); focus = config.get("segment_focus")
    if focus:
        row_weight *= np.asarray([float(config.get("segment_focus_weight", 1.75)) if row["segment"] == focus else 1.0 for row in rows])
    row_weight /= max(float(row_weight.mean()), 1e-12); grad_residual = np.zeros_like(normalized); total_loss = 0.0
    for kind, offset, predicted, target, h0, weight_name in (
        ("calls", 0, calls, target_calls, h0_calls, "calls_wape_weight"),
        ("bytes", BIN_COUNT, bytes_, target_bytes, h0_bytes, "bytes_wape_weight"),
    ):
        positive = target[target > 0]; delta = max(float(np.median(positive) if len(positive) else 1.0) * 1e-4, 1e-8)
        smooth, direction = _soft_l1(predicted - target, delta); denominator = max(float((row_weight[:, None] * target).sum()), 1e-12); loss_weight = float(config[weight_name])
        total_loss += loss_weight * float((row_weight[:, None] * smooth).sum() / denominator)
        grad_predicted = loss_weight * row_weight[:, None] * direction / denominator
        predicted_total = np.maximum(predicted.sum(axis=1, keepdims=True), 1e-12); share = predicted / predicted_total
        grad_share = grad_predicted * np.maximum(h0.sum(axis=1, keepdims=True), 0.0)
        if kind == "calls":
            target_share = target / np.maximum(target.sum(axis=1, keepdims=True), 1e-12); diff = share - target_share; tv_delta = 1e-5
            tv_smooth, tv_direction = _soft_l1(diff, tv_delta); tv_weight = float(config["tv_weight"])
            total_loss += tv_weight * float((row_weight[:, None] * tv_smooth).sum() / (2.0 * max(float(row_weight.sum()), 1e-12)))
            grad_share += tv_weight * row_weight[:, None] * tv_direction / (2.0 * max(float(row_weight.sum()), 1e-12))
            cumulative = np.cumsum(diff, axis=1); emd_smooth, emd_direction = _soft_l1(cumulative, tv_delta); emd_weight = float(config["emd_weight"])
            total_loss += emd_weight * float((row_weight[:, None] * emd_smooth).sum() / (11.0 * max(float(row_weight.sum()), 1e-12)))
            reverse = np.flip(np.cumsum(np.flip(emd_direction, axis=1), axis=1), axis=1)
            grad_share += emd_weight * row_weight[:, None] * reverse / (11.0 * max(float(row_weight.sum()), 1e-12))
        grad_logits = share * (grad_share - (grad_share * share).sum(axis=1, keepdims=True)); grad_residual[:, offset:offset + BIN_COUNT] = grad_logits
    regularization = float(config.get("residual_l2", 1e-4)); total_loss += regularization * float(np.mean(normalized * normalized))
    if gradient:
        grad_normalized = bound * grad_residual + regularization * 2.0 * normalized / max(normalized.size, 1)
    else:
        grad_normalized = None
    def wape(predicted: np.ndarray, target: np.ndarray) -> float:
        return float(np.abs(predicted - target).sum() / max(float(target.sum()), 1e-12))
    calls_shape = calls / np.maximum(calls.sum(axis=1, keepdims=True), 1e-12); target_shape = target_calls / np.maximum(target_calls.sum(axis=1, keepdims=True), 1e-12)
    metrics = {
        "calls_histogram_wape": wape(calls, target_calls), "bytes_histogram_wape": wape(bytes_, target_bytes),
        "calls_tv": float((0.5 * np.abs(calls_shape - target_shape).sum(axis=1)).mean()),
        "calls_emd": float((np.abs(np.cumsum(calls_shape, axis=1) - np.cumsum(target_shape, axis=1)).sum(axis=1) / 11.0).mean()),
    }
    return total_loss, grad_normalized, metrics


def fit_shape_model(x: np.ndarray, rows: list[dict[str, str]], config: dict[str, Any], seed: int, histogram_arrays: Callable, *, validation: tuple[np.ndarray, list[dict[str, str]]] | None = None, fixed_epochs: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    model = init_model(x.shape[1], int(config["width"]), int(config["depth"]), seed); m_w = [np.zeros_like(v) for v in model["weights"]]; v_w = [np.zeros_like(v) for v in model["weights"]]; m_b = [np.zeros_like(v) for v in model["biases"]]; v_b = [np.zeros_like(v) for v in model["biases"]]
    lr = float(config["learning_rate"]); wd = float(config["weight_decay"]); maximum = int(fixed_epochs if fixed_epochs is not None else config["max_epochs"]); eval_every = int(config.get("eval_every", 5)); patience = int(config.get("patience_evals", 24)); best = copy.deepcopy(model); best_loss = math.inf; best_epoch = 0; stale = 0
    for epoch in range(1, maximum + 1):
        normalized, activations = forward(model, x, cache=True); training_loss, gradient, _ = objective_and_gradient(rows, normalized, config, histogram_arrays, gradient=True); assert gradient is not None
        delta = gradient * (1.0 - normalized * normalized); grad_w = [np.empty(0)] * len(model["weights"]); grad_b = [np.empty(0)] * len(model["biases"])
        for layer in range(len(model["weights"]) - 1, -1, -1):
            grad_w[layer] = activations[layer].T @ delta + wd * model["weights"][layer]; grad_b[layer] = delta.sum(axis=0)
            if layer:
                delta = (delta @ model["weights"][layer].T) * (1.0 - activations[layer] * activations[layer])
        for layer in range(len(model["weights"])):
            m_w[layer] = 0.9 * m_w[layer] + 0.1 * grad_w[layer]; v_w[layer] = 0.999 * v_w[layer] + 0.001 * grad_w[layer] ** 2; m_b[layer] = 0.9 * m_b[layer] + 0.1 * grad_b[layer]; v_b[layer] = 0.999 * v_b[layer] + 0.001 * grad_b[layer] ** 2
            correction1 = 1.0 - 0.9 ** epoch; correction2 = 1.0 - 0.999 ** epoch
            model["weights"][layer] -= lr * (m_w[layer] / correction1) / (np.sqrt(v_w[layer] / correction2) + 1e-8); model["biases"][layer] -= lr * (m_b[layer] / correction1) / (np.sqrt(v_b[layer] / correction2) + 1e-8)
        if fixed_epochs is not None:
            best = copy.deepcopy(model); best_epoch = epoch; continue
        if epoch % eval_every:
            continue
        if validation is None:
            validation_loss = training_loss
        else:
            validation_loss, _, _ = objective_and_gradient(validation[1], forward(model, validation[0]), config, histogram_arrays, gradient=False)
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss; best_epoch = epoch; best = copy.deepcopy(model); stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    final_loss, _, final_metrics = objective_and_gradient(rows, forward(best, x), config, histogram_arrays, gradient=False)
    return best, {"best_epoch": best_epoch, "best_validation_loss": None if validation is None else best_loss, "training_loss": final_loss, "training_metrics": final_metrics, "epochs_ran": epoch}
