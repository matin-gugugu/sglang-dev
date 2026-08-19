#!/usr/bin/env python3
"""Phase54 histogram metrics; formulas remain compatible with Phase42/50."""

from __future__ import annotations

from typing import Any

import numpy as np


SCORE_KEYS = (
    "calls_histogram_wape",
    "bytes_histogram_wape",
    "mean_calls_histogram_tv",
    "mean_normalized_log_payload_emd",
)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, 1e-12))


def metric_bundle(pred_calls: np.ndarray, pred_bytes: np.ndarray, target_calls: np.ndarray, target_bytes: np.ndarray) -> dict[str, float]:
    arrays = (pred_calls, pred_bytes, target_calls, target_bytes)
    if any(value.ndim != 2 or value.shape[1] != 12 or not np.isfinite(value).all() for value in arrays):
        raise ValueError("histogram arrays must be finite Nx12 matrices")
    if any((value < -1e-9).any() for value in arrays):
        raise ValueError("histogram arrays must be nonnegative")
    pred_calls_total = pred_calls.sum(axis=1)
    target_calls_total = target_calls.sum(axis=1)
    pred_bytes_total = pred_bytes.sum(axis=1)
    target_bytes_total = target_bytes.sum(axis=1)
    pred_shape = pred_calls / np.maximum(pred_calls_total[:, None], 1e-12)
    target_shape = target_calls / np.maximum(target_calls_total[:, None], 1e-12)
    tv = 0.5 * np.abs(pred_shape - target_shape).sum(axis=1)
    emd = np.abs(np.cumsum(pred_shape, axis=1) - np.cumsum(target_shape, axis=1)).sum(axis=1) / 11.0
    return {
        "profiles": int(pred_calls.shape[0]),
        "calls_histogram_wape": _safe_ratio(float(np.abs(pred_calls - target_calls).sum()), float(target_calls.sum())),
        "bytes_histogram_wape": _safe_ratio(float(np.abs(pred_bytes - target_bytes).sum()), float(target_bytes.sum())),
        "calls_total_wape": _safe_ratio(float(np.abs(pred_calls_total - target_calls_total).sum()), float(target_calls_total.sum())),
        "bytes_total_wape": _safe_ratio(float(np.abs(pred_bytes_total - target_bytes_total).sum()), float(target_bytes_total.sum())),
        "mean_profile_calls_l1": float((np.abs(pred_calls - target_calls).sum(axis=1) / np.maximum(target_calls_total, 1e-12)).mean()),
        "mean_profile_bytes_l1": float((np.abs(pred_bytes - target_bytes).sum(axis=1) / np.maximum(target_bytes_total, 1e-12)).mean()),
        "mean_calls_histogram_tv": float(tv.mean()),
        "mean_normalized_log_payload_emd": float(emd.mean()),
    }


def compare_to_h0(candidate: dict[str, float], h0: dict[str, float]) -> dict[str, Any]:
    ratios = {key: _safe_ratio(float(candidate[key]), float(h0[key])) for key in SCORE_KEYS}
    return {
        "metric_ratios_to_h0": ratios,
        "composite_ratio": float(np.mean(list(ratios.values()))),
        "outcome": "IMPROVES_COMPOSITE" if all(value < 1.0 for value in ratios.values()) else "DOES_NOT_IMPROVE_COMPOSITE",
    }


def target_gate(metrics: dict[str, float], *, histogram_limit: float, total_limit: float) -> bool:
    return (
        float(metrics["calls_histogram_wape"]) <= histogram_limit
        and float(metrics["bytes_histogram_wape"]) <= histogram_limit
        and float(metrics["calls_total_wape"]) <= total_limit
        and float(metrics["bytes_total_wape"]) <= total_limit
    )
