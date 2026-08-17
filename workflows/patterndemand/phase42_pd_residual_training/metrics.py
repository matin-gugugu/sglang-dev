#!/usr/bin/env python3
"""Phase42/43 topology-independent histogram metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, 1e-12))


def metric_bundle(pred_calls: np.ndarray, pred_bytes: np.ndarray, target_calls: np.ndarray, target_bytes: np.ndarray) -> dict[str, float]:
    for value in (pred_calls, pred_bytes, target_calls, target_bytes):
        if value.ndim != 2 or value.shape[1] != 12 or not np.isfinite(value).all():
            raise ValueError("histogram arrays must be finite Nx12 matrices")
        if (value < -1e-9).any():
            raise ValueError("histogram arrays must be nonnegative")
    calls_total_pred = pred_calls.sum(axis=1)
    calls_total_target = target_calls.sum(axis=1)
    bytes_total_pred = pred_bytes.sum(axis=1)
    bytes_total_target = target_bytes.sum(axis=1)
    pred_shape = pred_calls / np.maximum(calls_total_pred[:, None], 1e-12)
    target_shape = target_calls / np.maximum(calls_total_target[:, None], 1e-12)
    tv = 0.5 * np.abs(pred_shape - target_shape).sum(axis=1)
    emd = np.abs(np.cumsum(pred_shape, axis=1) - np.cumsum(target_shape, axis=1)).sum(axis=1) / 11.0
    profile_calls_l1 = np.abs(pred_calls - target_calls).sum(axis=1) / np.maximum(calls_total_target, 1e-12)
    profile_bytes_l1 = np.abs(pred_bytes - target_bytes).sum(axis=1) / np.maximum(bytes_total_target, 1e-12)
    return {
        "profiles": int(pred_calls.shape[0]),
        "calls_histogram_wape": _safe_ratio(float(np.abs(pred_calls - target_calls).sum()), float(target_calls.sum())),
        "bytes_histogram_wape": _safe_ratio(float(np.abs(pred_bytes - target_bytes).sum()), float(target_bytes.sum())),
        "calls_total_wape": _safe_ratio(float(np.abs(calls_total_pred - calls_total_target).sum()), float(calls_total_target.sum())),
        "bytes_total_wape": _safe_ratio(float(np.abs(bytes_total_pred - bytes_total_target).sum()), float(bytes_total_target.sum())),
        "mean_profile_calls_l1": float(profile_calls_l1.mean()),
        "mean_profile_bytes_l1": float(profile_bytes_l1.mean()),
        "mean_calls_histogram_tv": float(tv.mean()),
        "mean_normalized_log_payload_emd": float(emd.mean()),
    }


SCORE_KEYS = (
    "calls_histogram_wape",
    "bytes_histogram_wape",
    "mean_calls_histogram_tv",
    "mean_normalized_log_payload_emd",
)


def compare_to_h0(dnn: dict[str, float], h0: dict[str, float]) -> dict[str, Any]:
    ratios = {key: _safe_ratio(float(dnn[key]), float(h0[key])) for key in SCORE_KEYS}
    composite = float(np.mean(list(ratios.values())))
    improves = (
        composite < 1.0
        and ratios["calls_histogram_wape"] <= 1.25
        and ratios["bytes_histogram_wape"] <= 1.25
    )
    return {
        "metric_ratios_to_h0": ratios,
        "composite_ratio": composite,
        "outcome": "IMPROVES_COMPOSITE" if improves else "DOES_NOT_IMPROVE_COMPOSITE",
    }
