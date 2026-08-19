#!/usr/bin/env python3
"""Phase57: bounded, OOF-only, multi-family iterative PD histogram optimization."""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
P54 = HERE.parent / "phase54_pd_histogram_accuracy_refinement"
P56 = HERE.parent / "phase56_pd_structural_histogram_search"
sys.path.insert(0, str(P54)); sys.path.insert(0, str(P56)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

from common import environment_record, load_json, refresh_manifest, repo_root, utc_now, write_json  # noqa: E402
from model_loader import read_csv_gz  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


P54RUN = _load("phase57_p54_run", P54 / "run.py")
P56RUN = _load("phase57_p56_run", P56 / "run.py")
PREFLIGHT = _load("phase57_preflight", HERE / "preflight.py")

MODEL_IDS = P54RUN.MODEL_IDS
SEGMENTS = P54RUN.SEGMENTS
BIN_COUNT = 12
ENCODED = 26
SCORE_KEYS = P54RUN.SCORE_KEYS
ARRIVAL_TOKENS = ("_rps", "interarrival", "peak_to_mean", "fano")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def score(metrics: dict[str, float]) -> float:
    return float(np.mean([float(metrics[key]) for key in SCORE_KEYS]))


def scope_key(row: dict[str, str], scope: str) -> str:
    if scope in ("global", "shared"):
        return "global"
    if scope == "model":
        return f"model::{row['model']}"
    if scope == "segment":
        return f"segment::{row['segment']}"
    if scope == "model_segment":
        return f"model_segment::{row['model']}::{row['segment']}"
    raise ValueError(scope)


def groups_for(rows: list[dict[str, str]], scope: str) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        result.setdefault(scope_key(row, scope), []).append(index)
    return result


def target_residual(rows: list[dict[str, str]], representation: str = "residual") -> np.ndarray:
    h0_calls, h0_bytes = P54RUN.histogram_arrays(rows, "h0")
    target_calls, target_bytes = P54RUN.histogram_arrays(rows, "target")
    value = P54RUN.encode_histograms(target_calls, target_bytes) - P54RUN.encode_histograms(h0_calls, h0_bytes)
    if representation == "direct_shape":
        value[:, 0] = 0.0; value[:, BIN_COUNT + 1] = 0.0
    elif representation != "residual":
        raise ValueError(representation)
    return value


def feature_matrix(rows: list[dict[str, str]], mode: str) -> tuple[np.ndarray, list[str]]:
    names = sorted(name for name in rows[0] if name.startswith("feature_") or name.startswith("h0_"))
    names = [name for name in names if not any(token in name for token in ARRIVAL_TOKENS)]
    values = []
    for row in rows:
        values.append([np.log1p(max(float(row[name]), 0.0)) if name.startswith("h0_") else float(row[name]) for name in names])
    base = np.asarray(values, dtype=np.float64)
    columns = [base]; labels = list(names)
    if mode in ("causal_with_h0_shape", "causal_structural_interactions"):
        columns.extend([
            np.asarray([[float(row["model"] == model) for model in MODEL_IDS] for row in rows]),
            np.asarray([[float(row["segment"] == segment) for segment in SEGMENTS] for row in rows]),
        ])
        labels.extend([f"cat_model_{model}" for model in MODEL_IDS] + [f"cat_segment_{segment}" for segment in SEGMENTS])
    if mode == "causal_structural_interactions":
        selected = [index for index, name in enumerate(names) if (
            name.startswith("h0_") or name.startswith("feature_profile_") or name in {
                "feature_model_hidden_size", "feature_model_num_hidden_layers", "feature_model_kv_bytes_per_token",
                "feature_pd_chunk_tokens", "feature_pd_page_size_tokens", "feature_pd_wave_size"
            }
        )]
        selected = selected[:64]
        interaction = base[:, selected] ** 2
        columns.append(interaction); labels.extend([f"sq_{names[index]}" for index in selected])
    matrix = np.concatenate(columns, axis=1)
    if not np.isfinite(matrix).all():
        raise ValueError("non-finite causal feature matrix")
    return matrix, labels


def fit_ridge(x: np.ndarray, y: np.ndarray, l2: float) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    mean = x.mean(axis=0); scale = x.std(axis=0); scale[scale < 1e-10] = 1.0
    z = np.clip((x - mean) / scale, -10.0, 10.0)
    design = np.c_[np.ones(len(z)), z]
    gram = design.T @ design; gram.flat[:: gram.shape[0] + 1] += float(l2)
    try:
        weights = np.linalg.solve(gram, design.T @ y)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(gram) @ design.T @ y
    return {"input_mean": mean.tolist(), "input_scale": scale.tolist(), "weights": weights.tolist()}, design @ weights, (mean, scale)


def ridge_predict(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    mean = np.asarray(model["input_mean"], dtype=np.float64); scale = np.asarray(model["input_scale"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    z = np.clip((x - mean) / np.maximum(scale, 1e-10), -10.0, 10.0)
    return np.c_[np.ones(len(z)), z] @ weights


def ridge_oof(train: list[dict[str, str]], config: dict[str, Any], folds: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    residual = np.zeros((len(train), ENCODED), dtype=np.float64)
    support_calls = np.ones((len(train), BIN_COUNT), dtype=bool); support_bytes = np.ones_like(support_calls)
    epochs: list[int] = []
    for fold in range(4):
        fit_indices = [i for i, row in enumerate(train) if folds[row["profile_id"]] != fold]
        hold_indices = [i for i, row in enumerate(train) if folds[row["profile_id"]] == fold]
        groups = groups_for(train, config["scope"])
        for group, group_indices in sorted(groups.items()):
            fit_i = [i for i in group_indices if i in set(fit_indices)]; hold_i = [i for i in group_indices if i in set(hold_indices)]
            if not hold_i:
                continue
            fit_rows = [train[i] for i in fit_i]; hold_rows = [train[i] for i in hold_i]
            x_fit, _ = feature_matrix(fit_rows, config["feature_mode"]); x_hold, _ = feature_matrix(hold_rows, config["feature_mode"])
            y_fit = target_residual(fit_rows, config["representation"])
            fitted, fit_prediction, _ = fit_ridge(x_fit, y_fit, float(config["l2"]))
            hold_prediction = ridge_predict(fitted, x_hold)
            if config.get("calibration", False):
                hold_prediction += float(config.get("calibration_strength", 0.5)) * np.mean(y_fit - fit_prediction, axis=0)
            residual[hold_i] = np.clip(hold_prediction, -2.0, 2.0)
            if config.get("support_aware", False):
                fit_calls, fit_bytes = P54RUN.histogram_arrays(fit_rows, "target")
                support_calls[hold_i] = (fit_calls > 1e-9).mean(axis=0)[None, :] >= float(config.get("support_threshold", 0.08))
                support_bytes[hold_i] = (fit_bytes > 1e-9).mean(axis=0)[None, :] >= float(config.get("support_threshold", 0.08))
            epochs.append(1)
    return residual, support_calls, support_bytes, epochs


def support_project(calls: np.ndarray, bytes_: np.ndarray, masks_calls: np.ndarray, masks_bytes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    calls = np.where(masks_calls, calls, 0.0); bytes_ = np.where(masks_bytes, bytes_, 0.0)
    # Keep the predicted total while projecting onto fit-fold-observed support.
    for value, masks in ((calls, masks_calls), (bytes_, masks_bytes)):
        total_before = np.maximum(value.sum(axis=1), 1e-12)
        value[masks.sum(axis=1) == 0] = 0.0
        value *= (total_before / np.maximum(value.sum(axis=1), 1e-12))[:, None]
    return np.maximum(calls, 0.0), np.maximum(bytes_, 0.0)


def apply_alpha(rows: list[dict[str, str]], calls: np.ndarray, bytes_: np.ndarray, alpha_map: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    h0_calls, h0_bytes = P54RUN.histogram_arrays(rows, "h0"); base = P54RUN.encode_histograms(h0_calls, h0_bytes)
    predicted = P54RUN.encode_histograms(calls, bytes_)
    alpha = np.asarray([float(alpha_map[row["model"]]) for row in rows], dtype=np.float64)[:, None]
    return P54RUN.decode_histograms(base + alpha * (predicted - base))


def bias_rows(rows: list[dict[str, str]], calls: np.ndarray, bytes_: np.ndarray, candidate_id: str, round_index: int) -> list[dict[str, Any]]:
    target_calls, target_bytes = P54RUN.histogram_arrays(rows, "target"); output: list[dict[str, Any]] = []
    for kind, predicted, target in (("calls", calls, target_calls), ("logical_bytes", bytes_, target_bytes)):
        for index in range(BIN_COUNT):
            target_sum = float(target[:, index].sum()); predicted_sum = float(predicted[:, index].sum())
            output.append({"round": round_index, "candidate_id": candidate_id, "kind": kind, "bin": index, "target_sum": target_sum, "predicted_sum": predicted_sum, "signed_bias": predicted_sum - target_sum, "relative_signed_bias": (predicted_sum - target_sum) / max(target_sum, 1e-12)})
    return output


def seed_candidates(round_index: int, signal: dict[str, Any]) -> list[dict[str, Any]]:
    tag = f"r{round_index}"
    focus = "tail_shape_focus" if signal.get("tail_abs_bias", 0.0) >= signal.get("head_abs_bias", 0.0) else "shape_focus"
    mlp = [
        ("model", "model", "none", 0.0, focus, "full_target_free", 96, 2, 0.004),
        ("model_segment", "model_segment", "head_residual", 0.5, focus, "full_target_free", 96, 2, 0.004),
        ("model_segment", "model_segment", "head_residual", 0.75, "tail_shape_focus", "fixed_draining_causal", 128, 2, 0.003),
        ("segment", "model_segment", "head_residual", 0.5, "shape_focus", "full_target_free", 128, 2, 0.003),
    ]
    ridge = [
        ("global", "causal_with_h0_shape", "residual", 0.1, False, False),
        ("model", "causal_with_h0_shape", "residual", 1.0, True, False),
        ("model_segment", "causal_with_h0_shape", "residual", 10.0, True, False),
        ("model_segment", "causal_structural_interactions", "residual", 10.0, True, False),
        ("model_segment", "causal_structural_interactions", "direct_shape", 10.0, True, True),
        ("segment", "causal_structural_interactions", "residual", 100.0, True, True),
        ("model", "causal_structural_interactions", "residual", 10.0, False, True),
        ("global", "causal_structural_interactions", "direct_shape", 100.0, True, True),
        ("segment", "causal_with_h0_shape", "residual", 1.0, True, False),
        ("model", "causal_with_h0_shape", "direct_shape", 10.0, True, True),
        ("model_segment", "causal_with_h0_shape", "direct_shape", 1.0, False, False),
        ("global", "causal_structural_interactions", "residual", 10.0, True, False),
    ]
    output: list[dict[str, Any]] = []
    for index, (head, alpha_scope, calibration, strength, loss, feature, width, depth, lr) in enumerate(mlp):
        output.append({"family": "phase56_mlp", "candidate_id": f"p57_{tag}_mlp_{index:02d}", "head_scope": head, "alpha_scope": alpha_scope, "calibration_mode": calibration, "calibration_strength": strength, "loss_mode": loss, "feature_mode": feature, "width": width, "depth": depth, "learning_rate": lr, "weight_decay": 0.002, "max_epochs": 520, "patience": 90, "stage": "seed", "round": round_index})
    for index, (scope, feature, representation, l2, calibration, support) in enumerate(ridge):
        output.append({"family": "ridge", "candidate_id": f"p57_{tag}_ridge_{index:02d}", "scope": scope, "feature_mode": feature, "representation": representation, "l2": l2, "calibration": calibration, "calibration_strength": 0.5, "support_aware": support, "support_threshold": 0.08, "stage": "seed", "round": round_index})
    return output


def adaptive_candidates(top: list[dict[str, Any]], round_index: int, signal: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for parent_index, parent in enumerate(top[:4]):
        cfg = copy.deepcopy(parent["config"]); cfg["stage"] = "adaptive"; cfg["round"] = round_index
        if cfg["family"] == "phase56_mlp":
            cfg["candidate_id"] = f"p57_r{round_index}_from_{parent_index}_mlp_adapt"
            cfg["head_scope"] = "model_segment"; cfg["alpha_scope"] = "model_segment"
            cfg["loss_mode"] = "tail_shape_focus" if signal.get("tail_abs_bias", 0.0) >= signal.get("head_abs_bias", 0.0) else "shape_focus"
            cfg["width"] = min(192, int(cfg["width"]) + 32); cfg["depth"] = min(3, int(cfg["depth"]) + 1); cfg["max_epochs"] = 760; cfg["patience"] = 140
        elif cfg["family"] == "ridge":
            cfg["candidate_id"] = f"p57_r{round_index}_from_{parent_index}_ridge_adapt"
            cfg["scope"] = "model_segment"; cfg["feature_mode"] = "causal_structural_interactions"; cfg["l2"] = float(cfg["l2"]) * (10.0 if parent_index % 2 else 0.1)
            cfg["support_aware"] = True; cfg["support_threshold"] = 0.03 if signal.get("support_gap", 0.0) > 0.02 else 0.12; cfg["calibration"] = True
        else:
            cfg["candidate_id"] = f"p57_r{round_index}_from_{parent_index}_blend_adapt"
            cfg["weights"] = [0.8, 0.2] if len(cfg.get("parent_ids", [])) == 2 else [1.0]
        output.append(cfg)
        blend = {"family": "blend", "candidate_id": f"p57_r{round_index}_from_{parent_index}_blend", "stage": "adaptive", "round": round_index, "parent_ids": [parent["config"]["candidate_id"]], "weights": [0.5], "alpha_scope": "model", "support_aware": False}
        if parent_index + 1 < len(top):
            blend["parent_ids"].append(top[parent_index + 1]["config"]["candidate_id"]); blend["weights"] = [0.65, 0.35]
        output.append(blend)
    return output


def signal_from(results: list[dict[str, Any]]) -> dict[str, Any]:
    selected = sorted(results, key=lambda value: value["sort"])[0]
    values = selected["bias"]
    head = [abs(float(row["relative_signed_bias"])) for row in values if int(row["bin"]) < 4]
    tail = [abs(float(row["relative_signed_bias"])) for row in values if int(row["bin"]) >= 8]
    support_gap = float(np.mean([abs(float(row["relative_signed_bias"])) for row in values if int(row["bin"]) in (2, 3, 4, 9)])) if values else 0.0
    return {"selected_candidate_id": selected["config"]["candidate_id"], "head_abs_bias": float(np.mean(head) if head else 0.0), "tail_abs_bias": float(np.mean(tail) if tail else 0.0), "support_gap": support_gap, "model_failures": sum(not value["target_guard"] for value in selected["models"].values()), "segment_failures": sum(not value["target_guard"] for value in selected["segments"].values())}


def phase56_oof(train: list[dict[str, str]], config: dict[str, Any], folds: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    raw, calibrated, epochs = P56RUN.fit_predict_oof(train, config, folds)
    return calibrated, np.ones((len(train), BIN_COUNT), dtype=bool), np.ones((len(train), BIN_COUNT), dtype=bool), epochs


def evaluate_candidate(train: list[dict[str, str]], config: dict[str, Any], folds: dict[str, int], contract: dict[str, Any], parent_predictions: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    if config["family"] == "phase56_mlp":
        residual, mask_calls, mask_bytes, epochs = phase56_oof(train, config, folds)
        base_calls, base_bytes = P54RUN.decode_histograms(P54RUN.encode_histograms(*P54RUN.histogram_arrays(train, "h0")) + residual)
    elif config["family"] == "ridge":
        residual, mask_calls, mask_bytes, epochs = ridge_oof(train, config, folds)
        h0_calls, h0_bytes = P54RUN.histogram_arrays(train, "h0")
        base_calls, base_bytes = P54RUN.decode_histograms(P54RUN.encode_histograms(h0_calls, h0_bytes) + residual)
    elif config["family"] == "blend":
        parents = [parent_predictions[parent_id] for parent_id in config["parent_ids"]]
        weights = np.asarray(config["weights"], dtype=np.float64); weights = weights / max(weights.sum(), 1e-12)
        base_calls = sum(float(weight) * value["base_calls"] for weight, value in zip(weights, parents)); base_bytes = sum(float(weight) * value["base_bytes"] for weight, value in zip(weights, parents))
        mask_calls = np.ones((len(train), BIN_COUNT), dtype=bool); mask_bytes = np.ones_like(mask_calls); epochs = [1]
    else:
        raise ValueError(config["family"])
    h0_calls, h0_bytes = P54RUN.histogram_arrays(train, "h0")
    h0_metrics = P54RUN.metric_bundle(h0_calls, h0_bytes, *P54RUN.histogram_arrays(train, "target"))
    alpha_map, alpha_audits, _ = P54RUN.choose_alphas(train, base_calls, base_bytes, h0_metrics, [float(value) for value in contract["search_contract"]["alpha_grid"]])
    calls, bytes_ = apply_alpha(train, base_calls, base_bytes, alpha_map)
    if config.get("support_aware", False):
        calls, bytes_ = support_project(calls, bytes_, mask_calls, mask_bytes)
    overall, models, segments, target = P54RUN.development_audits(train, calls, bytes_)
    protection = P54RUN.strict_h0(overall) and all(P54RUN.strict_h0(value) for value in models.values()) and all(P54RUN.strict_h0(value) for value in segments.values())
    return {"config": config, "base_calls": base_calls, "base_bytes": base_bytes, "calls": calls, "bytes": bytes_, "epochs": epochs, "alpha_map": alpha_map, "alpha_audits": alpha_audits, "overall": overall, "models": models, "segments": segments, "oof_target": bool(target), "oof_protection": bool(protection), "bias": bias_rows(train, calls, bytes_, config["candidate_id"], int(config["round"])), "sort": (not target, not protection, score(overall["h0_plus_dnn_refined"]), config["candidate_id"])}


def fit_final_ridge(train: list[dict[str, str]], rows: list[dict[str, str]], config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    calls = np.zeros((len(rows), BIN_COUNT)); bytes_ = np.zeros_like(calls); bundle = {"family": "ridge", "scope": config["scope"], "groups": {}}
    for group, fit_indices in sorted(groups_for(train, config["scope"]).items()):
        fit_rows = [train[index] for index in fit_indices]; hold_indices = [index for index, row in enumerate(rows) if scope_key(row, config["scope"]) == group]
        if not hold_indices: continue
        hold_rows = [rows[index] for index in hold_indices]; x_fit, names = feature_matrix(fit_rows, config["feature_mode"]); x_hold, _ = feature_matrix(hold_rows, config["feature_mode"])
        fitted, _, _ = fit_ridge(x_fit, target_residual(fit_rows, config["representation"]), float(config["l2"]))
        residual = ridge_predict(fitted, x_hold); h0_calls, h0_bytes = P54RUN.histogram_arrays(hold_rows, "h0")
        pred_calls, pred_bytes = P54RUN.decode_histograms(P54RUN.encode_histograms(h0_calls, h0_bytes) + residual)
        calls[hold_indices] = pred_calls; bytes_[hold_indices] = pred_bytes; bundle["groups"][group] = {"model": fitted, "feature_names": names}
    return calls, bytes_, bundle


def fit_final_config(train: list[dict[str, str]], rows: list[dict[str, str]], config: dict[str, Any], config_map: dict[str, dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if config["family"] == "ridge":
        return fit_final_ridge(train, rows, config)
    if config["family"] == "phase56_mlp":
        epochs = int(config.get("selected_epochs", config.get("max_epochs", 520)))
        bundle = P56RUN.fit_final_bundle(train, config, epochs, [57001, 57002, 57003]); calls, bytes_ = P56RUN.predict_final_bundle(rows, bundle); return calls, bytes_, {"family": "phase56_mlp", "bundle": bundle, "epochs": epochs}
    if config["family"] == "blend":
        calls = np.zeros((len(rows), BIN_COUNT)); bytes_ = np.zeros_like(calls); weights = np.asarray(config["weights"], dtype=np.float64); weights /= max(weights.sum(), 1e-12); bundles = []
        for weight, parent_id in zip(weights, config["parent_ids"]):
            pc, pb, bundle = fit_final_config(train, rows, config_map[parent_id], config_map); calls += float(weight) * pc; bytes_ += float(weight) * pb; bundles.append(bundle)
        return calls, bytes_, {"family": "blend", "parent_ids": config["parent_ids"], "weights": weights.tolist(), "bundles": bundles}
    raise ValueError(config["family"])


def prediction_rows(rows: list[dict[str, str]], calls: np.ndarray, bytes_: np.ndarray, method: str) -> list[dict[str, Any]]:
    result = []
    for row, cv, bv in zip(rows, calls, bytes_):
        value = {name: row[name] for name in ("profile_id", "split_role", "source", "segment", "source_split", "window_id", "cutoff_ms", "model")}; value["method"] = method
        for index in range(BIN_COUNT): value[f"predicted_calls_bin_{index:02d}"] = float(cv[index]); value[f"predicted_logical_bytes_bin_{index:02d}"] = float(bv[index])
        result.append(value)
    return result


def run(expected: str, output: Path) -> dict[str, Any]:
    preflight = PREFLIGHT.run_checks(expected); contract = load_json(HERE / "experiment.json")
    if output.exists(): raise RuntimeError(f"refuse overwrite: {output}")
    source = repo_root() / contract["pinned_inputs"][1]["path"]; rows = read_csv_gz(source)
    train = [row for row in rows if row["split_role"] == "expanded_train"]; validation = [row for row in rows if row["split_role"] == "expanded_validation"]
    folds = P54RUN.fold_map(train); all_results: list[dict[str, Any]] = []; trace: list[dict[str, Any]] = []; all_bias: list[dict[str, Any]] = []; config_map: dict[str, dict[str, Any]] = {}; carry: dict[str, dict[str, np.ndarray]] = {}; signal: dict[str, Any] = {}
    rounds_completed = 0
    for round_index in range(int(contract["search_contract"]["max_rounds"])):
        seeds = seed_candidates(round_index, signal); seed_results: list[dict[str, Any]] = []
        for cfg in seeds:
            result = evaluate_candidate(train, cfg, folds, contract, carry); seed_results.append(result); config_map[cfg["candidate_id"]] = cfg; trace.append({"round": round_index, "stage": "seed", "candidate_id": cfg["candidate_id"], "family": cfg["family"], "oof_target": result["oof_target"], "oof_protection": result["oof_protection"], "oof_score": score(result["overall"]["h0_plus_dnn_refined"]), "calls_histogram_wape": result["overall"]["h0_plus_dnn_refined"]["calls_histogram_wape"], "bytes_histogram_wape": result["overall"]["h0_plus_dnn_refined"]["bytes_histogram_wape"]})
        seed_top = sorted(seed_results, key=lambda value: value["sort"])[:4]; parent_pool = {value["config"]["candidate_id"]: {"base_calls": value["base_calls"], "base_bytes": value["base_bytes"]} for value in seed_top}; carry.update(parent_pool)
        adaptive = adaptive_candidates(seed_top, round_index, signal); adaptive_results: list[dict[str, Any]] = []
        for cfg in adaptive:
            if cfg["family"] == "blend" and any(parent not in carry for parent in cfg["parent_ids"]): continue
            result = evaluate_candidate(train, cfg, folds, contract, carry); adaptive_results.append(result); config_map[cfg["candidate_id"]] = cfg; trace.append({"round": round_index, "stage": "adaptive", "candidate_id": cfg["candidate_id"], "family": cfg["family"], "parent_ids": json.dumps(cfg.get("parent_ids", []), sort_keys=True), "oof_target": result["oof_target"], "oof_protection": result["oof_protection"], "oof_score": score(result["overall"]["h0_plus_dnn_refined"]), "calls_histogram_wape": result["overall"]["h0_plus_dnn_refined"]["calls_histogram_wape"], "bytes_histogram_wape": result["overall"]["h0_plus_dnn_refined"]["bytes_histogram_wape"]})
        round_results = seed_results + adaptive_results; all_results.extend(round_results); all_bias.extend([row for value in round_results for row in value["bias"]]); rounds_completed = round_index + 1
        best = sorted(all_results, key=lambda value: value["sort"])[0]
        carry.update({value["config"]["candidate_id"]: {"base_calls": value["base_calls"], "base_bytes": value["base_bytes"]} for value in sorted(round_results, key=lambda value: value["sort"])[:4]})
        signal = signal_from(round_results)
        if best["oof_target"] and best["oof_protection"]:
            break
    hard_budget = int(contract["search_contract"]["max_total_candidates"])
    if len(all_results) > hard_budget or len(all_results) == 0: raise RuntimeError({"candidate_count": len(all_results), "hard_budget": hard_budget})
    selected = sorted(all_results, key=lambda value: value["sort"])[0]; selected_config = copy.deepcopy(selected["config"])
    selected_config["selected_epochs"] = int(np.clip(round(statistics.median(selected["epochs"])), 100, int(selected_config.get("max_epochs", 1))))
    if selected_config["family"] == "blend":
        selected_config["parent_configs"] = [copy.deepcopy(config_map[parent]) for parent in selected_config["parent_ids"]]
    final_config_map = dict(config_map); final_config_map.update({cfg["candidate_id"]: cfg for cfg in selected_config.get("parent_configs", [])})
    validation_base_calls, validation_base_bytes, final_bundle = fit_final_config(train, validation, selected_config, final_config_map)
    alpha_map = selected["alpha_map"]; validation_calls, validation_bytes = apply_alpha(validation, validation_base_calls, validation_base_bytes, alpha_map)
    overall, model_audits, segment_audits, target_met = P54RUN.development_audits(validation, validation_calls, validation_bytes)
    output.mkdir(parents=True)
    candidate_rows = [{"round": value["config"]["round"], "stage": value["config"]["stage"], "candidate_id": value["config"]["candidate_id"], "family": value["config"]["family"], "oof_target": value["oof_target"], "oof_protection": value["oof_protection"], "oof_score": score(value["overall"]["h0_plus_dnn_refined"]), "oof_composite_ratio": value["overall"]["composite_ratio"], "oof_calls_histogram_wape": value["overall"]["h0_plus_dnn_refined"]["calls_histogram_wape"], "oof_bytes_histogram_wape": value["overall"]["h0_plus_dnn_refined"]["bytes_histogram_wape"], "selected": value is selected} for value in all_results]
    write_csv(output / "analysis/round_trace.csv", trace); write_csv(output / "analysis/oof_candidate_metrics.csv", candidate_rows); write_csv(output / "analysis/oof_bin_bias.csv", all_bias)
    write_json(output / "analysis/oof_selection.json", {"selected_candidate": selected_config, "alpha_map": alpha_map, "oof_overall": selected["overall"], "oof_models": selected["models"], "oof_segments": selected["segments"], "oof_target": selected["oof_target"], "oof_protection": selected["oof_protection"], "candidate_count": len(all_results), "rounds_completed": rounds_completed, "signal": signal})
    write_csv(output / "analysis/development_validation_metrics.csv", [{"method": "h0", **overall["h0"], "composite_ratio_to_h0": 1.0, "formal_target_gate": False}, {"method": "h0_plus_dnn_iterative", **overall["h0_plus_dnn_refined"], "composite_ratio_to_h0": overall["composite_ratio"], "formal_target_gate": overall["target_gate"]}])
    write_json(output / "analysis/model_validation.json", model_audits); write_json(output / "analysis/segment_validation.json", segment_audits)
    h0_calls, h0_bytes = P54RUN.histogram_arrays(validation, "h0"); P54RUN.write_csv_gz(output / "predictions/development_validation_predictions.csv.gz", prediction_rows(validation, h0_calls, h0_bytes, "h0") + prediction_rows(validation, validation_calls, validation_bytes, "h0_plus_dnn_iterative"))
    checkpoint = {"schema_version": "phase57-pd-iterative-histogram-optimization-checkpoint-v1", "workflow_commit": expected, "selected_candidate": selected_config, "alpha_map": alpha_map, "ensemble_seeds": contract["search_contract"]["ensemble_seeds"], "bundle": final_bundle, "phase50_blind_accessed": False, "complete_requests_accessed": False}
    P54RUN.write_json_gz(output / "checkpoints/pd_iterative_histogram_optimization.json.gz", checkpoint)
    write_json(output / "audit/input_freeze.json", preflight); write_json(output / "audit/search.json", {"candidate_budget": len(all_results), "max_candidate_budget": hard_budget, "rounds_completed": rounds_completed, "selected_candidate_id": selected_config["candidate_id"], "oof_target": selected["oof_target"], "oof_protection": selected["oof_protection"], "development_target_met": target_met, "validation_opened_once_after_freeze": True, "phase50_blind_accessed": False, "complete_requests_accessed": False, "adaptive_signal_source": "OOF only"})
    write_json(output / "audit/environment.json", {**environment_record(), "gpu_used": False, "network_used": False, "raw_accessed": False, "phase50_blind_accessed": False, "complete_requests_accessed": False, "training_used": True})
    summary = {"schema_version": "phase57-pd-iterative-histogram-optimization-result-v1", "status": "PASS", "workflow_commit": expected, "completed_at_utc": utc_now(), "counts": {"profiles": 1200, "train_profiles": 960, "validation_profiles": 240, "models": 6, "segments": 3, "example_rows": 7200, "train_rows": 5760, "validation_rows": 1440, "candidates": len(all_results), "rounds_completed": rounds_completed, "complete_request_rows_in_git": 0}, "selected": {"candidate_id": selected_config["candidate_id"], "family": selected_config["family"], "alpha_map": alpha_map}, "gates": {"oof_target": selected["oof_target"], "oof_protection": selected["oof_protection"], "development_overall": overall["target_gate"], "development_all_models": all(value["target_guard"] for value in model_audits.values()), "development_all_segments": all(value["target_guard"] for value in segment_audits.values()), "target_met": bool(target_met), "next_phase_permitted": bool(target_met)}, "development_validation": overall, "models": model_audits, "segments": segment_audits, "scientific_outcome": "DEVELOPMENT_TARGET_MET" if target_met else "DEVELOPMENT_TARGET_NOT_MET", "proved": "bounded multi-family OOF iterative search with one-shot validation", "not_proved": "fresh blind generalization, unseen-model extrapolation, physical communication time, placement, latency, queueing or online scheduling"}
    write_json(output / "summary.json", summary); (output / "README.md").write_text(f"# Phase57：PD 直方图迭代优化\n\n状态：`PASS`（流程完整）。完成 {rounds_completed} 轮、{len(all_results)} 个候选；选中 `{selected_config['candidate_id']}`；development target={target_met}。\n\n所有候选、OOF 校准和融合只使用 Phase48 train OOF；validation 只在冻结后打开一次；未读取 Phase50 blind、raw 或完整请求。\n", encoding="utf-8")
    (output / "logs").mkdir(); (output / "logs/runtime.log").write_text(f"completed={utc_now()} workflow_commit={expected}\nrounds={rounds_completed} candidates={len(all_results)} selected={selected_config['candidate_id']} family={selected_config['family']}\noof_target={selected['oof_target']} oof_protection={selected['oof_protection']} development_target={target_met}\ngpu=false network=false phase50_blind=false complete_requests=false\n", encoding="utf-8"); (output / "DONE").write_text("PASS\n", encoding="utf-8"); refresh_manifest(output); return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--expected-workflow-commit", required=True); parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase57_pd_iterative_histogram_optimization")
    args = parser.parse_args(); print(json.dumps(run(args.expected_workflow_commit, args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
