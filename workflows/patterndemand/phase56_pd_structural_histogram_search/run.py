#!/usr/bin/env python3
"""Phase56 structural, calibrated OOF search for pure-PD histogram prediction."""

from __future__ import annotations

import argparse
import csv
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
sys.path.insert(0, str(P54)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

from common import environment_record, load_json, refresh_manifest, repo_root, utc_now, write_json  # noqa: E402
from model_loader import read_csv_gz  # noqa: E402

_RUN_SPEC = importlib.util.spec_from_file_location("phase54_run_for_phase56", P54 / "run.py")
if _RUN_SPEC is None or _RUN_SPEC.loader is None:
    raise RuntimeError("cannot load pinned Phase54 predictor")
P54RUN = importlib.util.module_from_spec(_RUN_SPEC); _RUN_SPEC.loader.exec_module(P54RUN)
_PREFLIGHT_SPEC = importlib.util.spec_from_file_location("phase56_preflight", HERE / "preflight.py")
if _PREFLIGHT_SPEC is None or _PREFLIGHT_SPEC.loader is None:
    raise RuntimeError("cannot load Phase56 preflight")
PREFLIGHT = importlib.util.module_from_spec(_PREFLIGHT_SPEC); _PREFLIGHT_SPEC.loader.exec_module(PREFLIGHT)

MODEL_IDS = P54RUN.MODEL_IDS
SEGMENTS = P54RUN.SEGMENTS
SCORE_KEYS = P54RUN.SCORE_KEYS


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def score(metrics: dict[str, float]) -> float:
    return float(np.mean([float(metrics[key]) for key in SCORE_KEYS]))


def augment_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Add target-free categorical one-hot features without changing identifiers."""
    augmented: list[dict[str, str]] = []
    for row in rows:
        value = dict(row)
        for segment in SEGMENTS:
            value[f"feature_cat_segment_{segment}"] = str(float(row["segment"] == segment))
        for model in MODEL_IDS:
            safe = hashlib.sha256(model.encode()).hexdigest()[:10]
            value[f"feature_cat_model_{safe}"] = str(float(row["model"] == model))
        augmented.append(value)
    return augmented


def scope_key(row: dict[str, str], scope: str) -> str:
    if scope == "global" or scope == "shared":
        return "global"
    if scope == "model":
        return f"model::{row['model']}"
    if scope == "segment":
        return f"segment::{row['segment']}"
    if scope == "model_segment":
        return f"model_segment::{row['model']}::{row['segment']}"
    raise ValueError(scope)


def groups_for(rows: list[dict[str, str]], scope: str) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(scope_key(row, scope), []).append(index)
    return groups


def seed_candidates(contract: dict[str, Any]) -> list[dict[str, Any]]:
    epochs = int(contract["search_contract"]["stage_a_max_epochs"] if "stage_a_max_epochs" in contract["search_contract"] else 420)
    specs = [
        ("p56_a_shared_shape_w64_d1", "shared", "global", "none", 0.0, "shape_focus", "full_target_free", 64, 1, .005, .001),
        ("p56_a_shared_tail_w96_d2", "shared", "model", "head_residual", .5, "tail_shape_focus", "full_target_free", 96, 2, .004, .002),
        ("p56_a_model_uniform_causal_w64_d1", "model", "model", "none", 0.0, "uniform", "fixed_draining_causal", 64, 1, .006, .001),
        ("p56_a_model_shape_full_w96_d2", "model", "model", "none", 0.0, "shape_focus", "full_target_free", 96, 2, .004, .002),
        ("p56_a_model_tail_full_w128_d2", "model", "model", "head_residual", .5, "tail_shape_focus", "full_target_free", 128, 2, .003, .002),
        ("p56_a_model_shape_full_w128_d3", "model", "model", "head_residual", .75, "shape_focus", "full_target_free", 128, 3, .003, .002),
        ("p56_a_model_shape_causal_w128_d2_alpha_group", "model", "model_segment", "head_residual", .5, "shape_focus", "fixed_draining_causal", 128, 2, .003, .002),
        ("p56_a_segment_shape_full_w96_d2", "segment", "model_segment", "head_residual", .5, "shape_focus", "full_target_free", 96, 2, .004, .002),
        ("p56_a_segment_tail_full_w128_d2", "segment", "model_segment", "head_residual", .75, "tail_shape_focus", "full_target_free", 128, 2, .003, .002),
        ("p56_a_ms_uniform_full_w64_d1", "model_segment", "model_segment", "none", 0.0, "uniform", "full_target_free", 64, 1, .006, .001),
        ("p56_a_ms_shape_full_w64_d2_cal25", "model_segment", "model_segment", "head_residual", .25, "shape_focus", "full_target_free", 64, 2, .005, .001),
        ("p56_a_ms_shape_full_w96_d2_cal50", "model_segment", "model_segment", "head_residual", .5, "shape_focus", "full_target_free", 96, 2, .004, .002),
        ("p56_a_ms_tail_full_w96_d2_cal50", "model_segment", "model_segment", "head_residual", .5, "tail_shape_focus", "full_target_free", 96, 2, .004, .002),
        ("p56_a_ms_tail_full_w128_d3_cal75", "model_segment", "model_segment", "head_residual", .75, "tail_shape_focus", "full_target_free", 128, 3, .003, .002),
        ("p56_a_ms_shape_causal_w96_d2_cal50", "model_segment", "model_segment", "head_residual", .5, "shape_focus", "fixed_draining_causal", 96, 2, .004, .002),
        ("p56_a_ms_tail_causal_w128_d2_cal75", "model_segment", "model_segment", "head_residual", .75, "tail_shape_focus", "fixed_draining_causal", 128, 2, .003, .002),
        ("p56_a_ms_shape_full_w128_d2_cal100", "model_segment", "model_segment", "head_residual", 1.0, "shape_focus", "full_target_free", 128, 2, .003, .002),
        ("p56_a_ms_uniform_full_w128_d3_cal50", "model_segment", "model_segment", "head_residual", .5, "uniform", "full_target_free", 128, 3, .003, .002),
        ("p56_a_model_shape_full_w192_d3_cal75", "model", "model_segment", "head_residual", .75, "shape_focus", "full_target_free", 192, 3, .0025, .002),
        ("p56_a_ms_shape_full_w192_d3_cal100", "model_segment", "model_segment", "head_residual", 1.0, "shape_focus", "full_target_free", 192, 3, .0025, .002),
    ]
    return [{
        "candidate_id": cid, "head_scope": head, "alpha_scope": alpha_scope,
        "calibration_mode": calibration_mode, "calibration_strength": strength,
        "loss_mode": loss, "feature_mode": features, "width": width, "depth": depth,
        "learning_rate": lr, "weight_decay": wd, "max_epochs": epochs, "patience": 70,
        "stage": "A", "parent_candidate_id": "", "adaptation_policy": "seed",
    } for cid, head, alpha_scope, calibration_mode, strength, loss, features, width, depth, lr, wd in specs]


def refinement_signal(result: dict[str, Any]) -> dict[str, Any]:
    segment_ratios = [float(result["segments"][segment]["composite_ratio"]) for segment in SEGMENTS]
    bias = result["oof_bias"]
    head = [abs(float(row["relative_signed_bias"])) for row in bias if int(row["bin"]) < 4]
    tail = [abs(float(row["relative_signed_bias"])) for row in bias if int(row["bin"]) >= 8]
    head_abs = float(np.mean(head)) if head else 0.0
    tail_abs = float(np.mean(tail)) if tail else 0.0
    return {
        "policy": "model_segment_tail" if max(segment_ratios) - min(segment_ratios) >= .03 and tail_abs >= head_abs else
                  "model_segment_shape" if max(segment_ratios) - min(segment_ratios) >= .03 else
                  "tail_shape" if tail_abs >= head_abs else "shape",
        "segment_ratio_gap": max(segment_ratios) - min(segment_ratios),
        "head_abs_bias": head_abs,
        "tail_abs_bias": tail_abs,
    }


def refine_candidates(top: list[dict[str, Any]], contract: dict[str, Any]) -> list[dict[str, Any]]:
    maximum = int(contract["search_contract"].get("stage_a_max_epochs", 420)) * 2
    variants: list[dict[str, Any]] = []
    for parent_result in top:
        parent = dict(parent_result["config"])
        signal = refinement_signal(parent_result)
        focus = "tail_shape_focus" if "tail" in signal["policy"] else "shape_focus"
        head = "model_segment" if "model_segment" in signal["policy"] else parent["head_scope"]
        fields = {
            "adaptation_policy": signal["policy"], "adaptation_segment_ratio_gap": signal["segment_ratio_gap"],
            "adaptation_head_abs_bias": signal["head_abs_bias"], "adaptation_tail_abs_bias": signal["tail_abs_bias"],
        }
        capacity = dict(parent); capacity.update({
            "candidate_id": f"{parent['candidate_id']}__capacity_head", "head_scope": head,
            "alpha_scope": "model_segment" if head == "model_segment" else parent["alpha_scope"],
            "loss_mode": focus, "width": min(192, int(parent["width"]) + 32), "depth": min(3, int(parent["depth"]) + 1),
            "learning_rate": float(parent["learning_rate"]) * .75, "max_epochs": maximum,
            "patience": 140, "stage": "B", "parent_candidate_id": parent["candidate_id"], **fields,
        }); variants.append(capacity)
        calibrated = dict(parent); calibrated.update({
            "candidate_id": f"{parent['candidate_id']}__calibrated_shape", "head_scope": head,
            "alpha_scope": "model_segment" if head == "model_segment" else parent["alpha_scope"],
            "calibration_mode": "head_residual", "calibration_strength": min(1.0, max(.25, float(parent["calibration_strength"]) + .25)),
            "loss_mode": focus, "feature_mode": "full_target_free", "max_epochs": maximum,
            "patience": 140, "stage": "B", "parent_candidate_id": parent["candidate_id"], **fields,
        }); variants.append(calibrated)
    return variants


def residual_from_hist(rows: list[dict[str, str]], calls: np.ndarray, bytes_: np.ndarray) -> np.ndarray:
    predicted = P54RUN.encode_histograms(calls, bytes_)
    h0_calls, h0_bytes = P54RUN.histogram_arrays(rows, "h0")
    return predicted - P54RUN.encode_histograms(h0_calls, h0_bytes)


def target_residual(rows: list[dict[str, str]]) -> np.ndarray:
    h0_calls, h0_bytes = P54RUN.histogram_arrays(rows, "h0")
    target_calls, target_bytes = P54RUN.histogram_arrays(rows, "target")
    return P54RUN.encode_histograms(target_calls, target_bytes) - P54RUN.encode_histograms(h0_calls, h0_bytes)


def decode_residual(rows: list[dict[str, str]], residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h0_calls, h0_bytes = P54RUN.histogram_arrays(rows, "h0")
    return P54RUN.decode_histograms(P54RUN.encode_histograms(h0_calls, h0_bytes) + residual)


def stable_seed(candidate_id: str, fold: int, group: str) -> int:
    digest = hashlib.sha256(f"phase56:{candidate_id}:{fold}:{group}".encode()).hexdigest()
    return 560000 + int(digest[:8], 16) % 900000


def fit_predict_oof(train: list[dict[str, str]], config: dict[str, Any], folds: dict[str, int]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    raw = np.zeros((len(train), 26), dtype=np.float64); calibrated = np.zeros_like(raw); epochs: list[int] = []
    target_all = target_residual(train)
    for fold in range(4):
        fit_indices = [i for i, row in enumerate(train) if folds[row["profile_id"]] != fold]
        hold_indices = [i for i, row in enumerate(train) if folds[row["profile_id"]] == fold]
        groups = groups_for(train, config["head_scope"])
        for group, group_indices in sorted(groups.items()):
            fit_i = [i for i in group_indices if i in set(fit_indices)]
            hold_i = [i for i in group_indices if i in set(hold_indices)]
            if not hold_i:
                continue
            fit_rows = [train[i] for i in fit_i]; hold_rows = [train[i] for i in hold_i]
            transform = P54RUN.fit_transform(fit_rows, config["feature_mode"])
            model, audit = P54RUN.fit_model(
                P54RUN.transform_inputs(fit_rows, transform), P54RUN.transform_targets(fit_rows, transform), config,
                stable_seed(config["candidate_id"], fold, group),
                validation=(P54RUN.transform_inputs(hold_rows, transform), P54RUN.transform_targets(hold_rows, transform)),
            )
            fit_calls, fit_bytes = P54RUN.predict_histograms(fit_rows, transform, [model])
            hold_calls, hold_bytes = P54RUN.predict_histograms(hold_rows, transform, [model])
            fit_residual = residual_from_hist(fit_rows, fit_calls, fit_bytes); hold_residual = residual_from_hist(hold_rows, hold_calls, hold_bytes)
            raw[hold_i] = hold_residual
            offset = np.zeros(26, dtype=np.float64)
            if config["calibration_mode"] == "head_residual":
                offset = np.mean(target_all[fit_i] - fit_residual, axis=0)
            calibrated[hold_i] = hold_residual + float(config["calibration_strength"]) * offset
            epochs.append(int(audit["best_epoch"]))
    return raw, calibrated, epochs


def choose_alphas(rows: list[dict[str, str]], calls: np.ndarray, bytes_: np.ndarray, scope: str, grid: list[float]) -> tuple[dict[str, float], dict[str, Any]]:
    h0_calls, h0_bytes = P54RUN.histogram_arrays(rows, "h0"); h0_encoded = P54RUN.encode_histograms(h0_calls, h0_bytes)
    predicted_encoded = P54RUN.encode_histograms(calls, bytes_); alpha_map: dict[str, float] = {}; audits: dict[str, Any] = {}
    for group, indices in sorted(groups_for(rows, scope).items()):
        idx = np.asarray(indices, dtype=int); subset = [rows[i] for i in indices]
        target_calls, target_bytes = P54RUN.histogram_arrays(subset, "target"); base_calls, base_bytes = P54RUN.histogram_arrays(subset, "h0")
        h0_metrics = P54RUN.metric_bundle(base_calls, base_bytes, target_calls, target_bytes); choices = []
        for alpha in grid:
            candidate_calls, candidate_bytes = P54RUN.decode_histograms(h0_encoded[idx] + float(alpha) * (predicted_encoded[idx] - h0_encoded[idx]))
            candidate = P54RUN.metric_bundle(candidate_calls, candidate_bytes, target_calls, target_bytes); comparison = P54RUN.compare_to_h0(candidate, h0_metrics)
            strict = P54RUN.strict_h0(comparison)
            choices.append((not strict, score(candidate), float(alpha), comparison, candidate))
        choices.sort(key=lambda value: value[:3]); rejected, _, alpha, comparison, candidate = choices[0]
        alpha_map[group] = alpha; audits[group] = {"alpha": alpha, "strict_h0": not rejected, "metrics": candidate, "comparison": comparison}
    return alpha_map, audits


def apply_alpha(rows: list[dict[str, str]], calls: np.ndarray, bytes_: np.ndarray, scope: str, alpha_map: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    h0_calls, h0_bytes = P54RUN.histogram_arrays(rows, "h0"); h0_encoded = P54RUN.encode_histograms(h0_calls, h0_bytes)
    predicted = P54RUN.encode_histograms(calls, bytes_); alpha = np.asarray([float(alpha_map[scope_key(row, scope)]) for row in rows], dtype=np.float64)[:, None]
    return P54RUN.decode_histograms(h0_encoded + alpha * (predicted - h0_encoded))


def bin_bias_rows(rows: list[dict[str, str]], calls: np.ndarray, bytes_: np.ndarray, candidate_id: str, stage: str) -> list[dict[str, Any]]:
    target_calls, target_bytes = P54RUN.histogram_arrays(rows, "target"); output: list[dict[str, Any]] = []
    for kind, predicted, target in (("calls", calls, target_calls), ("logical_bytes", bytes_, target_bytes)):
        for index in range(12):
            target_sum = float(target[:, index].sum()); predicted_sum = float(predicted[:, index].sum())
            output.append({"candidate_id": candidate_id, "stage": stage, "kind": kind, "bin": index, "target_sum": target_sum, "predicted_sum": predicted_sum, "signed_bias": predicted_sum - target_sum, "relative_signed_bias": (predicted_sum - target_sum) / max(target_sum, 1e-12)})
    return output


def honest_oof_offsets(rows: list[dict[str, str]], raw_residual: np.ndarray, scope: str) -> dict[str, list[float]]:
    target = target_residual(rows); return {group: np.mean(target[idx] - raw_residual[idx], axis=0).tolist() for group, idx in groups_for(rows, scope).items()}


def evaluate_candidate(train: list[dict[str, str]], config: dict[str, Any], folds: dict[str, int], contract: dict[str, Any]) -> dict[str, Any]:
    raw_residual, calibrated_residual, epochs = fit_predict_oof(train, config, folds)
    oof_calls, oof_bytes = decode_residual(train, calibrated_residual)
    alpha_map, alpha_audits = choose_alphas(train, oof_calls, oof_bytes, config["alpha_scope"], [float(value) for value in contract["search_contract"]["alpha_grid"]])
    calls, bytes_ = apply_alpha(train, oof_calls, oof_bytes, config["alpha_scope"], alpha_map)
    overall, models, segments, target = P54RUN.development_audits(train, calls, bytes_)
    protection = P54RUN.strict_h0(overall) and all(P54RUN.strict_h0(value) for value in models.values()) and all(P54RUN.strict_h0(value) for value in segments.values())
    return {
        "config": config, "raw_residual": raw_residual, "calibrated_residual": calibrated_residual,
        "calls": calls, "bytes": bytes_, "epochs": epochs, "alpha_map": alpha_map, "alpha_audits": alpha_audits,
        "overall": overall, "models": models, "segments": segments, "oof_target": target, "oof_protection": protection,
        "oof_score": score(overall["h0_plus_dnn_refined"]), "oof_bias": bin_bias_rows(train, calls, bytes_, config["candidate_id"], config["stage"]),
        "oof_offsets": honest_oof_offsets(train, raw_residual, config["head_scope"]),
        "sort": (not target, not protection, score(overall["h0_plus_dnn_refined"]), float(overall["h0_plus_dnn_refined"]["calls_histogram_wape"]) + float(overall["h0_plus_dnn_refined"]["bytes_histogram_wape"]), config["candidate_id"]),
    }


def fit_final_bundle(train: list[dict[str, str]], config: dict[str, Any], epochs: int, seeds: list[int]) -> dict[str, Any]:
    bundle = {"head_scope": config["head_scope"], "groups": {}}
    for group, indices in sorted(groups_for(train, config["head_scope"]).items()):
        rows = [train[i] for i in indices]; transform = P54RUN.fit_transform(rows, config["feature_mode"]); models = []
        for seed in seeds:
            model, _audit = P54RUN.fit_model(P54RUN.transform_inputs(rows, transform), P54RUN.transform_targets(rows, transform), config, int(seed), fixed_epochs=epochs)
            models.append(P54RUN.model_to_json(model))
        bundle["groups"][group] = {"transform": transform, "models": models}
    return bundle


def _model_from_json(value: dict[str, Any]) -> dict[str, Any]:
    return {"weights": [np.asarray(item, dtype=np.float64) for item in value["weights"]], "biases": [np.asarray(item, dtype=np.float64) for item in value["biases"]]}


def predict_final_bundle(rows: list[dict[str, str]], bundle: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    calls = np.zeros((len(rows), 12), dtype=np.float64); bytes_ = np.zeros_like(calls); groups = groups_for(rows, bundle["head_scope"])
    for group, indices in groups.items():
        subset = [rows[i] for i in indices]; spec = bundle["groups"][group]; models = [_model_from_json(value) for value in spec["models"]]
        predicted_calls, predicted_bytes = P54RUN.predict_histograms(subset, spec["transform"], models); calls[indices] = predicted_calls; bytes_[indices] = predicted_bytes
    return calls, bytes_


def prediction_rows(rows: list[dict[str, str]], calls: np.ndarray, bytes_: np.ndarray, method: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row, cv, bv in zip(rows, calls, bytes_):
        value = {name: row[name] for name in ("profile_id", "split_role", "source", "segment", "source_split", "window_id", "cutoff_ms", "model")}; value["method"] = method
        for index in range(12): value[f"predicted_calls_bin_{index:02d}"] = float(cv[index]); value[f"predicted_logical_bytes_bin_{index:02d}"] = float(bv[index])
        output.append(value)
    return output


def run(expected: str, output: Path) -> dict[str, Any]:
    preflight = PREFLIGHT.run_checks(expected); contract = load_json(HERE / "experiment.json")
    if output.exists():
        raise RuntimeError(f"refuse overwrite: {output}")
    source = repo_root() / contract["pinned_inputs"][2]["path"]
    rows = augment_rows(read_csv_gz(source)); train = [row for row in rows if row["split_role"] == "expanded_train"]; validation = [row for row in rows if row["split_role"] == "expanded_validation"]
    folds = P54RUN.fold_map(train); evaluations: list[dict[str, Any]] = []; trace: list[dict[str, Any]] = []; seeds = [int(value) for value in contract["search_contract"]["ensemble_seeds"]]
    stage_a = seed_candidates(contract)
    for config in stage_a:
        result = evaluate_candidate(train, config, folds, contract); evaluations.append(result); trace.append({"stage": "A", "candidate_id": config["candidate_id"], "parent_candidate_id": "", "action": "seed_candidate", "oof_target": result["oof_target"], "oof_protection": result["oof_protection"], "oof_score": result["oof_score"]})
    top = sorted(evaluations, key=lambda value: value["sort"])[:int(contract["search_contract"]["stage_a_top_k"])]
    for config in refine_candidates(top, contract):
        result = evaluate_candidate(train, config, folds, contract); evaluations.append(result); trace.append({"stage": "B", "candidate_id": config["candidate_id"], "parent_candidate_id": config["parent_candidate_id"], "action": "oof_structural_adaptive_variant", "oof_target": result["oof_target"], "oof_protection": result["oof_protection"], "oof_score": result["oof_score"], "adaptation_policy": config.get("adaptation_policy"), "adaptation_segment_ratio_gap": config.get("adaptation_segment_ratio_gap", ""), "adaptation_head_abs_bias": config.get("adaptation_head_abs_bias", ""), "adaptation_tail_abs_bias": config.get("adaptation_tail_abs_bias", "")})
    if len(evaluations) != int(contract["search_contract"]["max_total_candidates"]):
        raise RuntimeError(f"candidate budget mismatch: {len(evaluations)}")
    selected = sorted(evaluations, key=lambda value: value["sort"])[0]; config = selected["config"]
    selected_epochs = int(np.clip(round(statistics.median(selected["epochs"])), 100, int(config["max_epochs"])))
    bundle = fit_final_bundle(train, config, selected_epochs, seeds)
    raw_val_calls, raw_val_bytes = predict_final_bundle(validation, bundle); raw_val_residual = residual_from_hist(validation, raw_val_calls, raw_val_bytes)
    offsets = {key: np.asarray(value, dtype=np.float64) for key, value in selected["oof_offsets"].items()}; val_residual = raw_val_residual.copy()
    if config["calibration_mode"] == "head_residual":
        val_residual += float(config["calibration_strength"]) * np.asarray([offsets[scope_key(row, config["head_scope"])] for row in validation])
    val_raw_calls, val_raw_bytes = decode_residual(validation, val_residual); validation_calls, validation_bytes = apply_alpha(validation, val_raw_calls, val_raw_bytes, config["alpha_scope"], selected["alpha_map"])
    overall, model_audits, segment_audits, target_met = P54RUN.development_audits(validation, validation_calls, validation_bytes)
    output.mkdir(parents=True)
    candidate_rows = [{"stage": value["config"]["stage"], "candidate_id": value["config"]["candidate_id"], "parent_candidate_id": value["config"]["parent_candidate_id"], "head_scope": value["config"]["head_scope"], "alpha_scope": value["config"]["alpha_scope"], "calibration_mode": value["config"]["calibration_mode"], "calibration_strength": value["config"]["calibration_strength"], "loss_mode": value["config"]["loss_mode"], "oof_target": value["oof_target"], "oof_protection": value["oof_protection"], "oof_score": value["oof_score"], "oof_calls_histogram_wape": value["overall"]["h0_plus_dnn_refined"]["calls_histogram_wape"], "oof_bytes_histogram_wape": value["overall"]["h0_plus_dnn_refined"]["bytes_histogram_wape"], "epochs_median": int(statistics.median(value["epochs"])), "selected": value is selected} for value in evaluations]
    group_rows = []
    for group, audit in selected["alpha_audits"].items(): group_rows.append({"candidate_id": config["candidate_id"], "alpha_scope": config["alpha_scope"], "group": group, "alpha": audit["alpha"], "strict_h0": audit["strict_h0"], "composite_ratio": audit["comparison"]["composite_ratio"]})
    write_csv(output / "analysis/search_trace.csv", trace); write_csv(output / "analysis/oof_candidate_metrics.csv", candidate_rows); write_csv(output / "analysis/oof_group_metrics.csv", group_rows); write_csv(output / "analysis/oof_bin_bias.csv", [row for value in evaluations for row in value["oof_bias"]])
    write_json(output / "analysis/oof_selection.json", {"selected_candidate": config, "selected_epochs": selected_epochs, "alpha_map": selected["alpha_map"], "oof_offsets": selected["oof_offsets"], "oof_overall": selected["overall"], "oof_models": selected["models"], "oof_segments": selected["segments"], "oof_target": selected["oof_target"], "oof_protection": selected["oof_protection"], "candidate_count": len(evaluations), "stage_a_count": len(stage_a), "stage_b_count": len(evaluations) - len(stage_a)})
    write_csv(output / "analysis/development_validation_metrics.csv", [{"method": "h0", **overall["h0"], "composite_ratio_to_h0": 1.0, "formal_target_gate": False}, {"method": "h0_plus_dnn_structural", **overall["h0_plus_dnn_refined"], "composite_ratio_to_h0": overall["composite_ratio"], "formal_target_gate": overall["target_gate"]}])
    write_json(output / "analysis/model_validation.json", model_audits); write_json(output / "analysis/segment_validation.json", segment_audits)
    h0_calls, h0_bytes = P54RUN.histogram_arrays(validation, "h0"); P54RUN.write_csv_gz(output / "predictions/development_validation_predictions.csv.gz", prediction_rows(validation, h0_calls, h0_bytes, "h0") + prediction_rows(validation, validation_calls, validation_bytes, "h0_plus_dnn_structural"))
    checkpoint = {"schema_version": "phase56-pd-structural-histogram-search-checkpoint-v1", "workflow_commit": expected, "selected_candidate": config, "selected_epochs": selected_epochs, "alpha_map": selected["alpha_map"], "oof_offsets": selected["oof_offsets"], "ensemble_seeds": seeds, "bundle": bundle, "phase50_blind_accessed": False, "complete_requests_accessed": False}
    P54RUN.write_json_gz(output / "checkpoints/pd_structural_histogram_search.json.gz", checkpoint)
    write_json(output / "audit/input_freeze.json", preflight); write_json(output / "audit/search.json", {"candidate_budget": len(evaluations), "stage_a_count": len(stage_a), "stage_b_count": len(evaluations) - len(stage_a), "selected_candidate_id": config["candidate_id"], "selected_epochs": selected_epochs, "oof_target": selected["oof_target"], "oof_protection": selected["oof_protection"], "development_target_met": target_met, "validation_opened_once_after_freeze": True, "phase50_blind_accessed": False, "complete_requests_accessed": False, "calibration_offsets_source": "fit-fold for OOF; honest train OOF for final validation"})
    write_json(output / "audit/environment.json", {**environment_record(), "gpu_used": False, "network_used": False, "raw_accessed": False, "phase50_blind_accessed": False, "complete_requests_accessed": False, "training_used": True})
    summary = {"schema_version": "phase56-pd-structural-histogram-search-result-v1", "status": "PASS", "workflow_commit": expected, "completed_at_utc": utc_now(), "counts": {"profiles": 1200, "train_profiles": 960, "validation_profiles": 240, "models": 6, "segments": 3, "example_rows": 7200, "train_rows": 5760, "validation_rows": 1440, "candidates": len(evaluations), "complete_request_rows_in_git": 0}, "selected": {"candidate_id": config["candidate_id"], "head_scope": config["head_scope"], "alpha_scope": config["alpha_scope"], "calibration_mode": config["calibration_mode"], "calibration_strength": config["calibration_strength"], "loss_mode": config["loss_mode"], "epochs": selected_epochs, "alpha_map": selected["alpha_map"]}, "gates": {"oof_target": selected["oof_target"], "oof_protection": selected["oof_protection"], "development_overall": overall["target_gate"], "development_all_models": all(value["target_guard"] for value in model_audits.values()), "development_all_segments": all(value["target_guard"] for value in segment_audits.values()), "target_met": target_met, "next_phase_permitted": target_met}, "development_validation": overall, "models": model_audits, "segments": segment_audits, "scientific_outcome": "DEVELOPMENT_TARGET_MET" if target_met else "DEVELOPMENT_TARGET_NOT_MET", "proved": "bounded structural OOF search with head specialization and honest calibration", "not_proved": "fresh blind generalization, unseen-model extrapolation, physical communication time, placement, latency or online scheduling"}
    write_json(output / "summary.json", summary); (output / "README.md").write_text(f"# Phase56：结构化PD直方图搜索\n\n状态：`PASS`（流程完整）。共评估 {len(evaluations)} 个候选，选中 `{config['candidate_id']}`；head_scope={config['head_scope']}；calibration={config['calibration_mode']}:{config['calibration_strength']}；开发目标={target_met}。\n\n所有结构、校准、alpha和epoch只由train OOF决定；validation只在冻结后打开一次；未读取Phase50 blind、raw或完整请求。\n", encoding="utf-8")
    (output / "logs").mkdir(); (output / "logs/runtime.log").write_text(f"completed={utc_now()} workflow_commit={expected}\ncandidates={len(evaluations)} stage_a={len(stage_a)} stage_b={len(evaluations)-len(stage_a)} selected={config['candidate_id']}\nhead_scope={config['head_scope']} alpha_scope={config['alpha_scope']} calibration={config['calibration_mode']}:{config['calibration_strength']}\noof_target={selected['oof_target']} oof_protection={selected['oof_protection']} development_target={target_met}\ngpu=false network=false phase50_blind=false complete_requests=false\n", encoding="utf-8"); (output / "DONE").write_text("PASS\n", encoding="utf-8"); refresh_manifest(output); return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--expected-workflow-commit", required=True); parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase56_pd_structural_histogram_search")
    args = parser.parse_args(); print(json.dumps(run(args.expected_workflow_commit, args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
