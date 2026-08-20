#!/usr/bin/env python3
"""Phase59: time-bounded OOF iteration with a metric-aligned shape DNN."""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import importlib.util
import json
import os
import pickle
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
P54 = HERE.parent / "phase54_pd_histogram_accuracy_refinement"
sys.path.insert(0, str(P54)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

from common import environment_record, load_json, refresh_manifest, repo_root, utc_now, write_json  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


P54RUN = load_module("phase59_p54_run", P54 / "run.py")
PREFLIGHT = load_module("phase59_preflight", HERE / "preflight.py")
SHAPE = load_module("phase59_shape_model", HERE / "model.py")
MODEL_IDS = P54RUN.MODEL_IDS
SEGMENTS = P54RUN.SEGMENTS
BIN_COUNT = 12
RUNTIME_STATE_SCHEMA = "phase59-pd-metric-aligned-runtime-state-v2"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True); fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def stable_seed(candidate_id: str, fold: int, group: str, extra: int = 0) -> int:
    digest = hashlib.sha256(f"phase59:{candidate_id}:{fold}:{group}:{extra}".encode()).hexdigest(); return 590000 + int(digest[:8], 16) % 900000


def compact_result(value: dict[str, Any]) -> dict[str, Any]:
    """Keep scientific/audit fields while dropping large prediction arrays."""
    return {key: item for key, item in value.items() if key not in {"base_calls", "base_bytes", "calls", "bytes"}}


def checkpoint_partial_result(value: dict[str, Any]) -> dict[str, Any]:
    """A partial-round train result must retain raw OOF arrays for later blends."""
    return {key: item for key, item in value.items() if key not in {"calls", "bytes"}}


def atomic_save_runtime_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with gzip.open(temporary, "wb", compresslevel=3) as stream:
            pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_runtime_state(path: Path, expected: str, output: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as stream:
        state = pickle.load(stream)
    identity = {
        "schema_version": state.get("schema_version"),
        "workflow_commit": state.get("workflow_commit"),
        "output_dir": state.get("output_dir"),
    }
    required = {"schema_version": RUNTIME_STATE_SCHEMA, "workflow_commit": expected, "output_dir": str(output)}
    if identity != required:
        raise RuntimeError({"runtime_state_identity": identity, "required": required})
    return state


def model_groups(rows: list[dict[str, str]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(row["model"], []).append(index)
    return groups


def train_configs(round_index: int, signal: dict[str, Any], contract: dict[str, Any]) -> list[dict[str, Any]]:
    search = contract["search_contract"]; widths = search["width_grid"]; depths = search["depth_grid"]; bounds = search["residual_bound_grid"]; epochs = search["max_epochs_grid"]
    worst = signal.get("worst_kind", "balanced"); focus = signal.get("worst_segment") if round_index else None
    templates = [
        (1.0, 1.0, .20, .20, .0030),
        (1.50 if worst == "calls" else 1.15, 1.0, .35, .25, .0025),
        (1.0, 1.50 if worst == "bytes" else 1.15, .20, .35, .0020),
        (1.20, 1.20, .60, .60, .0015),
    ]
    output = []
    for index, (calls_weight, bytes_weight, tv_weight, emd_weight, learning_rate) in enumerate(templates):
        output.append({
            "family": "metric_shape_mlp", "candidate_id": f"p59_r{round_index:03d}_train_{index}", "stage": "train", "round": round_index,
            "feature_mode": "fixed_draining_causal", "head_scope": "model", "width": int(widths[(round_index + index) % len(widths)]),
            "depth": int(depths[(round_index * 2 + index) % len(depths)]), "residual_bound": float(bounds[(round_index + index * 2) % len(bounds)]),
            "calls_wape_weight": calls_weight, "bytes_wape_weight": bytes_weight, "tv_weight": tv_weight, "emd_weight": emd_weight,
            "segment_focus": focus if index >= 2 else None, "segment_focus_weight": 1.5 + 0.25 * (round_index % 3),
            "learning_rate": learning_rate, "weight_decay": .001 + .0005 * (index % 2), "residual_l2": 1e-4,
            "max_epochs": int(epochs[(round_index + index) % len(epochs)]), "eval_every": 5, "patience_evals": 20 + 4 * (index % 2),
        })
    return output


def fit_predict_oof(train: list[dict[str, str]], config: dict[str, Any], folds: dict[str, int]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    calls = np.zeros((len(train), BIN_COUNT), dtype=np.float64); bytes_ = np.zeros_like(calls); epochs: list[int] = []
    groups = model_groups(train)
    for fold in range(4):
        fit_set = {index for index, row in enumerate(train) if folds[row["profile_id"]] != fold}; hold_set = {index for index, row in enumerate(train) if folds[row["profile_id"]] == fold}
        for group, indices in sorted(groups.items()):
            fit_indices = [index for index in indices if index in fit_set]; hold_indices = [index for index in indices if index in hold_set]
            if not hold_indices:
                continue
            fit_rows = [train[index] for index in fit_indices]; hold_rows = [train[index] for index in hold_indices]; transform = P54RUN.fit_transform(fit_rows, config["feature_mode"])
            x_fit = P54RUN.transform_inputs(fit_rows, transform); x_hold = P54RUN.transform_inputs(hold_rows, transform)
            fitted, audit = SHAPE.fit_shape_model(x_fit, fit_rows, config, stable_seed(config["candidate_id"], fold, group), P54RUN.histogram_arrays, validation=(x_hold, hold_rows))
            normalized = SHAPE.forward(fitted, x_hold); pc, pb = SHAPE.decode_shape(hold_rows, normalized, float(config["residual_bound"]), P54RUN.histogram_arrays)
            calls[hold_indices] = pc; bytes_[hold_indices] = pb; epochs.append(int(audit["best_epoch"]))
    return calls, bytes_, epochs


def apply_alpha(rows: list[dict[str, str]], calls: np.ndarray, bytes_: np.ndarray, alpha_map: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    h0_calls, h0_bytes = P54RUN.histogram_arrays(rows, "h0"); h0 = P54RUN.encode_histograms(h0_calls, h0_bytes); predicted = P54RUN.encode_histograms(calls, bytes_)
    alpha = np.asarray([float(alpha_map[row["model"]]) for row in rows], dtype=np.float64)[:, None]
    return P54RUN.decode_histograms(h0 + alpha * (predicted - h0))


def safe_bias_rows(rows: list[dict[str, str]], calls: np.ndarray, bytes_: np.ndarray, candidate_id: str, round_index: int) -> list[dict[str, Any]]:
    target_calls, target_bytes = P54RUN.histogram_arrays(rows, "target"); output = []
    for kind, predicted, target in (("calls", calls, target_calls), ("logical_bytes", bytes_, target_bytes)):
        target_sums = target.sum(axis=0); positive = target_sums[target_sums > 0]; floor = max(float(np.median(positive) if len(positive) else 1.0) * 1e-3, 1.0)
        for index in range(BIN_COUNT):
            target_sum = float(target_sums[index]); predicted_sum = float(predicted[:, index].sum()); signed = predicted_sum - target_sum
            output.append({"round": round_index, "candidate_id": candidate_id, "kind": kind, "bin": index, "target_sum": target_sum, "predicted_sum": predicted_sum, "signed_bias": signed, "safe_relative_signed_bias": signed / max(target_sum, floor), "zero_target_bin": target_sum == 0.0})
    return output


def violation_score(overall: dict[str, Any], models: dict[str, Any], segments: dict[str, Any]) -> float:
    values = []
    candidate = overall["h0_plus_dnn_refined"]
    for key in ("calls_histogram_wape", "bytes_histogram_wape"):
        values.append(float(candidate[key]) / .10)
    for key in ("calls_total_wape", "bytes_total_wape"):
        values.append(float(candidate[key]) / .05)
    for audits in (models, segments):
        for audit in audits.values():
            metric = audit["h0_plus_dnn_refined"]
            values.extend([float(metric["calls_histogram_wape"]) / .15, float(metric["bytes_histogram_wape"]) / .15, float(metric["calls_total_wape"]) / .05, float(metric["bytes_total_wape"]) / .05])
    return float(max(values) + .05 * sum(max(value - 1.0, 0.0) for value in values))


def evaluate_arrays(train: list[dict[str, str]], config: dict[str, Any], raw_calls: np.ndarray, raw_bytes: np.ndarray, epochs: list[int], contract: dict[str, Any]) -> dict[str, Any]:
    h0_calls, h0_bytes = P54RUN.histogram_arrays(train, "h0"); h0_metrics = P54RUN.metric_bundle(h0_calls, h0_bytes, *P54RUN.histogram_arrays(train, "target"))
    alpha_map, alpha_audits, _ = P54RUN.choose_alphas(train, raw_calls, raw_bytes, h0_metrics, [float(value) for value in contract["search_contract"]["alpha_grid"]]); calls, bytes_ = apply_alpha(train, raw_calls, raw_bytes, alpha_map)
    overall, models, segments, target = P54RUN.development_audits(train, calls, bytes_); protection = P54RUN.strict_h0(overall) and all(P54RUN.strict_h0(value) for value in models.values()) and all(P54RUN.strict_h0(value) for value in segments.values()); violation = violation_score(overall, models, segments)
    return {"config": config, "base_calls": raw_calls, "base_bytes": raw_bytes, "calls": calls, "bytes": bytes_, "epochs": epochs, "alpha_map": alpha_map, "alpha_audits": alpha_audits, "overall": overall, "models": models, "segments": segments, "oof_target": bool(target), "oof_protection": bool(protection), "violation_score": violation, "bias": safe_bias_rows(train, calls, bytes_, config["candidate_id"], int(config["round"])), "sort": (not target, not protection, violation, P54RUN.score(overall["h0_plus_dnn_refined"]), config["candidate_id"])}


def blend_configs(base_results: list[dict[str, Any]], round_index: int) -> list[dict[str, Any]]:
    top = sorted(base_results, key=lambda value: value["sort"]); a = top[0]["config"]["candidate_id"]; b = top[1]["config"]["candidate_id"]; c = top[2]["config"]["candidate_id"]
    specs = [([a, b], [.75, .25]), ([a, b], [.5, .5]), ([a, b], [.25, .75]), ([a, b, c], [.5, .3, .2])]
    return [{"family": "oof_blend", "candidate_id": f"p59_r{round_index:03d}_blend_{index}", "stage": "blend", "round": round_index, "parent_ids": parents, "weights": weights} for index, (parents, weights) in enumerate(specs)]


def evaluate_blend(train: list[dict[str, str]], config: dict[str, Any], result_map: dict[str, dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    parents = [result_map[parent] for parent in config["parent_ids"]]; weights = np.asarray(config["weights"], dtype=np.float64); weights /= weights.sum(); calls = sum(float(weight) * parent["base_calls"] for weight, parent in zip(weights, parents)); bytes_ = sum(float(weight) * parent["base_bytes"] for weight, parent in zip(weights, parents))
    return evaluate_arrays(train, config, calls, bytes_, [1], contract)


def refinement_signal(result: dict[str, Any]) -> dict[str, Any]:
    candidate = result["overall"]["h0_plus_dnn_refined"]; worst_kind = "calls" if float(candidate["calls_histogram_wape"]) >= float(candidate["bytes_histogram_wape"]) else "bytes"
    worst_model = max(result["models"], key=lambda key: max(float(result["models"][key]["h0_plus_dnn_refined"]["calls_histogram_wape"]), float(result["models"][key]["h0_plus_dnn_refined"]["bytes_histogram_wape"])))
    worst_segment = max(result["segments"], key=lambda key: max(float(result["segments"][key]["h0_plus_dnn_refined"]["calls_histogram_wape"]), float(result["segments"][key]["h0_plus_dnn_refined"]["bytes_histogram_wape"])))
    usable = [row for row in result["bias"] if not row["zero_target_bin"]]; worst_bin = max(usable, key=lambda row: abs(float(row["safe_relative_signed_bias"]))) if usable else None
    return {"selected_candidate_id": result["config"]["candidate_id"], "worst_kind": worst_kind, "worst_model": worst_model, "worst_segment": worst_segment, "worst_bin": None if worst_bin is None else {"kind": worst_bin["kind"], "bin": worst_bin["bin"], "safe_relative_signed_bias": worst_bin["safe_relative_signed_bias"]}, "model_failures": sum(not value["target_guard"] for value in result["models"].values()), "segment_failures": sum(not value["target_guard"] for value in result["segments"].values())}


def fit_final_base(train: list[dict[str, str]], rows: list[dict[str, str]], config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    calls = np.zeros((len(rows), BIN_COUNT), dtype=np.float64); bytes_ = np.zeros_like(calls); bundle = {"family": "metric_shape_mlp", "candidate_id": config["candidate_id"], "groups": {}}
    for group, train_indices in sorted(model_groups(train).items()):
        fit_rows = [train[index] for index in train_indices]; hold_indices = [index for index, row in enumerate(rows) if row["model"] == group]; hold_rows = [rows[index] for index in hold_indices]; transform = P54RUN.fit_transform(fit_rows, config["feature_mode"]); x_fit = P54RUN.transform_inputs(fit_rows, transform); x_hold = P54RUN.transform_inputs(hold_rows, transform); models = []
        for extra in range(3):
            fitted, _ = SHAPE.fit_shape_model(x_fit, fit_rows, config, stable_seed(config["candidate_id"], 99, group, extra), P54RUN.histogram_arrays, fixed_epochs=int(config["selected_epochs"])); models.append(fitted)
        normalized = np.mean([SHAPE.forward(model, x_hold) for model in models], axis=0); pc, pb = SHAPE.decode_shape(hold_rows, normalized, float(config["residual_bound"]), P54RUN.histogram_arrays); calls[hold_indices] = pc; bytes_[hold_indices] = pb; bundle["groups"][group] = {"transform": transform, "models": [SHAPE.model_to_json(model) for model in models]}
    return calls, bytes_, bundle


def fit_final(train: list[dict[str, str]], rows: list[dict[str, str]], config: dict[str, Any], config_map: dict[str, dict[str, Any]], cache: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]], on_cache_update: Any = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    candidate_id = config["candidate_id"]
    if candidate_id in cache:
        return cache[candidate_id]
    if config["family"] == "metric_shape_mlp":
        result = fit_final_base(train, rows, config)
    else:
        weights = np.asarray(config["weights"], dtype=np.float64); weights /= weights.sum(); calls = np.zeros((len(rows), BIN_COUNT)); bytes_ = np.zeros_like(calls); bundles = []
        for weight, parent_id in zip(weights, config["parent_ids"]):
            pc, pb, bundle = fit_final(train, rows, config_map[parent_id], config_map, cache, on_cache_update); calls += float(weight) * pc; bytes_ += float(weight) * pb; bundles.append(bundle)
        result = (calls, bytes_, {"family": "oof_blend", "candidate_id": candidate_id, "parent_ids": config["parent_ids"], "weights": weights.tolist(), "bundles": bundles})
    cache[candidate_id] = result
    if on_cache_update is not None:
        on_cache_update(candidate_id, result)
    return result


def prediction_rows(rows: list[dict[str, str]], calls: np.ndarray, bytes_: np.ndarray, method: str) -> list[dict[str, Any]]:
    output = []
    for row, call_values, byte_values in zip(rows, calls, bytes_):
        value = {name: row[name] for name in ("profile_id", "split_role", "source", "segment", "source_split", "window_id", "cutoff_ms", "model")}; value["method"] = method
        for index in range(BIN_COUNT):
            value[f"predicted_calls_bin_{index:02d}"] = float(call_values[index]); value[f"predicted_logical_bytes_bin_{index:02d}"] = float(byte_values[index])
        output.append(value)
    return output


def run(expected: str, output: Path, runtime_state_path: Path | None = None) -> dict[str, Any]:
    session_started = time.monotonic(); current_started_utc = utc_now(); preflight = PREFLIGHT.run_checks(expected); contract = load_json(HERE / "experiment.json"); search = contract["search_contract"]
    if output.exists():
        raise RuntimeError(f"refuse overwrite: {output}")
    if runtime_state_path is None:
        runtime_state_path = Path("/tmp") / f"patterndemand-phase59-{expected[:12]}.resume.pkl.gz"
    runtime_state_path = runtime_state_path.resolve()
    rows = P54RUN.read_csv_gz(repo_root() / contract["pinned_inputs"][1]["path"]); train = [row for row in rows if row["split_role"] == "expanded_train"]; validation = [row for row in rows if row["split_role"] == "expanded_validation"]; folds = P54RUN.fold_map(train)
    resumed = runtime_state_path.exists()
    if resumed:
        saved = load_runtime_state(runtime_state_path, expected, output)
        started_utc = str(saved["started_at_utc"]); prior_elapsed = float(saved["elapsed_seconds"]); all_results = saved["all_results"]; trace = saved["trace"]; all_bias = saved["all_bias"]; config_map = saved["config_map"]; durations = saved["durations"]; signal = saved["signal"]; rounds_completed = int(saved["rounds_completed"]); partial_round_index = saved["partial_round_index"]; partial_train_results = saved["partial_train_results"]; search_complete = bool(saved["search_complete"]); stop_reason = str(saved["stop_reason"]); search_elapsed_at_completion = saved.get("search_elapsed_at_completion"); final_cache = saved.get("final_cache", {}); checkpoint_writes = int(saved.get("checkpoint_writes", 0)); restart_count = int(saved.get("restart_count", 0)) + 1
        print(json.dumps({"event": "runtime_state_restored", "path": str(runtime_state_path), "rounds_completed": rounds_completed, "partial_round_index": partial_round_index, "candidates": len(all_results), "elapsed_seconds": round(prior_elapsed, 3), "restart_count": restart_count}, sort_keys=True), flush=True)
    else:
        started_utc = current_started_utc; prior_elapsed = 0.0; all_results: list[dict[str, Any]] = []; trace: list[dict[str, Any]] = []; all_bias: list[dict[str, Any]] = []; config_map: dict[str, dict[str, Any]] = {}; durations: list[float] = []; signal: dict[str, Any] = {}; rounds_completed = 0; partial_round_index = None; partial_train_results: list[dict[str, Any]] = []; search_complete = False; stop_reason = "max_rounds"; search_elapsed_at_completion = None; final_cache: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}; checkpoint_writes = 0; restart_count = 0

    def active_elapsed() -> float:
        return prior_elapsed + time.monotonic() - session_started

    def save_progress() -> None:
        nonlocal checkpoint_writes
        checkpoint_writes += 1
        state = {
            "schema_version": RUNTIME_STATE_SCHEMA, "workflow_commit": expected, "output_dir": str(output), "started_at_utc": started_utc,
            "elapsed_seconds": active_elapsed(), "all_results": all_results, "trace": trace, "all_bias": all_bias, "config_map": config_map,
            "durations": durations, "signal": signal, "rounds_completed": rounds_completed, "partial_round_index": partial_round_index,
            "partial_train_results": [checkpoint_partial_result(value) for value in partial_train_results], "search_complete": search_complete,
            "stop_reason": stop_reason, "search_elapsed_at_completion": search_elapsed_at_completion, "final_cache": final_cache, "checkpoint_writes": checkpoint_writes, "restart_count": restart_count,
        }
        atomic_save_runtime_state(runtime_state_path, state)

    start_round = int(partial_round_index) if partial_round_index is not None else rounds_completed
    for round_index in range(start_round, int(search["max_rounds"])) if not search_complete else []:
        elapsed = active_elapsed(); median_duration = statistics.median(durations) if durations else 0.0; forecast = elapsed + 1.25 * median_duration * int(search["train_candidates_per_round"])
        if round_index >= int(search["minimum_rounds"]) and partial_round_index is None and (elapsed >= float(search["search_time_budget_seconds"]) or forecast >= float(search["search_time_budget_seconds"])):
            stop_reason = "search_time_budget"; search_complete = True; search_elapsed_at_completion = active_elapsed(); save_progress(); break
        if partial_round_index == round_index:
            base_results = partial_train_results
        else:
            partial_round_index = round_index; partial_train_results = []; base_results = partial_train_results; save_progress()
        result_map = {value["config"]["candidate_id"]: value for value in base_results}; completed_ids = {value["config"]["candidate_id"] for value in all_results}
        for config in train_configs(round_index, signal, contract):
            if config["candidate_id"] in completed_ids:
                continue
            candidate_started = time.monotonic(); raw_calls, raw_bytes, epochs = fit_predict_oof(train, config, folds); value = evaluate_arrays(train, config, raw_calls, raw_bytes, epochs, contract); duration = time.monotonic() - candidate_started; durations.append(duration); value["duration_seconds"] = duration; selected_epochs = int(np.clip(round(statistics.median(epochs)), 50, int(config["max_epochs"]))); value["config"] = {**config, "selected_epochs": selected_epochs}; config_map[config["candidate_id"]] = value["config"]; result_map[config["candidate_id"]] = value; base_results.append(value); all_results.append(compact_result(value)); all_bias.extend(value["bias"])
            trace.append({"round": round_index, "stage": "train", "candidate_id": config["candidate_id"], "family": config["family"], "duration_seconds": duration, "oof_target": value["oof_target"], "oof_protection": value["oof_protection"], "violation_score": value["violation_score"], "calls_histogram_wape": value["overall"]["h0_plus_dnn_refined"]["calls_histogram_wape"], "bytes_histogram_wape": value["overall"]["h0_plus_dnn_refined"]["bytes_histogram_wape"]})
            save_progress(); print(json.dumps({"event": "candidate_complete", "candidate": config["candidate_id"], "round": round_index, "duration_seconds": round(duration, 3), "oof_target": value["oof_target"], "oof_protection": value["oof_protection"], "violation_score": value["violation_score"], "elapsed_seconds": round(active_elapsed(), 3)}, sort_keys=True), flush=True)
        for config in blend_configs(base_results, round_index):
            if config["candidate_id"] in completed_ids:
                continue
            value = evaluate_blend(train, config, result_map, contract); value["duration_seconds"] = 0.0; config_map[config["candidate_id"]] = config; all_results.append(compact_result(value)); all_bias.extend(value["bias"])
            trace.append({"round": round_index, "stage": "blend", "candidate_id": config["candidate_id"], "family": config["family"], "duration_seconds": 0.0, "oof_target": value["oof_target"], "oof_protection": value["oof_protection"], "violation_score": value["violation_score"], "calls_histogram_wape": value["overall"]["h0_plus_dnn_refined"]["calls_histogram_wape"], "bytes_histogram_wape": value["overall"]["h0_plus_dnn_refined"]["bytes_histogram_wape"]})
            completed_ids.add(config["candidate_id"]); save_progress()
        rounds_completed = round_index + 1; best = sorted(all_results, key=lambda value: value["sort"])[0]; signal = refinement_signal(best); partial_round_index = None; partial_train_results = []
        print(json.dumps({"event": "round_complete", "round": round_index, "best": best["config"]["candidate_id"], "oof_target": best["oof_target"], "oof_protection": best["oof_protection"], "signal": signal, "elapsed_seconds": round(active_elapsed(), 3)}, sort_keys=True), flush=True)
        if rounds_completed >= int(search["minimum_rounds"]) and best["oof_target"] and best["oof_protection"]:
            stop_reason = "oof_contract_met"; search_complete = True; search_elapsed_at_completion = active_elapsed()
        save_progress()
        if search_complete:
            break
    if not search_complete:
        search_complete = True; stop_reason = "max_rounds"; search_elapsed_at_completion = active_elapsed(); save_progress()
    if not all_results or len(all_results) > int(search["max_total_candidates"]):
        raise RuntimeError({"candidate_count": len(all_results), "max": search["max_total_candidates"]})
    if search_elapsed_at_completion is None:
        raise RuntimeError("search marked complete without frozen elapsed time")
    search_elapsed = float(search_elapsed_at_completion); selected = sorted(all_results, key=lambda value: value["sort"])[0]; selected_config = copy.deepcopy(selected["config"])
    def cache_update(candidate_id: str, value: tuple[np.ndarray, np.ndarray, dict[str, Any]]) -> None:
        final_cache[candidate_id] = value; save_progress()
    validation_base_calls, validation_base_bytes, final_bundle = fit_final(train, validation, selected_config, config_map, final_cache, cache_update); validation_calls, validation_bytes = apply_alpha(validation, validation_base_calls, validation_base_bytes, selected["alpha_map"]); overall, model_audits, segment_audits, validation_gate = P54RUN.development_audits(validation, validation_calls, validation_bytes); target_met = bool(selected["oof_target"] and selected["oof_protection"] and validation_gate); total_elapsed = active_elapsed()
    staging = output.parent / f".{output.name}.staging-{expected[:12]}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True); candidate_rows = [{"round": value["config"]["round"], "stage": value["config"]["stage"], "candidate_id": value["config"]["candidate_id"], "family": value["config"]["family"], "duration_seconds": value["duration_seconds"], "oof_target": value["oof_target"], "oof_protection": value["oof_protection"], "violation_score": value["violation_score"], "oof_score": P54RUN.score(value["overall"]["h0_plus_dnn_refined"]), "oof_calls_histogram_wape": value["overall"]["h0_plus_dnn_refined"]["calls_histogram_wape"], "oof_bytes_histogram_wape": value["overall"]["h0_plus_dnn_refined"]["bytes_histogram_wape"], "selected": value is selected} for value in all_results]
    write_csv(staging / "analysis/search_trace.csv", trace); write_csv(staging / "analysis/oof_candidate_metrics.csv", candidate_rows); write_csv(staging / "analysis/oof_bin_bias.csv", all_bias)
    write_json(staging / "analysis/oof_selection.json", {"selected_candidate": selected_config, "alpha_map": selected["alpha_map"], "oof_overall": selected["overall"], "oof_models": selected["models"], "oof_segments": selected["segments"], "oof_target": selected["oof_target"], "oof_protection": selected["oof_protection"], "violation_score": selected["violation_score"], "candidate_count": len(all_results), "rounds_completed": rounds_completed, "stop_reason": stop_reason, "final_signal": signal})
    write_csv(staging / "analysis/development_validation_metrics.csv", [{"method": "h0", **overall["h0"], "composite_ratio_to_h0": 1.0, "formal_target_gate": False}, {"method": "h0_plus_dnn_metric_aligned", **overall["h0_plus_dnn_refined"], "composite_ratio_to_h0": overall["composite_ratio"], "formal_target_gate": overall["target_gate"]}]); write_json(staging / "analysis/model_validation.json", model_audits); write_json(staging / "analysis/segment_validation.json", segment_audits)
    continuation = {"continuation_required": not target_met, "reason": None if target_met else "unchanged accuracy contract not met within Phase59 OOF/time budget", "thresholds_must_remain_unchanged": True, "phase50_blind_must_remain_closed": True, "last_oof_signal": signal, "selected_oof_violation_score": selected["violation_score"], "next_action": "freeze successful predictor for later blind evaluation" if target_met else "design the next development-only refinement from Phase59 OOF diagnostics; do not tune on development validation or Phase50 blind labels"}; write_json(staging / "analysis/continuation_spec.json", continuation)
    h0_calls, h0_bytes = P54RUN.histogram_arrays(validation, "h0"); P54RUN.write_csv_gz(staging / "predictions/development_validation_predictions.csv.gz", prediction_rows(validation, h0_calls, h0_bytes, "h0") + prediction_rows(validation, validation_calls, validation_bytes, "h0_plus_dnn_metric_aligned"))
    checkpoint = {"schema_version": "phase59-pd-metric-aligned-shape-checkpoint-v1", "workflow_commit": expected, "selected_candidate": selected_config, "alpha_map": selected["alpha_map"], "bundle": final_bundle, "phase50_blind_accessed": False, "complete_requests_accessed": False}; P54RUN.write_json_gz(staging / "checkpoints/pd_metric_aligned_shape_search.json.gz", checkpoint)
    write_json(staging / "audit/input_freeze.json", preflight); write_json(staging / "audit/search.json", {"candidate_budget": len(all_results), "max_candidate_budget": search["max_total_candidates"], "rounds_completed": rounds_completed, "selected_candidate_id": selected_config["candidate_id"], "stop_reason": stop_reason, "search_elapsed_seconds": search_elapsed, "total_elapsed_seconds": total_elapsed, "search_time_budget_seconds": search["search_time_budget_seconds"], "hard_total_runtime_seconds": search["hard_total_runtime_seconds"], "runtime_budget_respected": total_elapsed <= float(search["hard_total_runtime_seconds"]), "resumed_from_checkpoint": resumed, "restart_count": restart_count, "runtime_checkpoint_writes": checkpoint_writes, "oof_target": selected["oof_target"], "oof_protection": selected["oof_protection"], "development_target_met": validation_gate, "validation_opened_once_after_freeze": True, "phase50_blind_accessed": False, "complete_requests_accessed": False, "adaptive_signal_source": "OOF only"}); write_json(staging / "audit/environment.json", {**environment_record(), "gpu_used": False, "network_used": False, "raw_accessed": False, "phase50_blind_accessed": False, "complete_requests_accessed": False, "training_used": True})
    summary = {"schema_version": "phase59-pd-metric-aligned-shape-result-v1", "status": "PASS", "workflow_commit": expected, "started_at_utc": started_utc, "completed_at_utc": utc_now(), "counts": {"profiles": 1200, "train_profiles": 960, "validation_profiles": 240, "models": 6, "segments": 3, "example_rows": 7200, "train_rows": 5760, "validation_rows": 1440, "candidates": len(all_results), "rounds_completed": rounds_completed, "complete_request_rows_in_git": 0}, "runtime": {"search_elapsed_seconds": search_elapsed, "total_elapsed_seconds": total_elapsed, "stop_reason": stop_reason}, "selected": {"candidate_id": selected_config["candidate_id"], "family": selected_config["family"], "alpha_map": selected["alpha_map"]}, "gates": {"oof_target": selected["oof_target"], "oof_protection": selected["oof_protection"], "development_overall": overall["target_gate"], "development_all_models": all(value["target_guard"] for value in model_audits.values()), "development_all_segments": all(value["target_guard"] for value in segment_audits.values()), "target_met": target_met, "next_phase_permitted": target_met}, "development_validation": overall, "models": model_audits, "segments": segment_audits, "scientific_outcome": "DEVELOPMENT_TARGET_MET" if target_met else "DEVELOPMENT_TARGET_NOT_MET", "proved": "time-bounded OOF metric-aligned total-preserving H0+DNN shape search with one-shot development validation", "not_proved": "fresh blind generalization, unseen-model extrapolation, physical communication time, placement, latency or online scheduling"}
    write_json(staging / "summary.json", summary); (staging / "README.md").write_text(f"# Phase59：PD metric-aligned shape搜索\n\n状态：`PASS`（流程完整）。完成 {rounds_completed} 轮、{len(all_results)} 个候选，停止原因 `{stop_reason}`，选中 `{selected_config['candidate_id']}`；最终合同通过={target_met}。\n\n搜索只使用OOF；validation在冻结后打开一次；未读取Phase50 blind、raw或完整请求。\n", encoding="utf-8"); (staging / "logs").mkdir(); (staging / "logs/runtime.log").write_text(f"started={started_utc}\ncompleted={utc_now()} workflow_commit={expected}\nrounds={rounds_completed} candidates={len(all_results)} selected={selected_config['candidate_id']} stop_reason={stop_reason}\nsearch_elapsed_seconds={search_elapsed:.3f} total_elapsed_seconds={total_elapsed:.3f}\nresumed_from_checkpoint={resumed} restart_count={restart_count} checkpoint_writes={checkpoint_writes}\noof_target={selected['oof_target']} oof_protection={selected['oof_protection']} development_target={validation_gate} final_target={target_met}\ngpu=false network=false phase50_blind=false complete_requests=false\n", encoding="utf-8"); (staging / "DONE").write_text("PASS\n", encoding="utf-8"); refresh_manifest(staging); staging.replace(output); runtime_state_path.unlink(missing_ok=True); return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--expected-workflow-commit", required=True); parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase59_pd_metric_aligned_shape_search"); parser.add_argument("--runtime-state", type=Path); args = parser.parse_args(); print(json.dumps(run(args.expected_workflow_commit, args.output_dir.resolve(), args.runtime_state), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
