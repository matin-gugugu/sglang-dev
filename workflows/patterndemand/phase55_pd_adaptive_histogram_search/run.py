#!/usr/bin/env python3
"""Phase55 bounded OOF-adaptive search; validation is opened only once at the end."""

from __future__ import annotations

import argparse
import csv
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

_P54RUN_SPEC = importlib.util.spec_from_file_location("phase54_run_for_phase55", P54 / "run.py")
if _P54RUN_SPEC is None or _P54RUN_SPEC.loader is None:
    raise RuntimeError("cannot load Phase54 predictor implementation")
P54RUN = importlib.util.module_from_spec(_P54RUN_SPEC); _P54RUN_SPEC.loader.exec_module(P54RUN)
_P55PREFLIGHT_SPEC = importlib.util.spec_from_file_location("phase55_preflight", HERE / "preflight.py")
if _P55PREFLIGHT_SPEC is None or _P55PREFLIGHT_SPEC.loader is None:
    raise RuntimeError("cannot load Phase55 preflight")
P55PREFLIGHT = importlib.util.module_from_spec(_P55PREFLIGHT_SPEC); _P55PREFLIGHT_SPEC.loader.exec_module(P55PREFLIGHT)


MODEL_IDS = P54RUN.MODEL_IDS
SEGMENTS = P54RUN.SEGMENTS
SCORE_KEYS = P54RUN.SCORE_KEYS


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def score(metrics: dict[str, float]) -> float:
    return float(np.mean([float(metrics[key]) for key in SCORE_KEYS]))


def seed_candidates(contract: dict[str, Any]) -> list[dict[str, Any]]:
    epochs = int(contract["search_contract"]["stage_a_max_epochs"])
    base = [
        ("p55_a_shared_uniform_causal_w48_d1", "shared", "uniform", "fixed_draining_causal", 48, 1, 0.006, 0.001),
        ("p55_a_shared_shape_causal_w64_d1", "shared", "shape_focus", "fixed_draining_causal", 64, 1, 0.005, 0.001),
        ("p55_a_shared_tail_causal_w64_d2", "shared", "tail_shape_focus", "fixed_draining_causal", 64, 2, 0.004, 0.002),
        ("p55_a_shared_shape_full_w64_d2", "shared", "shape_focus", "full_target_free", 64, 2, 0.004, 0.002),
        ("p55_a_per_model_shape_causal_w64_d1", "per_model", "shape_focus", "fixed_draining_causal", 64, 1, 0.005, 0.001),
        ("p55_a_per_model_shape_causal_w64_d2", "per_model", "shape_focus", "fixed_draining_causal", 64, 2, 0.004, 0.002),
        ("p55_a_per_model_tail_causal_w64_d2", "per_model", "tail_shape_focus", "fixed_draining_causal", 64, 2, 0.004, 0.002),
        ("p55_a_per_model_tail_full_w96_d2", "per_model", "tail_shape_focus", "full_target_free", 96, 2, 0.003, 0.002),
        ("p55_a_per_model_shape_full_w96_d2", "per_model", "shape_focus", "full_target_free", 96, 2, 0.003, 0.002),
        ("p55_a_shared_tail_full_w96_d2", "shared", "tail_shape_focus", "full_target_free", 96, 2, 0.003, 0.002),
    ]
    return [{"candidate_id": cid, "scope": scope, "loss_mode": loss, "feature_mode": features, "width": width, "depth": depth, "learning_rate": lr, "weight_decay": wd, "max_epochs": epochs, "patience": 70, "stage": "A", "parent_candidate_id": ""} for cid, scope, loss, features, width, depth, lr, wd in base]


def refinement_signal(value: dict[str, Any]) -> dict[str, Any]:
    """Turn a parent's OOF 12-bin bias into a deterministic next-round hint."""
    bias_rows = value.get("oof_bias", []) if "config" in value else []
    if not bias_rows:
        return {"policy": "default_shape", "tail_abs_bias": 0.0, "head_abs_bias": 0.0, "worst_bin": -1}
    grouped: dict[tuple[str, int], float] = {}
    for row in bias_rows:
        grouped[(str(row["kind"]), int(row["bin"]))] = abs(float(row["relative_signed_bias"]))
    head = [amount for (kind, index), amount in grouped.items() if index < 4]
    tail = [amount for (kind, index), amount in grouped.items() if index >= 8]
    worst_kind, worst_bin = max(grouped, key=grouped.get)
    head_abs = float(np.mean(head)) if head else 0.0
    tail_abs = float(np.mean(tail)) if tail else 0.0
    policy = "tail_focus" if tail_abs >= head_abs else "shape_focus"
    return {"policy": policy, "tail_abs_bias": tail_abs, "head_abs_bias": head_abs, "worst_bin": int(worst_bin), "worst_kind": worst_kind}


def refine_candidates(top: list[dict[str, Any]], contract: dict[str, Any]) -> list[dict[str, Any]]:
    maximum = int(contract["search_contract"]["stage_a_max_epochs"]) * 2
    variants: list[dict[str, Any]] = []
    for parent_value in top:
        parent = dict(parent_value["config"] if "config" in parent_value else parent_value)
        signal = refinement_signal(parent_value)
        focus = "tail_shape_focus" if signal["policy"] == "tail_focus" else "shape_focus"
        signal_fields = {"adaptation_policy": signal["policy"], "adaptation_worst_bin": signal["worst_bin"], "adaptation_tail_abs_bias": signal["tail_abs_bias"], "adaptation_head_abs_bias": signal["head_abs_bias"]}
        deeper = dict(parent); deeper.update({"candidate_id": f"{parent['candidate_id']}__deeper", "loss_mode": focus, "width": min(128, int(parent["width"]) + 32), "depth": min(3, int(parent["depth"]) + 1), "learning_rate": float(parent["learning_rate"]) * 0.75, "max_epochs": maximum, "patience": 140, "stage": "B", "parent_candidate_id": parent["candidate_id"], **signal_fields}); variants.append(deeper)
        shape = dict(parent); shape.update({"candidate_id": f"{parent['candidate_id']}__shape_variant", "scope": "per_model", "loss_mode": focus, "feature_mode": "full_target_free", "width": max(64, int(parent["width"])), "depth": max(2, int(parent["depth"])), "max_epochs": maximum, "patience": 140, "stage": "B", "parent_candidate_id": parent["candidate_id"], **signal_fields}); variants.append(shape)
    return variants


def bin_bias_rows(rows: list[dict[str, str]], calls: np.ndarray, bytes_: np.ndarray, candidate_id: str, stage: str) -> list[dict[str, Any]]:
    target_calls, target_bytes = P54RUN.histogram_arrays(rows, "target"); output: list[dict[str, Any]] = []
    for kind, predicted, target in (("calls", calls, target_calls), ("logical_bytes", bytes_, target_bytes)):
        for index in range(12):
            target_sum = float(target[:, index].sum()); predicted_sum = float(predicted[:, index].sum())
            output.append({"candidate_id": candidate_id, "stage": stage, "kind": kind, "bin": index, "target_sum": target_sum, "predicted_sum": predicted_sum, "signed_bias": predicted_sum - target_sum, "relative_signed_bias": (predicted_sum - target_sum) / max(target_sum, 1e-12)})
    return output


def evaluate_candidate(train: list[dict[str, str]], config: dict[str, Any], folds: dict[str, int], alpha_grid: list[float]) -> dict[str, Any]:
    raw_calls, raw_bytes, epochs = P54RUN.fit_predict_oof(train, config, folds)
    h0_calls, h0_bytes = P54RUN.histogram_arrays(train, "h0"); target_calls, target_bytes = P54RUN.histogram_arrays(train, "target")
    h0 = P54RUN.metric_bundle(h0_calls, h0_bytes, target_calls, target_bytes)
    alpha_by_model, alpha_audits, alpha_rows = P54RUN.choose_alphas(train, raw_calls, raw_bytes, h0, alpha_grid)
    calls, bytes_ = P54RUN.shrink(train, raw_calls, raw_bytes, alpha_by_model)
    overall, models, segments, target = P54RUN.development_audits(train, calls, bytes_)
    oof_protection = P54RUN.strict_h0(overall) and all(P54RUN.strict_h0(value) for value in models.values()) and all(P54RUN.strict_h0(value) for value in segments.values())
    return {"config": config, "raw_calls": raw_calls, "raw_bytes": raw_bytes, "calls": calls, "bytes": bytes_, "epochs": epochs, "alpha_by_model": alpha_by_model, "alpha_audits": alpha_audits, "alpha_rows": alpha_rows, "overall": overall, "models": models, "segments": segments, "oof_target": target, "oof_protection": oof_protection, "oof_score": score(overall["h0_plus_dnn_refined"]), "oof_bias": bin_bias_rows(train, calls, bytes_, config["candidate_id"], config["stage"]), "sort": (not target, not oof_protection, score(overall["h0_plus_dnn_refined"]), float(overall["h0_plus_dnn_refined"]["calls_histogram_wape"]) + float(overall["h0_plus_dnn_refined"]["bytes_histogram_wape"]), config["candidate_id"])}


def fit_final_bundle(train: list[dict[str, str]], config: dict[str, Any], epochs: int, seeds: list[int]) -> dict[str, Any]:
    if config["scope"] == "shared":
        transform = P54RUN.fit_transform(train, config["feature_mode"]); models = []
        for seed in seeds:
            model, _audit = P54RUN.fit_model(P54RUN.transform_inputs(train, transform), P54RUN.transform_targets(train, transform), config, int(seed), fixed_epochs=epochs); models.append(P54RUN.model_to_json(model))
        return {"scope": "shared", "transform": transform, "models": models}
    per_model: dict[str, Any] = {}
    for model_id in MODEL_IDS:
        rows = [row for row in train if row["model"] == model_id]; transform = P54RUN.fit_transform(rows, config["feature_mode"]); models = []
        for seed in seeds:
            model, _audit = P54RUN.fit_model(P54RUN.transform_inputs(rows, transform), P54RUN.transform_targets(rows, transform), config, int(seed), fixed_epochs=epochs); models.append(P54RUN.model_to_json(model))
        per_model[model_id] = {"transform": transform, "models": models}
    return {"scope": "per_model", "per_model": per_model}


def prediction_rows(rows: list[dict[str, str]], calls: np.ndarray, bytes_: np.ndarray, method: str) -> list[dict[str, Any]]:
    output = []
    for row, cv, bv in zip(rows, calls, bytes_):
        value = {name: row[name] for name in ("profile_id", "split_role", "source", "segment", "source_split", "window_id", "cutoff_ms", "model")}; value["method"] = method
        for index in range(12): value[f"predicted_calls_bin_{index:02d}"] = float(cv[index]); value[f"predicted_logical_bytes_bin_{index:02d}"] = float(bv[index])
        output.append(value)
    return output


def run(expected: str, output: Path) -> dict[str, Any]:
    preflight = P55PREFLIGHT.run_checks(expected); contract = load_json(HERE / "experiment.json")
    if output.exists():
        raise RuntimeError(f"refuse overwrite: {output}")
    source = repo_root() / contract["pinned_inputs"][2]["path"]; rows = read_csv_gz(source)
    train = [row for row in rows if row["split_role"] == "expanded_train"]; validation = [row for row in rows if row["split_role"] == "expanded_validation"]
    folds = P54RUN.fold_map(train); alpha_grid = [float(value) for value in contract["search_contract"]["alpha_grid"]]
    stage_a = seed_candidates(contract); evaluations: list[dict[str, Any]] = []; trace: list[dict[str, Any]] = []
    for config in stage_a:
        result = evaluate_candidate(train, config, folds, alpha_grid); evaluations.append(result); trace.append({"stage": "A", "candidate_id": config["candidate_id"], "parent_candidate_id": "", "action": "seed_candidate", "oof_target": result["oof_target"], "oof_protection": result["oof_protection"], "oof_score": result["oof_score"]})
    stage_a_sorted = sorted(evaluations, key=lambda value: value["sort"]); top = stage_a_sorted[:int(contract["search_contract"]["stage_a_top_k"])]
    for config in refine_candidates(top, contract):
        result = evaluate_candidate(train, config, folds, alpha_grid); evaluations.append(result); signal = {name: config.get(name) for name in ("adaptation_policy", "adaptation_worst_bin", "adaptation_tail_abs_bias", "adaptation_head_abs_bias")}; trace.append({"stage": "B", "candidate_id": config["candidate_id"], "parent_candidate_id": config["parent_candidate_id"], "action": "oof_adaptive_variant", "oof_target": result["oof_target"], "oof_protection": result["oof_protection"], "oof_score": result["oof_score"], **signal})
    if len(evaluations) != int(contract["search_contract"]["max_total_candidates"]):
        raise RuntimeError(f"candidate budget mismatch: {len(evaluations)}")
    selected = sorted(evaluations, key=lambda value: value["sort"])[0]; config = selected["config"]
    selected_epochs = int(np.clip(round(statistics.median(selected["epochs"])), 100, int(config["max_epochs"])))
    seeds = [int(value) for value in contract["search_contract"]["ensemble_seeds"]]; bundle = fit_final_bundle(train, config, selected_epochs, seeds)
    raw_val_calls, raw_val_bytes = P54RUN.predict_bundle(validation, bundle); validation_calls, validation_bytes = P54RUN.shrink(validation, raw_val_calls, raw_val_bytes, selected["alpha_by_model"])
    overall, model_audits, segment_audits, target_met = P54RUN.development_audits(validation, validation_calls, validation_bytes)
    output.mkdir(parents=True)
    candidate_rows = [{"stage": value["config"]["stage"], "candidate_id": value["config"]["candidate_id"], "parent_candidate_id": value["config"]["parent_candidate_id"], "scope": value["config"]["scope"], "loss_mode": value["config"]["loss_mode"], "adaptation_policy": value["config"].get("adaptation_policy", "seed"), "adaptation_worst_bin": value["config"].get("adaptation_worst_bin", -1), "oof_target": value["oof_target"], "oof_protection": value["oof_protection"], "oof_score": value["oof_score"], "oof_composite_ratio": value["overall"]["composite_ratio"], "oof_calls_histogram_wape": value["overall"]["h0_plus_dnn_refined"]["calls_histogram_wape"], "oof_bytes_histogram_wape": value["overall"]["h0_plus_dnn_refined"]["bytes_histogram_wape"], "epochs_median": int(statistics.median(value["epochs"])), "selected": value is selected} for value in evaluations]
    write_csv(output / "analysis/search_trace.csv", trace); write_csv(output / "analysis/oof_candidate_metrics.csv", candidate_rows); write_csv(output / "analysis/oof_bin_bias.csv", [row for value in evaluations for row in value["oof_bias"]])
    write_json(output / "analysis/oof_selection.json", {"selected_candidate": config, "selected_epochs": selected_epochs, "alpha_by_model": selected["alpha_by_model"], "oof_overall": selected["overall"], "oof_models": selected["models"], "oof_segments": selected["segments"], "oof_target": selected["oof_target"], "oof_protection": selected["oof_protection"], "candidate_count": len(evaluations), "stage_a_count": len(stage_a), "stage_b_count": len(evaluations) - len(stage_a)})
    write_csv(output / "analysis/development_validation_metrics.csv", [{"method": "h0", **overall["h0"], "composite_ratio_to_h0": 1.0, "formal_target_gate": False}, {"method": "h0_plus_dnn_adaptive", **overall["h0_plus_dnn_refined"], "composite_ratio_to_h0": overall["composite_ratio"], "formal_target_gate": overall["target_gate"]}])
    write_json(output / "analysis/model_validation.json", model_audits); write_json(output / "analysis/segment_validation.json", segment_audits)
    h0_calls, h0_bytes = P54RUN.histogram_arrays(validation, "h0"); P54RUN.write_csv_gz(output / "predictions/development_validation_predictions.csv.gz", prediction_rows(validation, h0_calls, h0_bytes, "h0") + prediction_rows(validation, validation_calls, validation_bytes, "h0_plus_dnn_adaptive"))
    checkpoint = {"schema_version": "phase55-pd-adaptive-histogram-search-checkpoint-v1", "workflow_commit": expected, "selected_candidate": config, "selected_epochs": selected_epochs, "alpha_by_model": selected["alpha_by_model"], "ensemble_seeds": seeds, "bundle": bundle, "phase50_blind_accessed": False, "complete_requests_accessed": False}
    P54RUN.write_json_gz(output / "checkpoints/pd_adaptive_histogram_search.json.gz", checkpoint)
    write_json(output / "audit/input_freeze.json", preflight); write_json(output / "audit/search.json", {"candidate_budget": len(evaluations), "stage_a_count": len(stage_a), "stage_b_count": len(evaluations) - len(stage_a), "selected_candidate_id": config["candidate_id"], "selected_epochs": selected_epochs, "oof_target": selected["oof_target"], "oof_protection": selected["oof_protection"], "development_target_met": target_met, "validation_opened_once_after_freeze": True, "phase50_blind_accessed": False, "complete_requests_accessed": False})
    write_json(output / "audit/environment.json", {**environment_record(), "gpu_used": False, "network_used": False, "raw_accessed": False, "phase50_blind_accessed": False, "complete_requests_accessed": False, "training_used": True})
    summary = {"schema_version": "phase55-pd-adaptive-histogram-search-result-v1", "status": "PASS", "workflow_commit": expected, "completed_at_utc": utc_now(), "counts": {"profiles": 1200, "train_profiles": 960, "validation_profiles": 240, "models": 6, "example_rows": 7200, "train_rows": 5760, "validation_rows": 1440, "candidates": len(evaluations), "complete_request_rows_in_git": 0}, "selected": {"candidate_id": config["candidate_id"], "scope": config["scope"], "loss_mode": config["loss_mode"], "epochs": selected_epochs, "alpha_by_model": selected["alpha_by_model"]}, "gates": {"oof_target": selected["oof_target"], "oof_protection": selected["oof_protection"], "development_overall": overall["target_gate"], "development_all_models": all(value["target_guard"] for value in model_audits.values()), "development_all_segments": all(value["target_guard"] for value in segment_audits.values()), "target_met": target_met, "phase56_permitted": target_met}, "development_validation": overall, "models": model_audits, "segments": segment_audits, "scientific_outcome": "DEVELOPMENT_TARGET_MET" if target_met else "DEVELOPMENT_TARGET_NOT_MET", "proved": "bounded OOF-adaptive search without blind access", "not_proved": "fresh blind generalization, unseen-model extrapolation, physical communication time, placement, latency or online scheduling"}
    write_json(output / "summary.json", summary); (output / "README.md").write_text(f"# Phase55：PD受约束自适应直方图搜索\n\n状态：`PASS`（搜索流程完整）。共评估 {len(evaluations)} 个候选，选中 `{config['candidate_id']}`；OOF protection={selected['oof_protection']}；开发集目标={target_met}。\n\nvalidation 只在搜索和 alpha/epoch 冻结后打开一次；未读取 Phase50 blind、raw 或完整请求。\n", encoding="utf-8")
    (output / "logs").mkdir(); (output / "logs/runtime.log").write_text(f"completed={utc_now()} workflow_commit={expected}\ncandidates={len(evaluations)} stage_a={len(stage_a)} stage_b={len(evaluations)-len(stage_a)} selected={config['candidate_id']}\noof_target={selected['oof_target']} oof_protection={selected['oof_protection']} development_target={target_met}\ngpu=false network=false phase50_blind=false complete_requests=false\n", encoding="utf-8"); (output / "DONE").write_text("PASS\n", encoding="utf-8"); refresh_manifest(output); return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--expected-workflow-commit", required=True); parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase55_pd_adaptive_histogram_search")
    args = parser.parse_args(); print(json.dumps(run(args.expected_workflow_commit, args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
