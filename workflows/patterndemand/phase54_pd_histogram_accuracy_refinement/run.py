#!/usr/bin/env python3
"""Train and select Phase54 PD histogram-refinement candidates on Phase48 development data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

from common import environment_record, load_json, refresh_manifest, repo_root, utc_now, write_json  # noqa: E402
from contracts import validate_rows  # noqa: E402
from metrics import SCORE_KEYS, compare_to_h0, metric_bundle, target_gate  # noqa: E402
from model import decode_histograms, encode_histograms, fit_model, fit_transform, histogram_arrays, model_to_json, predict_histograms, read_csv_gz, write_csv_gz, write_json_gz  # noqa: E402
import preflight as phase54_preflight  # noqa: E402


MODEL_IDS = ("deepseek-v2-lite", "qwen3-8b", "qwen3-30b-a3b", "llama-3.2-3b-instruct", "qwen2.5-14b-instruct", "mixtral-8x7b-instruct-v0.1")
SEGMENTS = ("burstgpt_1", "burstgpt_2", "burstgpt_3")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def fold_map(rows: list[dict[str, str]], folds: int = 4) -> dict[str, int]:
    representatives: dict[str, dict[str, str]] = {}
    for row in rows:
        representatives.setdefault(row["profile_id"], row)
    groups: dict[tuple[str, str], list[str]] = {}
    for profile_id, row in representatives.items():
        count_bucket = str(int(float(row["feature_profile_request_count"]) // 1000))
        groups.setdefault((row["segment"], count_bucket), []).append(profile_id)
    result: dict[str, int] = {}
    for key, values in sorted(groups.items()):
        ordered = sorted(values, key=lambda value: hashlib.sha256(f"phase54-fold:{key}:{value}".encode()).hexdigest())
        for index, profile_id in enumerate(ordered):
            result[profile_id] = index % folds
    if set(result) != set(representatives):
        raise RuntimeError("OOF fold assignment lost profiles")
    return result


def strict_h0(comparison: dict[str, Any]) -> bool:
    return all(float(comparison["metric_ratios_to_h0"][key]) < 1.0 for key in SCORE_KEYS)


def score(metrics: dict[str, float]) -> float:
    return float(np.mean([float(metrics[key]) for key in SCORE_KEYS]))


def shrink(rows: list[dict[str, str]], calls: np.ndarray, logical_bytes: np.ndarray, alpha_by_model: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    h0_calls, h0_bytes = histogram_arrays(rows, "h0")
    alpha = np.asarray([float(alpha_by_model[row["model"]]) for row in rows], dtype=np.float64)[:, None]
    h0_encoded = encode_histograms(h0_calls, h0_bytes)
    raw_encoded = encode_histograms(calls, logical_bytes)
    return decode_histograms(h0_encoded + alpha * (raw_encoded - h0_encoded))


def subset_metrics(rows: list[dict[str, str]], calls: np.ndarray, logical_bytes: np.ndarray, indices: list[int]) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    subset = [rows[index] for index in indices]
    h0_calls, h0_bytes = histogram_arrays(subset, "h0"); target_calls, target_bytes = histogram_arrays(subset, "target")
    h0 = metric_bundle(h0_calls, h0_bytes, target_calls, target_bytes)
    candidate = metric_bundle(calls[indices], logical_bytes[indices], target_calls, target_bytes)
    return h0, candidate, compare_to_h0(candidate, h0)


def fit_predict_oof(train: list[dict[str, str]], config: dict[str, Any], folds: dict[str, int]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    raw_calls = np.zeros((len(train), 12), dtype=np.float64); raw_bytes = np.zeros((len(train), 12), dtype=np.float64); epochs: list[int] = []
    for fold in range(4):
        fit_indices = [i for i, row in enumerate(train) if folds[row["profile_id"]] != fold]
        hold_indices = [i for i, row in enumerate(train) if folds[row["profile_id"]] == fold]
        if config["scope"] == "shared":
            fit_rows = [train[i] for i in fit_indices]; hold_rows = [train[i] for i in hold_indices]
            transform = fit_transform(fit_rows, config["feature_mode"])
            model, audit = fit_model(transform_inputs(fit_rows, transform), transform_targets(fit_rows, transform), config, 540000 + fold, validation=(transform_inputs(hold_rows, transform), transform_targets(hold_rows, transform)))
            calls, bytes_ = predict_histograms(hold_rows, transform, [model]); raw_calls[hold_indices] = calls; raw_bytes[hold_indices] = bytes_; epochs.append(int(audit["best_epoch"]))
        else:
            for model_id in MODEL_IDS:
                fit_indices_model = [i for i in fit_indices if train[i]["model"] == model_id]
                hold_indices_model = [i for i in hold_indices if train[i]["model"] == model_id]
                if not hold_indices_model:
                    continue
                fit_rows = [train[i] for i in fit_indices_model]; hold_rows = [train[i] for i in hold_indices_model]
                transform = fit_transform(fit_rows, config["feature_mode"])
                model, audit = fit_model(transform_inputs(fit_rows, transform), transform_targets(fit_rows, transform), config, 540000 + fold * 10 + MODEL_IDS.index(model_id), validation=(transform_inputs(hold_rows, transform), transform_targets(hold_rows, transform)))
                calls, bytes_ = predict_histograms(hold_rows, transform, [model]); raw_calls[hold_indices_model] = calls; raw_bytes[hold_indices_model] = bytes_; epochs.append(int(audit["best_epoch"]))
    return raw_calls, raw_bytes, epochs


def choose_alphas(train: list[dict[str, str]], raw_calls: np.ndarray, raw_bytes: np.ndarray, h0_metrics: dict[str, float], alpha_grid: list[float]) -> tuple[dict[str, float], dict[str, Any], list[dict[str, Any]]]:
    alpha_by_model: dict[str, float] = {}; audits: dict[str, Any] = {}; rows: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        indices = [i for i, row in enumerate(train) if row["model"] == model_id]
        h0, _candidate_unused, _comparison_unused = subset_metrics(train, raw_calls, raw_bytes, indices)
        choices = []
        for alpha in alpha_grid:
            calls, bytes_ = shrink([train[i] for i in indices], raw_calls[indices], raw_bytes[indices], {model_id: alpha})
            target_calls, target_bytes = histogram_arrays([train[i] for i in indices], "target")
            candidate = metric_bundle(calls, bytes_, target_calls, target_bytes); comparison = compare_to_h0(candidate, h0)
            accepted = strict_h0(comparison)
            row = {"model": model_id, "alpha": alpha, "strict_h0": accepted, "composite_ratio": comparison["composite_ratio"], "absolute_score": score(candidate), **candidate, **{f"ratio_{key}": comparison["metric_ratios_to_h0"][key] for key in SCORE_KEYS}}
            rows.append(row); choices.append((not accepted, score(candidate), alpha, comparison, candidate))
        choices.sort(key=lambda value: value[:3]); rejected, _score, alpha, comparison, candidate = choices[0]
        alpha_by_model[model_id] = float(alpha); audits[model_id] = {"alpha": alpha, "strict_h0": not rejected, "metrics": candidate, "comparison": comparison}
    return alpha_by_model, audits, rows


def development_audits(rows: list[dict[str, str]], calls: np.ndarray, bytes_: np.ndarray) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bool]:
    target_calls, target_bytes = histogram_arrays(rows, "target"); h0_calls, h0_bytes = histogram_arrays(rows, "h0")
    h0 = metric_bundle(h0_calls, h0_bytes, target_calls, target_bytes); candidate = metric_bundle(calls, bytes_, target_calls, target_bytes); overall = compare_to_h0(candidate, h0)
    models: dict[str, Any] = {}; model_gate = True
    for model_id in MODEL_IDS:
        indices = [i for i, row in enumerate(rows) if row["model"] == model_id]
        h0m, dm, comp = subset_metrics(rows, calls, bytes_, indices)
        gate = strict_h0(comp) and target_gate(dm, histogram_limit=0.15, total_limit=0.05)
        models[model_id] = {"h0": h0m, "h0_plus_dnn_refined": dm, **comp, "target_guard": gate}; model_gate &= gate
    segments: dict[str, Any] = {}; segment_gate = True
    for segment in SEGMENTS:
        indices = [i for i, row in enumerate(rows) if row["segment"] == segment]
        h0s, ds, comp = subset_metrics(rows, calls, bytes_, indices)
        gate = strict_h0(comp) and target_gate(ds, histogram_limit=0.15, total_limit=0.05)
        segments[segment] = {"h0": h0s, "h0_plus_dnn_refined": ds, **comp, "target_guard": gate}; segment_gate &= gate
    overall_gate = strict_h0(overall) and target_gate(candidate, histogram_limit=0.10, total_limit=0.05)
    return {"h0": h0, "h0_plus_dnn_refined": candidate, **overall, "target_gate": overall_gate}, models, segments, bool(overall_gate and model_gate and segment_gate)


def fit_final_bundle(train: list[dict[str, str]], config: dict[str, Any], epochs: int) -> dict[str, Any]:
    if config["scope"] == "shared":
        transform = fit_transform(train, config["feature_mode"]); models = []
        for seed in [54001, 54002, 54003]:
            model, _audit = fit_model(transform_inputs(train, transform), transform_targets(train, transform), config, seed, fixed_epochs=epochs); models.append(model_to_json(model))
        return {"scope": "shared", "transform": transform, "models": models}
    per_model: dict[str, Any] = {}
    for model_id in MODEL_IDS:
        model_rows = [row for row in train if row["model"] == model_id]; transform = fit_transform(model_rows, config["feature_mode"]); models = []
        for seed in [54001, 54002, 54003]:
            model, _audit = fit_model(transform_inputs(model_rows, transform), transform_targets(model_rows, transform), config, seed, fixed_epochs=epochs); models.append(model_to_json(model))
        per_model[model_id] = {"transform": transform, "models": models}
    return {"scope": "per_model", "per_model": per_model}


def _model_from_json(value: dict[str, Any]) -> dict[str, Any]:
    return {"weights": [np.asarray(item, dtype=np.float64) for item in value["weights"]], "biases": [np.asarray(item, dtype=np.float64) for item in value["biases"]]}


def predict_bundle(rows: list[dict[str, str]], bundle: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    calls = np.zeros((len(rows), 12), dtype=np.float64); bytes_ = np.zeros((len(rows), 12), dtype=np.float64)
    if bundle["scope"] == "shared":
        return predict_histograms(rows, bundle["transform"], [_model_from_json(model) for model in bundle["models"]])
    for model_id in MODEL_IDS:
        indices = [i for i, row in enumerate(rows) if row["model"] == model_id]; subset = [rows[i] for i in indices]; spec = bundle["per_model"][model_id]
        predicted_calls, predicted_bytes = predict_histograms(subset, spec["transform"], [_model_from_json(model) for model in spec["models"]]); calls[indices] = predicted_calls; bytes_[indices] = predicted_bytes
    return calls, bytes_


def prediction_rows(rows: list[dict[str, str]], calls: np.ndarray, bytes_: np.ndarray, method: str) -> list[dict[str, Any]]:
    output = []
    for row, cv, bv in zip(rows, calls, bytes_):
        value = {name: row[name] for name in ("profile_id", "split_role", "source", "segment", "source_split", "window_id", "cutoff_ms", "model")}; value["method"] = method
        for i in range(12): value[f"predicted_calls_bin_{i:02d}"] = float(cv[i]); value[f"predicted_logical_bytes_bin_{i:02d}"] = float(bv[i])
        output.append(value)
    return output


def bin_bias_rows(rows: list[dict[str, str]], calls: np.ndarray, bytes_: np.ndarray, method: str) -> list[dict[str, Any]]:
    target_calls, target_bytes = histogram_arrays(rows, "target")
    output: list[dict[str, Any]] = []
    for kind, predicted, target in (("calls", calls, target_calls), ("logical_bytes", bytes_, target_bytes)):
        for index in range(12):
            target_sum = float(target[:, index].sum()); predicted_sum = float(predicted[:, index].sum())
            output.append({"method": method, "kind": kind, "bin": index, "target_sum": target_sum, "predicted_sum": predicted_sum, "signed_bias": predicted_sum - target_sum, "relative_signed_bias": (predicted_sum - target_sum) / max(target_sum, 1e-12)})
    return output


def run(expected: str, output: Path) -> dict[str, Any]:
    preflight = phase54_preflight.run_checks(expected); contract = load_json(HERE / "experiment.json")
    if output.exists():
        raise RuntimeError(f"refuse overwrite: {output}")
    source = repo_root() / contract["pinned_inputs"][2]["path"]
    rows = read_csv_gz(source); validate_rows(rows)
    train = [row for row in rows if row["split_role"] == "expanded_train"]; validation = [row for row in rows if row["split_role"] == "expanded_validation"]
    folds = fold_map(train); alpha_grid = [float(value) for value in contract["predictor_contract"]["alpha_grid"]]
    candidate_configs = contract["predictor_contract"]["candidate_grid"]; candidates: list[dict[str, Any]] = []; candidate_rows: list[dict[str, Any]] = []
    for candidate_index, config in enumerate(candidate_configs):
        raw_calls, raw_bytes, epochs = fit_predict_oof(train, config, folds)
        h0_train_calls, h0_train_bytes = histogram_arrays(train, "h0"); target_train_calls, target_train_bytes = histogram_arrays(train, "target")
        h0_train = metric_bundle(h0_train_calls, h0_train_bytes, target_train_calls, target_train_bytes)
        alpha_by_model, alpha_audits, alpha_rows = choose_alphas(train, raw_calls, raw_bytes, h0_train, alpha_grid)
        oof_calls, oof_bytes = shrink(train, raw_calls, raw_bytes, alpha_by_model); overall, models, segments, all_target = development_audits(train, oof_calls, oof_bytes)
        oof_protection = strict_h0(overall) and all(value["target_guard"] or strict_h0(value) for value in models.values())
        candidate_row = {"candidate_id": config["candidate_id"], "scope": config["scope"], "loss_mode": config["loss_mode"], "oof_target": all_target, "oof_protection": oof_protection, "oof_score": score(overall["h0_plus_dnn_refined"]), "oof_composite_ratio": overall["composite_ratio"], "oof_calls_histogram_wape": overall["h0_plus_dnn_refined"]["calls_histogram_wape"], "oof_bytes_histogram_wape": overall["h0_plus_dnn_refined"]["bytes_histogram_wape"], "epochs_median": int(statistics.median(epochs)), "alpha_by_model": json.dumps(alpha_by_model, sort_keys=True)}
        candidate_rows.append(candidate_row)
        candidates.append({"config": config, "raw_calls": raw_calls, "raw_bytes": raw_bytes, "alpha_by_model": alpha_by_model, "alpha_audits": alpha_audits, "alpha_rows": alpha_rows, "overall": overall, "models": models, "segments": segments, "oof_target": all_target, "oof_protection": oof_protection, "epochs": epochs, "sort": (not all_target, not oof_protection, score(overall["h0_plus_dnn_refined"]), config["candidate_id"])})
    candidates.sort(key=lambda value: value["sort"]); selected = candidates[0]; config = selected["config"]; selected_epochs = int(np.clip(round(statistics.median(selected["epochs"])), 100, int(config["max_epochs"])))
    bundle = fit_final_bundle(train, config, selected_epochs); validation_raw_calls, validation_raw_bytes = predict_bundle(validation, bundle); alpha_by_model = selected["alpha_by_model"]; validation_calls, validation_bytes = shrink(validation, validation_raw_calls, validation_raw_bytes, alpha_by_model)
    overall, model_audits, segment_audits, target_met = development_audits(validation, validation_calls, validation_bytes)
    output.mkdir(parents=True)
    write_csv(output / "analysis/candidate_oof_metrics.csv", candidate_rows)
    write_json(output / "analysis/oof_selection.json", {"selected_candidate": config, "selected_epochs": selected_epochs, "alpha_by_model": alpha_by_model, "oof_overall": selected["overall"], "oof_models": selected["models"], "oof_segments": selected["segments"], "oof_target": selected["oof_target"], "oof_protection": selected["oof_protection"], "candidate_count": len(candidates)})
    write_csv(output / "analysis/development_validation_metrics.csv", [{"method": "h0", **overall["h0"], "composite_ratio_to_h0": 1.0, "formal_target_gate": False}, {"method": "h0_plus_dnn_refined", **overall["h0_plus_dnn_refined"], "composite_ratio_to_h0": overall["composite_ratio"], "formal_target_gate": overall["target_gate"]}])
    write_json(output / "analysis/model_validation.json", model_audits); write_json(output / "analysis/segment_validation.json", segment_audits)
    h0_calls, h0_bytes = histogram_arrays(validation, "h0")
    write_csv(output / "analysis/development_bin_bias.csv", bin_bias_rows(validation, h0_calls, h0_bytes, "h0") + bin_bias_rows(validation, validation_calls, validation_bytes, "h0_plus_dnn_refined"))
    write_csv_gz(output / "predictions/development_validation_predictions.csv.gz", prediction_rows(validation, h0_calls, h0_bytes, "h0") + prediction_rows(validation, validation_calls, validation_bytes, "h0_plus_dnn_refined"))
    checkpoint = {"schema_version": "phase54-pd-histogram-refinement-checkpoint-v1", "workflow_commit": expected, "selected_candidate": config, "selected_epochs": selected_epochs, "alpha_by_model": alpha_by_model, "ensemble_seeds": contract["predictor_contract"]["ensemble_seeds"], "bundle": bundle, "phase50_blind_accessed": False, "complete_requests_accessed": False}
    write_json_gz(output / "checkpoints/pd_histogram_refinement.json.gz", checkpoint)
    write_json(output / "audit/input_freeze.json", preflight); write_json(output / "audit/training.json", {"fold_assignment": folds, "candidate_count": len(candidates), "selected_candidate_id": config["candidate_id"], "selected_epochs": selected_epochs, "alpha_by_model": alpha_by_model, "oof_target": selected["oof_target"], "oof_protection": selected["oof_protection"], "development_target_met": target_met, "phase50_blind_accessed": False, "complete_requests_accessed": False})
    write_json(output / "audit/environment.json", {**environment_record(), "gpu_used": False, "network_used": False, "raw_accessed": False, "phase50_blind_accessed": False, "complete_requests_accessed": False, "training_used": True})
    summary = {"schema_version": "phase54-pd-histogram-accuracy-refinement-result-v1", "status": "PASS", "workflow_commit": expected, "completed_at_utc": utc_now(), "counts": {"profiles": 1200, "train_profiles": 960, "validation_profiles": 240, "models": 6, "example_rows": 7200, "train_rows": 5760, "validation_rows": 1440, "complete_request_rows_in_git": 0}, "selected": {"candidate_id": config["candidate_id"], "scope": config["scope"], "loss_mode": config["loss_mode"], "epochs": selected_epochs, "alpha_by_model": alpha_by_model}, "gates": {"oof_target": selected["oof_target"], "oof_protection": selected["oof_protection"], "development_overall": overall["target_gate"], "development_all_models": all(value["target_guard"] for value in model_audits.values()), "development_all_segments": all(value["target_guard"] for value in segment_audits.values()), "target_met": target_met, "phase55_permitted": target_met}, "development_validation": overall, "models": model_audits, "segments": segment_audits, "scientific_outcome": "DEVELOPMENT_TARGET_MET" if target_met else "DEVELOPMENT_TARGET_NOT_MET", "proved": "Phase48 development-only candidate selection without Phase50 blind access", "not_proved": "fresh blind generalization, unseen-model extrapolation, physical communication time, placement, latency or online scheduling"}
    write_json(output / "summary.json", summary)
    (output / "README.md").write_text(f"# Phase54：PD直方图精度改进\n\n状态：`PASS`（开发流程完整）。候选 `{config['candidate_id']}`；OOF protection={selected['oof_protection']}；开发集10%目标={target_met}。\n\n本结果没有读取Phase50 blind、raw或完整请求，不能当作新的盲测证据。只有 `phase55_permitted=true` 才能冻结候选进入后续一次性blind评估。\n", encoding="utf-8")
    (output / "logs").mkdir(); (output / "logs/runtime.log").write_text(f"completed={utc_now()} workflow_commit={expected}\ncandidate={config['candidate_id']} scope={config['scope']} loss={config['loss_mode']} epochs={selected_epochs}\noof_target={selected['oof_target']} oof_protection={selected['oof_protection']} development_target={target_met}\ngpu=false network=false phase50_blind=false complete_requests=false\n", encoding="utf-8")
    (output / "DONE").write_text("PASS\n", encoding="utf-8"); refresh_manifest(output); return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--expected-workflow-commit", required=True); parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase54_pd_histogram_accuracy_refinement")
    args = parser.parse_args(); print(json.dumps(run(args.expected_workflow_commit, args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
