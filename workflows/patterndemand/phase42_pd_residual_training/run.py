#!/usr/bin/env python3
"""Train, evaluate and freeze the Phase42 pure-PD residual predictor."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import environment_record, load_json, refresh_manifest, repo_root, utc_now, write_json  # noqa: E402
from metrics import compare_to_h0, metric_bundle  # noqa: E402
from model import (  # noqa: E402
    fit_model, fit_transform, histogram_arrays, model_to_json, predict_histograms,
    read_csv_gz, transform_inputs, transform_targets, write_csv_gz, write_json_gz,
)
from preflight import run_checks  # noqa: E402


IDENTIFIERS = ("profile_id", "split_role", "source", "segment", "source_split", "window_id", "cutoff_ms", "model")


def fold_map(rows: list[dict[str, str]], folds: int) -> dict[str, int]:
    result: dict[str, int] = {}
    by_segment: dict[str, list[str]] = {}
    for row in rows: by_segment.setdefault(row["segment"], []).append(row["profile_id"])
    for segment, values in sorted(by_segment.items()):
        ordered = sorted(values, key=lambda value: hashlib.sha256(f"phase42-fold:{segment}:{value}".encode()).hexdigest())
        for index, profile_id in enumerate(ordered): result[profile_id] = index % folds
    return result


def arrays(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h0_calls, h0_bytes = histogram_arrays(rows, "h0")
    target_calls, target_bytes = histogram_arrays(rows, "target")
    return h0_calls, h0_bytes, target_calls, target_bytes


def prediction_rows(rows: list[dict[str, str]], calls: np.ndarray, logical_bytes: np.ndarray, method: str, *, include_target: bool) -> list[dict[str, Any]]:
    output = []
    for row, call_vector, byte_vector in zip(rows, calls, logical_bytes):
        value: dict[str, Any] = {name: row[name] for name in IDENTIFIERS}
        value["method"] = method
        value["predicted_total_calls_per_1000"] = float(call_vector.sum())
        value["predicted_total_logical_bytes_per_1000"] = float(byte_vector.sum())
        for index in range(12): value[f"predicted_calls_bin_{index:02d}"] = float(call_vector[index])
        for index in range(12): value[f"predicted_logical_bytes_bin_{index:02d}"] = float(byte_vector[index])
        if include_target:
            for kind in ("calls", "logical_bytes"):
                for index in range(12): value[f"target_{kind}_bin_{index:02d}"] = float(row[f"target_{kind}_bin_{index:02d}"])
        output.append(value)
    return output


def metric_rows(h0: dict[str, float], dnn: dict[str, float], comparison: dict[str, Any], split: str) -> list[dict[str, Any]]:
    result = []
    for method, metrics in (("h0", h0), ("h0_plus_dnn_residual", dnn)):
        row: dict[str, Any] = {"split": split, "method": method, **metrics}
        row["composite_ratio_to_h0"] = 1.0 if method == "h0" else comparison["composite_ratio"]
        row["outcome"] = "BASELINE" if method == "h0" else comparison["outcome"]
        result.append(row)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def run(expected: str, output: Path) -> dict[str, Any]:
    preflight = run_checks(expected)
    contract = load_json(HERE / "experiment.json")
    if output.exists(): raise RuntimeError(f"refuse to overwrite output: {output}")
    base = repo_root() / "experiment-results/phase41_pd_full_window_dataset/dataset"
    rows = read_csv_gz(base / "pd_development_h0_residual_examples.csv.gz")
    blind = read_csv_gz(base / "pd_blind_target_free_features.csv.gz")
    train = [row for row in rows if row["split_role"] == "development_train"]
    validation = [row for row in rows if row["split_role"] == "development_validation"]
    if (len(train), len(validation), len(blind)) != (75, 19, 12): raise RuntimeError("unexpected Phase41 split")
    forbidden = [name for name in blind[0] if name.startswith("target_") or name.startswith("residual_")]
    if forbidden: raise RuntimeError(f"blind target leakage: {forbidden}")
    folds = int(contract["predictor"]["folds"]); mapping = fold_map(train, folds)
    candidates = []
    best_payload: tuple[float, str, dict[str, Any], list[int], np.ndarray, np.ndarray] | None = None
    for candidate_index, config in enumerate(contract["predictor"]["candidate_grid"]):
        oof_calls = np.zeros((len(train), 12)); oof_bytes = np.zeros((len(train), 12)); best_epochs = []
        for fold in range(folds):
            fit_indices = [index for index, row in enumerate(train) if mapping[row["profile_id"]] != fold]
            val_indices = [index for index, row in enumerate(train) if mapping[row["profile_id"]] == fold]
            fit_rows = [train[index] for index in fit_indices]; val_rows = [train[index] for index in val_indices]
            transform = fit_transform(fit_rows)
            x_fit = transform_inputs(fit_rows, transform); y_fit = transform_targets(fit_rows, transform)
            x_val = transform_inputs(val_rows, transform); y_val = transform_targets(val_rows, transform)
            model, audit = fit_model(x_fit, y_fit, config, 420000 + candidate_index * 1000 + fold, validation=(x_val, y_val))
            calls, logical_bytes = predict_histograms(val_rows, transform, [model])
            oof_calls[val_indices] = calls; oof_bytes[val_indices] = logical_bytes
            best_epochs.append(int(audit["best_epoch"]))
        h0_calls, h0_bytes, target_calls, target_bytes = arrays(train)
        h0_metrics = metric_bundle(h0_calls, h0_bytes, target_calls, target_bytes)
        dnn_metrics = metric_bundle(oof_calls, oof_bytes, target_calls, target_bytes)
        comparison = compare_to_h0(dnn_metrics, h0_metrics)
        candidate = {"candidate_id": config["candidate_id"], "mean_best_epoch": float(np.mean(best_epochs)), "median_best_epoch": int(statistics.median(best_epochs)), **{f"dnn_{key}": value for key, value in dnn_metrics.items()}, **{f"h0_{key}": value for key, value in h0_metrics.items()}, "composite_ratio": comparison["composite_ratio"], "outcome": comparison["outcome"]}
        candidates.append(candidate)
        key = (float(comparison["composite_ratio"]), str(config["candidate_id"]))
        payload = (key[0], key[1], config, best_epochs, oof_calls, oof_bytes)
        if best_payload is None or key < (best_payload[0], best_payload[1]): best_payload = payload
    assert best_payload is not None
    selected = best_payload[2]; selected_epochs = int(np.clip(round(statistics.median(best_payload[3])), 100, int(selected["max_epochs"])))
    final_transform = fit_transform(train)
    x_train = transform_inputs(train, final_transform); y_train = transform_targets(train, final_transform)
    final_models = []; final_audits = []
    for seed in contract["predictor"]["final_ensemble_seeds"]:
        model, audit = fit_model(x_train, y_train, selected, int(seed), fixed_epochs=selected_epochs)
        final_models.append(model); final_audits.append({"seed": seed, **audit})
    dnn_val_calls, dnn_val_bytes = predict_histograms(validation, final_transform, final_models)
    h0_val_calls, h0_val_bytes, target_val_calls, target_val_bytes = arrays(validation)
    h0_metrics = metric_bundle(h0_val_calls, h0_val_bytes, target_val_calls, target_val_bytes)
    dnn_metrics = metric_bundle(dnn_val_calls, dnn_val_bytes, target_val_calls, target_val_bytes)
    comparison = compare_to_h0(dnn_metrics, h0_metrics)
    blind_dnn_calls, blind_dnn_bytes = predict_histograms(blind, final_transform, final_models)
    blind_h0_calls, blind_h0_bytes = histogram_arrays(blind, "h0")
    output.mkdir(parents=True)
    write_csv(output / "analysis/candidate_metrics.csv", sorted(candidates, key=lambda row: (float(row["composite_ratio"]), row["candidate_id"])))
    write_csv(output / "analysis/development_validation_metrics.csv", metric_rows(h0_metrics, dnn_metrics, comparison, "development_validation"))
    write_csv_gz(output / "predictions/development_validation_predictions.csv.gz", prediction_rows(validation, h0_val_calls, h0_val_bytes, "h0", include_target=True) + prediction_rows(validation, dnn_val_calls, dnn_val_bytes, "h0_plus_dnn_residual", include_target=True))
    write_csv_gz(output / "predictions/blind_frozen_predictions.csv.gz", prediction_rows(blind, blind_h0_calls, blind_h0_bytes, "h0", include_target=False) + prediction_rows(blind, blind_dnn_calls, blind_dnn_bytes, "h0_plus_dnn_residual", include_target=False))
    checkpoint = {
        "schema_version": "phase42-pd-numpy-h0-dnn-checkpoint-v1", "workflow_commit": expected,
        "selected_candidate": selected, "selected_epochs": selected_epochs,
        "ensemble_seeds": contract["predictor"]["final_ensemble_seeds"], "transform": final_transform,
        "models": [model_to_json(model) for model in final_models],
        "training_profile_ids": [row["profile_id"] for row in train],
        "forbidden_assets_seen": {"raw": False, "complete_requests": False, "blind_targets": False, "gpu": False},
    }
    write_json_gz(output / "checkpoints/pd_qwen3_h0_dnn_residual.json.gz", checkpoint)
    write_json(output / "audit/input_freeze.json", preflight)
    environment = environment_record(); environment.update({"numpy": np.__version__, "gpu_used": False, "raw_visible": False, "network_used": False})
    write_json(output / "audit/environment.json", environment)
    training_audit = {"schema_version": "phase42-training-audit-v1", "candidate_selection_profiles": 75, "development_validation_profiles": 19, "blind_profiles": 12, "fold_assignment": mapping, "selected_candidate_id": selected["candidate_id"], "selected_epochs": selected_epochs, "final_models": final_audits, "validation_used_for_candidate_selection": False, "blind_target_accessed": False, "deterministic": True}
    write_json(output / "audit/training.json", training_audit)
    summary = {
        "schema_version": "phase42-pd-residual-training-result-v1", "status": "PASS", "workflow_commit": expected,
        "completed_at_utc": utc_now(), "selected_candidate_id": selected["candidate_id"], "selected_epochs": selected_epochs,
        "counts": {"development_train_profiles": 75, "development_validation_profiles": 19, "blind_profiles": 12, "blind_prediction_rows": 24, "blind_target_rows": 0, "candidate_rows": len(candidates), "checkpoint_models": len(final_models)},
        "development_validation": {"h0": h0_metrics, "h0_plus_dnn_residual": dnn_metrics, **comparison},
        "blind_state": "12 H0 and 12 H0+DNN residual predictions frozen before target generation; no blind full requests or targets accessed",
        "proved": "target-isolated deterministic CPU training and pre-target prediction freeze for one Qwen3 pure-PD predictor",
        "not_proved": "blind generalization, other models, physical RDMA cost, placement or online scheduling",
    }
    write_json(output / "summary.json", summary)
    (output / "README.md").write_text(f"# Phase42：纯PD H0+DNN residual训练与预测冻结\n\n状态：`PASS`。仅用75个development_train画像完成候选选择和最终训练，19个development_validation画像只做一次性开发评估。选中`{selected['candidate_id']}`，开发集结论为`{comparison['outcome']}`，composite ratio为`{comparison['composite_ratio']:.6f}`。\n\n12个blind画像的H0与H0+DNN预测已经冻结；未读取完整blind请求或target。Phase43只能在本commit合入后打开target并评分。\n", encoding="utf-8")
    (output / "logs").mkdir(); (output / "logs/runtime.log").write_text(f"completed={utc_now()} workflow_commit={expected}\nselected={selected['candidate_id']} epochs={selected_epochs}\nvalidation_outcome={comparison['outcome']} composite_ratio={comparison['composite_ratio']:.12f}\nblind_predictions=24 blind_targets=0 gpu=false raw=false\n", encoding="utf-8")
    (output / "DONE").write_text("PASS\n", encoding="utf-8")
    refresh_manifest(output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--expected-workflow-commit", required=True)
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase42_pd_residual_training")
    args = parser.parse_args(); print(json.dumps(run(args.expected_workflow_commit, args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
