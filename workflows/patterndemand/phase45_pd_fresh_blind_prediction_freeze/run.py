#!/usr/bin/env python3
"""Freeze target-free Phase45 features and R44 predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
P41 = HERE.parent / "phase41_pd_full_window_dataset"
P42 = HERE.parent / "phase42_pd_residual_training"
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(P42)); sys.path.insert(0, str(P41)); sys.path.insert(0, str(HERE.parents[2] / "scripts")); sys.path.insert(0, str(HERE))
from common import environment_record, load_json, refresh_manifest, repo_root, utc_now, write_json  # noqa: E402
from contracts import profile_example_rows  # noqa: E402
from model import decode_histograms, encode_histograms, histogram_arrays, model_from_json, predict_histograms, read_json_gz, write_csv_gz  # noqa: E402
from preflight import read_csv, run_checks  # noqa: E402
from prepare_bundle import reconstruct_profile  # noqa: E402
from prepare_phase15_trace_windows import BURST_FILES, MOONCAKE_FILES, load_segment  # noqa: E402


IDENTIFIERS = ("profile_id", "split_role", "source", "segment", "source_split", "window_id", "cutoff_ms", "model")


def shrink(rows: list[dict[str, Any]], calls: np.ndarray, logical_bytes: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    h0_calls, h0_bytes = histogram_arrays(rows, "h0")
    baseline = encode_histograms(h0_calls, h0_bytes); predicted = encode_histograms(calls, logical_bytes)
    return decode_histograms(baseline + float(alpha) * (predicted - baseline))


def prediction_rows(rows: list[dict[str, Any]], calls: np.ndarray, logical_bytes: np.ndarray, method: str) -> list[dict[str, Any]]:
    output = []
    for row, call_vector, byte_vector in zip(rows, calls, logical_bytes):
        value = {name: row[name] for name in IDENTIFIERS}; value["method"] = method
        value["predicted_total_calls_per_1000"] = float(call_vector.sum()); value["predicted_total_logical_bytes_per_1000"] = float(byte_vector.sum())
        for index in range(12): value[f"predicted_calls_bin_{index:02d}"] = float(call_vector[index])
        for index in range(12): value[f"predicted_logical_bytes_bin_{index:02d}"] = float(byte_vector[index])
        output.append(value)
    return output


def run(expected: str, raw_dir: Path, output: Path) -> dict[str, Any]:
    preflight = run_checks(expected, raw_dir); contract = load_json(HERE / "experiment.json"); phase41 = load_json(P41 / "experiment.json"); feature_contract = load_json(P41 / "feature_contract.json")
    if output.exists(): raise RuntimeError(f"refuse overwrite: {output}")
    selected = read_csv(repo_root() / contract["selection_contract"]["path"])
    file_by_segment = {segment: raw_dir.expanduser().resolve() / name for name, (segment, _split) in {**BURST_FILES, **MOONCAKE_FILES}.items()}
    arrays = {segment: load_segment(file_by_segment[segment]) for segment in contract["selection_contract"]["segments"]}
    model_contract = load_json(repo_root() / "experiment-results/phase41_pd_full_window_dataset/contracts/model_contract.json"); kv_bytes = int(model_contract["derived"]["kv_bytes_per_page"])
    features = []; profiles = []; total_requests = 0
    for row in selected:
        profile, requests = reconstruct_profile({**row, "split_role": row["role"]}, arrays)
        feature, target = profile_example_rows(profile=profile, requests=None, contract=phase41, feature_contract=feature_contract, kv_bytes_per_page=kv_bytes)
        if target is not None or any(name.startswith("target_") or name.startswith("residual_") for name in feature): raise RuntimeError("target leakage")
        features.append(feature); profiles.append(profile); total_requests += len(requests)
    if len(features) != 300 or total_requests != 115083: raise RuntimeError({"profiles": len(features), "requests": total_requests})
    checkpoint = read_json_gz(repo_root() / "experiment-results/phase44_pd_expanded_protected_training/checkpoints/pd_qwen3_expanded_h0_protected_dnn.json.gz")
    models = [model_from_json(value) for value in checkpoint["models"]]
    raw_calls, raw_bytes = predict_histograms(features, checkpoint["transform"], models)
    dnn_calls, dnn_bytes = shrink(features, raw_calls, raw_bytes, float(checkpoint["selected_alpha"]))
    h0_calls, h0_bytes = histogram_arrays(features, "h0")
    predictions = prediction_rows(features, h0_calls, h0_bytes, "h0") + prediction_rows(features, dnn_calls, dnn_bytes, "h0_plus_dnn_residual")
    output.mkdir(parents=True)
    write_csv_gz(output / "dataset/pd_fresh_blind_target_free_features.csv.gz", features)
    write_csv_gz(output / "profiles/fresh_blind_lowdim_profiles.csv.gz", profiles)
    write_csv_gz(output / "predictions/pd_fresh_blind_frozen_predictions.csv.gz", predictions)
    write_json(output / "audit/input_freeze.json", preflight)
    write_json(output / "audit/prediction_freeze.json", {"schema_version": "phase45-prediction-freeze-audit-v1", "profiles": 300, "prediction_rows": 600, "complete_requests_reconstructed_outside_git": total_requests, "complete_request_rows_committed": 0, "target_rows": 0, "training_used": False, "checkpoint_changed": False, "selected_candidate_id": checkpoint["selected_candidate"]["candidate_id"], "selected_alpha": checkpoint["selected_alpha"], "selected_epochs": checkpoint["selected_epochs"], "ensemble_seeds": checkpoint["ensemble_seeds"]})
    write_json(output / "audit/environment.json", {**environment_record(), "numpy": np.__version__, "gpu_used": False, "network_used": False, "training_used": False, "targets_accessed": False, "raw_mutated": False})
    summary = {"schema_version": "phase45-pd-fresh-blind-prediction-freeze-result-v1", "status": "PASS", "workflow_commit": expected, "completed_at_utc": utc_now(), "counts": {"blind_profiles": 300, "blind_complete_requests_reconstructed_outside_git": total_requests, "frozen_prediction_rows": 600, "target_rows": 0, "complete_request_rows_in_git": 0}, "predictor": {"candidate_id": checkpoint["selected_candidate"]["candidate_id"], "feature_mode": checkpoint["selected_candidate"]["feature_mode"], "alpha": checkpoint["selected_alpha"], "epochs": checkpoint["selected_epochs"], "ensemble_seeds": checkpoint["ensemble_seeds"]}, "blind_state": "target-free features and H0/H0+DNN predictions frozen; no Hfull generated or accessed", "next": "only after R45 formal integration may Phase46 reconstruct the same windows and reveal Hfull once", "proved": "target-isolated fresh blind cohort and exact R44 prediction freeze", "not_proved": "blind accuracy or improvement, Mooncake, other models, physical RDMA cost, placement, latency or online scheduling"}
    write_json(output / "summary.json", summary)
    (output / "README.md").write_text(f"# Phase45：纯PD fresh blind预测冻结\n\n状态：`PASS`。冻结300个历史隔离窗口的低维画像及600行H0/H0+DNN预测，共重建{total_requests}个请求；没有生成Hfull标签，完整请求未进入Git。\n\n使用R44原checkpoint：`{checkpoint['selected_candidate']['candidate_id']}`、alpha=`{checkpoint['selected_alpha']}`、epochs=`{checkpoint['selected_epochs']}`。只有R45正式合入后，Phase46才能一次性揭示标签。\n", encoding="utf-8")
    (output / "logs").mkdir(); (output / "logs/runtime.log").write_text(f"completed={utc_now()} workflow_commit={expected}\nprofiles=300 complete_requests={total_requests} predictions=600 targets=0\ncandidate={checkpoint['selected_candidate']['candidate_id']} alpha={checkpoint['selected_alpha']} epochs={checkpoint['selected_epochs']}\ngpu=false training=false target_access=false raw_committed=false\n", encoding="utf-8")
    (output / "DONE").write_text("PASS\n", encoding="utf-8"); refresh_manifest(output); return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--expected-workflow-commit", required=True); parser.add_argument("--raw-dir", type=Path, required=True); parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase45_pd_fresh_blind_prediction_freeze")
    args = parser.parse_args(); print(json.dumps(run(args.expected_workflow_commit, args.raw_dir, args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
