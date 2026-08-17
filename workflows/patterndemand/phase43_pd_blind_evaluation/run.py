#!/usr/bin/env python3
"""Open Phase43 blind targets only after R42 predictions are frozen."""

from __future__ import annotations

import argparse
import csv
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
from metrics import compare_to_h0, metric_bundle  # noqa: E402
from model import read_csv_gz, write_csv_gz  # noqa: E402
from preflight import run_checks  # noqa: E402
from prepare_bundle import reconstruct_profile, reproduce_blind_selection  # noqa: E402
from prepare_phase15_trace_windows import BURST_FILES, MOONCAKE_FILES, load_segment  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def feature_difference(saved: dict[str, str], generated: dict[str, Any]) -> dict[str, Any]:
    if set(saved) != set(generated): return {"schema_exact": False, "identifiers_exact": False, "max_absolute_difference": float("inf")}
    identifiers = {"profile_id", "split_role", "source", "segment", "source_split", "window_id", "cutoff_ms", "model"}
    identifiers_exact = all(str(saved[name]) == str(generated[name]) for name in identifiers)
    differences = [abs(float(saved[name]) - float(generated[name])) for name in saved if name not in identifiers]
    return {"schema_exact": True, "identifiers_exact": identifiers_exact, "max_absolute_difference": max(differences, default=0.0)}


def frozen_arrays(predictions: list[dict[str, str]], profile_ids: list[str], method: str) -> tuple[np.ndarray, np.ndarray]:
    by_id = {row["profile_id"]: row for row in predictions if row["method"] == method}
    if set(by_id) != set(profile_ids): raise RuntimeError(f"frozen prediction IDs differ for {method}")
    calls = np.asarray([[float(by_id[profile_id][f"predicted_calls_bin_{index:02d}"]) for index in range(12)] for profile_id in profile_ids])
    logical_bytes = np.asarray([[float(by_id[profile_id][f"predicted_logical_bytes_bin_{index:02d}"]) for index in range(12)] for profile_id in profile_ids])
    return calls, logical_bytes


def target_arrays(targets: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    calls = np.asarray([[float(row[f"target_calls_bin_{index:02d}"]) for index in range(12)] for row in targets])
    logical_bytes = np.asarray([[float(row[f"target_logical_bytes_bin_{index:02d}"]) for index in range(12)] for row in targets])
    return calls, logical_bytes


def per_profile_rows(profile_ids: list[str], methods: dict[str, tuple[np.ndarray, np.ndarray]], targets: tuple[np.ndarray, np.ndarray]) -> list[dict[str, Any]]:
    output = []
    for method, (calls, logical_bytes) in methods.items():
        for index, profile_id in enumerate(profile_ids):
            metrics = metric_bundle(calls[index:index+1], logical_bytes[index:index+1], targets[0][index:index+1], targets[1][index:index+1])
            output.append({"profile_id": profile_id, "method": method, **metrics})
    return output


def paired_bootstrap(per_profile: list[dict[str, Any]]) -> dict[str, Any]:
    by_method = {method: {row["profile_id"]: row for row in per_profile if row["method"] == method} for method in ("h0", "h0_plus_dnn_residual")}
    profile_ids = sorted(by_method["h0"])
    rng = np.random.default_rng(20260817); draws = 10000
    output: dict[str, Any] = {"schema_version": "phase43-paired-bootstrap-v1", "seed": 20260817, "draws": draws, "difference": "H0 error minus H0+DNN error; positive favors DNN"}
    for key in ("mean_profile_calls_l1", "mean_profile_bytes_l1"):
        paired = np.asarray([float(by_method["h0"][profile_id][key]) - float(by_method["h0_plus_dnn_residual"][profile_id][key]) for profile_id in profile_ids])
        sampled = paired[rng.integers(0, len(paired), size=(draws, len(paired)))].mean(axis=1)
        output[key] = {"observed_mean_difference": float(paired.mean()), "ci95_low": float(np.quantile(sampled, 0.025)), "ci95_high": float(np.quantile(sampled, 0.975)), "fraction_bootstrap_positive": float(np.mean(sampled > 0))}
    return output


def run(expected: str, raw_dir: Path, output: Path) -> dict[str, Any]:
    preflight = run_checks(expected, raw_dir)
    if output.exists(): raise RuntimeError(f"refuse to overwrite output: {output}")
    phase41 = load_json(P41 / "experiment.json"); feature_contract = load_json(P41 / "feature_contract.json")
    model_contract = load_json(repo_root() / "experiment-results/phase41_pd_full_window_dataset/contracts/model_contract.json")
    frozen_features = read_csv_gz(repo_root() / "experiment-results/phase41_pd_full_window_dataset/dataset/pd_blind_target_free_features.csv.gz")
    frozen_predictions = read_csv_gz(repo_root() / "experiment-results/phase42_pd_residual_training/predictions/blind_frozen_predictions.csv.gz")
    forbidden = [name for row in frozen_predictions for name in row if name.startswith("target_") or name.startswith("residual_")]
    if forbidden: raise RuntimeError(f"R42 blind prediction leakage: {sorted(set(forbidden))}")
    selection = reproduce_blind_selection(phase41)
    rows = [{**item, "split_role": item["role"]} for item in selection["rows"]]
    file_by_segment = {segment: raw_dir.expanduser().resolve() / name for name, (segment, _split) in {**BURST_FILES, **MOONCAKE_FILES}.items()}
    source_arrays = {segment: load_segment(file_by_segment[segment]) for segment in sorted({row["segment"] for row in rows})}
    frozen_by_id = {row["profile_id"]: row for row in frozen_features}
    targets = []; reconstruction = []; reconstructed = []; total_requests = 0
    kv_bytes = int(model_contract["derived"]["kv_bytes_per_page"])
    for row in rows:
        profile, requests = reconstruct_profile(row, source_arrays)
        feature, no_target = profile_example_rows(profile=profile, requests=None, contract=phase41, feature_contract=feature_contract, kv_bytes_per_page=kv_bytes)
        assert no_target is None
        difference = feature_difference(frozen_by_id[profile["profile_id"]], feature)
        difference.update({"profile_id": profile["profile_id"], "request_count": len(requests)})
        reconstruction.append(difference); reconstructed.append((profile, requests)); total_requests += len(requests)
    tolerance = float(load_json(HERE / "experiment.json")["blind_contract"]["feature_reconstruction_tolerance_lt"])
    if total_requests != 2887 or len(reconstructed) != 12 or not all(row["schema_exact"] and row["identifiers_exact"] and float(row["max_absolute_difference"]) < tolerance for row in reconstruction):
        raise RuntimeError({"total_requests": total_requests, "reconstruction": reconstruction})
    for profile, requests in reconstructed:
        _example, target = profile_example_rows(profile=profile, requests=[tuple(pair) for pair in requests], contract=phase41, feature_contract=feature_contract, kv_bytes_per_page=kv_bytes)
        assert target is not None
        targets.append(target)
    profile_ids = [row["profile_id"] for row in targets]
    target_calls, target_bytes = target_arrays(targets)
    methods = {method: frozen_arrays(frozen_predictions, profile_ids, method) for method in ("h0", "h0_plus_dnn_residual")}
    aggregates = {method: metric_bundle(value[0], value[1], target_calls, target_bytes) for method, value in methods.items()}
    comparison = compare_to_h0(aggregates["h0_plus_dnn_residual"], aggregates["h0"])
    per_profile = per_profile_rows(profile_ids, methods, (target_calls, target_bytes)); bootstrap = paired_bootstrap(per_profile)
    output.mkdir(parents=True)
    write_csv_gz(output / "labels/pd_blind_hfull_targets.csv.gz", targets)
    aggregate_rows = []
    for method in ("h0", "h0_plus_dnn_residual"):
        aggregate_rows.append({"method": method, **aggregates[method], "composite_ratio_to_h0": 1.0 if method == "h0" else comparison["composite_ratio"], "scientific_outcome": "BASELINE" if method == "h0" else comparison["outcome"]})
    write_csv(output / "analysis/aggregate_metrics.csv", aggregate_rows)
    write_csv_gz(output / "analysis/per_profile_metrics.csv.gz", per_profile)
    write_json(output / "analysis/paired_bootstrap.json", bootstrap)
    write_json(output / "audit/input_freeze.json", preflight)
    write_json(output / "audit/target_generation.json", {"schema_version": "phase43-target-generation-audit-v1", "workflow_commit": expected, "prediction_parent_result_commit": "88dd1a8f5a4b9452e226118ade270aa3eb6fed7e", "profiles": 12, "complete_requests_used_outside_git": total_requests, "full_request_rows_committed": 0, "selection_checks": selection["checks"], "reconstruction": reconstruction})
    write_json(output / "audit/environment.json", {**environment_record(), "numpy": np.__version__, "gpu_used": False, "checkpoint_loaded": False, "prediction_recomputed": False, "network_used": False, "raw_mutated": False})
    write_json(output / "contracts/experiment.json", load_json(HERE / "experiment.json"))
    summary = {"schema_version": "phase43-pd-blind-evaluation-result-v1", "status": "PASS", "workflow_commit": expected, "completed_at_utc": utc_now(), "counts": {"blind_profiles": 12, "blind_complete_requests": total_requests, "target_rows": 12, "frozen_prediction_rows": 24, "aggregate_metric_rows": 2, "per_profile_metric_rows": 24, "full_request_rows_in_git": 0}, "blind_metrics": {"h0": aggregates["h0"], "h0_plus_dnn_residual": aggregates["h0_plus_dnn_residual"], **comparison}, "paired_bootstrap": bootstrap, "proved": "one-time 12-profile Qwen3 pure-PD blind evaluation against targets opened after R42 prediction freeze", "not_proved": "other-model generalization, physical RDMA time, placement, latency or online scheduling"}
    write_json(output / "summary.json", summary)
    (output / "README.md").write_text(f"# Phase43：纯PD一次性blind评估\n\n状态：`PASS`。R42冻结以后才从受保护raw重建12个blind完整窗口，共{total_requests}个请求；Git只保存12行Hfull直方图标签，不保存完整请求。\n\nH0+DNN相对H0的blind composite ratio为`{comparison['composite_ratio']:.6f}`，科学结论为`{comparison['outcome']}`。无论正负，本阶段都没有重训、调参、加载checkpoint或重算预测。\n", encoding="utf-8")
    (output / "logs").mkdir(); (output / "logs/runtime.log").write_text(f"completed={utc_now()} workflow_commit={expected}\nprofiles=12 complete_requests={total_requests} frozen_predictions=24\noutcome={comparison['outcome']} composite_ratio={comparison['composite_ratio']:.12f}\ngpu=false training=false checkpoint_loaded=false prediction_recomputed=false raw_committed=false\n", encoding="utf-8")
    (output / "DONE").write_text("PASS\n", encoding="utf-8")
    refresh_manifest(output); return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--expected-workflow-commit", required=True); parser.add_argument("--raw-dir", type=Path, required=True); parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase43_pd_blind_evaluation")
    args = parser.parse_args(); print(json.dumps(run(args.expected_workflow_commit, args.raw_dir, args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
