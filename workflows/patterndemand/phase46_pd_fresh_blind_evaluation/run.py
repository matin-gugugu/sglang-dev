#!/usr/bin/env python3
"""Reveal Phase46 Hfull labels only after exact R45 freeze reconstruction."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent; P41 = HERE.parent / "phase41_pd_full_window_dataset"; P42 = HERE.parent / "phase42_pd_residual_training"; P45 = HERE.parent / "phase45_pd_fresh_blind_prediction_freeze"
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(P42)); sys.path.insert(0, str(P41)); sys.path.insert(0, str(P45)); sys.path.insert(0, str(HERE.parents[2] / "scripts")); sys.path.insert(0, str(HERE))
from common import environment_record, load_json, refresh_manifest, repo_root, utc_now, write_json  # noqa: E402
from contracts import profile_example_rows  # noqa: E402
from metrics import SCORE_KEYS, metric_bundle  # noqa: E402
from model import read_csv_gz, write_csv_gz  # noqa: E402
from preflight import run_checks  # noqa: E402
from prepare_bundle import reconstruct_profile  # noqa: E402
from prepare_phase15_trace_windows import BURST_FILES, MOONCAKE_FILES, load_segment  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source: return list(csv.DictReader(source))


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
    if set(by_id) != set(profile_ids): raise RuntimeError(f"frozen prediction IDs differ: {method}")
    calls = np.asarray([[float(by_id[profile_id][f"predicted_calls_bin_{index:02d}"]) for index in range(12)] for profile_id in profile_ids])
    logical_bytes = np.asarray([[float(by_id[profile_id][f"predicted_logical_bytes_bin_{index:02d}"]) for index in range(12)] for profile_id in profile_ids])
    return calls, logical_bytes


def target_arrays(targets: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    calls = np.asarray([[float(row[f"target_calls_bin_{index:02d}"]) for index in range(12)] for row in targets])
    logical_bytes = np.asarray([[float(row[f"target_logical_bytes_bin_{index:02d}"]) for index in range(12)] for row in targets])
    return calls, logical_bytes


def compare(dnn: dict[str, float], h0: dict[str, float]) -> dict[str, Any]:
    ratios = {key: float(dnn[key]) / max(float(h0[key]), 1e-12) for key in SCORE_KEYS}
    return {"metric_ratios_to_h0": ratios, "composite_ratio": float(np.mean(list(ratios.values()))), "strict_four_metric_gate": all(value < 1.0 for value in ratios.values())}


def per_profile_rows(profile_ids: list[str], methods: dict[str, tuple[np.ndarray, np.ndarray]], targets: tuple[np.ndarray, np.ndarray]) -> list[dict[str, Any]]:
    output = []
    for method, (calls, logical_bytes) in methods.items():
        for index, profile_id in enumerate(profile_ids):
            output.append({"profile_id": profile_id, "method": method, **metric_bundle(calls[index:index + 1], logical_bytes[index:index + 1], targets[0][index:index + 1], targets[1][index:index + 1])})
    return output


def paired_bootstrap(per_profile: list[dict[str, Any]]) -> dict[str, Any]:
    by_method = {method: {row["profile_id"]: row for row in per_profile if row["method"] == method} for method in ("h0", "h0_plus_dnn_residual")}; profile_ids = sorted(by_method["h0"])
    rng = np.random.default_rng(460017); draws = 20000
    output: dict[str, Any] = {"schema_version": "phase46-paired-bootstrap-v1", "seed": 460017, "draws": draws, "difference": "H0 error minus H0+DNN error; positive favors DNN"}
    for key in ("mean_profile_calls_l1", "mean_profile_bytes_l1"):
        paired = np.asarray([float(by_method["h0"][profile_id][key]) - float(by_method["h0_plus_dnn_residual"][profile_id][key]) for profile_id in profile_ids])
        sampled = paired[rng.integers(0, len(paired), size=(draws, len(paired)))].mean(axis=1)
        output[key] = {"observed_mean_difference": float(paired.mean()), "ci95_low": float(np.quantile(sampled, 0.025)), "ci95_high": float(np.quantile(sampled, 0.975)), "fraction_bootstrap_positive": float(np.mean(sampled > 0))}
    output["both_ci95_strictly_positive"] = all(float(output[key]["ci95_low"]) > 0 for key in ("mean_profile_calls_l1", "mean_profile_bytes_l1"))
    return output


def group_comparison(indices: list[int], methods: dict[str, tuple[np.ndarray, np.ndarray]], targets: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
    h0 = metric_bundle(methods["h0"][0][indices], methods["h0"][1][indices], targets[0][indices], targets[1][indices]); dnn = metric_bundle(methods["h0_plus_dnn_residual"][0][indices], methods["h0_plus_dnn_residual"][1][indices], targets[0][indices], targets[1][indices]); comparison = compare(dnn, h0)
    return {"h0": h0, "h0_plus_dnn_residual": dnn, **comparison}


def run(expected: str, raw_dir: Path, output: Path) -> dict[str, Any]:
    preflight = run_checks(expected, raw_dir)
    if output.exists(): raise RuntimeError(f"refuse overwrite: {output}")
    contract = load_json(HERE / "experiment.json"); phase41 = load_json(P41 / "experiment.json"); feature_contract = load_json(P41 / "feature_contract.json")
    frozen_features = read_csv_gz(repo_root() / "experiment-results/phase45_pd_fresh_blind_prediction_freeze/dataset/pd_fresh_blind_target_free_features.csv.gz"); frozen_predictions = read_csv_gz(repo_root() / "experiment-results/phase45_pd_fresh_blind_prediction_freeze/predictions/pd_fresh_blind_frozen_predictions.csv.gz")
    selection = read_csv(repo_root() / "workflows/patterndemand/phase45_pd_fresh_blind_prediction_freeze/selection/fresh_blind_windows.csv"); frozen_by_id = {row["profile_id"]: row for row in frozen_features}
    file_by_segment = {segment: raw_dir.expanduser().resolve() / name for name, (segment, _split) in {**BURST_FILES, **MOONCAKE_FILES}.items()}; arrays = {segment: load_segment(file_by_segment[segment]) for segment in contract["blind_contract"]["segments"]}
    model_contract = load_json(repo_root() / "experiment-results/phase41_pd_full_window_dataset/contracts/model_contract.json"); kv_bytes = int(model_contract["derived"]["kv_bytes_per_page"])
    reconstructed = []; reconstruction = []; total_requests = 0
    for row in selection:
        profile, requests = reconstruct_profile({**row, "split_role": row["role"]}, arrays)
        feature, target = profile_example_rows(profile=profile, requests=None, contract=phase41, feature_contract=feature_contract, kv_bytes_per_page=kv_bytes)
        if target is not None: raise RuntimeError("target opened during reconstruction gate")
        difference = feature_difference(frozen_by_id[profile["profile_id"]], feature); difference.update({"profile_id": profile["profile_id"], "request_count": len(requests)})
        reconstruction.append(difference); reconstructed.append((profile, requests)); total_requests += len(requests)
    tolerance = float(contract["blind_contract"]["feature_reconstruction_tolerance_lt"])
    reconstruction_passed = len(reconstructed) == 300 and total_requests == 115083 and all(row["schema_exact"] and row["identifiers_exact"] and float(row["max_absolute_difference"]) < tolerance for row in reconstruction)
    if not reconstruction_passed: raise RuntimeError({"profiles": len(reconstructed), "requests": total_requests, "reconstruction": reconstruction[:10]})
    targets = []
    for profile, requests in reconstructed:
        _example, target = profile_example_rows(profile=profile, requests=[tuple(pair) for pair in requests], contract=phase41, feature_contract=feature_contract, kv_bytes_per_page=kv_bytes)
        if target is None: raise RuntimeError("teacher target missing")
        targets.append(target)
    profile_ids = [row["profile_id"] for row in targets]; target_values = target_arrays(targets)
    methods = {method: frozen_arrays(frozen_predictions, profile_ids, method) for method in ("h0", "h0_plus_dnn_residual")}
    overall = group_comparison(list(range(300)), methods, target_values)
    selection_by_id = {row["profile_id"]: row for row in selection}; segments = {}; segment_gate = True
    for segment in contract["blind_contract"]["segments"]:
        indices = [index for index, profile_id in enumerate(profile_ids) if selection_by_id[profile_id]["segment"] == segment]; value = group_comparison(indices, methods, target_values)
        ratios = value["metric_ratios_to_h0"]; gate = float(value["composite_ratio"]) < 1.0 and float(ratios["calls_histogram_wape"]) <= 1.05 and float(ratios["bytes_histogram_wape"]) <= 1.05
        segments[segment] = {**value, "gate": gate}; segment_gate &= gate
    strata = {}
    for stratum in range(10):
        indices = [index for index, profile_id in enumerate(profile_ids) if int(selection_by_id[profile_id]["request_count_stratum"]) == stratum]; strata[str(stratum)] = group_comparison(indices, methods, target_values)
    overall_gate = bool(overall["strict_four_metric_gate"]); confirmed = bool(overall_gate and segment_gate); outcome = "CONFIRMS_H0_PROTECTED_IMPROVEMENT" if confirmed else "DOES_NOT_CONFIRM"
    per_profile = per_profile_rows(profile_ids, methods, target_values); bootstrap = paired_bootstrap(per_profile)
    output.mkdir(parents=True); write_csv_gz(output / "labels/pd_fresh_blind_hfull_targets.csv.gz", targets)
    write_csv(output / "analysis/aggregate_metrics.csv", [{"method": method, **overall["h0" if method == "h0" else "h0_plus_dnn_residual"], "composite_ratio_to_h0": 1.0 if method == "h0" else overall["composite_ratio"], "scientific_outcome": "BASELINE" if method == "h0" else outcome} for method in ("h0", "h0_plus_dnn_residual")])
    write_csv_gz(output / "analysis/per_profile_metrics.csv.gz", per_profile); write_json(output / "analysis/segment_metrics.json", segments); write_json(output / "analysis/request_count_stratum_metrics.json", strata); write_json(output / "analysis/paired_bootstrap.json", bootstrap)
    write_json(output / "audit/input_freeze.json", preflight); write_json(output / "audit/target_generation.json", {"schema_version": "phase46-target-generation-audit-v1", "workflow_commit": expected, "prediction_parent_result_commit": "284f4b796b57bfee5002efb52937da26d0fe748f", "profiles": 300, "complete_requests_used_outside_git": total_requests, "complete_request_rows_committed": 0, "reconstruction_gate_passed_before_target_access": reconstruction_passed, "reconstruction": reconstruction})
    write_json(output / "audit/environment.json", {**environment_record(), "numpy": np.__version__, "gpu_used": False, "network_used": False, "training_used": False, "checkpoint_loaded": False, "prediction_recomputed": False, "raw_mutated": False})
    summary = {"schema_version": "phase46-pd-fresh-blind-evaluation-result-v1", "status": "PASS", "workflow_commit": expected, "completed_at_utc": utc_now(), "counts": {"blind_profiles": 300, "blind_complete_requests": total_requests, "target_rows": 300, "frozen_prediction_rows": 600, "per_profile_metric_rows": 600, "complete_request_rows_in_git": 0}, "gates": {"overall_strict_four_metrics": overall_gate, "all_segments": segment_gate, "confirmed": confirmed}, "scientific_outcome": outcome, "blind_metrics": overall, "segments": segments, "paired_bootstrap": bootstrap, "proved": "one-time target-isolated 300-profile Qwen3 pure-PD fresh blind evaluation after R45 prediction freeze", "not_proved": "Mooncake, other-model generalization, physical RDMA time, placement, latency or online scheduling"}
    write_json(output / "summary.json", summary)
    (output / "README.md").write_text(f"# Phase46：纯PD 300画像fresh blind评估\n\n状态：`PASS`，科学结论：`{outcome}`。R45正式冻结以后才为300个窗口、{total_requests}个完整请求生成Hfull；Git只保存300行直方图标签。\n\n四指标overall gate=`{overall_gate}`，三个segment gate=`{segment_gate}`，composite ratio=`{overall['composite_ratio']:.6f}`。无训练、checkpoint加载、预测重算或删样本。\n", encoding="utf-8")
    (output / "logs").mkdir(); (output / "logs/runtime.log").write_text(f"completed={utc_now()} workflow_commit={expected}\nprofiles=300 requests={total_requests} frozen_predictions=600 targets=300\noutcome={outcome} overall_gate={overall_gate} segment_gate={segment_gate} composite_ratio={overall['composite_ratio']:.12f}\ngpu=false training=false checkpoint_loaded=false prediction_recomputed=false raw_committed=false\n", encoding="utf-8")
    (output / "DONE").write_text("PASS\n", encoding="utf-8"); refresh_manifest(output); return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--expected-workflow-commit", required=True); parser.add_argument("--raw-dir", type=Path, required=True); parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase46_pd_fresh_blind_evaluation")
    args = parser.parse_args(); print(json.dumps(run(args.expected_workflow_commit, args.raw_dir, args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
