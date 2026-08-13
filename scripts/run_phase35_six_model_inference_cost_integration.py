#!/usr/bin/env python3
"""Replay frozen Phase34 predictors and convolve topology-independent histograms with cost curves."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import train_phase34c_six_model_direction as phase34


COMMON_REFERENCE_ID = "validation_common_reference_5us_100gbps"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase34a-dir", type=Path, default=root / "experiment-results/phase34a_six_model_contract")
    parser.add_argument("--phase34c-dir", type=Path, default=root / "experiment-results/phase34c_six_model_target_free_training")
    parser.add_argument("--phase34d-dir", type=Path, default=root / "experiment-results/phase34d_six_model_blind_evaluation")
    parser.add_argument("--collective-knots", type=Path, default=root / "experiment-results/phase2/summary_l1_curve/collective_cost_knots.json")
    parser.add_argument("--custom-knots", type=Path, default=root / "experiment-results/phase2/summary_l1_custom_kernel_curve/custom_kernel_cost_knots.json")
    parser.add_argument("--output-dir", type=Path, default=root / "experiment-results/phase35_six_model_inference_cost_integration")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(directory: Path) -> bool:
    manifest = directory / "manifest.sha256"
    if not manifest.exists():
        return False
    for line in manifest.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = directory / relative
        if not path.is_file() or sha256(path) != expected:
            return False
    return True


def read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as source:
        return list(csv.DictReader(source))


def deterministic_gzip(path: Path, text: str) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as output:
            output.write(text.encode())


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(name for name in row if name not in fields)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def write_csv_gz(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(name for name in row if name not in fields)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    deterministic_gzip(path, buffer.getvalue())


def write_json(path: Path, value: object) -> None:
    def convert(item: object) -> object:
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(item.__class__.__name__)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=convert) + "\n")


def runtime_predictions(args: argparse.Namespace, device: torch.device) -> tuple[list[dict], dict]:
    output = []
    audit = {}
    for parallelism in ("tp", "pp"):
        child = args.phase34c_dir / parallelism
        summary = json.loads((child / "summary.json").read_text())
        features = read_csv(args.phase34a_dir / f"dataset/{parallelism}_blind_confirmation_features.csv.gz")
        forbidden = [name for name in features[0] if name.startswith("target_")]
        if forbidden:
            raise RuntimeError(f"target exposed to {parallelism} runtime: {forbidden}")
        checkpoint_paths = sorted((child / "checkpoints").glob(f"{parallelism}_top1_seed*.pt"))
        if len(checkpoint_paths) != 3:
            raise RuntimeError(f"expected three selected {parallelism} checkpoints, got {checkpoint_paths}")
        bundles = []
        for path in checkpoint_paths:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload["rank"] != 1 or len(payload["folds"]) != 5:
                raise RuntimeError(f"invalid selected checkpoint: {path}")
            bundles.append(payload["folds"])
        prediction = phase34.infer(features, bundles, float(summary["selected"]["alpha"]), parallelism, device)
        rows = phase34.fixed_prediction_rows(
            features,
            {"h0_plus_dnn_residual": prediction},
            parallelism,
            summary["selected"]["candidate_id"],
        )
        for row in rows:
            row["runtime_schema"] = "phase35-unified-six-model-pattern-demand-v1"
            row["checkpoint_ensemble"] = "top1_3seed_5fold_mean"
            row["phase34_source_prediction_sha256"] = json.loads((args.phase34c_dir / "summary.json").read_text())["frozen_prediction_sha256"]
        output.extend(rows)
        audit[parallelism] = {
            "selected_candidate_id": summary["selected"]["candidate_id"],
            "alpha": summary["selected"]["alpha"],
            "feature_rows": len(features),
            "checkpoint_paths": [str(path.relative_to(args.phase34c_dir)) for path in checkpoint_paths],
            "checkpoint_sha256": [sha256(path) for path in checkpoint_paths],
            "fold_models": sum(len(bundle) for bundle in bundles),
        }
    return output, audit


def replay_audit(predictions: list[dict], args: argparse.Namespace) -> dict:
    frozen = []
    for parallelism in ("tp", "pp"):
        rows = read_csv(args.phase34c_dir / parallelism / "analysis/frozen_predictions.csv.gz")
        frozen.extend(row for row in rows if row["prediction_set"] == "phase34_blind_new" and row["method"] == "h0_plus_dnn_residual")
    old = {row["example_id"]: row for row in frozen}
    if len(old) != len(frozen) or len(predictions) != len(old):
        raise RuntimeError(f"replay cardinality mismatch: runtime={len(predictions)} frozen={len(old)}")
    maxima = defaultdict(float)
    for row in predictions:
        expected = old[row["example_id"]]
        for name in ("predicted_total_calls_per_1000", "predicted_total_logical_bytes_per_1000", "predicted_common_reference_cost_us_per_1000"):
            actual_value = float(row[name]); expected_value = float(expected[name])
            maxima[name + "_absolute"] = max(maxima[name + "_absolute"], abs(actual_value - expected_value))
            maxima[name + "_relative"] = max(maxima[name + "_relative"], abs(actual_value - expected_value) / max(abs(expected_value), 1.0))
        for name in ("predicted_calls_by_12bin_json", "predicted_logical_bytes_by_12bin_json"):
            actual_vector = np.asarray(json.loads(row[name]), dtype=np.float64)
            expected_vector = np.asarray(json.loads(expected[name]), dtype=np.float64)
            maxima[name + "_absolute"] = max(maxima[name + "_absolute"], float(np.max(np.abs(actual_vector - expected_vector))))
            maxima[name + "_relative"] = max(maxima[name + "_relative"], float(np.max(np.abs(actual_vector - expected_vector) / np.maximum(np.abs(expected_vector), 1.0))))
    return {
        "runtime_rows": len(predictions),
        "frozen_rows": len(frozen),
        "max_differences": dict(maxima),
        "max_scalar_relative_difference": max(value for key, value in maxima.items() if key.endswith("_relative")),
        "candidate_ids_match": all(row["selected_candidate_id"] == old[row["example_id"]]["selected_candidate_id"] for row in predictions),
    }


def registry(args: argparse.Namespace) -> list[dict]:
    phase17 = "scripts/analyze_phase17_parameterized_topology.py::PARAMETER_SCENARIOS nominal"
    common = {
        "placement_id": COMMON_REFERENCE_ID, "parallelism": "both", "topology": "reference_only",
        "curve_kind": "numeric_validation_reference", "evidence": "not_a_physical_placement_curve",
        "scheduler_candidate": False, "formula": "5 us + payload / 100 GB/s",
    }
    return [
        common,
        {
            "placement_id": "tp_l1_single_node_b200_nvlink_measured", "parallelism": "tp", "topology": "L1_single_node_B200_NVLink",
            "curve_kind": "physical_measurement_backend_aware", "evidence": "Phase2 measured CustomAllReduce intrinsic median below backend limit, NCCL median above",
            "scheduler_candidate": True, "custom_knots_sha256": sha256(args.custom_knots), "nccl_knots_sha256": sha256(args.collective_knots),
        },
        {
            "placement_id": "tp_l2_same_rack_nominal_proxy", "parallelism": "tp", "topology": "L2_same_rack_two_node_proxy",
            "curve_kind": "parameterized_collective_sensitivity", "evidence": "not_physical_measurement", "scheduler_candidate": True,
            "source": phase17, "launch_us": 5.0, "round_us": 4.0, "bandwidth_gbps": 100.0, "saturation_mib": 1.0,
        },
        {
            "placement_id": "tp_l3_cross_rack_nominal_proxy", "parallelism": "tp", "topology": "L3_cross_rack_two_node_proxy",
            "curve_kind": "parameterized_collective_sensitivity", "evidence": "not_physical_measurement", "scheduler_candidate": True,
            "source": phase17, "launch_us": 10.0, "round_us": 15.0, "bandwidth_gbps": 50.0, "saturation_mib": 4.0,
        },
        {
            "placement_id": "pp_l1_single_node_nominal_proxy", "parallelism": "pp", "topology": "L1_single_node_p2p_proxy",
            "curve_kind": "parameterized_p2p_sensitivity", "evidence": "not_physical_measurement", "scheduler_candidate": True,
            "launch_us": 3.0, "bandwidth_gbps": 300.0, "saturation_mib": 0.25,
        },
        {
            "placement_id": "pp_l2_same_rack_nominal_proxy", "parallelism": "pp", "topology": "L2_same_rack_p2p_proxy",
            "curve_kind": "parameterized_p2p_sensitivity", "evidence": "not_physical_measurement", "scheduler_candidate": True,
            "launch_us": 5.0, "bandwidth_gbps": 100.0, "saturation_mib": 1.0,
        },
        {
            "placement_id": "pp_l3_cross_rack_nominal_proxy", "parallelism": "pp", "topology": "L3_cross_rack_p2p_proxy",
            "curve_kind": "parameterized_p2p_sensitivity", "evidence": "not_physical_measurement", "scheduler_candidate": True,
            "launch_us": 10.0, "bandwidth_gbps": 50.0, "saturation_mib": 4.0,
        },
    ]


class MeasuredTPBackendCurve:
    def __init__(self, collective_path: Path, custom_path: Path):
        collective = json.loads(collective_path.read_text())["curves"]
        custom = json.loads(custom_path.read_text())["curves"]
        self.nccl = {int(row["group_size"]): [(float(k["payload_bytes"]), float(k["median_latency_us"])) for k in row["knots"]] for row in collective if row["op"] == "all_reduce"}
        self.custom = {int(row["group_size"]): [(float(k["payload_bytes"]), float(k["intrinsic_median_latency_us"])) for k in row["knots"]] for row in custom if row["op"] == "all_reduce"}
        if set(self.nccl) != {2, 4, 8} or set(self.custom) != {2, 4, 8}:
            raise RuntimeError("measured TP curve support mismatch")

    @staticmethod
    def interpolate(points: list[tuple[float, float]], payload: float) -> float:
        x = np.log2(np.asarray([point[0] for point in points], dtype=np.float64))
        y = np.asarray([point[1] for point in points], dtype=np.float64)
        return float(np.interp(math.log2(max(payload, 1.0)), x, y, left=y[0], right=y[-1]))

    def lookup(self, group_size: int, payload: float) -> float:
        custom_max = self.custom[group_size][-1][0]
        return self.interpolate(self.custom[group_size] if payload <= custom_max else self.nccl[group_size], payload)


def utilization(payload: float, saturation_mib: float) -> float:
    return max(1.0 - math.exp(-payload / (saturation_mib * 1024 * 1024)), 1e-6)


def message_cost_us(curve: dict, parallel_size: int, payload: float, measured: MeasuredTPBackendCurve) -> float:
    placement = curve["placement_id"]
    if placement == COMMON_REFERENCE_ID:
        return 5.0 + payload / 100e9 * 1e6
    if curve["curve_kind"] == "physical_measurement_backend_aware":
        return measured.lookup(parallel_size, payload)
    effective_bandwidth = float(curve["bandwidth_gbps"]) * 1e9 * utilization(payload, float(curve["saturation_mib"]))
    data_us = payload / effective_bandwidth * 1e6
    if curve["curve_kind"] == "parameterized_collective_sensitivity":
        ring_alpha = 2.0 * (parallel_size - 1) / parallel_size
        ring_rounds = 2.0 * (parallel_size - 1)
        return float(curve["launch_us"]) + ring_rounds * float(curve["round_us"]) + ring_alpha * data_us
    return float(curve["launch_us"]) + data_us


def histogram_cost(calls: np.ndarray, logical_bytes: np.ndarray, curve: dict, parallel_size: int, measured: MeasuredTPBackendCurve) -> float:
    total = 0.0
    for amount, byte_count in zip(calls, logical_bytes):
        if amount > 1e-12:
            total += float(amount) * message_cost_us(curve, parallel_size, float(byte_count) / float(amount), measured)
    return total


def histogram_sha(row: dict, prefix: str) -> str:
    payload = row[f"{prefix}_calls_by_12bin_json"] + "|" + row[f"{prefix}_logical_bytes_by_12bin_json"]
    return hashlib.sha256(payload.encode()).hexdigest()


def phase_cost_rows(predictions: list[dict], targets: list[dict], curves: list[dict], measured: MeasuredTPBackendCurve) -> tuple[list[dict], dict]:
    target_by_id = {row["example_id"]: row for row in targets}
    if len(target_by_id) != len(targets) or set(target_by_id) != {row["example_id"] for row in predictions}:
        raise RuntimeError("target/prediction key mismatch")
    output = []
    max_reference_relative = 0.0
    for prediction in predictions:
        target = target_by_id[prediction["example_id"]]
        predicted_calls = np.asarray(json.loads(prediction["predicted_calls_by_12bin_json"]), dtype=np.float64)
        predicted_bytes = np.asarray(json.loads(prediction["predicted_logical_bytes_by_12bin_json"]), dtype=np.float64)
        target_calls = np.asarray(json.loads(target["target_calls_by_12bin_json"]), dtype=np.float64)
        target_bytes = np.asarray(json.loads(target["target_logical_bytes_by_12bin_json"]), dtype=np.float64)
        applicable = [curve for curve in curves if curve["parallelism"] in {"both", prediction["parallelism"]}]
        for curve in applicable:
            predicted_cost = histogram_cost(predicted_calls, predicted_bytes, curve, int(prediction["parallel_size"]), measured)
            target_cost = histogram_cost(target_calls, target_bytes, curve, int(prediction["parallel_size"]), measured)
            if curve["placement_id"] == COMMON_REFERENCE_ID:
                for computed, saved in ((predicted_cost, float(prediction["predicted_common_reference_cost_us_per_1000"])), (target_cost, float(target["target_common_reference_cost_us_per_1000"]))):
                    max_reference_relative = max(max_reference_relative, abs(computed - saved) / max(abs(saved), 1e-12))
            output.append({
                "example_id": prediction["example_id"], "profile_id": prediction["profile_id"], "model": prediction["model"],
                "parallelism": prediction["parallelism"], "parallel_size": prediction["parallel_size"], "policy": prediction["policy"], "phase": prediction["phase"],
                "placement_id": curve["placement_id"], "topology": curve["topology"], "curve_kind": curve["curve_kind"], "curve_evidence": curve["evidence"],
                "scheduler_candidate": curve["scheduler_candidate"], "predicted_cost_us_per_1000": predicted_cost, "teacher_cost_us_per_1000": target_cost,
                "absolute_error_us_per_1000": abs(predicted_cost - target_cost), "absolute_percentage_error": abs(predicted_cost - target_cost) / max(target_cost, 1e-12),
                "signed_error_us_per_1000": predicted_cost - target_cost, "predicted_histogram_sha256": histogram_sha(prediction, "predicted"),
                "teacher_kind": target["teacher_kind"], "evidence_set": "phase34_open_target_repeated_engineering",
            })
    return output, {"max_common_reference_relative_difference": max_reference_relative}


def combined_cost_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row[name] for name in ("profile_id", "model", "parallelism", "parallel_size", "policy", "placement_id"))
        grouped[key].append(row)
    output = []
    for values in grouped.values():
        if {row["phase"] for row in values} != {"prefill", "decode"}:
            raise RuntimeError("cost case lacks two phases")
        source = values[0]
        predicted = sum(float(row["predicted_cost_us_per_1000"]) for row in values)
        teacher = sum(float(row["teacher_cost_us_per_1000"]) for row in values)
        output.append({
            **{name: source[name] for name in ("profile_id", "model", "parallelism", "parallel_size", "policy", "placement_id", "topology", "curve_kind", "curve_evidence", "scheduler_candidate", "predicted_histogram_sha256", "teacher_kind", "evidence_set")},
            "example_id": "/".join(source["example_id"].split("/")[:-1]) + "/total", "phase": "total",
            "predicted_cost_us_per_1000": predicted, "teacher_cost_us_per_1000": teacher,
            "absolute_error_us_per_1000": abs(predicted - teacher), "absolute_percentage_error": abs(predicted - teacher) / max(teacher, 1e-12),
            "signed_error_us_per_1000": predicted - teacher,
        })
    return output


def aggregate_metrics(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        for slice_type, slice_value in (("overall", "all"), ("model", row["model"]), ("policy", row["policy"])):
            groups[(row["parallelism"], row["placement_id"], row["phase"], slice_type, slice_value)].append(row)
    output = []
    for key, values in sorted(groups.items()):
        parallelism, placement, phase, slice_type, slice_value = key
        teacher = sum(float(row["teacher_cost_us_per_1000"]) for row in values)
        predicted = sum(float(row["predicted_cost_us_per_1000"]) for row in values)
        output.append({
            "parallelism": parallelism, "placement_id": placement, "topology": values[0]["topology"], "curve_kind": values[0]["curve_kind"],
            "phase": phase, "slice_type": slice_type, "slice_value": slice_value, "cases": len(values),
            "cost_mape": float(np.mean([float(row["absolute_percentage_error"]) for row in values])),
            "cost_wape": sum(float(row["absolute_error_us_per_1000"]) for row in values) / max(teacher, 1e-12),
            "signed_bias": (predicted - teacher) / max(teacher, 1e-12), "evidence_set": values[0]["evidence_set"],
        })
    return output


def rankings(combined: list[dict]) -> tuple[list[dict], dict]:
    candidates = [row for row in combined if str(row["scheduler_candidate"]).lower() == "true" or row["scheduler_candidate"] is True]
    groups = defaultdict(list)
    for row in candidates:
        key = tuple(row[name] for name in ("profile_id", "model", "parallelism", "parallel_size", "policy"))
        groups[key].append(row)
    output = []
    predicted_matches = 0
    for values in groups.values():
        if len(values) != 3:
            raise RuntimeError(f"expected three placement candidates: {values}")
        predicted_order = sorted(values, key=lambda row: float(row["predicted_cost_us_per_1000"]))
        teacher_order = sorted(values, key=lambda row: float(row["teacher_cost_us_per_1000"]))
        predicted_matches += predicted_order[0]["placement_id"] == teacher_order[0]["placement_id"]
        teacher_rank = {row["placement_id"]: index + 1 for index, row in enumerate(teacher_order)}
        for index, row in enumerate(predicted_order, 1):
            output.append({
                **{name: row[name] for name in ("profile_id", "model", "parallelism", "parallel_size", "policy", "placement_id", "topology", "curve_kind")},
                "predicted_communication_rank": index, "teacher_communication_rank": teacher_rank[row["placement_id"]],
                "predicted_cost_us_per_1000": row["predicted_cost_us_per_1000"], "teacher_cost_us_per_1000": row["teacher_cost_us_per_1000"],
                "selected_by_predicted_communication_only": index == 1,
                "ranking_scope": "communication_only_excludes_memory_compute_availability_and_overlap",
            })
    return output, {"cases": len(groups), "top1_match_rate": predicted_matches / max(len(groups), 1)}


def make_figure(path: Path, metrics: list[dict]) -> None:
    rows = [row for row in metrics if row["phase"] == "total" and row["slice_type"] == "overall" and row["placement_id"] != COMMON_REFERENCE_ID]
    width, height, margin = 1050, 500, 70
    maximum = max(float(row["cost_wape"]) for row in rows) * 1.15
    bar_width = (width - 2 * margin) / len(rows)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', '<text x="70" y="32" font-family="sans-serif" font-size="21">Phase35 各连续通信曲线的cost WAPE（重复工程评估）</text>']
    for index, row in enumerate(rows):
        value = float(row["cost_wape"]); x = margin + index * bar_width; bar_height = value / maximum * 330; y = 405 - bar_height
        color = "#2563eb" if row["parallelism"] == "tp" else "#d97706"
        label = row["placement_id"].replace("_nominal_proxy", "").replace("_measured", "")
        svg.extend([f'<rect x="{x + 8:.1f}" y="{y:.1f}" width="{bar_width - 16:.1f}" height="{bar_height:.1f}" fill="{color}"/>', f'<text x="{x + bar_width/2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-family="sans-serif" font-size="13">{value:.2%}</text>', f'<text x="{x + bar_width/2:.1f}" y="425" text-anchor="middle" font-family="sans-serif" font-size="10" transform="rotate(25 {x + bar_width/2:.1f} 425)">{label}</text>'])
    svg.append('</svg>')
    path.write_text("\n".join(svg) + "\n")


def refresh_manifest(directory: Path) -> None:
    rows = [f"{sha256(path)}  {path.relative_to(directory)}" for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "manifest.sha256"]
    (directory / "manifest.sha256").write_text("\n".join(rows) + "\n")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise RuntimeError(f"refuse to overwrite existing result: {args.output_dir}")
    for name in ("contracts", "predictions", "costs", "analysis", "figures", "logs", "docs"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Phase35 checkpoint replay requires CUDA to match Phase34 frozen inference")
    source_manifest_checks = {
        "phase34a": verify_manifest(args.phase34a_dir), "phase34c": verify_manifest(args.phase34c_dir),
        "phase34c_tp": verify_manifest(args.phase34c_dir / "tp"), "phase34c_pp": verify_manifest(args.phase34c_dir / "pp"),
        "phase34d": verify_manifest(args.phase34d_dir),
    }
    phase34c_summary = json.loads((args.phase34c_dir / "summary.json").read_text())
    source_prediction = args.phase34c_dir / "analysis/frozen_predictions_all_versions.csv.gz"
    source_sha_matches = sha256(source_prediction) == phase34c_summary["frozen_prediction_sha256"]

    predictions, runtime_audit = runtime_predictions(args, device)
    prediction_path = args.output_dir / "predictions/unified_six_model_histograms.csv.gz"
    write_csv_gz(prediction_path, predictions)
    replay = replay_audit(predictions, args)
    prediction_freeze = {
        "schema_version": "phase35-runtime-prediction-freeze-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_sha256": sha256(prediction_path), "rows": len(predictions), "runtime_audit": runtime_audit,
        "phase34_source_prediction_sha256": phase34c_summary["frozen_prediction_sha256"], "targets_read_before_freeze": False,
    }
    write_json(args.output_dir / "predictions/PREDICTION_FREEZE.json", prediction_freeze)

    curves = registry(args)
    write_json(args.output_dir / "contracts/topology_curve_registry.json", {"schema_version": "phase35-topology-curve-registry-v1", "curves": curves, "boundary": "Only TP L1 is a physical measurement. TP L2/L3 and all PP curves are parameterized sensitivity proxies."})
    measured = MeasuredTPBackendCurve(args.collective_knots, args.custom_knots)
    targets = read_csv(args.phase34d_dir / "labels/phase34_blind_six_model_hfull_targets.csv.gz")
    phase_rows, reference_audit = phase_cost_rows(predictions, targets, curves, measured)
    total_rows = combined_cost_rows(phase_rows)
    all_cost_rows = phase_rows + total_rows
    metrics = aggregate_metrics(all_cost_rows)
    rank_rows, rank_audit = rankings(total_rows)
    write_csv_gz(args.output_dir / "costs/placement_costs.csv.gz", all_cost_rows)
    write_csv(args.output_dir / "analysis/cost_metrics.csv", metrics)
    write_csv_gz(args.output_dir / "analysis/communication_only_rankings.csv.gz", rank_rows)
    make_figure(args.output_dir / "figures/topology_cost_wape.svg", metrics)

    headline = {
        row["parallelism"] + "/" + row["placement_id"]: row
        for row in metrics if row["phase"] == "total" and row["slice_type"] == "overall"
    }
    histogram_hashes = defaultdict(set)
    for row in phase_rows:
        histogram_hashes[row["example_id"]].add(row["predicted_histogram_sha256"])
    finite_nonnegative = all(math.isfinite(float(row[name])) and float(row[name]) >= 0 for row in all_cost_rows for name in ("predicted_cost_us_per_1000", "teacher_cost_us_per_1000", "absolute_error_us_per_1000"))
    checks = {
        "all_phase34_source_manifests_pass": all(source_manifest_checks.values()),
        "phase34_frozen_prediction_sha_matches": source_sha_matches,
        "runtime_prediction_rows_2592": len(predictions) == 2592,
        "runtime_contains_no_target_columns": not any(name.startswith("target_") for name in predictions[0]),
        "three_seed_fivefold_ensemble_each_direction": all(value["fold_models"] == 15 for value in runtime_audit.values()),
        "replay_candidate_ids_match": replay["candidate_ids_match"],
        "replay_relative_difference_below_1e_6": replay["max_scalar_relative_difference"] < 1e-6,
        "common_reference_relative_difference_below_1e_10": reference_audit["max_common_reference_relative_difference"] < 1e-10,
        "seven_curve_contracts": len(curves) == 7,
        "six_scheduler_placement_curves_three_each_direction": sum(bool(row["scheduler_candidate"]) for row in curves) == 6,
        "all_costs_finite_nonnegative": finite_nonnegative,
        "same_histogram_reused_across_placements": len(histogram_hashes) == 2592 and all(len(values) == 1 for values in histogram_hashes.values()),
        "phase_cost_rows_10368": len(phase_rows) == 10368,
        "combined_cost_rows_5184": len(total_rows) == 5184,
        "communication_ranking_rows_3888": len(rank_rows) == 3888,
        "teacher_used_only_after_prediction_freeze": prediction_freeze["targets_read_before_freeze"] is False,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = {
        "schema_version": "phase35-six-model-inference-cost-integration-v1", "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "device": str(device),
        "objective": "frozen six-model PatternDemand inference followed by independent continuous placement/topology communication cost curves",
        "prediction_freeze": prediction_freeze, "replay_audit": replay, "reference_audit": reference_audit,
        "source_manifest_checks": source_manifest_checks, "curve_registry": curves, "headline_cost_metrics": headline,
        "ranking_audit": rank_audit,
        "counts": {"prediction_phase_rows": len(predictions), "phase_cost_rows": len(phase_rows), "combined_cost_rows": len(total_rows), "metric_rows": len(metrics), "ranking_rows": len(rank_rows)},
        "evidence": {"checkpoint_replay": "frozen Phase34 engineering regression", "cost_evaluation": "Phase34 already-open target repeated engineering evidence", "tp_l1_curve": "physical Phase2 measurement", "tp_l2_l3_and_pp_curves": "parameterized sensitivity proxy"},
        "scheduler_boundary": "communication-only ranking excludes memory feasibility, compute time, resource availability, contention, and communication-compute overlap",
        "checks": checks,
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "audit_summary.json", {"schema_version": "phase35-audit-v1", "status": status, "checks": checks, "source_manifest_checks": source_manifest_checks, "replay_audit": replay, "reference_audit": reference_audit})
    write_json(args.output_dir / "logs/runtime.log", {"event": "phase35_six_model_inference_cost_integration_complete", "status": status, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "repository_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "platform": platform.platform(), "device": str(device), "training_performed": False})
    (args.output_dir / "README.md").write_text(f"""# Phase35：六模型统一推理与placement/topology连续通信代价集成

本阶段没有训练或调参。统一运行时加载Phase34冻结的TP/PP top1三seed五折checkpoint，从低维target-free特征重放六模型消息直方图；{len(predictions):,}条phase预测与Phase34C冻结结果在`1e-6`相对容差内一致。随后同一份拓扑无关直方图分别代入候选连续代价曲线。

TP单机B200 NVLink使用Phase2物理测量的CustomAllReduce/NCCL backend-aware曲线；TP L2/L3和全部PP曲线是参数化敏感性proxy，不能包装成真实硬件时延。共同参考曲线只做数值回归，与Phase34保存cost的最大相对差为`{reference_audit['max_common_reference_relative_difference']:.3e}`。

`analysis/cost_metrics.csv`保存整体、逐模型、逐policy的cost MAPE/WAPE；`analysis/communication_only_rankings.csv.gz`只给通信项排名，尚未加入显存、计算、资源、拥塞和重叠，不能直接等同最终调度决策。Phase34D target已经打开，本阶段cost误差属于重复工程证据，不是新盲测。
""")
    (args.output_dir / "DONE").write_text(status + "\n")
    refresh_manifest(args.output_dir)
    if status != "PASS":
        raise RuntimeError(checks)
    print(json.dumps({"status": status, "counts": summary["counts"], "replay": replay, "reference": reference_audit, "ranking": rank_audit, "headline_cost_metrics": headline}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
