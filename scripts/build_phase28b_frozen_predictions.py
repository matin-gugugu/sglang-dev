#!/usr/bin/env python3
"""Freeze Phase 28 H0/enhanced predictions without generating or reading Hfull targets."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from build_phase21b_pp_h0 import pseudo_requests
from build_phase27b_pp_hfull_dataset import (
    HISTORY_SECONDS,
    MICROBATCH_SIZES,
    PHASES,
    PP_SIZES,
    exact_histogram,
    model_features,
    summarize_profile,
    training_features,
)
from prepare_phase15_trace_windows import BURST_FILES, MOONCAKE_FILES, load_segment


METHODS = ("h0", "enhanced_bounded_residual")
BIN_COUNT = 12
COMMON_REFERENCE_LAUNCH_US = 5.0
COMMON_REFERENCE_BANDWIDTH_GBPS = 100.0


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument(
        "--selection",
        type=Path,
        default=root
        / "experiment-results/phase28a_second_confirmation_contract/selection/selected_windows.csv",
    )
    parser.add_argument(
        "--phase28a-summary",
        type=Path,
        default=root
        / "experiment-results/phase28a_second_confirmation_contract/summary.json",
    )
    parser.add_argument(
        "--frozen-mapping",
        type=Path,
        default=root
        / "experiment-results/phase28a_second_confirmation_contract/frozen_method_mapping.json",
    )
    parser.add_argument(
        "--numpy-checkpoint",
        type=Path,
        default=root
        / "experiment-results/phase28b_frozen_predictions/checkpoint/pp_enhanced_bounded_residual_numpy.npz",
    )
    parser.add_argument(
        "--checkpoint-export-audit",
        type=Path,
        default=root
        / "experiment-results/phase28b_frozen_predictions/checkpoint/export_audit.json",
    )
    parser.add_argument(
        "--phase27c-audit",
        type=Path,
        default=root
        / "experiment-results/phase27c_pp_scheduler_feature_training/audit_summary.json",
    )
    parser.add_argument(
        "--model-features",
        type=Path,
        default=root / "experiment-results/phase16_model_features/model_features.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase28b_frozen_predictions",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_gzip(path: Path, text: str) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as output:
            output.write(text.encode())


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_csv_gz(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    deterministic_gzip(path, buffer.getvalue())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def target_encode(calls: np.ndarray, logical_bytes: np.ndarray) -> np.ndarray:
    encoded: list[float] = []
    for vector in (calls, logical_bytes):
        total = max(float(np.sum(vector)), 0.0)
        smoothing = max(total, 1.0) * 1e-6 / BIN_COUNT
        shares = (vector + smoothing) / (total + smoothing * BIN_COUNT)
        encoded.extend([math.log1p(total), *np.log(shares)])
    return np.asarray(encoded, dtype=np.float32)


def target_decode(encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vectors = []
    offset = 0
    for _ in range(2):
        total = max(math.expm1(float(np.clip(encoded[offset], 0, 40))), 0.0)
        logits = np.clip(encoded[offset + 1 : offset + BIN_COUNT + 1], -50, 50)
        probabilities = np.exp(logits - np.max(logits))
        probabilities /= probabilities.sum()
        vectors.append(total * probabilities)
        offset += BIN_COUNT + 1
    return vectors[0].astype(np.float64), vectors[1].astype(np.float64)


def common_reference_cost(calls: np.ndarray, logical_bytes: np.ndarray) -> float:
    return float(
        COMMON_REFERENCE_LAUNCH_US * calls.sum()
        + logical_bytes.sum() / (COMMON_REFERENCE_BANDWIDTH_GBPS * 1e9) * 1e6
    )


def predict_numpy(
    rows: list[dict[str, object]], checkpoint: np.lib.npyio.NpzFile, h0_encoded: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    names = checkpoint["feature_names"].tolist()
    log_names = set(checkpoint["log_feature_names"].tolist())
    features = np.asarray(
        [
            [
                math.log1p(max(float(row[name]), 0.0))
                if name in log_names
                else float(row[name])
                for name in names
            ]
            for row in rows
        ],
        dtype=np.float32,
    )
    scaled = np.clip(
        (features - checkpoint["feature_mean"]) / checkpoint["feature_std"],
        -6.0,
        6.0,
    ).astype(np.float32)
    hidden1 = np.maximum(
        scaled @ checkpoint["network.0.weight"].T + checkpoint["network.0.bias"],
        0.0,
    )
    hidden2 = np.maximum(
        hidden1 @ checkpoint["network.2.weight"].T + checkpoint["network.2.bias"],
        0.0,
    )
    raw = np.tanh(
        hidden2 @ checkpoint["network.4.weight"].T + checkpoint["network.4.bias"]
    )
    encoded = h0_encoded + raw * checkpoint["target_std_or_residual_bounds"]
    calls, logical_bytes = zip(*(target_decode(row) for row in encoded))
    return np.stack(calls), np.stack(logical_bytes)


def main() -> None:
    args = parse_args()
    for name in ("profiles", "dataset", "analysis", "logs"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    contract = json.loads(args.phase28a_summary.read_text())
    mapping = json.loads(args.frozen_mapping.read_text())["mapping"]
    if contract["label_state_at_freeze"] != "no_phase28_hfull_labels_generated":
        raise RuntimeError("Phase 28A label contract is not clean")
    if mapping != {
        "mb1": "h0",
        "mb4": "enhanced_bounded_residual",
        "mb16": "enhanced_bounded_residual",
    }:
        raise RuntimeError(mapping)
    audit27c = json.loads(args.phase27c_audit.read_text())
    expected_checkpoint_hash = audit27c["checkpoint_sha256"][
        "enhanced_bounded_residual"
    ]
    export_audit = json.loads(args.checkpoint_export_audit.read_text())
    if export_audit["status"] != "PASS":
        raise RuntimeError("NumPy checkpoint export audit failed")
    if export_audit["source_checkpoint_sha256"] != expected_checkpoint_hash:
        raise RuntimeError("source checkpoint hash mismatch")
    if sha256(args.numpy_checkpoint) != export_audit["numpy_checkpoint_sha256"]:
        raise RuntimeError("NumPy checkpoint hash mismatch")

    raw_manifest_path = args.raw_dir / "source_manifest.json"
    raw_manifest = json.loads(raw_manifest_path.read_text())
    raw_checks = {}
    for row in raw_manifest["sources"]:
        path = args.raw_dir / row["name"]
        raw_checks[row["name"]] = (
            path.stat().st_size == int(row["actual_size"])
            and sha256(path) == row["sha256"]
        )
    if not all(raw_checks.values()):
        raise RuntimeError(raw_checks)

    selected = read_csv(args.selection)
    if len(selected) != 18:
        raise ValueError(f"expected 18 windows, got {len(selected)}")
    file_by_segment = {
        segment: args.raw_dir / name
        for name, (segment, _) in {**BURST_FILES, **MOONCAKE_FILES}.items()
    }
    arrays = {segment: load_segment(path) for segment, path in file_by_segment.items()}
    profiles = []
    full_count_checks = []
    for row in selected:
        timestamps, inputs, outputs = arrays[row["segment"]]
        cutoff = int(row["cutoff_ms"])
        left = int(np.searchsorted(timestamps, cutoff - HISTORY_SECONDS * 1000, side="left"))
        right = int(np.searchsorted(timestamps, cutoff, side="left"))
        compatibility = {
            **row,
            "phase27_profile_id": row["profile_id"],
            "phase27_role": row["role"],
        }
        profile, requests = summarize_profile(
            compatibility,
            timestamps[left:right],
            inputs[left:right],
            outputs[left:right],
        )
        full_count_checks.append(len(requests) == int(row["history_count"]))
        profiles.append(profile)

    _, model_feature_values = model_features(args.model_features)
    feature_rows = []
    h0_simulation_checks = []
    for profile in profiles:
        compact = pseudo_requests(profile)
        for pp_size in PP_SIZES:
            for microbatch in MICROBATCH_SIZES:
                h0_histograms, audit = exact_histogram(compact, pp_size, microbatch)
                h0_simulation_checks.append(
                    audit["all_requests_complete"]
                    and audit["prefill_token_mass"] == sum(row[0] for row in compact)
                    and audit["decode_token_mass"] == sum(row[1] - 1 for row in compact)
                )
                for phase in PHASES:
                    calls = np.zeros(12, dtype=np.float64)
                    logical_bytes = np.zeros(12, dtype=np.float64)
                    edges = np.geomspace(4 * 1024, 8 * 1024 * 1024 * 1024, 13)
                    for payload, count in h0_histograms[phase].items():
                        index = int(
                            np.clip(
                                np.searchsorted(edges, payload, side="right") - 1,
                                0,
                                11,
                            )
                        )
                        calls[index] += count
                        logical_bytes[index] += payload * count
                    feature_rows.append(
                        {
                            "training_id": f"qwen3-8b/pp{pp_size}/mb{microbatch}/{profile['profile_id']}/hfull/{phase}",
                            "profile_id": profile["profile_id"],
                            "role": profile["phase27_role"],
                            "source": profile["source"],
                            "segment": profile["segment"],
                            "window_id": profile["window_id"],
                            "model": "qwen3-8b",
                            "parallelism": "pp",
                            "parallel_size": pp_size,
                            "policy": f"mb{microbatch}",
                            "phase": phase,
                            **training_features(
                                profile,
                                model_feature_values,
                                pp_size,
                                microbatch,
                                phase,
                            ),
                            "h0_total_calls_per_1000": float(calls.sum()),
                            "h0_total_logical_bytes_per_1000": float(
                                logical_bytes.sum()
                            ),
                            "h0_common_reference_cost_us_per_1000": common_reference_cost(
                                calls, logical_bytes
                            ),
                            "h0_calls_by_12bin_json": json.dumps(
                                calls.tolist(), separators=(",", ":")
                            ),
                            "h0_logical_bytes_by_12bin_json": json.dumps(
                                logical_bytes.tolist(), separators=(",", ":")
                            ),
                        }
                    )

    checkpoint = np.load(args.numpy_checkpoint, allow_pickle=False)
    h0_calls = np.stack(
        [np.asarray(json.loads(row["h0_calls_by_12bin_json"])) for row in feature_rows]
    )
    h0_bytes = np.stack(
        [
            np.asarray(json.loads(row["h0_logical_bytes_by_12bin_json"]))
            for row in feature_rows
        ]
    )
    h0_encoded = np.stack(
        [target_encode(calls, byte_values) for calls, byte_values in zip(h0_calls, h0_bytes)]
    )
    enhanced_calls, enhanced_bytes = predict_numpy(feature_rows, checkpoint, h0_encoded)
    method_arrays = {
        "h0": (h0_calls, h0_bytes),
        "enhanced_bounded_residual": (enhanced_calls, enhanced_bytes),
    }
    predictions = []
    for method in METHODS:
        calls, logical_bytes = method_arrays[method]
        for index, row in enumerate(feature_rows):
            predictions.append(
                {
                    "training_id": row["training_id"],
                    "profile_id": row["profile_id"],
                    "segment": row["segment"],
                    "window_id": row["window_id"],
                    "parallel_size": row["parallel_size"],
                    "policy": row["policy"],
                    "phase": row["phase"],
                    "method": method,
                    "selected_by_frozen_mapping": mapping[row["policy"]] == method,
                    "predicted_total_calls_per_1000": float(calls[index].sum()),
                    "predicted_total_logical_bytes_per_1000": float(
                        logical_bytes[index].sum()
                    ),
                    "predicted_common_reference_cost_us_per_1000": common_reference_cost(
                        calls[index], logical_bytes[index]
                    ),
                    "predicted_calls_by_12bin_json": json.dumps(
                        calls[index].tolist(), separators=(",", ":")
                    ),
                    "predicted_logical_bytes_by_12bin_json": json.dumps(
                        logical_bytes[index].tolist(), separators=(",", ":")
                    ),
                }
            )

    profile_rows = []
    for profile in profiles:
        profile_rows.append(profile)
    write_csv_gz(args.output_dir / "profiles/low_dimensional_profiles.csv.gz", profile_rows)
    write_csv_gz(args.output_dir / "dataset/confirmation_features.csv.gz", feature_rows)
    write_csv_gz(args.output_dir / "analysis/frozen_predictions.csv.gz", predictions)
    write_csv(
        args.output_dir / "analysis/prediction_inventory.csv",
        [
            {
                "method": method,
                "prediction_rows": sum(row["method"] == method for row in predictions),
                "selected_rows": sum(
                    row["method"] == method
                    and bool(row["selected_by_frozen_mapping"])
                    for row in predictions
                ),
            }
            for method in METHODS
        ],
    )

    prediction_path = args.output_dir / "analysis/frozen_predictions.csv.gz"
    summary = {
        "schema_version": "phase28b-frozen-predictions-v1",
        "status": "PASS",
        "profiles": len(profiles),
        "feature_rows": len(feature_rows),
        "prediction_rows": len(predictions),
        "selected_prediction_rows": sum(
            bool(row["selected_by_frozen_mapping"]) for row in predictions
        ),
        "feature_columns": len(
            [name for name in feature_rows[0] if name.startswith("feature_")]
        ),
        "frozen_mapping": mapping,
        "inference_runtime": "numpy_cpu",
        "source_checkpoint_sha256": expected_checkpoint_hash,
        "numpy_checkpoint_sha256": sha256(args.numpy_checkpoint),
        "frozen_predictions_sha256": sha256(prediction_path),
        "raw_source_checks": raw_checks,
        "hfull_targets_generated_or_read": False,
        "inputs": {
            "selection_sha256": sha256(args.selection),
            "phase28a_summary_sha256": sha256(args.phase28a_summary),
            "frozen_mapping_sha256": sha256(args.frozen_mapping),
            "phase27c_audit_sha256": sha256(args.phase27c_audit),
            "checkpoint_export_audit_sha256": sha256(args.checkpoint_export_audit),
            "model_features_sha256": sha256(args.model_features),
            "raw_manifest_sha256": sha256(raw_manifest_path),
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    forbidden = {"input_lens", "output_lens", "requests", "target_calls_by_12bin_json"}
    checks = {
        "raw_source_hashes_6_of_6": len(raw_checks) == 6 and all(raw_checks.values()),
        "profiles_18": len(profiles) == 18,
        "history_counts_match_18_of_18": all(full_count_checks),
        "feature_rows_324": len(feature_rows) == 324,
        "feature_columns_108": summary["feature_columns"] == 108,
        "prediction_rows_648": len(predictions) == 648,
        "selected_prediction_rows_324": summary["selected_prediction_rows"] == 324,
        "source_checkpoint_hash_matches_phase27c": export_audit[
            "source_checkpoint_sha256"
        ]
        == expected_checkpoint_hash,
        "numpy_checkpoint_hash_matches_export_audit": sha256(args.numpy_checkpoint)
        == export_audit["numpy_checkpoint_sha256"],
        "h0_simulations_exact_162_of_162": len(h0_simulation_checks) == 162
        and all(h0_simulation_checks),
        "no_request_lists_or_targets_in_saved_features": not (
            forbidden & set(feature_rows[0])
        ),
        "hfull_targets_not_generated_or_read": not summary[
            "hfull_targets_generated_or_read"
        ],
    }
    audit = {
        "schema_version": "phase28b-frozen-predictions-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "source_checkpoint_sha256": summary["source_checkpoint_sha256"],
        "numpy_checkpoint_sha256": summary["numpy_checkpoint_sha256"],
        "frozen_predictions_sha256": summary["frozen_predictions_sha256"],
    }
    write_json(args.output_dir / "audit_summary.json", audit)
    if audit["status"] != "PASS":
        raise RuntimeError(audit)
    (args.output_dir / "README.md").write_text(
        f"""# Phase 28B：第二确认集预测冻结

本阶段从Phase 28A冻结的18个窗口聚合108列低维画像，使用Phase 27C已经冻结的增强
bounded-residual checkpoint，并与无参数H0一起生成{len(predictions)}行预测。按Phase 28A
方法映射实际选中的预测为324行：MB1使用H0，MB4/MB16使用增强residual。

脚本没有Hfull target参数，没有生成或读取Hfull标签。完整请求列表只在内存中聚合画像，
没有进入`profiles/`、`dataset/`或Git。`analysis/frozen_predictions.csv.gz`的SHA-256已写入
summary和audit；下一阶段必须先核验该hash，随后才可生成Hfull真值并评测。
"""
    )
    (args.output_dir / "DONE").write_text("PASS\n")
    write_json(
        args.output_dir / "logs/prediction.log",
        {
            "schema_version": "phase28b-prediction-log-v1",
            "status": "PASS",
            "inference_runtime": "numpy_cpu",
            "prediction_rows": len(predictions),
            "hfull_targets_generated_or_read": False,
        },
    )
    files = sorted(
        path
        for path in args.output_dir.rglob("*")
        if path.is_file() and path.name != "manifest.sha256"
    )
    (args.output_dir / "manifest.sha256").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(args.output_dir)}\n" for path in files)
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "profiles": len(profiles),
                "prediction_rows": len(predictions),
                "selected_prediction_rows": summary["selected_prediction_rows"],
                "frozen_predictions_sha256": summary["frozen_predictions_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
