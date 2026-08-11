#!/usr/bin/env python3
"""Train the pure-PP service-profile PatternDemand predictor.

The direct network is an ablation.  The deployable path is the transparent PP
microbatch H0 plus a bounded neural residual.  All reported rows come from
grouped holdouts rather than random row splits.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from train_profiledemand_v1 import (
    bounded_residual,
    fit_network,
    sha256,
    target_decode,
    target_encode,
    validation_indices,
    write_csv,
)


BIN_COUNT = 12
METHODS = ("direct_dnn", "h0", "h0_residual")
PROFILE_SCALARS = (
    "rps",
    "interarrival_cv",
    "peak_to_mean_1s",
    "fano_1s",
    "input_mean_capped",
    "output_mean_capped",
    "lm_correlation_capped",
    "survival_m_gt_8",
    "survival_m_gt_16",
    "survival_m_gt_32",
    "survival_m_gt_64",
)
LOG_PROFILE_FEATURES = {
    "rps",
    "interarrival_cv",
    "peak_to_mean_1s",
    "fano_1s",
    "input_mean_capped",
    "output_mean_capped",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=Path,
        default=root
        / "experiment-results/phase21b_pp_offline_profiledemand/qwen3-8b-labels-v1/labels.csv",
    )
    parser.add_argument(
        "--h0",
        type=Path,
        default=root
        / "experiment-results/phase21b_pp_offline_profiledemand/h0-v1/h0_samples.csv",
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=root / "experiment-results/phase16_service_profiles/service_profiles.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase22_pp_predictor/qwen3-8b-offline-v1",
    )
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument(
        "--dataset-mode",
        choices=("canonical_draining", "profiled_online"),
        default="canonical_draining",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def base_features(row: dict, profile: dict) -> list[float]:
    values = []
    for name in PROFILE_SCALARS:
        value = float(profile[name])
        values.append(math.log1p(max(value, 0.0)) if name in LOG_PROFILE_FEATURES else value)
    values.extend(float(value) for value in json.loads(profile["joint_lm_4x4_json"]))
    values.extend(
        [
            math.log2(int(row["pp_size"])),
            math.log2(int(row["pp_max_micro_batch_size"])),
            float(row["phase"] == "prefill"),
            float(row["phase"] == "decode"),
            math.log2(4096),
            math.log2(2),
            math.log2(2),
            math.log2(4096),
        ]
    )
    return values


def h0_summary_features(calls: np.ndarray, logical_bytes: np.ndarray) -> list[float]:
    calls_total = max(float(calls.sum()), 1e-9)
    bytes_total = max(float(logical_bytes.sum()), 1e-9)
    return [
        math.log1p(calls_total),
        math.log1p(bytes_total),
        *(calls / calls_total).tolist(),
        *(logical_bytes / bytes_total).tolist(),
    ]


def grouped_folds(rows: list[dict]) -> dict[str, dict[str, set[int]]]:
    definitions: dict[str, dict[str, set[int]]] = defaultdict(dict)
    for profile_id in sorted({row["profile_id"] for row in rows}):
        definitions["profile_holdout"][profile_id] = {
            index for index, row in enumerate(rows) if row["profile_id"] == profile_id
        }
    for value in sorted({row["pp_max_micro_batch_size"] for row in rows}, key=int):
        definitions["strategy_holdout"][f"mb{value}"] = {
            index
            for index, row in enumerate(rows)
            if row["pp_max_micro_batch_size"] == value
        }
    for value in sorted({row["pp_size"] for row in rows}, key=int):
        definitions["pp_holdout"][f"pp{value}"] = {
            index for index, row in enumerate(rows) if row["pp_size"] == value
        }
    return definitions


def normalized(vector: np.ndarray) -> np.ndarray:
    total = float(vector.sum())
    return vector / total if total else np.zeros_like(vector)


def prediction_record(
    row: dict,
    evaluation: str,
    fold: str,
    method: str,
    actual_calls: np.ndarray,
    actual_bytes: np.ndarray,
    predicted_calls: np.ndarray,
    predicted_bytes: np.ndarray,
) -> dict:
    actual_calls_total = float(actual_calls.sum())
    actual_bytes_total = float(actual_bytes.sum())
    predicted_calls_total = float(predicted_calls.sum())
    predicted_bytes_total = float(predicted_bytes.sum())
    actual_share = normalized(actual_calls)
    predicted_share = normalized(predicted_calls)
    return {
        "evaluation": evaluation,
        "fold": fold,
        "method": method,
        "sample_id": row["sample_id"],
        "model": row["model"],
        "profile_id": row["profile_id"],
        "phase": row["phase"],
        "pp_size": row["pp_size"],
        "pp_max_micro_batch_size": row["pp_max_micro_batch_size"],
        "calls_vector_absolute_error": float(
            np.abs(predicted_calls - actual_calls).sum()
        ),
        "bytes_vector_absolute_error": float(
            np.abs(predicted_bytes - actual_bytes).sum()
        ),
        "actual_total_calls": actual_calls_total,
        "predicted_total_calls": predicted_calls_total,
        "actual_total_bytes": actual_bytes_total,
        "predicted_total_bytes": predicted_bytes_total,
        "calls_ape": abs(predicted_calls_total - actual_calls_total)
        / max(actual_calls_total, 1e-9),
        "bytes_ape": abs(predicted_bytes_total - actual_bytes_total)
        / max(actual_bytes_total, 1e-9),
        "histogram_l1": float(np.abs(predicted_share - actual_share).sum()),
        "log_payload_emd": float(
            np.abs(np.cumsum(predicted_share - actual_share)[:-1]).sum()
            / (BIN_COUNT - 1)
        ),
        "actual_calls_by_12bin_json": json.dumps(
            actual_calls.tolist(), separators=(",", ":")
        ),
        "predicted_calls_by_12bin_json": json.dumps(
            predicted_calls.tolist(), separators=(",", ":")
        ),
        "actual_bytes_by_12bin_json": json.dumps(
            actual_bytes.tolist(), separators=(",", ":")
        ),
        "predicted_bytes_by_12bin_json": json.dumps(
            predicted_bytes.tolist(), separators=(",", ":")
        ),
    }


def aggregate_metrics(predictions: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in predictions:
        for scope in ("all", row["phase"]):
            groups[(row["evaluation"], row["method"], scope)].append(row)
    result = []
    for (evaluation, method, scope), rows in sorted(groups.items()):
        calls_denominator = sum(float(row["actual_total_calls"]) for row in rows)
        bytes_denominator = sum(float(row["actual_total_bytes"]) for row in rows)
        calls_ape = [float(row["calls_ape"]) for row in rows]
        bytes_ape = [float(row["bytes_ape"]) for row in rows]
        result.append(
            {
                "evaluation": evaluation,
                "method": method,
                "scope": scope,
                "samples": len(rows),
                "calls_vector_wape": sum(
                    float(row["calls_vector_absolute_error"]) for row in rows
                )
                / calls_denominator,
                "bytes_vector_wape": sum(
                    float(row["bytes_vector_absolute_error"]) for row in rows
                )
                / bytes_denominator,
                "total_calls_mape": float(np.mean(calls_ape)),
                "total_calls_p95_ape": float(np.percentile(calls_ape, 95)),
                "total_bytes_mape": float(np.mean(bytes_ape)),
                "total_bytes_p95_ape": float(np.percentile(bytes_ape, 95)),
                "mean_histogram_l1": float(
                    np.mean([float(row["histogram_l1"]) for row in rows])
                ),
                "mean_log_payload_emd": float(
                    np.mean([float(row["log_payload_emd"]) for row in rows])
                ),
            }
        )
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = read_csv(args.labels)
    h0_by_id = {row["sample_id"]: row for row in read_csv(args.h0)}
    profiles = {row["profile_id"]: row for row in read_csv(args.profiles)}
    h0_keys = [row.get("h0_sample_id", row["sample_id"]) for row in labels]
    missing_h0 = sorted(set(h0_keys) - set(h0_by_id))
    if missing_h0:
        raise ValueError(f"labels reference missing H0 samples: {missing_h0[:5]}")

    actual_calls = np.stack(
        [
            np.asarray(json.loads(row["calls_by_12bin_per_1000_json"]), dtype=np.float64)
            for row in labels
        ]
    )
    actual_bytes = np.stack(
        [
            np.asarray(
                json.loads(row["logical_bytes_by_12bin_per_1000_json"]),
                dtype=np.float64,
            )
            for row in labels
        ]
    )
    h0_calls = np.stack(
        [
            np.asarray(
                json.loads(
                    h0_by_id[row.get("h0_sample_id", row["sample_id"])][
                        "calls_by_12bin_per_1000_json"
                    ]
                ),
                dtype=np.float64,
            )
            for row in labels
        ]
    )
    h0_bytes = np.stack(
        [
            np.asarray(
                json.loads(
                    h0_by_id[row.get("h0_sample_id", row["sample_id"])][
                        "logical_bytes_by_12bin_per_1000_json"
                    ]
                ),
                dtype=np.float64,
            )
            for row in labels
        ]
    )
    encoded = np.stack(
        [target_encode(calls, byte) for calls, byte in zip(actual_calls, actual_bytes)]
    )
    h0_encoded = np.stack(
        [target_encode(calls, byte) for calls, byte in zip(h0_calls, h0_bytes)]
    )
    residual = bounded_residual(encoded, h0_encoded)
    direct_features = np.asarray(
        [base_features(row, profiles[row["profile_id"]]) for row in labels],
        dtype=np.float32,
    )
    residual_features = np.asarray(
        [
            base_features(row, profiles[row["profile_id"]])
            + h0_summary_features(h0_calls[index], h0_bytes[index])
            for index, row in enumerate(labels)
        ],
        dtype=np.float32,
    )

    predictions = []
    training_history = []
    definitions = grouped_folds(labels)
    all_indices = np.arange(len(labels), dtype=int)
    trained_folds = 0
    for evaluation, folds in definitions.items():
        for fold_number, (fold, test_set) in enumerate(sorted(folds.items())):
            test = np.asarray(sorted(test_set), dtype=int)
            train = np.asarray(
                [index for index in all_indices if index not in test_set], dtype=int
            )
            fit, validation = validation_indices(
                labels, train, args.seed + fold_number
            )
            direct_prediction, _, direct_history = fit_network(
                direct_features,
                encoded,
                fit,
                validation,
                args,
                args.seed + trained_folds * 2,
            )
            residual_prediction, _, residual_history = fit_network(
                residual_features,
                residual,
                fit,
                validation,
                args,
                args.seed + trained_folds * 2 + 1,
            )
            for method, history in (
                ("direct_dnn", direct_history),
                ("h0_residual", residual_history),
            ):
                for item in history:
                    training_history.append(
                        {
                            "evaluation": evaluation,
                            "fold": fold,
                            "method": method,
                            **item,
                        }
                    )
            for index in test:
                encoded_methods = {
                    "direct_dnn": direct_prediction[index],
                    "h0_residual": h0_encoded[index] + residual_prediction[index],
                }
                for method in METHODS:
                    if method == "h0":
                        predicted_calls, predicted_bytes = h0_calls[index], h0_bytes[index]
                    else:
                        predicted_calls, predicted_bytes = target_decode(
                            encoded_methods[method]
                        )
                    predictions.append(
                        prediction_record(
                            labels[index],
                            evaluation,
                            fold,
                            method,
                            actual_calls[index],
                            actual_bytes[index],
                            predicted_calls,
                            predicted_bytes,
                        )
                    )
            trained_folds += 1

    metrics = aggregate_metrics(predictions)
    write_csv(args.output_dir / "metrics.csv", metrics)
    with gzip.open(args.output_dir / "holdout_predictions.csv.gz", "wt", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(predictions[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(predictions)
    with (args.output_dir / "training_history.jsonl").open("w") as output:
        for row in training_history:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")

    fit, validation = validation_indices(labels, all_indices, args.seed + 9999)
    _, checkpoint, final_history = fit_network(
        residual_features,
        residual,
        fit,
        validation,
        args,
        args.seed + 9999,
    )
    checkpoint.update(
        {
            "schema_version": "phase21b-pure-pp-h0-residual-v1",
            "model": "qwen3-8b",
            "input_contract": (
                "steady service profile + numeric model structure + candidate PP size + "
                "microbatch/chunk policy + transparent H0"
            ),
            "output_contract": (
                "per-boundary 12-bin calls and logical bytes per 1000 requests"
            ),
            "labels_sha256": sha256(args.labels),
            "h0_sha256": sha256(args.h0),
        }
    )
    torch.save(checkpoint, args.output_dir / "formal_h0_residual_model.pt")
    with (args.output_dir / "final_training_history.jsonl").open("w") as output:
        for row in final_history:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")

    headline = {
        f"{row['evaluation']}:{row['method']}:{row['scope']}": row
        for row in metrics
        if row["scope"] == "all"
    }
    summary = {
        "schema_version": "phase21b-pure-pp-predictor-evaluation-v1",
        "status": "PASS",
        "labels": len(labels),
        "profiles": len(profiles),
        "pp_sizes": sorted({int(row["pp_size"]) for row in labels}),
        "microbatch_sizes": sorted(
            {int(row["pp_max_micro_batch_size"]) for row in labels}
        ),
        "methods": list(METHODS),
        "evaluation_regimes": {
            name: sorted(folds) for name, folds in definitions.items()
        },
        "trained_outer_folds": trained_folds,
        "dataset_mode": args.dataset_mode,
        "headline": headline,
        "boundary": (
            "This is a one-model PP checkpoint. Canonical draining labels close the "
            "structural base; profiled-online labels calibrate arrival-driven batching. "
            "Leave-one-model-out remains a separate three-model addition."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "DONE").write_text("PASS\n")
    files = sorted(
        path
        for path in args.output_dir.iterdir()
        if path.is_file() and path.name != "manifest.sha256"
    )
    (args.output_dir / "manifest.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
