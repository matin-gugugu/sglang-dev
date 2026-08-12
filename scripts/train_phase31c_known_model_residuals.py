#!/usr/bin/env python3
"""Finite TP/PP H0+residual search with target-free fixed prediction freezing."""

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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from train_phase27c_pp_scheduler_feature_predictors import (
    case_record,
    choose_device,
    fit_model,
    parse_histograms,
    predict,
    prepare_development,
    target_decode,
    target_encode,
)


FIT_ROLE = "development_train"
VALIDATION_ROLE = "development_validation"
FIXED_ROLE = "fixed_prediction"
POLICIES = {
    "tp": ("latency", "balanced", "throughput"),
    "pp": ("mb1", "mb4", "mb16"),
}
BIN_EDGES = {
    "tp": np.geomspace(4 * 1024, 512 * 1024 * 1024, 13).tolist(),
    "pp": np.geomspace(4 * 1024, 8 * 1024 * 1024 * 1024, 13).tolist(),
}
THRESHOLDS = {
    "tp": {"calls_wape": 0.10, "bytes_wape": 0.02, "mean_histogram_tv": 0.20, "mean_normalized_log_payload_emd": 0.025, "common_reference_cost_wape": 0.05},
    "pp": {"calls_wape": 0.15, "bytes_wape": 0.03, "mean_histogram_tv": 0.22, "mean_normalized_log_payload_emd": 0.04, "common_reference_cost_wape": 0.05},
}
ALPHAS = (0.25, 0.5, 0.75, 1.0)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    base = root / "experiment-results/phase31b_known_model_hfull_dataset"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=base)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase31c_known_model_residual_training",
    )
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--patience", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


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


def load_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as source:
        return list(csv.DictReader(source))


def feature_sets(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    full = [name for name in rows[0] if name.startswith("feature_")]
    forbidden = [name for name in full if "role" in name or "target" in name]
    if forbidden:
        raise RuntimeError(f"split/target metadata exposed as model features: {forbidden}")
    for name in full:
        try:
            float(rows[0][name])
        except ValueError as error:
            raise RuntimeError(f"non-numeric model feature: {name}") from error
    arrival_tokens = ("_rps", "interarrival", "peak_to_mean", "fano")
    causal = [name for name in full if not any(token in name for token in arrival_tokens)]
    return {"full": full, "no_arrival": causal}


def encoded_from_vectors(calls: np.ndarray, logical_bytes: np.ndarray) -> np.ndarray:
    return np.stack([target_encode(c, b) for c, b in zip(calls, logical_bytes)])


def vectors_from_encoded(encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    calls, logical_bytes = zip(*(target_decode(row) for row in encoded))
    return np.stack(calls), np.stack(logical_bytes)


def calibrated(
    h0_encoded: np.ndarray,
    predicted_calls: np.ndarray,
    predicted_bytes: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    predicted_encoded = encoded_from_vectors(predicted_calls, predicted_bytes)
    final_encoded = h0_encoded + alpha * (predicted_encoded - h0_encoded)
    return vectors_from_encoded(final_encoded)


def records_for_validation(
    rows: list[dict[str, str]],
    arrays: dict[str, np.ndarray],
    predicted: tuple[np.ndarray, np.ndarray],
    parallelism: str,
    method: str,
) -> list[dict]:
    calls, logical_bytes = predicted
    grouped: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["split_role"] == VALIDATION_ROLE:
            grouped[(row["profile_id"], row["model"], row["parallel_size"], row["policy"])].append(index)
    output = []
    for indices in grouped.values():
        if len(indices) != 2 or {rows[index]["phase"] for index in indices} != {"prefill", "decode"}:
            raise RuntimeError("configuration does not contain two phases")
        indices.sort(key=lambda index: rows[index]["phase"])
        for index in indices:
            record = case_record(
                rows[index], method, rows[index]["phase"],
                arrays["target_calls"][index], arrays["target_bytes"][index],
                calls[index], logical_bytes[index], BIN_EDGES[parallelism],
            )
            record["model"] = rows[index]["model"]
            record["profile_id"] = rows[index]["profile_id"]
            output.append(record)
        representative = rows[indices[0]]
        actual_calls = sum((arrays["target_calls"][index] for index in indices))
        actual_bytes = sum((arrays["target_bytes"][index] for index in indices))
        predicted_calls = sum((calls[index] for index in indices))
        predicted_bytes = sum((logical_bytes[index] for index in indices))
        total = case_record(representative, method, "total", actual_calls, actual_bytes, predicted_calls, predicted_bytes, BIN_EDGES[parallelism])
        actual_phase = np.concatenate([arrays["target_calls"][index] for index in indices])
        predicted_phase = np.concatenate([calls[index] for index in indices])
        total["histogram_l1"] = float(np.abs(actual_phase / max(actual_phase.sum(), 1e-12) - predicted_phase / max(predicted_phase.sum(), 1e-12)).sum())
        total["histogram_tv"] = total["histogram_l1"] / 2
        total["model"] = representative["model"]
        total["profile_id"] = representative["profile_id"]
        output.append(total)
    return output


def aggregate(records: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in records:
        for model in ("all", row["model"]):
            for policy in ("all", row["policy"]):
                groups[(row["method"], row["phase"], model, policy)].append(row)
    output = []
    for (method, phase, model, policy), values in sorted(groups.items()):
        actual_calls = sum(float(row["actual_total_calls"]) for row in values)
        actual_bytes = sum(float(row["actual_total_logical_bytes"]) for row in values)
        actual_cost = sum(float(row["actual_common_reference_cost_us"]) for row in values)
        output.append(
            {
                "method": method,
                "phase": phase,
                "model": model,
                "policy": policy,
                "cases": len(values),
                "calls_mape": float(np.mean([float(row["calls_ape"]) for row in values])),
                "calls_wape": sum(float(row["calls_absolute_error"]) for row in values) / max(actual_calls, 1e-12),
                "bytes_mape": float(np.mean([float(row["bytes_ape"]) for row in values])),
                "bytes_wape": sum(float(row["bytes_absolute_error"]) for row in values) / max(actual_bytes, 1e-12),
                "mean_histogram_tv": float(np.mean([float(row["histogram_tv"]) for row in values])),
                "mean_normalized_log_payload_emd": float(np.mean([float(row["normalized_log_payload_emd"]) for row in values])),
                "common_reference_cost_mape": float(np.mean([float(row["cost_ape"]) for row in values])),
                "common_reference_cost_wape": sum(float(row["cost_absolute_error"]) for row in values) / max(actual_cost, 1e-12),
            }
        )
    return output


def headline(metrics: list[dict]) -> dict:
    return next(row for row in metrics if row["phase"] == "total" and row["model"] == "all" and row["policy"] == "all")


def candidate_score(row: dict, h0: dict, parallelism: str) -> float:
    thresholds = THRESHOLDS[parallelism]
    score = sum(float(row[key]) / threshold for key, threshold in thresholds.items())
    if float(row["calls_wape"]) >= float(h0["calls_wape"]):
        score += 3.0
    if float(row["common_reference_cost_wape"]) >= float(h0["common_reference_cost_wape"]):
        score += 3.0
    if float(row["bytes_wape"]) > max(thresholds["bytes_wape"] * 1.5, float(h0["bytes_wape"]) * 1.5):
        score += 2.0
    return score


def model_args(args: argparse.Namespace, learning_rate: float) -> SimpleNamespace:
    return SimpleNamespace(
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=learning_rate,
    )


def fit_configuration(
    rows: list[dict[str, str]],
    arrays: dict[str, np.ndarray],
    features: list[str],
    mode: str,
    learning_rate: float,
    seed: int,
    parallelism: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, dict], tuple[np.ndarray, np.ndarray]]:
    compatible = [{**row, "phase27_role": row["split_role"]} for row in rows]
    if mode == "shared":
        checkpoint, _ = fit_model(
            method="enhanced_bounded_residual", rows=compatible, arrays=arrays,
            feature_names=features, args=model_args(args, learning_rate), device=device, seed=seed,
        )
        prediction = predict(compatible, checkpoint, arrays["h0_encoded"], device)
        return {"shared": checkpoint}, prediction

    calls = np.zeros_like(arrays["h0_calls"])
    logical_bytes = np.zeros_like(arrays["h0_bytes"])
    checkpoints = {}
    for policy_index, policy in enumerate(POLICIES[parallelism]):
        indices = [index for index, row in enumerate(compatible) if row["policy"] == policy]
        subset = [compatible[index] for index in indices]
        subset_arrays = {key: value[indices] for key, value in arrays.items()}
        checkpoint, _ = fit_model(
            method="enhanced_bounded_residual", rows=subset, arrays=subset_arrays,
            feature_names=features, args=model_args(args, learning_rate), device=device, seed=seed + policy_index,
        )
        part_calls, part_bytes = predict(subset, checkpoint, subset_arrays["h0_encoded"], device)
        calls[indices] = part_calls
        logical_bytes[indices] = part_bytes
        checkpoints[policy] = checkpoint
    return checkpoints, (calls, logical_bytes)


def predict_configuration(
    rows: list[dict[str, str]],
    arrays: dict[str, np.ndarray],
    checkpoints: dict[str, dict],
    mode: str,
    parallelism: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    compatible = [{**row, "phase27_role": row["split_role"]} for row in rows]
    if mode == "shared":
        return predict(compatible, checkpoints["shared"], arrays["h0_encoded"], device)
    calls = np.zeros_like(arrays["h0_calls"])
    logical_bytes = np.zeros_like(arrays["h0_bytes"])
    for policy in POLICIES[parallelism]:
        indices = [index for index, row in enumerate(compatible) if row["policy"] == policy]
        subset = [compatible[index] for index in indices]
        part_calls, part_bytes = predict(subset, checkpoints[policy], arrays["h0_encoded"][indices], device)
        calls[indices] = part_calls
        logical_bytes[indices] = part_bytes
    return calls, logical_bytes


def load_parallelism(args: argparse.Namespace, parallelism: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    development = load_rows(args.dataset_dir / f"dataset/{parallelism}_development_examples.csv.gz")
    fixed = load_rows(args.dataset_dir / f"dataset/{parallelism}_fixed_prediction_features.csv.gz")
    return development, fixed


def fixed_prediction_rows(
    rows: list[dict[str, str]],
    methods: dict[str, tuple[np.ndarray, np.ndarray]],
    parallelism: str,
    candidate_id: str,
) -> list[dict]:
    output = []
    for method, (calls, logical_bytes) in methods.items():
        for index, row in enumerate(rows):
            output.append(
                {
                    "example_id": row["example_id"],
                    "profile_id": row["profile_id"],
                    "split_role": row["split_role"],
                    "source": row["source"],
                    "segment": row["segment"],
                    "window_id": row["window_id"],
                    "model": row["model"],
                    "parallelism": parallelism,
                    "parallel_size": row["parallel_size"],
                    "policy": row["policy"],
                    "phase": row["phase"],
                    "method": method,
                    "selected_candidate_id": candidate_id if method == "h0_plus_dnn_residual" else "h0",
                    "predicted_total_calls_per_1000": float(calls[index].sum()),
                    "predicted_total_logical_bytes_per_1000": float(logical_bytes[index].sum()),
                    "predicted_common_reference_cost_us_per_1000": float(5.0 * calls[index].sum() + logical_bytes[index].sum() / 100e9 * 1e6),
                    "predicted_calls_by_12bin_json": json.dumps(calls[index].tolist(), separators=(",", ":")),
                    "predicted_logical_bytes_by_12bin_json": json.dumps(logical_bytes[index].tolist(), separators=(",", ":")),
                }
            )
    return output


def main() -> None:
    args = parse_args()
    for name in ("checkpoints", "analysis", "figures", "logs"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    dataset_summary = json.loads((args.dataset_dir / "summary.json").read_text())
    if dataset_summary["status"] != "PASS" or dataset_summary["fixed_target_state"] != "not_generated":
        raise RuntimeError("Phase31B is not a target-isolated PASS")
    device = choose_device(args.device)
    all_grid_rows = []
    all_validation_records = []
    all_frozen_rows = []
    selected_summaries = {}
    checkpoint_inventory = []

    for parallelism in ("tp", "pp"):
        development, fixed = load_parallelism(args, parallelism)
        if any(name.startswith("target_") for name in fixed[0]):
            raise RuntimeError(f"{parallelism} fixed feature file contains targets")
        role_counts = Counter({row["profile_id"]: row["split_role"] for row in development}.values())
        if role_counts != Counter({FIT_ROLE: 39, VALIDATION_ROLE: 10}):
            raise ValueError({parallelism: role_counts})
        arrays = prepare_development(development)
        fixed_h0_calls, fixed_h0_bytes = parse_histograms(fixed, "h0")
        fixed_arrays = {
            "h0_calls": fixed_h0_calls,
            "h0_bytes": fixed_h0_bytes,
            "h0_encoded": encoded_from_vectors(fixed_h0_calls, fixed_h0_bytes),
        }
        sets = feature_sets(development)
        h0_records = records_for_validation(development, arrays, (arrays["h0_calls"], arrays["h0_bytes"]), parallelism, "h0")
        h0_metrics = aggregate(h0_records)
        h0_headline = headline(h0_metrics)

        configs = []
        for feature_set in ("full", "no_arrival"):
            for mode in ("shared", "policy_heads"):
                for learning_rate in (3e-4, 1e-3, 3e-3):
                    configs.append({"feature_set": feature_set, "mode": mode, "learning_rate": learning_rate})
        candidate_results = []
        for config_index, config in enumerate(configs):
            candidate_id = f"{parallelism}_c{config_index + 1:02d}_{config['feature_set']}_{config['mode']}_lr{config['learning_rate']:g}"
            checkpoints, raw_prediction = fit_configuration(
                development, arrays, sets[config["feature_set"]], config["mode"], config["learning_rate"],
                args.seed + config_index * 17, parallelism, args, device,
            )
            best = None
            for alpha in ALPHAS:
                prediction = calibrated(arrays["h0_encoded"], *raw_prediction, alpha)
                records = records_for_validation(development, arrays, prediction, parallelism, candidate_id)
                metrics = aggregate(records)
                head = headline(metrics)
                score = candidate_score(head, h0_headline, parallelism)
                row = {"parallelism": parallelism, "candidate_id": candidate_id, **config, "alpha": alpha, "score": score, **{key: head[key] for key in THRESHOLDS[parallelism]}}
                if best is None or score < best["row"]["score"]:
                    best = {"row": row, "checkpoints": checkpoints, "prediction": prediction}
            all_grid_rows.append(best["row"])
            candidate_results.append(best)

        candidate_results.sort(key=lambda value: value["row"]["score"])
        top_two = candidate_results[:2]
        ensemble_results = []
        for rank, initial in enumerate(top_two, 1):
            config = initial["row"]
            seed_predictions_development = []
            seed_predictions_fixed = []
            saved = []
            for seed_offset in (0, 101, 202):
                seed = args.seed + seed_offset
                checkpoints, raw_development = fit_configuration(
                    development, arrays, sets[config["feature_set"]], config["mode"], float(config["learning_rate"]),
                    seed, parallelism, args, device,
                )
                raw_fixed = predict_configuration(fixed, fixed_arrays, checkpoints, config["mode"], parallelism, device)
                seed_predictions_development.append(encoded_from_vectors(*raw_development) - arrays["h0_encoded"])
                seed_predictions_fixed.append(encoded_from_vectors(*raw_fixed) - fixed_arrays["h0_encoded"])
                path = args.output_dir / "checkpoints" / f"{parallelism}_top{rank}_seed{seed}.pt"
                torch.save({"parallelism": parallelism, "candidate": config, "seed": seed, "heads": checkpoints}, path)
                item = {"parallelism": parallelism, "candidate_rank": rank, "candidate_id": config["candidate_id"], "seed": seed, "path": str(path.relative_to(args.output_dir)), "sha256": sha256(path), "bytes": path.stat().st_size}
                checkpoint_inventory.append(item)
                saved.append(item)

            mean_development_residual = np.mean(seed_predictions_development, axis=0)
            mean_fixed_residual = np.mean(seed_predictions_fixed, axis=0)
            best_ensemble = None
            for alpha in ALPHAS:
                prediction = vectors_from_encoded(arrays["h0_encoded"] + alpha * mean_development_residual)
                records = records_for_validation(development, arrays, prediction, parallelism, f"{config['candidate_id']}_ensemble")
                metrics = aggregate(records)
                head = headline(metrics)
                score = candidate_score(head, h0_headline, parallelism)
                if best_ensemble is None or score < best_ensemble["score"]:
                    best_ensemble = {"score": score, "alpha": alpha, "prediction": prediction, "records": records, "metrics": metrics, "headline": head}
            fixed_prediction = vectors_from_encoded(fixed_arrays["h0_encoded"] + best_ensemble["alpha"] * mean_fixed_residual)
            ensemble_results.append({"config": config, "checkpoints": saved, "fixed_prediction": fixed_prediction, **best_ensemble})

        ensemble_results.sort(key=lambda value: value["score"])
        selected = ensemble_results[0]
        selected_candidate_id = f"{selected['config']['candidate_id']}_3seed_alpha{selected['alpha']}"
        selected_summaries[parallelism] = {
            "candidate_id": selected_candidate_id,
            "feature_set": selected["config"]["feature_set"],
            "mode": selected["config"]["mode"],
            "learning_rate": selected["config"]["learning_rate"],
            "alpha": selected["alpha"],
            "validation_headline": selected["headline"],
            "h0_validation_headline": h0_headline,
            "nonzero_residual_fraction": float(np.mean(np.abs(encoded_from_vectors(*selected["prediction"]) - arrays["h0_encoded"]) > 1e-8)),
            "checkpoints": selected["checkpoints"],
        }
        all_validation_records.extend(h0_records)
        all_validation_records.extend(selected["records"])
        all_frozen_rows.extend(
            fixed_prediction_rows(
                fixed,
                {"h0": (fixed_h0_calls, fixed_h0_bytes), "h0_plus_dnn_residual": selected["fixed_prediction"]},
                parallelism,
                selected_candidate_id,
            )
        )

    write_csv(args.output_dir / "analysis/candidate_grid.csv", all_grid_rows)
    write_csv_gz(args.output_dir / "analysis/selected_validation_predictions.csv.gz", all_validation_records)
    write_csv(args.output_dir / "analysis/checkpoint_inventory.csv", checkpoint_inventory)
    write_csv_gz(args.output_dir / "analysis/frozen_fixed_prediction.csv.gz", all_frozen_rows)
    prediction_path = args.output_dir / "analysis/frozen_fixed_prediction.csv.gz"
    prediction_sha = sha256(prediction_path)

    try:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        for axis, parallelism in zip(axes, ("tp", "pp")):
            h0 = selected_summaries[parallelism]["h0_validation_headline"]
            dnn = selected_summaries[parallelism]["validation_headline"]
            labels = ["calls WAPE", "bytes WAPE", "TV", "cost WAPE"]
            h0_values = [h0["calls_wape"], h0["bytes_wape"], h0["mean_histogram_tv"], h0["common_reference_cost_wape"]]
            dnn_values = [dnn["calls_wape"], dnn["bytes_wape"], dnn["mean_histogram_tv"], dnn["common_reference_cost_wape"]]
            x = np.arange(len(labels))
            axis.bar(x - 0.18, h0_values, 0.36, label="H0")
            axis.bar(x + 0.18, dnn_values, 0.36, label="H0+DNN residual")
            axis.set_xticks(x, labels, rotation=25, ha="right")
            axis.set_title(parallelism.upper())
            axis.legend()
        figure.tight_layout()
        figure.savefig(args.output_dir / "figures/validation_comparison.png", dpi=180)
        plt.close(figure)
    except Exception as error:
        write_json(args.output_dir / "figures/plot_failure.json", {"error": repr(error)})

    checks = {
        "dataset_pass_and_fixed_targets_absent": dataset_summary["status"] == "PASS" and dataset_summary["fixed_target_state"] == "not_generated",
        "grid_12_per_parallelism": Counter(row["parallelism"] for row in all_grid_rows) == Counter({"tp": 12, "pp": 12}),
        "top2_three_seed_checkpoints_each": Counter(row["parallelism"] for row in checkpoint_inventory) == Counter({"tp": 6, "pp": 6}),
        "fixed_prediction_rows_2160": len(all_frozen_rows) == 10 * 2 * 3 * 3 * 3 * 2 * 2,
        "fixed_prediction_methods_h0_and_residual": {row["method"] for row in all_frozen_rows} == {"h0", "h0_plus_dnn_residual"},
        "nonzero_dnn_residual_tp_pp": all(selected_summaries[p]["nonzero_residual_fraction"] > 0 for p in ("tp", "pp")),
        "fixed_targets_not_a_script_input": not any("target" in name for name in vars(args) if name != "dataset_dir"),
        "all_metrics_finite": all(math.isfinite(float(value)) for row in all_grid_rows for key, value in row.items() if key in {"score", *THRESHOLDS[row["parallelism"]]}),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = {
        "schema_version": "phase31c-known-model-residual-training-v1",
        "status": status,
        "objective": "finite H0 plus DNN residual iteration for known three-model first-stage closure",
        "device": str(device),
        "search_limits": {"candidates_per_parallelism": 12, "top_candidates_for_confirmation": 2, "seeds_per_top_candidate": 3, "alphas": list(ALPHAS)},
        "selected": selected_summaries,
        "frozen_prediction_sha256": prediction_sha,
        "fixed_targets_read": False,
        "counts": {"grid_rows": len(all_grid_rows), "checkpoints": len(checkpoint_inventory), "frozen_prediction_rows": len(all_frozen_rows)},
        "inputs": {"dataset_summary_sha256": sha256(args.dataset_dir / "summary.json"), "tp_development_sha256": sha256(args.dataset_dir / "dataset/tp_development_examples.csv.gz"), "pp_development_sha256": sha256(args.dataset_dir / "dataset/pp_development_examples.csv.gz"), "tp_fixed_features_sha256": sha256(args.dataset_dir / "dataset/tp_fixed_prediction_features.csv.gz"), "pp_fixed_features_sha256": sha256(args.dataset_dir / "dataset/pp_fixed_prediction_features.csv.gz")},
        "checks": checks,
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "audit_summary.json", {"schema_version": "phase31c-training-audit-v1", "status": status, "checks": checks, "frozen_prediction_sha256": prediction_sha})
    (args.output_dir / "README.md").write_text(f"""# Phase 31C：三模型 TP/PP H0+DNN residual 有限训练

本阶段只读取Phase31B的39个训练画像和10个验证画像。10个固定预测画像只有低维特征与H0，不存在Hfull target；预测文件已在target生成前冻结，SHA-256为`{prediction_sha}`。

## 有限搜索

TP和PP各筛选12组配置：完整/去arrival特征、共享/按policy小头、三档学习率。每个方向只取验证最好的2组做3-seed训练，最终仍由验证集选择一组。没有整模型留出，三个已知模型同时参与训练、验证和固定预测。

最终模型始终是`H0 + DNN residual`。DNN输出经过H0空间的有界残差和验证集校准alpha；H0同时保留为对照。固定预测集没有参与网络、特征、alpha或checkpoint选择。

## 验证结果入口

TP与PP的最终验证指标见`summary.json`中的`selected`；全部24组初筛见`analysis/candidate_grid.csv`。下一步必须先归档本阶段与冻结预测，然后另行生成固定预测Hfull target并评测。
""")
    write_json(args.output_dir / "logs/training.log", {"event": "phase31c_training_complete", "status": status, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "repository_head_at_training": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "platform": platform.platform(), "device": str(device), "fixed_targets_read": False})
    (args.output_dir / "DONE").write_text(f"{status}\n")
    manifest = [f"{sha256(path)}  {path.relative_to(args.output_dir)}" for path in sorted(args.output_dir.rglob("*")) if path.is_file() and path.name != "manifest.sha256"]
    (args.output_dir / "manifest.sha256").write_text("\n".join(manifest) + "\n")
    if status != "PASS":
        raise RuntimeError(checks)
    print(json.dumps({"status": status, "selected": selected_summaries, "prediction_sha256": prediction_sha}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
