#!/usr/bin/env python3
"""Train aligned TP predictors without reading either confirmation target file."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from train_phase27c_pp_scheduler_feature_predictors import (
    LEARNED_METHODS,
    METHODS,
    PHASES,
    case_record,
    choose_device,
    deterministic_gzip,
    fit_model,
    parse_histograms,
    predict,
    prepare_development,
    sha256,
    target_encode,
    write_csv,
    write_csv_gz,
    write_json,
)


FIT_ROLE = "development_train"
VALIDATION_ROLE = "development_validation"
FIRST_CONFIRMATION_ROLE = "independent_confirmation"
SECOND_CONFIRMATION_ROLE = "second_independent_confirmation"
POLICIES = ("latency", "balanced", "throughput")
TP_SIZES = (2, 4, 8)
BIN_EDGES = np.geomspace(4 * 1024, 512 * 1024 * 1024, 13).tolist()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    base = root / "experiment-results/phase29b_tp_hfull_dataset"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--development-dataset",
        type=Path,
        default=base / "dataset/development_examples.csv.gz",
    )
    parser.add_argument(
        "--first-confirmation-features",
        type=Path,
        default=base / "dataset/first_confirmation_features.csv.gz",
    )
    parser.add_argument(
        "--second-confirmation-features",
        type=Path,
        default=base / "dataset/second_confirmation_features.csv.gz",
    )
    parser.add_argument(
        "--dataset-summary", type=Path, default=base / "summary.json"
    )
    parser.add_argument(
        "--feature-columns", type=Path, default=base / "feature_columns.json"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase29c_tp_aligned_training",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as source:
        return list(csv.DictReader(source))


def validation_records(
    rows: list[dict[str, str]],
    arrays: dict[str, np.ndarray],
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["phase27_role"] == VALIDATION_ROLE:
            grouped[
                (row["profile_id"], row["model"], row["parallel_size"], row["policy"])
            ].append(index)
    records = []
    for method in METHODS:
        predicted_calls, predicted_bytes = predictions[method]
        for indices in grouped.values():
            if len(indices) != 2 or {rows[index]["phase"] for index in indices} != set(
                PHASES
            ):
                raise ValueError("validation configuration lacks exactly two phases")
            indices = sorted(indices, key=lambda index: rows[index]["phase"])
            for index in indices:
                record = case_record(
                    rows[index],
                    method,
                    rows[index]["phase"],
                    arrays["target_calls"][index],
                    arrays["target_bytes"][index],
                    predicted_calls[index],
                    predicted_bytes[index],
                    BIN_EDGES,
                )
                record["model"] = rows[index]["model"]
                records.append(record)
            representative = rows[indices[0]]
            actual_calls = sum((arrays["target_calls"][index] for index in indices))
            actual_bytes = sum((arrays["target_bytes"][index] for index in indices))
            predicted_total_calls = sum((predicted_calls[index] for index in indices))
            predicted_total_bytes = sum((predicted_bytes[index] for index in indices))
            total = case_record(
                representative,
                method,
                "total",
                actual_calls,
                actual_bytes,
                predicted_total_calls,
                predicted_total_bytes,
                BIN_EDGES,
            )
            actual_phase_aware = np.concatenate(
                [arrays["target_calls"][index] for index in indices]
            )
            predicted_phase_aware = np.concatenate(
                [predicted_calls[index] for index in indices]
            )
            actual_distribution = actual_phase_aware / max(
                float(actual_phase_aware.sum()), 1e-12
            )
            predicted_distribution = predicted_phase_aware / max(
                float(predicted_phase_aware.sum()), 1e-12
            )
            total["histogram_l1"] = float(
                np.abs(predicted_distribution - actual_distribution).sum()
            )
            total["histogram_tv"] = total["histogram_l1"] / 2
            total["model"] = representative["model"]
            records.append(total)
    return records


def aggregate_records(records: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str, str, str, str], list[dict]] = defaultdict(list)
    for row in records:
        for model in ("all", row["model"]):
            for parallel_size in ("all", row["parallel_size"]):
                for policy in ("all", row["policy"]):
                    for segment in ("all", row["segment"]):
                        groups[
                            (
                                row["method"],
                                row["phase"],
                                model,
                                parallel_size,
                                policy,
                                segment,
                            )
                        ].append(row)
    output = []
    for key, values in sorted(groups.items()):
        method, phase, model, parallel_size, policy, segment = key
        actual_calls = sum(float(row["actual_total_calls"]) for row in values)
        actual_bytes = sum(float(row["actual_total_logical_bytes"]) for row in values)
        actual_cost = sum(float(row["actual_common_reference_cost_us"]) for row in values)
        output.append(
            {
                "method": method,
                "phase": phase,
                "model": model,
                "parallel_size": parallel_size,
                "policy": policy,
                "segment": segment,
                "cases": len(values),
                "calls_mape": float(np.mean([float(row["calls_ape"]) for row in values])),
                "calls_wape": sum(float(row["calls_absolute_error"]) for row in values)
                / actual_calls,
                "bytes_mape": float(np.mean([float(row["bytes_ape"]) for row in values])),
                "bytes_wape": sum(float(row["bytes_absolute_error"]) for row in values)
                / actual_bytes,
                "mean_histogram_l1": float(
                    np.mean([float(row["histogram_l1"]) for row in values])
                ),
                "mean_histogram_tv": float(
                    np.mean([float(row["histogram_tv"]) for row in values])
                ),
                "mean_normalized_log_payload_emd": float(
                    np.mean(
                        [float(row["normalized_log_payload_emd"]) for row in values]
                    )
                ),
                "common_reference_cost_mape": float(
                    np.mean([float(row["cost_ape"]) for row in values])
                ),
                "common_reference_cost_wape": sum(
                    float(row["cost_absolute_error"]) for row in values
                )
                / actual_cost,
            }
        )
    return output


def select_candidates(metrics: list[dict]) -> list[dict]:
    decisions = []
    residuals = ("legacy_bounded_residual", "enhanced_bounded_residual")
    for policy in POLICIES:
        lookup = {
            row["method"]: row
            for row in metrics
            if row["phase"] == "total"
            and row["model"] == "all"
            and row["parallel_size"] == "all"
            and row["policy"] == policy
            and row["segment"] == "all"
        }
        h0 = lookup["h0"]
        scored = []
        for method in residuals:
            value = lookup[method]
            fields = (
                "calls_mape",
                "mean_histogram_tv",
                "common_reference_cost_mape",
            )
            wins = sum(float(value[field]) < float(h0[field]) for field in fields)
            cost_guard = float(value["common_reference_cost_mape"]) <= 1.10 * float(
                h0["common_reference_cost_mape"]
            )
            ratio_sum = sum(
                float(value[field]) / max(float(h0[field]), 1e-12) for field in fields
            )
            scored.append((method, wins, cost_guard, ratio_sum))
        eligible = [row for row in scored if row[1] >= 2 and row[2]]
        selected = (
            min(eligible, key=lambda row: (-row[1], row[3], row[0]))[0]
            if eligible
            else "h0"
        )
        decisions.append(
            {
                "policy": policy,
                "selected_method": selected,
                "selection_source": "development_validation_only",
                "rule": "at_least_2_of_calls_tv_cost_wins_and_cost_mape_within_110pct_of_h0",
                "h0_calls_mape": h0["calls_mape"],
                "h0_histogram_tv": h0["mean_histogram_tv"],
                "h0_cost_mape": h0["common_reference_cost_mape"],
                "legacy_wins": scored[0][1],
                "legacy_cost_guard": scored[0][2],
                "enhanced_wins": scored[1][1],
                "enhanced_cost_guard": scored[1][2],
            }
        )
    return decisions


def frozen_prediction_rows(
    rows: list[dict[str, str]],
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[dict]:
    output = []
    for method in METHODS:
        calls, logical_bytes = predictions[method]
        for index, row in enumerate(rows):
            output.append(
                {
                    "training_id": row["training_id"],
                    "profile_id": row["profile_id"],
                    "role": row["role"],
                    "source": row["source"],
                    "segment": row["segment"],
                    "model": row["model"],
                    "parallelism": "tp",
                    "parallel_size": row["parallel_size"],
                    "policy": row["policy"],
                    "phase": row["phase"],
                    "method": method,
                    "predicted_total_calls_per_1000": float(calls[index].sum()),
                    "predicted_total_logical_bytes_per_1000": float(
                        logical_bytes[index].sum()
                    ),
                    "predicted_common_reference_cost_us_per_1000": float(
                        5.0 * calls[index].sum()
                        + logical_bytes[index].sum() / (100.0 * 1e9) * 1e6
                    ),
                    "predicted_calls_by_12bin_json": json.dumps(
                        calls[index].tolist(), separators=(",", ":")
                    ),
                    "predicted_logical_bytes_by_12bin_json": json.dumps(
                        logical_bytes[index].tolist(), separators=(",", ":")
                    ),
                    "prediction_frozen_before_confirmation_target_access": True,
                }
            )
    return output


def predict_feature_artifact(
    rows: list[dict[str, str]], checkpoints: dict[str, dict], device: torch.device
) -> list[dict]:
    h0_calls, h0_bytes = parse_histograms(rows, "h0")
    h0_encoded = np.stack(
        [target_encode(calls, values) for calls, values in zip(h0_calls, h0_bytes)]
    )
    predictions = {"h0": (h0_calls, h0_bytes)}
    for method, checkpoint in checkpoints.items():
        predictions[method] = predict(rows, checkpoint, h0_encoded, device)
    return frozen_prediction_rows(rows, predictions)


def plot_validation(path: Path, headline: dict[str, dict]) -> None:
    import matplotlib.pyplot as plt

    labels = ("H0", "Legacy residual", "Enhanced residual", "Direct")
    colors = ("#4C78A8", "#A0A0A0", "#F58518", "#B8B8B8")
    specs = (
        ("calls_mape", "Total calls MAPE", 100.0, "%"),
        ("mean_histogram_tv", "Histogram TV", 1.0, ""),
        ("common_reference_cost_mape", "Common cost MAPE", 100.0, "%"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for axis, (metric, title, scale, suffix) in zip(axes, specs):
        values = [headline[method][metric] * scale for method in METHODS]
        bars = axis.bar(labels, values, color=colors, width=0.68)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22, linewidth=0.8)
        axis.tick_params(axis="x", rotation=18)
        axis.spines[["top", "right"]].set_visible(False)
        upper = max(values) * 1.18 if max(values) > 0 else 1.0
        axis.set_ylim(0, upper)
        for bar, value in zip(bars, values):
            label = f"{value:.1f}{suffix}" if suffix else f"{value:.3f}"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + upper * 0.025,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )
    figure.suptitle("Phase 29C development validation: aligned TP predictors")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def readme(summary: dict) -> str:
    table = [
        "| 方法 | calls MAPE/WAPE | bytes MAPE/WAPE | TV | norm EMD | cost MAPE/WAPE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = summary["validation_headline"][method]
        table.append(
            "| {method} | {cm:.2%} / {cw:.2%} | {bm:.2%} / {bw:.2%} | {tv:.4f} | {emd:.4f} | {cost:.2%} / {costw:.2%} |".format(
                method=method,
                cm=row["calls_mape"],
                cw=row["calls_wape"],
                bm=row["bytes_mape"],
                bw=row["bytes_wape"],
                tv=row["mean_histogram_tv"],
                emd=row["mean_normalized_log_payload_emd"],
                cost=row["common_reference_cost_mape"],
                costw=row["common_reference_cost_wape"],
            )
        )
    decisions = "\n".join(
        f"- {row['policy']}：`{row['selected_method']}`"
        for row in summary["candidate_decisions"]
    )
    return f"""# Phase 29C：对齐三模型TP残差DNN训练

状态：**{summary['status']}**。本阶段在Phase 29B的30个开发训练画像上拟合，并只用12个
开发验证画像早停。训练输入是低维历史画像、模型结构、固定TP size、固定执行策略、phase
和compact32 H0；训练真值是完整历史窗口Hfull teacher消息直方图。

## 开发验证集结果

{chr(10).join(table)}

`legacy_bounded_residual`使用Phase 26同口径55列；`enhanced_bounded_residual`使用113列，
加入TP批处理敏感画像；`enhanced_direct`是控制组。最终设计仍是H0结构先验加残差DNN，H0
只作为基线与受保护回退，不代表取消DNN。

开发验证集冻结的候选为：

{decisions}

第一、第二独立确认集各972条feature均未包含target。本阶段已同时写出各四种方法的3,888条
预测，两个确认target文件都不是训练脚本参数。下一阶段只能按hash连接预先冻结的预测与真值，
不能重新训练、调参或改写这些预测。因此本阶段只能给出开发验证结论，不能替代独立泛化结论。
"""


def main() -> None:
    args = parse_args()
    for name in ("checkpoints", "analysis", "figures", "logs"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    dataset_summary = json.loads(args.dataset_summary.read_text())
    feature_contract = json.loads(args.feature_columns.read_text())
    if dataset_summary["status"] != "PASS":
        raise ValueError("Phase 29B dataset is not PASS")
    development = load_rows(args.development_dataset)
    first_confirmation = load_rows(args.first_confirmation_features)
    second_confirmation = load_rows(args.second_confirmation_features)
    if (len(development), len(first_confirmation), len(second_confirmation)) != (
        2268,
        972,
        972,
    ):
        raise ValueError("unexpected Phase 29C input row counts")
    if any(
        name.startswith("target_")
        for rows in (first_confirmation, second_confirmation)
        for name in rows[0]
    ):
        raise ValueError("confirmation feature artifact contains target columns")

    legacy_features = feature_contract["legacy_feature_columns"]
    enhanced_features = feature_contract["enhanced_feature_columns"]
    if len(legacy_features) != 55 or len(enhanced_features) != 113:
        raise ValueError("feature contract count mismatch")
    for name in [*legacy_features, *enhanced_features]:
        if name not in development[0]:
            raise ValueError(f"missing feature {name}")
    roles = {
        row["profile_id"]: row["phase27_role"] for row in development
    }
    role_counts = Counter(roles.values())
    if role_counts != Counter({FIT_ROLE: 30, VALIDATION_ROLE: 12}):
        raise ValueError(role_counts)

    device = choose_device(args.device)
    arrays = prepare_development(development)
    checkpoints = {}
    checkpoint_inventory = []
    histories = []
    for method_index, method in enumerate(LEARNED_METHODS):
        names = legacy_features if method.startswith("legacy_") else enhanced_features
        checkpoint, history = fit_model(
            method=method,
            rows=development,
            arrays=arrays,
            feature_names=names,
            args=args,
            device=device,
            seed=args.seed + method_index,
        )
        checkpoint["schema_version"] = "phase29c-tp-aligned-predictor-checkpoint-v1"
        checkpoint["bin_schema_id"] = "tp_native_12bin_4k_512m_v1"
        checkpoint["parallelism"] = "tp"
        checkpoint["forbidden_roles"] = [
            FIRST_CONFIRMATION_ROLE,
            SECOND_CONFIRMATION_ROLE,
        ]
        path = args.output_dir / "checkpoints" / f"tp_{method}.pt"
        torch.save(checkpoint, path)
        checkpoints[method] = checkpoint
        histories.extend(history)
        checkpoint_inventory.append(
            {
                "method": method,
                "path": str(path.relative_to(args.output_dir)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "feature_columns": len(names),
                "best_epoch": checkpoint["best_epoch"],
                "best_validation_loss": checkpoint["best_validation_loss"],
            }
        )

    predictions = {"h0": (arrays["h0_calls"], arrays["h0_bytes"])}
    for method, checkpoint in checkpoints.items():
        predictions[method] = predict(development, checkpoint, arrays["h0_encoded"], device)
    validation = validation_records(development, arrays, predictions)
    metrics = aggregate_records(validation)
    headline = {
        method: next(
            row
            for row in metrics
            if row["method"] == method
            and row["phase"] == "total"
            and row["model"] == "all"
            and row["parallel_size"] == "all"
            and row["policy"] == "all"
            and row["segment"] == "all"
        )
        for method in METHODS
    }
    decisions = select_candidates(metrics)
    first_predictions = predict_feature_artifact(first_confirmation, checkpoints, device)
    second_predictions = predict_feature_artifact(second_confirmation, checkpoints, device)

    write_csv_gz(args.output_dir / "analysis/training_history.csv.gz", histories)
    write_csv_gz(args.output_dir / "analysis/validation_case_metrics.csv.gz", validation)
    write_csv(args.output_dir / "analysis/validation_metrics.csv", metrics)
    write_csv(args.output_dir / "analysis/checkpoint_inventory.csv", checkpoint_inventory)
    write_csv(args.output_dir / "analysis/candidate_decisions.csv", decisions)
    write_csv_gz(
        args.output_dir / "analysis/first_confirmation_predictions.csv.gz",
        first_predictions,
    )
    write_csv_gz(
        args.output_dir / "analysis/second_confirmation_predictions.csv.gz",
        second_predictions,
    )
    plot_validation(args.output_dir / "figures/validation_tp_comparison.png", headline)

    checks = {
        "phase29b_status_pass": dataset_summary["status"] == "PASS",
        "development_rows_2268": len(development) == 2268,
        "fit_validation_profiles_30_12": role_counts
        == Counter({FIT_ROLE: 30, VALIDATION_ROLE: 12}),
        "first_confirmation_features_972_no_targets": len(first_confirmation) == 972
        and not any(name.startswith("target_") for name in first_confirmation[0]),
        "second_confirmation_features_972_no_targets": len(second_confirmation) == 972
        and not any(name.startswith("target_") for name in second_confirmation[0]),
        "features_55_legacy_113_enhanced": len(legacy_features) == 55
        and len(enhanced_features) == 113,
        "three_frozen_checkpoints": len(checkpoint_inventory) == 3,
        "validation_records_3888": len(validation) == 3888,
        "first_confirmation_predictions_3888": len(first_predictions) == 3888,
        "second_confirmation_predictions_3888": len(second_predictions) == 3888,
        "candidate_mapping_three_policies": len(decisions) == 3,
        "all_metrics_finite": all(
            math.isfinite(float(row[field]))
            for row in metrics
            for field in (
                "calls_mape",
                "calls_wape",
                "bytes_mape",
                "bytes_wape",
                "mean_histogram_tv",
                "mean_normalized_log_payload_emd",
                "common_reference_cost_mape",
            )
        ),
        "confirmation_targets_not_script_inputs": not any(
            "target" in name for name in vars(args)
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    if status != "PASS":
        raise RuntimeError(checks)

    summary = {
        "schema_version": "phase29c-tp-aligned-training-v1",
        "status": status,
        "objective": "retrain TP H0 plus residual DNN on aligned Hfull labels and freeze two confirmation prediction sets before target access",
        "device": str(device),
        "counts": {
            "development_phase_rows": len(development),
            "fit_phase_rows": sum(
                row["phase27_role"] == FIT_ROLE for row in development
            ),
            "validation_phase_rows": sum(
                row["phase27_role"] == VALIDATION_ROLE for row in development
            ),
            "fit_profiles": 30,
            "validation_profiles": 12,
            "first_confirmation_feature_rows": len(first_confirmation),
            "second_confirmation_feature_rows": len(second_confirmation),
            "legacy_feature_columns": len(legacy_features),
            "enhanced_feature_columns": len(enhanced_features),
            "checkpoints": len(checkpoint_inventory),
            "validation_case_metric_rows": len(validation),
            "first_confirmation_prediction_rows": len(first_predictions),
            "second_confirmation_prediction_rows": len(second_predictions),
        },
        "inputs": {
            "development_dataset_sha256": sha256(args.development_dataset),
            "first_confirmation_features_sha256": sha256(
                args.first_confirmation_features
            ),
            "second_confirmation_features_sha256": sha256(
                args.second_confirmation_features
            ),
            "dataset_summary_sha256": sha256(args.dataset_summary),
            "feature_columns_sha256": sha256(args.feature_columns),
        },
        "split_contract": {
            "fit": FIT_ROLE,
            "early_stopping": VALIDATION_ROLE,
            "first_confirmation_features_only": FIRST_CONFIRMATION_ROLE,
            "second_confirmation_features_only": SECOND_CONFIRMATION_ROLE,
            "confirmation_targets_read": False,
        },
        "validation_headline": headline,
        "candidate_decisions": decisions,
        "checkpoints": checkpoint_inventory,
        "checks": checks,
        "can_conclude": [
            "three DNN controls were retrained on aligned multi-model TP Hfull supervision",
            "both confirmation prediction artifacts were frozen without target access",
            "development validation can select guarded candidates for independent confirmation",
        ],
        "cannot_conclude": [
            "development validation improvements imply independent-window improvements",
            "the final residual DNN is accepted before both confirmation evaluations",
            "the common reference cost is measured physical topology time",
        ],
        "next_step": "archive this run, then evaluate first confirmation by joining only frozen predictions and isolated targets",
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(
        args.output_dir / "feature_contract.json",
        {
            "schema_version": "phase29c-tp-training-feature-contract-v1",
            "legacy_feature_columns": legacy_features,
            "enhanced_feature_columns": enhanced_features,
            "methods": list(METHODS),
            "primary_architecture": "compact32_H0_plus_enhanced_bounded_residual_DNN",
            "h0_role": "structural_prior_baseline_and_guarded_fallback",
            "candidate_rule": decisions[0]["rule"],
            "target_encoding": "log1p total plus log smoothed 12-bin shares for calls and logical bytes",
            "bounded_residual": {
                "total_log_bound": math.log(2.0),
                "share_logit_bound": 2.0,
                "network_output": "tanh",
            },
        },
    )
    write_json(
        args.output_dir / "audit_summary.json",
        {
            "schema_version": "phase29c-tp-aligned-training-audit-v1",
            "status": status,
            "checks": checks,
            "checkpoint_sha256": {
                row["method"]: row["sha256"] for row in checkpoint_inventory
            },
            "first_confirmation_predictions_sha256": sha256(
                args.output_dir / "analysis/first_confirmation_predictions.csv.gz"
            ),
            "second_confirmation_predictions_sha256": sha256(
                args.output_dir / "analysis/second_confirmation_predictions.csv.gz"
            ),
        },
    )
    (args.output_dir / "README.md").write_text(readme(summary))
    (args.output_dir / "DONE").write_text("PASS\n")
    try:
        repository_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        repository_head = "unknown"
    write_json(
        args.output_dir / "logs/training.log",
        {
            "schema_version": "phase29c-training-log-v1",
            "status": status,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository_head_at_training": repository_head,
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "CPU",
            "args": {
                **vars(args),
                "development_dataset": str(args.development_dataset),
                "first_confirmation_features": str(
                    args.first_confirmation_features
                ),
                "second_confirmation_features": str(
                    args.second_confirmation_features
                ),
                "dataset_summary": str(args.dataset_summary),
                "feature_columns": str(args.feature_columns),
                "output_dir": str(args.output_dir),
            },
            "training_runs": checkpoint_inventory,
            "confirmation_targets_read": False,
        },
    )
    files = sorted(
        path
        for path in args.output_dir.rglob("*")
        if path.is_file() and path.name != "manifest.sha256"
    )
    (args.output_dir / "manifest.sha256").write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(args.output_dir)}\n" for path in files
        )
    )
    print(
        json.dumps(
            {
                "status": status,
                "device": str(device),
                "candidate_decisions": {
                    row["policy"]: row["selected_method"] for row in decisions
                },
                "validation_headline": {
                    method: {
                        "calls_mape": headline[method]["calls_mape"],
                        "histogram_tv": headline[method]["mean_histogram_tv"],
                        "cost_mape": headline[method]["common_reference_cost_mape"],
                    }
                    for method in METHODS
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
