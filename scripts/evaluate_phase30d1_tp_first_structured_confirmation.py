#!/usr/bin/env python3
"""Evaluate frozen Phase30C predictions on isolated first-confirmation targets."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from build_phase25_full_window_teacher import PHASES, TP_BIN_EDGES
from build_phase29b_tp_hfull_dataset import MODELS, TP_SIZES, all_model_features
from build_phase30b_tp_structured_event_dataset import event_names, reconstruct_message_vectors
from train_phase27c_pp_scheduler_feature_predictors import case_record, sha256
from train_phase30c_tp_structured_event_predictors import METHODS, POLICIES


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    phase30b = root / "experiment-results/phase30b_tp_structured_event_dataset"
    phase30c = root / "experiment-results/phase30c_tp_structured_event_training"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frozen-predictions",
        type=Path,
        default=phase30c / "analysis/first_confirmation_predictions.csv.gz",
    )
    parser.add_argument(
        "--training-audit", type=Path, default=phase30c / "audit_summary.json"
    )
    parser.add_argument(
        "--candidate-decisions",
        type=Path,
        default=phase30c / "analysis/candidate_decisions.csv",
    )
    parser.add_argument(
        "--first-targets",
        type=Path,
        default=phase30b / "labels/first_confirmation_event_targets.csv.gz",
    )
    parser.add_argument(
        "--event-contract",
        type=Path,
        default=root
        / "experiment-results/phase30a_tp_structured_event_contract/event_contract.json",
    )
    parser.add_argument(
        "--model-features",
        type=Path,
        default=root / "experiment-results/phase16_model_features/model_features.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase30d1_tp_first_structured_confirmation",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as source:
        return list(csv.DictReader(source))


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


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def target_event_matrix(rows: list[dict[str, str]], names: list[str]) -> np.ndarray:
    return np.asarray(
        [[float(row[f"target_event_{name}"]) for name in names] for row in rows],
        dtype=np.float64,
    )


def event_dict(vector: np.ndarray, names: list[str]) -> dict[str, float]:
    return {name: float(value) for name, value in zip(names, vector)}


def actual_lookup(
    targets: list[dict[str, str]],
    target_events: np.ndarray,
    names: list[str],
    contract: dict,
    models: dict[str, tuple[dict, dict]],
) -> dict[tuple[str, str, str, str, str], tuple[np.ndarray, np.ndarray]]:
    output = {}
    for index, target in enumerate(targets):
        events = event_dict(target_events[index], names)
        for model_name in MODELS:
            vectors = reconstruct_message_vectors(events, models[model_name][0], contract)
            for tp_size in TP_SIZES:
                for phase in PHASES:
                    key = (
                        target["profile_id"],
                        model_name,
                        str(tp_size),
                        target["policy"],
                        phase,
                    )
                    if key in output:
                        raise ValueError(f"duplicate target expansion: {key}")
                    output[key] = vectors[phase]
    return output


def prediction_vectors(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(json.loads(row["predicted_calls_by_12bin_json"]), dtype=np.float64),
        np.asarray(
            json.loads(row["predicted_logical_bytes_by_12bin_json"]),
            dtype=np.float64,
        ),
    )


def confirmation_records(
    predictions: list[dict[str, str]],
    actual: dict[tuple[str, str, str, str, str], tuple[np.ndarray, np.ndarray]],
) -> list[dict]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        grouped[
            (
                row["method"],
                row["profile_id"],
                row["model"],
                row["parallel_size"],
                row["policy"],
            )
        ].append(row)
    records = []
    for (method, profile_id, model, parallel_size, policy), rows in grouped.items():
        if len(rows) != 2 or {row["phase"] for row in rows} != set(PHASES):
            raise ValueError("confirmation configuration lacks two phases")
        rows = sorted(rows, key=lambda row: row["phase"])
        actual_phase = []
        predicted_phase = []
        for row in rows:
            key = (profile_id, model, parallel_size, policy, row["phase"])
            actual_calls, actual_bytes = actual[key]
            predicted_calls, predicted_bytes = prediction_vectors(row)
            record = case_record(
                row,
                method,
                row["phase"],
                actual_calls,
                actual_bytes,
                predicted_calls,
                predicted_bytes,
                TP_BIN_EDGES.tolist(),
            )
            record["model"] = model
            record["role"] = row["role"]
            records.append(record)
            actual_phase.append((actual_calls, actual_bytes))
            predicted_phase.append((predicted_calls, predicted_bytes))
        representative = rows[0]
        actual_calls = sum((value[0] for value in actual_phase))
        actual_bytes = sum((value[1] for value in actual_phase))
        predicted_calls = sum((value[0] for value in predicted_phase))
        predicted_bytes = sum((value[1] for value in predicted_phase))
        total = case_record(
            representative,
            method,
            "total",
            actual_calls,
            actual_bytes,
            predicted_calls,
            predicted_bytes,
            TP_BIN_EDGES.tolist(),
        )
        actual_phase_aware = np.concatenate([value[0] for value in actual_phase])
        predicted_phase_aware = np.concatenate([value[0] for value in predicted_phase])
        actual_share = actual_phase_aware / max(float(actual_phase_aware.sum()), 1e-12)
        predicted_share = predicted_phase_aware / max(
            float(predicted_phase_aware.sum()), 1e-12
        )
        total["histogram_l1"] = float(np.abs(predicted_share - actual_share).sum())
        total["histogram_tv"] = total["histogram_l1"] / 2
        total["model"] = model
        total["role"] = representative["role"]
        records.append(total)
    return records


def aggregate_records(records: list[dict], mapping: dict[str, str] | None = None) -> list[dict]:
    selected = (
        [row for row in records if row["method"] == mapping[row["policy"]]]
        if mapping is not None
        else records
    )
    groups: dict[tuple[str, str, str, str, str, str], list[dict]] = defaultdict(list)
    for row in selected:
        method = "frozen_mapping" if mapping is not None else row["method"]
        for model in ("all", row["model"]):
            for parallel_size in ("all", row["parallel_size"]):
                for policy in ("all", row["policy"]):
                    for segment in ("all", row["segment"]):
                        groups[(method, row["phase"], model, parallel_size, policy, segment)].append(row)
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
                "calls_wape": sum(float(row["calls_absolute_error"]) for row in values) / actual_calls,
                "bytes_mape": float(np.mean([float(row["bytes_ape"]) for row in values])),
                "bytes_wape": sum(float(row["bytes_absolute_error"]) for row in values) / actual_bytes,
                "mean_histogram_l1": float(np.mean([float(row["histogram_l1"]) for row in values])),
                "mean_histogram_tv": float(np.mean([float(row["histogram_tv"]) for row in values])),
                "mean_normalized_log_payload_emd": float(
                    np.mean([float(row["normalized_log_payload_emd"]) for row in values])
                ),
                "common_reference_cost_mape": float(np.mean([float(row["cost_ape"]) for row in values])),
                "common_reference_cost_wape": sum(float(row["cost_absolute_error"]) for row in values) / actual_cost,
            }
        )
    return output


def headline(metrics: list[dict]) -> dict[str, dict]:
    return {
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


def plot_confirmation(path: Path, values: dict[str, dict]) -> None:
    import matplotlib.pyplot as plt

    labels = ("H0", "Phase29 residual", "Structured residual", "Direct control")
    colors = ("#4C78A8", "#A0A0A0", "#F58518", "#B8B8B8")
    specs = (
        ("calls_mape", "Total calls MAPE", 100.0, "%"),
        ("mean_histogram_tv", "Histogram TV", 1.0, ""),
        ("common_reference_cost_mape", "Common cost MAPE", 100.0, "%"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for axis, (metric, title, scale, suffix) in zip(axes, specs):
        numbers = [values[method][metric] * scale for method in METHODS]
        bars = axis.bar(labels, numbers, color=colors, width=0.68)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22, linewidth=0.8)
        axis.tick_params(axis="x", rotation=18)
        axis.spines[["top", "right"]].set_visible(False)
        upper = max(numbers) * 1.18 if max(numbers) > 0 else 1.0
        axis.set_ylim(0, upper)
        for bar, number in zip(bars, numbers):
            label = f"{number:.1f}{suffix}" if suffix else f"{number:.3f}"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + upper * 0.025,
                label,
                ha="center",
                fontsize=9,
            )
    figure.suptitle("Phase 30D1 first independent TP confirmation")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def readme(summary: dict) -> str:
    table = [
        "| 方法 | calls MAPE/WAPE | bytes MAPE/WAPE | TV | norm EMD | cost MAPE/WAPE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = summary["first_confirmation_headline"][method]
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
    mapping = "\n".join(
        f"- {policy}：`{method}`"
        for policy, method in summary["second_confirmation_mapping"].items()
    )
    return f"""# Phase 30D1：TP结构事件模型第一独立确认

状态：**{summary['status']}**。本阶段先核验Phase30C冻结预测的SHA与候选映射，再读取15个全新画像
的45个结构事件teacher target。真值经同一确定性适配器展开为三模型、TP2/4/8和两个phase；
所有四种方法只使用Phase30C已经冻结的3,240条预测，没有重训或改写。

## 第一确认结果

{chr(10).join(table)}

开发验证已将三个策略全部回退为H0，因此第一确认只核验这一冻结决策，不允许从诊断对照中重新
挑选赢家。第二确认映射继续冻结为：

{mapping}

本阶段可以判断冻结候选在第一批全新窗口上的误差；不能把一次确认解释为最终接受DNN，也不能
用第一确认结果调参。第二确认target仍未生成，下一阶段只能按上述映射和既有预测SHA继续。
"""


def main() -> None:
    args = parse_args()
    for name in ("analysis", "figures", "logs"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)

    # The frozen predictions, audit, and mapping are loaded and verified first.
    training_audit = json.loads(args.training_audit.read_text())
    predictions_sha = sha256(args.frozen_predictions)
    if predictions_sha != training_audit["first_confirmation_predictions_sha256"]:
        raise ValueError("frozen first prediction SHA mismatch")
    predictions = read_csv_gz(args.frozen_predictions)
    decisions = read_csv(args.candidate_decisions)
    development_mapping = {row["policy"]: row["selected_method"] for row in decisions}
    if development_mapping != {policy: "h0" for policy in POLICIES}:
        raise ValueError("unexpected frozen Phase30C mapping")
    if len(predictions) != 3240 or not all(
        row["prediction_frozen_before_first_confirmation_target_access"] == "True"
        for row in predictions
    ):
        raise ValueError("predictions are not frozen")

    # Target access begins only after all frozen-artifact checks above pass.
    targets = read_csv_gz(args.first_targets)
    contract = json.loads(args.event_contract.read_text())
    names = event_names(contract)
    models = all_model_features(args.model_features)
    target_events = target_event_matrix(targets, names)
    actual = actual_lookup(targets, target_events, names, contract, models)
    records = confirmation_records(predictions, actual)
    metrics = aggregate_records(records)
    values = headline(metrics)
    mapping_metrics = aggregate_records(records, development_mapping)
    second_mapping = development_mapping.copy()
    mapping_rows = [
        {
            "policy": policy,
            "selected_method": second_mapping[policy],
            "source": "phase30c_development_validation_frozen_before_first_target_access",
            "first_confirmation_used_for_retuning": False,
        }
        for policy in POLICIES
    ]

    write_csv_gz(args.output_dir / "analysis/first_case_metrics.csv.gz", records)
    write_csv(args.output_dir / "analysis/first_metrics.csv", metrics)
    write_csv(args.output_dir / "analysis/frozen_mapping_metrics.csv", mapping_metrics)
    write_csv(args.output_dir / "analysis/second_confirmation_mapping.csv", mapping_rows)
    plot_confirmation(args.output_dir / "figures/first_tp_comparison.png", values)

    checks = {
        "phase30c_status_pass": training_audit["status"] == "PASS",
        "frozen_prediction_sha_matches": predictions_sha
        == training_audit["first_confirmation_predictions_sha256"],
        "frozen_predictions_3240": len(predictions) == 3240,
        "prediction_freeze_flags_true": all(
            row["prediction_frozen_before_first_confirmation_target_access"] == "True"
            for row in predictions
        ),
        "first_targets_45": len(targets) == 45,
        "target_profiles_15": len({row["profile_id"] for row in targets}) == 15,
        "actual_expansions_810": len(actual) == 810,
        "case_metrics_4860": len(records) == 4860,
        "methods_four": {row["method"] for row in predictions} == set(METHODS),
        "development_mapping_all_h0": development_mapping
        == {policy: "h0" for policy in POLICIES},
        "second_mapping_frozen_all_h0": second_mapping
        == {policy: "h0" for policy in POLICIES},
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
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    if status != "PASS":
        raise RuntimeError(checks)
    frozen_headline = next(
        row
        for row in mapping_metrics
        if row["phase"] == "total"
        and row["model"] == "all"
        and row["parallel_size"] == "all"
        and row["policy"] == "all"
        and row["segment"] == "all"
    )
    summary = {
        "schema_version": "phase30d1-tp-first-structured-confirmation-v1",
        "status": status,
        "objective": "evaluate Phase30C frozen predictions on the isolated first new confirmation set without retraining",
        "counts": {
            "profiles": 15,
            "profile_policy_target_units": len(targets),
            "actual_phase_expansions": len(actual),
            "frozen_prediction_rows": len(predictions),
            "case_metric_rows": len(records),
        },
        "inputs": {
            "frozen_predictions_sha256": predictions_sha,
            "training_audit_sha256": sha256(args.training_audit),
            "candidate_decisions_sha256": sha256(args.candidate_decisions),
            "first_targets_sha256": sha256(args.first_targets),
            "event_contract_sha256": sha256(args.event_contract),
            "model_features_sha256": sha256(args.model_features),
        },
        "first_confirmation_headline": values,
        "frozen_mapping_headline": frozen_headline,
        "second_confirmation_mapping": second_mapping,
        "checks": checks,
        "can_conclude": [
            "the frozen Phase30C methods were evaluated on 15 unseen first-confirmation profiles",
            "the all-H0 development mapping was preserved without first-confirmation retuning",
        ],
        "cannot_conclude": [
            "first-confirmation diagnostics may be used to promote a noncandidate method",
            "the DNN route is finally accepted or rejected before second confirmation",
            "common reference cost is measured topology time",
        ],
        "next_step": "archive this run, then generate second targets from raw traces and evaluate the already-frozen second predictions",
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(
        args.output_dir / "audit_summary.json",
        {
            "schema_version": "phase30d1-tp-first-structured-confirmation-audit-v1",
            "status": status,
            "checks": checks,
            "first_prediction_sha256": predictions_sha,
            "second_mapping_sha256": sha256(
                args.output_dir / "analysis/second_confirmation_mapping.csv"
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
        args.output_dir / "logs/evaluation.log",
        {
            "schema_version": "phase30d1-evaluation-log-v1",
            "status": status,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository_head_at_evaluation": repository_head,
            "frozen_prediction_verified_before_target_access": True,
            "first_confirmation_used_for_retuning": False,
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
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
                "second_mapping": second_mapping,
                "first_confirmation_headline": {
                    method: {
                        "calls_mape": values[method]["calls_mape"],
                        "histogram_tv": values[method]["mean_histogram_tv"],
                        "cost_mape": values[method]["common_reference_cost_mape"],
                    }
                    for method in METHODS
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
