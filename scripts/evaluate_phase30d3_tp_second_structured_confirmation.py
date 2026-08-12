#!/usr/bin/env python3
"""Evaluate frozen Phase30C predictions on isolated second-confirmation targets."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from build_phase29b_tp_hfull_dataset import all_model_features
from build_phase30b_tp_structured_event_dataset import event_names
from evaluate_phase30d1_tp_first_structured_confirmation import (
    actual_lookup,
    aggregate_records,
    confirmation_records,
    headline,
    read_csv,
    read_csv_gz,
    target_event_matrix,
    write_csv,
    write_csv_gz,
    write_json,
)
from train_phase27c_pp_scheduler_feature_predictors import sha256
from train_phase30c_tp_structured_event_predictors import METHODS, POLICIES


METRIC_FIELDS = (
    "calls_mape",
    "calls_wape",
    "bytes_mape",
    "bytes_wape",
    "mean_histogram_tv",
    "mean_normalized_log_payload_emd",
    "common_reference_cost_mape",
    "common_reference_cost_wape",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    phase30c = root / "experiment-results/phase30c_tp_structured_event_training"
    phase30d1 = root / "experiment-results/phase30d1_tp_first_structured_confirmation"
    phase30d2 = root / "experiment-results/phase30d2_tp_second_structured_targets"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frozen-predictions",
        type=Path,
        default=phase30c / "analysis/second_confirmation_predictions.csv.gz",
    )
    parser.add_argument(
        "--training-audit", type=Path, default=phase30c / "audit_summary.json"
    )
    parser.add_argument(
        "--second-mapping",
        type=Path,
        default=phase30d1 / "analysis/second_confirmation_mapping.csv",
    )
    parser.add_argument(
        "--first-audit", type=Path, default=phase30d1 / "audit_summary.json"
    )
    parser.add_argument(
        "--first-summary", type=Path, default=phase30d1 / "summary.json"
    )
    parser.add_argument(
        "--second-targets",
        type=Path,
        default=phase30d2 / "labels/second_confirmation_event_targets.csv.gz",
    )
    parser.add_argument(
        "--second-target-audit",
        type=Path,
        default=phase30d2 / "audit_summary.json",
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
        default=root / "experiment-results/phase30d3_tp_second_structured_confirmation",
    )
    return parser.parse_args()


def comparison_rows(first: dict[str, dict], second: dict[str, dict]) -> list[dict]:
    rows = []
    for method in METHODS:
        for metric in METRIC_FIELDS:
            first_value = float(first[method][metric])
            second_value = float(second[method][metric])
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "first_confirmation": first_value,
                    "second_confirmation": second_value,
                    "second_minus_first": second_value - first_value,
                }
            )
    return rows


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
    figure.suptitle("Phase 30D3 second independent TP confirmation")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def readme(summary: dict) -> str:
    table = [
        "| 方法 | calls MAPE/WAPE | bytes MAPE/WAPE | TV | norm EMD | cost MAPE/WAPE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = summary["second_confirmation_headline"][method]
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
        f"- {policy}：`{method}`" for policy, method in summary["final_mapping"].items()
    )
    return f"""# Phase 30D3：TP结构事件模型第二独立确认与最终映射

状态：**{summary['status']}**。本阶段只把Phase30C预先冻结的第二确认预测连接到Phase30D2真值；
预测SHA、D1映射SHA和D2 target SHA均先核验通过。没有重训、调参或改写任何预测。

## 第二确认结果

{chr(10).join(table)}

当前Phase30最终受保护映射为：

{mapping}

这一结论表示当前91维画像→62维事件的有界残差DNN未通过开发验证和两级确认协议，部署候选应回退
到H0。它不表示研究设计取消DNN；相反，后续若继续TP DNN，应重新设计事件目标、loss或增加独立
训练窗口，并使用全新封闭确认集，不能再用Phase30两批确认结果调参。

共同参考cost仍只是同一连续代价曲线下的比较量，不是测得的真实placement/topology时延。
"""


def main() -> None:
    args = parse_args()
    for name in ("analysis", "figures", "logs"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)

    training_audit = json.loads(args.training_audit.read_text())
    first_audit = json.loads(args.first_audit.read_text())
    second_target_audit = json.loads(args.second_target_audit.read_text())
    prediction_sha = sha256(args.frozen_predictions)
    target_sha = sha256(args.second_targets)
    if prediction_sha != training_audit["second_confirmation_predictions_sha256"]:
        raise ValueError("frozen second prediction SHA mismatch")
    if target_sha != second_target_audit["second_targets_sha256"]:
        raise ValueError("second target SHA mismatch")
    mapping_rows = read_csv(args.second_mapping)
    mapping = {row["policy"]: row["selected_method"] for row in mapping_rows}
    if sha256(args.second_mapping) != first_audit["second_mapping_sha256"]:
        raise ValueError("second mapping SHA mismatch")
    expected_mapping = {policy: "h0" for policy in POLICIES}
    if mapping != expected_mapping:
        raise ValueError("unexpected frozen mapping")
    predictions = read_csv_gz(args.frozen_predictions)
    if len(predictions) != 3240 or not all(
        row["prediction_frozen_before_first_confirmation_target_access"] == "True"
        for row in predictions
    ):
        raise ValueError("second predictions are not frozen")

    targets = read_csv_gz(args.second_targets)
    contract = json.loads(args.event_contract.read_text())
    names = event_names(contract)
    models = all_model_features(args.model_features)
    target_events = target_event_matrix(targets, names)
    actual = actual_lookup(targets, target_events, names, contract, models)
    records = confirmation_records(predictions, actual)
    metrics = aggregate_records(records)
    values = headline(metrics)
    mapping_metrics = aggregate_records(records, mapping)
    first_summary = json.loads(args.first_summary.read_text())
    comparisons = comparison_rows(
        first_summary["first_confirmation_headline"], values
    )
    final_mapping_rows = [
        {
            "policy": policy,
            "selected_method": mapping[policy],
            "status": "FINAL_PHASE30_GUARDED_MAPPING",
            "source": "phase30c_development_mapping_preserved_through_two_confirmations",
        }
        for policy in POLICIES
    ]

    write_csv_gz(args.output_dir / "analysis/second_case_metrics.csv.gz", records)
    write_csv(args.output_dir / "analysis/second_metrics.csv", metrics)
    write_csv(args.output_dir / "analysis/frozen_mapping_metrics.csv", mapping_metrics)
    write_csv(
        args.output_dir / "analysis/cross_confirmation_comparison.csv", comparisons
    )
    write_csv(args.output_dir / "analysis/final_mapping.csv", final_mapping_rows)
    plot_confirmation(args.output_dir / "figures/second_tp_comparison.png", values)

    checks = {
        "phase30c_status_pass": training_audit["status"] == "PASS",
        "phase30d1_status_pass": first_audit["status"] == "PASS",
        "phase30d2_status_pass": second_target_audit["status"] == "PASS",
        "frozen_second_prediction_sha_matches": prediction_sha
        == training_audit["second_confirmation_predictions_sha256"],
        "second_target_sha_matches": target_sha
        == second_target_audit["second_targets_sha256"],
        "second_mapping_sha_matches": sha256(args.second_mapping)
        == first_audit["second_mapping_sha256"],
        "second_mapping_all_h0": mapping == expected_mapping,
        "frozen_predictions_3240": len(predictions) == 3240,
        "prediction_freeze_flags_true": all(
            row["prediction_frozen_before_first_confirmation_target_access"] == "True"
            for row in predictions
        ),
        "second_targets_45": len(targets) == 45,
        "target_profiles_15": len({row["profile_id"] for row in targets}) == 15,
        "actual_expansions_810": len(actual) == 810,
        "case_metrics_4860": len(records) == 4860,
        "methods_four": {row["method"] for row in predictions} == set(METHODS),
        "final_mapping_all_h0": mapping == expected_mapping,
        "all_metrics_finite": all(
            math.isfinite(float(row[field]))
            for row in metrics
            for field in METRIC_FIELDS
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
        "schema_version": "phase30d3-tp-second-structured-confirmation-v1",
        "status": status,
        "objective": "evaluate the already-frozen second predictions and finalize the guarded Phase30 TP mapping",
        "counts": {
            "profiles": 15,
            "profile_policy_target_units": len(targets),
            "actual_phase_expansions": len(actual),
            "frozen_prediction_rows": len(predictions),
            "case_metric_rows": len(records),
        },
        "inputs": {
            "frozen_predictions_sha256": prediction_sha,
            "training_audit_sha256": sha256(args.training_audit),
            "second_mapping_sha256": sha256(args.second_mapping),
            "first_audit_sha256": sha256(args.first_audit),
            "first_summary_sha256": sha256(args.first_summary),
            "second_targets_sha256": target_sha,
            "second_target_audit_sha256": sha256(args.second_target_audit),
            "event_contract_sha256": sha256(args.event_contract),
            "model_features_sha256": sha256(args.model_features),
        },
        "second_confirmation_headline": values,
        "frozen_mapping_headline": frozen_headline,
        "final_mapping": mapping,
        "checks": checks,
        "can_conclude": [
            "the Phase30 all-H0 guarded mapping completed two independent confirmations without retuning",
            "the current structured-event residual DNN is not an accepted deployment candidate",
            "the TP research architecture still includes a residual DNN route for future redesigned supervision",
        ],
        "cannot_conclude": [
            "TP intrinsically never benefits from DNN residual modeling",
            "Phase30 confirmation sets may be reused to tune a replacement model",
            "common reference cost is measured topology time",
        ],
        "next_step": "archive and synchronize Phase30D3, delete raw temporary traces, then update the Chinese guide and asset index",
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(
        args.output_dir / "audit_summary.json",
        {
            "schema_version": "phase30d3-tp-second-structured-confirmation-audit-v1",
            "status": status,
            "checks": checks,
            "frozen_second_prediction_sha256": prediction_sha,
            "second_target_sha256": target_sha,
            "final_mapping_sha256": sha256(
                args.output_dir / "analysis/final_mapping.csv"
            ),
        },
    )
    (args.output_dir / "README.md").write_text(readme(summary))
    (args.output_dir / "DONE").write_text("PASS\n")
    root = Path(__file__).resolve().parents[1]
    try:
        repository_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        repository_head = "unknown"
    write_json(
        args.output_dir / "logs/evaluation.log",
        {
            "schema_version": "phase30d3-evaluation-log-v1",
            "status": status,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository_head_at_evaluation": repository_head,
            "prediction_mapping_and_target_hashes_verified": True,
            "retrained_or_retuned": False,
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
                "final_mapping": mapping,
                "second_confirmation_headline": {
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
