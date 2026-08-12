#!/usr/bin/env python3
"""Evaluate the post-first frozen TP mapping on second confirmation targets."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from evaluate_phase29d1_tp_first_confirmation import (
    METHODS,
    POLICIES,
    add_total_records,
    aggregate_records,
    candidate_records,
    load_rows,
    phase_record,
    plot_confirmation,
    sha256,
    verify_manifest,
    write_csv,
    write_csv_gz,
    write_json,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    phase29c = root / "experiment-results/phase29c_tp_aligned_training"
    phase29d1 = root / "experiment-results/phase29d1_tp_first_confirmation"
    phase29d2 = root / "experiment-results/phase29d2_tp_second_confirmation_targets"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=phase29c / "analysis/second_confirmation_predictions.csv.gz",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=phase29d2 / "labels/second_confirmation_hfull_targets.csv.gz",
    )
    parser.add_argument(
        "--phase29c-audit", type=Path, default=phase29c / "audit_summary.json"
    )
    parser.add_argument(
        "--phase29d1-summary", type=Path, default=phase29d1 / "summary.json"
    )
    parser.add_argument(
        "--phase29d1-manifest", type=Path, default=phase29d1 / "manifest.sha256"
    )
    parser.add_argument(
        "--phase29d2-summary", type=Path, default=phase29d2 / "summary.json"
    )
    parser.add_argument(
        "--phase29d2-audit", type=Path, default=phase29d2 / "audit_summary.json"
    )
    parser.add_argument(
        "--phase29d2-manifest", type=Path, default=phase29d2 / "manifest.sha256"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase29d3_tp_second_confirmation",
    )
    return parser.parse_args()


def mapping_decisions(
    metrics: list[dict],
    mapping: dict[str, str],
    first_decisions: list[dict],
) -> list[dict]:
    first_lookup = {row["policy"]: row for row in first_decisions}
    output = []
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
        selected_method = mapping[policy]
        h0 = lookup["h0"]
        candidate = lookup[selected_method]
        if selected_method == "h0":
            wins = 0
            cost_guard = True
            second_confirmed = True
            outcome = "h0_fallback_retained"
        else:
            fields = (
                "calls_mape",
                "mean_histogram_tv",
                "common_reference_cost_mape",
            )
            wins = sum(float(candidate[field]) < float(h0[field]) for field in fields)
            cost_guard = float(candidate["common_reference_cost_mape"]) <= 1.10 * float(
                h0["common_reference_cost_mape"]
            )
            second_confirmed = wins >= 2 and cost_guard
            outcome = "residual_accepted" if second_confirmed else "fallback_to_h0"
        first_confirmed = bool(first_lookup[policy]["validation_candidate_confirmed"])
        final_method = selected_method if second_confirmed else "h0"
        output.append(
            {
                "policy": policy,
                "validation_frozen_candidate": first_lookup[policy][
                    "validation_frozen_candidate"
                ],
                "first_confirmation_confirmed": first_confirmed,
                "second_confirmation_frozen_method": selected_method,
                "second_confirmation_wins_of_calls_tv_cost": wins,
                "second_confirmation_cost_guard": cost_guard,
                "second_confirmation_confirmed": second_confirmed,
                "final_method": final_method,
                "final_outcome": outcome,
                "h0_calls_mape": h0["calls_mape"],
                "selected_calls_mape": candidate["calls_mape"],
                "h0_histogram_tv": h0["mean_histogram_tv"],
                "selected_histogram_tv": candidate["mean_histogram_tv"],
                "h0_cost_mape": h0["common_reference_cost_mape"],
                "selected_cost_mape": candidate["common_reference_cost_mape"],
                "mapping_selected_without_second_target_access": True,
                "second_confirmation_is_unbiased_for_frozen_mapping": True,
            }
        )
    return output


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
    mapping = summary["frozen_mapping_headline"]
    decision_lines = "\n".join(
        f"- {row['policy']}：第二确认冻结方法 `{row['second_confirmation_frozen_method']}`，"
        f"确认=`{row['second_confirmation_confirmed']}`，最终 `{row['final_method']}`。"
        for row in summary["final_policy_decisions"]
    )
    return f"""# Phase 29D3：TP第二独立确认与最终冻结结论

状态：**{summary['status']}**。本阶段没有训练、早停、调参或改写预测，只把Phase 29C已冻结
的3,888条第二确认预测，与Phase 29D2在预测/映射冻结后才生成的972条Hfull真值精确连接。

## 第二独立确认的四方法total结果

{chr(10).join(table)}

## 第一确认后冻结映射的第二确认结果

冻结映射整体：calls MAPE {mapping['calls_mape']:.2%} / WAPE {mapping['calls_wape']:.2%}，
bytes MAPE {mapping['bytes_mape']:.2%} / WAPE {mapping['bytes_wape']:.2%}，TV
{mapping['mean_histogram_tv']:.4f}，norm EMD {mapping['mean_normalized_log_payload_emd']:.4f}，
cost MAPE {mapping['common_reference_cost_mape']:.2%} / WAPE
{mapping['common_reference_cost_wape']:.2%}。

{decision_lines}

第二确认对上述映射是无偏的，因为映射、四方法预测及其hash都在第二真值生成前冻结。这里
可以决定每种固定TP策略首版采用残差DNN还是H0受保护回退；不能把同一第二确认结果继续用于
挑选新的特征、超参数或映射后再声称无偏。

最终架构仍是H0结构先验加有界残差DNN；若某策略未通过两轮确认，则该策略使用H0回退，
不等于删除DNN研究路线。cost仍是5 μs + 100 GB/s统一参考曲线，不是placement/topology
物理链路实测；结论仅覆盖fixed-draining和已冻结TP配置/策略。
"""


def main() -> None:
    args = parse_args()
    for name in ("analysis", "figures", "logs"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    phase29c = json.loads(args.phase29c_audit.read_text())
    phase29d1 = json.loads(args.phase29d1_summary.read_text())
    phase29d2 = json.loads(args.phase29d2_summary.read_text())
    phase29d2_audit = json.loads(args.phase29d2_audit.read_text())
    if any(
        value["status"] != "PASS"
        for value in (phase29c, phase29d1, phase29d2, phase29d2_audit)
    ):
        raise ValueError("upstream Phase 29C/D1/D2 is not PASS")
    d1_manifest_checks = verify_manifest(args.phase29d1_manifest)
    d2_manifest_checks = verify_manifest(args.phase29d2_manifest)
    frozen_prediction_hash = phase29d1["second_confirmation_predictions_sha256"]
    if sha256(args.predictions) != frozen_prediction_hash:
        raise RuntimeError("second-confirmation prediction hash mismatch")
    if phase29d2["frozen_prediction_sha256"] != frozen_prediction_hash:
        raise RuntimeError("Phase 29D2 prediction freeze mismatch")

    predictions = load_rows(args.predictions)
    targets = load_rows(args.targets)
    if len(predictions) != 3888 or len(targets) != 972:
        raise ValueError("unexpected second-confirmation row counts")
    target_lookup = {row["label_id"]: row for row in targets}
    if len(target_lookup) != len(targets):
        raise ValueError("duplicate targets")
    phase_records = []
    join_failures = []
    for prediction in predictions:
        target = target_lookup.get(prediction["training_id"])
        if target is None:
            join_failures.append(prediction["training_id"])
        else:
            phase_records.append(phase_record(prediction, target))
    records = add_total_records(phase_records)
    metrics = aggregate_records(records)
    frozen_mapping = phase29d1["second_confirmation_frozen_mapping"]
    frozen_records = [
        {**row, "method": "frozen_mapping"}
        for row in candidate_records(records, frozen_mapping)
    ]
    frozen_metrics = aggregate_records(frozen_records)
    final_decisions = mapping_decisions(
        metrics,
        frozen_mapping,
        phase29d1["post_first_confirmation_decisions"],
    )
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
    frozen_headline = next(
        row
        for row in frozen_metrics
        if row["phase"] == "total"
        and row["model"] == "all"
        and row["parallel_size"] == "all"
        and row["policy"] == "all"
        and row["segment"] == "all"
    )

    write_csv_gz(
        args.output_dir / "analysis/second_confirmation_predictions_and_errors.csv.gz",
        records,
    )
    write_csv(args.output_dir / "analysis/second_confirmation_metrics.csv", metrics)
    write_csv(
        args.output_dir / "analysis/frozen_mapping_metrics.csv", frozen_metrics
    )
    write_csv(args.output_dir / "analysis/final_policy_decisions.csv", final_decisions)
    plot_confirmation(
        args.output_dir / "figures/second_confirmation_comparison.png", headline
    )

    checks = {
        "upstream_all_pass": all(
            value["status"] == "PASS"
            for value in (phase29c, phase29d1, phase29d2, phase29d2_audit)
        ),
        "phase29d1_manifest_10_of_10": len(d1_manifest_checks) == 10
        and all(d1_manifest_checks.values()),
        "phase29d2_manifest_7_of_7": len(d2_manifest_checks) == 7
        and all(d2_manifest_checks.values()),
        "prediction_hash_matches_pre_target_freeze": sha256(args.predictions)
        == frozen_prediction_hash
        == phase29d2["frozen_prediction_sha256"],
        "predictions_3888_targets_972": len(predictions) == 3888
        and len(targets) == 972,
        "join_3888_of_3888": len(phase_records) == 3888 and not join_failures,
        "phase_plus_total_records_5832": len(records) == 5832,
        "profiles_18_models_3_tp_3_policies_3": len(
            {row["profile_id"] for row in phase_records}
        )
        == 18
        and len({row["model"] for row in phase_records}) == 3
        and {int(row["parallel_size"]) for row in phase_records} == {2, 4, 8}
        and {row["policy"] for row in phase_records} == set(POLICIES),
        "methods_four_balanced": Counter(row["method"] for row in phase_records)
        == Counter({method: 972 for method in METHODS}),
        "frozen_mapping_exactly_carried_from_phase29d1": frozen_mapping
        == phase29d2["second_confirmation_frozen_mapping"],
        "final_decisions_three_policies": len(final_decisions) == 3
        and {row["policy"] for row in final_decisions} == set(POLICIES),
        "all_metrics_finite": all(
            math.isfinite(float(row[field]))
            for row in [*metrics, *frozen_metrics]
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

    final_mapping = {row["policy"]: row["final_method"] for row in final_decisions}
    summary = {
        "schema_version": "phase29d3-tp-second-confirmation-v1",
        "status": status,
        "objective": "unbiased second confirmation of the TP mapping frozen after first confirmation",
        "counts": {
            "profiles": 18,
            "target_phase_rows": len(targets),
            "frozen_prediction_rows": len(predictions),
            "phase_error_records": len(phase_records),
            "phase_plus_total_records": len(records),
        },
        "inputs": {
            "predictions_sha256": sha256(args.predictions),
            "targets_sha256": sha256(args.targets),
            "phase29c_audit_sha256": sha256(args.phase29c_audit),
            "phase29d1_summary_sha256": sha256(args.phase29d1_summary),
            "phase29d1_manifest_sha256": sha256(args.phase29d1_manifest),
            "phase29d2_summary_sha256": sha256(args.phase29d2_summary),
            "phase29d2_audit_sha256": sha256(args.phase29d2_audit),
            "phase29d2_manifest_sha256": sha256(args.phase29d2_manifest),
        },
        "second_confirmation_headline": headline,
        "second_confirmation_frozen_mapping": frozen_mapping,
        "frozen_mapping_headline": frozen_headline,
        "final_policy_decisions": final_decisions,
        "final_policy_mapping": final_mapping,
        "second_confirmation_unbiased_for_frozen_mapping": True,
        "checks": checks,
        "can_conclude": [
            "whether the post-first TP mapping repeats on a distinct second window set",
            "which fixed TP strategies accept a bounded residual DNN versus guarded H0 fallback in the first release",
        ],
        "cannot_conclude": [
            "that a new mapping chosen after this second result is unbiased on the same windows",
            "physical placement/topology communication time from the common reference curve",
            "online arrival-aware behavior",
        ],
        "next_step": "archive and synchronize the final TP confirmation, clean temporary raw traces, then update the experiment guide and compare the aligned TP and PP evidence",
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(
        args.output_dir / "audit_summary.json",
        {
            "schema_version": "phase29d3-tp-second-confirmation-audit-v1",
            "status": status,
            "checks": checks,
            "prediction_hash_verified": frozen_prediction_hash,
            "join_failures": join_failures,
        },
    )
    (args.output_dir / "README.md").write_text(readme(summary))
    (args.output_dir / "DONE").write_text("PASS\n")
    write_json(
        args.output_dir / "logs/evaluation.log",
        {
            "schema_version": "phase29d3-evaluation-log-v1",
            "status": status,
            "training_performed": False,
            "predictions_rewritten": False,
            "mapping_changed_before_evaluation": False,
            "prediction_hash_verified": True,
            "phase29d1_manifest_checks": d1_manifest_checks,
            "phase29d2_manifest_checks": d2_manifest_checks,
            "joined_phase_rows": len(phase_records),
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
                "headline": {
                    method: {
                        "calls_mape": headline[method]["calls_mape"],
                        "histogram_tv": headline[method]["mean_histogram_tv"],
                        "cost_mape": headline[method]["common_reference_cost_mape"],
                    }
                    for method in METHODS
                },
                "frozen_mapping_headline": {
                    "calls_mape": frozen_headline["calls_mape"],
                    "histogram_tv": frozen_headline["mean_histogram_tv"],
                    "cost_mape": frozen_headline["common_reference_cost_mape"],
                },
                "final_policy_mapping": final_mapping,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
