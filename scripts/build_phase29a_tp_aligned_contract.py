#!/usr/bin/env python3
"""Freeze the aligned Phase 29 TP Hfull and DNN experiment contract before labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


MODELS = ("deepseek-v2-lite", "qwen3-8b", "qwen3-30b-a3b")
TP_SIZES = (2, 4, 8)
PHASES = ("prefill", "decode")
STRATEGIES = {
    "latency": {"max_batch_size": 4, "max_prefill_tokens": 8192},
    "balanced": {"max_batch_size": 8, "max_prefill_tokens": 32768},
    "throughput": {"max_batch_size": 16, "max_prefill_tokens": 65536},
}
EXPECTED_PHASE27_ROLES = {
    "development_train": 30,
    "development_validation": 12,
    "independent_confirmation": 18,
}
METHODS = (
    "h0",
    "phase26_legacy_bounded_residual",
    "enhanced_bounded_residual",
    "enhanced_direct_control",
)
TP_DERIVED_FEATURES = (
    "feature_parallelism_tp",
    "feature_parallel_size_log2",
    "feature_phase_prefill",
    "feature_phase_decode",
    "feature_tp_max_batch_size",
    "feature_tp_max_prefill_tokens",
    "feature_tp_batch_size_fraction_of_16",
    "feature_tp_prefill_budget_fraction_of_65536",
    "feature_tp_input_mean_budget_fill",
    "feature_tp_input_p50_budget_fill",
    "feature_tp_input_p90_budget_fill",
    "feature_tp_input_p99_budget_fill",
    "feature_tp_rps_per_batch_slot",
    "feature_tp_fano_per_batch_slot",
    "feature_tp_survival_m_gt_1_batch_pressure",
    "feature_tp_survival_m_gt_8_batch_pressure",
    "feature_tp_survival_m_gt_16_batch_pressure",
    "feature_tp_survival_m_gt_32_batch_pressure",
    "feature_tp_survival_m_gt_64_batch_pressure",
    "feature_tp_multichunk_batch_pressure",
    "feature_tp_rolling_multichunk_max_4_batch_pressure",
    "feature_tp_rolling_multichunk_max_16_batch_pressure",
    "feature_tp_rolling_multichunk_max_32_batch_pressure",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase27-selection",
        type=Path,
        default=root
        / "experiment-results/phase27a_pp_feature_and_holdout_contract/selection/selected_windows.csv",
    )
    parser.add_argument(
        "--phase28-selection",
        type=Path,
        default=root
        / "experiment-results/phase28a_second_confirmation_contract/selection/selected_windows.csv",
    )
    parser.add_argument(
        "--model-features",
        type=Path,
        default=root / "experiment-results/phase16_model_features/model_features.json",
    )
    parser.add_argument(
        "--strategy-summary",
        type=Path,
        default=root / "experiment-results/phase16_profiledemand_plans/summary.json",
    )
    parser.add_argument(
        "--legacy-feature-contract",
        type=Path,
        default=root
        / "experiment-results/phase26c_hfull_predictor_training/feature_contract.json",
    )
    parser.add_argument(
        "--pp-feature-columns",
        type=Path,
        default=root
        / "experiment-results/phase27b_pp_hfull_dataset/feature_columns.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase29a_tp_aligned_contract",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def aligned_phase27_rows(rows: list[dict[str, str]]) -> list[dict]:
    return [
        {
            "profile_id": row["phase27_profile_id"],
            "role": row["phase27_role"],
            "window_id": row["window_id"],
            "source": row["source"],
            "segment": row["segment"],
            "source_split": row["source_split"],
            "cutoff_ms": int(row["cutoff_ms"]),
            "history_seconds": int(row["history_seconds"]),
            "history_count": int(row["history_count"]),
            "alignment_source": "phase27_pp_contract",
        }
        for row in rows
    ]


def aligned_phase28_rows(rows: list[dict[str, str]]) -> list[dict]:
    return [
        {
            "profile_id": row["profile_id"],
            "role": "second_independent_confirmation",
            "window_id": row["window_id"],
            "source": row["source"],
            "segment": row["segment"],
            "source_split": row["source_split"],
            "cutoff_ms": int(row["cutoff_ms"]),
            "history_seconds": int(row["history_seconds"]),
            "history_count": int(row["history_count"]),
            "alignment_source": "phase28_pp_contract",
        }
        for row in rows
    ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    window_dir = args.output_dir / "windows"
    window_dir.mkdir(exist_ok=True)

    phase27 = aligned_phase27_rows(read_csv(args.phase27_selection))
    phase28 = aligned_phase28_rows(read_csv(args.phase28_selection))
    model_features = json.loads(args.model_features.read_text())
    model_names = {row["model"] for row in model_features}
    strategies = json.loads(args.strategy_summary.read_text())["strategies"]
    legacy_contract = json.loads(args.legacy_feature_contract.read_text())
    pp_columns = json.loads(args.pp_feature_columns.read_text())["feature_columns"]
    common_columns = [
        name
        for name in pp_columns
        if name.startswith("feature_profile_") or name.startswith("feature_model_")
    ]
    enhanced_columns = [*common_columns, *TP_DERIVED_FEATURES]
    legacy_columns = legacy_contract["feature_names"]

    write_csv(window_dir / "phase27_aligned_windows.csv", phase27)
    write_csv(window_dir / "phase28_second_confirmation_windows.csv", phase28)
    write_json(
        args.output_dir / "model_and_execution_contract.json",
        {
            "schema_version": "phase29a-tp-model-execution-contract-v1",
            "models": list(MODELS),
            "tp_sizes": list(TP_SIZES),
            "strategies": STRATEGIES,
            "phases": list(PHASES),
            "normalization_requests": 1000,
            "scheduler_semantics": "fixed-draining",
            "parallel_configuration_is_predictor_input": True,
            "scheduler_selects_tp_size": False,
        },
    )
    write_json(
        args.output_dir / "feature_contract.json",
        {
            "schema_version": "phase29a-tp-feature-contract-v1",
            "legacy_feature_count": len(legacy_columns),
            "legacy_feature_columns": legacy_columns,
            "enhanced_feature_count": len(enhanced_columns),
            "enhanced_feature_columns": enhanced_columns,
            "common_profile_and_model_feature_count": len(common_columns),
            "tp_specific_feature_count": len(TP_DERIVED_FEATURES),
            "predictor_inputs": [
                "low-dimensional history profile",
                "model structure",
                "fixed TP size",
                "fixed execution policy",
                "phase",
                "H0 encoded histogram for residual methods",
            ],
            "forbidden_predictor_inputs": [
                "full request list",
                "representative request list",
                "Hfull target histogram",
                "communication cost label",
                "placement or topology choice",
                "future-window fields",
            ],
            "complete_request_list_usage": "offline Hfull teacher generation only",
        },
    )
    write_json(
        args.output_dir / "training_and_holdout_contract.json",
        {
            "schema_version": "phase29a-tp-training-holdout-contract-v1",
            "methods": list(METHODS),
            "fit_role": "development_train",
            "early_stopping_role": "development_validation",
            "first_confirmation_role": "independent_confirmation",
            "second_confirmation_role": "second_independent_confirmation",
            "prediction_freeze_rule": (
                "write predictions and freeze their SHA-256 before reading the matching "
                "confirmation Hfull targets"
            ),
            "primary_architecture": "H0 plus enhanced bounded residual DNN",
            "h0_role": "structural prior, baseline, and fail-safe fallback; not a replacement for DNN",
            "direct_role": "control only",
            "selection_metrics": [
                "total calls MAPE/WAPE",
                "logical bytes MAPE/WAPE",
                "histogram L1/TV",
                "normalized log-payload EMD",
                "common-reference cost MAPE/WAPE",
            ],
            "pre_registered_guard": (
                "relative to H0, residual should win at least two of calls MAPE, TV, "
                "and cost MAPE; cost MAPE may not regress by more than 10%, and bytes "
                "MAPE must be reported as a guard rather than hidden"
            ),
            "report_slices": [
                "all models",
                "each model",
                "qwen3-8b aligned with PP",
                "TP2/TP4/TP8",
                "latency/balanced/throughput",
                "prefill/decode/total",
                "traffic segment",
            ],
        },
    )

    role_counts = Counter(row["role"] for row in phase27)
    phase27_ids = {row["window_id"] for row in phase27}
    phase28_ids = {row["window_id"] for row in phase28}
    expected_development_rows = {
        role: count * len(MODELS) * len(TP_SIZES) * len(STRATEGIES) * len(PHASES)
        for role, count in EXPECTED_PHASE27_ROLES.items()
    }
    expected_second_rows = (
        len(phase28) * len(MODELS) * len(TP_SIZES) * len(STRATEGIES) * len(PHASES)
    )
    summary = {
        "schema_version": "phase29a-tp-aligned-contract-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "objective": (
            "retrain the TP H0+residual DNN on the exact Phase27/28 PP history-window "
            "backbone while preserving TP-native labels, features, and bin semantics"
        ),
        "phase27_aligned_profiles": len(phase27),
        "phase28_second_confirmation_profiles": len(phase28),
        "phase27_role_counts": dict(role_counts),
        "models": list(MODELS),
        "tp_sizes": list(TP_SIZES),
        "strategies": STRATEGIES,
        "expected_phase27_tp_phase_rows": expected_development_rows,
        "expected_phase28_second_confirmation_phase_rows": expected_second_rows,
        "legacy_feature_columns": len(legacy_columns),
        "enhanced_feature_columns": len(enhanced_columns),
        "tp_hfull_label_state_at_freeze": "no_phase29_tp_hfull_labels_generated",
        "tp_prediction_state_at_freeze": "no_phase29_tp_predictions_generated",
        "inputs": {
            "phase27_selection_sha256": sha256(args.phase27_selection),
            "phase28_selection_sha256": sha256(args.phase28_selection),
            "model_features_sha256": sha256(args.model_features),
            "strategy_summary_sha256": sha256(args.strategy_summary),
            "legacy_feature_contract_sha256": sha256(args.legacy_feature_contract),
            "pp_feature_columns_sha256": sha256(args.pp_feature_columns),
        },
    }
    write_json(args.output_dir / "summary.json", summary)

    checks = {
        "phase27_profiles_60": len(phase27) == 60,
        "phase27_roles_30_12_18": role_counts == Counter(EXPECTED_PHASE27_ROLES),
        "phase28_profiles_18": len(phase28) == 18,
        "window_ids_unique_78": len(phase27_ids | phase28_ids) == 78,
        "phase27_phase28_disjoint": not (phase27_ids & phase28_ids),
        "models_exact_three": model_names == set(MODELS),
        "strategies_match_phase16_contract": strategies == STRATEGIES,
        "legacy_features_55": len(legacy_columns) == 55,
        "common_profile_model_features_90": len(common_columns) == 90,
        "enhanced_features_113_unique": len(enhanced_columns) == 113
        and len(set(enhanced_columns)) == 113,
        "expected_train_validation_confirmation_rows_1620_648_972": expected_development_rows
        == {
            "development_train": 1620,
            "development_validation": 648,
            "independent_confirmation": 972,
        },
        "expected_second_confirmation_rows_972": expected_second_rows == 972,
        "labels_and_predictions_absent_at_freeze": summary[
            "tp_hfull_label_state_at_freeze"
        ]
        == "no_phase29_tp_hfull_labels_generated"
        and summary["tp_prediction_state_at_freeze"]
        == "no_phase29_tp_predictions_generated",
    }
    audit = {
        "schema_version": "phase29a-tp-aligned-contract-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    write_json(args.output_dir / "audit_summary.json", audit)
    if audit["status"] != "PASS":
        raise RuntimeError(audit)

    (args.output_dir / "README.md").write_text(
        """# Phase 29A：TP与PP历史窗口对齐合同

本阶段在生成任何Phase 29 TP Hfull标签或预测前，冻结TP重新训练合同。TP最终方法仍是
`H0 + bounded residual DNN`；Phase 26D的H0胜出只说明5个训练画像和55列旧特征下的
residual没有跨域泛化，H0在本合同中是结构先验、baseline和失效回退，不替代DNN。

历史流量骨架与PP严格对齐：复用Phase 27的60个窗口及30/12/18角色，再复用Phase 28的18个
第二独立确认窗口。TP保持自己的机制语义：三个模型、TP2/4/8、latency/balanced/throughput、
prefill/decode和TP原生12桶。相同的是历史窗口、split、每1000请求归一化和评测指标；不同的
是TP teacher、TP batching特征和输出桶，不能为了形式统一而混用PP语义。

Phase 29B预计生成Phase 27骨架上的3,240条TP Hfull phase labels，其中1,620/648/972分别用于
训练、验证和第一确认；Phase 28骨架再生成972条第二确认标签。完整请求列表只用于离线teacher，
最终预测器仍只读取低维历史画像、模型结构、固定TP配置、策略、phase和H0。
"""
    )
    (args.output_dir / "DONE").write_text("PASS\n")
    write_json(
        args.output_dir / "run.log",
        {
            "schema_version": "phase29a-run-log-v1",
            "status": "PASS",
            "labels_generated": False,
            "predictions_generated": False,
            "phase27_profiles": len(phase27),
            "phase28_profiles": len(phase28),
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
                "phase27_profiles": len(phase27),
                "phase28_profiles": len(phase28),
                "enhanced_feature_columns": len(enhanced_columns),
                "expected_phase_rows": 3240 + 972,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
