#!/usr/bin/env python3
"""Freeze new windows and a structured TP scheduler-event prediction contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from build_phase25_full_window_teacher import TP_BIN_EDGES
from build_phase27a_pp_feature_and_holdout_contract import (
    HISTORY_ONLY_SOURCE_COLUMNS,
    SELECTION_FEATURES,
    choose_medoids,
    selection_vector,
)


SEED = "phase30-tp-structured-event-new-window-contract-20260812-v1"
SEGMENTS = (
    "burstgpt_1",
    "burstgpt_2",
    "burstgpt_3",
    "mooncake_conversation",
    "mooncake_toolagent",
)
EXHAUSTED_SEGMENT = "mooncake_synthetic"
SELECTED_PER_SEGMENT = 18
ROLE_QUOTAS = {
    "development_train": 9,
    "development_validation": 3,
    "independent_confirmation": 3,
    "second_independent_confirmation": 3,
}
MODELS = ("deepseek-v2-lite", "qwen3-8b", "qwen3-30b-a3b")
TP_SIZES = (2, 4, 8)
POLICIES = ("latency", "balanced", "throughput")
BYTES_PER_TOKEN_VALUES = (4096, 8192)
MAX_BATCH_SIZE = 16
MAX_PREFILL_TOKENS = 65536
METHODS = (
    "h0",
    "phase29_enhanced_bounded_residual_diagnostic",
    "structured_event_bounded_residual",
    "structured_event_direct_control",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--windows",
        type=Path,
        default=root / "experiment-results/phase15_trace_data/windows.csv.gz",
    )
    parser.add_argument(
        "--phase16-profiles",
        type=Path,
        default=root
        / "experiment-results/phase16_service_profiles/service_profiles.csv",
    )
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
        "--phase29-feature-contract",
        type=Path,
        default=root
        / "experiment-results/phase29a_tp_aligned_contract/feature_contract.json",
    )
    parser.add_argument(
        "--phase29d3-summary",
        type=Path,
        default=root
        / "experiment-results/phase29d3_tp_second_confirmation/summary.json",
    )
    parser.add_argument(
        "--model-features",
        type=Path,
        default=root / "experiment-results/phase16_model_features/model_features.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root
        / "experiment-results/phase30a_tp_structured_event_contract",
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


def role_order(window_id: str) -> str:
    return hashlib.sha256(f"{SEED}:{window_id}".encode()).hexdigest()


def output_bin(payload: int) -> int:
    return int(
        np.clip(np.searchsorted(TP_BIN_EDGES, payload, side="right") - 1, 0, 11)
    )


def prefill_joint_categories() -> list[dict]:
    categories = []
    previous = None
    start = 1
    for tokens in range(1, MAX_PREFILL_TOKENS + 1):
        pair = tuple(output_bin(tokens * value) for value in BYTES_PER_TOKEN_VALUES)
        if previous is None:
            previous = pair
            start = tokens
        elif pair != previous:
            categories.append(
                {
                    "category": len(categories),
                    "token_sum_min_inclusive": start,
                    "token_sum_max_inclusive": tokens - 1,
                    "tp_bin_for_4096_bytes_per_token": previous[0],
                    "tp_bin_for_8192_bytes_per_token": previous[1],
                }
            )
            previous = pair
            start = tokens
    categories.append(
        {
            "category": len(categories),
            "token_sum_min_inclusive": start,
            "token_sum_max_inclusive": MAX_PREFILL_TOKENS,
            "tp_bin_for_4096_bytes_per_token": previous[0],
            "tp_bin_for_8192_bytes_per_token": previous[1],
        }
    )
    return categories


def event_contract(categories: list[dict]) -> dict:
    count_names = [
        f"prefill_joint_category_{index}_batch_count_per_1000"
        for index in range(len(categories))
    ]
    mass_names = [
        f"prefill_joint_category_{index}_input_token_mass_per_1000"
        for index in range(len(categories))
    ]
    decode_names = [
        f"decode_active_lanes_{lanes}_step_count_per_1000"
        for lanes in range(1, MAX_BATCH_SIZE + 1)
    ]
    return {
        "schema_version": "phase30-tp-structured-scheduler-event-v1",
        "teacher_scheduler": "tp_fixed_order_token_budget_batches_v1",
        "normalization_requests": 1000,
        "event_target_count": len(count_names) + len(mass_names) + len(decode_names),
        "prefill_joint_category_count": len(categories),
        "prefill_joint_categories": categories,
        "target_groups": {
            "prefill_batch_counts": count_names,
            "prefill_input_token_mass": mass_names,
            "decode_active_lane_step_counts": decode_names,
        },
        "exact_reconstruction": {
            "prefill_calls": (
                "category batch count multiplied by model logical_collectives_per_forward; "
                "category maps losslessly to the TP output bin for both 4096 and 8192 bytes/token"
            ),
            "prefill_logical_bytes": (
                "category input-token mass multiplied by model bytes/token and "
                "logical_collectives_per_forward"
            ),
            "decode_calls": (
                "active-lane step count multiplied by model logical_collectives_per_forward"
            ),
            "decode_logical_bytes": (
                "active-lane step count multiplied by active lanes, model bytes/token, "
                "and logical_collectives_per_forward"
            ),
        },
        "model_adapter_inputs": [
            "logical_collectives_per_forward_prior",
            "payload_bytes_per_active_token_prior",
        ],
        "tp_size_semantics": (
            "TP size remains a predictor/deployment contract input and audit key, but the "
            "current GPU-validated topology-independent teacher formula is invariant to TP2/4/8"
        ),
        "event_predictor_inputs": [
            "low-dimensional history profile",
            "fixed TP batching policy",
            "compact32 H0 event vector for residual methods",
        ],
        "forbidden_inputs": [
            "full_request_list",
            "representative_request_list",
            "Hfull_event_target",
            "Hfull_histogram_target",
            "communication_cost_target",
            "placement_or_topology_choice",
            "future_window_columns",
        ],
    }


def modeling_contract(feature_contract: dict) -> dict:
    return {
        "schema_version": "phase30-tp-structured-event-modeling-contract-v1",
        "objective": (
            "predict normalized fixed-draining TP scheduler events, then deterministically "
            "reconstruct topology-independent message histograms with model structure"
        ),
        "methods": list(METHODS),
        "primary_architecture": "compact32_H0_events_plus_bounded_event_residual_DNN",
        "direct_role": "negative_control_only",
        "training_unit": "one independent history profile times one fixed batching policy",
        "do_not_inflate_training_unit_by": ["model", "TP size", "phase"],
        "feature_views": {
            "phase29_enhanced_profile_and_policy": feature_contract[
                "enhanced_feature_columns"
            ],
            "remove_model_tp_phase_columns_for_event_predictor": True,
            "model_structure_used_only_by_deterministic_adapter": True,
        },
        "development_data": {
            "reusable_phase29_profiles": {
                "development_train": 30,
                "development_validation": 12,
            },
            "new_phase30_profiles": {
                "development_train": 45,
                "development_validation": 15,
            },
            "combined_profiles": {
                "development_train": 75,
                "development_validation": 27,
            },
            "combined_profile_policy_training_units": 75 * len(POLICIES),
            "combined_profile_policy_validation_units": 27 * len(POLICIES),
        },
        "closed_for_tuning": [
            "all Phase29D1 first-confirmation profiles",
            "all Phase29D3 second-confirmation profiles",
        ],
        "new_confirmation_protocol": {
            "first_profiles": 15,
            "second_profiles": 15,
            "freeze_all_method_predictions_for_both_sets_before_reading_first targets": True,
            "generate_second_targets_only_after_first-confirmation mapping freeze": True,
        },
        "event_encoding": {
            "value_transform": "log1p for all 62 nonnegative event count/mass targets",
            "residual_prior": "compact32 H0 structured event vector",
            "residual_bound": "tanh times frozen per-target bound derived on development_train only",
        },
        "training_loss": {
            "components": [
                "normalized 62-target SmoothL1 event loss",
                "log total calls loss for prefill and decode",
                "log total logical-bytes loss for both model bytes/token classes",
                "12-bin normalized histogram TV surrogate",
                "log common-reference-cost loss",
            ],
            "component_weights": "freeze in Phase30B implementation before any confirmation target access",
            "selection_metrics": [
                "total calls MAPE",
                "histogram TV",
                "common reference cost MAPE",
            ],
        },
        "candidate_rule": (
            "per policy, residual must beat H0 on at least two of calls MAPE, TV, and cost "
            "MAPE on development_validation, with cost MAPE no greater than 110% of H0"
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_dir = args.output_dir / "selection"
    selection_dir.mkdir(exist_ok=True)
    (args.output_dir / "logs").mkdir(exist_ok=True)
    phase29_features = json.loads(args.phase29_feature_contract.read_text())
    phase29d3 = json.loads(args.phase29d3_summary.read_text())
    if phase29d3["status"] != "PASS":
        raise ValueError("Phase 29D3 is not PASS")
    if phase29d3["final_policy_mapping"] != {policy: "h0" for policy in POLICIES}:
        raise ValueError("unexpected Phase 29 final fallback mapping")

    windows = pd.read_csv(args.windows, usecols=list(HISTORY_ONLY_SOURCE_COLUMNS))
    phase16 = set(
        pd.read_csv(args.phase16_profiles, usecols=["window_id"])[
            "window_id"
        ].astype(str)
    )
    phase27 = {row["window_id"] for row in read_csv(args.phase27_selection)}
    phase28 = {row["window_id"] for row in read_csv(args.phase28_selection)}
    excluded = phase16 | phase27 | phase28
    selected_rows = []
    candidate_rows = []
    for segment in (*SEGMENTS, EXHAUSTED_SEGMENT):
        minimum = 32 if segment.startswith("burstgpt") else 128
        candidates = windows[
            (windows["segment"] == segment)
            & (windows["history_count"] >= minimum)
            & (~windows["window_id"].astype(str).isin(excluded))
        ].copy()
        candidates = candidates.sort_values("window_id", kind="stable").reset_index(
            drop=True
        )
        quota = 0 if segment == EXHAUSTED_SEGMENT else SELECTED_PER_SEGMENT
        candidate_rows.append(
            {
                "segment": segment,
                "eligible_unused_windows": len(candidates),
                "minimum_history_count": minimum,
                "selected_windows": quota,
                "selection_status": "exhausted_by_prior_frozen_experiments"
                if segment == EXHAUSTED_SEGMENT
                else "selected",
            }
        )
        if not quota:
            continue
        if len(candidates) < quota:
            raise RuntimeError(f"{segment}: only {len(candidates)} candidates")
        matrix = np.stack(
            [selection_vector(row) for _, row in candidates.iterrows()]
        )
        medoids, labels, distances = choose_medoids(matrix, quota)
        selected = []
        for cluster, index in enumerate(medoids):
            row = candidates.iloc[index]
            members = np.flatnonzero(labels == cluster)
            selected.append(
                {
                    "window_id": str(row["window_id"]),
                    "source": str(row["source"]),
                    "segment": segment,
                    "source_split": str(row["split"]),
                    "cutoff_ms": int(row["cutoff_ms"]),
                    "history_seconds": int(row["history_seconds"]),
                    "history_count": int(row["history_count"]),
                    "selection_cluster": cluster,
                    "selection_cluster_members": int(len(members)),
                    "selection_distance_to_medoid_mean": float(
                        np.mean(distances[members])
                    ),
                    "role_order_sha256": role_order(str(row["window_id"])),
                }
            )
        selected.sort(key=lambda row: row["role_order_sha256"])
        boundaries = np.cumsum(list(ROLE_QUOTAS.values()))
        roles = list(ROLE_QUOTAS)
        for index, row in enumerate(selected):
            role_index = int(np.searchsorted(boundaries, index, side="right"))
            row["role"] = roles[role_index]
            row["profile_id"] = f"phase30_{segment}_{index + 1:02d}"
            selected_rows.append(row)

    selected_rows.sort(key=lambda row: (row["segment"], row["profile_id"]))
    write_csv(selection_dir / "selected_windows.csv", selected_rows)
    write_csv(selection_dir / "candidate_counts.csv", candidate_rows)
    categories = prefill_joint_categories()
    events = event_contract(categories)
    modeling = modeling_contract(phase29_features)
    write_json(args.output_dir / "event_contract.json", events)
    write_json(args.output_dir / "modeling_contract.json", modeling)

    role_counts = Counter(row["role"] for row in selected_rows)
    segment_counts = Counter(row["segment"] for row in selected_rows)
    summary = {
        "schema_version": "phase30a-tp-structured-event-contract-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "objective": modeling["objective"],
        "selection_seed": SEED,
        "selection_rule": (
            "exclude all Phase16/27/28 windows, select 18 robust-scaled history-only "
            "medoids from each of five non-exhausted segments, and assign 9/3/3/3 roles "
            "by seeded SHA-256 order before any Phase30 target or prediction exists"
        ),
        "selected_profiles": len(selected_rows),
        "selected_segments": list(SEGMENTS),
        "exhausted_segments": [EXHAUSTED_SEGMENT],
        "segment_counts": dict(segment_counts),
        "role_counts": dict(role_counts),
        "selection_feature_count": len(SELECTION_FEATURES),
        "structured_event_targets": events["event_target_count"],
        "prefill_joint_categories": len(categories),
        "decode_active_lane_targets": MAX_BATCH_SIZE,
        "models": list(MODELS),
        "tp_sizes": list(TP_SIZES),
        "policies": list(POLICIES),
        "methods": list(METHODS),
        "excluded_windows": {
            "phase16": len(phase16),
            "phase27": len(phase27),
            "phase28": len(phase28),
            "union": len(excluded),
        },
        "target_state_at_freeze": "no_phase30_event_or_hfull_targets_generated",
        "prediction_state_at_freeze": "no_phase30_predictions_generated",
        "inputs": {
            "phase15_windows_sha256": sha256(args.windows),
            "phase16_profiles_sha256": sha256(args.phase16_profiles),
            "phase27_selection_sha256": sha256(args.phase27_selection),
            "phase28_selection_sha256": sha256(args.phase28_selection),
            "phase29_feature_contract_sha256": sha256(
                args.phase29_feature_contract
            ),
            "phase29d3_summary_sha256": sha256(args.phase29d3_summary),
            "model_features_sha256": sha256(args.model_features),
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    exhausted_count = next(
        row["eligible_unused_windows"]
        for row in candidate_rows
        if row["segment"] == EXHAUSTED_SEGMENT
    )
    checks = {
        "phase29d3_pass_and_final_h0_mapping": phase29d3["status"] == "PASS"
        and phase29d3["final_policy_mapping"]
        == {policy: "h0" for policy in POLICIES},
        "selected_profiles_90": len(selected_rows) == 90,
        "five_segments_18_each": segment_counts
        == Counter({segment: 18 for segment in SEGMENTS}),
        "roles_45_15_15_15": role_counts
        == Counter(
            {
                "development_train": 45,
                "development_validation": 15,
                "independent_confirmation": 15,
                "second_independent_confirmation": 15,
            }
        ),
        "all_window_ids_unique": len(
            {row["window_id"] for row in selected_rows}
        )
        == 90,
        "no_phase16_27_28_reuse": not (
            {row["window_id"] for row in selected_rows} & excluded
        ),
        "synthetic_exhausted_zero_eligible": exhausted_count == 0,
        "history_only_selection": all(
            not name.startswith("future_")
            for name in HISTORY_ONLY_SOURCE_COLUMNS
        ),
        "event_targets_62_categories_23_decode_16": events[
            "event_target_count"
        ]
        == 62
        and len(categories) == 23
        and MAX_BATCH_SIZE == 16,
        "joint_categories_cover_1_to_65536_without_gaps": categories[0][
            "token_sum_min_inclusive"
        ]
        == 1
        and categories[-1]["token_sum_max_inclusive"]
        == MAX_PREFILL_TOKENS
        and all(
            left["token_sum_max_inclusive"] + 1
            == right["token_sum_min_inclusive"]
            for left, right in zip(categories, categories[1:])
        ),
        "phase29_confirmations_closed_for_tuning": len(
            modeling["closed_for_tuning"]
        )
        == 2,
        "targets_and_predictions_absent_at_freeze": summary[
            "target_state_at_freeze"
        ]
        == "no_phase30_event_or_hfull_targets_generated"
        and summary["prediction_state_at_freeze"]
        == "no_phase30_predictions_generated",
    }
    audit = {
        "schema_version": "phase30a-tp-structured-event-contract-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    write_json(args.output_dir / "audit_summary.json", audit)
    if audit["status"] != "PASS":
        raise RuntimeError(audit)

    (args.output_dir / "README.md").write_text(
        f"""# Phase 30A：TP结构化batch事件与新窗口合同

状态：**PASS**。Phase 29第二确认否决了当前直方图residual checkpoint，但不取消DNN路线。
本阶段在任何Phase 30 target或预测生成前，冻结新的结构化目标和全新窗口。

TP teacher可拆为两类scheduler事件：prefill每个batch的输入token总和，以及decode每一步的
活跃lane数。对当前4096/8192 bytes-per-token两类模型，1–65,536 token可划为
{len(categories)}个联合区间，无损映射到两类模型的TP原生12桶。保存每个区间的batch count和
token mass，再加1–16活跃lane的decode step count，共{events['event_target_count']}个非负目标。
模型的collectives-per-forward与bytes-per-token由确定性结构适配器加入，不再把模型、TP size
和phase展开行误当独立流量样本。

窗口选择排除Phase 16的24个、Phase 27的60个和Phase 28的18个窗口。从BurstGPT三段及
Mooncake conversation/toolagent各选18个history-only medoid，共90个；冻结为45 train、
15 validation、15 first confirmation和15 second confirmation。Mooncake synthetic的12个
合格窗口已被前三轮冻结实验全部使用，因此本轮明确记录为0个可用，而不复用旧确认窗口。

Phase 29的30个train和12个validation画像允许作为开发数据复用；Phase 29两批确认画像永久
关闭，不得调参。Phase 30B训练单位是“独立画像×固定策略”，primary仍是compact32 H0事件
先验加bounded residual DNN，direct仅为控制。两批新确认预测必须在读取第一真值前同时冻结。
"""
    )
    (args.output_dir / "DONE").write_text("PASS\n")
    write_json(
        args.output_dir / "logs/contract.log",
        {
            "schema_version": "phase30a-contract-log-v1",
            "status": "PASS",
            "selected_profiles": len(selected_rows),
            "structured_event_targets": events["event_target_count"],
            "targets_generated": False,
            "predictions_generated": False,
            "phase29_confirmation_profiles_reused": False,
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
                "status": "PASS",
                "selected_profiles": len(selected_rows),
                "role_counts": dict(role_counts),
                "structured_event_targets": events["event_target_count"],
                "prefill_joint_categories": len(categories),
                "synthetic_unused_candidates": exhausted_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
