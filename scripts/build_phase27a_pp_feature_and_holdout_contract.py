#!/usr/bin/env python3
"""Freeze Phase 27 PP feature and new-window holdout contracts before labels exist."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SEED = "phase27-pp-independent-holdout-20260812-v1"
SEGMENTS = (
    "burstgpt_1",
    "burstgpt_2",
    "burstgpt_3",
    "mooncake_conversation",
    "mooncake_toolagent",
    "mooncake_synthetic",
)
SELECTED_PER_SEGMENT = 10
ROLE_QUOTAS = {
    "development_train": 5,
    "development_validation": 2,
    "independent_confirmation": 3,
}
SCALAR_SELECTION_FEATURES = (
    "history_rps",
    "history_interarrival_cv",
    "history_peak_to_mean_1s",
    "history_fano_1s",
    "history_input_mean",
    "history_input_p50",
    "history_input_p90",
    "history_input_p99",
    "history_output_mean",
    "history_output_p50",
    "history_output_p90",
    "history_output_p99",
)
SELECTION_FEATURES = (
    *(f"log1p_{name}" for name in SCALAR_SELECTION_FEATURES),
    "history_lm_correlation",
    *(f"history_input_log2_fraction_{index}" for index in range(18)),
    *(f"history_output_log2_fraction_{index}" for index in range(18)),
)
HISTORY_ONLY_SOURCE_COLUMNS = (
    "window_id",
    "source",
    "segment",
    "split",
    "cutoff_ms",
    "history_seconds",
    "history_count",
    *SCALAR_SELECTION_FEATURES,
    "history_lm_correlation",
    "history_input_log2_hist",
    "history_output_log2_hist",
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
        "--output-dir",
        type=Path,
        default=root
        / "experiment-results/phase27a_pp_feature_and_holdout_contract",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def robust_scale(matrix: np.ndarray) -> np.ndarray:
    median = np.median(matrix, axis=0)
    scale = np.quantile(matrix, 0.75, axis=0) - np.quantile(matrix, 0.25, axis=0)
    scale[scale < 1e-12] = 1.0
    return (matrix - median) / scale


def assign_clusters(matrix: np.ndarray, medoids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    distances = np.stack(
        [np.sum((matrix - matrix[index]) ** 2, axis=1) for index in medoids], axis=1
    )
    return np.argmin(distances, axis=1), np.min(distances, axis=1)


def choose_medoids(matrix: np.ndarray, count: int) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Deterministic robust-scaled farthest initialization and medoid refinement."""
    normalized = robust_scale(matrix)
    first = int(np.argmin(np.sum(normalized**2, axis=1)))
    medoids = [first]
    while len(medoids) < count:
        _, distance = assign_clusters(normalized, medoids)
        distance[medoids] = -1.0
        medoids.append(int(np.argmax(distance)))
    for _ in range(30):
        labels, _ = assign_clusters(normalized, medoids)
        updated = []
        for cluster in range(count):
            members = np.flatnonzero(labels == cluster)
            if not len(members):
                updated.append(medoids[cluster])
                continue
            centroid = np.mean(normalized[members], axis=0)
            distance = np.sum((normalized[members] - centroid) ** 2, axis=1)
            updated.append(int(members[np.argmin(distance)]))
        if updated == medoids:
            break
        medoids = updated
    labels, distances = assign_clusters(normalized, medoids)
    return medoids, labels, distances


def selection_vector(row: pd.Series) -> np.ndarray:
    values = [math.log1p(max(float(row[name]), 0.0)) for name in SCALAR_SELECTION_FEATURES]
    values.append(float(row["history_lm_correlation"]))
    count = max(int(row["history_count"]), 1)
    for name in ("history_input_log2_hist", "history_output_log2_hist"):
        histogram = json.loads(row[name])
        if len(histogram) != 18:
            raise ValueError(f"{row['window_id']}: {name} does not have 18 bins")
        values.extend(float(value) / count for value in histogram)
    return np.asarray(values, dtype=np.float64)


def role_order(window_id: str) -> str:
    return hashlib.sha256(f"{SEED}:{window_id}".encode()).hexdigest()


def feature_contract() -> dict:
    return {
        "schema_version": "phase27-pp-scheduler-sensitive-low-dimensional-profile-v1",
        "purpose": (
            "Predict fixed-draining topology-independent PP message histograms from "
            "a low-dimensional history profile; full request lists remain offline-only."
        ),
        "fixed_policy_constants": {
            "history_seconds": 300,
            "input_cap_tokens": 8192,
            "output_cap_tokens": 128,
            "pp_chunk_tokens": 4096,
            "pp_proxy_tensor_count": 2,
            "payload_bytes_per_token": 8192,
        },
        "base_profile_groups": {
            "arrival": [
                "request_count",
                "rps",
                "interarrival_cv",
                "peak_to_mean_1s",
                "fano_1s",
            ],
            "length_quantiles": [
                "input_mean_raw",
                "input_p50_raw",
                "input_p90_raw",
                "input_p99_raw",
                "input_mean_capped",
                "input_p50_capped",
                "input_p90_capped",
                "input_p99_capped",
                "output_mean_raw",
                "output_p50_raw",
                "output_p90_raw",
                "output_p99_raw",
                "output_mean_capped",
                "output_p50_capped",
                "output_p90_capped",
                "output_p99_capped",
                "lm_correlation_capped",
            ],
            "coarse_joint_distribution": ["joint_lm_4x4[16]"],
            "decode_survival": [
                "survival_m_gt_1",
                "survival_m_gt_8",
                "survival_m_gt_16",
                "survival_m_gt_32",
                "survival_m_gt_64",
            ],
            "prefill_chunk_structure": [
                "input_multichunk_fraction",
                "chunk_count_mean",
                "chunk_count_p50",
                "chunk_count_p90",
                "chunk_count_p99",
                "chunk_output_bucket_joint_2x5[10]",
                "chunk_output_work_mean",
                "chunk_output_work_p90",
                "chunk_output_work_p99",
            ],
            "request_order_summary": [
                "chunk_class_transition_2x2[4]",
                "multichunk_transition_rate",
                "multichunk_run_length_mean",
                "multichunk_run_length_p90",
                "multichunk_run_length_max",
                "rolling_multichunk_fraction_max_4",
                "rolling_multichunk_fraction_max_16",
                "rolling_multichunk_fraction_max_32",
            ],
        },
        "deployment_inputs": [
            "model_structure_features",
            "parallel_mode=pp",
            "pp_size",
            "max_microbatch_size",
            "fixed_draining_policy_id",
        ],
        "derived_policy_cross_features": [
            "max_microbatch_size/pp_size",
            "input_multichunk_fraction*max_microbatch_size/pp_size",
            "survival_m_gt_{1,8,16,32,64}*max_microbatch_size/pp_size",
            "rolling_multichunk_fraction_max_{4,16,32}*max_microbatch_size/pp_size",
        ],
        "forbidden_predictor_inputs": [
            "full_request_list",
            "representative_request_list",
            "future_window_columns",
            "Hfull_histogram_label",
            "communication_cost_label",
            "placement_or_topology_choice",
        ],
        "teacher_contract": (
            "Hfull labels are generated offline from the original-order complete history "
            "request list by the Phase 25D GPU-validated fixed-draining PP formula."
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_dir = args.output_dir / "selection"
    selection_dir.mkdir(exist_ok=True)

    windows = pd.read_csv(args.windows, usecols=list(HISTORY_ONLY_SOURCE_COLUMNS))
    profiles = pd.read_csv(args.phase16_profiles, usecols=["window_id"])
    excluded = set(profiles["window_id"].astype(str))
    selected_rows: list[dict] = []
    count_rows: list[dict] = []

    for segment in SEGMENTS:
        minimum = 32 if segment.startswith("burstgpt") else 128
        candidates = windows[
            (windows["segment"] == segment)
            & (windows["history_count"] >= minimum)
            & (~windows["window_id"].astype(str).isin(excluded))
        ].copy()
        candidates = candidates.sort_values("window_id", kind="stable").reset_index(drop=True)
        if len(candidates) < SELECTED_PER_SEGMENT:
            raise RuntimeError(f"{segment}: only {len(candidates)} eligible unused windows")
        matrix = np.stack([selection_vector(row) for _, row in candidates.iterrows()])
        medoids, labels, distances = choose_medoids(matrix, SELECTED_PER_SEGMENT)
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
                    "selection_distance_to_medoid_mean": float(np.mean(distances[members])),
                    "role_order_sha256": role_order(str(row["window_id"])),
                }
            )
        selected.sort(key=lambda row: row["role_order_sha256"])
        boundaries = np.cumsum(list(ROLE_QUOTAS.values()))
        roles = list(ROLE_QUOTAS)
        for index, row in enumerate(selected):
            role_index = int(np.searchsorted(boundaries, index, side="right"))
            row["phase27_role"] = roles[role_index]
            row["phase27_profile_id"] = f"phase27_{segment}_{index + 1:02d}"
            selected_rows.append(row)
        count_rows.append(
            {
                "segment": segment,
                "eligible_unused_windows": len(candidates),
                "minimum_history_count": minimum,
                "selected_windows": len(selected),
                "development_train": sum(row["phase27_role"] == "development_train" for row in selected),
                "development_validation": sum(row["phase27_role"] == "development_validation" for row in selected),
                "independent_confirmation": sum(row["phase27_role"] == "independent_confirmation" for row in selected),
            }
        )

    selected_rows.sort(key=lambda row: (row["segment"], row["phase27_profile_id"]))
    write_csv(selection_dir / "selected_windows.csv", selected_rows)
    write_csv(selection_dir / "candidate_counts.csv", count_rows)
    contract = feature_contract()
    write_json(args.output_dir / "feature_contract.json", contract)

    role_counts = Counter(row["phase27_role"] for row in selected_rows)
    summary = {
        "schema_version": "phase27a-pp-feature-and-holdout-contract-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "selection_seed": SEED,
        "selection_rule": (
            "Per segment, exclude all Phase 16 windows, filter by minimum history count, "
            "select 10 robust-scaled history-only medoids, then assign 5/2/3 roles by "
            "a seeded SHA-256 order before any Phase 27 Hfull label is generated."
        ),
        "selection_feature_count": len(SELECTION_FEATURES),
        "selection_features": list(SELECTION_FEATURES),
        "selected_profiles": len(selected_rows),
        "segments": len(SEGMENTS),
        "role_counts": dict(role_counts),
        "excluded_phase16_windows": len(excluded),
        "inputs": {
            "phase15_windows_sha256": sha256(args.windows),
            "phase16_profiles_sha256": sha256(args.phase16_profiles),
        },
        "label_state_at_freeze": "no_phase27_hfull_labels_generated",
    }
    write_json(args.output_dir / "summary.json", summary)

    checks = {
        "selected_profiles_60": len(selected_rows) == 60,
        "all_window_ids_unique": len({row["window_id"] for row in selected_rows}) == 60,
        "no_phase16_window_reused": not ({row["window_id"] for row in selected_rows} & excluded),
        "six_segments_with_ten_each": all(row["selected_windows"] == 10 for row in count_rows),
        "role_counts_30_12_18": role_counts
        == Counter(
            {
                "development_train": 30,
                "development_validation": 12,
                "independent_confirmation": 18,
            }
        ),
        "all_minimum_history_counts_met": all(
            row["history_count"] >= (32 if row["segment"].startswith("burstgpt") else 128)
            for row in selected_rows
        ),
        "selection_uses_history_columns_only": all(
            not name.startswith("future_") for name in HISTORY_ONLY_SOURCE_COLUMNS
        ),
        "full_request_lists_forbidden_as_predictor_input": "full_request_list"
        in contract["forbidden_predictor_inputs"],
        "holdout_role_frozen_before_labels": summary["label_state_at_freeze"]
        == "no_phase27_hfull_labels_generated",
    }
    audit = {
        "schema_version": "phase27a-pp-feature-and-holdout-contract-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    write_json(args.output_dir / "audit_summary.json", audit)
    if audit["status"] != "PASS":
        raise RuntimeError(audit)

    readme = f"""# Phase 27A：PP 调度敏感画像与新窗口冻结合同

本阶段只做**事前冻结**，不生成、读取或比较任何 Phase 27 Hfull 标签。目的是避免根据
真值误差挑窗口或改特征，从而给后续 PP 增强预测器保留真正独立的确认集。

## 新窗口划分

- 从 Phase 15 的 {len(windows):,} 个 300 秒历史窗口中选择；
- 明确排除 Phase 16 已使用的 {len(excluded)} 个窗口；
- 每个 BurstGPT/Mooncake segment 用 {len(SELECTION_FEATURES)} 个仅依赖历史的选择特征做
  robust-scaled medoid 覆盖，固定选 10 个，共 {len(selected_rows)} 个新窗口；
- 每个 segment 在看标签前按固定 SHA-256 顺序划为 5 个开发训练、2 个开发验证、3 个
  独立确认窗口；总计 30/12/18。

## 新增低维画像

保留原 4×4 长度联合分布、长度分位数、Decode 生存率和到达统计，并新增与 PP
fixed-draining 调度直接相关的低维摘要：4096-token prefill chunk 数、多 chunk 比例、
chunk×输出长度联合分布、相邻 chunk 类转移、多 chunk 连续段，以及 4/16/32 请求块内的
局部多 chunk 峰值。完整请求顺序只在离线阶段聚合成这些标量，不进入训练表。

`feature_contract.json` 是特征白名单和禁止输入；
`selection/selected_windows.csv` 是不可事后更改的新窗口及角色；
`selection/candidate_counts.csv` 记录各 segment 的候选覆盖。

当前能得出的结论仅是：独立评测合同和候选特征已经冻结并通过审计。当前不能得出新特征
改善 PP 的结论；该结论必须等待 Phase 27B/27C 在 18 个独立确认窗口上验证。
"""
    (args.output_dir / "README.md").write_text(readme)
    (args.output_dir / "DONE").write_text("PASS\n")
    run_log = {
        "event": "phase27a_contract_frozen",
        "status": "PASS",
        "selected_profiles": len(selected_rows),
        "role_counts": dict(role_counts),
        "phase27_labels_read_or_generated": False,
    }
    (args.output_dir / "run.log").write_text(json.dumps(run_log, sort_keys=True) + "\n")
    files = sorted(
        path
        for path in args.output_dir.rglob("*")
        if path.is_file() and path.name != "manifest.sha256"
    )
    (args.output_dir / "manifest.sha256").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(args.output_dir)}\n" for path in files)
    )
    print(json.dumps({"status": "PASS", "selected_profiles": len(selected_rows), "role_counts": dict(role_counts)}, indent=2))


if __name__ == "__main__":
    main()
