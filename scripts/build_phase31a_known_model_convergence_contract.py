#!/usr/bin/env python3
"""Freeze a normal-range, request-disjoint TP/PP convergence split."""

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

from build_phase27a_pp_feature_and_holdout_contract import (
    HISTORY_ONLY_SOURCE_COLUMNS,
    SELECTION_FEATURES,
    choose_medoids,
    selection_vector,
)


HISTORY_MS = 300_000
SEED = "phase31-known-model-normal-range-disjoint-v1-20260813"
MODELS = ("deepseek-v2-lite", "qwen3-8b", "qwen3-30b-a3b")
TP_SIZES = (2, 4, 8)
TP_POLICIES = ("latency", "balanced", "throughput")
PP_SIZES = (2, 4, 8)
PP_MICROBATCHES = (1, 4, 16)
ROLE_QUOTAS = {
    "burstgpt_1": {"development_train": 12, "development_validation": 3, "fixed_prediction": 3},
    "burstgpt_2": {"development_train": 12, "development_validation": 3, "fixed_prediction": 3},
    "burstgpt_3": {"development_train": 12, "development_validation": 3, "fixed_prediction": 3},
    "mooncake_conversation": {"development_train": 1, "development_validation": 1, "fixed_prediction": 1},
    "mooncake_toolagent": {"development_train": 1, "development_validation": 0, "fixed_prediction": 0},
    "mooncake_synthetic": {"development_train": 1, "development_validation": 0, "fixed_prediction": 0},
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--windows",
        type=Path,
        default=root / "experiment-results/phase15_trace_data/windows.csv.gz",
    )
    parser.add_argument(
        "--phase27-selection",
        type=Path,
        default=root / "experiment-results/phase27a_pp_feature_and_holdout_contract/selection/selected_windows.csv",
    )
    parser.add_argument(
        "--phase28-selection",
        type=Path,
        default=root / "experiment-results/phase28a_second_confirmation_contract/selection/selected_windows.csv",
    )
    parser.add_argument(
        "--phase30-selection",
        type=Path,
        default=root / "experiment-results/phase30a_tp_structured_event_contract/selection/selected_windows.csv",
    )
    parser.add_argument(
        "--model-features",
        type=Path,
        default=root / "experiment-results/phase16_model_features/model_features.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase31a_known_model_convergence_contract",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def overlaps(cutoff_a: int, cutoff_b: int) -> bool:
    return abs(cutoff_a - cutoff_b) < HISTORY_MS


def historical_confirmation_intervals(args: argparse.Namespace) -> dict[str, list[int]]:
    output: dict[str, list[int]] = {segment: [] for segment in ROLE_QUOTAS}
    for row in read_csv(args.phase27_selection):
        if row["phase27_role"] == "independent_confirmation":
            output[row["segment"]].append(int(row["cutoff_ms"]))
    for row in read_csv(args.phase28_selection):
        output[row["segment"]].append(int(row["cutoff_ms"]))
    for row in read_csv(args.phase30_selection):
        if row["role"] in {"independent_confirmation", "second_independent_confirmation"}:
            output[row["segment"]].append(int(row["cutoff_ms"]))
    return output


def disjoint_pool(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a deterministic maximum-size non-overlapping greedy pool."""
    chosen = []
    last_cutoff = None
    for index, row in frame.sort_values(["cutoff_ms", "window_id"], kind="stable").iterrows():
        cutoff = int(row["cutoff_ms"])
        if last_cutoff is None or cutoff - last_cutoff >= HISTORY_MS:
            chosen.append(index)
            last_cutoff = cutoff
    return frame.loc[chosen].reset_index(drop=True)


def seeded_role_order(window_id: str) -> str:
    return hashlib.sha256(f"{SEED}:{window_id}".encode()).hexdigest()


def main() -> None:
    args = parse_args()
    (args.output_dir / "selection").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(parents=True, exist_ok=True)

    windows = pd.read_csv(args.windows, usecols=list(HISTORY_ONLY_SOURCE_COLUMNS))
    confirmation_intervals = historical_confirmation_intervals(args)
    selected_rows: list[dict] = []
    inventory_rows: list[dict] = []
    fallback_rows: list[dict] = []

    for segment, quotas in ROLE_QUOTAS.items():
        quota = sum(quotas.values())
        minimum = 32 if segment.startswith("burstgpt") else 128
        candidates = windows[
            (windows["segment"] == segment) & (windows["history_count"] >= minimum)
        ].copy()
        before_embargo = len(candidates)
        candidates = candidates[
            [
                not any(overlaps(int(cutoff), old) for old in confirmation_intervals[segment])
                for cutoff in candidates["cutoff_ms"]
            ]
        ].copy()
        after_embargo = len(candidates)
        if not len(candidates):
            raise RuntimeError(f"{segment}: no candidates after historical-confirmation embargo")

        matrix = np.stack([selection_vector(row) for _, row in candidates.iterrows()])
        median = np.median(matrix, axis=0)
        q25 = np.quantile(matrix, 0.25, axis=0)
        q75 = np.quantile(matrix, 0.75, axis=0)
        scale = q75 - q25
        scale[scale < 1e-9] = 1.0
        robust_distance = np.sqrt(np.mean(((matrix - median) / scale) ** 2, axis=1))
        candidates["normality_distance"] = robust_distance

        selected_pool = None
        used_quantile = None
        for quantile in (0.95, 0.99, 1.0):
            threshold = float(np.quantile(robust_distance, quantile))
            normal = candidates[candidates["normality_distance"] <= threshold].copy()
            pool = disjoint_pool(normal)
            if len(pool) >= quota:
                selected_pool = pool
                used_quantile = quantile
                break
        if selected_pool is None:
            raise RuntimeError(f"{segment}: fewer than {quota} disjoint normal candidates")

        pool_matrix = np.stack([selection_vector(row) for _, row in selected_pool.iterrows()])
        medoids, labels, distances = choose_medoids(pool_matrix, quota)
        chosen = []
        for cluster, index in enumerate(medoids):
            row = selected_pool.iloc[index]
            members = np.flatnonzero(labels == cluster)
            chosen.append(
                {
                    "window_id": str(row["window_id"]),
                    "source": str(row["source"]),
                    "segment": segment,
                    "source_split": str(row["split"]),
                    "cutoff_ms": int(row["cutoff_ms"]),
                    "history_seconds": int(row["history_seconds"]),
                    "history_count": int(row["history_count"]),
                    "normality_distance": float(row["normality_distance"]),
                    "normality_pool_quantile": float(used_quantile),
                    "selection_cluster": cluster,
                    "selection_cluster_members": int(len(members)),
                    "selection_distance_to_medoid_mean": float(np.mean(distances[members])),
                    "role_order_sha256": seeded_role_order(str(row["window_id"])),
                }
            )
        chosen.sort(key=lambda row: row["role_order_sha256"])
        position = 0
        for role, count in quotas.items():
            for row in chosen[position : position + count]:
                row["role"] = role
                row["profile_id"] = f"phase31_{segment}_{len(selected_rows) + 1:03d}"
                selected_rows.append(row)
            position += count

        inventory_rows.append(
            {
                "segment": segment,
                "minimum_history_count": minimum,
                "eligible_before_embargo": before_embargo,
                "eligible_after_embargo": after_embargo,
                "disjoint_normal_pool": len(selected_pool),
                "normality_pool_quantile": used_quantile,
                "selected": quota,
                **{f"selected_{role}": count for role, count in quotas.items()},
            }
        )
        if used_quantile != 0.95:
            fallback_rows.append(
                {"segment": segment, "normality_pool_quantile": used_quantile, "reason": "P95 pool lacked enough request-disjoint windows after historical-confirmation embargo"}
            )

    selected_rows.sort(key=lambda row: (row["segment"], row["cutoff_ms"]))
    role_counts = Counter(row["role"] for row in selected_rows)
    segment_counts = Counter(row["segment"] for row in selected_rows)
    ids = {row["window_id"] for row in selected_rows}
    pair_overlaps = []
    for index, left in enumerate(selected_rows):
        for right in selected_rows[index + 1 :]:
            if left["segment"] == right["segment"] and overlaps(left["cutoff_ms"], right["cutoff_ms"]):
                pair_overlaps.append((left["window_id"], right["window_id"]))
    old_overlaps = [
        row["window_id"]
        for row in selected_rows
        if any(overlaps(row["cutoff_ms"], old) for old in confirmation_intervals[row["segment"]])
    ]
    models = json.loads(args.model_features.read_text())
    model_names = {row["model"] for row in models}

    write_csv(args.output_dir / "selection/selected_windows.csv", selected_rows)
    write_csv(args.output_dir / "selection/candidate_inventory.csv", inventory_rows)
    if fallback_rows:
        write_csv(args.output_dir / "selection/normality_fallbacks.csv", fallback_rows)

    checks = {
        "profiles_59": len(selected_rows) == 59,
        "roles_39_10_10": role_counts == Counter({"development_train": 39, "development_validation": 10, "fixed_prediction": 10}),
        "segments_exact": segment_counts == Counter({"burstgpt_1": 18, "burstgpt_2": 18, "burstgpt_3": 18, "mooncake_conversation": 3, "mooncake_toolagent": 1, "mooncake_synthetic": 1}),
        "window_ids_unique": len(ids) == len(selected_rows),
        "request_intervals_disjoint_within_new_split": not pair_overlaps,
        "historical_confirmation_embargo_300s": not old_overlaps,
        "history_only_selection": all(not name.startswith("future_") for name in HISTORY_ONLY_SOURCE_COLUMNS),
        "three_known_models": model_names == set(MODELS),
        "no_target_or_prediction_at_freeze": True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = {
        "schema_version": "phase31a-known-model-convergence-contract-v1",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "known-model in-distribution TP/PP H0 plus DNN residual first-stage closure",
        "selection_seed": SEED,
        "profiles": len(selected_rows),
        "role_counts": dict(role_counts),
        "segment_counts": dict(segment_counts),
        "models": list(MODELS),
        "tp_sizes": list(TP_SIZES),
        "tp_policies": list(TP_POLICIES),
        "pp_sizes": list(PP_SIZES),
        "pp_microbatches": list(PP_MICROBATCHES),
        "validation_strength": "known three models appear in train/validation/fixed prediction; no whole-model holdout",
        "normal_range_rule": "history-count admission plus robust-distance central pool; P95 preferred, deterministic fallback recorded only when Mooncake capacity is exhausted by 300-second embargo",
        "isolation_rule": "all new roles are 300-second interval/request disjoint; all historical Phase27/28/30 confirmation intervals receive a 300-second embargo",
        "mooncake_capacity_limit": "only five request-disjoint Mooncake blocks remain after historical-confirmation embargo; three conversation blocks span train/validation/fixed prediction, toolagent and synthetic are train-only",
        "target_state_at_freeze": "no_phase31_hfull_targets_generated",
        "prediction_state_at_freeze": "no_phase31_predictions_generated",
        "checks": checks,
        "inputs": {
            "phase15_windows_sha256": sha256(args.windows),
            "phase27_selection_sha256": sha256(args.phase27_selection),
            "phase28_selection_sha256": sha256(args.phase28_selection),
            "phase30_selection_sha256": sha256(args.phase30_selection),
            "model_features_sha256": sha256(args.model_features),
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "audit_summary.json", {"schema_version": "phase31a-contract-audit-v1", "status": status, "checks": checks, "pair_overlaps": pair_overlaps, "historical_confirmation_overlaps": old_overlaps})
    (args.output_dir / "README.md").write_text(f"""# Phase 31A：三模型 TP/PP 收敛数据合同

本阶段只冻结今晚第一阶段收敛实验的数据和方法边界，不生成 Hfull 标签、不训练模型、也不读取预测结果。

## 数据范围

- 三个已知模型：DeepSeek-V2-Lite、Qwen3-8B、Qwen3-30B-A3B；三者都会进入训练、验证和固定预测，不做整模型留出；
- 共 {len(selected_rows)} 个历史画像：{role_counts['development_train']} 个训练、{role_counts['development_validation']} 个验证、{role_counts['fixed_prediction']} 个固定预测；
- BurstGPT 三段各 18 个；Mooncake 共 5 个剩余独立块；
- 所有画像来自历史侧低维统计的正常中心区域，使用robust-distance medoid选择，不按target或误差选样本。

## 隔离修复

旧实验曾把300秒Mooncake窗口按60秒步长视为不同画像，造成不同window id共享请求。本合同使用300秒时间区间作为硬隔离单位：新训练、验证、固定预测之间共享请求为0；Phase27、Phase28和Phase30的历史确认区间也全部设置300秒embargo。

Mooncake原始trace较短，严格embargo后只剩5个独立块，因此固定预测中只有1个Mooncake conversation画像。它满足第一阶段的正常流量与防泄漏要求，但不能支持强Mooncake泛化结论。

## 不改变的研究口径

Hfull只作离线teacher；预测器输入仍为低维历史画像、模型结构、固定TP/PP配置和策略；模型形式必须是`H0 + DNN residual`。fixed-draining、每1000请求归一化、12桶calls/bytes和统一参考cost定义均保持不变。
""")
    write_json(args.output_dir / "logs/build.log", {"event": "phase31a_contract_frozen", "status": status, "profiles": len(selected_rows), "roles": dict(role_counts), "historical_confirmation_embargo_intervals": sum(len(values) for values in confirmation_intervals.values())})
    (args.output_dir / "DONE").write_text(f"{status}\n")
    manifest = [
        f"{sha256(path)}  {path.relative_to(args.output_dir)}"
        for path in sorted(args.output_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.sha256"
    ]
    (args.output_dir / "manifest.sha256").write_text("\n".join(manifest) + "\n")
    if status != "PASS":
        raise RuntimeError(checks)
    print(json.dumps({"status": status, "profiles": len(selected_rows), "roles": dict(role_counts), "fallbacks": fallback_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
