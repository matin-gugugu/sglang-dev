#!/usr/bin/env python3
"""Freeze a second untouched 18-window PP confirmation contract before labels."""

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
    SEGMENTS,
    choose_medoids,
    selection_vector,
    sha256,
)


QUOTAS = {
    "burstgpt_1": 3,
    "burstgpt_2": 3,
    "burstgpt_3": 3,
    "mooncake_conversation": 4,
    "mooncake_toolagent": 4,
    "mooncake_synthetic": 1,
}
SEED = "phase28-second-independent-confirmation-20260812-v1"
FROZEN_MAPPING = {
    "mb1": "h0",
    "mb4": "enhanced_bounded_residual",
    "mb16": "enhanced_bounded_residual",
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
        "--phase16-profiles",
        type=Path,
        default=root / "experiment-results/phase16_service_profiles/service_profiles.csv",
    )
    parser.add_argument(
        "--phase27-selection",
        type=Path,
        default=root
        / "experiment-results/phase27a_pp_feature_and_holdout_contract/selection/selected_windows.csv",
    )
    parser.add_argument(
        "--phase27d-decisions",
        type=Path,
        default=root
        / "experiment-results/phase27d_pp_independent_confirmation/analysis/post_confirmation_decisions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root
        / "experiment-results/phase28a_second_confirmation_contract",
    )
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_dir = args.output_dir / "selection"
    selection_dir.mkdir(exist_ok=True)

    decisions = read_csv(args.phase27d_decisions)
    actual_mapping = {
        row["policy"]: row["post_confirmation_recommendation"] for row in decisions
    }
    if actual_mapping != FROZEN_MAPPING:
        raise RuntimeError({"expected": FROZEN_MAPPING, "actual": actual_mapping})

    windows = pd.read_csv(args.windows, usecols=list(HISTORY_ONLY_SOURCE_COLUMNS))
    phase16 = set(pd.read_csv(args.phase16_profiles, usecols=["window_id"])["window_id"])
    phase27 = {row["window_id"] for row in read_csv(args.phase27_selection)}
    excluded = phase16 | phase27
    selected_rows = []
    candidate_rows = []
    for segment in SEGMENTS:
        minimum = 32 if segment.startswith("burstgpt") else 128
        candidates = windows[
            (windows["segment"] == segment)
            & (windows["history_count"] >= minimum)
            & (~windows["window_id"].astype(str).isin(excluded))
        ].copy()
        candidates = candidates.sort_values("window_id", kind="stable").reset_index(drop=True)
        quota = QUOTAS[segment]
        if len(candidates) < quota:
            raise RuntimeError(f"{segment}: only {len(candidates)} candidates")
        matrix = np.stack([selection_vector(row) for _, row in candidates.iterrows()])
        medoids, labels, distances = choose_medoids(matrix, quota)
        for cluster, index in enumerate(medoids):
            row = candidates.iloc[index]
            members = np.flatnonzero(labels == cluster)
            profile_id = f"phase28_{segment}_{cluster + 1:02d}"
            selected_rows.append(
                {
                    "profile_id": profile_id,
                    "role": "second_independent_confirmation",
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
                }
            )
        candidate_rows.append(
            {
                "segment": segment,
                "eligible_unused_windows": len(candidates),
                "minimum_history_count": minimum,
                "selected_windows": quota,
            }
        )
    selected_rows.sort(key=lambda row: row["profile_id"])
    write_csv(selection_dir / "selected_windows.csv", selected_rows)
    write_csv(selection_dir / "candidate_counts.csv", candidate_rows)
    write_json(
        args.output_dir / "frozen_method_mapping.json",
        {
            "schema_version": "phase28-frozen-pp-method-mapping-v1",
            "mapping": FROZEN_MAPPING,
            "source": "Phase 27D post-confirmation decisions",
            "selection_rule_changes_allowed": False,
            "hybrid_score_previously_computed": False,
        },
    )

    summary = {
        "schema_version": "phase28a-second-confirmation-contract-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "selection_seed": SEED,
        "selection_rule": (
            "exclude Phase16 and Phase27 windows, then select robust-scaled history-only "
            "medoids with frozen quotas 3/3/3/4/4/1 before any Phase28 Hfull label; "
            "Mooncake synthetic has only one eligible unused window"
        ),
        "selected_profiles": len(selected_rows),
        "segments": len(SEGMENTS),
        "selection_feature_count": len(SELECTION_FEATURES),
        "excluded_windows": {
            "phase16": len(phase16),
            "phase27": len(phase27),
            "union": len(excluded),
        },
        "frozen_mapping": FROZEN_MAPPING,
        "label_state_at_freeze": "no_phase28_hfull_labels_generated",
        "prediction_state_at_freeze": "no_phase28_predictions_generated",
        "inputs": {
            "phase15_windows_sha256": sha256(args.windows),
            "phase16_profiles_sha256": sha256(args.phase16_profiles),
            "phase27_selection_sha256": sha256(args.phase27_selection),
            "phase27d_decisions_sha256": sha256(args.phase27d_decisions),
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    checks = {
        "selected_profiles_18": len(selected_rows) == 18,
        "frozen_segment_quotas": Counter(row["segment"] for row in selected_rows)
        == Counter(QUOTAS),
        "all_window_ids_unique": len({row["window_id"] for row in selected_rows}) == 18,
        "no_phase16_or_phase27_reuse": not (
            {row["window_id"] for row in selected_rows} & excluded
        ),
        "history_only_selection": all(
            not name.startswith("future_") for name in HISTORY_ONLY_SOURCE_COLUMNS
        ),
        "mapping_matches_phase27d": actual_mapping == FROZEN_MAPPING,
        "mapping_frozen_before_predictions_and_labels": summary["label_state_at_freeze"]
        == "no_phase28_hfull_labels_generated"
        and summary["prediction_state_at_freeze"]
        == "no_phase28_predictions_generated",
    }
    audit = {
        "schema_version": "phase28a-second-confirmation-contract-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    write_json(args.output_dir / "audit_summary.json", audit)
    if audit["status"] != "PASS":
        raise RuntimeError(audit)
    (args.output_dir / "README.md").write_text(
        f"""# Phase 28A：第二批PP独立确认合同

本阶段在生成任何Phase 28预测和Hfull标签前，冻结Phase 27D确认后形成的方法映射：
`MB1=H0、MB4/MB16=增强bounded residual`。从Phase 15窗口中排除Phase 16的24个窗口和
Phase 27的60个窗口，再用相同{len(SELECTION_FEATURES)}个历史侧特征按3/3/3/4/4/1配额
选择medoid，共18个第二独立确认画像。Mooncake synthetic总共只有12个候选，排除前两轮后
仅剩1个，因此将多出的2个配额分给conversation和toolagent；该调整发生在任何选择清单、
预测或Hfull标签生成之前。

`selection/selected_windows.csv`是不可事后更改的窗口清单；
`frozen_method_mapping.json`是不可事后更改的方法映射。当前没有Phase 28预测或Hfull标签，
因此后续可以为这份混合映射提供真正独立的成绩。
"""
    )
    (args.output_dir / "DONE").write_text("PASS\n")
    (args.output_dir / "run.log").write_text(
        json.dumps(
            {
                "event": "phase28a_contract_frozen",
                "status": "PASS",
                "selected_profiles": len(selected_rows),
                "labels_generated": False,
                "predictions_generated": False,
            },
            sort_keys=True,
        )
        + "\n"
    )
    files = sorted(
        path
        for path in args.output_dir.rglob("*")
        if path.is_file() and path.name != "manifest.sha256"
    )
    (args.output_dir / "manifest.sha256").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(args.output_dir)}\n" for path in files)
    )
    print(json.dumps({"status": "PASS", "selected_profiles": len(selected_rows), "mapping": FROZEN_MAPPING}, indent=2))


if __name__ == "__main__":
    main()
