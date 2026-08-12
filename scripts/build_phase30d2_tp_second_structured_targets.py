#!/usr/bin/env python3
"""Generate Phase30 second-confirmation event targets after mapping freeze."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from build_phase27b_pp_hfull_dataset import HISTORY_SECONDS, summarize_profile
from build_phase29b_tp_hfull_dataset import MODELS, STRATEGIES, TP_SIZES, all_model_features
from build_phase30b_tp_structured_event_dataset import (
    SECOND_ROLE,
    events_from_requests,
    identifiers,
    max_vector_error,
    prefix_events,
    read_csv,
    reconstruct_message_vectors,
    sha256,
    target_vectors_from_requests,
    verify_raw,
    write_csv,
    write_csv_gz,
    write_json,
)
from prepare_phase15_trace_windows import BURST_FILES, MOONCAKE_FILES, load_segment


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    phase30c = root / "experiment-results/phase30c_tp_structured_event_training"
    phase30d1 = root / "experiment-results/phase30d1_tp_first_structured_confirmation"
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument(
        "--selection",
        type=Path,
        default=root
        / "experiment-results/phase30a_tp_structured_event_contract/selection/selected_windows.csv",
    )
    parser.add_argument(
        "--event-contract",
        type=Path,
        default=root
        / "experiment-results/phase30a_tp_structured_event_contract/event_contract.json",
    )
    parser.add_argument(
        "--frozen-second-predictions",
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
        "--model-features",
        type=Path,
        default=root / "experiment-results/phase16_model_features/model_features.json",
    )
    parser.add_argument(
        "--official-raw-manifest",
        type=Path,
        default=root / "experiment-results/phase15_trace_data/source_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase30d2_tp_second_structured_targets",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in ("labels", "analysis", "logs"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)

    # Verify all frozen artifacts before reading raw requests or creating targets.
    training_audit = json.loads(args.training_audit.read_text())
    first_audit = json.loads(args.first_audit.read_text())
    prediction_sha = sha256(args.frozen_second_predictions)
    if prediction_sha != training_audit["second_confirmation_predictions_sha256"]:
        raise ValueError("frozen second prediction SHA mismatch")
    mapping_rows = read_csv(args.second_mapping)
    mapping = {row["policy"]: row["selected_method"] for row in mapping_rows}
    expected_mapping = {policy: "h0" for policy in STRATEGIES}
    if mapping != expected_mapping:
        raise ValueError("unexpected second confirmation mapping")
    if sha256(args.second_mapping) != first_audit["second_mapping_sha256"]:
        raise ValueError("second mapping SHA mismatch")

    root = Path(__file__).resolve().parents[1]
    raw_checks = verify_raw(args.raw_dir, args.official_raw_manifest)
    if len(raw_checks) != 6 or not all(raw_checks.values()):
        raise RuntimeError(raw_checks)
    contract = json.loads(args.event_contract.read_text())
    models = all_model_features(args.model_features)
    selection = [row for row in read_csv(args.selection) if row["role"] == SECOND_ROLE]
    if len(selection) != 15:
        raise ValueError("expected 15 second-confirmation profiles")
    file_by_segment = {
        segment: args.raw_dir / name
        for name, (segment, _) in {**BURST_FILES, **MOONCAKE_FILES}.items()
    }
    needed_segments = {row["segment"] for row in selection}
    raw_arrays = {
        segment: load_segment(path)
        for segment, path in file_by_segment.items()
        if segment in needed_segments
    }

    target_rows = []
    request_total = 0
    count_checks = []
    partition_checks = []
    adapter_errors = []
    for selected in selection:
        timestamps, inputs, outputs = raw_arrays[selected["segment"]]
        cutoff = int(selected["cutoff_ms"])
        left = int(
            np.searchsorted(
                timestamps, cutoff - HISTORY_SECONDS * 1000, side="left"
            )
        )
        right = int(np.searchsorted(timestamps, cutoff, side="left"))
        compatibility = {
            **selected,
            "phase27_profile_id": selected["profile_id"],
            "phase27_role": selected["role"],
        }
        profile, requests = summarize_profile(
            compatibility,
            timestamps[left:right],
            inputs[left:right],
            outputs[left:right],
        )
        request_total += len(requests)
        count_checks.append(len(requests) == int(selected["history_count"]))
        for policy, strategy in STRATEGIES.items():
            events, partition = events_from_requests(requests, strategy, contract)
            partition_checks.append(partition)
            target_rows.append(
                {
                    **identifiers(
                        profile,
                        policy,
                        "phase30_raw_full_window_structured_event_teacher",
                    ),
                    **prefix_events("target_event_", events),
                }
            )
            for model_name in MODELS:
                model = models[model_name][0]
                reconstructed = reconstruct_message_vectors(events, model, contract)
                for _tp_size in TP_SIZES:
                    actual = target_vectors_from_requests(requests, strategy, model)
                    adapter_errors.append(max_vector_error(reconstructed, actual))

    max_calls_absolute = max(row[0] for row in adapter_errors)
    max_bytes_absolute = max(row[1] for row in adapter_errors)
    max_calls_relative = max(row[2] for row in adapter_errors)
    max_bytes_relative = max(row[3] for row in adapter_errors)
    adapter_rows = [
        {
            "profiles": len(selection),
            "profile_policy_target_units": len(target_rows),
            "expanded_model_tp_configurations": len(adapter_errors),
            "expanded_model_tp_phase_rows": len(adapter_errors) * 2,
            "max_calls_bin_absolute_error": max_calls_absolute,
            "max_logical_bytes_bin_absolute_error": max_bytes_absolute,
            "max_calls_bin_relative_error": max_calls_relative,
            "max_logical_bytes_bin_relative_error": max_bytes_relative,
        }
    ]
    write_csv_gz(
        args.output_dir / "labels/second_confirmation_event_targets.csv.gz",
        target_rows,
    )
    write_csv(args.output_dir / "analysis/adapter_exactness.csv", adapter_rows)

    checks = {
        "phase30c_status_pass": training_audit["status"] == "PASS",
        "phase30d1_status_pass": first_audit["status"] == "PASS",
        "frozen_second_prediction_sha_matches": prediction_sha
        == training_audit["second_confirmation_predictions_sha256"],
        "second_mapping_sha_matches": sha256(args.second_mapping)
        == first_audit["second_mapping_sha256"],
        "second_mapping_all_h0": mapping == expected_mapping,
        "raw_source_hashes_6_of_6": len(raw_checks) == 6 and all(raw_checks.values()),
        "profiles_15_target_units_45": len(selection) == 15 and len(target_rows) == 45,
        "history_counts_match_15_of_15": len(count_checks) == 15
        and all(count_checks),
        "full_partition_checks_45_of_45": len(partition_checks) == 45
        and all(partition_checks),
        "adapter_expansions_405_phase_rows_810": len(adapter_errors) == 405,
        "adapter_relative_errors_below_one_part_per_billion": max_calls_relative
        <= 1e-9
        and max_bytes_relative <= 1e-9,
        "no_request_arrays_saved": not any(
            key.lower()
            in {"requests", "request_list", "request_array", "full_requests"}
            for row in target_rows
            for key in row
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    if status != "PASS":
        raise RuntimeError(checks)
    summary = {
        "schema_version": "phase30d2-tp-second-structured-targets-v1",
        "status": status,
        "objective": "generate isolated second-confirmation structured-event targets only after prediction and mapping freeze",
        "profiles": len(selection),
        "profile_policy_target_units": len(target_rows),
        "full_window_requests": request_total,
        "expanded_adapter_configurations": len(adapter_errors),
        "expanded_adapter_phase_rows": len(adapter_errors) * 2,
        "second_mapping": mapping,
        "inputs": {
            "selection_sha256": sha256(args.selection),
            "event_contract_sha256": sha256(args.event_contract),
            "frozen_second_predictions_sha256": prediction_sha,
            "training_audit_sha256": sha256(args.training_audit),
            "second_mapping_sha256": sha256(args.second_mapping),
            "first_audit_sha256": sha256(args.first_audit),
            "model_features_sha256": sha256(args.model_features),
            "official_raw_manifest_sha256": sha256(args.official_raw_manifest),
        },
        "raw_source_checks": raw_checks,
        "adapter_exactness": adapter_rows[0],
        "checks": checks,
        "can_conclude": [
            "15 second-confirmation profiles were converted into 45 fixed-draining event targets after mapping freeze",
            "the deterministic adapter matches direct structural histograms within relative floating tolerance",
        ],
        "cannot_conclude": [
            "the second-confirmation performance before joining frozen predictions",
            "the full request arrays are valid online predictor inputs",
        ],
        "next_step": "archive this target artifact and evaluate only the already-frozen Phase30C second predictions",
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(
        args.output_dir / "audit_summary.json",
        {
            "schema_version": "phase30d2-tp-second-structured-targets-audit-v1",
            "status": status,
            "checks": checks,
            "second_targets_sha256": sha256(
                args.output_dir / "labels/second_confirmation_event_targets.csv.gz"
            ),
        },
    )
    (args.output_dir / "README.md").write_text(
        f"""# Phase 30D2：TP第二独立确认结构事件真值

状态：**PASS**。在核验Phase30C第二预测SHA与Phase30D1冻结映射后，本阶段才读取raw trace，
为15个第二确认画像生成45个profile×policy Hfull结构事件target。共处理{request_total:,}个完整
历史请求；请求数组只在内存中用于离线teacher，没有保存到Git。

62维事件经确定性适配器展开为405个模型×TP×策略配置、810条phase行；最大calls/bytes
相对误差分别为{max_calls_relative:.3e}和{max_bytes_relative:.3e}，审计通过。当前仍不能给出第二
确认误差；下一阶段只能把本真值连接到已经冻结的3,240条预测，不能重训或改写映射。
"""
    )
    (args.output_dir / "DONE").write_text("PASS\n")
    try:
        repository_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        repository_head = "unknown"
    write_json(
        args.output_dir / "logs/build.log",
        {
            "schema_version": "phase30d2-build-log-v1",
            "status": status,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository_head_at_generation": repository_head,
            "profiles": len(selection),
            "target_units": len(target_rows),
            "full_window_requests": request_total,
            "request_arrays_saved": False,
            "frozen_prediction_and_mapping_verified_before_target_generation": True,
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
                "profiles": len(selection),
                "target_units": len(target_rows),
                "full_window_requests": request_total,
                "adapter_max_relative_errors": {
                    "calls": max_calls_relative,
                    "logical_bytes": max_bytes_relative,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
