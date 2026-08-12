#!/usr/bin/env python3
"""Build aligned multi-model TP Hfull data while keeping confirmation targets isolated."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from build_phase21b_pp_h0 import pseudo_requests
from build_phase25_full_window_teacher import (
    PHASES,
    TP_BIN_EDGES,
    normalize,
    tp_batches,
    tp_histograms,
)
from build_phase27b_pp_hfull_dataset import (
    HISTORY_SECONDS,
    scalar_profile_features,
    summarize_profile,
)
from prepare_phase15_trace_windows import BURST_FILES, MOONCAKE_FILES, load_segment


MODELS = ("deepseek-v2-lite", "qwen3-8b", "qwen3-30b-a3b")
TP_SIZES = (2, 4, 8)
STRATEGIES = {
    "latency": {"max_batch_size": 4, "max_prefill_tokens": 8192},
    "balanced": {"max_batch_size": 8, "max_prefill_tokens": 32768},
    "throughput": {"max_batch_size": 16, "max_prefill_tokens": 65536},
}
TEACHER_STATUS = "GPU_VALIDATED_STRUCTURAL_FORMULA_SENTINELS_4_CELLS"
TEACHER_KIND = "full_window_fixed_draining_structural_teacher"
COMMON_REFERENCE_LAUNCH_US = 5.0
COMMON_REFERENCE_BANDWIDTH_GBPS = 100.0


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument(
        "--phase27-windows",
        type=Path,
        default=root
        / "experiment-results/phase29a_tp_aligned_contract/windows/phase27_aligned_windows.csv",
    )
    parser.add_argument(
        "--phase28-windows",
        type=Path,
        default=root
        / "experiment-results/phase29a_tp_aligned_contract/windows/phase28_second_confirmation_windows.csv",
    )
    parser.add_argument(
        "--phase29a-summary",
        type=Path,
        default=root / "experiment-results/phase29a_tp_aligned_contract/summary.json",
    )
    parser.add_argument(
        "--feature-contract",
        type=Path,
        default=root / "experiment-results/phase29a_tp_aligned_contract/feature_contract.json",
    )
    parser.add_argument(
        "--model-features",
        type=Path,
        default=root / "experiment-results/phase16_model_features/model_features.json",
    )
    parser.add_argument(
        "--phase26a-summary",
        type=Path,
        default=root / "experiment-results/phase26a_tp_hfull_teacher_audit/summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase29b_tp_hfull_dataset",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_gzip(path: Path, text: str) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as output:
            output.write(text.encode())


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


def bin_vectors(histogram: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
    calls = np.zeros(12, dtype=np.float64)
    logical_bytes = np.zeros(12, dtype=np.float64)
    for payload, count in histogram.items():
        index = int(
            np.clip(np.searchsorted(TP_BIN_EDGES, payload, side="right") - 1, 0, 11)
        )
        calls[index] += count
        logical_bytes[index] += payload * count
    return calls, logical_bytes


def reference_cost(calls: np.ndarray, logical_bytes: np.ndarray) -> float:
    return float(
        COMMON_REFERENCE_LAUNCH_US * calls.sum()
        + logical_bytes.sum() / (COMMON_REFERENCE_BANDWIDTH_GBPS * 1e9) * 1e6
    )


def target_row(
    profile: dict,
    model: dict,
    tp_size: int,
    policy: str,
    phase: str,
    histogram: dict[int, float],
) -> dict:
    calls, logical_bytes = bin_vectors(histogram)
    exact_bytes = {str(payload): payload * count for payload, count in histogram.items()}
    return {
        "label_id": f"{model['model']}/tp{tp_size}/{policy}/{profile['profile_id']}/hfull/{phase}",
        "label_status": TEACHER_STATUS,
        "teacher_kind": TEACHER_KIND,
        "model": model["model"],
        "model_config_sha256": model["config_sha256"],
        "profile_id": profile["profile_id"],
        "role": profile["phase27_role"],
        "source": profile["source"],
        "segment": profile["segment"],
        "source_split": profile["source_split"],
        "window_id": profile["window_id"],
        "parallelism": "tp",
        "parallel_size": tp_size,
        "policy": policy,
        "phase": phase,
        "requests": profile["request_count"],
        "normalization_requests": 1000,
        "boundary_multiplier": 1,
        "bin_schema_id": "tp_native_12bin_4k_512m_v1",
        "bin_edges_bytes_json": json.dumps(TP_BIN_EDGES.tolist(), separators=(",", ":")),
        "total_calls_per_1000": float(calls.sum()),
        "total_logical_bytes_per_1000": float(logical_bytes.sum()),
        "common_reference_cost_us_per_1000": reference_cost(calls, logical_bytes),
        "calls_by_12bin_json": json.dumps(calls.tolist(), separators=(",", ":")),
        "logical_bytes_by_12bin_json": json.dumps(
            logical_bytes.tolist(), separators=(",", ":")
        ),
        "exact_calls_histogram_per_1000_json": json.dumps(
            {str(payload): count for payload, count in histogram.items()},
            separators=(",", ":"),
        ),
        "exact_logical_bytes_histogram_per_1000_json": json.dumps(
            exact_bytes, separators=(",", ":")
        ),
        "scheduler_contract": "tp_fixed_order_token_budget_batches_v1",
    }


def all_model_features(path: Path) -> dict[str, tuple[dict, dict]]:
    excluded = {
        "model",
        "config_path",
        "architecture_audit_only",
        "model_type_audit_only",
        "raw_op_template_audit_only",
        "config_sha256",
    }
    result = {}
    for row in json.loads(path.read_text()):
        result[row["model"]] = (
            row,
            {
                f"feature_model_{key}": value
                for key, value in row.items()
                if key not in excluded
            },
        )
    return result


def feature_values(
    profile: dict,
    model_values: dict,
    tp_size: int,
    policy: str,
    phase: str,
    legacy_columns: list[str],
) -> dict:
    strategy = STRATEGIES[policy]
    max_batch = float(strategy["max_batch_size"])
    max_prefill = float(strategy["max_prefill_tokens"])
    result = {
        **scalar_profile_features(profile),
        **model_values,
        "feature_parallelism_tp": 1,
        "feature_parallelism_pp": 0,
        "feature_parallel_size_log2": math.log2(tp_size),
        "feature_phase_prefill": int(phase == "prefill"),
        "feature_phase_decode": int(phase == "decode"),
        "feature_tp_max_batch_size": int(max_batch),
        "feature_tp_max_prefill_tokens": int(max_prefill),
        "feature_pp_max_microbatch_size": 0,
        "feature_pp_chunk_tokens": 0,
        "feature_pp_page_size": 0,
        "feature_pp_proxy_tensor_count": 0,
        "feature_tp_batch_size_fraction_of_16": max_batch / 16.0,
        "feature_tp_prefill_budget_fraction_of_65536": max_prefill / 65536.0,
        "feature_tp_input_mean_budget_fill": profile["input_mean_capped"]
        * max_batch
        / max_prefill,
        "feature_tp_input_p50_budget_fill": profile["input_p50_capped"]
        * max_batch
        / max_prefill,
        "feature_tp_input_p90_budget_fill": profile["input_p90_capped"]
        * max_batch
        / max_prefill,
        "feature_tp_input_p99_budget_fill": profile["input_p99_capped"]
        * max_batch
        / max_prefill,
        "feature_tp_rps_per_batch_slot": profile["rps"] / max_batch,
        "feature_tp_fano_per_batch_slot": profile["fano_1s"] / max_batch,
        "feature_tp_multichunk_batch_pressure": profile[
            "input_multichunk_fraction"
        ]
        * max_batch,
    }
    for threshold in (1, 8, 16, 32, 64):
        result[f"feature_tp_survival_m_gt_{threshold}_batch_pressure"] = (
            profile[f"survival_m_gt_{threshold}"] * max_batch
        )
    for width in (4, 16, 32):
        result[f"feature_tp_rolling_multichunk_max_{width}_batch_pressure"] = (
            profile[f"rolling_multichunk_fraction_max_{width}"]
            * min(max_batch, float(width))
            / float(width)
        )
    missing_legacy = set(legacy_columns) - set(result)
    if missing_legacy:
        raise RuntimeError(f"legacy features missing: {sorted(missing_legacy)}")
    return result


def inventory_rows(profiles: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for profile in profiles:
        groups[(profile["phase27_role"], profile["segment"])].append(profile)
    rows = []
    for (role, segment), group in sorted(groups.items()):
        counts = [int(row["request_count"]) for row in group]
        rows.append(
            {
                "role": role,
                "segment": segment,
                "profiles": len(group),
                "requests_total": sum(counts),
                "requests_min": min(counts),
                "requests_median": statistics.median(counts),
                "requests_max": max(counts),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    for name in ("profiles", "labels", "dataset", "analysis", "logs"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    contract = json.loads(args.phase29a_summary.read_text())
    features_contract = json.loads(args.feature_contract.read_text())
    phase26a = json.loads(args.phase26a_summary.read_text())
    if contract["status"] != "PASS":
        raise RuntimeError("Phase 29A is not PASS")
    if contract["tp_hfull_label_state_at_freeze"] != "no_phase29_tp_hfull_labels_generated":
        raise RuntimeError("TP labels were not cleanly frozen")
    if phase26a["status"] != "PASS" or phase26a["promoted_label_status"] != TEACHER_STATUS:
        raise RuntimeError("Phase 26A teacher is not promoted")

    raw_manifest_path = args.raw_dir / "source_manifest.json"
    raw_manifest = json.loads(raw_manifest_path.read_text())
    raw_checks = {}
    official = json.loads(
        (Path(__file__).resolve().parents[1] / "experiment-results/phase15_trace_data/source_manifest.json").read_text()
    )
    official_by_name = {row["name"]: row for row in official["sources"]}
    for source in raw_manifest["sources"]:
        path = args.raw_dir / source["name"]
        expected = official_by_name[source["name"]]
        raw_checks[source["name"]] = (
            path.stat().st_size == int(expected["actual_size"])
            and sha256(path) == expected["sha256"]
        )
    if len(raw_checks) != 6 or not all(raw_checks.values()):
        raise RuntimeError(raw_checks)

    phase27_selection = read_csv(args.phase27_windows)
    phase28_selection = read_csv(args.phase28_windows)
    file_by_segment = {
        segment: args.raw_dir / name
        for name, (segment, _) in {**BURST_FILES, **MOONCAKE_FILES}.items()
    }
    arrays = {segment: load_segment(path) for segment, path in file_by_segment.items()}
    profiles = []
    windows = {}
    count_checks = []
    for selected in [*phase27_selection, *phase28_selection]:
        timestamps, inputs, outputs = arrays[selected["segment"]]
        cutoff = int(selected["cutoff_ms"])
        left = int(
            np.searchsorted(timestamps, cutoff - HISTORY_SECONDS * 1000, side="left")
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
        count_checks.append(len(requests) == int(selected["history_count"]))
        profiles.append(profile)
        windows[profile["profile_id"]] = requests

    models = all_model_features(args.model_features)
    if set(models) != set(MODELS):
        raise RuntimeError(set(models))
    legacy_columns = features_contract["legacy_feature_columns"]
    enhanced_columns = features_contract["enhanced_feature_columns"]
    union_columns = list(dict.fromkeys([*legacy_columns, *enhanced_columns]))
    targets = []
    baselines = []
    development_examples = []
    first_confirmation_features = []
    second_confirmation_features = []
    full_batch_checks = []
    compact_batch_checks = []

    for profile in profiles:
        full_requests = windows[profile["profile_id"]]
        compact_requests = pseudo_requests(profile)
        is_phase27 = profile["phase27_role"] != "second_independent_confirmation"
        for model_name in MODELS:
            model, model_values = models[model_name]
            for policy, strategy in STRATEGIES.items():
                full_histograms = (
                    tp_histograms(full_requests, strategy, model) if is_phase27 else None
                )
                compact_histograms = tp_histograms(compact_requests, strategy, model)
                if is_phase27:
                    full_batch_checks.append(
                        sum(len(batch) for batch in tp_batches(full_requests, strategy))
                        == len(full_requests)
                    )
                compact_batch_checks.append(
                    sum(len(batch) for batch in tp_batches(compact_requests, strategy))
                    == len(compact_requests)
                )
                for tp_size in TP_SIZES:
                    for phase in PHASES:
                        h0_hist = normalize(compact_histograms[phase], len(compact_requests))
                        h0 = target_row(
                            {**profile, "request_count": len(compact_requests)},
                            model,
                            tp_size,
                            policy,
                            phase,
                            h0_hist,
                        )
                        h0["label_id"] = h0["label_id"].replace(
                            "/hfull/", "/compact32_h0/"
                        )
                        h0["label_status"] = "PARAMETER_FREE_LOW_DIMENSIONAL_BASELINE"
                        h0["teacher_kind"] = (
                            "compact32_reconstruction_plus_tp_fixed_order_token_budget_batches_v1"
                        )
                        h0["full_window_requests_audit_only"] = profile["request_count"]
                        baselines.append(h0)
                        values = feature_values(
                            profile,
                            model_values,
                            tp_size,
                            policy,
                            phase,
                            legacy_columns,
                        )
                        if set(enhanced_columns) - set(values):
                            raise RuntimeError("enhanced feature missing")
                        identifiers = {
                            "training_id": h0["label_id"].replace(
                                "/compact32_h0/", "/hfull/"
                            ),
                            "profile_id": profile["profile_id"],
                            "role": profile["phase27_role"],
                            "source": profile["source"],
                            "segment": profile["segment"],
                            "window_id": profile["window_id"],
                            "model": model_name,
                            "parallelism": "tp",
                            "parallel_size": tp_size,
                            "policy": policy,
                            "phase": phase,
                        }
                        h0_fields = {
                            "h0_total_calls_per_1000": h0["total_calls_per_1000"],
                            "h0_total_logical_bytes_per_1000": h0[
                                "total_logical_bytes_per_1000"
                            ],
                            "h0_common_reference_cost_us_per_1000": h0[
                                "common_reference_cost_us_per_1000"
                            ],
                            "h0_calls_by_12bin_json": h0["calls_by_12bin_json"],
                            "h0_logical_bytes_by_12bin_json": h0[
                                "logical_bytes_by_12bin_json"
                            ],
                        }
                        feature_row = {
                            **identifiers,
                            **{name: values[name] for name in union_columns},
                            **h0_fields,
                        }
                        if not is_phase27:
                            second_confirmation_features.append(feature_row)
                            continue
                        target_hist = normalize(
                            full_histograms[phase], len(full_requests)
                        )
                        target = target_row(
                            profile,
                            model,
                            tp_size,
                            policy,
                            phase,
                            target_hist,
                        )
                        targets.append(target)
                        if profile["phase27_role"] == "independent_confirmation":
                            first_confirmation_features.append(feature_row)
                        else:
                            development_examples.append(
                                {
                                    **feature_row,
                                    "target_total_calls_per_1000": target[
                                        "total_calls_per_1000"
                                    ],
                                    "target_total_logical_bytes_per_1000": target[
                                        "total_logical_bytes_per_1000"
                                    ],
                                    "target_common_reference_cost_us_per_1000": target[
                                        "common_reference_cost_us_per_1000"
                                    ],
                                    "target_calls_by_12bin_json": target[
                                        "calls_by_12bin_json"
                                    ],
                                    "target_logical_bytes_by_12bin_json": target[
                                        "logical_bytes_by_12bin_json"
                                    ],
                                }
                            )

    development_targets = [
        row for row in targets if row["role"] != "independent_confirmation"
    ]
    first_confirmation_targets = [
        row for row in targets if row["role"] == "independent_confirmation"
    ]
    profile_rows = [
        {**profile, **scalar_profile_features(profile)} for profile in profiles
    ]
    write_csv_gz(args.output_dir / "profiles/low_dimensional_profiles.csv.gz", profile_rows)
    write_csv_gz(
        args.output_dir / "labels/development_hfull_targets.csv.gz",
        development_targets,
    )
    write_csv_gz(
        args.output_dir / "labels/first_confirmation_hfull_targets.csv.gz",
        first_confirmation_targets,
    )
    write_csv_gz(
        args.output_dir / "labels/compact32_h0_baselines.csv.gz", baselines
    )
    write_csv_gz(
        args.output_dir / "dataset/development_examples.csv.gz", development_examples
    )
    write_csv_gz(
        args.output_dir / "dataset/first_confirmation_features.csv.gz",
        first_confirmation_features,
    )
    write_csv_gz(
        args.output_dir / "dataset/second_confirmation_features.csv.gz",
        second_confirmation_features,
    )
    write_csv(args.output_dir / "analysis/profile_inventory.csv", inventory_rows(profiles))
    write_csv(
        args.output_dir / "analysis/label_inventory.csv",
        [
            {
                "artifact": "development_examples",
                "profiles": 42,
                "phase_rows": len(development_examples),
                "contains_target": True,
                "allowed_training_access": True,
            },
            {
                "artifact": "first_confirmation_features",
                "profiles": 18,
                "phase_rows": len(first_confirmation_features),
                "contains_target": False,
                "allowed_training_access": True,
            },
            {
                "artifact": "first_confirmation_hfull_targets",
                "profiles": 18,
                "phase_rows": len(first_confirmation_targets),
                "contains_target": True,
                "allowed_training_access": False,
            },
            {
                "artifact": "second_confirmation_features",
                "profiles": 18,
                "phase_rows": len(second_confirmation_features),
                "contains_target": False,
                "allowed_training_access": True,
            },
            {
                "artifact": "second_confirmation_hfull_targets",
                "profiles": 18,
                "phase_rows": 0,
                "contains_target": False,
                "allowed_training_access": False,
            },
        ],
    )
    write_json(
        args.output_dir / "feature_columns.json",
        {
            "schema_version": "phase29b-tp-feature-columns-v1",
            "saved_feature_count": len(union_columns),
            "legacy_feature_count": len(legacy_columns),
            "enhanced_feature_count": len(enhanced_columns),
            "saved_feature_columns": union_columns,
            "legacy_feature_columns": legacy_columns,
            "enhanced_feature_columns": enhanced_columns,
            "confirmation_target_columns_absent": True,
        },
    )

    role_counts = Counter(profile["phase27_role"] for profile in profiles)
    total_requests = sum(int(profile["request_count"]) for profile in profiles)
    summary = {
        "schema_version": "phase29b-tp-hfull-dataset-v1",
        "status": "PASS",
        "profiles": len(profiles),
        "profile_role_counts": dict(role_counts),
        "full_window_requests": total_requests,
        "models": list(MODELS),
        "tp_sizes": list(TP_SIZES),
        "policies": list(STRATEGIES),
        "saved_feature_columns": len(union_columns),
        "legacy_feature_columns": len(legacy_columns),
        "enhanced_feature_columns": len(enhanced_columns),
        "development_phase_rows": len(development_examples),
        "first_confirmation_phase_rows": len(first_confirmation_features),
        "second_confirmation_feature_rows": len(second_confirmation_features),
        "hfull_target_phase_rows": len(targets),
        "h0_baseline_phase_rows": len(baselines),
        "teacher_status": TEACHER_STATUS,
        "teacher_kind": TEACHER_KIND,
        "second_confirmation_hfull_targets_generated": False,
        "raw_source_checks": raw_checks,
        "inputs": {
            "phase27_windows_sha256": sha256(args.phase27_windows),
            "phase28_windows_sha256": sha256(args.phase28_windows),
            "phase29a_summary_sha256": sha256(args.phase29a_summary),
            "feature_contract_sha256": sha256(args.feature_contract),
            "model_features_sha256": sha256(args.model_features),
            "phase26a_summary_sha256": sha256(args.phase26a_summary),
            "raw_manifest_sha256": sha256(raw_manifest_path),
        },
    }
    write_json(args.output_dir / "summary.json", summary)

    target_names = {
        "target_total_calls_per_1000",
        "target_total_logical_bytes_per_1000",
        "target_calls_by_12bin_json",
        "target_logical_bytes_by_12bin_json",
    }
    request_fields = {
        "input_lens",
        "output_lens",
        "requests_json",
        "full_request_list",
    }
    checks = {
        "raw_source_hashes_6_of_6": len(raw_checks) == 6
        and all(raw_checks.values()),
        "profiles_78_roles_30_12_18_18": len(profiles) == 78
        and role_counts
        == Counter(
            {
                "development_train": 30,
                "development_validation": 12,
                "independent_confirmation": 18,
                "second_independent_confirmation": 18,
            }
        ),
        "history_counts_match_78_of_78": all(count_checks),
        "models_three_tp_sizes_three_policies_three": set(models) == set(MODELS)
        and len(TP_SIZES) == 3
        and len(STRATEGIES) == 3,
        "development_rows_2268": len(development_examples) == 2268,
        "first_confirmation_features_and_targets_972": len(
            first_confirmation_features
        )
        == 972
        and len(first_confirmation_targets) == 972,
        "second_confirmation_features_972_targets_zero": len(
            second_confirmation_features
        )
        == 972
        and not summary["second_confirmation_hfull_targets_generated"],
        "hfull_targets_3240": len(targets) == 3240,
        "h0_baselines_4212": len(baselines) == 4212,
        "full_batch_partition_checks_540_of_540": len(full_batch_checks) == 540
        and all(full_batch_checks),
        "compact_batch_partition_checks_702_of_702": len(compact_batch_checks)
        == 702
        and all(compact_batch_checks),
        "features_118_saved_55_legacy_113_enhanced": len(union_columns) == 118
        and len(legacy_columns) == 55
        and len(enhanced_columns) == 113,
        "confirmation_features_have_no_targets": not (
            target_names & set(first_confirmation_features[0])
        )
        and not (target_names & set(second_confirmation_features[0])),
        "no_complete_request_lists_saved": not (
            request_fields & set(profile_rows[0])
        )
        and not (request_fields & set(development_examples[0])),
        "teacher_promoted_by_phase26a": phase26a["promoted_labels"] == 1296
        and phase26a["promoted_label_status"] == TEACHER_STATUS,
    }
    audit = {
        "schema_version": "phase29b-tp-hfull-dataset-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    write_json(args.output_dir / "audit_summary.json", audit)
    if audit["status"] != "PASS":
        raise RuntimeError(audit)

    (args.output_dir / "README.md").write_text(
        f"""# Phase 29B：三模型TP对齐Hfull数据集

本阶段把Phase 27/28已经冻结的同一批历史窗口扩展到三个TP模型。60个Phase 27窗口生成
{len(targets):,}条Hfull phase labels；其中30/12个开发训练与验证画像形成
{len(development_examples):,}条带target样本，18个第一确认画像的{len(first_confirmation_features):,}
条feature与同数量target物理隔离。Phase 28的18个第二确认画像只生成{len(second_confirmation_features):,}
条无target feature和compact32 H0，尚未生成第二确认Hfull真值，供后续预测先冻结。

覆盖DeepSeek-V2-Lite、Qwen3-8B、Qwen3-30B-A3B，TP2/4/8，latency/balanced/throughput和
prefill/decode。Hfull teacher沿用Phase 26A四个跨模型/TP/策略GPU sentinel精确验证的结构公式；
无需为每个新窗口重跑GPU。每条样本按1000请求归一化，输出保持TP原生4 KiB–512 MiB 12桶。

保存118列特征并冻结两个视图：Phase 26旧55列legacy和Phase 29增强113列。完整请求数组只在
构建器内存中生成Hfull，没有写入profiles、dataset、labels或Git。最终预测器输入仍是低维
历史画像、模型结构、固定TP size、固定策略、phase和H0。
"""
    )
    (args.output_dir / "DONE").write_text("PASS\n")
    write_json(
        args.output_dir / "logs/build.log",
        {
            "schema_version": "phase29b-build-log-v1",
            "status": "PASS",
            "profiles": len(profiles),
            "full_window_requests": total_requests,
            "hfull_target_phase_rows": len(targets),
            "second_confirmation_targets_generated": False,
            "full_request_lists_saved": False,
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
                "profiles": len(profiles),
                "full_window_requests": total_requests,
                "hfull_target_phase_rows": len(targets),
                "development_rows": len(development_examples),
                "first_confirmation_rows": len(first_confirmation_features),
                "second_confirmation_feature_rows": len(second_confirmation_features),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
