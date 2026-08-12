#!/usr/bin/env python3
"""Build three-model TP/PP Hfull development data and target-free fixed features."""

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
    PP_BIN_EDGES,
    TP_BIN_EDGES,
    normalize,
    tp_histograms,
)
from build_phase25b_pp_scheduler_teacher import PP_PROXY_COUNT, simulate_scheduler
from build_phase27b_pp_hfull_dataset import (
    HISTORY_SECONDS,
    PHASES,
    scalar_profile_features,
    summarize_profile,
    training_features as pp_training_features,
)
from build_phase29b_tp_hfull_dataset import (
    MODELS,
    STRATEGIES,
    TP_SIZES,
    all_model_features,
    feature_values as tp_feature_values,
)
from prepare_phase15_trace_windows import BURST_FILES, MOONCAKE_FILES, load_segment


PP_SIZES = (2, 4, 8)
MICROBATCHES = (1, 4, 16)
DEVELOPMENT_ROLES = {"development_train", "development_validation"}
FIXED_ROLE = "fixed_prediction"
COMMON_REFERENCE_LAUNCH_US = 5.0
COMMON_REFERENCE_BANDWIDTH_GBPS = 100.0


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument(
        "--selection",
        type=Path,
        default=root / "experiment-results/phase31a_known_model_convergence_contract/selection/selected_windows.csv",
    )
    parser.add_argument(
        "--contract-summary",
        type=Path,
        default=root / "experiment-results/phase31a_known_model_convergence_contract/summary.json",
    )
    parser.add_argument(
        "--model-features",
        type=Path,
        default=root / "experiment-results/phase16_model_features/model_features.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase31b_known_model_hfull_dataset",
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


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def bin_vectors(histogram: dict[int, float], edges: np.ndarray) -> tuple[list[float], list[float]]:
    calls = np.zeros(12, dtype=np.float64)
    logical_bytes = np.zeros(12, dtype=np.float64)
    for payload, count in histogram.items():
        index = int(np.clip(np.searchsorted(edges, payload, side="right") - 1, 0, 11))
        calls[index] += count
        logical_bytes[index] += payload * count
    return calls.tolist(), logical_bytes.tolist()


def reference_cost(calls: list[float], logical_bytes: list[float]) -> float:
    return float(
        COMMON_REFERENCE_LAUNCH_US * sum(calls)
        + sum(logical_bytes) / (COMMON_REFERENCE_BANDWIDTH_GBPS * 1e9) * 1e6
    )


def histogram_fields(histogram: dict[int, float], edges: np.ndarray) -> dict:
    calls, logical_bytes = bin_vectors(histogram, edges)
    return {
        "total_calls_per_1000": float(sum(calls)),
        "total_logical_bytes_per_1000": float(sum(logical_bytes)),
        "common_reference_cost_us_per_1000": reference_cost(calls, logical_bytes),
        "calls_by_12bin_json": json.dumps(calls, separators=(",", ":")),
        "logical_bytes_by_12bin_json": json.dumps(logical_bytes, separators=(",", ":")),
        "exact_calls_histogram_per_1000_json": json.dumps({str(key): value for key, value in sorted(histogram.items())}, separators=(",", ":")),
    }


def pp_histograms(
    requests: list[tuple[int, int]], pp_size: int, microbatch: int, bytes_per_token: int
) -> tuple[dict[str, dict[int, float]], dict]:
    simulated = simulate_scheduler(requests, pp_size=pp_size, max_microbatch=microbatch)
    scale = 1000.0 / len(requests)
    histograms = {
        phase: {
            int(active_tokens) * bytes_per_token: float(events * PP_PROXY_COUNT) * scale
            for active_tokens, events in sorted(simulated.event_histograms[phase].items())
        }
        for phase in PHASES
    }
    audit = {
        "all_requests_complete": simulated.all_requests_complete,
        "prefill_token_mass": simulated.prefill_token_mass,
        "decode_token_mass": simulated.decode_token_mass,
    }
    return histograms, audit


def identifiers(
    profile: dict,
    *,
    model: str,
    parallelism: str,
    parallel_size: int,
    policy: str,
    phase: str,
) -> dict:
    return {
        "example_id": f"{parallelism}/{model}/p{parallel_size}/{policy}/{profile['profile_id']}/{phase}",
        "profile_id": profile["profile_id"],
        "split_role": profile["split_role"],
        "source": profile["source"],
        "segment": profile["segment"],
        "window_id": profile["window_id"],
        "model": model,
        "parallelism": parallelism,
        "parallel_size": parallel_size,
        "policy": policy,
        "phase": phase,
    }


def feature_safe_profile(profile: dict) -> dict:
    """Remove split metadata before calling reused Phase27/29 feature builders."""
    return {key: value for key, value in profile.items() if key != "split_role"}


def prefixed_fields(prefix: str, values: dict) -> dict:
    return {
        f"{prefix}_total_calls_per_1000": values["total_calls_per_1000"],
        f"{prefix}_total_logical_bytes_per_1000": values["total_logical_bytes_per_1000"],
        f"{prefix}_common_reference_cost_us_per_1000": values["common_reference_cost_us_per_1000"],
        f"{prefix}_calls_by_12bin_json": values["calls_by_12bin_json"],
        f"{prefix}_logical_bytes_by_12bin_json": values["logical_bytes_by_12bin_json"],
    }


def inventory(profiles: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for profile in profiles:
        grouped[(profile["split_role"], profile["segment"])].append(profile)
    output = []
    for (role, segment), rows in sorted(grouped.items()):
        counts = [int(row["request_count"]) for row in rows]
        output.append(
            {
                "split_role": role,
                "segment": segment,
                "profiles": len(rows),
                "requests_total": sum(counts),
                "requests_min": min(counts),
                "requests_median": statistics.median(counts),
                "requests_max": max(counts),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    for name in ("profiles", "dataset", "labels", "baselines", "analysis", "logs"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)

    contract = json.loads(args.contract_summary.read_text())
    if contract["status"] != "PASS" or contract["target_state_at_freeze"] != "no_phase31_hfull_targets_generated":
        raise RuntimeError("Phase31A contract is not a target-free PASS")
    selection = read_csv(args.selection)
    if len(selection) != 59:
        raise ValueError(f"expected 59 profiles, got {len(selection)}")

    manifest_path = args.raw_dir / "source_manifest.json"
    raw_manifest = json.loads(manifest_path.read_text())
    raw_checks = {}
    for row in raw_manifest["sources"]:
        path = args.raw_dir / row["name"]
        raw_checks[row["name"]] = path.stat().st_size == int(row["actual_size"]) and sha256(path) == row["sha256"]
    if len(raw_checks) != 6 or not all(raw_checks.values()):
        raise RuntimeError({"raw_source_checks": raw_checks})

    file_by_segment = {
        segment: args.raw_dir / name
        for name, (segment, _) in {**BURST_FILES, **MOONCAKE_FILES}.items()
    }
    arrays = {segment: load_segment(path) for segment, path in file_by_segment.items()}
    profiles = []
    request_windows: dict[str, list[tuple[int, int]]] = {}
    count_matches = []
    for selected in selection:
        timestamps, inputs, outputs = arrays[selected["segment"]]
        cutoff = int(selected["cutoff_ms"])
        left = int(np.searchsorted(timestamps, cutoff - HISTORY_SECONDS * 1000, side="left"))
        right = int(np.searchsorted(timestamps, cutoff, side="left"))
        compatible = {**selected, "phase27_profile_id": selected["profile_id"], "phase27_role": selected["role"]}
        profile, requests = summarize_profile(compatible, timestamps[left:right], inputs[left:right], outputs[left:right])
        profile["split_role"] = profile.pop("phase27_role")
        count_matches.append(len(requests) == int(selected["history_count"]))
        profiles.append(profile)
        request_windows[profile["profile_id"]] = requests

    model_map = all_model_features(args.model_features)
    if set(model_map) != set(MODELS):
        raise ValueError(f"unexpected models: {set(model_map)}")
    target_rows = []
    baseline_rows = []
    tp_development = []
    pp_development = []
    tp_fixed = []
    pp_fixed = []
    simulation_checks = []

    for profile in profiles:
        full_requests = request_windows[profile["profile_id"]]
        compact_requests = pseudo_requests(profile)
        is_development = profile["split_role"] in DEVELOPMENT_ROLES
        feature_profile = feature_safe_profile(profile)

        for model_name in MODELS:
            model_meta, model_values = model_map[model_name]
            for tp_size in TP_SIZES:
                for policy, strategy in STRATEGIES.items():
                    full_by_phase = (
                        {phase: normalize(hist, len(full_requests)) for phase, hist in tp_histograms(full_requests, strategy, model_meta).items()}
                        if is_development
                        else None
                    )
                    h0_by_phase = {
                        phase: normalize(hist, len(compact_requests))
                        for phase, hist in tp_histograms(compact_requests, strategy, model_meta).items()
                    }
                    for phase in PHASES:
                        ids = identifiers(profile, model=model_name, parallelism="tp", parallel_size=tp_size, policy=policy, phase=phase)
                        features = tp_feature_values(feature_profile, model_values, tp_size, policy, phase, [])
                        h0 = histogram_fields(h0_by_phase[phase], TP_BIN_EDGES)
                        base = {**ids, **features, **prefixed_fields("h0", h0)}
                        baseline_rows.append({**ids, **h0, "baseline_kind": "compact32_h0"})
                        if is_development:
                            target = histogram_fields(full_by_phase[phase], TP_BIN_EDGES)
                            target_rows.append({**ids, **target, "teacher_kind": "tp_full_window_fixed_draining_structural_teacher"})
                            tp_development.append({**base, **prefixed_fields("target", target)})
                        else:
                            tp_fixed.append(base)

            bytes_per_token = int(model_meta["payload_bytes_per_active_token_prior"])
            for pp_size in PP_SIZES:
                for microbatch in MICROBATCHES:
                    full_pp = None
                    full_audit = None
                    if is_development:
                        full_pp, full_audit = pp_histograms(full_requests, pp_size, microbatch, bytes_per_token)
                    h0_pp, h0_audit = pp_histograms(compact_requests, pp_size, microbatch, bytes_per_token)
                    simulation_checks.append(
                        {
                            "profile_id": profile["profile_id"],
                            "model": model_name,
                            "pp_size": pp_size,
                            "microbatch": microbatch,
                            "split_role": profile["split_role"],
                            "target_complete": None if full_audit is None else full_audit["all_requests_complete"],
                            "target_prefill_mass_exact": None if full_audit is None else full_audit["prefill_token_mass"] == sum(row[0] for row in full_requests),
                            "target_decode_mass_exact": None if full_audit is None else full_audit["decode_token_mass"] == sum(row[1] - 1 for row in full_requests),
                            "h0_complete": h0_audit["all_requests_complete"],
                            "h0_prefill_mass_exact": h0_audit["prefill_token_mass"] == sum(row[0] for row in compact_requests),
                            "h0_decode_mass_exact": h0_audit["decode_token_mass"] == sum(row[1] - 1 for row in compact_requests),
                        }
                    )
                    for phase in PHASES:
                        policy = f"mb{microbatch}"
                        ids = identifiers(profile, model=model_name, parallelism="pp", parallel_size=pp_size, policy=policy, phase=phase)
                        features = pp_training_features(feature_profile, model_values, pp_size, microbatch, phase)
                        h0 = histogram_fields(h0_pp[phase], PP_BIN_EDGES)
                        base = {**ids, **features, **prefixed_fields("h0", h0)}
                        baseline_rows.append({**ids, **h0, "baseline_kind": "compact32_h0"})
                        if is_development:
                            target = histogram_fields(full_pp[phase], PP_BIN_EDGES)
                            target_rows.append({**ids, **target, "teacher_kind": "pp_scheduler_faithful_full_window_teacher_adapted_by_model_hidden_size"})
                            pp_development.append({**base, **prefixed_fields("target", target)})
                        else:
                            pp_fixed.append(base)

    profile_rows = [
        {**profile, **scalar_profile_features(feature_safe_profile(profile))}
        for profile in profiles
    ]
    write_csv_gz(args.output_dir / "profiles/low_dimensional_profiles.csv.gz", profile_rows)
    write_csv_gz(args.output_dir / "labels/development_hfull_targets.csv.gz", target_rows)
    write_csv_gz(args.output_dir / "baselines/compact32_h0.csv.gz", baseline_rows)
    write_csv_gz(args.output_dir / "dataset/tp_development_examples.csv.gz", tp_development)
    write_csv_gz(args.output_dir / "dataset/pp_development_examples.csv.gz", pp_development)
    write_csv_gz(args.output_dir / "dataset/tp_fixed_prediction_features.csv.gz", tp_fixed)
    write_csv_gz(args.output_dir / "dataset/pp_fixed_prediction_features.csv.gz", pp_fixed)
    write_csv(args.output_dir / "analysis/profile_inventory.csv", inventory(profiles))
    write_csv(args.output_dir / "analysis/simulation_checks.csv", simulation_checks)

    tp_features = [name for name in tp_development[0] if name.startswith("feature_")]
    pp_features = [name for name in pp_development[0] if name.startswith("feature_")]
    write_json(args.output_dir / "feature_columns.json", {"tp": tp_features, "pp": pp_features})

    role_counts = Counter(profile["split_role"] for profile in profiles)
    simulation_exact = all(
        value is None or bool(value)
        for row in simulation_checks
        for key, value in row.items()
        if key.endswith("_complete") or key.endswith("_exact")
    )
    checks = {
        "phase31a_contract_pass": contract["status"] == "PASS",
        "raw_source_hashes_6_of_6": len(raw_checks) == 6 and all(raw_checks.values()),
        "profiles_59_roles_39_10_10": role_counts == Counter({"development_train": 39, "development_validation": 10, "fixed_prediction": 10}),
        "history_counts_match_59_of_59": all(count_matches),
        "models_exact_three": set(model_map) == set(MODELS),
        "development_targets_5292": len(target_rows) == 49 * 2 * 3 * 3 * 3 * 2,
        "tp_development_rows_2646": len(tp_development) == 49 * 3 * 3 * 3 * 2,
        "pp_development_rows_2646": len(pp_development) == 49 * 3 * 3 * 3 * 2,
        "fixed_features_540_each": len(tp_fixed) == len(pp_fixed) == 10 * 3 * 3 * 3 * 2,
        "fixed_features_have_no_target_columns": not any(name.startswith("target_") for name in set(tp_fixed[0]) | set(pp_fixed[0])),
        "split_role_not_exposed_as_feature": "feature_profile_split_role" not in set(tp_development[0]) | set(pp_development[0]) | set(tp_fixed[0]) | set(pp_fixed[0]),
        "full_request_lists_not_saved": not any(name in set(profile_rows[0]) | set(tp_development[0]) | set(pp_development[0]) for name in {"input_lens", "output_lens", "full_request_list", "requests"}),
        "pp_scheduler_mass_and_completion_exact": simulation_exact,
        "fixed_hfull_targets_not_generated": all(row["split_role"] != FIXED_ROLE for row in target_rows),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = {
        "schema_version": "phase31b-known-model-hfull-dataset-v1",
        "status": status,
        "profiles": len(profiles),
        "role_counts": dict(role_counts),
        "full_window_requests_development": sum(profile["request_count"] for profile in profiles if profile["split_role"] in DEVELOPMENT_ROLES),
        "full_window_requests_fixed_features_only": sum(profile["request_count"] for profile in profiles if profile["split_role"] == FIXED_ROLE),
        "models": list(MODELS),
        "target_phase_rows": len(target_rows),
        "tp_development_rows": len(tp_development),
        "pp_development_rows": len(pp_development),
        "tp_fixed_feature_rows": len(tp_fixed),
        "pp_fixed_feature_rows": len(pp_fixed),
        "tp_feature_columns": len(tp_features),
        "pp_feature_columns": len(pp_features),
        "fixed_target_state": "not_generated",
        "teacher_contract": {
            "tp": "Phase26A GPU-validated structural formula sentinels",
            "pp": "Phase25B/25C scheduler-faithful event simulator; active-token events deterministically adapted by each model hidden-size bytes/token",
        },
        "inputs": {
            "raw_manifest_sha256": sha256(manifest_path),
            "selection_sha256": sha256(args.selection),
            "contract_summary_sha256": sha256(args.contract_summary),
            "model_features_sha256": sha256(args.model_features),
        },
        "checks": checks,
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "audit_summary.json", {"schema_version": "phase31b-dataset-audit-v1", "status": status, "checks": checks, "raw_source_checks": raw_checks})
    (args.output_dir / "README.md").write_text(f"""# Phase 31B：三模型 TP/PP Hfull 开发数据

本阶段使用Phase31A冻结的59个请求级不重叠画像。39个训练和10个验证画像生成Hfull teacher；10个固定预测画像只生成低维特征和compact32 H0，尚未生成Hfull真值。

## 数据规模

- 三个模型：DeepSeek-V2-Lite、Qwen3-8B、Qwen3-30B-A3B；
- TP：TP2/4/8 × latency/balanced/throughput × prefill/decode；
- PP：PP2/4/8 × MB1/4/16 × prefill/decode；
- 开发Hfull标签：{len(target_rows):,}条phase rows，TP/PP各{len(tp_development):,}条；
- 固定预测特征：TP/PP各{len(tp_fixed):,}条，不含任何`target_`字段；
- 完整请求列表只在构建进程内存中用于低维画像聚合和teacher，不落盘、不进入Git。

## teacher口径

TP继续使用Phase26A GPU哨兵验证过的fixed-draining结构公式。PP先由Phase25B/25C验证过的scheduler-faithful模拟器生成模型无关active-token事件，再按三个模型的hidden-size/dtype确定性映射payload；这属于当前源码与调度合同内的结构teacher，不声称三个模型的全部PP配置均已逐项GPU实测。

下一步只允许读取开发数据训练`H0 + DNN residual`，先冻结固定预测文件和SHA，再生成固定预测Hfull真值。
""")
    write_json(args.output_dir / "logs/build.log", {"event": "phase31b_dataset_built", "status": status, "profiles": len(profiles), "target_rows": len(target_rows), "fixed_targets_generated": False})
    (args.output_dir / "DONE").write_text(f"{status}\n")
    manifest = [
        f"{sha256(path)}  {path.relative_to(args.output_dir)}"
        for path in sorted(args.output_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.sha256"
    ]
    (args.output_dir / "manifest.sha256").write_text("\n".join(manifest) + "\n")
    if status != "PASS":
        raise RuntimeError(checks)
    print(json.dumps({"status": status, "profiles": len(profiles), "target_rows": len(target_rows), "tp_features": len(tp_features), "pp_features": len(pp_features)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
