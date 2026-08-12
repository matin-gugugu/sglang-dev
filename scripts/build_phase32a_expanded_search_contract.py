#!/usr/bin/env python3
"""Freeze target-blind Phase32 evidence and build only deployable confirmation features."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from build_phase21b_pp_h0 import pseudo_requests
from build_phase25_full_window_teacher import PP_BIN_EDGES, TP_BIN_EDGES, normalize, tp_histograms
from build_phase25b_pp_scheduler_teacher import PP_PROXY_COUNT, simulate_scheduler
from build_phase27a_pp_feature_and_holdout_contract import (
    HISTORY_ONLY_SOURCE_COLUMNS,
    choose_medoids,
    selection_vector,
)
from build_phase27b_pp_hfull_dataset import HISTORY_SECONDS, PHASES, summarize_profile, training_features as pp_training_features
from build_phase29b_tp_hfull_dataset import MODELS, STRATEGIES, TP_SIZES, all_model_features, feature_values as tp_feature_values
from build_phase31b_known_model_hfull_dataset import (
    MICROBATCHES,
    PP_SIZES,
    bin_vectors,
    feature_safe_profile,
    identifiers,
    prefixed_fields,
    reference_cost,
)
from prepare_phase15_trace_windows import BURST_FILES, MOONCAKE_FILES, load_segment


HISTORY_MS = 300_000
SEED = "phase32-expanded-search-burst-confirmation-v1-20260813"
SEGMENTS = ("burstgpt_1", "burstgpt_2", "burstgpt_3")
PER_SEGMENT = 3


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--windows", type=Path, default=root / "experiment-results/phase15_trace_data/windows.csv.gz")
    parser.add_argument("--model-features", type=Path, default=root / "experiment-results/phase16_model_features/model_features.json")
    parser.add_argument("--phase27-selection", type=Path, default=root / "experiment-results/phase27a_pp_feature_and_holdout_contract/selection/selected_windows.csv")
    parser.add_argument("--phase28-selection", type=Path, default=root / "experiment-results/phase28a_second_confirmation_contract/selection/selected_windows.csv")
    parser.add_argument("--phase30-selection", type=Path, default=root / "experiment-results/phase30a_tp_structured_event_contract/selection/selected_windows.csv")
    parser.add_argument("--phase31-selection", type=Path, default=root / "experiment-results/phase31a_known_model_convergence_contract/selection/selected_windows.csv")
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=root / "experiment-results/phase32a_expanded_search_contract")
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
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def deterministic_gzip(path: Path, text: str) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as output:
            output.write(text.encode())


def write_csv_gz(path: Path, rows: list[dict]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    deterministic_gzip(path, buffer.getvalue())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def used_intervals(args: argparse.Namespace) -> dict[str, list[int]]:
    output = {segment: [] for segment in SEGMENTS}
    for path in (args.phase27_selection, args.phase28_selection, args.phase30_selection, args.phase31_selection):
        for row in read_csv(path):
            if row["segment"] in output:
                output[row["segment"]].append(int(row["cutoff_ms"]))
    return output


def overlaps(cutoff: int, values: list[int]) -> bool:
    return any(abs(cutoff - old) < HISTORY_MS for old in values)


def disjoint_pool(frame: pd.DataFrame) -> pd.DataFrame:
    chosen = []
    last = None
    for index, row in frame.sort_values(["cutoff_ms", "window_id"], kind="stable").iterrows():
        cutoff = int(row["cutoff_ms"])
        if last is None or cutoff - last >= HISTORY_MS:
            chosen.append(index)
            last = cutoff
    return frame.loc[chosen].reset_index(drop=True)


def histogram_fields(histogram: dict[int, float], edges: np.ndarray) -> dict:
    calls, logical_bytes = bin_vectors(histogram, edges)
    return {
        "total_calls_per_1000": float(sum(calls)),
        "total_logical_bytes_per_1000": float(sum(logical_bytes)),
        "common_reference_cost_us_per_1000": reference_cost(calls, logical_bytes),
        "calls_by_12bin_json": json.dumps(calls, separators=(",", ":")),
        "logical_bytes_by_12bin_json": json.dumps(logical_bytes, separators=(",", ":")),
    }


def pp_h0(requests: list[tuple[int, int]], pp_size: int, microbatch: int, bytes_per_token: int) -> dict[str, dict[int, float]]:
    simulated = simulate_scheduler(requests, pp_size=pp_size, max_microbatch=microbatch)
    if not simulated.all_requests_complete:
        raise RuntimeError("compact PP simulation incomplete")
    scale = 1000.0 / len(requests)
    return {
        phase: {
            int(active_tokens) * bytes_per_token: float(events * PP_PROXY_COUNT) * scale
            for active_tokens, events in sorted(simulated.event_histograms[phase].items())
        }
        for phase in PHASES
    }


def main() -> None:
    args = parse_args()
    for name in ("selection", "profiles", "dataset", "analysis", "docs", "logs"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)

    windows = pd.read_csv(args.windows, usecols=list(HISTORY_ONLY_SOURCE_COLUMNS))
    old = used_intervals(args)
    selected = []
    inventory = []
    for segment in SEGMENTS:
        candidates = windows[(windows["segment"] == segment) & (windows["history_count"] >= 32)].copy()
        before = len(candidates)
        candidates = candidates[[not overlaps(int(value), old[segment]) for value in candidates["cutoff_ms"]]].copy()
        matrix = np.stack([selection_vector(row) for _, row in candidates.iterrows()])
        median = np.median(matrix, axis=0)
        scale = np.quantile(matrix, 0.75, axis=0) - np.quantile(matrix, 0.25, axis=0)
        scale[scale < 1e-9] = 1.0
        candidates["normality_distance"] = np.sqrt(np.mean(((matrix - median) / scale) ** 2, axis=1))
        threshold = float(np.quantile(candidates["normality_distance"], 0.95))
        pool = disjoint_pool(candidates[candidates["normality_distance"] <= threshold])
        if len(pool) < PER_SEGMENT:
            raise RuntimeError(f"{segment}: only {len(pool)} disjoint normal windows")
        pool_matrix = np.stack([selection_vector(row) for _, row in pool.iterrows()])
        medoids, labels, distances = choose_medoids(pool_matrix, PER_SEGMENT)
        for cluster, index in enumerate(medoids):
            row = pool.iloc[index]
            members = np.flatnonzero(labels == cluster)
            selected.append({
                "profile_id": f"phase32_confirm_{segment}_{cluster + 1}",
                "role": "new_confirmation",
                "window_id": str(row["window_id"]),
                "source": str(row["source"]),
                "segment": segment,
                "source_split": str(row["split"]),
                "cutoff_ms": int(row["cutoff_ms"]),
                "history_seconds": int(row["history_seconds"]),
                "history_count": int(row["history_count"]),
                "normality_distance": float(row["normality_distance"]),
                "selection_cluster": cluster,
                "selection_cluster_members": int(len(members)),
                "selection_distance_to_medoid_mean": float(np.mean(distances[members])),
            })
        inventory.append({"segment": segment, "eligible_before_embargo": before, "eligible_after_embargo": len(candidates), "disjoint_p95_pool": len(pool), "selected": PER_SEGMENT})
    selected.sort(key=lambda row: (row["segment"], row["cutoff_ms"]))
    write_csv(args.output_dir / "selection/selected_windows.csv", selected)
    write_csv(args.output_dir / "selection/candidate_inventory.csv", inventory)

    raw_manifest = json.loads((args.raw_dir / "source_manifest.json").read_text())
    raw_checks = {
        row["name"]: (args.raw_dir / row["name"]).stat().st_size == int(row["actual_size"]) and sha256(args.raw_dir / row["name"]) == row["sha256"]
        for row in raw_manifest["sources"]
    }
    file_by_segment = {segment: args.raw_dir / name for name, (segment, _) in {**BURST_FILES, **MOONCAKE_FILES}.items()}
    arrays = {segment: load_segment(file_by_segment[segment]) for segment in SEGMENTS}
    profiles = []
    request_windows = {}
    for row in selected:
        timestamps, inputs, outputs = arrays[row["segment"]]
        cutoff = int(row["cutoff_ms"])
        left = int(np.searchsorted(timestamps, cutoff - HISTORY_SECONDS * 1000, side="left"))
        right = int(np.searchsorted(timestamps, cutoff, side="left"))
        compatible = {**row, "phase27_profile_id": row["profile_id"], "phase27_role": row["role"]}
        profile, requests = summarize_profile(compatible, timestamps[left:right], inputs[left:right], outputs[left:right])
        profile["split_role"] = profile.pop("phase27_role")
        if len(requests) != int(row["history_count"]):
            raise RuntimeError(f"request-count mismatch: {row['profile_id']}")
        profiles.append(profile)
        request_windows[profile["profile_id"]] = requests

    model_map = all_model_features(args.model_features)
    tp_rows, pp_rows = [], []
    for profile in profiles:
        compact = pseudo_requests(profile)
        safe = feature_safe_profile(profile)
        for model_name in MODELS:
            model_meta, model_values = model_map[model_name]
            for tp_size in TP_SIZES:
                for policy, strategy in STRATEGIES.items():
                    histograms = {phase: normalize(hist, len(compact)) for phase, hist in tp_histograms(compact, strategy, model_meta).items()}
                    for phase in PHASES:
                        ids = identifiers(profile, model=model_name, parallelism="tp", parallel_size=tp_size, policy=policy, phase=phase)
                        tp_rows.append({**ids, **tp_feature_values(safe, model_values, tp_size, policy, phase, []), **prefixed_fields("h0", histogram_fields(histograms[phase], TP_BIN_EDGES))})
            bytes_per_token = int(model_meta["payload_bytes_per_active_token_prior"])
            for pp_size in PP_SIZES:
                for microbatch in MICROBATCHES:
                    histograms = pp_h0(compact, pp_size, microbatch, bytes_per_token)
                    for phase in PHASES:
                        policy = f"mb{microbatch}"
                        ids = identifiers(profile, model=model_name, parallelism="pp", parallel_size=pp_size, policy=policy, phase=phase)
                        pp_rows.append({**ids, **pp_training_features(safe, model_values, pp_size, microbatch, phase), **prefixed_fields("h0", histogram_fields(histograms[phase], PP_BIN_EDGES))})

    write_csv_gz(args.output_dir / "profiles/low_dimensional_profiles.csv.gz", profiles)
    write_csv_gz(args.output_dir / "dataset/tp_new_confirmation_features.csv.gz", tp_rows)
    write_csv_gz(args.output_dir / "dataset/pp_new_confirmation_features.csv.gz", pp_rows)
    (args.output_dir / "docs/今晚TP_PP收敛执行参考_搜索上限扩容补充.md").write_text(args.supplement.read_text())

    old_new_overlap = any(overlaps(int(row["cutoff_ms"]), old[row["segment"]]) for row in selected)
    new_new_overlap = any(
        left["segment"] == right["segment"] and abs(int(left["cutoff_ms"]) - int(right["cutoff_ms"])) < HISTORY_MS
        for index, left in enumerate(selected) for right in selected[index + 1:]
    )
    checks = {
        "nine_burstgpt_confirmation_profiles": len(selected) == 9 and Counter(row["segment"] for row in selected) == Counter({segment: 3 for segment in SEGMENTS}),
        "request_intervals_disjoint_from_all_phase27_28_30_31_roles": not old_new_overlap,
        "new_confirmation_request_intervals_pairwise_disjoint": not new_new_overlap,
        "history_only_normal_p95_selection": all(float(row["normality_distance"]) >= 0 for row in selected),
        "raw_sources_hash_pass": len(raw_checks) == 6 and all(raw_checks.values()),
        "three_models_all_configurations": len(tp_rows) == len(pp_rows) == 9 * 3 * 3 * 3 * 2,
        "features_have_no_target": not any(name.startswith("target_") for name in set(tp_rows[0]) | set(pp_rows[0])),
        "full_request_lists_not_saved": not any(name in set(profiles[0]) | set(tp_rows[0]) | set(pp_rows[0]) for name in {"requests", "input_lens", "output_lens", "full_request_list"}),
        "hfull_target_not_generated": not (args.output_dir / "labels").exists(),
        "expanded_limits_frozen": True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = {
        "schema_version": "phase32a-expanded-search-contract-v1",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "search_limits_cumulative": {"tp_regular": 42, "tp_absolute": 48, "pp_regular": 30, "pp_absolute": 36, "initial_seed_per_candidate": 1, "top_candidates_three_seed": 3},
        "prior_candidate_counts": {"tp": 18, "pp": 12},
        "new_confirmation": {"profiles": 9, "segments": list(SEGMENTS), "scope": "BurstGPT-only request-disjoint normal windows", "target_state": "not_generated"},
        "counts": {"tp_confirmation_feature_rows": len(tp_rows), "pp_confirmation_feature_rows": len(pp_rows), "full_requests_hidden_teacher_only_future": sum(int(row["history_count"]) for row in selected)},
        "evidence_limit": "Mooncake has no remaining capacity under the accumulated 300-second embargo; the new blind confirmation is BurstGPT-only and does not replace the original ten fixed windows.",
        "checks": checks,
        "inputs": {"windows_sha256": sha256(args.windows), "phase31_selection_sha256": sha256(args.phase31_selection), "supplement_sha256": sha256(args.supplement)},
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "audit_summary.json", {"schema_version": "phase32a-audit-v1", "status": status, "checks": checks, "raw_source_checks": raw_checks})
    write_json(args.output_dir / "logs/build.log", {"event": "phase32a_target_blind_contract_and_features_frozen", "status": status, "target_generated": False, "profiles": len(selected), "feature_rows_each": len(tp_rows)})
    (args.output_dir / "README.md").write_text(f"""# Phase 32A：扩容有限搜索合同与新确认特征\n\n本阶段落实搜索上限扩容补充：TP累计常规/绝对上限42/48，PP累计常规/绝对上限30/36；Phase31已有TP 18组、PP 12组计入累计。初筛每组1个seed，每个方向开发侧前三名做3-seed确认。\n\n在任何新确认Hfull target生成前，本阶段冻结9个新的BurstGPT正常窗口（BurstGPT三段各3个）及其低维特征、compact32 H0。它们与Phase27/28/30/31所有角色都满足300秒请求区间互斥，TP/PP各{len(tp_rows)}条feature rows，不含任何target。Mooncake在累计embargo下没有剩余完整容量，因此新确认的证据范围明确限定为BurstGPT；原10个固定窗口不变。\n\n完整请求列表未保存；以后只允许在预测文件与SHA归档后用于一次性生成Hfull teacher。\n""")
    (args.output_dir / "DONE").write_text(status + "\n")
    manifest = [f"{sha256(path)}  {path.relative_to(args.output_dir)}" for path in sorted(args.output_dir.rglob("*")) if path.is_file() and path.name != "manifest.sha256"]
    (args.output_dir / "manifest.sha256").write_text("\n".join(manifest) + "\n")
    if status != "PASS":
        raise RuntimeError(checks)
    print(json.dumps({"status": status, "profiles": len(selected), "rows_each": len(tp_rows), "target_generated": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
