#!/usr/bin/env python3
"""Freeze fresh Phase33 development and blind-confirmation windows without targets."""

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
from build_phase27a_pp_feature_and_holdout_contract import HISTORY_ONLY_SOURCE_COLUMNS, choose_medoids, selection_vector
from build_phase27b_pp_hfull_dataset import HISTORY_SECONDS, PHASES, summarize_profile, training_features as pp_training_features
from build_phase29b_tp_hfull_dataset import MODELS, STRATEGIES, TP_SIZES, all_model_features, feature_values as tp_feature_values
from build_phase31b_known_model_hfull_dataset import MICROBATCHES, PP_SIZES, bin_vectors, feature_safe_profile, identifiers, prefixed_fields, reference_cost
from prepare_phase15_trace_windows import BURST_FILES, MOONCAKE_FILES, load_segment


HISTORY_MS = 300_000
SEED = "phase33-fresh-normal-development-blind-v1-20260813"
SEGMENTS = ("burstgpt_1", "burstgpt_2", "burstgpt_3")
ROLE_QUOTAS = {"development_train": 12, "development_validation": 3, "blind_confirmation": 3}
PRIOR_SELECTION_NAMES = (
    "phase27_selection", "phase28_selection", "phase30_selection", "phase31_selection", "phase32_selection",
)


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
    parser.add_argument("--phase32-selection", type=Path, default=root / "experiment-results/phase32a_expanded_search_contract/selection/selected_windows.csv")
    parser.add_argument("--execution-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=root / "experiment-results/phase33a_fresh_data_contract")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(name for name in row if name not in fields)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def deterministic_gzip(path: Path, text: str) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as output:
            output.write(text.encode())


def write_csv_gz(path: Path, rows: list[dict]) -> None:
    buffer = io.StringIO(newline="")
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(name for name in row if name not in fields)
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    deterministic_gzip(path, buffer.getvalue())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def overlaps(cutoff: int, old: int) -> bool:
    return abs(cutoff - old) < HISTORY_MS


def used_intervals(args: argparse.Namespace) -> dict[str, list[int]]:
    output = {segment: [] for segment in SEGMENTS}
    for name in PRIOR_SELECTION_NAMES:
        for row in read_csv(getattr(args, name)):
            if row["segment"] in output:
                output[row["segment"]].append(int(row["cutoff_ms"]))
    return output


def disjoint_pool(frame: pd.DataFrame) -> pd.DataFrame:
    chosen, last = [], None
    for index, row in frame.sort_values(["cutoff_ms", "window_id"], kind="stable").iterrows():
        cutoff = int(row["cutoff_ms"])
        if last is None or cutoff - last >= HISTORY_MS:
            chosen.append(index); last = cutoff
    return frame.loc[chosen].reset_index(drop=True)


def role_order(window_id: str) -> str:
    return hashlib.sha256(f"{SEED}:{window_id}".encode()).hexdigest()


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
        phase: {int(active_tokens) * bytes_per_token: float(events * PP_PROXY_COUNT) * scale for active_tokens, events in sorted(simulated.event_histograms[phase].items())}
        for phase in PHASES
    }


def main() -> None:
    args = parse_args()
    for name in ("selection", "profiles", "dataset", "analysis", "docs", "logs"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    windows = pd.read_csv(args.windows, usecols=list(HISTORY_ONLY_SOURCE_COLUMNS))
    old = used_intervals(args)
    selected, inventory = [], []
    quota = sum(ROLE_QUOTAS.values())
    for segment in SEGMENTS:
        candidates = windows[(windows["segment"] == segment) & (windows["history_count"] >= 32)].copy()
        before = len(candidates)
        candidates = candidates[[not any(overlaps(int(value), prior) for prior in old[segment]) for value in candidates["cutoff_ms"]]].copy()
        after = len(candidates)
        matrix = np.stack([selection_vector(row) for _, row in candidates.iterrows()])
        median = np.median(matrix, axis=0)
        scale = np.quantile(matrix, 0.75, axis=0) - np.quantile(matrix, 0.25, axis=0)
        scale[scale < 1e-9] = 1.0
        candidates["normality_distance"] = np.sqrt(np.mean(((matrix - median) / scale) ** 2, axis=1))
        threshold = float(np.quantile(candidates["normality_distance"], 0.95))
        pool = disjoint_pool(candidates[candidates["normality_distance"] <= threshold])
        if len(pool) < quota:
            raise RuntimeError(f"{segment}: only {len(pool)} disjoint normal windows")
        pool_matrix = np.stack([selection_vector(row) for _, row in pool.iterrows()])
        medoids, labels, distances = choose_medoids(pool_matrix, quota)
        chosen = []
        for cluster, index in enumerate(medoids):
            row = pool.iloc[index]; members = np.flatnonzero(labels == cluster)
            chosen.append({
                "window_id": str(row["window_id"]), "source": str(row["source"]), "segment": segment,
                "source_split": str(row["split"]), "cutoff_ms": int(row["cutoff_ms"]),
                "history_seconds": int(row["history_seconds"]), "history_count": int(row["history_count"]),
                "normality_distance": float(row["normality_distance"]), "normality_pool_quantile": 0.95,
                "selection_cluster": cluster, "selection_cluster_members": int(len(members)),
                "selection_distance_to_medoid_mean": float(np.mean(distances[members])),
                "role_order_sha256": role_order(str(row["window_id"])),
            })
        chosen.sort(key=lambda row: row["role_order_sha256"])
        offset = 0
        for role, count in ROLE_QUOTAS.items():
            for local, row in enumerate(chosen[offset:offset + count], 1):
                row["role"] = role
                row["profile_id"] = f"phase33_{segment}_{role}_{local:02d}"
                selected.append(row)
            offset += count
        inventory.append({
            "segment": segment, "eligible_before_embargo": before, "eligible_after_all_prior_embargo": after,
            "disjoint_p95_pool": len(pool), "selected": quota,
            **{f"selected_{role}": count for role, count in ROLE_QUOTAS.items()},
        })
    selected.sort(key=lambda row: (row["segment"], row["cutoff_ms"]))
    write_csv(args.output_dir / "selection/selected_windows.csv", selected)
    write_csv(args.output_dir / "selection/candidate_inventory.csv", inventory)

    raw_manifest = json.loads((args.raw_dir / "source_manifest.json").read_text())
    raw_checks = {row["name"]: (args.raw_dir / row["name"]).stat().st_size == int(row["actual_size"]) and sha256(args.raw_dir / row["name"]) == row["sha256"] for row in raw_manifest["sources"]}
    file_by_segment = {segment: args.raw_dir / name for name, (segment, _) in {**BURST_FILES, **MOONCAKE_FILES}.items()}
    arrays = {segment: load_segment(file_by_segment[segment]) for segment in SEGMENTS}
    profiles, request_windows = [], {}
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
        profiles.append(profile); request_windows[profile["profile_id"]] = requests

    model_map = all_model_features(args.model_features)
    tp_rows, pp_rows = [], []
    for profile in profiles:
        compact = pseudo_requests(profile); safe = feature_safe_profile(profile)
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
    for parallelism, rows in (("tp", tp_rows), ("pp", pp_rows)):
        development = [row for row in rows if row["split_role"] in {"development_train", "development_validation"}]
        confirmation = [row for row in rows if row["split_role"] == "blind_confirmation"]
        write_csv_gz(args.output_dir / f"dataset/{parallelism}_new_development_features.csv.gz", development)
        write_csv_gz(args.output_dir / f"dataset/{parallelism}_blind_confirmation_features.csv.gz", confirmation)
    (args.output_dir / "docs/Phase33_TP继续收敛与PP保守改进执行参考.md").write_text(args.execution_reference.read_text())

    role_counts = Counter(row["role"] for row in selected)
    pair_overlaps = [(left["profile_id"], right["profile_id"]) for index, left in enumerate(selected) for right in selected[index + 1:] if left["segment"] == right["segment"] and overlaps(int(left["cutoff_ms"]), int(right["cutoff_ms"]))]
    prior_overlaps = [row["profile_id"] for row in selected if any(overlaps(int(row["cutoff_ms"]), prior) for prior in old[row["segment"]])]
    feature_names = set(tp_rows[0]) | set(pp_rows[0])
    checks = {
        "profiles_54_roles_36_9_9": len(selected) == 54 and role_counts == Counter({"development_train": 36, "development_validation": 9, "blind_confirmation": 9}),
        "three_burst_segments_18_each": Counter(row["segment"] for row in selected) == Counter({segment: 18 for segment in SEGMENTS}),
        "request_intervals_pairwise_disjoint": not pair_overlaps,
        "embargo_all_phase27_28_30_31_32_roles": not prior_overlaps,
        "history_only_p95_medoid_selection": all(float(row["normality_pool_quantile"]) == 0.95 for row in selected),
        "raw_sources_hash_pass": len(raw_checks) == 6 and all(raw_checks.values()),
        "three_models_all_configurations": len(tp_rows) == len(pp_rows) == 54 * 3 * 3 * 3 * 2,
        "features_have_no_target": not any(name.startswith("target_") for name in feature_names),
        "full_request_lists_not_saved": not any(name in feature_names | set(profiles[0]) for name in {"requests", "input_lens", "output_lens", "full_request_list"}),
        "blind_target_not_generated": not (args.output_dir / "labels").exists(),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    development_requests = sum(int(row["history_count"]) for row in selected if row["role"].startswith("development_"))
    blind_requests = sum(int(row["history_count"]) for row in selected if row["role"] == "blind_confirmation")
    summary = {
        "schema_version": "phase33a-fresh-data-contract-v1", "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "selection_seed": SEED,
        "profiles": len(selected), "role_counts": dict(role_counts), "segments": list(SEGMENTS),
        "search_limits_new": {"tp_regular": 18, "tp_absolute": 24, "pp_regular": 8, "pp_absolute": 12, "screen_seed_per_candidate": 1, "top_candidates_three_seed": 3, "folds": 5},
        "target_state": {"development": "not_generated_in_phase33a", "blind_confirmation": "not_generated"},
        "counts": {"development_full_requests_future_teacher_only": development_requests, "blind_full_requests_hidden_teacher_only_future": blind_requests, "tp_development_feature_rows": 45 * 54, "tp_blind_feature_rows": 9 * 54, "pp_development_feature_rows": 45 * 54, "pp_blind_feature_rows": 9 * 54},
        "evidence_limit": "All remaining request-disjoint Mooncake capacity is exhausted under accumulated 300-second embargo; Phase33 fresh evidence is BurstGPT-only.",
        "checks": checks, "raw_source_checks": raw_checks,
        "inputs": {"windows_sha256": sha256(args.windows), "model_features_sha256": sha256(args.model_features), "execution_reference_sha256": sha256(args.execution_reference), **{f"{name}_sha256": sha256(getattr(args, name)) for name in PRIOR_SELECTION_NAMES}},
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "audit_summary.json", {"schema_version": "phase33a-audit-v1", "status": status, "checks": checks, "pair_overlaps": pair_overlaps, "prior_overlaps": prior_overlaps, "raw_source_checks": raw_checks})
    write_json(args.output_dir / "logs/build.log", {"event": "phase33a_fresh_target_blind_contract_frozen", "status": status, "profiles": len(selected), "roles": dict(role_counts), "development_target_generated": False, "blind_target_generated": False})
    (args.output_dir / "README.md").write_text(f"""# Phase 33A：新增开发与盲确认数据合同

本阶段在任何Phase33 Hfull target生成前，冻结54个新的BurstGPT正常窗口：36个训练、9个验证、9个盲确认。三个BurstGPT分段各18个；所有Phase33角色彼此请求区间互斥，并对Phase27/28/30/31/32全部历史角色设置300秒embargo。

选择只使用历史侧低维统计，在P95正常中心池中做18类medoid覆盖，并用事前固定hash分配角色。TP/PP分别生成{45 * 54}条新增开发feature rows和{9 * 54}条盲确认feature rows，覆盖三个模型、三种并行规模、三种policy与两个phase；全部不含target或完整请求列表。

Mooncake在累计embargo下没有剩余完整块，因此Phase33新证据限定为BurstGPT。开发Hfull可在下一阶段生成；盲确认Hfull必须等模型、预测文件和SHA冻结后才允许一次性生成。
""")
    (args.output_dir / "DONE").write_text(status + "\n")
    manifest = [f"{sha256(path)}  {path.relative_to(args.output_dir)}" for path in sorted(args.output_dir.rglob("*")) if path.is_file() and path.name != "manifest.sha256"]
    (args.output_dir / "manifest.sha256").write_text("\n".join(manifest) + "\n")
    if status != "PASS":
        raise RuntimeError(checks)
    print(json.dumps({"status": status, "profiles": len(selected), "roles": dict(role_counts), "development_requests": development_requests, "blind_requests": blind_requests, "blind_target_generated": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
