#!/usr/bin/env python3
"""Freeze Phase34 six-model config and fresh target-free confirmation contract."""

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
from build_phase29b_tp_hfull_dataset import STRATEGIES, TP_SIZES, all_model_features, feature_values as tp_feature_values
from build_phase31b_known_model_hfull_dataset import MICROBATCHES, PP_SIZES, bin_vectors, feature_safe_profile, identifiers, prefixed_fields, reference_cost
from prepare_phase15_trace_windows import BURST_FILES, MOONCAKE_FILES, load_segment


HISTORY_MS = 300_000
SEED = "phase34-six-model-fresh-blind-v1-20260813"
SEGMENTS = ("burstgpt_1", "burstgpt_2", "burstgpt_3")
BLIND_PER_SEGMENT = 4
SIX_MODELS = (
    "deepseek-v2-lite",
    "qwen3-8b",
    "qwen3-30b-a3b",
    "llama-3.2-3b-instruct",
    "qwen2.5-14b-instruct",
    "mixtral-8x7b-instruct-v0.1",
)
PRIOR_SELECTION_NAMES = (
    "phase27_selection", "phase28_selection", "phase30_selection",
    "phase31_selection", "phase32_selection", "phase33_selection",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--windows", type=Path, default=root / "experiment-results/phase15_trace_data/windows.csv.gz")
    parser.add_argument("--base-model-features", type=Path, default=root / "experiment-results/phase16_model_features/model_features.json")
    parser.add_argument("--llama-config", type=Path, default=root / "experiment-configs/phase34/llama-3.2-3b-instruct.config.json")
    parser.add_argument("--qwen-config", type=Path, default=root / "experiment-configs/phase34/qwen2.5-14b-instruct.config.json")
    parser.add_argument("--mixtral-config", type=Path, default=root / "experiment-configs/phase34/mixtral-8x7b-instruct-v0.1.config.json")
    parser.add_argument("--phase27-selection", type=Path, default=root / "experiment-results/phase27a_pp_feature_and_holdout_contract/selection/selected_windows.csv")
    parser.add_argument("--phase28-selection", type=Path, default=root / "experiment-results/phase28a_second_confirmation_contract/selection/selected_windows.csv")
    parser.add_argument("--phase30-selection", type=Path, default=root / "experiment-results/phase30a_tp_structured_event_contract/selection/selected_windows.csv")
    parser.add_argument("--phase31-selection", type=Path, default=root / "experiment-results/phase31a_known_model_convergence_contract/selection/selected_windows.csv")
    parser.add_argument("--phase32-selection", type=Path, default=root / "experiment-results/phase32a_expanded_search_contract/selection/selected_windows.csv")
    parser.add_argument("--phase33-selection", type=Path, default=root / "experiment-results/phase33a_fresh_data_contract/selection/selected_windows.csv")
    parser.add_argument("--phase33c-dir", type=Path, default=root / "experiment-results/phase33c_target_free_model_selection")
    parser.add_argument("--phase33d-dir", type=Path, default=root / "experiment-results/phase33d_blind_confirmation_evaluation")
    parser.add_argument("--execution-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=root / "experiment-results/phase34a_six_model_contract")
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
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(name for name in row if name not in fields)
    buffer = io.StringIO(newline="")
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


def new_feature_row(model: str, config_path: Path, *, is_moe: bool, experts: int = 0, topk: int = 0) -> dict:
    config = json.loads(config_path.read_text())
    layers = int(config["num_hidden_layers"]); hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"]); kv_heads = int(config.get("num_key_value_heads", heads))
    intermediate = int(config["intermediate_size"])
    return {
        "model": model,
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "architecture_audit_only": config["architectures"][0],
        "model_type_audit_only": config["model_type"],
        "num_hidden_layers": layers,
        "hidden_size": hidden,
        "dense_intermediate_ratio": intermediate / hidden,
        "num_attention_heads": heads,
        "head_dim": float(config.get("head_dim", hidden / heads)),
        "kv_head_ratio": kv_heads / heads,
        "dtype_bytes": 2,
        "is_moe": int(is_moe),
        "num_experts": experts,
        "experts_per_token": topk,
        "moe_intermediate_ratio": intermediate / hidden if is_moe else 0.0,
        "num_shared_experts": 0,
        "first_dense_layers": 0,
        "moe_layer_frequency": 1 if is_moe else 0,
        "estimated_moe_layers": layers if is_moe else 0,
        "logical_collectives_per_forward_prior": 2 * layers + 1,
        "payload_bytes_per_active_token_prior": hidden * 2,
        "canonical_op_mask_all_reduce": 1,
        "raw_op_template_audit_only": "all_reduce:2L+1",
    }


def build_model_features(args: argparse.Namespace) -> list[dict]:
    existing = {row["model"]: row for row in json.loads(args.base_model_features.read_text())}
    added = {
        "llama-3.2-3b-instruct": new_feature_row("llama-3.2-3b-instruct", args.llama_config, is_moe=False),
        "qwen2.5-14b-instruct": new_feature_row("qwen2.5-14b-instruct", args.qwen_config, is_moe=False),
        "mixtral-8x7b-instruct-v0.1": new_feature_row("mixtral-8x7b-instruct-v0.1", args.mixtral_config, is_moe=True, experts=8, topk=2),
    }
    combined = {**existing, **added}
    return [combined[name] for name in SIX_MODELS]


def main() -> None:
    args = parse_args()
    for name in ("selection", "profiles", "dataset", "model_configs", "analysis", "docs", "logs"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)

    phase33c_manifest_before = sha256(args.phase33c_dir / "manifest.sha256")
    phase33d_manifest_before = sha256(args.phase33d_dir / "manifest.sha256")
    model_features = build_model_features(args)
    write_json(args.output_dir / "model_configs/model_features_six_models.json", model_features)
    model_map = all_model_features(args.output_dir / "model_configs/model_features_six_models.json")

    windows = pd.read_csv(args.windows, usecols=list(HISTORY_ONLY_SOURCE_COLUMNS))
    old = used_intervals(args)
    selected, inventory = [], []
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
        if len(pool) < BLIND_PER_SEGMENT:
            raise RuntimeError(f"{segment}: only {len(pool)} disjoint normal windows")
        pool_matrix = np.stack([selection_vector(row) for _, row in pool.iterrows()])
        medoids, labels, distances = choose_medoids(pool_matrix, BLIND_PER_SEGMENT)
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
                "role": "blind_confirmation",
            })
        chosen.sort(key=lambda row: row["role_order_sha256"])
        for local, row in enumerate(chosen, 1):
            row["profile_id"] = f"phase34_{segment}_blind_confirmation_{local:02d}"
            selected.append(row)
        inventory.append({
            "segment": segment, "eligible_before_embargo": before,
            "eligible_after_phase27_28_30_31_32_33_embargo": after,
            "disjoint_p95_pool": len(pool), "selected_blind": BLIND_PER_SEGMENT,
        })
    selected.sort(key=lambda row: (row["segment"], row["cutoff_ms"]))
    write_csv(args.output_dir / "selection/selected_windows.csv", selected)
    write_csv(args.output_dir / "selection/candidate_inventory.csv", inventory)

    raw_manifest = json.loads((args.raw_dir / "source_manifest.json").read_text())
    raw_checks = {row["name"]: (args.raw_dir / row["name"]).stat().st_size == int(row["actual_size"]) and sha256(args.raw_dir / row["name"]) == row["sha256"] for row in raw_manifest["sources"]}
    file_by_segment = {segment: args.raw_dir / name for name, (segment, _) in {**BURST_FILES, **MOONCAKE_FILES}.items()}
    arrays = {segment: load_segment(file_by_segment[segment]) for segment in SEGMENTS}
    profiles = []
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

    tp_rows, pp_rows = [], []
    for profile in profiles:
        compact = pseudo_requests(profile); safe = feature_safe_profile(profile)
        for model_name in SIX_MODELS:
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
    write_csv_gz(args.output_dir / "dataset/tp_blind_confirmation_features.csv.gz", tp_rows)
    write_csv_gz(args.output_dir / "dataset/pp_blind_confirmation_features.csv.gz", pp_rows)
    (args.output_dir / "docs/Phase34_六模型扩展执行参考.md").write_text(args.execution_reference.read_text())
    model_inventory = [{
        "model": row["model"], "existing_or_added": "existing" if row["model"] in {"deepseek-v2-lite", "qwen3-8b", "qwen3-30b-a3b"} else "added",
        "architecture": row["architecture_audit_only"], "layers": row["num_hidden_layers"],
        "hidden_size": row["hidden_size"], "is_moe": row["is_moe"], "experts": row["num_experts"],
        "experts_per_token": row["experts_per_token"], "payload_bytes_per_active_token": row["payload_bytes_per_active_token_prior"],
        "logical_collectives_per_forward": row["logical_collectives_per_forward_prior"], "config_sha256": row["config_sha256"],
    } for row in model_features]
    write_csv(args.output_dir / "analysis/model_inventory.csv", model_inventory)

    pair_overlaps = [(left["profile_id"], right["profile_id"]) for index, left in enumerate(selected) for right in selected[index + 1:] if left["segment"] == right["segment"] and overlaps(int(left["cutoff_ms"]), int(right["cutoff_ms"]))]
    prior_overlaps = [row["profile_id"] for row in selected if any(overlaps(int(row["cutoff_ms"]), prior) for prior in old[row["segment"]])]
    feature_names = set(tp_rows[0]) | set(pp_rows[0])
    expected_rows = len(selected) * len(SIX_MODELS) * 3 * 3 * 2
    checks = {
        "phase33_frozen_manifests_unchanged": phase33c_manifest_before == sha256(args.phase33c_dir / "manifest.sha256") and phase33d_manifest_before == sha256(args.phase33d_dir / "manifest.sha256"),
        "six_models_three_existing_three_added": len(model_features) == 6 and Counter("existing" if row["model"] in {"deepseek-v2-lite", "qwen3-8b", "qwen3-30b-a3b"} else "added" for row in model_features) == Counter({"existing": 3, "added": 3}),
        "positive_model_structure_and_unique_payload_coverage": all(int(row["num_hidden_layers"]) > 0 and int(row["hidden_size"]) > 0 for row in model_features) and len({int(row["payload_bytes_per_active_token_prior"]) for row in model_features}) >= 4,
        "twelve_blind_four_per_segment": len(selected) == 12 and Counter(row["segment"] for row in selected) == Counter({segment: 4 for segment in SEGMENTS}),
        "request_intervals_pairwise_disjoint": not pair_overlaps,
        "embargo_all_phase27_28_30_31_32_33_roles": not prior_overlaps,
        "history_only_p95_medoid_selection": all(float(row["normality_pool_quantile"]) == 0.95 for row in selected),
        "raw_sources_hash_pass": len(raw_checks) == 6 and all(raw_checks.values()),
        "six_models_all_configurations": len(tp_rows) == len(pp_rows) == expected_rows,
        "features_have_no_target": not any(name.startswith("target_") for name in feature_names),
        "full_request_lists_not_saved": not any(name in feature_names | set(profiles[0]) for name in {"requests", "input_lens", "output_lens", "full_request_list"}),
        "blind_target_not_generated": not (args.output_dir / "labels").exists(),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    blind_requests = sum(int(row["history_count"]) for row in selected)
    summary = {
        "schema_version": "phase34a-six-model-target-free-contract-v1", "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "selection_seed": SEED,
        "models": {"names": list(SIX_MODELS), "existing": list(SIX_MODELS[:3]), "added": list(SIX_MODELS[3:])},
        "model_config_sources": {
            "llama-3.2-3b-instruct": "normalized config cross-checked against Meta model card, SGLang registered test, and public compatible config; gated official HF config not anonymously readable",
            "qwen2.5-14b-instruct": "official Hugging Face config.json plus SGLang registered test",
            "mixtral-8x7b-instruct-v0.1": "official Hugging Face config.json plus SGLang implementation/tests",
        },
        "blind_confirmation": {"profiles": len(selected), "requests": blind_requests, "segments": list(SEGMENTS), "target_state": "not_generated"},
        "search_limits": {"tp_regular": 18, "tp_absolute": 24, "pp_regular": 18, "pp_absolute": 24, "screen_seed_per_candidate": 1, "top_candidates_three_seed": 3, "folds": 5},
        "counts": {"tp_blind_feature_rows": len(tp_rows), "pp_blind_feature_rows": len(pp_rows)},
        "evidence_limit": "累计300秒embargo后Mooncake没有剩余完整块；Phase34全新确认窗口为BurstGPT-only。",
        "checks": checks, "raw_source_checks": raw_checks,
        "inputs": {"windows_sha256": sha256(args.windows), "base_model_features_sha256": sha256(args.base_model_features), "execution_reference_sha256": sha256(args.execution_reference), "phase33c_manifest_sha256": phase33c_manifest_before, "phase33d_manifest_sha256": phase33d_manifest_before, **{f"{name}_sha256": sha256(getattr(args, name)) for name in PRIOR_SELECTION_NAMES}},
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "audit_summary.json", {"schema_version": "phase34a-audit-v1", "status": status, "checks": checks, "pair_overlaps": pair_overlaps, "prior_overlaps": prior_overlaps, "raw_source_checks": raw_checks})
    write_json(args.output_dir / "logs/build.log", {"event": "phase34a_six_model_target_free_contract_frozen", "status": status, "models": list(SIX_MODELS), "blind_profiles": len(selected), "blind_requests": blind_requests, "blind_target_generated": False})
    (args.output_dir / "README.md").write_text(f"""# Phase 34A：六模型配置与全新确认数据合同

本阶段冻结六模型集合：保留deepseek-v2-lite、qwen3-8b、qwen3-30b-a3b，新增llama-3.2-3b-instruct、qwen2.5-14b-instruct和mixtral-8x7b-instruct-v0.1。新增模型覆盖小型dense、大hidden dense和少专家top-2 MoE；只固化配置，没有下载权重或运行GPU profiling。

在Phase27/28/30/31/32/33所有已使用窗口的300秒embargo之外，从三个BurstGPT分段各冻结4个P95正常中心medoid，共12个请求级互斥的新确认画像、{blind_requests:,}个未来teacher请求。TP和PP各生成{len(tp_rows):,}条六模型低维feature/H0记录，不含target或完整请求列表。

Phase33三模型结果与manifest保持不变。下一阶段可在固定94个开发画像上生成六模型Hfull开发标签；必须先完成六模型训练、预测、checkpoint和SHA归档，才能一次性打开本批确认target。
""")
    (args.output_dir / "DONE").write_text(status + "\n")
    manifest = [f"{sha256(path)}  {path.relative_to(args.output_dir)}" for path in sorted(args.output_dir.rglob("*")) if path.is_file() and path.name != "manifest.sha256"]
    (args.output_dir / "manifest.sha256").write_text("\n".join(manifest) + "\n")
    if status != "PASS":
        raise RuntimeError(checks)
    print(json.dumps({"status": status, "models": list(SIX_MODELS), "blind_profiles": len(selected), "blind_requests": blind_requests, "feature_rows_each": expected_rows, "blind_target_generated": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
