#!/usr/bin/env python3
"""Expand the frozen 94 Phase33 development profiles to six-model Hfull data."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from build_phase21b_pp_h0 import pseudo_requests
from build_phase25_full_window_teacher import PP_BIN_EDGES, TP_BIN_EDGES, normalize, tp_histograms
from build_phase27b_pp_hfull_dataset import HISTORY_SECONDS, PHASES, summarize_profile, training_features as pp_training_features
from build_phase29b_tp_hfull_dataset import STRATEGIES, TP_SIZES, all_model_features, feature_values as tp_feature_values
from build_phase31b_known_model_hfull_dataset import MICROBATCHES, PP_SIZES, feature_safe_profile, histogram_fields, identifiers, pp_histograms, prefixed_fields
from prepare_phase15_trace_windows import BURST_FILES, MOONCAKE_FILES, load_segment


EXISTING_MODELS = ("deepseek-v2-lite", "qwen3-8b", "qwen3-30b-a3b")
ADDED_MODELS = ("llama-3.2-3b-instruct", "qwen2.5-14b-instruct", "mixtral-8x7b-instruct-v0.1")
SIX_MODELS = EXISTING_MODELS + ADDED_MODELS
DEVELOPMENT_ROLES = {"development_train", "development_validation"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--phase34a-dir", type=Path, default=root / "experiment-results/phase34a_six_model_contract")
    parser.add_argument("--phase33b-dir", type=Path, default=root / "experiment-results/phase33b_expanded_development_dataset")
    parser.add_argument("--phase31b-dir", type=Path, default=root / "experiment-results/phase31b_known_model_hfull_dataset")
    parser.add_argument("--output-dir", type=Path, default=root / "experiment-results/phase34b_six_model_hfull_dataset")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as source:
        return list(csv.DictReader(source))


def deterministic_gzip(path: Path, text: str) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as output:
            output.write(text.encode())


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(name for name in row if name not in fields)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def write_csv_gz(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(name for name in row if name not in fields)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    deterministic_gzip(path, buffer.getvalue())


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def inventory(profiles: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for profile in profiles:
        grouped[(profile["split_role"], profile["segment"])].append(profile)
    output = []
    for (role, segment), rows in sorted(grouped.items()):
        counts = [int(row["request_count"]) for row in rows]
        output.append({"split_role": role, "segment": segment, "profiles": len(rows), "requests_total": sum(counts), "requests_min": min(counts), "requests_median": statistics.median(counts), "requests_max": max(counts)})
    return output


def reconstruct_profiles(args: argparse.Namespace) -> tuple[list[dict], dict[str, list[tuple[int, int]]], list[dict]]:
    saved31 = [row for row in read_csv(args.phase31b_dir / "profiles/low_dimensional_profiles.csv.gz") if row["split_role"] in DEVELOPMENT_ROLES]
    saved33 = read_csv(args.phase33b_dir / "profiles/new_development_low_dimensional_profiles.csv.gz")
    saved = saved31 + saved33
    file_by_segment = {segment: args.raw_dir / name for name, (segment, _) in {**BURST_FILES, **MOONCAKE_FILES}.items()}
    arrays = {segment: load_segment(file_by_segment[segment]) for segment in sorted({row["segment"] for row in saved})}
    profiles, requests_by_profile, audits = [], {}, []
    for old in saved:
        timestamps, inputs, outputs = arrays[old["segment"]]
        cutoff = int(old["cutoff_ms"])
        left = int(np.searchsorted(timestamps, cutoff - HISTORY_SECONDS * 1000, side="left"))
        right = int(np.searchsorted(timestamps, cutoff, side="left"))
        selection = {
            "phase27_profile_id": old["profile_id"], "phase27_role": old["split_role"],
            "source": old["source"], "segment": old["segment"], "source_split": old["source_split"],
            "window_id": old["window_id"], "cutoff_ms": cutoff,
        }
        profile, requests = summarize_profile(selection, timestamps[left:right], inputs[left:right], outputs[left:right])
        profile["split_role"] = profile.pop("phase27_role")
        numeric_diffs = []; structured_matches = []
        for key, value in profile.items():
            if key in old and key not in {"profile_id", "split_role", "source", "segment", "source_split", "window_id"}:
                try:
                    numeric_diffs.append(abs(float(value) - float(old[key])))
                except (TypeError, ValueError):
                    structured_matches.append(json.loads(value) == json.loads(old[key]))
        audits.append({"profile_id": profile["profile_id"], "request_count_match": len(requests) == int(old["request_count"]), "structured_fields_match": all(structured_matches), "max_saved_profile_absolute_difference": max(numeric_diffs, default=0.0)})
        profiles.append(profile); requests_by_profile[profile["profile_id"]] = requests
    return profiles, requests_by_profile, audits


def teacher_and_example(profile: dict, requests: list[tuple[int, int]], model_name: str, model_meta: dict, model_values: dict) -> tuple[list[dict], list[dict], list[dict]]:
    compact = pseudo_requests(profile); safe = feature_safe_profile(profile)
    targets, tp_examples, pp_examples = [], [], []
    for tp_size in TP_SIZES:
        for policy, strategy in STRATEGIES.items():
            h0_histograms = {phase: normalize(hist, len(compact)) for phase, hist in tp_histograms(compact, strategy, model_meta).items()}
            target_histograms = {phase: normalize(hist, len(requests)) for phase, hist in tp_histograms(requests, strategy, model_meta).items()}
            for phase in PHASES:
                ids = identifiers(profile, model=model_name, parallelism="tp", parallel_size=tp_size, policy=policy, phase=phase)
                h0 = histogram_fields(h0_histograms[phase], TP_BIN_EDGES)
                target = histogram_fields(target_histograms[phase], TP_BIN_EDGES)
                targets.append({**ids, **target, "teacher_kind": "tp_full_window_fixed_draining_structural_teacher"})
                tp_examples.append({**ids, **tp_feature_values(safe, model_values, tp_size, policy, phase, []), **prefixed_fields("h0", h0), **prefixed_fields("target", target)})
    bytes_per_token = int(model_meta["payload_bytes_per_active_token_prior"])
    for pp_size in PP_SIZES:
        for microbatch in MICROBATCHES:
            h0_histograms, h0_audit = pp_histograms(compact, pp_size, microbatch, bytes_per_token)
            target_histograms, target_audit = pp_histograms(requests, pp_size, microbatch, bytes_per_token)
            if not h0_audit["all_requests_complete"] or not target_audit["all_requests_complete"]:
                raise RuntimeError(f"PP incomplete: {profile['profile_id']}/{model_name}/pp{pp_size}/mb{microbatch}")
            if target_audit["prefill_token_mass"] != sum(value[0] for value in requests) or target_audit["decode_token_mass"] != sum(value[1] - 1 for value in requests):
                raise RuntimeError(f"PP token mass mismatch: {profile['profile_id']}/{model_name}/pp{pp_size}/mb{microbatch}")
            for phase in PHASES:
                policy = f"mb{microbatch}"
                ids = identifiers(profile, model=model_name, parallelism="pp", parallel_size=pp_size, policy=policy, phase=phase)
                h0 = histogram_fields(h0_histograms[phase], PP_BIN_EDGES)
                target = histogram_fields(target_histograms[phase], PP_BIN_EDGES)
                targets.append({**ids, **target, "teacher_kind": "pp_scheduler_faithful_full_window_teacher_adapted_by_model_hidden_size"})
                pp_examples.append({**ids, **pp_training_features(safe, model_values, pp_size, microbatch, phase), **prefixed_fields("h0", h0), **prefixed_fields("target", target)})
    return targets, tp_examples, pp_examples


def bytes_anchor_relative_error(rows: list[dict]) -> tuple[float, list[dict]]:
    maximum, per_model = 0.0, defaultdict(list)
    for row in rows:
        tokens = float(row["feature_profile_input_mean_capped"]) if row["phase"] == "prefill" else max(float(row["feature_profile_output_mean_capped"]) - 1.0, 0.0)
        per_token = float(row["feature_model_payload_bytes_per_active_token_prior"])
        multiplier = float(row["feature_model_logical_collectives_per_forward_prior"]) if row["parallelism"] == "tp" else float(row["feature_pp_proxy_tensor_count"])
        anchor = tokens * per_token * multiplier * 1000.0
        target = float(row["target_total_logical_bytes_per_1000"])
        relative = abs(anchor - target) / max(target, 1e-12)
        maximum = max(maximum, relative); per_model[(row["parallelism"], row["model"])].append(relative)
    audit = [{"parallelism": key[0], "model": key[1], "rows": len(values), "max_relative_error": max(values), "mean_relative_error": float(np.mean(values))} for key, values in sorted(per_model.items())]
    return maximum, audit


def main() -> None:
    args = parse_args()
    for name in ("profiles", "dataset", "labels", "analysis", "logs"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    phase34a = json.loads((args.phase34a_dir / "summary.json").read_text())
    if phase34a["status"] != "PASS" or phase34a["blind_confirmation"]["target_state"] != "not_generated":
        raise RuntimeError("Phase34A target-free contract invalid")
    raw_manifest = json.loads((args.raw_dir / "source_manifest.json").read_text())
    raw_checks = {row["name"]: (args.raw_dir / row["name"]).stat().st_size == int(row["actual_size"]) and sha256(args.raw_dir / row["name"]) == row["sha256"] for row in raw_manifest["sources"]}
    if len(raw_checks) != 6 or not all(raw_checks.values()):
        raise RuntimeError(raw_checks)

    profiles, requests_by_profile, profile_audits = reconstruct_profiles(args)
    if Counter(profile["split_role"] for profile in profiles) != Counter({"development_train": 75, "development_validation": 19}):
        raise RuntimeError("profile roles mismatch")
    model_map = all_model_features(args.phase34a_dir / "model_configs/model_features_six_models.json")
    if tuple(model_map) != SIX_MODELS:
        raise RuntimeError(f"model order mismatch: {tuple(model_map)}")

    old_examples = {p: read_csv(args.phase33b_dir / f"dataset/{p}_combined_development_examples.csv.gz") for p in ("tp", "pp")}
    old_targets = read_csv(args.phase33b_dir / "labels/combined_development_hfull_targets.csv.gz")
    new_targets, new_examples = [], {"tp": [], "pp": []}
    for profile in profiles:
        for model_name in ADDED_MODELS:
            model_meta, model_values = model_map[model_name]
            targets, tp_rows, pp_rows = teacher_and_example(profile, requests_by_profile[profile["profile_id"]], model_name, model_meta, model_values)
            new_targets.extend(targets); new_examples["tp"].extend(tp_rows); new_examples["pp"].extend(pp_rows)

    combined_examples = {p: old_examples[p] + new_examples[p] for p in ("tp", "pp")}
    combined_targets = old_targets + new_targets
    write_csv_gz(args.output_dir / "profiles/low_dimensional_profiles_94.csv.gz", profiles)
    write_csv_gz(args.output_dir / "labels/new_three_model_hfull_targets.csv.gz", new_targets)
    write_csv_gz(args.output_dir / "labels/six_model_development_hfull_targets.csv.gz", combined_targets)
    for parallelism in ("tp", "pp"):
        write_csv_gz(args.output_dir / f"dataset/{parallelism}_new_three_model_examples.csv.gz", new_examples[parallelism])
        write_csv_gz(args.output_dir / f"dataset/{parallelism}_six_model_development_examples.csv.gz", combined_examples[parallelism])
    write_csv(args.output_dir / "analysis/profile_reconstruction_audit.csv", profile_audits)
    write_csv(args.output_dir / "analysis/profile_inventory.csv", inventory(profiles))
    anchor_max, anchor_audit = bytes_anchor_relative_error(combined_examples["tp"] + combined_examples["pp"])
    write_csv(args.output_dir / "analysis/bytes_anchor_audit.csv", anchor_audit)
    feature_columns = {parallelism: [name for name in combined_examples[parallelism][0] if name.startswith("feature_")] for parallelism in ("tp", "pp")}
    write_json(args.output_dir / "feature_columns.json", feature_columns)

    old_ids = {row["example_id"] for row in old_targets}; new_ids = {row["example_id"] for row in new_targets}
    blind_features = {p: read_csv(args.phase34a_dir / f"dataset/{p}_blind_confirmation_features.csv.gz") for p in ("tp", "pp")}
    blind_ids = {row["example_id"] for rows in blind_features.values() for row in rows}
    model_counts = {p: Counter(row["model"] for row in combined_examples[p]) for p in ("tp", "pp")}
    checks = {
        "phase34a_target_free_pass": phase34a["status"] == "PASS" and phase34a["blind_confirmation"]["target_state"] == "not_generated",
        "raw_source_hashes_6_of_6": len(raw_checks) == 6 and all(raw_checks.values()),
        "profiles_94_roles_75_19": len(profiles) == 94 and Counter(profile["split_role"] for profile in profiles) == Counter({"development_train": 75, "development_validation": 19}),
        "history_counts_match_94_of_94": all(row["request_count_match"] for row in profile_audits),
        "saved_profile_reconstruction_exact_within_float_tolerance": all(row["structured_fields_match"] for row in profile_audits) and max(float(row["max_saved_profile_absolute_difference"]) for row in profile_audits) < 1e-10,
        "phase33_existing_three_rows_reused_immutable": len(old_targets) == 10152 and all(len(old_examples[p]) == 5076 for p in ("tp", "pp")),
        "new_three_targets_10152": len(new_targets) == 10152 and not old_ids.intersection(new_ids),
        "six_model_targets_20304": len(combined_targets) == 20304 and len({row["example_id"] for row in combined_targets}) == 20304,
        "six_model_examples_10152_each": all(len(rows) == 10152 for rows in combined_examples.values()),
        "six_models_1692_rows_each_direction": all(model_counts[p] == Counter({model: 1692 for model in SIX_MODELS}) for p in ("tp", "pp")),
        "feature_columns_compatible": all(set(old_examples[p][0]) == set(new_examples[p][0]) for p in ("tp", "pp")),
        "bytes_anchor_all_six_models_matches_teacher": anchor_max < 1e-10,
        "blind_features_target_free": all(not any(name.startswith("target_") for name in rows[0]) for rows in blind_features.values()),
        "blind_ids_absent_from_all_development_targets": not blind_ids.intersection({row["example_id"] for row in combined_targets}),
        "full_request_lists_not_saved": not any(name in set(combined_examples["tp"][0]) | set(combined_examples["pp"][0]) | set(profiles[0]) for name in {"requests", "input_lens", "output_lens", "full_request_list"}),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    unique_requests = sum(len(value) for value in requests_by_profile.values())
    summary = {
        "schema_version": "phase34b-six-model-hfull-dataset-v1", "status": status,
        "models": {"all": list(SIX_MODELS), "existing_phase33": list(EXISTING_MODELS), "added_phase34": list(ADDED_MODELS)},
        "profiles": {"development": 94, "development_train": 75, "development_validation": 19, "burstgpt": 90, "mooncake": 4},
        "full_window_requests": {"unique_profile_requests": unique_requests, "expected_phase33_value": 35524, "usage": "offline_teacher_only_not_saved_as_features"},
        "targets": {"existing_three_rows": len(old_targets), "new_three_rows": len(new_targets), "six_model_rows": len(combined_targets), "tp_examples": len(combined_examples["tp"]), "pp_examples": len(combined_examples["pp"])},
        "teacher_contract": {"tp": "Phase26A GPU-validated fixed-draining structural formula", "pp": "Phase25B/25C scheduler-faithful event simulator adapted by model hidden size"},
        "bytes_anchor": {"definition": "low-dimensional capped mean × model bytes/token × structural communication multiplier × 1000", "maximum_relative_error_all_six_models": anchor_max, "status": "PASS" if anchor_max < 1e-10 else "FAIL"},
        "blind_confirmation_target_state": "not_generated", "checks": checks, "raw_source_checks": raw_checks,
        "inputs": {"phase34a_manifest_sha256": sha256(args.phase34a_dir / "manifest.sha256"), "phase33b_manifest_sha256": sha256(args.phase33b_dir / "manifest.sha256"), "phase31b_manifest_sha256": sha256(args.phase31b_dir / "manifest.sha256"), "raw_manifest_sha256": sha256(args.raw_dir / "source_manifest.json")},
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "audit_summary.json", {"schema_version": "phase34b-audit-v1", "status": status, "checks": checks, "bytes_anchor_max_relative_error": anchor_max, "raw_source_checks": raw_checks})
    write_json(args.output_dir / "logs/build.log", {"event": "phase34b_six_model_hfull_dataset_built", "status": status, "profiles": 94, "unique_teacher_requests": unique_requests, "six_model_target_rows": len(combined_targets), "blind_target_generated": False})
    (args.output_dir / "README.md").write_text(f"""# Phase 34B：六模型Hfull开发数据

本阶段保持Phase33的94个开发画像不变：75个训练、19个验证，覆盖90个BurstGPT与4个Mooncake窗口，共{unique_requests:,}个唯一完整teacher请求。Phase33三个模型的已冻结数据逐字复用；只为三个新增模型生成Hfull标签，再合并为六模型数据。

TP和PP各有{len(combined_examples['tp']):,}条phase训练样本，每个模型各1,692条，覆盖三种并行规模、三种policy与prefill/decode。TP使用Phase26A验证过的fixed-draining结构公式；PP使用Phase25B/25C验证过的scheduler-faithful事件模拟器。完整请求列表只在构建内存中使用，没有保存或进入特征。

六模型低维bytes均值结构锚点与Hfull teacher逐条一致，最大相对误差为`{anchor_max:.3e}`，因此后续TP/PP都允许继续使用该锚点并保留H0的12-bin bytes形状。Phase34A的12个全新确认画像仍只有低维feature/H0，Hfull target尚未生成。
""")
    (args.output_dir / "DONE").write_text(status + "\n")
    manifest = [f"{sha256(path)}  {path.relative_to(args.output_dir)}" for path in sorted(args.output_dir.rglob("*")) if path.is_file() and path.name != "manifest.sha256"]
    (args.output_dir / "manifest.sha256").write_text("\n".join(manifest) + "\n")
    if status != "PASS":
        raise RuntimeError(checks)
    print(json.dumps({"status": status, "profiles": len(profiles), "unique_teacher_requests": unique_requests, "models": list(SIX_MODELS), "tp_rows": len(combined_examples["tp"]), "pp_rows": len(combined_examples["pp"]), "bytes_anchor_max_relative_error": anchor_max, "blind_target_generated": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
