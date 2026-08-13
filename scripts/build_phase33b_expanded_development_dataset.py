#!/usr/bin/env python3
"""Generate fresh development Hfull targets and combine them with Phase31 development data."""

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

from build_phase25_full_window_teacher import PP_BIN_EDGES, TP_BIN_EDGES, normalize, tp_histograms
from build_phase27b_pp_hfull_dataset import HISTORY_SECONDS, PHASES, summarize_profile
from build_phase29b_tp_hfull_dataset import MODELS, STRATEGIES, TP_SIZES, all_model_features
from build_phase31b_known_model_hfull_dataset import MICROBATCHES, PP_SIZES, histogram_fields, pp_histograms, prefixed_fields
from prepare_phase15_trace_windows import BURST_FILES, MOONCAKE_FILES, load_segment


DEVELOPMENT_ROLES = {"development_train", "development_validation"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--phase33a-dir", type=Path, default=root / "experiment-results/phase33a_fresh_data_contract")
    parser.add_argument("--phase31b-dir", type=Path, default=root / "experiment-results/phase31b_known_model_hfull_dataset")
    parser.add_argument("--model-features", type=Path, default=root / "experiment-results/phase16_model_features/model_features.json")
    parser.add_argument("--output-dir", type=Path, default=root / "experiment-results/phase33b_expanded_development_dataset")
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
    buffer = io.StringIO(newline="")
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(name for name in row if name not in fields)
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    deterministic_gzip(path, buffer.getvalue())


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def identifier(profile: dict, model: str, parallelism: str, parallel_size: int, policy: str, phase: str) -> dict:
    return {
        "example_id": f"{parallelism}/{model}/p{parallel_size}/{policy}/{profile['profile_id']}/{phase}",
        "profile_id": profile["profile_id"], "split_role": profile["split_role"], "source": profile["source"],
        "segment": profile["segment"], "window_id": profile["window_id"], "model": model,
        "parallelism": parallelism, "parallel_size": parallel_size, "policy": policy, "phase": phase,
    }


def inventory(profiles: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for profile in profiles:
        grouped[(profile["split_role"], profile["segment"])].append(profile)
    output = []
    for (role, segment), rows in sorted(grouped.items()):
        counts = [int(row["request_count"]) for row in rows]
        output.append({"split_role": role, "segment": segment, "profiles": len(rows), "requests_total": sum(counts), "requests_min": min(counts), "requests_median": statistics.median(counts), "requests_max": max(counts)})
    return output


def main() -> None:
    args = parse_args()
    for name in ("profiles", "dataset", "labels", "analysis", "logs"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    phase33a = json.loads((args.phase33a_dir / "summary.json").read_text())
    if phase33a["status"] != "PASS" or phase33a["target_state"]["blind_confirmation"] != "not_generated":
        raise RuntimeError("Phase33A target-free contract is not valid")
    selection = [row for row in read_csv(args.phase33a_dir / "selection/selected_windows.csv") if row["role"] in DEVELOPMENT_ROLES]
    if len(selection) != 45:
        raise RuntimeError(f"expected 45 fresh development profiles, got {len(selection)}")

    raw_manifest = json.loads((args.raw_dir / "source_manifest.json").read_text())
    raw_checks = {row["name"]: (args.raw_dir / row["name"]).stat().st_size == int(row["actual_size"]) and sha256(args.raw_dir / row["name"]) == row["sha256"] for row in raw_manifest["sources"]}
    if len(raw_checks) != 6 or not all(raw_checks.values()):
        raise RuntimeError(raw_checks)
    file_by_segment = {segment: args.raw_dir / name for name, (segment, _) in {**BURST_FILES, **MOONCAKE_FILES}.items()}
    arrays = {segment: load_segment(file_by_segment[segment]) for segment in sorted({row["segment"] for row in selection})}
    profiles, request_windows, count_matches = [], {}, []
    for selected in selection:
        timestamps, inputs, outputs = arrays[selected["segment"]]
        cutoff = int(selected["cutoff_ms"])
        left = int(np.searchsorted(timestamps, cutoff - HISTORY_SECONDS * 1000, side="left"))
        right = int(np.searchsorted(timestamps, cutoff, side="left"))
        compatible = {**selected, "phase27_profile_id": selected["profile_id"], "phase27_role": selected["role"]}
        profile, requests = summarize_profile(compatible, timestamps[left:right], inputs[left:right], outputs[left:right])
        profile["split_role"] = profile.pop("phase27_role")
        count_matches.append(len(requests) == int(selected["history_count"]))
        profiles.append(profile); request_windows[profile["profile_id"]] = requests

    model_map = all_model_features(args.model_features)
    if set(model_map) != set(MODELS):
        raise RuntimeError("model mismatch")
    new_targets, simulation_checks = [], []
    for profile in profiles:
        requests = request_windows[profile["profile_id"]]
        for model_name in MODELS:
            model_meta, _ = model_map[model_name]
            for tp_size in TP_SIZES:
                for policy, strategy in STRATEGIES.items():
                    histograms = {phase: normalize(histogram, len(requests)) for phase, histogram in tp_histograms(requests, strategy, model_meta).items()}
                    for phase in PHASES:
                        ids = identifier(profile, model_name, "tp", tp_size, policy, phase)
                        target = histogram_fields(histograms[phase], TP_BIN_EDGES)
                        new_targets.append({**ids, **target, "teacher_kind": "tp_full_window_fixed_draining_structural_teacher"})
            bytes_per_token = int(model_meta["payload_bytes_per_active_token_prior"])
            for pp_size in PP_SIZES:
                for microbatch in MICROBATCHES:
                    histograms, audit = pp_histograms(requests, pp_size, microbatch, bytes_per_token)
                    simulation_checks.append({"profile_id": profile["profile_id"], "model": model_name, "pp_size": pp_size, "microbatch": microbatch, "all_requests_complete": audit["all_requests_complete"], "prefill_token_mass_exact": audit["prefill_token_mass"] == sum(value[0] for value in requests), "decode_token_mass_exact": audit["decode_token_mass"] == sum(value[1] - 1 for value in requests)})
                    for phase in PHASES:
                        policy = f"mb{microbatch}"; ids = identifier(profile, model_name, "pp", pp_size, policy, phase)
                        target = histogram_fields(histograms[phase], PP_BIN_EDGES)
                        new_targets.append({**ids, **target, "teacher_kind": "pp_scheduler_faithful_full_window_teacher_adapted_by_model_hidden_size"})

    target_by_id = {row["example_id"]: row for row in new_targets}
    if len(target_by_id) != len(new_targets):
        raise RuntimeError("duplicate target ids")
    new_examples = {}
    for parallelism in ("tp", "pp"):
        features = read_csv(args.phase33a_dir / f"dataset/{parallelism}_new_development_features.csv.gz")
        rows = []
        for feature in features:
            target = target_by_id[feature["example_id"]]
            rows.append({**feature, **prefixed_fields("target", target)})
        new_examples[parallelism] = rows

    old_examples = {parallelism: read_csv(args.phase31b_dir / f"dataset/{parallelism}_development_examples.csv.gz") for parallelism in ("tp", "pp")}
    combined_examples = {parallelism: old_examples[parallelism] + new_examples[parallelism] for parallelism in ("tp", "pp")}
    old_targets = read_csv(args.phase31b_dir / "labels/development_hfull_targets.csv.gz")
    combined_targets = old_targets + new_targets
    write_csv_gz(args.output_dir / "profiles/new_development_low_dimensional_profiles.csv.gz", profiles)
    write_csv_gz(args.output_dir / "labels/new_development_hfull_targets.csv.gz", new_targets)
    write_csv_gz(args.output_dir / "labels/combined_development_hfull_targets.csv.gz", combined_targets)
    for parallelism in ("tp", "pp"):
        write_csv_gz(args.output_dir / f"dataset/{parallelism}_new_development_examples.csv.gz", new_examples[parallelism])
        write_csv_gz(args.output_dir / f"dataset/{parallelism}_combined_development_examples.csv.gz", combined_examples[parallelism])
    write_csv(args.output_dir / "analysis/new_profile_inventory.csv", inventory(profiles))
    write_csv(args.output_dir / "analysis/pp_simulation_checks.csv", simulation_checks)
    feature_columns = {parallelism: [name for name in combined_examples[parallelism][0] if name.startswith("feature_")] for parallelism in ("tp", "pp")}
    write_json(args.output_dir / "feature_columns.json", feature_columns)

    roles = Counter(row["split_role"] for row in combined_examples["tp"] if row["model"] == MODELS[0] and row["parallel_size"] == "2" and row["policy"] == "latency" and row["phase"] == "prefill")
    blind_features = {parallelism: read_csv(args.phase33a_dir / f"dataset/{parallelism}_blind_confirmation_features.csv.gz") for parallelism in ("tp", "pp")}
    target_ids = {row["example_id"] for row in combined_targets}
    blind_ids = {row["example_id"] for rows in blind_features.values() for row in rows}
    checks = {
        "phase33a_target_free_pass": phase33a["status"] == "PASS" and phase33a["target_state"]["blind_confirmation"] == "not_generated",
        "raw_source_hashes_6_of_6": len(raw_checks) == 6 and all(raw_checks.values()),
        "new_profiles_45_roles_36_9": Counter(profile["split_role"] for profile in profiles) == Counter({"development_train": 36, "development_validation": 9}),
        "history_counts_match_45_of_45": all(count_matches),
        "new_targets_4860": len(new_targets) == 45 * 2 * 3 * 3 * 3 * 2,
        "combined_targets_10152": len(combined_targets) == 10152,
        "new_examples_2430_each": all(len(rows) == 2430 for rows in new_examples.values()),
        "combined_examples_5076_each": all(len(rows) == 5076 for rows in combined_examples.values()),
        "combined_profiles_94_roles_75_19": roles == Counter({"development_train": 75, "development_validation": 19}),
        "feature_columns_compatible": all(set(old_examples[p][0]) == set(new_examples[p][0]) for p in ("tp", "pp")),
        "pp_scheduler_mass_and_completion_exact": all(bool(value) for row in simulation_checks for key, value in row.items() if key.endswith("complete") or key.endswith("exact")),
        "blind_features_have_no_target": all(not any(name.startswith("target_") for name in rows[0]) for rows in blind_features.values()),
        "blind_target_ids_absent": not target_ids.intersection(blind_ids),
        "full_request_lists_not_saved": not any(name in set(combined_examples["tp"][0]) | set(combined_examples["pp"][0]) | set(profiles[0]) for name in {"requests", "input_lens", "output_lens", "full_request_list"}),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = {
        "schema_version": "phase33b-expanded-development-dataset-v1", "status": status,
        "profiles": {"phase31_existing_development": 49, "phase33_new_development": 45, "combined": 94, "combined_train": 75, "combined_validation": 19},
        "full_window_requests": {"phase31_existing": 21058, "phase33_new": sum(len(value) for value in request_windows.values()), "combined": 21058 + sum(len(value) for value in request_windows.values())},
        "targets": {"new_phase_rows": len(new_targets), "combined_phase_rows": len(combined_targets), "tp_combined_rows": len(combined_examples["tp"]), "pp_combined_rows": len(combined_examples["pp"])},
        "feature_columns": {key: len(value) for key, value in feature_columns.items()},
        "blind_confirmation_target_state": "not_generated",
        "teacher_contract": {"tp": "Phase26A GPU-validated structural formula", "pp": "Phase25B/25C scheduler-faithful event simulator adapted by model hidden-size bytes/token"},
        "checks": checks, "raw_source_checks": raw_checks,
        "inputs": {"phase33a_manifest_sha256": sha256(args.phase33a_dir / "manifest.sha256"), "phase31b_manifest_sha256": sha256(args.phase31b_dir / "manifest.sha256"), "raw_manifest_sha256": sha256(args.raw_dir / "source_manifest.json"), "model_features_sha256": sha256(args.model_features)},
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "audit_summary.json", {"schema_version": "phase33b-audit-v1", "status": status, "checks": checks, "raw_source_checks": raw_checks})
    write_json(args.output_dir / "logs/build.log", {"event": "phase33b_expanded_development_teacher_built", "status": status, "new_profiles": len(profiles), "combined_profiles": 94, "combined_target_rows": len(combined_targets), "blind_target_generated": False})
    (args.output_dir / "README.md").write_text(f"""# Phase 33B：扩充后的开发Hfull数据

本阶段只为Phase33A冻结的45个新增开发窗口生成Hfull teacher，并与Phase31的49个开发窗口合并。最终开发侧为75个训练、19个验证，共94个请求级互斥画像；完整teacher请求从21,058个增加到{21058 + sum(len(value) for value in request_windows.values()):,}个。

TP与PP各有{len(combined_examples['tp']):,}条phase训练样本，覆盖三个已知模型、三种并行规模、三种policy和prefill/decode。TP继续使用GPU验证过的full-window结构公式，PP继续使用scheduler-faithful teacher。完整请求列表只在构建内存中使用，没有保存或进入特征。

9个Phase33盲确认窗口仍只有低维特征与compact H0，Hfull target尚未生成；其example id与本阶段全部target互斥。
""")
    (args.output_dir / "DONE").write_text(status + "\n")
    manifest = [f"{sha256(path)}  {path.relative_to(args.output_dir)}" for path in sorted(args.output_dir.rglob("*")) if path.is_file() and path.name != "manifest.sha256"]
    (args.output_dir / "manifest.sha256").write_text("\n".join(manifest) + "\n")
    if status != "PASS":
        raise RuntimeError(checks)
    print(json.dumps({"status": status, "combined_profiles": 94, "combined_requests": summary["full_window_requests"]["combined"], "tp_rows": len(combined_examples["tp"]), "pp_rows": len(combined_examples["pp"]), "blind_target_generated": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
