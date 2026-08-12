#!/usr/bin/env python3
"""Generate second-confirmation TP Hfull targets after predictions are frozen."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np

from build_phase25_full_window_teacher import PHASES, normalize, tp_batches, tp_histograms
from build_phase27b_pp_hfull_dataset import HISTORY_SECONDS, summarize_profile
from build_phase29b_tp_hfull_dataset import (
    MODELS,
    STRATEGIES,
    TEACHER_KIND,
    TEACHER_STATUS,
    TP_SIZES,
    all_model_features,
    target_row,
)
from prepare_phase15_trace_windows import BURST_FILES, MOONCAKE_FILES, load_segment


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument(
        "--windows",
        type=Path,
        default=root
        / "experiment-results/phase29a_tp_aligned_contract/windows/phase28_second_confirmation_windows.csv",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=root
        / "experiment-results/phase29b_tp_hfull_dataset/dataset/second_confirmation_features.csv.gz",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=root
        / "experiment-results/phase29c_tp_aligned_training/analysis/second_confirmation_predictions.csv.gz",
    )
    parser.add_argument(
        "--phase29b-summary",
        type=Path,
        default=root / "experiment-results/phase29b_tp_hfull_dataset/summary.json",
    )
    parser.add_argument(
        "--phase29c-audit",
        type=Path,
        default=root
        / "experiment-results/phase29c_tp_aligned_training/audit_summary.json",
    )
    parser.add_argument(
        "--phase29d1-summary",
        type=Path,
        default=root
        / "experiment-results/phase29d1_tp_first_confirmation/summary.json",
    )
    parser.add_argument(
        "--model-features",
        type=Path,
        default=root / "experiment-results/phase16_model_features/model_features.json",
    )
    parser.add_argument(
        "--phase26a-summary",
        type=Path,
        default=root
        / "experiment-results/phase26a_tp_hfull_teacher_audit/summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root
        / "experiment-results/phase29d2_tp_second_confirmation_targets",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as source:
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
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as output:
            output.write(buffer.getvalue().encode())


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def verify_raw(raw_dir: Path, official_manifest: Path) -> dict[str, bool]:
    raw_manifest = json.loads((raw_dir / "source_manifest.json").read_text())
    official = json.loads(official_manifest.read_text())
    official_by_name = {row["name"]: row for row in official["sources"]}
    checks = {}
    for source in raw_manifest["sources"]:
        path = raw_dir / source["name"]
        expected = official_by_name[source["name"]]
        checks[source["name"]] = (
            path.stat().st_size == int(expected["actual_size"])
            and sha256(path) == expected["sha256"]
        )
    return checks


def main() -> None:
    args = parse_args()
    for name in ("labels", "analysis", "logs"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    phase29b = json.loads(args.phase29b_summary.read_text())
    phase29c = json.loads(args.phase29c_audit.read_text())
    phase29d1 = json.loads(args.phase29d1_summary.read_text())
    phase26a = json.loads(args.phase26a_summary.read_text())
    if phase29b["status"] != "PASS" or phase29c["status"] != "PASS":
        raise ValueError("upstream Phase 29B/29C is not PASS")
    if phase29d1["status"] != "PASS" or phase26a["status"] != "PASS":
        raise ValueError("upstream Phase 29D1/26A is not PASS")
    if phase29b["second_confirmation_hfull_targets_generated"]:
        raise RuntimeError("second targets existed before prediction freeze")
    frozen_prediction_hash = phase29c["second_confirmation_predictions_sha256"]
    if sha256(args.predictions) != frozen_prediction_hash:
        raise RuntimeError("second-confirmation prediction hash mismatch")
    if phase29d1["second_confirmation_predictions_sha256"] != frozen_prediction_hash:
        raise RuntimeError("Phase 29D1 did not carry forward the frozen prediction hash")

    official_manifest = root / "experiment-results/phase15_trace_data/source_manifest.json"
    raw_checks = verify_raw(args.raw_dir, official_manifest)
    if len(raw_checks) != 6 or not all(raw_checks.values()):
        raise RuntimeError(raw_checks)
    windows = read_csv(args.windows)
    features = read_csv_gz(args.features)
    predictions = read_csv_gz(args.predictions)
    if len(windows) != 18 or len(features) != 972 or len(predictions) != 3888:
        raise ValueError("unexpected second-confirmation input counts")
    if any(name.startswith("target_") for name in features[0]):
        raise RuntimeError("second confirmation feature artifact contains targets")

    file_by_segment = {
        segment: args.raw_dir / name
        for name, (segment, _) in {**BURST_FILES, **MOONCAKE_FILES}.items()
    }
    arrays = {segment: load_segment(path) for segment, path in file_by_segment.items()}
    profiles = []
    requests_by_profile = {}
    count_checks = []
    for selected in windows:
        timestamps, inputs, outputs = arrays[selected["segment"]]
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
        profiles.append(profile)
        requests_by_profile[profile["profile_id"]] = requests
        count_checks.append(len(requests) == int(selected["history_count"]))

    models = all_model_features(args.model_features)
    targets = []
    batch_checks = []
    for profile in profiles:
        requests = requests_by_profile[profile["profile_id"]]
        for model_name in MODELS:
            model, _ = models[model_name]
            for policy, strategy in STRATEGIES.items():
                histograms = tp_histograms(requests, strategy, model)
                batch_checks.append(
                    sum(len(batch) for batch in tp_batches(requests, strategy))
                    == len(requests)
                )
                for tp_size in TP_SIZES:
                    for phase in PHASES:
                        histogram = normalize(histograms[phase], len(requests))
                        targets.append(
                            target_row(
                                profile,
                                model,
                                tp_size,
                                policy,
                                phase,
                                histogram,
                            )
                        )

    prediction_ids = {row["training_id"] for row in predictions}
    feature_ids = {row["training_id"] for row in features}
    target_ids = {row["label_id"] for row in targets}
    inventory = []
    for profile in profiles:
        inventory.append(
            {
                "profile_id": profile["profile_id"],
                "role": profile["phase27_role"],
                "source": profile["source"],
                "segment": profile["segment"],
                "window_id": profile["window_id"],
                "full_window_requests": profile["request_count"],
                "models": len(MODELS),
                "tp_sizes": len(TP_SIZES),
                "policies": len(STRATEGIES),
                "target_phase_rows": len(MODELS)
                * len(TP_SIZES)
                * len(STRATEGIES)
                * len(PHASES),
            }
        )
    write_csv_gz(
        args.output_dir / "labels/second_confirmation_hfull_targets.csv.gz",
        targets,
    )
    write_csv(args.output_dir / "analysis/profile_inventory.csv", inventory)

    role_counts = Counter(profile["phase27_role"] for profile in profiles)
    checks = {
        "raw_source_hashes_6_of_6": len(raw_checks) == 6
        and all(raw_checks.values()),
        "predictions_frozen_before_targets": sha256(args.predictions)
        == frozen_prediction_hash
        and not phase29b["second_confirmation_hfull_targets_generated"],
        "profiles_18_all_second_confirmation": len(profiles) == 18
        and role_counts == Counter({"second_independent_confirmation": 18}),
        "history_counts_match_18_of_18": all(count_checks),
        "targets_972_unique": len(targets) == 972 and len(target_ids) == 972,
        "prediction_feature_target_ids_exact": prediction_ids
        == feature_ids
        == target_ids,
        "batch_partition_checks_162_of_162": len(batch_checks) == 162
        and all(batch_checks),
        "teacher_promoted_phase26a": phase26a["promoted_labels"] == 1296
        and phase26a["promoted_label_status"] == TEACHER_STATUS,
        "no_request_arrays_saved": not any(
            name in targets[0]
            for name in (
                "input_lens",
                "output_lens",
                "requests_json",
                "full_request_list",
            )
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    if status != "PASS":
        raise RuntimeError(checks)
    total_requests = sum(int(profile["request_count"]) for profile in profiles)
    summary = {
        "schema_version": "phase29d2-tp-second-confirmation-targets-v1",
        "status": status,
        "objective": "generate Hfull targets for the untouched second confirmation windows only after all method predictions and the post-first mapping were frozen",
        "profiles": len(profiles),
        "full_window_requests": total_requests,
        "target_phase_rows": len(targets),
        "models": list(MODELS),
        "tp_sizes": list(TP_SIZES),
        "policies": list(STRATEGIES),
        "teacher_status": TEACHER_STATUS,
        "teacher_kind": TEACHER_KIND,
        "frozen_prediction_sha256": frozen_prediction_hash,
        "second_confirmation_frozen_mapping": phase29d1[
            "second_confirmation_frozen_mapping"
        ],
        "target_generated_after_prediction_and_mapping_freeze": True,
        "inputs": {
            "windows_sha256": sha256(args.windows),
            "features_sha256": sha256(args.features),
            "predictions_sha256": sha256(args.predictions),
            "phase29b_summary_sha256": sha256(args.phase29b_summary),
            "phase29c_audit_sha256": sha256(args.phase29c_audit),
            "phase29d1_summary_sha256": sha256(args.phase29d1_summary),
            "model_features_sha256": sha256(args.model_features),
            "phase26a_summary_sha256": sha256(args.phase26a_summary),
            "raw_manifest_sha256": sha256(args.raw_dir / "source_manifest.json"),
        },
        "raw_source_checks": raw_checks,
        "checks": checks,
        "next_step": "archive these targets, then evaluate the already-frozen Phase29C predictions and Phase29D1 mapping without retraining",
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(
        args.output_dir / "audit_summary.json",
        {
            "schema_version": "phase29d2-target-generation-audit-v1",
            "status": status,
            "checks": checks,
            "prediction_hash_verified_before_target_generation": True,
            "request_arrays_saved": False,
        },
    )
    (args.output_dir / "README.md").write_text(
        f"""# Phase 29D2：TP第二独立确认Hfull真值

状态：**{status}**。本阶段在Phase 29C的第二确认四方法预测和Phase 29D1的分策略映射均已
写入Git并通过hash冻结后，才首次为Phase 28的18个第二独立窗口生成Hfull teacher真值。

共读取{total_requests:,}个完整历史请求，覆盖3个模型、TP2/4/8、3种固定策略和
prefill/decode，生成{len(targets):,}条按1000请求归一化的TP原生12桶标签。teacher是
Phase 26A经四个GPU sentinel精确验证并提升状态的fixed-draining结构公式，不需逐窗口重跑GPU。

完整请求数组只在构建器内存中用于生成标签，没有写入任何正式文件或Git。预测hash、映射、
窗口、模型配置、teacher审计和6份公共trace hash均已记录；本阶段只生成真值，没有训练、
模型选择或评测，因此不能从本目录单独得出泛化结论。
"""
    )
    (args.output_dir / "DONE").write_text("PASS\n")
    write_json(
        args.output_dir / "logs/build.log",
        {
            "schema_version": "phase29d2-build-log-v1",
            "status": status,
            "profiles": len(profiles),
            "full_window_requests": total_requests,
            "target_phase_rows": len(targets),
            "prediction_hash_verified_before_target_generation": True,
            "training_performed": False,
            "request_arrays_saved": False,
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
                "profiles": len(profiles),
                "full_window_requests": total_requests,
                "target_phase_rows": len(targets),
                "frozen_mapping": phase29d1["second_confirmation_frozen_mapping"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
