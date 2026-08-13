#!/usr/bin/env python3
"""Open Phase34 blind targets after Git-archived freeze and evaluate six models."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import evaluate_phase31d_known_model_fixed_predictions as eval31
from build_phase31b_known_model_hfull_dataset import HISTORY_SECONDS, all_model_features, summarize_profile
from prepare_phase15_trace_windows import BURST_FILES, MOONCAKE_FILES, load_segment


SIX_MODELS = (
    "deepseek-v2-lite", "qwen3-8b", "qwen3-30b-a3b",
    "llama-3.2-3b-instruct", "qwen2.5-14b-instruct", "mixtral-8x7b-instruct-v0.1",
)
EXISTING_MODELS = SIX_MODELS[:3]
METHODS = {"h0", "h0_plus_dnn_residual"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--phase34a-dir", type=Path, default=root / "experiment-results/phase34a_six_model_contract")
    parser.add_argument("--phase34b-dir", type=Path, default=root / "experiment-results/phase34b_six_model_hfull_dataset")
    parser.add_argument("--phase34c-dir", type=Path, default=root / "experiment-results/phase34c_six_model_target_free_training")
    parser.add_argument("--phase33d-dir", type=Path, default=root / "experiment-results/phase33d_blind_confirmation_evaluation")
    parser.add_argument("--output-dir", type=Path, default=root / "experiment-results/phase34d_six_model_blind_evaluation")
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
    for row in rows[1:]: fields.extend(name for name in row if name not in fields)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def write_csv_gz(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    for row in rows[1:]: fields.extend(name for name in row if name not in fields)
    buffer = io.StringIO(newline=""); writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    deterministic_gzip(path, buffer.getvalue())


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def load_blind_requests(args: argparse.Namespace) -> tuple[list[dict], dict[str, list[tuple[int, int]]], dict[str, bool]]:
    selection = [row for row in read_csv(args.phase34a_dir / "selection/selected_windows.csv") if row["role"] == "blind_confirmation"]
    if len(selection) != 12: raise RuntimeError(f"expected 12 blind profiles, got {len(selection)}")
    manifest = json.loads((args.raw_dir / "source_manifest.json").read_text())
    raw_checks = {row["name"]: (args.raw_dir / row["name"]).stat().st_size == int(row["actual_size"]) and sha256(args.raw_dir / row["name"]) == row["sha256"] for row in manifest["sources"]}
    if len(raw_checks) != 6 or not all(raw_checks.values()): raise RuntimeError(raw_checks)
    file_by_segment = {segment: args.raw_dir / name for name, (segment, _) in {**BURST_FILES, **MOONCAKE_FILES}.items()}
    arrays = {segment: load_segment(file_by_segment[segment]) for segment in sorted({row["segment"] for row in selection})}
    profiles, requests = [], {}
    for selected in selection:
        timestamps, inputs, outputs = arrays[selected["segment"]]; cutoff = int(selected["cutoff_ms"])
        left = int(np.searchsorted(timestamps, cutoff - HISTORY_SECONDS * 1000, side="left")); right = int(np.searchsorted(timestamps, cutoff, side="left"))
        compatible = {**selected, "phase27_profile_id": selected["profile_id"], "phase27_role": "blind_confirmation"}
        profile, window = summarize_profile(compatible, timestamps[left:right], inputs[left:right], outputs[left:right])
        profile["split_role"] = profile.pop("phase27_role")
        if len(window) != int(selected["history_count"]): raise RuntimeError(f"request count mismatch: {selected['profile_id']}")
        profiles.append(profile); requests[profile["profile_id"]] = window
    return profiles, requests, raw_checks


def strict_closure(metrics: list[dict], parallelism: str, models: tuple[str, ...]) -> dict:
    eval31.MODELS = models
    result = eval31.closure(metrics, parallelism)
    if parallelism == "tp" and result["decision"] != "formal_pass":
        result["legacy_conditional_ignored"] = result["decision"] == "conditional_pass"; result["decision"] = "fail"
    result["phase34_standard"] = "TP calls WAPE formal line is 10%; PP retains Phase33 model and MB16 official guards."
    return result


def evaluate_set(predictions: list[dict[str, str]], targets: list[dict], evidence_set: str, models: tuple[str, ...]) -> tuple[list[dict], list[dict], dict]:
    records = eval31.evaluation_records(predictions, targets)
    for row in records: row["evidence_set"] = evidence_set
    metrics = eval31.aggregate(records)
    for row in metrics: row["evidence_set"] = evidence_set
    decisions = {p: strict_closure(metrics, p, models) for p in ("tp", "pp")}
    return records, metrics, decisions


def comparison_row(phase34: dict, phase33: dict, parallelism: str) -> dict:
    return {
        "parallelism": parallelism,
        "phase34_calls_wape": phase34["calls_wape"], "phase33_calls_wape": phase33["calls_wape"],
        "phase34_vs_phase33_calls_relative_change": float(phase34["calls_wape"]) / max(float(phase33["calls_wape"]), 1e-12) - 1.0,
        "phase34_cost_wape": phase34["common_reference_cost_wape"], "phase33_cost_wape": phase33["common_reference_cost_wape"],
        "phase34_vs_phase33_cost_relative_change": float(phase34["common_reference_cost_wape"]) / max(float(phase33["common_reference_cost_wape"]), 1e-12) - 1.0,
        "phase34_tv": phase34["mean_histogram_tv"], "phase33_tv": phase33["mean_histogram_tv"],
        "phase34_emd": phase34["mean_normalized_log_payload_emd"], "phase33_emd": phase33["mean_normalized_log_payload_emd"],
    }


def minimal_svg(path: Path, decisions: dict) -> None:
    width, height = 1040, 500; keys = (("calls_wape", "calls"), ("bytes_wape", "bytes"), ("mean_histogram_tv", "TV"), ("mean_normalized_log_payload_emd", "EMD"), ("common_reference_cost_wape", "cost"))
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', '<text x="45" y="35" font-family="sans-serif" font-size="22">Phase34六模型全新盲测：H0与H0+DNN residual</text>']
    for panel, parallelism in enumerate(("tp", "pp")):
        x, y0 = 45 + panel * 500, 78; value = decisions[parallelism]
        svg.append(f'<text x="{x}" y="{y0}" font-family="sans-serif" font-size="16">{parallelism.upper()} / {value["decision"]}</text>')
        for offset, (key, label) in enumerate(keys):
            y = y0 + 35 + offset * 70; h0 = min(float(value["h0"][key]) * 1100, 310); dnn = min(float(value["h0_plus_dnn_residual"][key]) * 1100, 310)
            svg.append(f'<text x="{x}" y="{y+12}" font-family="sans-serif" font-size="12">{label}</text><rect x="{x+55}" y="{y}" width="{h0:.2f}" height="13" fill="#94a3b8"/><rect x="{x+55}" y="{y+19}" width="{dnn:.2f}" height="13" fill="#2563eb"/>')
    svg.append('</svg>'); path.write_text("\n".join(svg) + "\n")


def main() -> None:
    args = parse_args()
    for name in ("labels", "analysis", "figures", "logs", "docs"): (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    a = json.loads((args.phase34a_dir / "summary.json").read_text()); b = json.loads((args.phase34b_dir / "summary.json").read_text()); c = json.loads((args.phase34c_dir / "summary.json").read_text())
    prediction_path = args.phase34c_dir / "analysis/frozen_predictions_all_versions.csv.gz"
    if a["blind_confirmation"]["target_state"] != "not_generated" or b["blind_confirmation_target_state"] != "not_generated": raise RuntimeError("target not sealed at evaluation start")
    if c["status"] != "PASS" or c["target_isolation"]["phase34_blind_targets_read"] is not False: raise RuntimeError("target-free Phase34C PASS missing")
    if sha256(prediction_path) != c["frozen_prediction_sha256"]: raise RuntimeError("frozen prediction SHA mismatch")
    archived_commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", str(args.phase34c_dir)], text=True).strip()
    if not archived_commit: raise RuntimeError("Phase34C has no archived Git commit")

    predictions = read_csv(prediction_path)
    profiles, request_windows, raw_checks = load_blind_requests(args)
    model_map = all_model_features(args.phase34a_dir / "model_configs/model_features_six_models.json")
    if tuple(model_map) != SIX_MODELS: raise RuntimeError("model inventory mismatch")
    eval31.MODELS = SIX_MODELS
    blind_targets = eval31.generate_targets(profiles, request_windows, model_map)
    old_targets = read_csv(args.phase33d_dir / "labels/phase33_blind_hfull_targets.csv.gz")

    phase34_new = [row for row in predictions if row["prediction_set"] == "phase34_blind_new" and row["model_version"] == "phase34_six_model"]
    phase34_new_old3 = [row for row in phase34_new if row["model"] in EXISTING_MODELS]
    phase33_baseline = [row for row in predictions if row["prediction_set"] == "phase34_blind_new" and row["model_version"] == "phase33_three_model_baseline"]
    phase34_repeated = [row for row in predictions if row["prediction_set"] == "phase33_blind_repeated" and row["model_version"] == "phase34_six_model"]
    target_old3 = [row for row in blind_targets if row["model"] in EXISTING_MODELS]
    sets = {
        "phase34_blind_six_model": (phase34_new, blind_targets, SIX_MODELS),
        "phase34_blind_phase34_original_three": (phase34_new_old3, target_old3, EXISTING_MODELS),
        "phase34_blind_phase33_baseline_original_three": (phase33_baseline, target_old3, EXISTING_MODELS),
        "phase33_blind_repeated_phase34_original_three": (phase34_repeated, old_targets, EXISTING_MODELS),
    }
    all_records, all_metrics, decisions = [], [], {}
    for evidence, (pred, targets, models) in sets.items():
        records, metrics, decision = evaluate_set(pred, targets, evidence, models)
        all_records.extend(records); all_metrics.extend(metrics); decisions[evidence] = decision

    comparisons = []
    for parallelism in ("tp", "pp"):
        comparisons.append(comparison_row(decisions["phase34_blind_phase34_original_three"][parallelism]["h0_plus_dnn_residual"], decisions["phase34_blind_phase33_baseline_original_three"][parallelism]["h0_plus_dnn_residual"], parallelism))
    write_csv_gz(args.output_dir / "labels/phase34_blind_six_model_hfull_targets.csv.gz", blind_targets)
    write_csv_gz(args.output_dir / "analysis/per_case_metrics.csv.gz", all_records)
    write_csv(args.output_dir / "analysis/aggregate_metrics.csv", all_metrics)
    write_csv(args.output_dir / "analysis/phase34_vs_phase33_same_blind_comparison.csv", comparisons)
    minimal_svg(args.output_dir / "figures/six_model_blind_evaluation.svg", decisions["phase34_blind_six_model"])

    primary = decisions["phase34_blind_six_model"]
    checks = {
        "target_absent_at_phase34c_freeze": a["blind_confirmation"]["target_state"] == "not_generated" and b["blind_confirmation_target_state"] == "not_generated",
        "phase34c_archived_before_target_generation": bool(archived_commit),
        "frozen_prediction_sha_matches": sha256(prediction_path) == c["frozen_prediction_sha256"],
        "blind_profiles_12": len(profiles) == 12,
        "blind_requests_3803": sum(len(value) for value in request_windows.values()) == 3803,
        "blind_six_model_target_phase_rows_2592": len(blind_targets) == 2592,
        "six_model_prediction_phase_rows_5184": len(phase34_new) == 5184,
        "phase33_baseline_frozen_same_windows_original_three": len(phase33_baseline) == 2592,
        "phase33_repeated_engineering_prediction_rows_1944": len(phase34_repeated) == 1944,
        "all_metrics_finite": all(math.isfinite(float(row[key])) for row in all_metrics for key in ("calls_mape", "calls_wape", "bytes_mape", "bytes_wape", "mean_histogram_tv", "mean_normalized_log_payload_emd", "common_reference_cost_mape", "common_reference_cost_wape")),
        "raw_sources_pass": len(raw_checks) == 6 and all(raw_checks.values()),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = {
        "schema_version": "phase34d-six-model-new-blind-evaluation-v1", "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "primary_evidence": "phase34_blind_six_model",
        "secondary_evidence": ["phase34_blind_phase34_original_three", "phase34_blind_phase33_baseline_original_three", "phase33_blind_repeated_phase34_original_three"],
        "blind_scope": "12 fresh request-disjoint normal BurstGPT windows; six models and all TP/PP configurations",
        "counts": {"blind_profiles": len(profiles), "blind_full_teacher_requests": sum(len(value) for value in request_windows.values()), "blind_target_phase_rows": len(blind_targets), "blind_prediction_phase_rows": len(phase34_new), "all_per_case_rows_including_total": len(all_records)},
        "decisions": decisions, "phase34_vs_phase33_same_new_blind_original_three": comparisons,
        "frozen_prediction_sha256": c["frozen_prediction_sha256"], "phase34c_archived_commit": archived_commit,
        "evidence_limit": "Phase34 new blind confirmation is BurstGPT-only because Mooncake capacity is exhausted under accumulated 300-second embargo. Phase33 windows are repeated engineering evidence. The six models are all represented in training and validation; this is not unseen-model generalization.",
        "checks": checks,
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "audit_summary.json", {"schema_version": "phase34d-audit-v1", "status": status, "checks": checks, "raw_source_checks": raw_checks, "frozen_prediction_sha256": c["frozen_prediction_sha256"], "phase34c_archived_commit": archived_commit})
    write_json(args.output_dir / "logs/evaluation.log", {"event": "phase34d_blind_target_open_and_six_model_evaluation_complete", "status": status, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "repository_head_at_evaluation": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "primary_decisions": {p: primary[p]["decision"] for p in ("tp", "pp")}})
    (args.output_dir / "README.md").write_text(f"""# Phase 34D：六模型全新盲测与Phase33同窗基线

Phase34C的TP/PP模型、18个checkpoint、全部冻结预测和SHA已先由Git提交`{archived_commit}`归档，本阶段才一次性生成12个全新BurstGPT请求级互斥窗口的Hfull target。新盲测包含3,803个完整teacher请求、六个模型和全部TP/PP配置。

六模型新盲测裁定：TP=`{primary['tp']['decision']}`，PP=`{primary['pp']['decision']}`。TP继续使用calls WAPE≤10%的正式线；PP继续使用Phase33逐模型和MB16专门保护条件。整体、逐模型、逐policy、逐并行规模指标见`analysis/aggregate_metrics.csv`。

为了可比，Phase33三模型incumbent和Phase34六模型predictor都在开target前对同一12个新窗口的原三个模型冻结了预测；对比见`analysis/phase34_vs_phase33_same_blind_comparison.csv`。Phase33原9个窗口的Phase34复评只能作为重复工程证据。

证据边界：新盲测仍是BurstGPT-only，不能声称跨数据源或未见模型泛化；六模型全部进入了训练和验证。
""")
    (args.output_dir / "DONE").write_text(status + "\n")
    manifest = [f"{sha256(path)}  {path.relative_to(args.output_dir)}" for path in sorted(args.output_dir.rglob("*")) if path.is_file() and path.name != "manifest.sha256"]
    (args.output_dir / "manifest.sha256").write_text("\n".join(manifest) + "\n")
    if status != "PASS": raise RuntimeError(checks)
    print(json.dumps({"status": status, "primary_decisions": {p: primary[p]["decision"] for p in ("tp", "pp")}, "primary_metrics": {p: primary[p]["h0_plus_dnn_residual"] for p in ("tp", "pp")}, "comparison": comparisons}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
