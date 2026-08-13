#!/usr/bin/env python3
"""Open Phase33 blind Hfull targets after archived frozen predictions and evaluate."""

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

from build_phase31b_known_model_hfull_dataset import HISTORY_SECONDS, MODELS, all_model_features, summarize_profile
from evaluate_phase31d_known_model_fixed_predictions import aggregate, closure, evaluation_records, generate_targets
from prepare_phase15_trace_windows import BURST_FILES, MOONCAKE_FILES, load_segment


METHODS = {"h0", "h0_plus_dnn_residual"}
EVIDENCE = ("phase33_blind", "phase31_fixed_repeated", "phase32_confirmation_repeated")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--phase33a-dir", type=Path, default=root / "experiment-results/phase33a_fresh_data_contract")
    parser.add_argument("--phase33b-dir", type=Path, default=root / "experiment-results/phase33b_expanded_development_dataset")
    parser.add_argument("--phase33c-dir", type=Path, default=root / "experiment-results/phase33c_target_free_model_selection")
    parser.add_argument("--phase31d-dir", type=Path, default=root / "experiment-results/phase31d_known_model_fixed_evaluation")
    parser.add_argument("--phase32c-dir", type=Path, default=root / "experiment-results/phase32c_frozen_prediction_evaluation")
    parser.add_argument("--model-features", type=Path, default=root / "experiment-results/phase16_model_features/model_features.json")
    parser.add_argument("--output-dir", type=Path, default=root / "experiment-results/phase33d_blind_confirmation_evaluation")
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
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def load_blind_requests(args: argparse.Namespace) -> tuple[list[dict], dict[str, list[tuple[int, int]]], dict[str, bool]]:
    selection = [row for row in read_csv(args.phase33a_dir / "selection/selected_windows.csv") if row["role"] == "blind_confirmation"]
    if len(selection) != 9:
        raise RuntimeError(f"expected 9 Phase33 blind profiles, got {len(selection)}")
    manifest = json.loads((args.raw_dir / "source_manifest.json").read_text())
    raw_checks = {}
    for row in manifest["sources"]:
        path = args.raw_dir / row["name"]
        raw_checks[row["name"]] = path.stat().st_size == int(row["actual_size"]) and sha256(path) == row["sha256"]
    if len(raw_checks) != 6 or not all(raw_checks.values()):
        raise RuntimeError(raw_checks)
    file_by_segment = {segment: args.raw_dir / name for name, (segment, _) in {**BURST_FILES, **MOONCAKE_FILES}.items()}
    arrays = {segment: load_segment(file_by_segment[segment]) for segment in sorted({row["segment"] for row in selection})}
    profiles, requests = [], {}
    for selected in selection:
        timestamps, inputs, outputs = arrays[selected["segment"]]
        cutoff = int(selected["cutoff_ms"])
        left = int(np.searchsorted(timestamps, cutoff - HISTORY_SECONDS * 1000, side="left"))
        right = int(np.searchsorted(timestamps, cutoff, side="left"))
        compatible = {**selected, "phase27_profile_id": selected["profile_id"], "phase27_role": "blind_confirmation"}
        profile, window = summarize_profile(compatible, timestamps[left:right], inputs[left:right], outputs[left:right])
        profile["split_role"] = profile.pop("phase27_role")
        if len(window) != int(selected["history_count"]):
            raise RuntimeError(f"history count mismatch: {selected['profile_id']}")
        profiles.append(profile); requests[profile["profile_id"]] = window
    return profiles, requests, raw_checks


def strict_decision(metrics: list[dict], parallelism: str) -> dict:
    result = closure(metrics, parallelism)
    if parallelism == "tp" and result["decision"] != "formal_pass":
        result["legacy_phase31_conditional_ignored"] = result["decision"] == "conditional_pass"
        result["decision"] = "fail"
    result["phase33_standard"] = "TP has no 12% conditional line; PP retains formal closure including model and MB16 guards."
    return result


def minimal_svg(path: Path, decisions: dict) -> None:
    width, height = 1120, 560
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', '<text x="45" y="35" font-family="sans-serif" font-size="22">Phase33：新盲测与重复工程证据</text>']
    panels = [(evidence, parallelism) for evidence in EVIDENCE for parallelism in ("tp", "pp")]
    for index, (evidence, parallelism) in enumerate(panels):
        x = 45 + (index % 3) * 355; y0 = 75 + (index // 3) * 235
        value = decisions[evidence][parallelism]
        svg.append(f'<text x="{x}" y="{y0}" font-family="sans-serif" font-size="14">{evidence} / {parallelism.upper()} / {value["decision"]}</text>')
        for offset, (key, label) in enumerate((("calls_wape", "calls"), ("bytes_wape", "bytes"), ("mean_histogram_tv", "TV"), ("mean_normalized_log_payload_emd", "EMD"), ("common_reference_cost_wape", "cost"))):
            y = y0 + 25 + offset * 37
            h0 = min(float(value["h0"][key]) * 850, 200); dnn = min(float(value["h0_plus_dnn_residual"][key]) * 850, 200)
            svg.append(f'<text x="{x}" y="{y+11}" font-family="sans-serif" font-size="11">{label}</text><rect x="{x+45}" y="{y}" width="{h0:.2f}" height="11" fill="#94a3b8"/><rect x="{x+45}" y="{y+14}" width="{dnn:.2f}" height="11" fill="#2563eb"/>')
    svg.append('</svg>'); path.write_text("\n".join(svg) + "\n")


def main() -> None:
    args = parse_args()
    for name in ("labels", "analysis", "figures", "logs"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    phase33a = json.loads((args.phase33a_dir / "summary.json").read_text())
    phase33b = json.loads((args.phase33b_dir / "summary.json").read_text())
    phase33c = json.loads((args.phase33c_dir / "summary.json").read_text())
    prediction_path = args.phase33c_dir / "analysis/frozen_predictions.csv.gz"
    if phase33a["target_state"]["blind_confirmation"] != "not_generated" or phase33b["blind_confirmation_target_state"] != "not_generated":
        raise RuntimeError("blind target was not sealed at evaluation start")
    if phase33c["status"] != "PASS" or phase33c["target_isolation"]["phase33_blind_targets_read"] is not False:
        raise RuntimeError("Phase33C target-free PASS missing")
    if sha256(prediction_path) != phase33c["frozen_prediction_sha256"]:
        raise RuntimeError("frozen prediction SHA mismatch")
    archived_commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", str(args.phase33c_dir)], text=True).strip()
    if not archived_commit:
        raise RuntimeError("Phase33C has no archived Git commit")

    predictions = read_csv(prediction_path)
    if {row["prediction_set"] for row in predictions} != set(EVIDENCE) or {row["method"] for row in predictions} != METHODS:
        raise RuntimeError("frozen prediction inventory mismatch")
    profiles, request_windows, raw_checks = load_blind_requests(args)
    model_map = all_model_features(args.model_features)
    if set(model_map) != set(MODELS):
        raise RuntimeError("model inventory mismatch")
    blind_targets = generate_targets(profiles, request_windows, model_map)
    targets_by_evidence = {
        "phase33_blind": blind_targets,
        "phase31_fixed_repeated": read_csv(args.phase31d_dir / "labels/fixed_prediction_hfull_targets.csv.gz"),
        "phase32_confirmation_repeated": read_csv(args.phase32c_dir / "labels/new_confirmation_hfull_targets.csv.gz"),
    }

    all_records, all_metrics, decisions = [], [], {}
    for evidence in EVIDENCE:
        evidence_predictions = [row for row in predictions if row["prediction_set"] == evidence]
        records = evaluation_records(evidence_predictions, targets_by_evidence[evidence])
        for row in records: row["evidence_set"] = evidence
        metrics = aggregate(records)
        for row in metrics: row["evidence_set"] = evidence
        all_records.extend(records); all_metrics.extend(metrics)
        decisions[evidence] = {parallelism: strict_decision(metrics, parallelism) for parallelism in ("tp", "pp")}

    write_csv_gz(args.output_dir / "labels/phase33_blind_hfull_targets.csv.gz", blind_targets)
    write_csv_gz(args.output_dir / "analysis/per_case_metrics.csv.gz", all_records)
    write_csv(args.output_dir / "analysis/aggregate_metrics.csv", all_metrics)
    minimal_svg(args.output_dir / "figures/blind_and_repeated_evaluation.svg", decisions)
    checks = {
        "phase33_target_absent_at_freeze": phase33a["target_state"]["blind_confirmation"] == "not_generated" and phase33b["blind_confirmation_target_state"] == "not_generated",
        "phase33c_target_free_selection": phase33c["target_isolation"]["phase33_blind_targets_read"] is False,
        "phase33c_archived_before_target_generation": bool(archived_commit),
        "frozen_prediction_sha_matches": sha256(prediction_path) == phase33c["frozen_prediction_sha256"],
        "blind_profiles_9": len(profiles) == 9,
        "blind_requests_1742": sum(len(value) for value in request_windows.values()) == 1742,
        "blind_target_phase_rows_972": len(blind_targets) == 972,
        "blind_prediction_phase_rows_1944": sum(row["prediction_set"] == "phase33_blind" for row in predictions) == 1944,
        "repeated_target_rows_1080_and_972": len(targets_by_evidence["phase31_fixed_repeated"]) == 1080 and len(targets_by_evidence["phase32_confirmation_repeated"]) == 972,
        "all_metrics_finite": all(math.isfinite(float(row[key])) for row in all_metrics for key in ("calls_mape", "calls_wape", "bytes_mape", "bytes_wape", "mean_histogram_tv", "mean_normalized_log_payload_emd", "common_reference_cost_mape", "common_reference_cost_wape")),
        "raw_sources_pass": len(raw_checks) == 6 and all(raw_checks.values()),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = {
        "schema_version": "phase33d-blind-confirmation-evaluation-v1", "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_evidence": "phase33_blind", "secondary_evidence": ["phase31_fixed_repeated", "phase32_confirmation_repeated"],
        "blind_scope": "nine fresh request-disjoint normal BurstGPT windows; three known models and all TP/PP configurations",
        "counts": {"blind_profiles": len(profiles), "blind_full_teacher_requests": sum(len(value) for value in request_windows.values()), "blind_target_phase_rows": len(blind_targets), "blind_prediction_phase_rows": sum(row["prediction_set"] == "phase33_blind" for row in predictions), "all_per_case_rows_including_total": len(all_records)},
        "decisions": decisions, "frozen_prediction_sha256": phase33c["frozen_prediction_sha256"],
        "phase33c_archived_commit": archived_commit,
        "evidence_limit": "Phase33 fresh blind confirmation is BurstGPT-only because Mooncake capacity was exhausted under the accumulated 300-second embargo. Both older sets are repeated engineering evidence, not new blind tests.",
        "checks": checks,
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "audit_summary.json", {"schema_version": "phase33d-audit-v1", "status": status, "checks": checks, "raw_source_checks": raw_checks, "frozen_prediction_sha256": phase33c["frozen_prediction_sha256"], "phase33c_archived_commit": archived_commit})
    write_json(args.output_dir / "logs/evaluation.log", {"event": "phase33d_blind_target_open_and_evaluation_complete", "status": status, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "repository_head_at_evaluation": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "decisions": {evidence: {parallelism: value["decision"] for parallelism, value in directions.items()} for evidence, directions in decisions.items()}})
    (args.output_dir / "README.md").write_text(f"""# Phase 33D：全新盲测与重复工程评测

Phase33C模型、checkpoint、冻结预测和SHA已先由Git提交`{archived_commit}`归档，本阶段才生成9个全新BurstGPT请求级互斥窗口的Hfull target。新盲测包含1,742个完整teacher请求，覆盖三个已知模型和全部TP/PP配置。Phase31固定集与Phase32确认集没有更换，只标记为重复工程证据。

新盲测裁定：TP=`{decisions['phase33_blind']['tp']['decision']}`，PP=`{decisions['phase33_blind']['pp']['decision']}`。TP严格使用calls WAPE≤10%的正式线，不再接受12%有条件线。整体、逐模型、逐policy、逐并行规模、逐source指标见`analysis/aggregate_metrics.csv`，逐case结果见压缩明细。

证据边界：新盲测仅覆盖正常BurstGPT窗口，不能单独声称跨数据源泛化；旧两套结果只能作为重复工程稳定性证据。
""")
    (args.output_dir / "DONE").write_text(status + "\n")
    manifest = [f"{sha256(path)}  {path.relative_to(args.output_dir)}" for path in sorted(args.output_dir.rglob("*")) if path.is_file() and path.name != "manifest.sha256"]
    (args.output_dir / "manifest.sha256").write_text("\n".join(manifest) + "\n")
    if status != "PASS":
        raise RuntimeError(checks)
    print(json.dumps({"status": status, "decisions": {evidence: {parallelism: value["decision"] for parallelism, value in directions.items()} for evidence, directions in decisions.items()}, "blind": {parallelism: decisions["phase33_blind"][parallelism]["h0_plus_dnn_residual"] for parallelism in ("tp", "pp")}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
