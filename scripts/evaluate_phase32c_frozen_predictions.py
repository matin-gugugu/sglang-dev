#!/usr/bin/env python3
"""Open Phase32 confirmation targets only after frozen predictions are archived."""

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
from evaluate_phase31d_known_model_fixed_predictions import (
    aggregate,
    closure,
    evaluation_records,
    generate_targets,
)
from prepare_phase15_trace_windows import BURST_FILES, MOONCAKE_FILES, load_segment


METHODS = {"h0", "h0_plus_dnn_residual"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--phase32a-dir", type=Path, default=root / "experiment-results/phase32a_expanded_search_contract")
    parser.add_argument("--phase32b-dir", type=Path, default=root / "experiment-results/phase32b_expanded_residual_search")
    parser.add_argument("--phase31d-dir", type=Path, default=root / "experiment-results/phase31d_known_model_fixed_evaluation")
    parser.add_argument("--model-features", type=Path, default=root / "experiment-results/phase16_model_features/model_features.json")
    parser.add_argument("--output-dir", type=Path, default=root / "experiment-results/phase32c_frozen_prediction_evaluation")
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
    fieldnames = list(rows[0])
    for row in rows[1:]: fieldnames.extend(name for name in row if name not in fieldnames)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def write_csv_gz(path: Path, rows: list[dict]) -> None:
    buffer = io.StringIO(newline="")
    fieldnames = list(rows[0])
    for row in rows[1:]: fieldnames.extend(name for name in row if name not in fieldnames)
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    deterministic_gzip(path, buffer.getvalue())


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def load_new_confirmation(args: argparse.Namespace) -> tuple[list[dict], dict[str, list[tuple[int, int]]], dict]:
    selection = read_csv(args.phase32a_dir / "selection/selected_windows.csv")
    if len(selection) != 9 or {row["role"] for row in selection} != {"new_confirmation"}:
        raise RuntimeError("unexpected Phase32 confirmation selection")
    manifest = json.loads((args.raw_dir / "source_manifest.json").read_text())
    raw_checks = {row["name"]: (args.raw_dir / row["name"]).stat().st_size == int(row["actual_size"]) and sha256(args.raw_dir / row["name"]) == row["sha256"] for row in manifest["sources"]}
    if len(raw_checks) != 6 or not all(raw_checks.values()): raise RuntimeError(raw_checks)
    file_by_segment = {segment: args.raw_dir / name for name, (segment, _) in {**BURST_FILES, **MOONCAKE_FILES}.items()}
    arrays = {segment: load_segment(file_by_segment[segment]) for segment in sorted({row["segment"] for row in selection})}
    profiles, requests_by_profile = [], {}
    for selected in selection:
        timestamps, inputs, outputs = arrays[selected["segment"]]
        cutoff = int(selected["cutoff_ms"])
        left = int(np.searchsorted(timestamps, cutoff - HISTORY_SECONDS * 1000, side="left"))
        right = int(np.searchsorted(timestamps, cutoff, side="left"))
        compatible = {**selected, "phase27_profile_id": selected["profile_id"], "phase27_role": "new_confirmation"}
        profile, requests = summarize_profile(compatible, timestamps[left:right], inputs[left:right], outputs[left:right])
        profile["split_role"] = profile.pop("phase27_role")
        if len(requests) != int(selected["history_count"]): raise RuntimeError(f"history count mismatch: {selected['profile_id']}")
        profiles.append(profile); requests_by_profile[profile["profile_id"]] = requests
    return profiles, requests_by_profile, raw_checks


def minimal_svg(path: Path, decisions: dict) -> None:
    width, height = 1000, 520
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', '<text x="40" y="35" font-family="sans-serif" font-size="22">Phase32：冻结预测评测（H0 与 H0+DNN）</text>']
    x = 50
    for evidence in ("new_confirmation", "original_fixed_repeated"):
        for parallelism in ("tp", "pp"):
            value = decisions[evidence][parallelism]
            h0, dnn = value["h0"], value["h0_plus_dnn_residual"]
            svg.append(f'<text x="{x}" y="75" font-family="sans-serif" font-size="16">{evidence} / {parallelism.upper()} / {value["decision"]}</text>')
            for index, (key, label) in enumerate((("calls_wape", "calls"), ("bytes_wape", "bytes"), ("mean_histogram_tv", "TV"), ("common_reference_cost_wape", "cost"))):
                y = 105 + index * 75
                svg.append(f'<text x="{x}" y="{y+15}" font-family="sans-serif" font-size="13">{label}</text>')
                svg.append(f'<rect x="{x+55}" y="{y}" width="{min(float(h0[key])*800,150):.2f}" height="18" fill="#94a3b8"/>')
                svg.append(f'<rect x="{x+55}" y="{y+24}" width="{min(float(dnn[key])*800,150):.2f}" height="18" fill="#2563eb"/>')
            x += 235
    svg.append('</svg>')
    path.write_text("\n".join(svg) + "\n")


def main() -> None:
    args = parse_args()
    for name in ("labels", "analysis", "figures", "logs"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    phase32a = json.loads((args.phase32a_dir / "summary.json").read_text())
    phase32b = json.loads((args.phase32b_dir / "summary.json").read_text())
    prediction_path = args.phase32b_dir / "analysis/frozen_predictions.csv.gz"
    if phase32a["new_confirmation"]["target_state"] != "not_generated" or phase32b["fixed_targets_read"] is not False or phase32b["new_confirmation_targets_read"] is not False:
        raise RuntimeError("target isolation contract failed")
    if sha256(prediction_path) != phase32b["frozen_prediction_sha256"]: raise RuntimeError("frozen SHA mismatch")
    predictions = read_csv(prediction_path)
    new_predictions = [row for row in predictions if row["prediction_set"] == "new_confirmation"]
    fixed_predictions = [row for row in predictions if row["prediction_set"] == "original_fixed"]
    if {row["method"] for row in predictions} != METHODS: raise RuntimeError("method mismatch")

    profiles, request_windows, raw_checks = load_new_confirmation(args)
    model_map = all_model_features(args.model_features)
    if set(model_map) != set(MODELS): raise RuntimeError("model mismatch")
    new_targets = generate_targets(profiles, request_windows, model_map)
    original_targets = read_csv(args.phase31d_dir / "labels/fixed_prediction_hfull_targets.csv.gz")
    new_records = evaluation_records(new_predictions, new_targets)
    fixed_records = evaluation_records(fixed_predictions, original_targets)
    for row in new_records: row["evidence_set"] = "new_confirmation"
    for row in fixed_records: row["evidence_set"] = "original_fixed_repeated"
    new_metrics = aggregate(new_records); fixed_metrics = aggregate(fixed_records)
    for row in new_metrics: row["evidence_set"] = "new_confirmation"
    for row in fixed_metrics: row["evidence_set"] = "original_fixed_repeated"
    decisions = {
        "new_confirmation": {parallelism: closure(new_metrics, parallelism) for parallelism in ("tp", "pp")},
        "original_fixed_repeated": {parallelism: closure(fixed_metrics, parallelism) for parallelism in ("tp", "pp")},
    }

    write_csv_gz(args.output_dir / "labels/new_confirmation_hfull_targets.csv.gz", new_targets)
    write_csv_gz(args.output_dir / "analysis/per_case_metrics.csv.gz", new_records + fixed_records)
    write_csv(args.output_dir / "analysis/aggregate_metrics.csv", new_metrics + fixed_metrics)
    minimal_svg(args.output_dir / "figures/frozen_evaluation.svg", decisions)
    checks = {
        "phase32a_target_absent_at_freeze": phase32a["new_confirmation"]["target_state"] == "not_generated",
        "phase32b_target_free_training": phase32b["fixed_targets_read"] is False and phase32b["new_confirmation_targets_read"] is False,
        "frozen_prediction_sha_matches": sha256(prediction_path) == phase32b["frozen_prediction_sha256"],
        "new_confirmation_profiles_9": len(profiles) == 9,
        "new_confirmation_requests_2976": sum(len(value) for value in request_windows.values()) == 2976,
        "new_target_phase_rows_972": len(new_targets) == 972,
        "original_fixed_target_phase_rows_1080": len(original_targets) == 1080,
        "new_prediction_phase_rows_1944": len(new_predictions) == 1944,
        "fixed_prediction_phase_rows_2160": len(fixed_predictions) == 2160,
        "all_metrics_finite": all(math.isfinite(float(row[key])) for row in new_metrics + fixed_metrics for key in ("calls_wape", "bytes_wape", "mean_histogram_tv", "mean_normalized_log_payload_emd", "common_reference_cost_wape")),
        "raw_sources_pass": len(raw_checks) == 6 and all(raw_checks.values()),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = {"schema_version": "phase32c-frozen-evaluation-v1", "status": status, "generated_at_utc": datetime.now(timezone.utc).isoformat(), "primary_evidence": "new_confirmation", "secondary_evidence": "original_fixed_repeated", "new_confirmation_scope": "nine request-disjoint normal BurstGPT windows; all three known models and all TP/PP configurations", "counts": {"new_profiles": len(profiles), "new_full_requests": sum(len(value) for value in request_windows.values()), "new_target_phase_rows": len(new_targets), "new_prediction_phase_rows": len(new_predictions), "fixed_prediction_phase_rows": len(fixed_predictions)}, "decisions": decisions, "frozen_prediction_sha256": phase32b["frozen_prediction_sha256"], "evidence_limit": "The new confirmation is BurstGPT-only because no Mooncake block remained under the accumulated 300-second embargo. The original ten-window result is a repeated engineering evaluation, not a fresh blind test.", "checks": checks}
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "audit_summary.json", {"schema_version": "phase32c-audit-v1", "status": status, "checks": checks, "raw_source_checks": raw_checks})
    write_json(args.output_dir / "logs/evaluation.log", {"event": "phase32c_target_open_and_evaluation_complete", "status": status, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "repository_head_at_evaluation": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "frozen_prediction_sha256": phase32b["frozen_prediction_sha256"], "decisions": {evidence: {parallelism: value["decision"] for parallelism, value in rows.items()} for evidence, rows in decisions.items()}})
    (args.output_dir / "README.md").write_text(f"""# Phase 32C：冻结预测的新确认与重复固定集评测\n\nPhase32B预测和SHA已先归档，本阶段之后才生成9个新BurstGPT请求级互斥窗口的Hfull target。新确认覆盖三个已知模型、全部TP/PP配置和2,976个完整请求，是本轮主证据；原10个固定窗口没有更换，其复评明确标为重复工程证据。\n\n主证据裁定：TP=`{decisions['new_confirmation']['tp']['decision']}`，PP=`{decisions['new_confirmation']['pp']['decision']}`。重复固定集裁定：TP=`{decisions['original_fixed_repeated']['tp']['decision']}`，PP=`{decisions['original_fixed_repeated']['pp']['decision']}`。\n\n完整整体、逐模型、逐policy和逐并行规模指标见`analysis/aggregate_metrics.csv`；逐case结果见压缩明细。新确认不含Mooncake，因此不能把它单独表述为跨数据源验证。\n""")
    (args.output_dir / "DONE").write_text(status + "\n")
    manifest = [f"{sha256(path)}  {path.relative_to(args.output_dir)}" for path in sorted(args.output_dir.rglob("*")) if path.is_file() and path.name != "manifest.sha256"]
    (args.output_dir / "manifest.sha256").write_text("\n".join(manifest) + "\n")
    if status != "PASS": raise RuntimeError(checks)
    print(json.dumps({"status": status, "decisions": {evidence: {parallelism: value["decision"] for parallelism, value in rows.items()} for evidence, rows in decisions.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
