#!/usr/bin/env python3
"""Re-evaluate the frozen Phase31E TP prediction on the unchanged fixed set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from evaluate_phase31d_known_model_fixed_predictions import (
    METHODS,
    PHASES,
    closure,
    evaluation_records,
    metric_row,
    read_csv,
    write_csv,
    write_csv_gz,
    write_fallback_svg,
    write_json,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase31d-dir", type=Path, default=root / "experiment-results/phase31d_known_model_fixed_evaluation")
    parser.add_argument("--phase31e-dir", type=Path, default=root / "experiment-results/phase31e_tp_weighted_residual_round2")
    parser.add_argument("--output-dir", type=Path, default=root / "experiment-results/phase31f_tp_round2_fixed_evaluation")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_expected(directory: Path, relative_path: str) -> str:
    for line in (directory / "manifest.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if name == relative_path:
            return digest
    raise RuntimeError(f"missing manifest entry: {relative_path}")


def aggregate_tp(records: list[dict]) -> list[dict]:
    output = []
    for method in METHODS:
        for phase in (*PHASES, "total"):
            base = [row for row in records if row["method"] == method and row["phase"] == phase]
            output.append(metric_row(base, parallelism="tp", method=method, phase=phase, slice_type="overall", slice_value="all"))
            for field in ("model", "policy", "parallel_size", "source"):
                for value in sorted({str(row[field]) for row in base}):
                    subset = [row for row in base if str(row[field]) == value]
                    output.append(metric_row(subset, parallelism="tp", method=method, phase=phase, slice_type=field, slice_value=value))
    return output


def main() -> None:
    args = parse_args()
    for name in ("analysis", "figures", "logs"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    phase31d = json.loads((args.phase31d_dir / "summary.json").read_text())
    phase31e = json.loads((args.phase31e_dir / "summary.json").read_text())
    prediction_path = args.phase31e_dir / "analysis/frozen_fixed_prediction.csv.gz"
    target_path = args.phase31d_dir / "labels/fixed_prediction_hfull_targets.csv.gz"
    expected_target_sha = manifest_expected(args.phase31d_dir, "labels/fixed_prediction_hfull_targets.csv.gz")
    if phase31e["status"] != "PASS" or phase31e["fixed_targets_read"] is not False:
        raise RuntimeError("Phase31E training isolation failed")
    if sha256(prediction_path) != phase31e["frozen_prediction_sha256"]:
        raise RuntimeError("Phase31E prediction SHA mismatch")
    predictions = read_csv(prediction_path)
    targets = [row for row in read_csv(target_path) if row["parallelism"] == "tp"]
    if len(predictions) != 1080 or len(targets) != 540:
        raise RuntimeError("unexpected TP row count")
    records = evaluation_records(predictions, targets)
    metrics = aggregate_tp(records)
    decision = closure(metrics, "tp")
    old = phase31d["decisions"]["tp"]
    write_csv_gz(args.output_dir / "analysis/per_case_metrics.csv.gz", records)
    write_csv(args.output_dir / "analysis/aggregate_metrics.csv", metrics)
    write_fallback_svg(args.output_dir / "figures/tp_round2_comparison.svg", {"tp": decision, "pp": phase31d["decisions"]["pp"]})
    checks = {
        "phase31e_fixed_targets_not_read": phase31e["fixed_targets_read"] is False,
        "prediction_sha_matches": sha256(prediction_path) == phase31e["frozen_prediction_sha256"],
        "unchanged_phase31d_target": sha256(target_path) == expected_target_sha,
        "predictions_1080": len(predictions) == 1080,
        "targets_540": len(targets) == 540,
        "records_1620": len(records) == 1620,
        "same_fixed_h0_calls_wape": abs(float(decision["h0"]["calls_wape"]) - float(old["h0"]["calls_wape"])) < 1e-12,
        "same_fixed_h0_cost_wape": abs(float(decision["h0"]["common_reference_cost_wape"]) - float(old["h0"]["common_reference_cost_wape"])) < 1e-12,
        "all_metrics_finite": all(math.isfinite(float(row[key])) for row in metrics for key in ("calls_mape", "calls_wape", "bytes_mape", "bytes_wape", "mean_histogram_tv", "mean_normalized_log_payload_emd", "common_reference_cost_mape", "common_reference_cost_wape")),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = {
        "schema_version": "phase31f-tp-round2-fixed-evaluation-v1",
        "status": status,
        "decision": decision,
        "phase31d_initial_decision": old,
        "selected_source": phase31e["selected_source"],
        "selected_candidate_id": phase31e["selected_candidate_id"],
        "fixed_prediction_sha256": phase31e["frozen_prediction_sha256"],
        "target_sha256": sha256(target_path),
        "target_manifest_sha256": expected_target_sha,
        "checks": checks,
        "evidence_class": "repeated unchanged fixed-set evaluation after a development-only predeclared finite round; not fresh independent confirmation",
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "audit_summary.json", {"schema_version": "phase31f-evaluation-audit-v1", "status": status, "checks": checks})
    (args.output_dir / "README.md").write_text(f"""# Phase 31F：TP最后一轮固定集复评

本阶段只评测Phase31E已冻结的TP预测，不训练、不选模型、不改变10个固定窗口，也不降低阈值。Phase31E训练脚本未读取固定Hfull target；本阶段复用Phase31D同一target，因此H0指标必须逐值一致。

裁定为`{decision['decision']}`。需要明确：这是在Phase31D已经打开target之后，对预定有限路线所得模型进行的同一固定集重复评测，不是新的独立确认。它可以用于今晚工程收口判断，但不能包装成全新盲测证据。
""")
    write_json(args.output_dir / "logs/evaluation.log", {"event": "phase31f_tp_round2_fixed_evaluation", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "status": status, "repository_head_at_evaluation": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "python": sys.version, "platform": platform.platform(), "decision": decision["decision"]})
    (args.output_dir / "DONE").write_text(f"{status}\n")
    manifest = [f"{sha256(path)}  {path.relative_to(args.output_dir)}" for path in sorted(args.output_dir.rglob("*")) if path.is_file() and path.name != "manifest.sha256"]
    (args.output_dir / "manifest.sha256").write_text("\n".join(manifest) + "\n")
    if status != "PASS":
        raise RuntimeError(checks)
    print(json.dumps({"status": status, "decision": decision}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
