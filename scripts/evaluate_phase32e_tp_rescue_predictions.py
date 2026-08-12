#!/usr/bin/env python3
"""Evaluate frozen Phase32D TP rescue predictions as repeated engineering evidence."""

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

from evaluate_phase31d_known_model_fixed_predictions import (
    METHODS,
    PHASES,
    closure,
    evaluation_records,
    metric_row,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase32d-dir", type=Path, default=root / "experiment-results/phase32d_tp_gate_rescue")
    parser.add_argument("--phase32c-dir", type=Path, default=root / "experiment-results/phase32c_frozen_prediction_evaluation")
    parser.add_argument("--phase31d-dir", type=Path, default=root / "experiment-results/phase31d_known_model_fixed_evaluation")
    parser.add_argument("--output-dir", type=Path, default=root / "experiment-results/phase32e_tp_rescue_repeated_evaluation")
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
    for row in rows[1:]:
        fieldnames.extend(name for name in row if name not in fieldnames)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def write_csv_gz(path: Path, rows: list[dict]) -> None:
    buffer = io.StringIO(newline="")
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(name for name in row if name not in fieldnames)
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    deterministic_gzip(path, buffer.getvalue())


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


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


def minimal_svg(path: Path, decisions: dict) -> None:
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="920" height="440" viewBox="0 0 920 440">', '<rect width="100%" height="100%" fill="white"/>', '<text x="40" y="35" font-family="sans-serif" font-size="22">Phase32E：TP gate救援重复工程评估</text>']
    for panel, evidence in enumerate(("new_confirmation_repeated", "original_fixed_repeated")):
        value = decisions[evidence]
        x = 50 + panel * 440
        svg.append(f'<text x="{x}" y="75" font-family="sans-serif" font-size="16">{evidence} / {value["decision"]}</text>')
        for index, (key, label) in enumerate((("calls_wape", "calls WAPE"), ("bytes_wape", "bytes WAPE"), ("mean_histogram_tv", "TV"), ("mean_normalized_log_payload_emd", "EMD"), ("common_reference_cost_wape", "cost WAPE"))):
            y = 105 + index * 60
            h0 = float(value["h0"][key]); dnn = float(value["h0_plus_dnn_residual"][key])
            svg.append(f'<text x="{x}" y="{y+15}" font-family="sans-serif" font-size="12">{label}</text>')
            svg.append(f'<rect x="{x+85}" y="{y}" width="{min(h0*850,230):.2f}" height="17" fill="#94a3b8"/>')
            svg.append(f'<rect x="{x+85}" y="{y+21}" width="{min(dnn*850,230):.2f}" height="17" fill="#2563eb"/>')
    svg.append('</svg>')
    path.write_text("\n".join(svg) + "\n")


def main() -> None:
    args = parse_args()
    for name in ("analysis", "figures", "logs"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    phase32d = json.loads((args.phase32d_dir / "summary.json").read_text())
    predictions_path = args.phase32d_dir / "analysis/frozen_predictions.csv.gz"
    if phase32d["fixed_targets_read"] is not False or phase32d["new_confirmation_targets_read"] is not False:
        raise RuntimeError("Phase32D target isolation failed")
    if sha256(predictions_path) != phase32d["frozen_prediction_sha256"]:
        raise RuntimeError("Phase32D frozen SHA mismatch")
    predictions = read_csv(predictions_path)
    if {row["parallelism"] for row in predictions} != {"tp"} or {row["method"] for row in predictions} != METHODS:
        raise RuntimeError("unexpected prediction methods or parallelism")

    evidence_inputs = {
        "new_confirmation_repeated": (
            [row for row in predictions if row["prediction_set"] == "new_confirmation"],
            read_csv(args.phase32c_dir / "labels/new_confirmation_hfull_targets.csv.gz"),
        ),
        "original_fixed_repeated": (
            [row for row in predictions if row["prediction_set"] == "original_fixed"],
            read_csv(args.phase31d_dir / "labels/fixed_prediction_hfull_targets.csv.gz"),
        ),
    }
    all_records, all_metrics, decisions = [], [], {}
    for evidence, (evidence_predictions, targets) in evidence_inputs.items():
        tp_targets = [row for row in targets if row["parallelism"] == "tp"]
        records = evaluation_records(evidence_predictions, tp_targets)
        for row in records: row["evidence_set"] = evidence
        metrics = aggregate_tp(records)
        for row in metrics: row["evidence_set"] = evidence
        all_records.extend(records); all_metrics.extend(metrics)
        decisions[evidence] = closure(metrics, "tp")

    checks = {
        "phase32d_target_free_training_and_selection": phase32d["fixed_targets_read"] is False and phase32d["new_confirmation_targets_read"] is False,
        "frozen_prediction_sha_matches": sha256(predictions_path) == phase32d["frozen_prediction_sha256"],
        "cumulative_tp_candidates_48": phase32d["search"]["final_cumulative"] == 48,
        "new_prediction_phase_rows_972": len(evidence_inputs["new_confirmation_repeated"][0]) == 972,
        "fixed_prediction_phase_rows_1080": len(evidence_inputs["original_fixed_repeated"][0]) == 1080,
        "all_metrics_finite": all(math.isfinite(float(row[key])) for row in all_metrics for key in ("calls_wape", "bytes_wape", "mean_histogram_tv", "mean_normalized_log_payload_emd", "common_reference_cost_wape")),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    write_csv_gz(args.output_dir / "analysis/per_case_metrics.csv.gz", all_records)
    write_csv(args.output_dir / "analysis/aggregate_metrics.csv", all_metrics)
    minimal_svg(args.output_dir / "figures/tp_rescue_repeated_evaluation.svg", decisions)
    summary = {
        "schema_version": "phase32e-tp-rescue-repeated-evaluation-v1",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_candidate_id": phase32d["selected_candidate_id"],
        "frozen_prediction_sha256": phase32d["frozen_prediction_sha256"],
        "decisions": decisions,
        "checks": checks,
        "evidence_limit": "Phase32C targets were open before Phase32D was run. Although Phase32D selection was target-free, both evaluations here are repeated engineering evidence and are not a new blind confirmation.",
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "audit_summary.json", {"schema_version": "phase32e-audit-v1", "status": status, "checks": checks})
    write_json(args.output_dir / "logs/evaluation.log", {"event": "phase32e_tp_rescue_repeated_evaluation_complete", "status": status, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "repository_head_at_evaluation": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "decisions": {key: value["decision"] for key, value in decisions.items()}})
    (args.output_dir / "README.md").write_text(f"""# Phase 32E：TP定向救援后的重复工程评估

本阶段不训练，只对Phase32D已经冻结并归档SHA的TP H0与H0+DNN residual预测做评估。新确认target已在Phase32C开放，因此这里的新确认复评和原10窗口复评都只能称为重复工程证据，不能包装成新的盲测。

新确认重复工程裁定：`{decisions['new_confirmation_repeated']['decision']}`；原10窗口重复工程裁定：`{decisions['original_fixed_repeated']['decision']}`。整体、逐模型、逐policy、逐TP size指标见`analysis/aggregate_metrics.csv`。
""")
    (args.output_dir / "DONE").write_text(status + "\n")
    manifest = [f"{sha256(path)}  {path.relative_to(args.output_dir)}" for path in sorted(args.output_dir.rglob("*")) if path.is_file() and path.name != "manifest.sha256"]
    (args.output_dir / "manifest.sha256").write_text("\n".join(manifest) + "\n")
    if status != "PASS": raise RuntimeError(checks)
    print(json.dumps({"status": status, "decisions": {key: value["decision"] for key, value in decisions.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
