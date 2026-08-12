#!/usr/bin/env python3
"""Summarize Phase 25 full-window TP/PP GPU smoke teacher audits."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


MIN_PAYLOAD = 4 * 1024
MAX_PAYLOAD = 8 * 1024 * 1024 * 1024
LAUNCH_US = 5.0
BANDWIDTH_GBPS = 100.0


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=root / "experiment-results/phase25_full_window_teacher",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root
        / "experiment-results/phase25_full_window_teacher/analysis/gpu_smoke-v1",
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
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def histogram(row: dict[str, str]) -> dict[int, float]:
    return {
        int(payload): float(calls)
        for payload, calls in json.loads(
            row["exact_calls_histogram_per_1000_json"]
        ).items()
    }


def l1_tv(predicted: dict, truth: dict) -> tuple[float, float]:
    predicted_total = max(sum(predicted.values()), 1e-12)
    truth_total = max(sum(truth.values()), 1e-12)
    keys = set(predicted) | set(truth)
    l1 = sum(
        abs(predicted.get(key, 0.0) / predicted_total - truth.get(key, 0.0) / truth_total)
        for key in keys
    )
    return float(l1), float(l1 / 2.0)


def log_payload_emd(predicted: dict[int, float], truth: dict[int, float]) -> tuple[float, float]:
    supports = sorted(set(predicted) | set(truth))
    if len(supports) <= 1:
        return 0.0, 0.0
    predicted_total = max(sum(predicted.values()), 1e-12)
    truth_total = max(sum(truth.values()), 1e-12)
    predicted_cdf = truth_cdf = emd = 0.0
    for left, right in zip(supports[:-1], supports[1:]):
        predicted_cdf += predicted.get(left, 0.0) / predicted_total
        truth_cdf += truth.get(left, 0.0) / truth_total
        emd += abs(predicted_cdf - truth_cdf) * (math.log2(right) - math.log2(left))
    normalized = emd / (math.log2(MAX_PAYLOAD) - math.log2(MIN_PAYLOAD))
    return float(emd), float(normalized)


def cost(hist: dict[int, float]) -> float:
    return sum(
        calls * (LAUNCH_US + payload / (BANDWIDTH_GBPS * 1e9) * 1e6)
        for payload, calls in hist.items()
    )


def combine(rows: list[dict[str, str]]) -> dict[str, str]:
    if {row["phase"] for row in rows} != {"prefill", "decode"}:
        raise ValueError("total row requires exactly prefill and decode")
    hist = Counter()
    for row in rows:
        hist.update(histogram(row))
    return {
        **rows[0],
        "phase": "total",
        "total_calls_per_1000": str(sum(hist.values())),
        "total_logical_bytes_per_1000": str(
            sum(payload * calls for payload, calls in hist.items())
        ),
        "exact_calls_histogram_per_1000_json": json.dumps(
            {str(payload): calls for payload, calls in sorted(hist.items())},
            separators=(",", ":"),
        ),
    }


def metric_row(predicted: dict[str, str], truth: dict[str, str]) -> dict:
    predicted_hist, truth_hist = histogram(predicted), histogram(truth)
    calls_predicted, calls_truth = sum(predicted_hist.values()), sum(truth_hist.values())
    bytes_predicted = sum(payload * calls for payload, calls in predicted_hist.items())
    bytes_truth = sum(payload * calls for payload, calls in truth_hist.items())
    l1, tv = l1_tv(predicted_hist, truth_hist)
    byte_l1, byte_tv = l1_tv(
        {payload: payload * calls for payload, calls in predicted_hist.items()},
        {payload: payload * calls for payload, calls in truth_hist.items()},
    )
    emd, normalized_emd = log_payload_emd(predicted_hist, truth_hist)
    predicted_cost, truth_cost = cost(predicted_hist), cost(truth_hist)
    return {
        "parallelism": predicted["parallelism"],
        "parallel_size": int(predicted["parallel_size"]),
        "profile_id": predicted["profile_id"],
        "policy": predicted["policy"],
        "phase": predicted["phase"],
        "requests": int(predicted["requests"]),
        "calls_predicted": calls_predicted,
        "calls_truth": calls_truth,
        "calls_abs_error": abs(calls_predicted - calls_truth),
        "calls_ape": abs(calls_predicted - calls_truth) / max(calls_truth, 1e-12),
        "bytes_predicted": bytes_predicted,
        "bytes_truth": bytes_truth,
        "bytes_abs_error": abs(bytes_predicted - bytes_truth),
        "bytes_ape": abs(bytes_predicted - bytes_truth) / max(bytes_truth, 1e-12),
        "calls_histogram_l1": l1,
        "calls_histogram_tv": tv,
        "bytes_histogram_l1": byte_l1,
        "bytes_histogram_tv": byte_tv,
        "log_payload_emd_log2_bytes": emd,
        "normalized_log_payload_emd": normalized_emd,
        "common_reference_cost_predicted_us": predicted_cost,
        "common_reference_cost_truth_us": truth_cost,
        "common_reference_cost_ape": abs(predicted_cost - truth_cost)
        / max(truth_cost, 1e-12),
        "exact_histogram": predicted_hist == truth_hist,
    }


def aggregate(rows: list[dict], parallelism: str, phase: str, policy: str = "") -> dict:
    values = [
        row
        for row in rows
        if row["parallelism"] == parallelism
        and row["phase"] == phase
        and (not policy or row["policy"] == policy)
    ]
    return {
        "parallelism": parallelism,
        "phase": phase,
        "policy": policy or "all",
        "cases": len(values),
        "exact_histograms": sum(row["exact_histogram"] for row in values),
        "calls_mape": sum(row["calls_ape"] for row in values) / len(values),
        "calls_wape": sum(row["calls_abs_error"] for row in values)
        / max(sum(row["calls_truth"] for row in values), 1e-12),
        "bytes_mape": sum(row["bytes_ape"] for row in values) / len(values),
        "bytes_wape": sum(row["bytes_abs_error"] for row in values)
        / max(sum(row["bytes_truth"] for row in values), 1e-12),
        "mean_calls_histogram_tv": sum(row["calls_histogram_tv"] for row in values)
        / len(values),
        "mean_normalized_log_payload_emd": sum(
            row["normalized_log_payload_emd"] for row in values
        )
        / len(values),
        "common_reference_cost_mape": sum(
            row["common_reference_cost_ape"] for row in values
        )
        / len(values),
    }


def main() -> None:
    args = parse_args()
    results = args.teacher_root / "gpu_audit" / "results"
    pp_root = results / "pp" / "smoke"
    if (pp_root / "MATRIX_DONE").read_text().strip() not in {"PASS", "MEASURED_MISMATCH"}:
        raise RuntimeError("PP smoke matrix is incomplete")
    tp_dir = results / "tp" / "smoke" / "qwen3-8b" / "tp2" / "r0"
    tp_audit = json.loads((tp_dir / "teacher_audit.json").read_text())
    pp_audits = [
        json.loads(path.read_text())
        for path in sorted(pp_root.glob("pp*/mb*/r0/teacher_audit.json"))
    ]
    if tp_audit["status"] != "PASS" or len(pp_audits) != 9:
        raise RuntimeError("expected one passing TP sentinel and nine PP cells")
    if not all(row["checks"]["gpu_integrity"] for row in pp_audits):
        raise RuntimeError("a PP cell failed execution integrity")

    teacher_rows = []
    for name in ("tp_phase_labels.csv.gz", "pp_phase_labels.csv.gz"):
        teacher_rows.extend(read_csv_gz(args.teacher_root / "labels" / name))
    teacher = {
        (
            row["parallelism"],
            int(row["parallel_size"]),
            row["profile_id"],
            row["policy"],
            row["phase"],
        ): row
        for row in teacher_rows
    }
    gpu_rows = read_csv(tp_dir / "gpu_phase_labels.csv")
    for path in sorted(pp_root.glob("pp*/mb*/r0/gpu_phase_labels.csv")):
        gpu_rows.extend(read_csv(path))
    for row in gpu_rows:
        row["parallelism"] = "tp" if row["policy"] in {"latency", "balanced", "throughput"} else "pp"
    pairs = []
    for truth in gpu_rows:
        key = (
            truth["parallelism"],
            int(truth["parallel_size"]),
            truth["profile_id"],
            truth["policy"],
            truth["phase"],
        )
        predicted = teacher[key]
        pairs.append((predicted, truth))
    grouped: dict[tuple, list[tuple[dict, dict]]] = defaultdict(list)
    for predicted, truth in pairs:
        grouped[
            (
                predicted["parallelism"],
                predicted["parallel_size"],
                predicted["profile_id"],
                predicted["policy"],
            )
        ].append((predicted, truth))
    for values in grouped.values():
        pairs.append((combine([row[0] for row in values]), combine([row[1] for row in values])))
    metrics = [metric_row(predicted, truth) for predicted, truth in pairs]
    aggregates = []
    for parallelism in ("tp", "pp"):
        policies = (
            ("latency", "balanced", "throughput")
            if parallelism == "tp"
            else ("mb1", "mb4", "mb16")
        )
        for phase in ("prefill", "decode", "total"):
            aggregates.append(aggregate(metrics, parallelism, phase))
            aggregates.extend(aggregate(metrics, parallelism, phase, policy) for policy in policies)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "label_metrics.csv", metrics)
    write_csv(args.output_dir / "aggregate_metrics.csv", aggregates)
    pp_exact = sum(row["checks"]["teacher_exact_match"] for row in pp_audits)
    pp_total = next(
        row
        for row in aggregates
        if row["parallelism"] == "pp" and row["phase"] == "total" and row["policy"] == "all"
    )
    summary = {
        "schema_version": "phase25-full-window-gpu-smoke-analysis-v1",
        "status": "COMPLETE_MEASURED_PP_MISMATCH",
        "profile_id": "profile_13_burstgpt_3_c2",
        "requests": 42,
        "tp": {
            "cells": 1,
            "teacher_exact_cells": 1,
            "phase_labels": 6,
            "status": "PASS",
        },
        "pp": {
            "cells": len(pp_audits),
            "integrity_pass_cells": sum(row["checks"]["gpu_integrity"] for row in pp_audits),
            "teacher_exact_cells": pp_exact,
            "teacher_mismatch_cells": len(pp_audits) - pp_exact,
            "phase_labels": 18,
            "total_metrics": pp_total,
        },
        "interpretation": (
            "Full-window TP aggregation is exact for the sentinel. PP logical bytes are "
            "conserved, but MB>1 scheduler split/merge changes calls and payload histograms; "
            "the current static PP structural formula cannot be promoted as Y_full."
        ),
        "promotion_gate": "BLOCKED_BY_PP_MB_GT_1_SCHEDULER_MISMATCH",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(
        "# Phase 25 GPU smoke：完整窗口 teacher 审计\n\n"
        "42 个真实请求的完整 fixed-draining 窗口已完成 TP2 和 9 个 PP cell 的 GPU 审计。"
        f"TP 精确通过；PP {pp_exact}/9 个 cell 与静态 teacher 精确一致。"
        "所有 PP cell 的采集完整性和 sender boundary 一致性通过。\n\n"
        "MB>1 的 logical bytes 守恒但 calls/直方图不一致，说明误差来自真实 scheduler 的"
        "离散 microbatch 拆分/合并，而不是请求规模抽样。当前 provisional PP 标签不能晋升为"
        "训练真值；下一步应恢复 fixed-draining scheduler 语义或生成 GPU/full-scheduler teacher。\n"
    )
    (args.output_dir / "DONE").write_text("COMPLETE_MEASURED_PP_MISMATCH\n")
    files = sorted(
        path for path in args.output_dir.iterdir() if path.is_file() and path.name != "manifest.sha256"
    )
    (args.output_dir / "manifest.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
