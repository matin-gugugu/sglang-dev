#!/usr/bin/env python3
"""Audit strict fixed-request pure-PP draining repetitions and structural H0."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_phase20_pp_predictor import (
    extract_truth,
    h0_hist,
    load_cell,
    strip_repeat,
    validate_boundaries,
    vector_from_payload_hist,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "experiment-results/phase23_pp_draining_stability/qwen3-8b-v1"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "experiment-results/phase23_pp_draining_stability/qwen3-8b-analysis-v1"
        ),
    )
    parser.add_argument("--expected-repeats", type=int, default=10)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def relative_span(values: list[float]) -> float:
    mean = float(np.mean(values))
    return 0.0 if mean == 0 else (max(values) - min(values)) / mean


def main() -> None:
    args = parse_args()
    if not (args.input_dir / "MATRIX_DONE").exists():
        raise RuntimeError(f"matrix is not complete: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[tuple[str, str], list[tuple[dict, dict[int, int], dict]]] = (
        defaultdict(list)
    )
    boundary_checks = []
    for cell in sorted(args.input_dir.glob("pp*/mb*")):
        if not (cell / "DONE").exists():
            continue
        config, clients, profiles = load_cell(cell)
        representative, boundary_count = validate_boundaries(profiles)
        truth = extract_truth(representative)
        boundary_checks.append(
            {
                "cell": str(cell.relative_to(args.input_dir)),
                "boundaries": boundary_count,
                "status": "PASS",
            }
        )
        for client in clients:
            for phase in ("prefill", "decode"):
                key = (client["workload_id"], phase)
                if key not in truth:
                    raise ValueError(f"missing truth for {key} in {cell}")
                groups[(strip_repeat(client["workload_id"]), phase)].append(
                    (client, dict(truth[key]), config)
                )

    repeat_rows = []
    h0_rows = []
    for (base, phase), rows in sorted(groups.items()):
        if len(rows) != args.expected_repeats:
            raise ValueError(
                f"expected {args.expected_repeats} repeats for {base}/{phase}, "
                f"got {len(rows)}"
            )
        signatures = [histogram for _, histogram, _ in rows]
        exact = all(histogram == signatures[0] for histogram in signatures[1:])
        calls = [float(sum(histogram.values())) for histogram in signatures]
        logical_bytes = [
            float(sum(payload * count for payload, count in histogram.items()))
            for histogram in signatures
        ]
        client, truth_hist, config = rows[0]
        h0 = h0_hist(
            client["input_lens"],
            client["actual_output_lens"],
            phase,
            int(client["pp_max_micro_batch_size"]),
            int(config["chunked_prefill_size"]),
        )
        true_calls, true_bytes = vector_from_payload_hist(truth_hist)
        pred_calls, pred_bytes = vector_from_payload_hist(h0)
        true_call_total = float(true_calls.sum())
        true_byte_total = float(true_bytes.sum())
        true_dist = true_calls / max(true_call_total, 1.0)
        pred_dist = pred_calls / max(float(pred_calls.sum()), 1.0)
        repeat_rows.append(
            {
                "sample_id": f"{base}/{phase}",
                "phase": phase,
                "pp_size": int(client["pp_size"]),
                "microbatch_size": int(client["pp_max_micro_batch_size"]),
                "workload": client["workload"],
                "repeats": len(rows),
                "exact_histogram": exact,
                "calls_relative_span": relative_span(calls),
                "bytes_relative_span": relative_span(logical_bytes),
            }
        )
        h0_rows.append(
            {
                "sample_id": f"{base}/{phase}",
                "phase": phase,
                "pp_size": int(client["pp_size"]),
                "microbatch_size": int(client["pp_max_micro_batch_size"]),
                "workload": client["workload"],
                "calls_ape": abs(float(pred_calls.sum()) - true_call_total)
                / max(true_call_total, 1.0),
                "bytes_ape": abs(float(pred_bytes.sum()) - true_byte_total)
                / max(true_byte_total, 1.0),
                "histogram_l1": float(np.abs(pred_dist - true_dist).sum()),
                "histogram_emd": float(
                    np.abs(np.cumsum(pred_dist) - np.cumsum(true_dist)).sum()
                    / max(len(true_dist) - 1, 1)
                ),
            }
        )

    exact_count = sum(bool(row["exact_histogram"]) for row in repeat_rows)
    summary = {
        "schema_version": "phase23-pp-fixed-draining-stability-v1",
        "status": "PASS"
        if exact_count == len(repeat_rows)
        and all(float(row["bytes_relative_span"]) == 0 for row in repeat_rows)
        else "FAIL",
        "input_contract": (
            "identical request token IDs, lengths, order, simultaneous arrival, "
            "PP configuration and microbatch policy across repetitions"
        ),
        "groups": len(repeat_rows),
        "raw_phase_labels": len(repeat_rows) * args.expected_repeats,
        "expected_repeats": args.expected_repeats,
        "exact_histogram_groups": exact_count,
        "exact_histogram_rate": exact_count / max(len(repeat_rows), 1),
        "max_calls_relative_span": max(
            float(row["calls_relative_span"]) for row in repeat_rows
        ),
        "max_bytes_relative_span": max(
            float(row["bytes_relative_span"]) for row in repeat_rows
        ),
        "h0": {
            "mean_calls_ape": float(
                np.mean([float(row["calls_ape"]) for row in h0_rows])
            ),
            "p95_calls_ape": float(
                np.quantile([float(row["calls_ape"]) for row in h0_rows], 0.95)
            ),
            "mean_bytes_ape": float(
                np.mean([float(row["bytes_ape"]) for row in h0_rows])
            ),
            "p95_bytes_ape": float(
                np.quantile([float(row["bytes_ape"]) for row in h0_rows], 0.95)
            ),
            "mean_histogram_l1": float(
                np.mean([float(row["histogram_l1"]) for row in h0_rows])
            ),
            "mean_histogram_emd": float(
                np.mean([float(row["histogram_emd"]) for row in h0_rows])
            ),
        },
        "boundary_checks": boundary_checks,
        "interpretation": (
            "This is the strict PP counterpart of the TP fixed-workload H0 audit. "
            "It does not use profiled/online arrivals and does not estimate an "
            "expected service histogram."
        ),
    }

    write_csv(args.output_dir / "repeat_stability.csv", repeat_rows)
    write_csv(args.output_dir / "h0_metrics.csv", h0_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "README.md").write_text(
        f"""# Phase 23：纯PP固定draining严格稳定性加固

本实验固定精确请求token、长度、顺序、同时到达方式、PP配置和microbatch策略，作为TP固定
workload验证的PP严格对照。覆盖{len(boundary_checks)}个单元、{len(repeat_rows)}个阶段组，
每组重复{args.expected_repeats}次。

- 精确直方图一致：{exact_count}/{len(repeat_rows)}；
- 最大calls相对跨度：{summary['max_calls_relative_span']:.4%}；
- 最大logical bytes相对跨度：{summary['max_bytes_relative_span']:.4%}；
- H0 calls平均APE：{summary['h0']['mean_calls_ape']:.4%}；
- H0 bytes平均APE：{summary['h0']['mean_bytes_ape']:.4%}；
- H0直方图平均L1：{summary['h0']['mean_histogram_l1']:.6f}。

该结果只验证固定draining请求的稳定性和结构公式，不代表在线到达或期望直方图预测。
""",
        encoding="utf-8",
    )
    (args.output_dir / "DONE").write_text(summary["status"] + "\n", encoding="utf-8")
    manifest = []
    for path in sorted(args.output_dir.iterdir()):
        if not path.is_file() or path.name == "manifest.sha256":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.append(f"{digest}  {path.name}\n")
    (args.output_dir / "manifest.sha256").write_text("".join(manifest), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
