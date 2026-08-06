#!/usr/bin/env python3
"""Select ProfileDemand payload bins using measured L1 curves.

The script separates two questions:

1. ``center`` uses the measured/interpolated cost at each logarithmic bin center and
   represents a directly deployable fixed-bin encoding.
2. ``mean_payload`` preserves both calls and logical bytes in each bin, evaluates the
   curve at ``bytes/calls``, and is the deployable moment-augmented histogram.
3. ``oracle`` uses the call-weighted optimal *constant* cost inside each op/TP/bin.
   It is an optimistic reference for constant-representative encodings; it never
   uses workload time labels. Because ``mean_payload`` retains an extra first moment,
   it can legitimately outperform this constant reference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_phase14g_strict_ablation import (
    fit_scales,
    interpolate,
    load_curve,
    phase14d_folds,
    read_csv,
    write_csv,
)


BIN_COUNTS = (8, 12, 16, 24)
MIN_PAYLOAD = 4 * 1024
MAX_PAYLOAD = 512 * 1024 * 1024


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--curve-root",
        type=Path,
        default=root / "experiment-results/phase14f_post_rendezvous/curve",
    )
    parser.add_argument(
        "--curve-extension",
        type=Path,
        default=root / "experiment-results/phase15_l1_curve_extension/curve_summary.csv",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root
        / "experiment-results/phase14c/extended_dataset_analysis/aggregated_configurations.csv",
    )
    parser.add_argument(
        "--phase14d-predictions",
        type=Path,
        default=root
        / "experiment-results/phase14d/tp_phase_interaction_analysis/predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase16_profiledemand_binning",
    )
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_all_supports(curve_root, extension_path):
    supports = load_curve(curve_root)
    for row in read_csv(extension_path):
        supports.append(
            {
                "op": row["op"],
                "tp": int(row["tp"]),
                "payload": int(row["payload_bytes"]),
                "cost_us": float(row["median_post_rendezvous_us"]),
                "backend_proxy": row["backend_proxy"],
                "observed_backend": "not_recorded_in_extension",
            }
        )
    keyed = {}
    for row in supports:
        key = (row["op"], row["tp"], row["payload"])
        if key in keyed:
            raise ValueError(f"duplicate curve support {key}")
        keyed[key] = row
    return list(keyed.values())


def make_bins(count):
    log_edges = np.linspace(math.log2(MIN_PAYLOAD), math.log2(MAX_PAYLOAD), count + 1)
    edges = np.power(2.0, log_edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return edges, centers


def bin_index(payload, edges):
    if payload < edges[0] or payload > edges[-1]:
        raise ValueError(f"payload {payload} outside [{edges[0]}, {edges[-1]}]")
    return min(int(np.searchsorted(edges, payload, side="right") - 1), len(edges) - 2)


def curve_lookup(supports):
    exact = {}
    points = defaultdict(list)
    for row in supports:
        key = (row["op"], row["tp"], row["payload"])
        exact[key] = row["cost_us"]
        points[(row["op"], row["tp"])].append((row["payload"], row["cost_us"]))
    return exact, points


def load_workloads(path, fold_path, exact):
    folds = phase14d_folds(fold_path)
    workloads = []
    for row in read_csv(path):
        tp = int(row["tp"])
        hist = []
        exact_cost = 0.0
        for text, count in json.loads(row["calls_by_op_payload_json"]).items():
            op, payload_text = text.rsplit(":", 1)
            payload, count = int(payload_text), int(count)
            key = (op, tp, payload)
            if key not in exact:
                raise KeyError(f"workload support missing from L1 curve: {key}")
            hist.append((op, payload, count))
            exact_cost += count * exact[key]
        workloads.append(
            {
                **row,
                "tp": tp,
                "fold": folds[(row["workload_id"], tp)],
                "histogram": hist,
                "exact_cost_us": exact_cost,
                "target_post_us": float(row["target_post_us"]),
            }
        )
    return workloads


def oracle_costs(workloads, edges, exact):
    weighted_sum = defaultdict(float)
    call_sum = defaultdict(int)
    for row in workloads:
        for op, payload, count in row["histogram"]:
            key = (op, row["tp"], bin_index(payload, edges))
            weighted_sum[key] += count * exact[(op, row["tp"], payload)]
            call_sum[key] += count
    return {key: weighted_sum[key] / call_sum[key] for key in call_sum}


def add_binned_costs(workloads, supports, exact):
    _, points = curve_lookup(supports)
    bin_metadata = []
    for count in BIN_COUNTS:
        edges, centers = make_bins(count)
        oracle = oracle_costs(workloads, edges, exact)
        for index, center in enumerate(centers):
            bin_metadata.append(
                {
                    "bin_count": count,
                    "bin_index": index,
                    "left_bytes": int(round(edges[index])),
                    "right_bytes": int(round(edges[index + 1])),
                    "center_bytes": int(round(center)),
                }
            )
        for row in workloads:
            center_total = 0.0
            mean_payload_total = 0.0
            oracle_total = 0.0
            grouped = defaultdict(lambda: [0, 0])
            for op, payload, calls in row["histogram"]:
                index = bin_index(payload, edges)
                center_cost, _ = interpolate(points[(op, row["tp"])], centers[index])
                center_total += calls * center_cost
                oracle_total += calls * oracle[(op, row["tp"], index)]
                grouped[(op, index)][0] += calls
                grouped[(op, index)][1] += calls * payload
            for (op, _), (calls, logical_bytes) in grouped.items():
                average_payload = logical_bytes / calls
                average_cost, _ = interpolate(points[(op, row["tp"])], average_payload)
                mean_payload_total += calls * average_cost
            row[f"center_{count}_cost_us"] = center_total
            row[f"mean_payload_{count}_cost_us"] = mean_payload_total
            row[f"oracle_{count}_cost_us"] = oracle_total
    return bin_metadata


def error_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    ape = np.abs(predicted - actual) / actual
    return {
        "samples": int(len(actual)),
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
        "p95_ape": float(np.percentile(ape, 95)),
        "max_ape": float(np.max(ape)),
    }


def discretization_metrics(workloads):
    rows = []
    for count in BIN_COUNTS:
        for method in ("center", "mean_payload", "oracle"):
            for scope in ("all", "prefill", "decode"):
                selected = workloads if scope == "all" else [r for r in workloads if r["phase"] == scope]
                rows.append(
                    {
                        "evaluation": "binning_vs_exact_structural_cost",
                        "method": method,
                        "bin_count": count,
                        "scope": scope,
                        **error_metrics(
                            [r["exact_cost_us"] for r in selected],
                            [r[f"{method}_{count}_cost_us"] for r in selected],
                        ),
                    }
                )
    return rows


def workload_time_cv(workloads):
    predictions = []
    fields = [("exact", "exact_cost_us")]
    for count in BIN_COUNTS:
        fields.extend(
            [
                (f"center_{count}", f"center_{count}_cost_us"),
                (f"mean_payload_{count}", f"mean_payload_{count}_cost_us"),
                (f"oracle_{count}", f"oracle_{count}_cost_us"),
            ]
        )
    for name, field in fields:
        for fold in sorted({r["fold"] for r in workloads}):
            train = [r for r in workloads if r["fold"] != fold]
            test = [r for r in workloads if r["fold"] == fold]
            scales = fit_scales(train, field)
            for row in test:
                predicted = row[field] * scales[row["phase"]]
                predictions.append(
                    {
                        "method": name,
                        "fold": fold,
                        "workload_id": row["workload_id"],
                        "model": row["model"],
                        "tp": row["tp"],
                        "phase": row["phase"],
                        "actual_us": row["target_post_us"],
                        "predicted_us": predicted,
                        "absolute_percentage_error": abs(predicted - row["target_post_us"])
                        / row["target_post_us"],
                    }
                )
    metrics = []
    for name, _ in fields:
        base = [r for r in predictions if r["method"] == name]
        for scope in ("all", "prefill", "decode"):
            selected = base if scope == "all" else [r for r in base if r["phase"] == scope]
            metrics.append(
                {
                    "evaluation": "workload_time_cv",
                    "method": name,
                    "bin_count": 0 if name == "exact" else int(name.rsplit("_", 1)[1]),
                    "scope": scope,
                    **error_metrics(
                        [r["actual_us"] for r in selected],
                        [r["predicted_us"] for r in selected],
                    ),
                }
            )
    return predictions, metrics


def select_bins(discretization):
    by_count = defaultdict(dict)
    for row in discretization:
        if row["method"] == "mean_payload":
            by_count[row["bin_count"]][row["scope"]] = row
    passing = []
    for count in BIN_COUNTS:
        rows = by_count[count]
        if all(rows[scope]["mape"] < 0.02 and rows[scope]["p95_ape"] < 0.05 for scope in rows):
            passing.append(count)
    return min(passing) if passing else None


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    supports = load_all_supports(args.curve_root, args.curve_extension)
    exact, _ = curve_lookup(supports)
    workloads = load_workloads(args.dataset, args.phase14d_predictions, exact)
    bins = add_binned_costs(workloads, supports, exact)
    discretization = discretization_metrics(workloads)
    predictions, time_metrics = workload_time_cv(workloads)
    selected = select_bins(discretization)

    write_csv(args.output_dir / "bin_definitions.csv", bins)
    write_csv(args.output_dir / "discretization_metrics.csv", discretization)
    write_csv(args.output_dir / "workload_time_predictions.csv", predictions)
    write_csv(args.output_dir / "workload_time_metrics.csv", time_metrics)

    summary = {
        "schema_version": "profiledemand-payload-binning-v1",
        "payload_range_bytes": [MIN_PAYLOAD, MAX_PAYLOAD],
        "bin_counts": list(BIN_COUNTS),
        "workloads": len(workloads),
        "curve_supports": len(supports),
        "selection_rule": "smallest moment-augmented bins with phase/all MAPE<2% and P95 APE<5% versus exact structural cost",
        "selected_bin_count": selected,
        "discretization_metrics": discretization,
        "workload_time_metrics": time_metrics,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def row(method, count, scope="all"):
        return next(
            r
            for r in discretization
            if r["method"] == method and r["bin_count"] == count and r["scope"] == scope
        )

    table = []
    for count in BIN_COUNTS:
        center, mean_payload, oracle = (
            row("center", count),
            row("mean_payload", count),
            row("oracle", count),
        )
        table.append(
            f"| {count} | {center['mape']*100:.3f}% | {center['p95_ape']*100:.3f}% | "
            f"{mean_payload['mape']*100:.3f}% | {mean_payload['p95_ape']*100:.3f}% | "
            f"{oracle['mape']*100:.3f}% | {oracle['p95_ape']*100:.3f}% |"
        )
    decision = str(selected) if selected is not None else "没有等宽 log bins 通过，需 cost-aware bins"
    readme = f"""# Phase 16A：ProfileDemand payload 分桶选择

固定范围为 4 KiB–512 MiB。`center` 使用桶几何中心在实测 L1 曲线上的代价；
`mean_payload` 在每桶同时保留 calls 与 logical bytes，并以 `bytes/calls` 查询曲线；
`oracle` 使用每个 `op×TP×bin` 内按 calls 加权的最优常数，是“每桶只能用一个固定
代表代价”时的乐观参照，不使用 workload 时间标签。`calls+bytes` 多保留了一阶矩，
因此可以优于这个固定常数参照。

| bins | center MAPE | center P95 | calls+bytes MAPE | calls+bytes P95 | oracle MAPE | oracle P95 |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

按“整体和 Prefill/Decode 均 MAPE <2%、P95 <5%”选择的结果：**{decision}**。

`workload_time_metrics.csv` 另报告各编码乘 L1 曲线并经 workload-CV phase scale 后，
相对真实 all-rank post-rendezvous 标签的误差；它用于观察分桶误差是否改变已有 4.43%
闭环，不用于定义分桶 oracle。
"""
    (args.output_dir / "README.md").write_text(readme)

    checks = {
        "workloads_162": len(workloads) == 162,
        "base_curve_supports_105_plus_extension_21": len(supports) == 126,
        "all_payloads_in_range": all(
            MIN_PAYLOAD <= payload <= MAX_PAYLOAD
            for row in workloads
            for _, payload, _ in row["histogram"]
        ),
        "selected_12_moment_augmented_bins": selected == 12,
    }
    audit = {
        "schema_version": "profiledemand-payload-binning-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    (args.output_dir / "audit_summary.json").write_text(json.dumps(audit, indent=2) + "\n")
    if audit["status"] != "PASS":
        raise RuntimeError(audit)
    (args.output_dir / "DONE").write_text("PASS\n")
    files = sorted(
        path
        for path in args.output_dir.iterdir()
        if path.is_file() and path.name not in {"manifest.sha256", "run.log"}
    )
    (args.output_dir / "manifest.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files)
    )
    print(json.dumps({"selected_bin_count": selected, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
