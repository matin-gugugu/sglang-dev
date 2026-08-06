#!/usr/bin/env python3
"""Strict same-contract ablation and support holdout for corrected Phase 14F."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--curve-root",
        type=Path,
        default=root / "experiment-results/phase14f_post_rendezvous/curve",
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
        default=root / "experiment-results/phase14g_strict_ablation",
    )
    return parser.parse_args()


def read_csv(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"empty rows for {path}")
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(rows):
    actual = np.asarray([float(row["actual_us"]) for row in rows])
    pred = np.asarray([float(row["predicted_us"]) for row in rows])
    ape = np.abs(pred - actual) / actual
    residual = pred - actual
    denom = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        "samples": len(rows),
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
        "p95_ape": float(np.percentile(ape, 95)),
        "mae_us": float(np.mean(np.abs(residual))),
        "rmse_us": float(np.sqrt(np.mean(residual**2))),
        "r2": 1.0 - float(np.sum(residual**2)) / denom if denom else 0.0,
    }


def load_curve(root):
    records = []
    for path in sorted(root.glob("tp*/*/r*/curve.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    grouped = defaultdict(list)
    proxies = defaultdict(set)
    observed = defaultdict(set)
    for row in records:
        key = (row["op"], int(row["group_size"]), int(row["payload_bytes"]))
        grouped[key].extend(float(value) for value in row["post_rendezvous_samples_us"])
        proxies[key].add(row["backend_proxy_pre_run"])
        observed[key].add(row["observed_backend_audit_only"])
    supports = []
    for key, values in sorted(grouped.items()):
        if len(proxies[key]) != 1 or len(observed[key]) != 1:
            raise ValueError(f"unstable backend metadata at {key}")
        supports.append(
            {
                "op": key[0],
                "tp": key[1],
                "payload": key[2],
                "cost_us": float(np.median(values)),
                "backend_proxy": next(iter(proxies[key])),
                "observed_backend": next(iter(observed[key])),
            }
        )
    if len(supports) != 105:
        raise ValueError(f"expected 105 supports, got {len(supports)}")
    return supports


def phase14d_folds(path):
    rows = [
        row
        for row in read_csv(path)
        if row["evaluation"] == "workload_cv"
        and row["method"] == "tp_conditioned_pattern"
    ]
    keyed = {(row["workload_id"], int(row["tp"])): int(row["fold"]) for row in rows}
    if len(keyed) != 162:
        raise ValueError(f"expected 162 fold assignments, got {len(keyed)}")
    return keyed


def interpolate(points, payload):
    """Linear interpolation in log2(payload), with clipped two-point extrapolation."""
    points = sorted((int(x), float(y)) for x, y in points)
    if not points:
        raise ValueError("no interpolation support")
    if len(points) == 1:
        return points[0][1], "single_point"
    x = math.log2(payload)
    xs = np.asarray([math.log2(value[0]) for value in points], dtype=np.float64)
    ys = np.asarray([value[1] for value in points], dtype=np.float64)
    if x <= xs[0]:
        left, right, mode = 0, 1, "low_extrapolation"
    elif x >= xs[-1]:
        left, right, mode = len(xs) - 2, len(xs) - 1, "high_extrapolation"
    else:
        right = int(np.searchsorted(xs, x, side="right"))
        left, mode = right - 1, "interpolation"
    ratio = (x - xs[left]) / (xs[right] - xs[left])
    prediction = ys[left] + ratio * (ys[right] - ys[left])
    return max(float(prediction), 1e-6), mode


def support_holdout(supports):
    rows = []
    for target in supports:
        same_op = [
            row
            for row in supports
            if row is not target and row["op"] == target["op"] and row["tp"] == target["tp"]
        ]
        same_backend = [
            row for row in same_op if row["backend_proxy"] == target["backend_proxy"]
        ]
        op_pred, op_mode = interpolate(
            [(row["payload"], row["cost_us"]) for row in same_op], target["payload"]
        )
        fallback = not same_backend
        backend_points = same_backend if same_backend else same_op
        backend_pred, backend_mode = interpolate(
            [(row["payload"], row["cost_us"]) for row in backend_points], target["payload"]
        )
        ordered = sorted(
            [row for row in supports if row["op"] == target["op"] and row["tp"] == target["tp"]],
            key=lambda row: row["payload"],
        )
        index = ordered.index(target)
        neighbors = ordered[max(0, index - 1) : index] + ordered[index + 1 : index + 2]
        proxy_boundary = any(
            row["backend_proxy"] != target["backend_proxy"] for row in neighbors
        )
        observed_boundary = any(
            row["observed_backend"] != target["observed_backend"] for row in neighbors
        )
        rows.append(
            {
                **target,
                "op_interp_us": op_pred,
                "op_interp_mode": op_mode,
                "op_interp_ape": abs(op_pred - target["cost_us"]) / target["cost_us"],
                "backend_interp_us": backend_pred,
                "backend_interp_mode": backend_mode,
                "backend_fallback": fallback,
                "backend_interp_ape": abs(backend_pred - target["cost_us"])
                / target["cost_us"],
                "proxy_boundary": proxy_boundary,
                "observed_boundary_audit_only": observed_boundary,
            }
        )
    return rows


def curve_maps(supports, holdout):
    exact_op = {
        (row["op"], row["tp"], row["payload"]): row["cost_us"] for row in supports
    }
    pooled = defaultdict(list)
    for row in supports:
        pooled[(row["tp"], row["payload"])].append(row["cost_us"])
    exact_payload = {key: float(np.median(values)) for key, values in pooled.items()}
    op_loo = {
        (row["op"], row["tp"], row["payload"]): row["op_interp_us"] for row in holdout
    }
    backend_loo = {
        (row["op"], row["tp"], row["payload"]): row["backend_interp_us"]
        for row in holdout
    }
    return exact_op, exact_payload, op_loo, backend_loo


def load_workloads(path, folds, maps):
    exact_op, exact_payload, op_loo, backend_loo = maps
    rows = []
    for source in read_csv(path):
        tp = int(source["tp"])
        histogram = json.loads(source["calls_by_op_payload_json"])
        bases = {
            "total_bytes": float(source["logical_payload_bytes"]),
            "payload_tp_exact": 0.0,
            "raw_op_tp_exact": 0.0,
            "raw_op_tp_loo_interp": 0.0,
            "backend_proxy_loo_interp": 0.0,
        }
        for text, count in histogram.items():
            op, payload_text = text.rsplit(":", 1)
            payload, count = int(payload_text), int(count)
            bases["payload_tp_exact"] += count * exact_payload[(tp, payload)]
            key = (op, tp, payload)
            bases["raw_op_tp_exact"] += count * exact_op[key]
            bases["raw_op_tp_loo_interp"] += count * op_loo[key]
            bases["backend_proxy_loo_interp"] += count * backend_loo[key]
        rows.append(
            {
                **source,
                "tp": tp,
                "target_post_us": float(source["target_post_us"]),
                "fold": folds[(source["workload_id"], tp)],
                **bases,
            }
        )
    if len(rows) != 162:
        raise ValueError(f"expected 162 workloads, got {len(rows)}")
    return rows


def fit_scales(rows, field):
    scales = {}
    for phase in ("prefill", "decode"):
        selected = [row for row in rows if row["phase"] == phase]
        x = np.asarray([float(row[field]) for row in selected])
        y = np.asarray([float(row["target_post_us"]) for row in selected])
        scales[phase] = max(0.0, float(np.dot(x, y) / np.dot(x, x)))
    return scales


def workload_cv(rows, field, method, calibrated=True):
    predictions = []
    for fold in sorted({row["fold"] for row in rows}):
        train = [row for row in rows if row["fold"] != fold]
        test = [row for row in rows if row["fold"] == fold]
        scales = fit_scales(train, field) if calibrated else {"prefill": 1.0, "decode": 1.0}
        for row in test:
            predicted = float(row[field]) * scales[row["phase"]]
            predictions.append(
                {
                    "method": method,
                    "workload_id": row["workload_id"],
                    "fold": fold,
                    "model": row["model"],
                    "tp": row["tp"],
                    "phase": row["phase"],
                    "case_label": row["case_label"],
                    "actual_us": row["target_post_us"],
                    "predicted_us": predicted,
                    "absolute_percentage_error": abs(predicted - row["target_post_us"])
                    / row["target_post_us"],
                    "prefill_scale": scales["prefill"],
                    "decode_scale": scales["decode"],
                }
            )
    return predictions


def grouped_metrics(predictions):
    result = []
    for method in sorted({row["method"] for row in predictions}):
        base = [row for row in predictions if row["method"] == method]
        scopes = {"all": base}
        for phase in ("prefill", "decode"):
            scopes[phase] = [row for row in base if row["phase"] == phase]
        for scope, rows in scopes.items():
            result.append({"method": method, "scope": scope, **metrics(rows)})
    return result


def curve_metrics(rows, field, scope):
    selected = rows
    if scope == "proxy_boundary":
        selected = [row for row in rows if row["proxy_boundary"]]
    elif scope == "observed_boundary_audit_only":
        selected = [row for row in rows if row["observed_boundary_audit_only"]]
    values = np.asarray([float(row[field]) for row in selected])
    return {
        "scope": scope,
        "method": field.replace("_ape", ""),
        "samples": len(values),
        "mape": float(np.mean(values)),
        "median_ape": float(np.median(values)),
        "p95_ape": float(np.percentile(values, 95)),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    supports = load_curve(args.curve_root)
    holdout = support_holdout(supports)
    maps = curve_maps(supports, holdout)
    workloads = load_workloads(args.dataset, phase14d_folds(args.phase14d_predictions), maps)

    predictions = []
    for field, method, calibrated in (
        ("total_bytes", "total_bytes_phase_scaled", True),
        ("payload_tp_exact", "payload_histogram_tp_phase_scaled", True),
        ("raw_op_tp_exact", "raw_op_exact_no_phase_calibration", False),
        ("raw_op_tp_exact", "raw_op_exact_phase_scaled", True),
        ("raw_op_tp_loo_interp", "raw_op_support_loo_phase_scaled", True),
        ("backend_proxy_loo_interp", "backend_proxy_support_loo_phase_scaled", True),
    ):
        predictions.extend(workload_cv(workloads, field, method, calibrated))
    workload_summary = grouped_metrics(predictions)
    support_summary = []
    for scope in ("all", "proxy_boundary", "observed_boundary_audit_only"):
        support_summary.append(curve_metrics(holdout, "op_interp_ape", scope))
        support_summary.append(curve_metrics(holdout, "backend_interp_ape", scope))

    write_csv(args.output_dir / "support_holdout_predictions.csv", holdout)
    write_csv(args.output_dir / "workload_predictions.csv", predictions)
    write_csv(args.output_dir / "workload_metrics.csv", workload_summary)
    write_csv(args.output_dir / "support_holdout_metrics.csv", support_summary)

    selected = {
        row["method"]: row
        for row in workload_summary
        if row["scope"] == "all"
    }
    summary = {
        "schema_version": "phase14g-strict-ablation-v1",
        "time_contract": "all-rank post-rendezvous for both curve and target",
        "workloads": len(workloads),
        "supports": len(supports),
        "exact_support_identifiability": (
            "At exact supports, raw_op+payload+TP uniquely determines the pre-run "
            "backend proxy; backend has no separately identifiable exact-lookup score."
        ),
        "workload_metrics": selected,
        "support_holdout_metrics": support_summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def pct(method, key="mape"):
        return selected[method][key] * 100

    readme = f"""# Phase 14G：同口径消融与未见 payload 支撑留出

所有 workload 标签和微基准曲线均使用 all-rank post-rendezvous 时间契约。

## Workload-CV

| 方法 | MAPE | P95 APE |
|---|---:|---:|
| total bytes + phase scale | {pct('total_bytes_phase_scaled'):.3f}% | {pct('total_bytes_phase_scaled', 'p95_ape'):.3f}% |
| payload histogram × TP + phase scale | {pct('payload_histogram_tp_phase_scaled'):.3f}% | {pct('payload_histogram_tp_phase_scaled', 'p95_ape'):.3f}% |
| exact raw op，不校准 | {pct('raw_op_exact_no_phase_calibration'):.3f}% | {pct('raw_op_exact_no_phase_calibration', 'p95_ape'):.3f}% |
| exact raw op + phase scale | {pct('raw_op_exact_phase_scaled'):.3f}% | {pct('raw_op_exact_phase_scaled', 'p95_ape'):.3f}% |
| raw op，逐支撑点 LOO 插值 | {pct('raw_op_support_loo_phase_scaled'):.3f}% | {pct('raw_op_support_loo_phase_scaled', 'p95_ape'):.3f}% |
| backend-proxy 分段，逐支撑点 LOO 插值 | {pct('backend_proxy_support_loo_phase_scaled'):.3f}% | {pct('backend_proxy_support_loo_phase_scaled', 'p95_ape'):.3f}% |

在当前 105 个精确支撑点上，`raw_op + payload + TP` 唯一决定 pre-run backend
proxy，所以 exact raw-op 与 exact backend-proxy 查表不能得到两个独立分数。
backend proxy 的增量价值只在未见 payload 插值中评估。

## 未见 payload 支撑点

逐个隐藏 105 个曲线支撑点、只使用其余点插值时：

- 不区分 backend 的曲线级 MAPE 为 {support_summary[0]['mape'] * 100:.3f}%，
  P95 为 {support_summary[0]['p95_ape'] * 100:.3f}%；
- 按运行前 backend proxy 分段后，曲线级 MAPE 为
  {support_summary[1]['mape'] * 100:.3f}%，P95 为
  {support_summary[1]['p95_ape'] * 100:.3f}%；
- 在 12 个 proxy 边界点上，MAPE 从
  {support_summary[2]['mape'] * 100:.3f}% 降至
  {support_summary[3]['mape'] * 100:.3f}%；
- 将全部 LOO 插值曲线重新卷积到 162 个 workload 后，raw-op 插值和 backend
  分段插值 MAPE 分别为 {pct('raw_op_support_loo_phase_scaled'):.3f}% 与
  {pct('backend_proxy_support_loo_phase_scaled'):.3f}%。

因此，backend 分段对算法边界的单点插值明显有帮助，但没有改善当前 workload
总体 MAPE；主要增益顺序是 total bytes → payload histogram → raw op 精确曲线。
phase calibration 略微改善 P95/RMSE，但不改善平均 MAPE。

## 产物

- `workload_metrics.csv` / `workload_predictions.csv`；
- `support_holdout_metrics.csv` / `support_holdout_predictions.csv`；
- `summary.json`。
"""
    (args.output_dir / "README.md").write_text(readme)

    checks = {
        "workloads_162": len(workloads) == 162,
        "supports_105": len(supports) == 105,
        "exact_phase14f_mape_matches": abs(
            selected["raw_op_exact_phase_scaled"]["mape"] - 0.04425131816914883
        )
        < 1e-12,
        "support_loo_workload_p95_below_25pct": selected[
            "raw_op_support_loo_phase_scaled"
        ]["p95_ape"]
        < 0.25,
        "backend_proxy_improves_boundary_curve_mape": support_summary[3]["mape"]
        < support_summary[2]["mape"],
    }
    audit = {
        "schema_version": "phase14g-strict-ablation-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    if audit["status"] != "PASS":
        raise RuntimeError(audit)
    manifest = args.output_dir / "manifest.sha256"
    files = sorted(
        path
        for path in args.output_dir.iterdir()
        if path.is_file() and path.name not in {"manifest.sha256", "run.log"}
    )
    manifest.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
