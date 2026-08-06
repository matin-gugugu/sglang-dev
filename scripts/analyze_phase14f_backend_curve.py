#!/usr/bin/env python3
"""Evaluate Phase 14F op/backend-proxy curves on the Phase 14C dataset."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_pattern_cost_ablation import BackendAwareCostCurve


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--curve-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--phase14d-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--phase2-custom-curve",
        type=Path,
        default=repo_root
        / "experiment-results/phase2/summary_l1_custom_kernel_curve/custom_kernel_curve_summary.csv",
    )
    parser.add_argument(
        "--phase2-nccl-curve",
        type=Path,
        default=repo_root
        / "experiment-results/phase2/summary_l1_curve/collective_curve_summary.csv",
    )
    parser.add_argument("--expected-repeats", type=int, default=5)
    return parser.parse_args()


def read_csv(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def regression_metrics(rows):
    actual = np.asarray([row["actual_us"] for row in rows], dtype=np.float64)
    predicted = np.asarray([row["predicted_us"] for row in rows], dtype=np.float64)
    ape = np.abs(predicted - actual) / actual
    residual = predicted - actual
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        "samples": len(rows),
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
        "p95_ape": percentile(ape, 95),
        "mae_us": float(np.mean(np.abs(residual))),
        "rmse_us": float(np.sqrt(np.mean(residual**2))),
        "r2": 1.0 - float(np.sum(residual**2)) / denominator if denominator else 0.0,
    }


class ExactOpCurve:
    def __init__(self, rows):
        grouped = defaultdict(list)
        backend_proxy = defaultdict(set)
        observed_backend = defaultdict(set)
        for row in rows:
            key = (row["op"], int(row["group_size"]), int(row["payload_bytes"]))
            grouped[key].extend(row["post_rendezvous_samples_us"])
            backend_proxy[key].add(row["backend_proxy_pre_run"])
            observed_backend[key].add(row["observed_backend_audit_only"])
        self.cost = {}
        self.backend_proxy = {}
        self.observed_backend = {}
        for key, values in grouped.items():
            if len(backend_proxy[key]) != 1 or len(observed_backend[key]) != 1:
                raise ValueError(f"unstable backend mapping at {key}")
            self.cost[key] = float(np.median(values))
            self.backend_proxy[key] = next(iter(backend_proxy[key]))
            self.observed_backend[key] = next(iter(observed_backend[key]))

    def lookup(self, op, tp, payload):
        key = (op, tp, payload)
        if key not in self.cost:
            raise KeyError(f"missing exact Phase 14F support: {key}")
        return self.cost[key]


def load_curve_records(root, expected_repeats):
    records = []
    repeat_ids = defaultdict(set)
    source_files = sorted(root.glob("tp*/*/r*/curve.jsonl"))
    for path in source_files:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["op"], int(row["group_size"]), int(row["payload_bytes"]))
            repeat_ids[key].add(int(row["repeat_id"]))
            records.append(row)
    if len(repeat_ids) != 105:
        raise ValueError(f"expected 105 support points, got {len(repeat_ids)}")
    expected = set(range(expected_repeats))
    bad = {key: values for key, values in repeat_ids.items() if values != expected}
    if bad:
        raise ValueError(f"incomplete repeats: {bad}")
    return records, source_files


def load_dataset(path, phase2_curve, phase14f_curve):
    rows = []
    for source in read_csv(path):
        tp = int(source["tp"])
        by_op_payload = json.loads(source["calls_by_op_payload_json"])
        phase2_base = 0.0
        phase14f_base = 0.0
        for key, count in by_op_payload.items():
            op, payload = key.rsplit(":", 1)
            payload = int(payload)
            count = int(count)
            phase2_base += count * phase2_curve.lookup(tp, payload)
            phase14f_base += count * phase14f_curve.lookup(op, tp, payload)
        rows.append(
            {
                **source,
                "tp": tp,
                "target_post_us": float(source["target_post_us"]),
                "phase2_payload_only_base_us": phase2_base,
                "phase14f_op_backend_base_us": phase14f_base,
            }
        )
    if len(rows) != 162:
        raise ValueError(f"expected 162 configurations, got {len(rows)}")
    return rows


def phase14d_predictions(path):
    rows = [
        row
        for row in read_csv(path)
        if row["evaluation"] == "workload_cv"
        and row["method"] == "tp_conditioned_pattern"
    ]
    if len(rows) != 162:
        raise ValueError(f"expected 162 Phase 14D predictions, got {len(rows)}")
    keyed = {(row["workload_id"], int(row["tp"])): row for row in rows}
    if len(keyed) != 162:
        raise ValueError("Phase 14D predictions are not unique by workload_id and TP")
    return keyed


def fit_phase_scales(rows, base_field):
    scales = {}
    for phase in ("prefill", "decode"):
        subset = [row for row in rows if row["phase"] == phase]
        x = np.asarray([row[base_field] for row in subset], dtype=np.float64)
        y = np.asarray([row["target_post_us"] for row in subset], dtype=np.float64)
        denominator = float(np.dot(x, x))
        scales[phase] = max(0.0, float(np.dot(x, y)) / denominator) if denominator else 1.0
    return scales


def add_scaled_predictions(rows, base_field, method, group_field):
    predictions = []
    groups = sorted({row[group_field] for row in rows})
    for held_out in groups:
        train = [row for row in rows if row[group_field] != held_out]
        test = [row for row in rows if row[group_field] == held_out]
        scales = fit_phase_scales(train, base_field)
        for row in test:
            predicted = row[base_field] * scales[row["phase"]]
            predictions.append(
                {
                    "evaluation": (
                        "workload_cv" if group_field == "fold" else "leave_one_model_out"
                    ),
                    "outer_id": str(held_out),
                    "method": method,
                    "workload_id": row["workload_id"],
                    "fold": row["fold"],
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


def add_raw_predictions(rows, base_field, method):
    return [
        {
            "evaluation": "direct_no_label_fit",
            "outer_id": "all",
            "method": method,
            "workload_id": row["workload_id"],
            "fold": row["fold"],
            "model": row["model"],
            "tp": row["tp"],
            "phase": row["phase"],
            "case_label": row["case_label"],
            "actual_us": row["target_post_us"],
            "predicted_us": row[base_field],
            "absolute_percentage_error": abs(row[base_field] - row["target_post_us"])
            / row["target_post_us"],
            "prefill_scale": "",
            "decode_scale": "",
        }
        for row in rows
    ]


def build_metrics(predictions):
    metrics = []
    keys = sorted({(row["evaluation"], row["method"]) for row in predictions})
    for evaluation, method in keys:
        base = [
            row
            for row in predictions
            if row["evaluation"] == evaluation and row["method"] == method
        ]
        scopes = {"all": base}
        for phase in ("prefill", "decode"):
            scopes[phase] = [row for row in base if row["phase"] == phase]
        if evaluation == "leave_one_model_out":
            for model in sorted({row["model"] for row in base}):
                scopes[f"model:{model}"] = [row for row in base if row["model"] == model]
        for scope, subset in scopes.items():
            metrics.append(
                {
                    "evaluation": evaluation,
                    "scope": scope,
                    "method": method,
                    **regression_metrics(subset),
                }
            )
    return metrics


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve_records, source_files = load_curve_records(
        args.curve_root, args.expected_repeats
    )
    phase14f_curve = ExactOpCurve(curve_records)
    phase2_curve = BackendAwareCostCurve(
        args.phase2_custom_curve,
        args.phase2_nccl_curve,
        custom_latency_column="completion_median_latency_us",
        nccl_latency_column="intrinsic_min_median_latency_us",
    )
    rows = load_dataset(args.dataset, phase2_curve, phase14f_curve)
    phase14d = phase14d_predictions(args.phase14d_predictions)
    for row in rows:
        row["fold"] = int(phase14d[(row["workload_id"], row["tp"])]["fold"])

    predictions = []
    predictions.extend(
        add_raw_predictions(rows, "phase2_payload_only_base_us", "phase2_payload_only_raw")
    )
    predictions.extend(
        add_raw_predictions(rows, "phase14f_op_backend_base_us", "phase14f_op_backend_raw")
    )
    for base_field, method in (
        ("phase2_payload_only_base_us", "phase2_payload_only_scaled"),
        ("phase14f_op_backend_base_us", "phase14f_op_backend_scaled"),
    ):
        predictions.extend(add_scaled_predictions(rows, base_field, method, "fold"))
        predictions.extend(add_scaled_predictions(rows, base_field, method, "model"))
    for row in rows:
        existing = phase14d[(row["workload_id"], row["tp"])]
        predictions.append(
            {
                "evaluation": "workload_cv",
                "outer_id": existing["outer_id"],
                "method": "phase14d_tp_conditioned_pattern",
                "workload_id": row["workload_id"],
                "fold": row["fold"],
                "model": row["model"],
                "tp": row["tp"],
                "phase": row["phase"],
                "case_label": row["case_label"],
                "actual_us": row["target_post_us"],
                "predicted_us": float(existing["predicted_us"]),
                "absolute_percentage_error": float(existing["absolute_percentage_error"]),
                "prefill_scale": "",
                "decode_scale": "",
            }
        )

    curve_summary = []
    by_support = defaultdict(list)
    for row in curve_records:
        key = (row["op"], int(row["group_size"]), int(row["payload_bytes"]))
        by_support[key].append(row)
    for key, support_rows in sorted(by_support.items()):
        op, tp, payload = key
        post_rendezvous = [
            sample
            for row in support_rows
            for sample in row["post_rendezvous_samples_us"]
        ]
        intrinsic = [
            sample for row in support_rows for sample in row["intrinsic_samples_us"]
        ]
        curve_summary.append(
            {
                "op": op,
                "group_size": tp,
                "payload_bytes": payload,
                "repeats": len(support_rows),
                "pooled_samples": len(post_rendezvous),
                "backend_proxy_pre_run": phase14f_curve.backend_proxy[key],
                "observed_backend_audit_only": phase14f_curve.observed_backend[key],
                "intrinsic_median_us": float(np.median(intrinsic)),
                "post_rendezvous_median_us": float(np.median(post_rendezvous)),
                "post_rendezvous_p95_us": percentile(post_rendezvous, 95),
                "repeat_median_cv": float(
                    np.std(
                        [
                            row["post_rendezvous_latency_us"]["median"]
                            for row in support_rows
                        ]
                    )
                    / np.mean(
                        [
                            row["post_rendezvous_latency_us"]["median"]
                            for row in support_rows
                        ]
                    )
                ),
            }
        )

    metrics = build_metrics(predictions)
    write_csv(args.output_dir / "curve_summary.csv", curve_summary)
    write_csv(args.output_dir / "predictions.csv", predictions)
    write_csv(args.output_dir / "metrics.csv", metrics)

    selected = next(
        row
        for row in metrics
        if row["evaluation"] == "workload_cv"
        and row["scope"] == "all"
        and row["method"] == "phase14f_op_backend_scaled"
    )
    selected_decode = next(
        row
        for row in metrics
        if row["evaluation"] == "workload_cv"
        and row["scope"] == "decode"
        and row["method"] == "phase14f_op_backend_scaled"
    )
    lomo = [
        row
        for row in metrics
        if row["evaluation"] == "leave_one_model_out"
        and row["method"] == "phase14f_op_backend_scaled"
        and row["scope"].startswith("model:")
    ]
    gates = {
        "overall_mape_below_10pct": selected["mape"] < 0.10,
        "overall_p95_below_25pct": selected["p95_ape"] < 0.25,
        "decode_mape_below_10pct": selected_decode["mape"] < 0.10,
        "every_lomo_mape_below_15pct": all(row["mape"] < 0.15 for row in lomo),
    }
    summary_record = {
        "schema_version": "phase14f-backend-curve-analysis-v1",
        "dataset": {
            "configurations": len(rows),
            "models": sorted({row["model"] for row in rows}),
            "tensor_parallel_sizes": sorted({row["tp"] for row in rows}),
            "support_points": len(curve_summary),
            "curve_records": len(curve_records),
            "source_curve_files": [str(path) for path in source_files],
            "curve_samples": sum(
                len(row["post_rendezvous_samples_us"]) for row in curve_records
            ),
        },
        "predictive_contract": {
            "inputs": ["raw_op", "payload_bytes", "group_size", "topology"],
            "backend_proxy": "lookup table measured before inference evaluation",
            "observed_backend": "audit-only; never read from Phase 14C traces by the predictor",
            "structural_formula": "sum(count(op,payload,tp) * C(op,payload,tp,L1,backend_proxy))",
            "calibration": "one nonnegative scalar per phase fitted inside each outer training fold",
        },
        "selected_workload_cv": selected,
        "selected_decode_workload_cv": selected_decode,
        "selected_lomo": lomo,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "max_repeat_median_cv": max(row["repeat_median_cv"] for row in curve_summary),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary_record, indent=2) + "\n"
    )
    readme = f"""# Phase 14F: op/backend-proxy continuous cost curve

This analysis measures 105 exact `raw_op x payload x TP` supports, convolves the
curve with the Phase 14C PatternDemand histograms, and evaluates held-out workload
and leave-one-model-out prediction. Observed inference backend signatures are not
predictive features; they are retained only for post-hoc audit.

Selected workload-CV result (`phase14f_op_backend_scaled`):

- MAPE: {selected['mape'] * 100:.3f}%
- P95 APE: {selected['p95_ape'] * 100:.3f}%
- Decode MAPE: {selected_decode['mape'] * 100:.3f}%
- all convergence gates passed: {all(gates.values())}

See `curve_summary.csv`, `predictions.csv`, `metrics.csv`, and `summary.json` for
the complete evidence and statistical scopes.
"""
    (args.output_dir / "README.md").write_text(readme)
    print(json.dumps(summary_record, indent=2))


if __name__ == "__main__":
    main()
