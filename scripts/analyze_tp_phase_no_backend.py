#!/usr/bin/env python3
"""Evaluate TP/phase-conditioned communication models without backend features."""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TPS = (2, 4, 8)
PHASES = ("prefill", "decode")
MODELS = ("qwen3-8b", "qwen3-30b-a3b")
PAYLOAD_LOG2_CENTERS = np.arange(12.0, 29.0, 2.0)
RIDGE_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
METHODS = (
    "pattern_demand_only",
    "pattern_demand_plus_tp",
    "pattern_demand_plus_phase",
    "pattern_demand_plus_tp_phase",
)
PREFERRED_CANDIDATE = "pattern_demand_plus_tp_phase"
FORBIDDEN_PREDICTIVE_SUBSTRINGS = ("backend", "kernel_name", "model")


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=repo_root
        / "experiment-results/phase14/tp_group_size_timing_analysis"
        / "aggregated_configurations.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "experiment-results/phase14/tp_phase_no_backend_analysis",
    )
    return parser.parse_args()


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def parse_payload_histogram(serialized):
    return {int(payload): int(count) for payload, count in json.loads(serialized).items()}


def parse_op_payload_histogram(serialized):
    result = {}
    for key, count in json.loads(serialized).items():
        raw_op, payload = key.rsplit(":", 1)
        result[(raw_op, int(payload))] = int(count)
    return result


def load_rows(path):
    rows = list(csv.DictReader(path.open()))
    required = {
        "workload_id",
        "model",
        "tp",
        "phase",
        "mode",
        "case_label",
        "calls",
        "logical_payload_bytes",
        "calls_by_payload_json",
        "calls_by_op_payload_json",
        "backend_signature",
        "target_post_us",
    }
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    for row in rows:
        row["tp"] = int(row["tp"])
        row["calls"] = int(row["calls"])
        row["logical_payload_bytes"] = int(row["logical_payload_bytes"])
        row["target_post_us"] = float(row["target_post_us"])
        row["payload_histogram"] = parse_payload_histogram(
            row["calls_by_payload_json"]
        )
        row["op_payload_histogram"] = parse_op_payload_histogram(
            row["calls_by_op_payload_json"]
        )
        if row["tp"] not in TPS or row["phase"] not in PHASES:
            raise ValueError(f"unexpected TP/phase row: {row}")
        payload_calls = sum(row["payload_histogram"].values())
        payload_bytes = sum(
            payload * count for payload, count in row["payload_histogram"].items()
        )
        if payload_calls != row["calls"] or payload_bytes != row["logical_payload_bytes"]:
            raise ValueError(f"payload histogram totals disagree: {row['workload_id']}")
        marginal = defaultdict(int)
        for (raw_op, payload), count in row["op_payload_histogram"].items():
            if raw_op not in ("all_reduce", "fused_allreduce_residual_rmsnorm"):
                raise ValueError(f"unexpected raw op: {raw_op}")
            marginal[payload] += count
        if dict(sorted(marginal.items())) != dict(sorted(row["payload_histogram"].items())):
            raise ValueError(f"raw-op histogram marginal mismatch: {row['workload_id']}")

    if len(rows) != 90:
        raise ValueError(f"expected 90 aggregate rows, got {len(rows)}")
    groups = defaultdict(list)
    for row in rows:
        groups[row["workload_id"]].append(row)
    if len(groups) != 30:
        raise ValueError(f"expected 30 workload groups, got {len(groups)}")
    for identifier, group in groups.items():
        if {row["tp"] for row in group} != set(TPS) or len(group) != len(TPS):
            raise ValueError(f"{identifier}: incomplete or duplicate TP variants")
    return rows


def build_balanced_folds(rows):
    representatives = {}
    for row in rows:
        representatives.setdefault(row["workload_id"], row)
    strata = defaultdict(list)
    for identifier, row in representatives.items():
        if row["phase"] == "prefill":
            key = ("prefill", row["model"], row["case_label"])
        else:
            key = ("decode", row["model"])
        strata[key].append(identifier)

    assignment = {}
    for key, identifiers in sorted(strata.items()):
        identifiers = sorted(identifiers)
        if key[0] == "decode":
            if len(identifiers) != 3:
                raise ValueError(f"{key}: expected three Decode groups")
            slots = (0, 2, 4) if key[1] == "qwen3-30b-a3b" else (1, 3, 5)
        else:
            if len(identifiers) != 6:
                raise ValueError(f"{key}: expected six Prefill groups")
            slots = tuple(range(6))
        assignment.update(dict(zip(identifiers, slots)))

    if Counter(assignment.values()) != {fold: 5 for fold in range(6)}:
        raise ValueError("outer folds are not balanced at five workload groups each")
    for row in rows:
        row["fold"] = assignment[row["workload_id"]]
    for fold in range(6):
        subset = [row for row in rows if row["fold"] == fold]
        if len(subset) != 15:
            raise ValueError(f"fold {fold}: expected 15 TP-expanded rows")
        if Counter(row["phase"] for row in subset) != {"prefill": 12, "decode": 3}:
            raise ValueError(f"fold {fold}: phase balance failed")

    output = []
    for identifier, fold in sorted(assignment.items()):
        row = representatives[identifier]
        output.append(
            {
                "workload_id": identifier,
                "fold": fold,
                "model": row["model"],
                "phase": row["phase"],
                "mode": row["mode"],
                "case_label": row["case_label"],
                "tp_variants_kept_together": "2,4,8",
            }
        )
    return output


def soft_histogram_features(histogram):
    values = []
    for center in PAYLOAD_LOG2_CENTERS:
        value = sum(
            count * max(0.0, 1.0 - abs(math.log2(payload) - center) / 2.0)
            for payload, count in histogram.items()
        )
        values.append(math.log1p(value))
    return values


def core_feature_names():
    names = [
        "log_calls",
        "log_logical_payload_bytes",
        "weighted_log2_payload_mean",
        "weighted_log2_payload_std",
        "min_log2_payload",
        "max_log2_payload",
        "log_payload_support_count",
    ]
    names.extend(
        f"log_soft_call_count_log2_{int(center)}"
        for center in PAYLOAD_LOG2_CENTERS
    )
    names.extend(
        [
            "log_fused_raw_op_calls",
            "log_fused_raw_op_bytes",
            "fused_raw_op_call_fraction",
            "fused_raw_op_byte_fraction",
        ]
    )
    names.extend(
        f"log_fused_soft_call_count_log2_{int(center)}"
        for center in PAYLOAD_LOG2_CENTERS
    )
    return names


def core_features(row):
    histogram = row["payload_histogram"]
    payloads = np.asarray(list(histogram), dtype=np.float64)
    counts = np.asarray(list(histogram.values()), dtype=np.float64)
    logs = np.log2(payloads)
    calls = float(np.sum(counts))
    logical_bytes = float(np.sum(payloads * counts))
    mean = float(np.average(logs, weights=counts))
    std = float(np.sqrt(np.average((logs - mean) ** 2, weights=counts)))
    fused = {
        payload: count
        for (raw_op, payload), count in row["op_payload_histogram"].items()
        if raw_op == "fused_allreduce_residual_rmsnorm"
    }
    fused_calls = float(sum(fused.values()))
    fused_bytes = float(sum(payload * count for payload, count in fused.items()))
    return np.asarray(
        [
            math.log1p(calls),
            math.log1p(logical_bytes),
            mean,
            std,
            float(np.min(logs)),
            float(np.max(logs)),
            math.log1p(len(histogram)),
            *soft_histogram_features(histogram),
            math.log1p(fused_calls),
            math.log1p(fused_bytes),
            fused_calls / calls,
            fused_bytes / logical_bytes,
            *soft_histogram_features(fused),
        ],
        dtype=np.float64,
    )


def feature_names(method):
    names = core_feature_names()
    if method in ("pattern_demand_plus_tp", "pattern_demand_plus_tp_phase"):
        names.extend(("tp_is_4", "tp_is_8"))
    if method in ("pattern_demand_plus_phase", "pattern_demand_plus_tp_phase"):
        names.append("phase_is_decode")
    if method == "pattern_demand_plus_tp_phase":
        names.extend(("tp4_x_decode", "tp8_x_decode"))
    for name in names:
        if any(forbidden in name for forbidden in FORBIDDEN_PREDICTIVE_SUBSTRINGS):
            raise ValueError(f"forbidden predictive feature name: {name}")
    return names


def feature_vector(row, method):
    values = list(core_features(row))
    if method in ("pattern_demand_plus_tp", "pattern_demand_plus_tp_phase"):
        values.extend((row["tp"] == 4, row["tp"] == 8))
    if method in ("pattern_demand_plus_phase", "pattern_demand_plus_tp_phase"):
        values.append(row["phase"] == "decode")
    if method == "pattern_demand_plus_tp_phase":
        values.extend(
            (
                row["tp"] == 4 and row["phase"] == "decode",
                row["tp"] == 8 and row["phase"] == "decode",
            )
        )
    result = np.asarray(values, dtype=np.float64)
    if len(result) != len(feature_names(method)):
        raise ValueError(f"{method}: feature name/vector size mismatch")
    return result


def fit_ridge(rows, method, alpha):
    features = np.vstack([feature_vector(row, method) for row in rows])
    target = np.log(
        np.asarray([row["target_post_us"] for row in rows], dtype=np.float64)
    )
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale[scale < 1e-12] = 1.0
    design = np.column_stack(
        [np.ones(len(features)), (features - mean) / scale]
    )
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty, design.T @ target
    )
    return mean, scale, coefficients


def ridge_predict(model, rows, method):
    mean, scale, coefficients = model
    features = np.vstack([feature_vector(row, method) for row in rows])
    design = np.column_stack([np.ones(len(features)), (features - mean) / scale])
    return np.exp(design @ coefficients)


def select_alpha(train_rows, method):
    inner_folds = sorted({row["fold"] for row in train_rows})
    if len(inner_folds) < 2:
        raise ValueError("nested selection requires at least two inner folds")
    scores = []
    for alpha in RIDGE_ALPHAS:
        errors = []
        for fold in inner_folds:
            inner_train = [row for row in train_rows if row["fold"] != fold]
            inner_validation = [row for row in train_rows if row["fold"] == fold]
            if not inner_train or not inner_validation:
                raise ValueError("empty nested train/validation split")
            predictions = ridge_predict(
                fit_ridge(inner_train, method, alpha), inner_validation, method
            )
            errors.extend(
                abs(predicted - row["target_post_us"]) / row["target_post_us"]
                for predicted, row in zip(predictions, inner_validation)
            )
        scores.append((float(np.mean(errors)), alpha))
    return min(scores)


def prediction_row(evaluation, outer_id, method, row, predicted, alpha):
    actual = row["target_post_us"]
    return {
        "evaluation": evaluation,
        "outer_id": outer_id,
        "method": method,
        "selected_alpha": alpha,
        "workload_id": row["workload_id"],
        "fold": row["fold"],
        "model": row["model"],
        "tp": row["tp"],
        "phase": row["phase"],
        "case_label": row["case_label"],
        "actual_us": actual,
        "predicted_us": float(predicted),
        "absolute_percentage_error": abs(float(predicted) - actual) / actual,
        "backend_signature_posthoc_only": row["backend_signature"],
        "backend_used_as_predictive_feature": False,
    }


def run_workload_cv(rows):
    predictions = []
    selection = []
    for outer_fold in range(6):
        train = [row for row in rows if row["fold"] != outer_fold]
        test = [row for row in rows if row["fold"] == outer_fold]
        if {row["workload_id"] for row in train} & {
            row["workload_id"] for row in test
        }:
            raise ValueError("workload leakage across outer fold")
        for method in METHODS:
            inner_mape, alpha = select_alpha(train, method)
            predicted = ridge_predict(fit_ridge(train, method, alpha), test, method)
            selection.append(
                {
                    "evaluation": "workload_cv",
                    "outer_id": f"fold{outer_fold}",
                    "method": method,
                    "selected_alpha": alpha,
                    "inner_validation_mape": inner_mape,
                    "train_workload_groups": len(
                        {row["workload_id"] for row in train}
                    ),
                    "test_workload_groups": len(
                        {row["workload_id"] for row in test}
                    ),
                }
            )
            predictions.extend(
                prediction_row(
                    "workload_cv", f"fold{outer_fold}", method, row, value, alpha
                )
                for row, value in zip(test, predicted)
            )
    expected = len(rows) * len(METHODS)
    if len(predictions) != expected:
        raise ValueError(f"expected {expected} workload-CV predictions")
    return predictions, selection


def run_leave_one_model_out(rows):
    predictions = []
    selection = []
    for held_out_model in MODELS:
        train = [row for row in rows if row["model"] != held_out_model]
        test = [row for row in rows if row["model"] == held_out_model]
        for method in METHODS:
            inner_mape, alpha = select_alpha(train, method)
            predicted = ridge_predict(fit_ridge(train, method, alpha), test, method)
            selection.append(
                {
                    "evaluation": "leave_one_model_out",
                    "outer_id": held_out_model,
                    "method": method,
                    "selected_alpha": alpha,
                    "inner_validation_mape": inner_mape,
                    "train_workload_groups": len(
                        {row["workload_id"] for row in train}
                    ),
                    "test_workload_groups": len(
                        {row["workload_id"] for row in test}
                    ),
                }
            )
            predictions.extend(
                prediction_row(
                    "leave_one_model_out",
                    held_out_model,
                    method,
                    row,
                    value,
                    alpha,
                )
                for row, value in zip(test, predicted)
            )
    expected = len(rows) * len(METHODS)
    if len(predictions) != expected:
        raise ValueError(f"expected {expected} model-holdout predictions")
    return predictions, selection


def metric_row(evaluation, scope, method, predictions):
    actual = np.asarray([row["actual_us"] for row in predictions], dtype=np.float64)
    predicted = np.asarray(
        [row["predicted_us"] for row in predictions], dtype=np.float64
    )
    ape = np.abs(predicted - actual) / actual
    residual = float(np.sum((predicted - actual) ** 2))
    total = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        "evaluation": evaluation,
        "scope": scope,
        "method": method,
        "samples": len(predictions),
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
        "p95_ape": percentile(ape, 95),
        "mae_us": float(np.mean(np.abs(predicted - actual))),
        "rmse_us": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "r2": 1.0 - residual / total if total > 0 else float("nan"),
    }


def build_metrics(workload_predictions, model_predictions):
    output = []
    for method in METHODS:
        method_rows = [
            row for row in workload_predictions if row["method"] == method
        ]
        scopes = {"all": method_rows}
        for phase in PHASES:
            scopes[phase] = [row for row in method_rows if row["phase"] == phase]
        for tp in TPS:
            scopes[f"tp{tp}"] = [row for row in method_rows if row["tp"] == tp]
        for model in MODELS:
            scopes[model] = [row for row in method_rows if row["model"] == model]
        output.extend(
            metric_row("workload_cv", scope, method, subset)
            for scope, subset in scopes.items()
        )

        method_rows = [row for row in model_predictions if row["method"] == method]
        output.append(
            metric_row("leave_one_model_out", "all", method, method_rows)
        )
        for model in MODELS:
            subset = [row for row in method_rows if row["outer_id"] == model]
            output.append(
                metric_row("leave_one_model_out", model, method, subset)
            )
            for phase in PHASES:
                phase_subset = [row for row in subset if row["phase"] == phase]
                output.append(
                    metric_row(
                        "leave_one_model_out",
                        f"{model}_{phase}",
                        method,
                        phase_subset,
                    )
                )
    return output


def backend_family(signature):
    families = sorted({part.split(":", 1)[0] for part in signature.split("+")})
    return "+".join(families)


def build_posthoc_backend_diagnostics(rows, predictions):
    signatures = defaultdict(set)
    for row in rows:
        signatures[row["workload_id"]].add(row["backend_signature"])
    selected = [
        row
        for row in predictions
        if row["method"] == PREFERRED_CANDIDATE
    ]
    groups = defaultdict(list)
    for row in selected:
        transition = len(signatures[row["workload_id"]]) > 1
        groups[("backend_transition", str(transition).lower())].append(row)
        groups[
            (
                "observed_backend_family",
                backend_family(row["backend_signature_posthoc_only"]),
            )
        ].append(row)
    output = []
    for (diagnostic, value), subset in sorted(groups.items()):
        metric = metric_row(
            "workload_cv_posthoc", f"{diagnostic}={value}", PREFERRED_CANDIDATE, subset
        )
        metric["diagnostic_dimension"] = diagnostic
        metric["diagnostic_value"] = value
        metric["backend_used_as_predictive_feature"] = False
        output.append(metric)
    return output


def metric_lookup(metrics, evaluation, scope, method):
    return next(
        row
        for row in metrics
        if row["evaluation"] == evaluation
        and row["scope"] == scope
        and row["method"] == method
    )


def make_figure(metrics, path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    labels = ("PD", "+TP", "+phase", "+TP×phase")
    x = np.arange(len(METHODS))
    all_metrics = [
        metric_lookup(metrics, "workload_cv", "all", method)
        for method in METHODS
    ]
    axes[0].bar(x - 0.18, [100 * row["mape"] for row in all_metrics], 0.36, label="MAPE")
    axes[0].bar(x + 0.18, [100 * row["p95_ape"] for row in all_metrics], 0.36, label="P95 APE")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Error (%)")
    axes[0].set_title("6-fold workload CV")
    axes[0].legend()

    phase_metrics = [
        metric_lookup(metrics, "workload_cv", scope, PREFERRED_CANDIDATE)
        for scope in ("all", "prefill", "decode")
    ]
    axes[1].bar(
        np.arange(3) - 0.18,
        [100 * row["mape"] for row in phase_metrics],
        0.36,
        label="MAPE",
    )
    axes[1].bar(
        np.arange(3) + 0.18,
        [100 * row["p95_ape"] for row in phase_metrics],
        0.36,
        label="P95 APE",
    )
    axes[1].set_xticks(np.arange(3), ("All", "Prefill", "Decode"))
    axes[1].set_ylabel("Error (%)")
    axes[1].set_title("Backend-free TP×phase candidate")
    axes[1].legend()

    held_metrics = [
        metric_lookup(metrics, "leave_one_model_out", model, PREFERRED_CANDIDATE)
        for model in MODELS
    ]
    axes[2].bar(
        np.arange(2) - 0.18,
        [100 * row["mape"] for row in held_metrics],
        0.36,
        label="MAPE",
    )
    axes[2].bar(
        np.arange(2) + 0.18,
        [100 * row["p95_ape"] for row in held_metrics],
        0.36,
        label="P95 APE",
    )
    axes[2].set_xticks(np.arange(2), ("Hold Qwen3-8B", "Hold Qwen3-30B"))
    axes[2].set_ylabel("Error (%)")
    axes[2].set_title("Leave-one-model-out")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.input_csv)
    folds = build_balanced_folds(rows)
    workload_predictions, workload_selection = run_workload_cv(rows)
    model_predictions, model_selection = run_leave_one_model_out(rows)
    metrics = build_metrics(workload_predictions, model_predictions)
    posthoc = build_posthoc_backend_diagnostics(rows, workload_predictions)

    write_csv(args.output_dir / "fold_assignments.csv", folds)
    write_csv(
        args.output_dir / "cv_predictions.csv",
        workload_predictions + model_predictions,
    )
    write_csv(
        args.output_dir / "alpha_selection.csv",
        workload_selection + model_selection,
    )
    write_csv(args.output_dir / "metrics.csv", metrics)
    write_csv(args.output_dir / "posthoc_backend_diagnostics.csv", posthoc)
    make_figure(metrics, args.output_dir / "tp_phase_no_backend_analysis.png")

    best = metric_lookup(
        metrics, "workload_cv", "all", PREFERRED_CANDIDATE
    )
    prefill = metric_lookup(
        metrics, "workload_cv", "prefill", PREFERRED_CANDIDATE
    )
    decode = metric_lookup(
        metrics, "workload_cv", "decode", PREFERRED_CANDIDATE
    )
    held_out = {
        model: metric_lookup(
            metrics, "leave_one_model_out", model, PREFERRED_CANDIDATE
        )
        for model in MODELS
    }
    summary = {
        "schema_version": "tp-phase-no-backend-analysis-v1",
        "dataset": {
            "source": str(args.input_csv),
            "aggregated_configurations": len(rows),
            "workload_groups": len({row["workload_id"] for row in rows}),
            "models": list(MODELS),
            "tensor_parallel_sizes": list(TPS),
            "phases": list(PHASES),
            "target": "median of three repeats of all-rank post-rendezvous completion time",
        },
        "leakage_audit": {
            "actual_backend_feature_used": False,
            "backend_signature_use": "post-hoc residual diagnostics only",
            "kernel_name_feature_used": False,
            "model_identity_feature_used": False,
            "phase2_cost_curve_feature_used": False,
            "logical_raw_op_histogram_feature_used": True,
            "raw_op_note": "raw_op is pre-run logical PatternDemand, not the observed runtime backend",
            "outer_validation": "six balanced folds grouped by complete workload_id",
            "same_workload_tp_variants_kept_together": True,
            "ridge_alpha_selection": "nested within each outer training split",
            "repeat_leakage": False,
        },
        "feature_specification": {
            method: feature_names(method) for method in METHODS
        },
        "ridge_alphas": list(RIDGE_ALPHAS),
        "workload_cv": {
            "folds": 6,
            "test_workload_groups_per_fold": 5,
            "test_configurations_per_fold": 15,
            "preferred_candidate": PREFERRED_CANDIDATE,
            "all": best,
            "prefill": prefill,
            "decode": decode,
        },
        "leave_one_model_out": held_out,
        "decision_gates": {
            "overall_mape_below_10pct": best["mape"] < 0.10,
            "overall_p95_below_25pct": best["p95_ape"] < 0.25,
            "prefill_mape_below_15pct": prefill["mape"] < 0.15,
            "decode_mape_below_10pct": decode["mape"] < 0.10,
            "both_held_out_models_mape_below_15pct": all(
                row["mape"] < 0.15 for row in held_out.values()
            ),
        },
        "interpretation": {
            "result": "TP and phase conditioning materially improve workload-held-out prediction without using actual backend, but the model does not pass overall tail, Decode mean, or cross-model gates.",
            "recommended_status": "useful baseline, not production-ready default",
            "next_step": "Do not add observed backend as an input. Add more Decode/model diversity and test pre-run structural features or a deterministic dispatch proxy only if the backend-free residual remains unacceptable.",
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=False) + "\n"
    )
    print(
        f"analyzed {len(rows)} configurations; "
        f"backend-free TP×phase workload-CV MAPE={best['mape']:.6f}, "
        f"P95={best['p95_ape']:.6f}"
    )


if __name__ == "__main__":
    main()
