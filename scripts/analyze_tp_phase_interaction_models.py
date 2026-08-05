#!/usr/bin/env python3
"""Phase 14D: compare backend-free TP/phase-conditioned PatternDemand slopes."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import analyze_tp_phase_no_backend as baseline


METHODS = (
    "additive_tp_phase",
    "ring_equivalent_pattern",
    "tp_conditioned_pattern",
    "phase_conditioned_pattern",
    "tp_phase_conditioned_pattern",
)
PRIMARY_CANDIDATE = "tp_phase_conditioned_pattern"
BASELINE_CANDIDATE = "additive_tp_phase"
INTERCEPT_NAMES = (
    "tp_is_4",
    "tp_is_8",
    "phase_is_decode",
    "tp4_x_decode",
    "tp8_x_decode",
)
FORBIDDEN_PREDICTIVE_SUBSTRINGS = ("backend", "kernel_name", "model_identity")


def parse_args():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=repo
        / "experiment-results/phase14c/extended_dataset_analysis"
        / "aggregated_configurations.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "experiment-results/phase14d/tp_phase_interaction_analysis",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="validate features and fit one outer fold without nested selection",
    )
    return parser.parse_args()


def interaction_names(prefix):
    return [f"{prefix}_x_{name}" for name in baseline.core_feature_names()]


def feature_names(method):
    core = baseline.core_feature_names()
    if method == "ring_equivalent_pattern":
        names = [f"ring_equivalent_{name}" for name in core]
        names.append("phase_is_decode")
    else:
        names = list(core) + list(INTERCEPT_NAMES)
        if method in ("tp_conditioned_pattern", "tp_phase_conditioned_pattern"):
            names.extend(interaction_names("tp4"))
            names.extend(interaction_names("tp8"))
        if method in ("phase_conditioned_pattern", "tp_phase_conditioned_pattern"):
            names.extend(interaction_names("decode"))
        if method == "tp_phase_conditioned_pattern":
            names.extend(interaction_names("tp4_decode"))
            names.extend(interaction_names("tp8_decode"))
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    if len(names) != len(set(names)):
        raise ValueError(f"{method}: duplicate feature names")
    for name in names:
        if any(token in name for token in FORBIDDEN_PREDICTIVE_SUBSTRINGS):
            raise ValueError(f"forbidden predictive feature name: {name}")
    return names


def ring_equivalent_core_features(row):
    """Map each ring AllReduce call to 2*(TP-1) rounds of payload/TP bytes."""
    rounds = 2 * (row["tp"] - 1)
    ring_row = dict(row)
    ring_row["payload_histogram"] = {
        float(payload) / row["tp"]: count * rounds
        for payload, count in row["payload_histogram"].items()
    }
    ring_row["op_payload_histogram"] = {
        (raw_op, float(payload) / row["tp"]): count * rounds
        for (raw_op, payload), count in row["op_payload_histogram"].items()
    }
    return baseline.core_features(ring_row)


def feature_vector(row, method):
    decode = float(row["phase"] == "decode")
    tp4 = float(row["tp"] == 4)
    tp8 = float(row["tp"] == 8)
    if method == "ring_equivalent_pattern":
        values = list(ring_equivalent_core_features(row)) + [decode]
    else:
        core = baseline.core_features(row)
        values = list(core) + [tp4, tp8, decode, tp4 * decode, tp8 * decode]
        if method in ("tp_conditioned_pattern", "tp_phase_conditioned_pattern"):
            values.extend(tp4 * core)
            values.extend(tp8 * core)
        if method in ("phase_conditioned_pattern", "tp_phase_conditioned_pattern"):
            values.extend(decode * core)
        if method == "tp_phase_conditioned_pattern":
            values.extend(tp4 * decode * core)
            values.extend(tp8 * decode * core)
    result = np.asarray(values, dtype=np.float64)
    if len(result) != len(feature_names(method)):
        raise ValueError(f"{method}: feature name/vector size mismatch")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{method}: non-finite feature value")
    return result


def configure_baseline(models):
    """Reuse the audited split/ridge implementation with Phase 14D features."""
    baseline.MODELS = tuple(models)
    baseline.METHODS = METHODS
    baseline.PREFERRED_CANDIDATE = PRIMARY_CANDIDATE
    baseline.feature_names = feature_names
    baseline.feature_vector = feature_vector


def add_profile_metrics(metrics, predictions):
    labels = sorted({row["case_label"] for row in predictions})
    for method in METHODS:
        method_rows = [row for row in predictions if row["method"] == method]
        for label in labels:
            subset = [row for row in method_rows if row["case_label"] == label]
            if subset:
                metrics.append(
                    baseline.metric_row(
                        "workload_cv", f"profile:{label}", method, subset
                    )
                )


def metric(metrics, evaluation, scope, method):
    return baseline.metric_lookup(metrics, evaluation, scope, method)


def compact_metric(row):
    return {
        key: row[key]
        for key in (
            "samples",
            "mape",
            "median_ape",
            "p95_ape",
            "mae_us",
            "rmse_us",
            "r2",
        )
    }


def method_comparison(metrics):
    return {
        method: compact_metric(metric(metrics, "workload_cv", "all", method))
        for method in METHODS
    }


def make_figure(metrics, models, path):
    fig, axes = plt.subplots(2, 2, figsize=(17, 10.5))
    labels = ("Additive", "Ring equiv.", "TP slopes", "Phase slopes", "TP×phase slopes")
    x = np.arange(len(METHODS))
    all_rows = [metric(metrics, "workload_cv", "all", method) for method in METHODS]
    axes[0, 0].bar(x - 0.18, [100 * row["mape"] for row in all_rows], 0.36, label="MAPE")
    axes[0, 0].bar(x + 0.18, [100 * row["p95_ape"] for row in all_rows], 0.36, label="P95 APE")
    axes[0, 0].set_xticks(x, labels, rotation=15)
    axes[0, 0].set_ylabel("Error (%)")
    axes[0, 0].set_title("6-fold workload CV")
    axes[0, 0].legend()

    phase_scopes = ("all", "prefill", "decode")
    phase_rows = [metric(metrics, "workload_cv", scope, PRIMARY_CANDIDATE) for scope in phase_scopes]
    axes[0, 1].bar(np.arange(3) - 0.18, [100 * row["mape"] for row in phase_rows], 0.36, label="MAPE")
    axes[0, 1].bar(np.arange(3) + 0.18, [100 * row["p95_ape"] for row in phase_rows], 0.36, label="P95 APE")
    axes[0, 1].set_xticks(np.arange(3), ("All", "Prefill", "Decode"))
    axes[0, 1].set_ylabel("Error (%)")
    axes[0, 1].set_title("Primary TP×phase-conditioned slopes")
    axes[0, 1].legend()

    profiles = sorted(
        row["scope"].split(":", 1)[1]
        for row in metrics
        if row["evaluation"] == "workload_cv"
        and row["method"] == PRIMARY_CANDIDATE
        and row["scope"].startswith("profile:")
    )
    profile_rows = [
        metric(metrics, "workload_cv", f"profile:{profile}", PRIMARY_CANDIDATE)
        for profile in profiles
    ]
    px = np.arange(len(profiles))
    axes[1, 0].bar(px - 0.18, [100 * row["mape"] for row in profile_rows], 0.36, label="MAPE")
    axes[1, 0].bar(px + 0.18, [100 * row["p95_ape"] for row in profile_rows], 0.36, label="P95 APE")
    axes[1, 0].set_xticks(px, profiles, rotation=20)
    axes[1, 0].set_ylabel("Error (%)")
    axes[1, 0].set_title("Primary candidate by workload shape")
    axes[1, 0].legend()

    held_rows = [
        metric(metrics, "leave_one_model_out", model, PRIMARY_CANDIDATE)
        for model in models
    ]
    mx = np.arange(len(models))
    axes[1, 1].bar(mx - 0.18, [100 * row["mape"] for row in held_rows], 0.36, label="MAPE")
    axes[1, 1].bar(mx + 0.18, [100 * row["p95_ape"] for row in held_rows], 0.36, label="P95 APE")
    axes[1, 1].set_xticks(mx, [f"Hold {model}" for model in models], rotation=15)
    axes[1, 1].set_ylabel("Error (%)")
    axes[1, 1].set_title("Leave-one-model-out")
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_smoke(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    train = [row for row in rows if row["fold"] != 0]
    test = [row for row in rows if row["fold"] == 0]
    checks = {}
    for method in METHODS:
        matrix = np.vstack([feature_vector(row, method) for row in rows])
        predicted = baseline.ridge_predict(
            baseline.fit_ridge(train, method, 1.0), test, method
        )
        ape = np.abs(
            predicted - np.asarray([row["target_post_us"] for row in test])
        ) / np.asarray([row["target_post_us"] for row in test])
        checks[method] = {
            "features": matrix.shape[1],
            "rows": matrix.shape[0],
            "finite": bool(np.all(np.isfinite(matrix))),
            "fold0_predictions": len(predicted),
            "fold0_mape": float(np.mean(ape)),
        }
    summary = {
        "schema_version": "phase14d-interaction-smoke-v1",
        "status": "passed",
        "input_rows": len(rows),
        "workload_groups": len({row["workload_id"] for row in rows}),
        "methods": checks,
        "actual_backend_feature_used": False,
        "model_identity_feature_used": False,
    }
    (output_dir / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"Phase 14D smoke passed for {len(rows)} rows and {len(METHODS)} methods")


def write_readme(path):
    path.write_text(
        "# Phase 14D backend-free TP/phase interaction analysis\n\n"
        "This analysis compares the Phase 14C additive TP/phase baseline with "
        "ring-equivalent PatternDemand, TP-conditioned slopes, phase-conditioned "
        "slopes, and shared-plus-interaction TP×phase slopes. Actual backend, "
        "kernel names, and model identity are never predictive features. Backend "
        "signatures are retained only for post-hoc residual diagnostics.\n"
    )


def main():
    args = parse_args()
    rows = baseline.load_rows(args.input_csv)
    observed = {row["model"] for row in rows}
    preferred_order = ("qwen3-8b", "qwen3-30b-a3b", "deepseek-v2-lite")
    models = tuple(model for model in preferred_order if model in observed)
    models += tuple(sorted(observed - set(models)))
    configure_baseline(models)
    folds = baseline.build_balanced_folds(rows)
    for method in METHODS:
        for row in rows:
            feature_vector(row, method)

    if args.smoke:
        run_smoke(rows, args.output_dir)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    workload_predictions, workload_selection = baseline.run_workload_cv(rows)
    model_predictions, model_selection = baseline.run_leave_one_model_out(rows)
    metrics = baseline.build_metrics(workload_predictions, model_predictions)
    add_profile_metrics(metrics, workload_predictions)
    posthoc = baseline.build_posthoc_backend_diagnostics(
        rows, workload_predictions
    )

    baseline.write_csv(args.output_dir / "fold_assignments.csv", folds)
    baseline.write_csv(
        args.output_dir / "predictions.csv",
        workload_predictions + model_predictions,
    )
    baseline.write_csv(
        args.output_dir / "alpha_selection.csv",
        workload_selection + model_selection,
    )
    baseline.write_csv(args.output_dir / "metrics.csv", metrics)
    baseline.write_csv(
        args.output_dir / "posthoc_backend_diagnostics.csv", posthoc
    )
    make_figure(
        metrics,
        models,
        args.output_dir / "tp_phase_interaction_analysis.png",
    )

    comparison = method_comparison(metrics)
    best_method = min(METHODS, key=lambda method: comparison[method]["mape"])
    primary = {
        scope: compact_metric(
            metric(metrics, "workload_cv", scope, PRIMARY_CANDIDATE)
        )
        for scope in ("all", "prefill", "decode", "tp2", "tp4", "tp8")
    }
    held_out = {
        model: compact_metric(
            metric(metrics, "leave_one_model_out", model, PRIMARY_CANDIDATE)
        )
        for model in models
    }
    profiles = {
        scope.split(":", 1)[1]: compact_metric(
            metric(metrics, "workload_cv", scope, PRIMARY_CANDIDATE)
        )
        for scope in sorted(
            row["scope"]
            for row in metrics
            if row["evaluation"] == "workload_cv"
            and row["method"] == PRIMARY_CANDIDATE
            and row["scope"].startswith("profile:")
        )
    }
    additive = comparison[BASELINE_CANDIDATE]
    primary_all = primary["all"]
    summary = {
        "schema_version": "phase14d-tp-phase-interaction-analysis-v1",
        "dataset": {
            "source": str(args.input_csv),
            "aggregated_configurations": len(rows),
            "workload_groups": len({row["workload_id"] for row in rows}),
            "models": list(models),
            "tensor_parallel_sizes": list(baseline.TPS),
            "phases": list(baseline.PHASES),
        },
        "leakage_audit": {
            "actual_backend_feature_used": False,
            "backend_signature_use": "post-hoc residual diagnostics only",
            "kernel_name_feature_used": False,
            "model_identity_feature_used": False,
            "phase2_cost_curve_feature_used": False,
            "same_workload_tp_variants_kept_together": True,
            "ridge_alpha_selection": "nested within each outer training split",
            "repeat_leakage": False,
        },
        "feature_counts": {
            method: len(feature_names(method)) for method in METHODS
        },
        "feature_specification": {
            method: feature_names(method) for method in METHODS
        },
        "ridge_alphas": list(baseline.RIDGE_ALPHAS),
        "method_comparison": comparison,
        "best_workload_cv_mape_method": best_method,
        "primary_candidate": PRIMARY_CANDIDATE,
        "primary_workload_cv": primary,
        "primary_by_profile": profiles,
        "primary_leave_one_model_out": held_out,
        "relative_to_additive": {
            "mape_fractional_change": (
                primary_all["mape"] - additive["mape"]
            ) / additive["mape"],
            "p95_fractional_change": (
                primary_all["p95_ape"] - additive["p95_ape"]
            ) / additive["p95_ape"],
        },
        "decision_gates": {
            "overall_mape_below_10pct": primary_all["mape"] < 0.10,
            "overall_p95_below_25pct": primary_all["p95_ape"] < 0.25,
            "prefill_mape_below_15pct": primary["prefill"]["mape"] < 0.15,
            "decode_mape_below_10pct": primary["decode"]["mape"] < 0.10,
            "all_held_out_models_mape_below_15pct": all(
                row["mape"] < 0.15 for row in held_out.values()
            ),
            "primary_improves_additive_mape": (
                primary_all["mape"] < additive["mape"]
            ),
            "primary_improves_additive_p95": (
                primary_all["p95_ape"] < additive["p95_ape"]
            ),
        },
        "interpretation": {
            "ring_equivalent_definition": (
                "Each logical AllReduce call becomes 2*(TP-1) ring rounds at "
                "input_payload_bytes/TP bytes per round."
            ),
            "conditioned_slopes": (
                "Shared PatternDemand coefficients plus ridge-regularized TP, "
                "phase, and TP×phase interactions; no independent per-cell fit."
            ),
            "convergence_rule": (
                "Primary candidate must pass overall MAPE<10%, P95<25%, "
                "Prefill MAPE<15%, Decode MAPE<10%, and every held-out-model "
                "MAPE<15%."
            ),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    write_readme(args.output_dir / "README.md")
    print(
        f"Phase 14D analyzed {len(rows)} configurations; "
        f"primary MAPE={primary_all['mape']:.6f}, "
        f"P95={primary_all['p95_ape']:.6f}; best={best_method}"
    )


if __name__ == "__main__":
    main()
