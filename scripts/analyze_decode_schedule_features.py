#!/usr/bin/env python3
"""Phase 14E: test pre-run Decode schedule features without backend inputs."""

import argparse
import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import analyze_tp_phase_interaction_models as phase14d


baseline = phase14d.baseline
METHODS = (
    "additive_tp_phase",
    "tp_conditioned_pattern",
    "additive_plus_schedule",
    "tp_conditioned_plus_schedule",
    "tp_conditioned_plus_schedule_tp",
)
BASELINE_CANDIDATE = "tp_conditioned_pattern"
PRIMARY_CANDIDATE = "tp_conditioned_plus_schedule"
SCHEDULE_FEATURE_NAMES = (
    "decode_log_initial_batch",
    "decode_log_steps",
    "decode_log_total_tokens",
    "decode_mean_active_fraction",
    "decode_active_std_fraction",
    "decode_final_active_fraction",
    "decode_distinct_active_levels_fraction",
    "decode_change_rate",
    "decode_longest_plateau_fraction",
    "decode_completion_p25_fraction",
    "decode_completion_p50_fraction",
    "decode_completion_p75_fraction",
    "decode_completion_length_cv",
    "decode_front_to_back_log_ratio",
)
FORBIDDEN_PREDICTIVE_SUBSTRINGS = (
    "backend",
    "kernel_name",
    "model_identity",
    "target",
)


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
        default=repo / "experiment-results/phase14e/decode_schedule_analysis",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="validate feature reconstruction and fit one fold without nested selection",
    )
    return parser.parse_args()


def output_lengths(row):
    values = json.loads(row["output_lens_json"])
    if not isinstance(values, list) or not values:
        raise ValueError(f"{row['workload_id']}: output_lens_json must be non-empty")
    lengths = [int(value) for value in values]
    if any(value <= 0 for value in lengths):
        raise ValueError(f"{row['workload_id']}: output lengths must be positive")
    if len(lengths) != int(row["batch_size"]):
        raise ValueError(f"{row['workload_id']}: output-length/batch mismatch")
    return lengths


def active_batch_sequence(row):
    if row["phase"] != "decode":
        return []
    lengths = output_lengths(row)
    return [
        sum(length > step for length in lengths)
        for step in range(max(lengths))
    ]


def longest_run(values):
    return max(len(list(group)) for _, group in itertools.groupby(values))


def schedule_features(row):
    if row["phase"] != "decode":
        return np.zeros(len(SCHEDULE_FEATURE_NAMES), dtype=np.float64)
    lengths = np.asarray(output_lengths(row), dtype=np.float64)
    active = np.asarray(active_batch_sequence(row), dtype=np.float64)
    batch = float(len(lengths))
    steps = float(len(active))
    quarter = max(1, len(active) // 4)
    front = float(np.mean(active[:quarter]))
    back = float(np.mean(active[-quarter:]))
    values = np.asarray(
        [
            math.log1p(batch),
            math.log1p(steps),
            math.log1p(float(np.sum(lengths))),
            float(np.mean(active)) / batch,
            float(np.std(active)) / batch,
            float(active[-1]) / batch,
            float(len(set(int(value) for value in active))) / batch,
            float(np.count_nonzero(np.diff(active))) / max(1.0, steps - 1.0),
            float(longest_run(active)) / steps,
            float(np.percentile(lengths, 25)) / steps,
            float(np.percentile(lengths, 50)) / steps,
            float(np.percentile(lengths, 75)) / steps,
            float(np.std(lengths) / np.mean(lengths)),
            math.log((front + 1e-12) / (back + 1e-12)),
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{row['workload_id']}: non-finite schedule feature")
    return values


def feature_names(method):
    if method == "additive_tp_phase":
        names = phase14d.feature_names("additive_tp_phase")
    elif method == "tp_conditioned_pattern":
        names = phase14d.feature_names("tp_conditioned_pattern")
    elif method == "additive_plus_schedule":
        names = phase14d.feature_names("additive_tp_phase") + list(
            SCHEDULE_FEATURE_NAMES
        )
    else:
        names = phase14d.feature_names("tp_conditioned_pattern") + list(
            SCHEDULE_FEATURE_NAMES
        )
        if method == "tp_conditioned_plus_schedule_tp":
            names.extend(f"tp4_x_{name}" for name in SCHEDULE_FEATURE_NAMES)
            names.extend(f"tp8_x_{name}" for name in SCHEDULE_FEATURE_NAMES)
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    if len(names) != len(set(names)):
        raise ValueError(f"{method}: duplicate feature names")
    for name in names:
        if any(token in name for token in FORBIDDEN_PREDICTIVE_SUBSTRINGS):
            raise ValueError(f"forbidden predictive feature name: {name}")
    return names


def feature_vector(row, method):
    if method == "additive_tp_phase":
        values = phase14d.feature_vector(row, "additive_tp_phase")
    elif method == "tp_conditioned_pattern":
        values = phase14d.feature_vector(row, "tp_conditioned_pattern")
    elif method == "additive_plus_schedule":
        values = np.concatenate(
            [
                phase14d.feature_vector(row, "additive_tp_phase"),
                schedule_features(row),
            ]
        )
    else:
        schedule = schedule_features(row)
        values = np.concatenate(
            [phase14d.feature_vector(row, "tp_conditioned_pattern"), schedule]
        )
        if method == "tp_conditioned_plus_schedule_tp":
            values = np.concatenate(
                [
                    values,
                    float(row["tp"] == 4) * schedule,
                    float(row["tp"] == 8) * schedule,
                ]
            )
    result = np.asarray(values, dtype=np.float64)
    if len(result) != len(feature_names(method)):
        raise ValueError(f"{method}: feature name/vector size mismatch")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{method}: non-finite feature value")
    return result


def configure_baseline(models):
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


def schedule_rows(rows):
    representatives = {}
    for row in rows:
        representatives.setdefault(row["workload_id"], row)
    output = []
    for identifier, row in sorted(representatives.items()):
        sequence = active_batch_sequence(row)
        values = schedule_features(row)
        item = {
            "workload_id": identifier,
            "model_metadata_only": row["model"],
            "phase": row["phase"],
            "case_label": row["case_label"],
            "fold": row["fold"],
            "output_lens_json": row["output_lens_json"],
            "active_batch_sequence_json": json.dumps(sequence, separators=(",", ":")),
            "features_available_before_profiled_execution": True,
            "actual_backend_feature_used": False,
        }
        item.update(
            {
                name: float(value)
                for name, value in zip(SCHEDULE_FEATURE_NAMES, values)
            }
        )
        output.append(item)
    return output


def feasibility_audit(rows):
    decode = [row for row in rows if row["phase"] == "decode"]
    groups = {}
    for row in rows:
        groups.setdefault(row["workload_id"], []).append(row)
    if any(
        len({row["output_lens_json"] for row in group}) != 1
        for group in groups.values()
    ):
        raise ValueError("output_lens_json changed across TP")
    decode_groups = {
        row["workload_id"]: row
        for row in decode
        if row["tp"] == 2
    }
    sequence_keys = {
        tuple(active_batch_sequence(row)) for row in decode_groups.values()
    }
    profile_to_sequence = {}
    for row in decode_groups.values():
        profile_to_sequence.setdefault(row["case_label"], set()).add(
            tuple(active_batch_sequence(row))
        )
    if any(len(values) != 1 for values in profile_to_sequence.values()):
        raise ValueError("same Decode profile has inconsistent active-batch sequence")
    fold_profiles = {}
    for fold in range(6):
        fold_profiles[str(fold)] = sorted(
            {
                row["case_label"]
                for row in decode
                if row["fold"] == fold
            }
        )
    if any(len(values) != 1 for values in fold_profiles.values()):
        raise ValueError(f"Decode profile holdout is not aligned: {fold_profiles}")
    schedule = np.vstack([schedule_features(row) for row in rows])
    base = np.vstack(
        [phase14d.feature_vector(row, "tp_conditioned_pattern") for row in rows]
    )
    combined = np.column_stack([base, schedule])
    return {
        "all_rows_reconstructable": True,
        "decode_tp_expanded_rows": len(decode),
        "decode_workload_groups": len(decode_groups),
        "decode_profiles": sorted(profile_to_sequence),
        "unique_decode_active_batch_sequences": len(sequence_keys),
        "output_lens_invariant_across_tp": True,
        "same_profile_sequence_invariant_across_models": True,
        "decode_profile_held_out_across_all_models_per_fold": True,
        "decode_profile_by_fold": fold_profiles,
        "tp_conditioned_matrix_rank": int(np.linalg.matrix_rank(base)),
        "schedule_matrix_rank": int(np.linalg.matrix_rank(schedule)),
        "combined_matrix_rank": int(np.linalg.matrix_rank(combined)),
        "availability_note": (
            "Features use configured per-request output lengths. They are pre-run "
            "for this ignore-EOS benchmark, but production use would require "
            "requested or predicted output lengths rather than observed completion."
        ),
    }


def make_figure(metrics, models, path):
    fig, axes = plt.subplots(2, 2, figsize=(17, 10.5))
    labels = (
        "Additive",
        "TP slopes",
        "+schedule",
        "TP slopes\n+schedule",
        "+schedule×TP",
    )
    x = np.arange(len(METHODS))
    all_rows = [metric(metrics, "workload_cv", "all", method) for method in METHODS]
    axes[0, 0].bar(x - 0.18, [100 * row["mape"] for row in all_rows], 0.36, label="MAPE")
    axes[0, 0].bar(x + 0.18, [100 * row["p95_ape"] for row in all_rows], 0.36, label="P95 APE")
    axes[0, 0].set_xticks(x, labels, rotation=12)
    axes[0, 0].set_ylabel("Error (%)")
    axes[0, 0].set_title("6-fold workload/profile CV")
    axes[0, 0].legend()

    phase_rows = [
        metric(metrics, "workload_cv", scope, PRIMARY_CANDIDATE)
        for scope in ("all", "prefill", "decode")
    ]
    axes[0, 1].bar(np.arange(3) - 0.18, [100 * row["mape"] for row in phase_rows], 0.36, label="MAPE")
    axes[0, 1].bar(np.arange(3) + 0.18, [100 * row["p95_ape"] for row in phase_rows], 0.36, label="P95 APE")
    axes[0, 1].set_xticks(np.arange(3), ("All", "Prefill", "Decode"))
    axes[0, 1].set_ylabel("Error (%)")
    axes[0, 1].set_title("Predeclared schedule candidate")
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
    axes[1, 0].set_title("Schedule candidate by held-out shape")
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


def run_smoke(rows, audit, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    train = [row for row in rows if row["fold"] != 0]
    test = [row for row in rows if row["fold"] == 0]
    checks = {}
    for method in METHODS:
        matrix = np.vstack([feature_vector(row, method) for row in rows])
        predicted = baseline.ridge_predict(
            baseline.fit_ridge(train, method, 1.0), test, method
        )
        actual = np.asarray([row["target_post_us"] for row in test])
        checks[method] = {
            "features": matrix.shape[1],
            "rows": matrix.shape[0],
            "finite": bool(np.all(np.isfinite(matrix))),
            "fold0_predictions": len(predicted),
            "fold0_mape": float(np.mean(np.abs(predicted - actual) / actual)),
        }
    summary = {
        "schema_version": "phase14e-decode-schedule-smoke-v1",
        "status": "passed",
        "input_rows": len(rows),
        "workload_groups": len({row["workload_id"] for row in rows}),
        "feasibility_audit": audit,
        "methods": checks,
        "actual_backend_feature_used": False,
        "model_identity_feature_used": False,
        "target_derived_feature_used": False,
    }
    (output_dir / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"Phase 14E smoke passed for {len(rows)} rows and {len(METHODS)} methods")


def write_readme(path):
    path.write_text(
        "# Phase 14E Decode schedule analysis\n\n"
        "This analysis reconstructs active-batch sequences from configured "
        "per-request output lengths and tests compact schedule features on top "
        "of the backend-free Phase 14D models. Actual backend, kernel names, "
        "model identity, and measured target values are never predictive features.\n"
    )


def main():
    args = parse_args()
    rows = baseline.load_rows(args.input_csv)
    for row in rows:
        row["batch_size"] = int(row["batch_size"])
    observed = {row["model"] for row in rows}
    preferred_order = ("qwen3-8b", "qwen3-30b-a3b", "deepseek-v2-lite")
    models = tuple(model for model in preferred_order if model in observed)
    models += tuple(sorted(observed - set(models)))
    configure_baseline(models)
    folds = baseline.build_balanced_folds(rows)
    audit = feasibility_audit(rows)
    for method in METHODS:
        for row in rows:
            feature_vector(row, method)

    if args.smoke:
        run_smoke(rows, audit, args.output_dir)
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
    baseline.write_csv(args.output_dir / "schedule_features.csv", schedule_rows(rows))
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
    make_figure(metrics, models, args.output_dir / "decode_schedule_analysis.png")

    comparison = {
        method: compact_metric(metric(metrics, "workload_cv", "all", method))
        for method in METHODS
    }
    best_method = min(METHODS, key=lambda method: comparison[method]["mape"])

    def scoped(candidate):
        return {
            scope: compact_metric(metric(metrics, "workload_cv", scope, candidate))
            for scope in ("all", "prefill", "decode", "tp2", "tp4", "tp8")
        }

    def held(candidate):
        return {
            model: compact_metric(
                metric(metrics, "leave_one_model_out", model, candidate)
            )
            for model in models
        }

    def profiles(candidate):
        return {
            scope.split(":", 1)[1]: compact_metric(
                metric(metrics, "workload_cv", scope, candidate)
            )
            for scope in sorted(
                row["scope"]
                for row in metrics
                if row["evaluation"] == "workload_cv"
                and row["method"] == candidate
                and row["scope"].startswith("profile:")
            )
        }

    primary = scoped(PRIMARY_CANDIDATE)
    primary_held = held(PRIMARY_CANDIDATE)
    selected = scoped(best_method)
    selected_held = held(best_method)
    baseline_all = comparison[BASELINE_CANDIDATE]
    primary_all = primary["all"]
    summary = {
        "schema_version": "phase14e-decode-schedule-analysis-v1",
        "dataset": {
            "source": str(args.input_csv),
            "aggregated_configurations": len(rows),
            "workload_groups": len({row["workload_id"] for row in rows}),
            "models": list(models),
            "tensor_parallel_sizes": list(baseline.TPS),
            "phases": list(baseline.PHASES),
        },
        "feasibility_audit": audit,
        "leakage_audit": {
            "actual_backend_feature_used": False,
            "backend_signature_use": "post-hoc residual diagnostics only",
            "kernel_name_feature_used": False,
            "model_identity_feature_used": False,
            "target_derived_feature_used": False,
            "phase2_cost_curve_feature_used": False,
            "schedule_source": "configured output_lens_json",
            "same_workload_tp_variants_kept_together": True,
            "ridge_alpha_selection": "nested within each outer training split",
            "repeat_leakage": False,
        },
        "feature_counts": {method: len(feature_names(method)) for method in METHODS},
        "schedule_feature_names": list(SCHEDULE_FEATURE_NAMES),
        "ridge_alphas": list(baseline.RIDGE_ALPHAS),
        "method_comparison": comparison,
        "predeclared_candidate": PRIMARY_CANDIDATE,
        "predeclared_workload_cv": primary,
        "predeclared_by_profile": profiles(PRIMARY_CANDIDATE),
        "predeclared_leave_one_model_out": primary_held,
        "best_workload_cv_mape_method": best_method,
        "selected_candidate_note": (
            "Descriptive ranking on common outer-CV predictions; candidate-family "
            "selection itself is not nested and is not an unbiased production score."
        ),
        "selected_workload_cv": selected,
        "selected_by_profile": profiles(best_method),
        "selected_leave_one_model_out": selected_held,
        "relative_to_phase14d_tp_conditioned": {
            "predeclared_mape_fractional_change": (
                primary_all["mape"] - baseline_all["mape"]
            ) / baseline_all["mape"],
            "predeclared_p95_fractional_change": (
                primary_all["p95_ape"] - baseline_all["p95_ape"]
            ) / baseline_all["p95_ape"],
        },
        "predeclared_decision_gates": {
            "overall_mape_below_10pct": primary_all["mape"] < 0.10,
            "overall_p95_below_25pct": primary_all["p95_ape"] < 0.25,
            "prefill_mape_below_15pct": primary["prefill"]["mape"] < 0.15,
            "decode_mape_below_10pct": primary["decode"]["mape"] < 0.10,
            "all_held_out_models_mape_below_15pct": all(
                row["mape"] < 0.15 for row in primary_held.values()
            ),
            "improves_phase14d_tp_conditioned_mape": (
                primary_all["mape"] < baseline_all["mape"]
            ),
            "improves_phase14d_tp_conditioned_p95": (
                primary_all["p95_ape"] < baseline_all["p95_ape"]
            ),
        },
        "interpretation": {
            "schedule_definition": (
                "Compact statistics of the monotonically draining active-batch "
                "sequence reconstructed from configured per-request output lengths."
            ),
            "production_availability_limit": (
                "Exact output lengths are configured in this ignore-EOS benchmark; "
                "a production predictor would need requested limits or a separate "
                "length forecast available before execution."
            ),
            "convergence_rule": (
                "Predeclared candidate must pass overall MAPE<10%, P95<25%, "
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
        f"Phase 14E analyzed {len(rows)} configurations; "
        f"predeclared MAPE={primary_all['mape']:.6f}, "
        f"P95={primary_all['p95_ape']:.6f}; "
        f"best={best_method}, best MAPE={selected['all']['mape']:.6f}"
    )


if __name__ == "__main__":
    main()
