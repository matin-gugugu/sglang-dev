#!/usr/bin/env python3
"""Analyze Phase 14 representative TP2/4/8 all-rank timing labels."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluate_pattern_cost_ablation import BackendAwareCostCurve


MODELS = ("qwen3-8b", "qwen3-30b-a3b")
TPS = (2, 4, 8)
MIXED_PROFILES = ("balanced", "staircase", "bimodal")
CHUNK_INPUTS = {
    "c1024": {1023, 1024, 1025},
    "c4096": {4095, 4096, 4097},
}
CHUNK_BATCHES = {1, 4}
TARGET_FIELD = "post_rendezvous_completion_kernel_time_us"
METHODS = (
    "total_bytes_tp2_fit",
    "continuous_tp2_calibrated",
    "continuous_per_tp_calibrated",
)


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tp2-qwen-root",
        type=Path,
        default=repo_root
        / "experiment-results/phase11/multiscale_timing_ground_truth/qwen3-8b",
    )
    parser.add_argument(
        "--tp2-qwen30-root",
        type=Path,
        default=repo_root
        / "experiment-results/phase13/multiscale_timing_ground_truth/qwen3-30b-a3b",
    )
    parser.add_argument(
        "--phase14-root",
        type=Path,
        default=repo_root
        / "experiment-results/phase14/tp_group_size_timing_ground_truth",
    )
    parser.add_argument(
        "--custom-curve",
        type=Path,
        default=repo_root
        / "experiment-results/phase2/summary_l1_custom_kernel_curve"
        / "custom_kernel_curve_summary.csv",
    )
    parser.add_argument(
        "--nccl-curve",
        type=Path,
        default=repo_root
        / "experiment-results/phase2/summary_l1_curve"
        / "collective_curve_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "experiment-results/phase14/tp_group_size_timing_analysis",
    )
    return parser.parse_args()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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


def selected_record(record, mode, case_label):
    workload = record["workload"]
    if mode == "mixed_same_coarse":
        return case_label in MIXED_PROFILES
    return (
        case_label in CHUNK_INPUTS
        and int(workload["input_len"]) in CHUNK_INPUTS[case_label]
        and int(workload["batch_size"]) in CHUNK_BATCHES
    )


def workload_key(model, tp, mode, case_label, record):
    workload = record["workload"]
    return (
        model,
        tp,
        mode,
        case_label,
        int(workload["batch_size"]),
        int(workload["input_len"]),
        int(workload["output_len"]),
        tuple(int(value) for value in workload["output_lens_per_request"]),
        int(workload["prefill_chunk_size"]),
    )


def workload_id(key):
    model, _, mode, case_label, batch, input_len, output_len, output_lens, chunk = key
    if mode == "mixed_same_coarse":
        return f"{model}-mixed-{case_label}"
    return f"{model}-chunk-c{chunk}-b{batch}-l{input_len}-m{output_len}"


def op_histogram(pattern):
    entries = pattern.get("calls_by_raw_op_and_input_payload_bytes")
    if entries is None:
        return {
            ("all_reduce", int(payload)): int(count)
            for payload, count in pattern["calls_by_input_payload_bytes"].items()
        }
    result = {
        (entry["raw_op"], int(entry["input_payload_bytes"])): int(entry["count"])
        for entry in entries
    }
    if len(result) != len(entries):
        raise ValueError("duplicate raw-op/payload histogram entries")
    if any(entry["collective_family"] != "all_reduce" for entry in entries):
        raise ValueError("Phase 14 only supports the AllReduce collective family")
    return result


def payload_histogram(pattern, op_payload):
    result = defaultdict(int)
    for (_, payload), count in op_payload.items():
        result[payload] += count
    serialized = {
        int(payload): int(count)
        for payload, count in pattern["calls_by_input_payload_bytes"].items()
    }
    if dict(sorted(result.items())) != dict(sorted(serialized.items())):
        raise ValueError("raw-op histogram marginal disagrees with payload histogram")
    return dict(sorted(result.items()))


def load_rows(args, curve):
    roots = {
        (2, "qwen3-8b"): args.tp2_qwen_root,
        (2, "qwen3-30b-a3b"): args.tp2_qwen30_root,
    }
    for tp in (4, 8):
        for model in MODELS:
            roots[(tp, model)] = args.phase14_root / f"tp{tp}" / model

    grouped = defaultdict(list)
    source_files = []
    selected_raw_rows = 0
    for (tp, model), root in roots.items():
        paths = sorted(root.glob("*/*/r*/all_rank_ground_truth.jsonl"))
        if not paths:
            raise ValueError(f"no all-rank labels under {root}")
        for path in paths:
            mode, case_label, repeat_dir, _ = path.relative_to(root).parts
            if mode == "mixed_same_coarse" and case_label not in MIXED_PROFILES:
                continue
            if mode == "chunked_prefill" and case_label not in CHUNK_INPUTS:
                continue
            source_files.append(path)
            for record in read_jsonl(path):
                if not selected_record(record, mode, case_label):
                    continue
                selected_raw_rows += 1
                record["_source"] = str(path)
                record["_repeat_dir"] = repeat_dir
                grouped[workload_key(model, tp, mode, case_label, record)].append(record)

    rows = []
    for key, repeats in sorted(grouped.items(), key=lambda item: str(item[0])):
        if len(repeats) != 3:
            raise ValueError(f"{key}: expected three repeats, got {len(repeats)}")
        if sorted(int(record["repeat_id"]) for record in repeats) != [0, 1, 2]:
            raise ValueError(f"{key}: repeat ids are not 0,1,2")
        model, tp, mode, case_label, batch, input_len, output_len, output_lens, chunk = key
        patterns = [record["full_phase_pattern_demand"] for record in repeats]
        if any(pattern != patterns[0] for pattern in patterns[1:]):
            raise ValueError(f"{key}: PatternDemand changed across repeats")
        for record in repeats:
            if int(record["full_phase_pattern_demand"]["group_size"]) != tp:
                raise ValueError(f"{record['_source']}: group size mismatch")
            alignment = record["alignment"]
            required = (
                "exact_count_on_every_rank",
                "identical_backend_sequence",
                "identical_profiled_pattern_demand_on_every_rank",
                "identical_full_phase_pattern_demand_on_every_rank",
            )
            if not all(alignment[field] for field in required):
                raise ValueError(f"{record['_source']}: all-rank alignment failed")
            estimate = record["all_rank_ground_truth"]["full_phase_estimate"]
            if not math.isclose(
                float(estimate["profiled_to_full_call_scale"]), 1.0, abs_tol=1e-12
            ):
                raise ValueError(f"{record['_source']}: full-phase scale is not 1")

        pattern = patterns[0]
        by_op_payload = op_histogram(pattern)
        by_payload = payload_histogram(pattern, by_op_payload)
        calls = sum(by_payload.values())
        logical_bytes = sum(payload * count for payload, count in by_payload.items())
        if calls != int(pattern["all_reduce_calls"]):
            raise ValueError(f"{key}: call total mismatch")
        if logical_bytes != int(pattern["input_payload_bytes"]):
            raise ValueError(f"{key}: logical byte total mismatch")

        estimates = [
            record["all_rank_ground_truth"]["full_phase_estimate"]
            for record in repeats
        ]
        targets = [float(estimate[TARGET_FIELD]) for estimate in estimates]
        intrinsic = [
            float(estimate["skew_free_intrinsic_kernel_time_us"])
            for estimate in estimates
        ]
        sync = [
            float(estimate["synchronization_inclusive_max_duration_sum_us"])
            for estimate in estimates
        ]
        backend_signatures = sorted(
            {
                record["all_rank_ground_truth"]["backend_sequence_signature"]
                for record in repeats
            }
        )
        if len(backend_signatures) != 1:
            raise ValueError(f"{key}: backend changed across repeats")

        production_payloads = [payload for payload, _ in curve.production_points(tp)]
        curve_min, curve_max = min(production_payloads), max(production_payloads)
        extrapolated_calls = sum(
            count
            for payload, count in by_payload.items()
            if payload < curve_min or payload > curve_max
        )
        continuous_raw = sum(
            count * curve.lookup(tp, payload) for payload, count in by_payload.items()
        )
        target = float(statistics.median(targets))
        rows.append(
            {
                "workload_id": workload_id(key),
                "model": model,
                "tp": tp,
                "mode": mode,
                "case_label": case_label,
                "phase": "decode" if mode == "mixed_same_coarse" else "prefill",
                "batch_size": batch,
                "input_len": input_len,
                "output_len": output_len,
                "output_lens_json": json.dumps(output_lens, separators=(",", ":")),
                "prefill_chunk_size": chunk,
                "repeat_count": 3,
                "calls": calls,
                "logical_payload_bytes": logical_bytes,
                "ring_equivalent_bytes": float(pattern["ring_equivalent"]["bytes"]),
                "ring_equivalent_rounds": int(pattern["ring_equivalent"]["rounds"]),
                "payload_supports": len(by_payload),
                "op_payload_supports": len(by_op_payload),
                "calls_by_payload_json": json.dumps(
                    {str(payload): count for payload, count in by_payload.items()},
                    separators=(",", ":"),
                ),
                "calls_by_op_payload_json": json.dumps(
                    {
                        f"{op}:{payload}": count
                        for (op, payload), count in sorted(by_op_payload.items())
                    },
                    separators=(",", ":"),
                ),
                "raw_ops_json": json.dumps(
                    sorted({op for op, _ in by_op_payload}), separators=(",", ":")
                ),
                "backend_signature": backend_signatures[0],
                "target_post_us": target,
                "post_repeat_iqr_fraction": (
                    percentile(targets, 75) - percentile(targets, 25)
                )
                / target,
                "intrinsic_us": float(statistics.median(intrinsic)),
                "sync_inclusive_us": float(statistics.median(sync)),
                "continuous_raw_us": continuous_raw,
                "curve_min_payload_bytes": curve_min,
                "curve_max_payload_bytes": curve_max,
                "curve_extrapolated_calls": extrapolated_calls,
                "curve_extrapolated_call_fraction": extrapolated_calls / calls,
                "_by_payload": by_payload,
            }
        )

    expected_aggregated = len(MODELS) * len(TPS) * 15
    expected_raw = expected_aggregated * 3
    if len(rows) != expected_aggregated or selected_raw_rows != expected_raw:
        raise ValueError(
            f"expected {expected_raw} raw/{expected_aggregated} aggregate rows, "
            f"got {selected_raw_rows}/{len(rows)}"
        )
    return rows, source_files, selected_raw_rows


def fit_total_bytes_tp2(rows):
    models = {}
    for phase in ("prefill", "decode"):
        subset = [row for row in rows if row["tp"] == 2 and row["phase"] == phase]
        design = np.asarray(
            [[1.0, math.log1p(row["logical_payload_bytes"])] for row in subset],
            dtype=np.float64,
        )
        target = np.asarray(
            [math.log(row["target_post_us"]) for row in subset], dtype=np.float64
        )
        models[phase] = np.linalg.lstsq(design, target, rcond=None)[0]
    return models


def total_bytes_predict(row, models):
    intercept, slope = models[row["phase"]]
    return math.exp(intercept + slope * math.log1p(row["logical_payload_bytes"]))


def fit_calibration(rows, key_fields):
    ratios = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        ratios[key].append(math.log(row["target_post_us"] / row["continuous_raw_us"]))
    return {key: math.exp(statistics.median(values)) for key, values in ratios.items()}


def add_predictions(rows):
    total_models = fit_total_bytes_tp2(rows)
    tp2_calibration = fit_calibration(
        [row for row in rows if row["tp"] == 2], ("phase",)
    )
    per_tp_calibration = fit_calibration(rows, ("tp", "phase"))
    for row in rows:
        row["pred_total_bytes_tp2_fit_us"] = total_bytes_predict(row, total_models)
        row["pred_continuous_tp2_calibrated_us"] = (
            row["continuous_raw_us"] * tp2_calibration[(row["phase"],)]
        )
        row["pred_continuous_per_tp_calibrated_us"] = (
            row["continuous_raw_us"]
            * per_tp_calibration[(row["tp"], row["phase"])]
        )
    return {
        "total_bytes_tp2_fit": {
            phase: coefficients.tolist()
            for phase, coefficients in total_models.items()
        },
        "continuous_tp2_calibration": {
            key[0]: value for key, value in tp2_calibration.items()
        },
        "continuous_per_tp_calibration": {
            f"tp{key[0]}_{key[1]}": value
            for key, value in sorted(per_tp_calibration.items())
        },
    }


def metric_row(scope, method, rows):
    actual = np.asarray([row["target_post_us"] for row in rows], dtype=np.float64)
    predicted = np.asarray(
        [row[f"pred_{method}_us"] for row in rows], dtype=np.float64
    )
    ape = np.abs(predicted - actual) / actual
    residual = np.sum((predicted - actual) ** 2)
    total = np.sum((actual - np.mean(actual)) ** 2)
    return {
        "scope": scope,
        "method": method,
        "samples": len(rows),
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
        "p95_ape": percentile(ape, 95),
        "mae_us": float(np.mean(np.abs(predicted - actual))),
        "rmse_us": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "r2": float(1.0 - residual / total) if total > 0 else float("nan"),
    }


def build_metrics(rows):
    scopes = {
        "tp4_8_zero_shot": [row for row in rows if row["tp"] in (4, 8)],
    }
    for tp in TPS:
        scopes[f"tp{tp}_all"] = [row for row in rows if row["tp"] == tp]
        for phase in ("prefill", "decode"):
            scopes[f"tp{tp}_{phase}"] = [
                row for row in rows if row["tp"] == tp and row["phase"] == phase
            ]
        for model in MODELS:
            scopes[f"tp{tp}_{model}"] = [
                row for row in rows if row["tp"] == tp and row["model"] == model
            ]
    return [
        metric_row(scope, method, subset)
        for scope, subset in scopes.items()
        for method in METHODS
    ]


def build_scaling(rows):
    groups = defaultdict(dict)
    for row in rows:
        groups[row["workload_id"]][row["tp"]] = row
    output = []
    for identifier, by_tp in sorted(groups.items()):
        if set(by_tp) != set(TPS):
            raise ValueError(f"{identifier}: incomplete TP scaling group")
        base, tp4, tp8 = by_tp[2], by_tp[4], by_tp[8]
        calls_invariant = len({row["calls"] for row in by_tp.values()}) == 1
        bytes_invariant = (
            len({row["logical_payload_bytes"] for row in by_tp.values()}) == 1
        )
        histogram_invariant = (
            len({row["calls_by_op_payload_json"] for row in by_tp.values()}) == 1
        )
        output.append(
            {
                "workload_id": identifier,
                "model": base["model"],
                "mode": base["mode"],
                "case_label": base["case_label"],
                "phase": base["phase"],
                "batch_size": base["batch_size"],
                "input_len": base["input_len"],
                "calls_invariant": calls_invariant,
                "logical_bytes_invariant": bytes_invariant,
                "op_payload_histogram_invariant": histogram_invariant,
                "tp2_rounds": base["ring_equivalent_rounds"],
                "tp4_rounds": tp4["ring_equivalent_rounds"],
                "tp8_rounds": tp8["ring_equivalent_rounds"],
                "tp2_target_post_us": base["target_post_us"],
                "tp4_target_post_us": tp4["target_post_us"],
                "tp8_target_post_us": tp8["target_post_us"],
                "tp4_over_tp2": tp4["target_post_us"] / base["target_post_us"],
                "tp8_over_tp2": tp8["target_post_us"] / base["target_post_us"],
                "tp8_over_tp4": tp8["target_post_us"] / tp4["target_post_us"],
                "tp2_backend_signature": base["backend_signature"],
                "tp4_backend_signature": tp4["backend_signature"],
                "tp8_backend_signature": tp8["backend_signature"],
                "backend_changed": len(
                    {row["backend_signature"] for row in by_tp.values()}
                )
                > 1,
            }
        )
    if len(output) != len(MODELS) * 15:
        raise ValueError(f"expected 30 TP scaling groups, got {len(output)}")
    return output


def public_row(row):
    return {key: value for key, value in row.items() if not key.startswith("_")}


def make_figure(rows, metrics, scaling, path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    colors = {"qwen3-8b": "#4C78A8", "qwen3-30b-a3b": "#54A24B"}
    for model in MODELS:
        values = [
            statistics.median(
                row["target_post_us"]
                for row in rows
                if row["model"] == model and row["tp"] == tp
            )
            for tp in TPS
        ]
        axes[0].plot(TPS, values, marker="o", label=model, color=colors[model])
    axes[0].set_title("Measured post-rendezvous time")
    axes[0].set_xlabel("TP group size")
    axes[0].set_ylabel("Median across representative configs (us)")
    axes[0].set_xticks(TPS)
    axes[0].legend()

    display = {
        "total_bytes_tp2_fit": "Total bytes\nTP2 fit",
        "continuous_tp2_calibrated": "Continuous\nTP2 calibrated",
        "continuous_per_tp_calibrated": "Continuous\nper-TP descriptive",
    }
    x = np.arange(len(METHODS))
    width = 0.34
    for offset, tp in ((-width / 2, 4), (width / 2, 8)):
        values = [
            next(
                row["mape"]
                for row in metrics
                if row["scope"] == f"tp{tp}_all" and row["method"] == method
            )
            * 100
            for method in METHODS
        ]
        axes[1].bar(x + offset, values, width, label=f"TP{tp}")
    axes[1].set_xticks(x, [display[method] for method in METHODS])
    axes[1].set_ylabel("MAPE (%)")
    axes[1].set_title("Cross-TP prediction error")
    axes[1].legend()

    ratios4 = [row["tp4_over_tp2"] for row in scaling]
    ratios8 = [row["tp8_over_tp2"] for row in scaling]
    axes[2].boxplot([ratios4, ratios8], tick_labels=["TP4 / TP2", "TP8 / TP2"])
    axes[2].axhline(1.0, color="#777777", linestyle="--", linewidth=1)
    axes[2].set_ylabel("Measured time ratio")
    axes[2].set_title("Group-size timing effect")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve = BackendAwareCostCurve(
        args.custom_curve,
        args.nccl_curve,
        custom_latency_column="completion_median_latency_us",
        nccl_latency_column="median_latency_us",
    )
    rows, source_files, raw_rows = load_rows(args, curve)
    model_spec = add_predictions(rows)
    metrics = build_metrics(rows)
    scaling = build_scaling(rows)

    write_csv(
        args.output_dir / "aggregated_configurations.csv",
        [public_row(row) for row in rows],
    )
    prediction_rows = []
    for row in rows:
        base = {
            key: row[key]
            for key in (
                "workload_id",
                "model",
                "tp",
                "mode",
                "case_label",
                "phase",
                "batch_size",
                "input_len",
                "target_post_us",
            )
        }
        for method in METHODS:
            predicted = row[f"pred_{method}_us"]
            prediction_rows.append(
                {
                    **base,
                    "method": method,
                    "predicted_us": predicted,
                    "absolute_percentage_error": abs(
                        predicted - row["target_post_us"]
                    )
                    / row["target_post_us"],
                }
            )
    write_csv(args.output_dir / "predictions.csv", prediction_rows)
    write_csv(args.output_dir / "metrics.csv", metrics)
    write_csv(args.output_dir / "tp_scaling_comparison.csv", scaling)
    make_figure(
        rows,
        metrics,
        scaling,
        args.output_dir / "tp_group_size_timing_analysis.png",
    )

    stability = {}
    for tp in TPS:
        values = [row["post_repeat_iqr_fraction"] for row in rows if row["tp"] == tp]
        stability[f"tp{tp}"] = {
            "median_iqr_fraction": float(statistics.median(values)),
            "p95_iqr_fraction": percentile(values, 95),
            "configurations_above_20pct_iqr": sum(value > 0.2 for value in values),
        }
    summary = {
        "schema_version": "tp-group-size-timing-analysis-v1",
        "dataset": {
            "models": list(MODELS),
            "tensor_parallel_sizes": list(TPS),
            "selected_raw_label_rows": raw_rows,
            "aggregated_configurations": len(rows),
            "phase14_formal_units": 60,
            "phase14_new_raw_label_rows": sum(row["repeat_count"] for row in rows if row["tp"] in (4, 8)),
            "phase14_new_aggregated_configurations": sum(1 for row in rows if row["tp"] in (4, 8)),
            "source_ground_truth_files": len(source_files),
            "target": "median of three repeats of all-rank post-rendezvous completion time",
            "stability": stability,
            "curve_extrapolated_calls": sum(row["curve_extrapolated_calls"] for row in rows),
            "curve_total_calls": sum(row["calls"] for row in rows),
            "cost_curve_note": "The structural curve is group-size-aware but payload-marginal. Raw-op identities and observed backend transitions are preserved separately; the curve does not claim a dedicated fused-op microbenchmark.",
        },
        "model_specification": model_spec,
        "metrics": metrics,
        "tp_scaling": {
            "groups": len(scaling),
            "calls_invariant_groups": sum(row["calls_invariant"] for row in scaling),
            "logical_bytes_invariant_groups": sum(
                row["logical_bytes_invariant"] for row in scaling
            ),
            "op_payload_histogram_invariant_groups": sum(
                row["op_payload_histogram_invariant"] for row in scaling
            ),
            "backend_transition_groups": sum(row["backend_changed"] for row in scaling),
            "tp4_over_tp2_median": float(
                statistics.median(row["tp4_over_tp2"] for row in scaling)
            ),
            "tp4_over_tp2_p95": percentile(
                [row["tp4_over_tp2"] for row in scaling], 95
            ),
            "tp8_over_tp2_median": float(
                statistics.median(row["tp8_over_tp2"] for row in scaling)
            ),
            "tp8_over_tp2_p95": percentile(
                [row["tp8_over_tp2"] for row in scaling], 95
            ),
        },
        "interpretation_guardrails": [
            "continuous_per_tp_calibrated is descriptive because it uses labels from the evaluated TP; only continuous_tp2_calibrated is the zero-shot cross-TP test",
            "results cover representative single-node B200 TP2/4/8 workloads, not the full Phase 10/12 grid",
            "results do not cover cross-node topology, PP, PD, or expert-parallel All-to-All",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"analyzed {len(rows)} TP2/4/8 configurations from "
        f"{raw_rows} selected labels"
    )


if __name__ == "__main__":
    main()
