#!/usr/bin/env python3
"""Recompute PP H32/H64/H128/Hfull convergence with the Phase 25B scheduler."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from analyze_phase24_representative_request_convergence import (
    COMMON_REFERENCE_BANDWIDTH_GBPS,
    COMMON_REFERENCE_LAUNCH_US,
    CONVERGENCE_THRESHOLDS,
    PHASES,
    PP_CHUNK_TOKENS,
    PP_MICROBATCH_SIZES,
    PP_PROXY_TENSOR_COUNT,
    PP_SIZES,
    REQUESTED_ESTIMATORS,
    build_aggregates,
    common_reference_cost_us,
    histogram_cost,
    make_label_row,
    meets_thresholds,
    metric_row,
    normalize_histogram,
    overall_row,
    sha256,
    write_csv,
    write_csv_gz,
    write_json,
    write_jsonl_gz,
)
from build_phase25b_pp_scheduler_teacher import BYTES_PER_TOKEN, simulate_scheduler


SAMPLE_LABELS = ("h32", "h64", "h128", "hfull", "compact32")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase24-root",
        type=Path,
        default=root / "experiment-results/phase24_representative_request_convergence",
    )
    parser.add_argument(
        "--phase25b-root",
        type=Path,
        default=root / "experiment-results/phase25b_pp_scheduler_teacher",
    )
    parser.add_argument(
        "--phase25c-root",
        type=Path,
        default=root / "experiment-results/phase25c_pp_scheduler_tail_audit",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root
        / "experiment-results/phase25d_pp_scheduler_representative_convergence",
    )
    return parser.parse_args()


def read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as source:
        return [json.loads(line) for line in source if line.strip()]


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as source:
        return list(csv.DictReader(source))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def scheduler_histograms(
    requests: list[tuple[int, int]], pp_size: int, microbatch: int
) -> dict[str, dict[int, float]]:
    simulated = simulate_scheduler(
        requests, pp_size=pp_size, max_microbatch=microbatch
    )
    return {
        phase: normalize_histogram(
            {
                int(tokens) * BYTES_PER_TOKEN: int(events) * PP_PROXY_TENSOR_COUNT
                for tokens, events in simulated.event_histograms[phase].items()
            },
            len(requests),
        )
        for phase in PHASES
    }


def build_labels(
    request_rows: list[dict],
) -> tuple[list[dict], dict[tuple, dict[int, float]], list[dict]]:
    labels = []
    lookup = {}
    simulation_stats = []
    for profile in request_rows:
        requests = list(
            zip(map(int, profile["input_lens"]), map(int, profile["output_lens"]))
        )
        sample_label = profile["sample_label"]
        estimator_kind = (
            "compact_profile" if sample_label == "compact32" else "exact_representative"
        )
        for pp_size in PP_SIZES:
            for microbatch in PP_MICROBATCH_SIZES:
                histograms = scheduler_histograms(requests, pp_size, microbatch)
                simulation_stats.append(
                    {
                        "profile_id": profile["profile_id"],
                        "source": profile["source"],
                        "sample_label": sample_label,
                        "sample_requests": len(requests),
                        "pp_size": pp_size,
                        "microbatch": microbatch,
                        "prefill_proxy_calls_per_1000": sum(
                            histograms["prefill"].values()
                        ),
                        "decode_proxy_calls_per_1000": sum(
                            histograms["decode"].values()
                        ),
                        "total_logical_bytes_per_1000": sum(
                            payload * calls
                            for histogram in histograms.values()
                            for payload, calls in histogram.items()
                        ),
                    }
                )
                for phase in PHASES:
                    histogram = histograms[phase]
                    key = (
                        profile["profile_id"],
                        "pp",
                        pp_size,
                        f"mb{microbatch}",
                        sample_label,
                        phase,
                    )
                    lookup[key] = histogram
                    labels.append(
                        {
                            **make_label_row(
                                profile,
                                "pp",
                                pp_size,
                                f"mb{microbatch}",
                                estimator_kind,
                                sample_label,
                                len(requests),
                                int(profile["full_request_count"]),
                                phase,
                                histogram,
                                histogram_cost(histogram, common_reference_cost_us),
                                None,
                                boundary_multiplier=pp_size - 1,
                            ),
                            "teacher_kind": "sglang_pp_fcfs_lanes_v1",
                            "teacher_gpu_evidence": "phase25b_smoke_9_of_9_plus_phase25c_tail_3_of_3",
                        }
                    )
    return labels, lookup, simulation_stats


def build_metrics(
    profiles: list[dict], lookup: dict[tuple, dict[int, float]]
) -> tuple[list[dict], list[dict]]:
    metrics, decomposition = [], []
    for profile in profiles:
        profile_id = profile["profile_id"]
        for pp_size in PP_SIZES:
            for microbatch in PP_MICROBATCH_SIZES:
                policy = f"mb{microbatch}"
                truth = {
                    phase: lookup[(profile_id, "pp", pp_size, policy, "hfull", phase)]
                    for phase in PHASES
                }
                for sample_label in (*REQUESTED_ESTIMATORS, "compact32"):
                    predicted = {
                        phase: lookup[
                            (profile_id, "pp", pp_size, policy, sample_label, phase)
                        ]
                        for phase in PHASES
                    }
                    for phase in (*PHASES, "total"):
                        metrics.append(
                            metric_row(
                                profile,
                                "pp",
                                pp_size,
                                policy,
                                sample_label,
                                "compact_profile"
                                if sample_label == "compact32"
                                else "exact_representative",
                                phase,
                                predicted,
                                truth,
                                {},
                                "hfull",
                            )
                        )
                compact = {
                    phase: lookup[
                        (profile_id, "pp", pp_size, policy, "compact32", phase)
                    ]
                    for phase in PHASES
                }
                h32 = {
                    phase: lookup[(profile_id, "pp", pp_size, policy, "h32", phase)]
                    for phase in PHASES
                }
                for phase in (*PHASES, "total"):
                    decomposition.append(
                        metric_row(
                            profile,
                            "pp",
                            pp_size,
                            policy,
                            "compact32",
                            "compact_profile",
                            phase,
                            compact,
                            h32,
                            {},
                            "h32",
                        )
                    )
    return metrics, decomposition


def decisions(aggregates: list[dict]) -> dict:
    by_sample = {}
    sufficient = []
    for sample_label in REQUESTED_ESTIMATORS:
        row = overall_row(aggregates, "pp", sample_label)
        passed = meets_thresholds(row)
        if passed:
            sufficient.append(sample_label)
        by_sample[sample_label] = {
            "meets_all": passed,
            "calls_mape": row["calls_mape"],
            "calls_wape": row["calls_wape"],
            "bytes_mape": row["bytes_mape"],
            "bytes_wape": row["bytes_wape"],
            "histogram_tv": row["mean_calls_histogram_tv"],
            "normalized_log_payload_emd": row["mean_normalized_log_payload_emd"],
            "common_reference_cost_mape": row["common_reference_cost_mape"],
            "p95_calls_ape": row["p95_calls_ape"],
            "p95_bytes_ape": row["p95_bytes_ape"],
            "p95_common_reference_cost_ape": row[
                "p95_common_reference_cost_ape"
            ],
        }
    return {
        "minimum_sample_meeting_all_preregistered_thresholds": (
            sufficient[0] if sufficient else None
        ),
        "by_sample": by_sample,
    }


def compare_phase25b_hfull(
    labels: list[dict], phase25b_path: Path
) -> tuple[int, list[dict]]:
    expected = {
        (
            row["profile_id"],
            int(row["parallel_size"]),
            row["policy"],
            row["phase"],
        ): row
        for row in read_csv_gz(phase25b_path)
    }
    actual = {
        (
            row["profile_id"],
            int(row["parallel_size"]),
            row["policy"],
            row["phase"],
        ): row
        for row in labels
        if row["sample_label"] == "hfull"
    }
    failures = []
    for key in sorted(set(expected) | set(actual)):
        left, right = actual.get(key), expected.get(key)
        exact = (
            left is not None
            and right is not None
            and json.loads(left["exact_calls_histogram_per_1000_json"])
            == json.loads(right["exact_calls_histogram_per_1000_json"])
            and math.isclose(
                float(left["total_calls_per_1000"]),
                float(right["total_calls_per_1000"]),
                rel_tol=0,
                abs_tol=1e-9,
            )
            and math.isclose(
                float(left["total_logical_bytes_per_1000"]),
                float(right["total_logical_bytes_per_1000"]),
                rel_tol=0,
                abs_tol=1e-6,
            )
        )
        if not exact:
            failures.append({"key": key, "exact": False})
    return len(expected) - len(failures), failures


def old_new_rows(
    old_path: Path, new_aggregates: list[dict]
) -> list[dict]:
    old = read_csv_rows(old_path)
    output = []
    for sample_label in REQUESTED_ESTIMATORS:
        old_matches = [
            row
            for row in old
            if row["aggregation_scope"] == "overall"
            and row["parallelism"] == "pp"
            and row["sample_label"] == sample_label
            and row["reference_label"] == "hfull"
            and row["phase"] == "total"
        ]
        if len(old_matches) != 1:
            raise ValueError(f"missing Phase 24 aggregate for {sample_label}")
        new = overall_row(new_aggregates, "pp", sample_label)
        output.append(
            {
                "sample_label": sample_label,
                "old_teacher": "phase24_static_pp_groups",
                "new_teacher": "phase25b_sglang_pp_fcfs_lanes_v1",
                **{
                    f"old_{field}": float(old_matches[0][field])
                    for field in (
                        "calls_mape",
                        "calls_wape",
                        "bytes_mape",
                        "bytes_wape",
                        "mean_calls_histogram_tv",
                        "mean_normalized_log_payload_emd",
                        "common_reference_cost_mape",
                        "p95_calls_ape",
                    )
                },
                **{
                    f"new_{field}": float(new[field])
                    for field in (
                        "calls_mape",
                        "calls_wape",
                        "bytes_mape",
                        "bytes_wape",
                        "mean_calls_histogram_tv",
                        "mean_normalized_log_payload_emd",
                        "common_reference_cost_mape",
                        "p95_calls_ape",
                    )
                },
            }
        )
    return output


def plot(path: Path, decision: dict) -> None:
    import matplotlib.pyplot as plt

    labels = list(REQUESTED_ESTIMATORS)
    panels = (
        ("calls_mape", "Calls MAPE", 100.0, "%"),
        ("calls_wape", "Calls WAPE", 100.0, "%"),
        ("bytes_mape", "Logical bytes MAPE", 100.0, "%"),
        ("histogram_tv", "Histogram TV", 1.0, ""),
        ("normalized_log_payload_emd", "Normalized log-payload EMD", 1.0, ""),
        ("common_reference_cost_mape", "Reference-cost MAPE", 100.0, "%"),
    )
    colors = ("#4C78A8", "#F58518", "#E45756")
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 6.4), constrained_layout=True)
    for axis, (field, title, scale, suffix) in zip(axes.flat, panels):
        values = [float(decision["by_sample"][label][field]) * scale for label in labels]
        bars = axis.bar([label.upper() for label in labels], values, color=colors)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}{suffix}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axis.margins(y=0.20)
    figure.suptitle("Phase 25D scheduler-faithful PP representative convergence")
    figure.savefig(path, dpi=180, metadata={"Software": "matplotlib"})
    plt.close(figure)


def git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.strip()


def readme(summary: dict) -> str:
    rows = summary["convergence"]["by_sample"]
    lines = []
    for label in REQUESTED_ESTIMATORS:
        row = rows[label]
        lines.append(
            "| {label} | {calls:.2%} | {wape:.2%} | {bytes:.2%} | {tv:.4f} | "
            "{emd:.4f} | {cost:.2%} | {p95:.2%} | {passed} |".format(
                label=label.upper(),
                calls=row["calls_mape"],
                wape=row["calls_wape"],
                bytes=row["bytes_mape"],
                tv=row["histogram_tv"],
                emd=row["normalized_log_payload_emd"],
                cost=row["common_reference_cost_mape"],
                p95=row["p95_calls_ape"],
                passed="PASS" if row["meets_all"] else "FAIL",
            )
        )
    return f"""# Phase 25D: scheduler-faithful PP representative convergence

Status: **{summary['status']}**. This reruns H32/H64/H128/Hfull and compact32
with the Phase 25B SGLang lane scheduler, replacing the static PP grouping used
by Phase 24. Hfull reproduces all {summary['phase25b_regression']['exact_rows']}/432
Phase 25B teacher rows exactly.

| Sample | calls MAPE | calls WAPE | bytes MAPE | hist TV | norm EMD | cost MAPE | P95 calls APE | all gates |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
{chr(10).join(lines)}

The experiment covers 24 BurstGPT/Mooncake windows, PP2/4/8, MB1/4/16,
Prefill/Decode/total, and normalization per 1,000 requests. It stores exact
payload histograms, per-case and aggregate metrics, compact32 decomposition,
the old-vs-new teacher comparison, figure, logs, DONE, and manifest.

Phase 25B has GPU evidence from all nine configurations on the 42-request smoke;
Phase 25C adds exact BurstGPT and Mooncake long-prompt tail evidence on the three
diagonal configurations. This validates the teacher contract in those audited
regions. It does not turn H32/H64/H128 into GPU-measured labels or cover online
arrival-aware scheduling.

Complete request lists remain offline label-generation inputs. The next model
still consumes only the compact history profile, model structure, fixed PP
configuration, policy, and phase.
"""


def main() -> None:
    started = time.time()
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    for directory in ("labels", "analysis", "figures", "logs"):
        (args.output_dir / directory).mkdir(parents=True, exist_ok=True)

    request_path = args.phase24_root / "input_windows/selected_requests.jsonl.gz"
    request_rows = read_jsonl_gz(request_path)
    if len(request_rows) != 24 * len(SAMPLE_LABELS):
        raise ValueError(f"unexpected request rows: {len(request_rows)}")
    profiles = [row for row in request_rows if row["sample_label"] == "hfull"]
    labels, lookup, simulation_stats = build_labels(request_rows)
    metrics, decomposition = build_metrics(profiles, lookup)
    aggregates = build_aggregates(metrics)
    decomposition_aggregates = build_aggregates(decomposition)
    decision = decisions(aggregates)

    write_jsonl_gz(args.output_dir / "labels/histogram_labels.jsonl.gz", labels)
    write_csv_gz(args.output_dir / "analysis/per_case_metrics.csv.gz", metrics)
    write_csv_gz(
        args.output_dir / "analysis/decomposition_metrics.csv.gz", decomposition
    )
    write_csv(
        args.output_dir / "analysis/aggregate_metrics.csv",
        aggregates + decomposition_aggregates,
    )
    write_csv(args.output_dir / "analysis/simulation_statistics.csv", simulation_stats)

    phase25b_labels = args.phase25b_root / "labels/pp_phase_labels.csv.gz"
    exact_rows, regression_failures = compare_phase25b_hfull(labels, phase25b_labels)
    old_new = old_new_rows(
        args.phase24_root / "analysis/aggregate_metrics.csv", aggregates
    )
    write_csv(args.output_dir / "analysis/phase24_old_vs_phase25d.csv", old_new)
    plot(args.output_dir / "figures/convergence.png", decision)

    finite = all(
        math.isfinite(float(row[field]))
        for row in metrics + decomposition
        for field in (
            "calls_ape",
            "bytes_ape",
            "calls_histogram_l1",
            "normalized_log_payload_emd",
            "common_reference_cost_ape",
        )
    )
    checks = {
        "request_rows_120": len(request_rows) == 120,
        "profiles_24": len(profiles) == 24,
        "histogram_labels_2160": len(labels) == 24 * 5 * 9 * 2,
        "per_case_metrics_2592": len(metrics) == 24 * 9 * 4 * 3,
        "decomposition_metrics_648": len(decomposition) == 24 * 9 * 3,
        "simulation_configurations_1080": len(simulation_stats) == 24 * 5 * 9,
        "phase25b_hfull_exact_432_of_432": exact_rows == 432
        and not regression_failures,
        "phase25c_tail_audit_pass": json.loads(
            (args.phase25c_root / "summary.json").read_text()
        )["status"]
        == "PASS",
        "all_metrics_finite": finite,
        "all_histograms_positive": all(
            float(row["total_calls_per_1000"]) > 0
            and float(row["total_logical_bytes_per_1000"]) > 0
            for row in labels
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = {
        "schema_version": "phase25d-pp-scheduler-representative-convergence-v1",
        "status": status,
        "objective": "Recompute PP H32/H64/H128/Hfull and compact32 convergence with the scheduler-faithful Phase 25B formula",
        "teacher_contract": "sglang_pp_fcfs_lanes_v1; fixed-draining; full request order; 4096-token chunk; page size 64; no radix/mixed chunk/async PP depth",
        "inputs": {
            "profiles": 24,
            "request_rows": len(request_rows),
            "phase24_selected_requests_sha256": sha256(request_path),
            "phase25b_labels_sha256": sha256(phase25b_labels),
            "phase25c_summary_sha256": sha256(args.phase25c_root / "summary.json"),
            "repository_head_at_build": git_head(root),
        },
        "outputs": {
            "histogram_labels": len(labels),
            "per_case_metrics": len(metrics),
            "decomposition_metrics": len(decomposition),
            "simulation_configurations": len(simulation_stats),
        },
        "phase25b_regression": {
            "expected_rows": 432,
            "exact_rows": exact_rows,
            "failures": regression_failures,
        },
        "gpu_evidence": {
            "phase25b_smoke": "9/9 PP2/4/8 x MB1/4/16 cells exact on 42-request BurstGPT window",
            "phase25c_tail": "3/3 diagonal cells, 6/6 profile-cells, 12/12 phase comparisons exact on BurstGPT/Mooncake tails",
        },
        "convergence_thresholds": CONVERGENCE_THRESHOLDS,
        "convergence": decision,
        "old_vs_new_teacher": old_new,
        "cost_contract": {
            "kind": "common parameterized reference, not physical PP measurement",
            "launch_us": COMMON_REFERENCE_LAUNCH_US,
            "bandwidth_gbps": COMMON_REFERENCE_BANDWIDTH_GBPS,
        },
        "checks": checks,
        "can_conclude": [
            "representative request convergence under the scheduler-faithful PP contract can be assessed against full-window labels",
            "the reported Phase 25D PP metrics supersede Phase 24 PP convergence metrics that used static groups",
        ],
        "cannot_conclude": [
            "representative structural labels are direct GPU measurements",
            "the same convergence applies to online arrivals or a different scheduler contract",
        ],
        "next_step": "use the Phase 25B Hfull labels as PP supervision and retrain compact-profile H0/direct/bounded-residual predictors; keep Phase 16 TP retraining on its separately audited Hfull teacher",
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(
        args.output_dir / "contract.json",
        {
            "schema_version": "phase25d-contract-v1",
            "status": status,
            "teacher": summary["teacher_contract"],
            "predictor_input": "compact history profile + model structure + fixed PP + fixed policy + phase",
            "offline_only_input": "H32/H64/H128/Hfull exact request lengths and order",
            "gpu_evidence": summary["gpu_evidence"],
        },
    )
    (args.output_dir / "README.md").write_text(readme(summary))
    write_json(
        args.output_dir / "logs/run.log",
        {
            "status": status,
            "argv": sys.argv,
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "duration_seconds": time.time() - started,
            "checks": checks,
        },
    )
    if status != "PASS":
        raise RuntimeError(json.dumps(summary, indent=2))
    (args.output_dir / "DONE").write_text("PASS\n")
    files = sorted(
        path
        for path in args.output_dir.rglob("*")
        if path.is_file() and path.name != "manifest.sha256"
    )
    (args.output_dir / "manifest.sha256").write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(args.output_dir)}\n" for path in files
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
