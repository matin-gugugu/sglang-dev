#!/usr/bin/env python3
"""Generate Phase 28 Hfull labels after prediction freeze and evaluate the frozen mapping."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from build_phase27b_pp_hfull_dataset import (
    HISTORY_SECONDS,
    LABEL_STATUS,
    MICROBATCH_SIZES,
    PHASES,
    PP_SIZES,
    TEACHER_KIND,
    exact_histogram,
    label_row,
    summarize_profile,
)
from evaluate_phase27d_pp_independent_confirmation import (
    add_total_records,
    aggregate_records,
    case_record,
)
from prepare_phase15_trace_windows import BURST_FILES, MOONCAKE_FILES, load_segment


METHODS = ("h0", "enhanced_bounded_residual")
FROZEN_HYBRID = "frozen_hybrid"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument(
        "--selection",
        type=Path,
        default=root
        / "experiment-results/phase28a_second_confirmation_contract/selection/selected_windows.csv",
    )
    parser.add_argument(
        "--phase28a-summary",
        type=Path,
        default=root
        / "experiment-results/phase28a_second_confirmation_contract/summary.json",
    )
    parser.add_argument(
        "--frozen-mapping",
        type=Path,
        default=root
        / "experiment-results/phase28a_second_confirmation_contract/frozen_method_mapping.json",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=root
        / "experiment-results/phase28b_frozen_predictions/analysis/frozen_predictions.csv.gz",
    )
    parser.add_argument(
        "--phase28b-summary",
        type=Path,
        default=root / "experiment-results/phase28b_frozen_predictions/summary.json",
    )
    parser.add_argument(
        "--phase28b-audit",
        type=Path,
        default=root / "experiment-results/phase28b_frozen_predictions/audit_summary.json",
    )
    parser.add_argument(
        "--phase28b-manifest",
        type=Path,
        default=root / "experiment-results/phase28b_frozen_predictions/manifest.sha256",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase28c_second_confirmation_evaluation",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_gzip(path: Path, text: str) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as output:
            output.write(text.encode())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_csv_gz(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    deterministic_gzip(path, buffer.getvalue())


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def verify_manifest(path: Path) -> dict[str, bool]:
    root = path.parent
    checks = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        checks[relative] = sha256(root / relative) == expected
    return checks


def profile_inventory(profiles: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for profile in profiles:
        groups[profile["segment"]].append(profile)
    rows = []
    for segment, group in sorted(groups.items()):
        counts = [int(row["request_count"]) for row in group]
        rows.append(
            {
                "segment": segment,
                "profiles": len(group),
                "requests_total": sum(counts),
                "requests_min": min(counts),
                "requests_median": statistics.median(counts),
                "requests_max": max(counts),
            }
        )
    return rows


def metric_row(
    rows: list[dict], method: str, policy: str = "all", phase: str = "total"
) -> dict:
    return next(
        row
        for row in rows
        if row["method"] == method
        and row["phase"] == phase
        and row["policy"] == policy
        and row["segment"] == "all"
    )


def comparison_rows(diagnostic_metrics: list[dict], hybrid_metrics: list[dict]) -> list[dict]:
    fields = (
        "calls_mape",
        "calls_wape",
        "bytes_mape",
        "bytes_wape",
        "mean_histogram_l1",
        "mean_histogram_tv",
        "mean_normalized_log_payload_emd",
        "common_reference_cost_mape",
        "common_reference_cost_wape",
    )
    rows = []
    for policy in ("all", "mb1", "mb4", "mb16"):
        h0 = metric_row(diagnostic_metrics, "h0", policy)
        hybrid = metric_row(hybrid_metrics, FROZEN_HYBRID, policy)
        row: dict[str, object] = {
            "policy": policy,
            "frozen_method": (
                "mb1=h0;mb4/mb16=enhanced_bounded_residual"
                if policy == "all"
                else ("h0" if policy == "mb1" else "enhanced_bounded_residual")
            ),
        }
        for field in fields:
            baseline = float(h0[field])
            candidate = float(hybrid[field])
            row[f"h0_{field}"] = baseline
            row[f"frozen_hybrid_{field}"] = candidate
            row[f"delta_{field}"] = candidate - baseline
            row[f"relative_change_{field}"] = (
                candidate / baseline - 1 if baseline > 0 else 0.0
            )
        row["wins_calls_tv_cost"] = sum(
            float(hybrid[field]) < float(h0[field])
            for field in (
                "calls_mape",
                "mean_histogram_tv",
                "common_reference_cost_mape",
            )
        )
        rows.append(row)
    return rows


def plot_comparison(path: Path, comparisons: list[dict]) -> None:
    import matplotlib.pyplot as plt

    policies = ("all", "mb1", "mb4", "mb16")
    labels = ("All", "MB1", "MB4", "MB16")
    lookup = {row["policy"]: row for row in comparisons}
    facets = (
        ("calls_mape", "Total calls MAPE", 100.0, "%"),
        ("bytes_mape", "Logical bytes MAPE", 100.0, "%"),
        ("mean_histogram_tv", "Histogram TV", 1.0, ""),
        ("common_reference_cost_mape", "Common cost MAPE", 100.0, "%"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), constrained_layout=True)
    positions = np.arange(len(policies), dtype=np.float64)
    width = 0.36
    for axis, (metric, title, scale, suffix) in zip(axes.reshape(-1), facets):
        h0_values = [float(lookup[policy][f"h0_{metric}"]) * scale for policy in policies]
        hybrid_values = [
            float(lookup[policy][f"frozen_hybrid_{metric}"]) * scale
            for policy in policies
        ]
        h0_bars = axis.bar(
            positions - width / 2,
            h0_values,
            width,
            label="H0",
            color="#4C78A8",
        )
        hybrid_bars = axis.bar(
            positions + width / 2,
            hybrid_values,
            width,
            label="Frozen hybrid",
            color="#F58518",
        )
        axis.set_title(title)
        axis.set_xticks(positions, labels)
        axis.grid(axis="y", alpha=0.22, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        upper = max(h0_values + hybrid_values) * 1.22
        axis.set_ylim(0, upper if upper > 0 else 1.0)
        for bars, values in ((h0_bars, h0_values), (hybrid_bars, hybrid_values)):
            for bar, value in zip(bars, values):
                text = f"{value:.1f}{suffix}" if suffix else f"{value:.3f}"
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + axis.get_ylim()[1] * 0.018,
                    text,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper right")
    figure.suptitle(
        "Phase 28C second independent confirmation: H0 vs frozen PP mapping",
        fontsize=14,
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def readme(summary: dict) -> str:
    headline = summary["headline"]
    h0 = headline["h0"]
    hybrid = headline["frozen_hybrid"]
    table = [
        "| 方法 | calls MAPE/WAPE | bytes MAPE/WAPE | TV | norm EMD | common cost MAPE/WAPE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in (("H0", h0), ("冻结混合映射", hybrid)):
        table.append(
            "| {name} | {cm:.2%} / {cw:.2%} | {bm:.2%} / {bw:.2%} | "
            "{tv:.4f} | {emd:.4f} | {cost:.2%} / {costw:.2%} |".format(
                name=name,
                cm=row["calls_mape"],
                cw=row["calls_wape"],
                bm=row["bytes_mape"],
                bw=row["bytes_wape"],
                tv=row["mean_histogram_tv"],
                emd=row["mean_normalized_log_payload_emd"],
                cost=row["common_reference_cost_mape"],
                costw=row["common_reference_cost_wape"],
            )
        )
    policies = []
    for policy, row in summary["policy_headline"].items():
        policies.append(
            f"- {policy}：冻结方法 `{row['frozen_method']}`；calls MAPE "
            f"{row['frozen_hybrid_calls_mape']:.2%}（H0 {row['h0_calls_mape']:.2%}），"
            f"TV {row['frozen_hybrid_mean_histogram_tv']:.4f}（H0 "
            f"{row['h0_mean_histogram_tv']:.4f}），cost MAPE "
            f"{row['frozen_hybrid_common_reference_cost_mape']:.2%}（H0 "
            f"{row['h0_common_reference_cost_mape']:.2%}）。"
        )
    return f"""# Phase 28C：PP 冻结混合映射第二独立确认

状态：**{summary['status']}**。Phase 28B 的预测文件先以SHA-256冻结并提交Git，本阶段核验
hash和manifest之后，才由18个此前未用于Phase16/27的历史窗口生成Hfull teacher。没有
训练、早停、调参或改变 `MB1=H0，MB4/MB16=enhanced_bounded_residual` 映射。

## 18个第二确认画像的total结果

{chr(10).join(table)}

## 分microbatch结果

{chr(10).join(policies)}

对比图见 `figures/frozen_mapping_second_confirmation.png`。

Hfull标签来自完整窗口请求列表的GPU验证结构公式，完整请求列表只在内存中使用，未写入
结果目录或Git；保存的是324条归一化teacher phase rows。冻结映射的无偏结论只适用于
Qwen3-8B、PP2/4/8、fixed-draining和当前三种microbatch策略。common cost仍是5 μs +
100 GB/s统一参考曲线，不能当作PP P2P物理链路实测；也不能外推到跨模型PP或online
arrival-aware调度。
"""


def main() -> None:
    args = parse_args()
    for name in ("labels", "analysis", "figures", "logs"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)

    contract = json.loads(args.phase28a_summary.read_text())
    mapping = json.loads(args.frozen_mapping.read_text())["mapping"]
    summary28b = json.loads(args.phase28b_summary.read_text())
    audit28b = json.loads(args.phase28b_audit.read_text())
    manifest28b = verify_manifest(args.phase28b_manifest)
    expected_mapping = {
        "mb1": "h0",
        "mb4": "enhanced_bounded_residual",
        "mb16": "enhanced_bounded_residual",
    }
    if contract["label_state_at_freeze"] != "no_phase28_hfull_labels_generated":
        raise RuntimeError("Phase 28A label freeze is not clean")
    if mapping != expected_mapping or summary28b["frozen_mapping"] != expected_mapping:
        raise RuntimeError("frozen mapping mismatch")
    if summary28b["status"] != "PASS" or audit28b["status"] != "PASS":
        raise RuntimeError("Phase 28B is not PASS")
    frozen_prediction_hash = audit28b["frozen_predictions_sha256"]
    if sha256(args.predictions) != frozen_prediction_hash:
        raise RuntimeError("frozen prediction hash mismatch")

    # Prediction integrity is established before any Phase 28 Hfull target is generated.
    predictions = read_csv_gz(args.predictions)
    selected_predictions = [
        row for row in predictions if row["selected_by_frozen_mapping"] == "True"
    ]
    selection = read_csv(args.selection)
    if len(predictions) != 648 or len(selected_predictions) != 324 or len(selection) != 18:
        raise RuntimeError("unexpected frozen artifact row counts")

    raw_manifest_path = args.raw_dir / "source_manifest.json"
    raw_manifest = json.loads(raw_manifest_path.read_text())
    raw_checks = {}
    for source in raw_manifest["sources"]:
        path = args.raw_dir / source["name"]
        raw_checks[source["name"]] = (
            path.stat().st_size == int(source["actual_size"])
            and sha256(path) == source["sha256"]
        )
    if len(raw_checks) != 6 or not all(raw_checks.values()):
        raise RuntimeError(raw_checks)

    file_by_segment = {
        segment: args.raw_dir / name
        for name, (segment, _) in {**BURST_FILES, **MOONCAKE_FILES}.items()
    }
    arrays = {segment: load_segment(path) for segment, path in file_by_segment.items()}
    profiles = []
    windows: dict[str, list[tuple[int, int]]] = {}
    history_count_checks = []
    for selected in selection:
        timestamps, inputs, outputs = arrays[selected["segment"]]
        cutoff = int(selected["cutoff_ms"])
        left = int(
            np.searchsorted(timestamps, cutoff - HISTORY_SECONDS * 1000, side="left")
        )
        right = int(np.searchsorted(timestamps, cutoff, side="left"))
        compatibility = {
            **selected,
            "phase27_profile_id": selected["profile_id"],
            "phase27_role": selected["role"],
        }
        profile, requests = summarize_profile(
            compatibility,
            timestamps[left:right],
            inputs[left:right],
            outputs[left:right],
        )
        history_count_checks.append(len(requests) == int(selected["history_count"]))
        profiles.append(profile)
        windows[profile["profile_id"]] = requests

    targets = []
    simulation_checks = []
    for profile in profiles:
        requests = windows[profile["profile_id"]]
        for pp_size in PP_SIZES:
            for microbatch in MICROBATCH_SIZES:
                histograms, scheduler_audit = exact_histogram(
                    requests, pp_size, microbatch
                )
                simulation_checks.append(
                    scheduler_audit["all_requests_complete"]
                    and scheduler_audit["prefill_token_mass"]
                    == sum(row[0] for row in requests)
                    and scheduler_audit["decode_token_mass"]
                    == sum(row[1] - 1 for row in requests)
                )
                for phase in PHASES:
                    targets.append(
                        label_row(
                            profile,
                            pp_size,
                            microbatch,
                            phase,
                            histograms[phase],
                        )
                    )

    target_lookup = {row["label_id"]: row for row in targets}
    join_failures = []
    diagnostic_phase_records = []
    for prediction in predictions:
        target = target_lookup.get(prediction["training_id"])
        if target is None:
            join_failures.append(prediction["training_id"])
            continue
        diagnostic_phase_records.append(case_record(prediction, target))
    selected_ids = {
        (row["training_id"], row["method"]) for row in selected_predictions
    }
    hybrid_phase_records = [
        {**row, "method": FROZEN_HYBRID}
        for row in diagnostic_phase_records
        if (row["training_id"], row["method"]) in selected_ids
    ]
    diagnostic_records = add_total_records(diagnostic_phase_records)
    hybrid_records = add_total_records(hybrid_phase_records)
    diagnostic_metrics = aggregate_records(diagnostic_records)
    hybrid_metrics = aggregate_records(hybrid_records)
    comparisons = comparison_rows(diagnostic_metrics, hybrid_metrics)
    headline = {
        "h0": metric_row(diagnostic_metrics, "h0"),
        FROZEN_HYBRID: metric_row(hybrid_metrics, FROZEN_HYBRID),
        "enhanced_bounded_residual": metric_row(
            diagnostic_metrics, "enhanced_bounded_residual"
        ),
    }
    policy_headline = {
        row["policy"]: row for row in comparisons if row["policy"] != "all"
    }

    write_csv_gz(args.output_dir / "labels/hfull_targets.csv.gz", targets)
    write_csv_gz(
        args.output_dir / "analysis/diagnostic_predictions_and_errors.csv.gz",
        diagnostic_records,
    )
    write_csv_gz(
        args.output_dir / "analysis/frozen_hybrid_predictions_and_errors.csv.gz",
        hybrid_records,
    )
    write_csv(args.output_dir / "analysis/diagnostic_metrics.csv", diagnostic_metrics)
    write_csv(args.output_dir / "analysis/frozen_hybrid_metrics.csv", hybrid_metrics)
    write_csv(args.output_dir / "analysis/frozen_mapping_vs_h0.csv", comparisons)
    write_csv(args.output_dir / "analysis/profile_inventory.csv", profile_inventory(profiles))
    plot_comparison(
        args.output_dir / "figures/frozen_mapping_second_confirmation.png",
        comparisons,
    )

    finite_fields = (
        "calls_mape",
        "calls_wape",
        "bytes_mape",
        "bytes_wape",
        "mean_histogram_tv",
        "mean_normalized_log_payload_emd",
        "common_reference_cost_mape",
        "common_reference_cost_wape",
    )
    checks = {
        "phase28a_contract_pass": contract["status"] == "PASS",
        "phase28b_status_pass": summary28b["status"] == "PASS"
        and audit28b["status"] == "PASS",
        "phase28b_manifest_11_of_11_pass": len(manifest28b) == 11
        and all(manifest28b.values()),
        "prediction_hash_matches_frozen_audit": sha256(args.predictions)
        == frozen_prediction_hash,
        "frozen_mapping_unchanged": mapping == expected_mapping,
        "raw_source_hashes_6_of_6": len(raw_checks) == 6
        and all(raw_checks.values()),
        "profiles_18": len(profiles) == 18,
        "history_counts_match_18_of_18": all(history_count_checks),
        "full_window_requests_not_saved": not any(
            key in targets[0]
            for key in ("input_lens", "output_lens", "request_list", "requests_json")
        ),
        "hfull_scheduler_simulations_exact_162_of_162": len(simulation_checks) == 162
        and all(simulation_checks),
        "targets_324": len(targets) == 324 and len(target_lookup) == 324,
        "predictions_648_and_selected_324": len(predictions) == 648
        and len(selected_predictions) == 324,
        "diagnostic_join_648_of_648": len(diagnostic_phase_records) == 648
        and not join_failures,
        "hybrid_phase_records_324": len(hybrid_phase_records) == 324,
        "phase_plus_total_records_972_and_486": len(diagnostic_records) == 972
        and len(hybrid_records) == 486,
        "methods_two_by_324": Counter(
            row["method"] for row in diagnostic_phase_records
        )
        == Counter({method: 324 for method in METHODS}),
        "all_metrics_finite": all(
            math.isfinite(float(row[field]))
            for row in diagnostic_metrics + hybrid_metrics
            for field in finite_fields
        ),
        "training_or_selection_changes_none": True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    if status != "PASS":
        raise RuntimeError(checks)

    summary = {
        "schema_version": "phase28c-second-confirmation-evaluation-v1",
        "status": status,
        "objective": "unbiased second confirmation of the Phase 27D post-confirmation PP method mapping on untouched windows",
        "model": "qwen3-8b",
        "parallel_sizes": list(PP_SIZES),
        "policies": [f"mb{value}" for value in MICROBATCH_SIZES],
        "frozen_mapping": mapping,
        "counts": {
            "profiles": len(profiles),
            "full_window_requests": sum(int(row["request_count"]) for row in profiles),
            "hfull_target_phase_rows": len(targets),
            "frozen_prediction_rows": len(predictions),
            "selected_prediction_rows": len(selected_predictions),
            "hfull_scheduler_simulations": len(simulation_checks),
        },
        "teacher_status": LABEL_STATUS,
        "teacher_kind": TEACHER_KIND,
        "inputs": {
            "phase28a_summary_sha256": sha256(args.phase28a_summary),
            "selection_sha256": sha256(args.selection),
            "frozen_mapping_sha256": sha256(args.frozen_mapping),
            "phase28b_summary_sha256": sha256(args.phase28b_summary),
            "phase28b_audit_sha256": sha256(args.phase28b_audit),
            "phase28b_manifest_sha256": sha256(args.phase28b_manifest),
            "frozen_predictions_sha256": sha256(args.predictions),
            "raw_manifest_sha256": sha256(raw_manifest_path),
        },
        "headline": headline,
        "policy_headline": policy_headline,
        "checks": checks,
        "training_performed": False,
        "model_selection_changed": False,
        "can_conclude": [
            "the unbiased error of the pre-frozen MB1=H0 and MB4/MB16=enhanced-residual mapping on 18 new windows",
            "whether that frozen mapping improves over H0 for Qwen3-8B fixed-draining PP2/4/8 under the common reference curve",
        ],
        "cannot_conclude": [
            "cross-model PP generalization",
            "physical PP P2P communication-time accuracy from the common reference curve",
            "online arrival-aware scheduling behavior",
        ],
        "next_step": "use the second-confirmation result to decide whether the frozen Qwen3-8B PP mapping can be promoted, then add new PP models without reopening this holdout",
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(
        args.output_dir / "audit_summary.json",
        {
            "schema_version": "phase28c-second-confirmation-audit-v1",
            "status": status,
            "checks": checks,
            "frozen_prediction_sha256": frozen_prediction_hash,
            "hfull_targets_sha256": sha256(
                args.output_dir / "labels/hfull_targets.csv.gz"
            ),
            "join_failures": join_failures,
        },
    )
    (args.output_dir / "README.md").write_text(readme(summary))
    (args.output_dir / "DONE").write_text("PASS\n")
    write_json(
        args.output_dir / "logs/teacher_generation.log",
        {
            "schema_version": "phase28c-teacher-generation-log-v1",
            "status": status,
            "prediction_hash_verified_before_target_generation": True,
            "profiles": len(profiles),
            "full_window_requests": sum(int(row["request_count"]) for row in profiles),
            "scheduler_simulations": len(simulation_checks),
            "target_phase_rows": len(targets),
            "full_request_lists_saved": False,
        },
    )
    write_json(
        args.output_dir / "logs/evaluation.log",
        {
            "schema_version": "phase28c-evaluation-log-v1",
            "status": status,
            "prediction_hash_verified": True,
            "phase28b_manifest_checks": manifest28b,
            "training_performed": False,
            "model_selection_changed": False,
            "diagnostic_joined_phase_rows": len(diagnostic_phase_records),
            "frozen_hybrid_phase_rows": len(hybrid_phase_records),
        },
    )
    files = sorted(
        path
        for path in args.output_dir.rglob("*")
        if path.is_file() and path.name != "manifest.sha256"
    )
    (args.output_dir / "manifest.sha256").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(args.output_dir)}\n" for path in files)
    )
    print(
        json.dumps(
            {
                "status": status,
                "profiles": len(profiles),
                "full_window_requests": summary["counts"]["full_window_requests"],
                "hfull_target_phase_rows": len(targets),
                "headline": {
                    method: {
                        "calls_mape": headline[method]["calls_mape"],
                        "calls_wape": headline[method]["calls_wape"],
                        "bytes_mape": headline[method]["bytes_mape"],
                        "bytes_wape": headline[method]["bytes_wape"],
                        "histogram_tv": headline[method]["mean_histogram_tv"],
                        "normalized_log_payload_emd": headline[method][
                            "mean_normalized_log_payload_emd"
                        ],
                        "cost_mape": headline[method]["common_reference_cost_mape"],
                        "cost_wape": headline[method]["common_reference_cost_wape"],
                    }
                    for method in ("h0", FROZEN_HYBRID)
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
