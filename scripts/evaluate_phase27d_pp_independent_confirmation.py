#!/usr/bin/env python3
"""Evaluate frozen Phase 27C predictions on independent Phase 27B targets."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


METHODS = (
    "h0",
    "legacy_bounded_residual",
    "enhanced_bounded_residual",
    "enhanced_direct",
)
PHASES = ("prefill", "decode")
COMMON_REFERENCE_LAUNCH_US = 5.0
COMMON_REFERENCE_BANDWIDTH_GBPS = 100.0
BIN_EDGES = np.geomspace(4 * 1024, 8 * 1024 * 1024 * 1024, 13).tolist()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=root
        / "experiment-results/phase27c_pp_scheduler_feature_training/analysis/independent_confirmation_predictions.csv.gz",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=root
        / "experiment-results/phase27b_pp_hfull_dataset/labels/independent_confirmation_hfull_targets.csv.gz",
    )
    parser.add_argument(
        "--phase27c-summary",
        type=Path,
        default=root
        / "experiment-results/phase27c_pp_scheduler_feature_training/summary.json",
    )
    parser.add_argument(
        "--phase27c-audit",
        type=Path,
        default=root
        / "experiment-results/phase27c_pp_scheduler_feature_training/audit_summary.json",
    )
    parser.add_argument(
        "--phase27c-manifest",
        type=Path,
        default=root
        / "experiment-results/phase27c_pp_scheduler_feature_training/manifest.sha256",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root
        / "experiment-results/phase27d_pp_independent_confirmation",
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


def load_rows(path: Path) -> list[dict[str, str]]:
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


def verify_manifest(path: Path) -> dict:
    root = path.parent
    results = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        results[relative] = sha256(root / relative) == expected
    return results


def histogram_tv(predicted: np.ndarray, actual: np.ndarray) -> float:
    predicted_total = max(float(predicted.sum()), 1e-12)
    actual_total = max(float(actual.sum()), 1e-12)
    return float(np.abs(predicted / predicted_total - actual / actual_total).sum() / 2)


def normalized_log_emd(predicted: np.ndarray, actual: np.ndarray) -> float:
    predicted_total = max(float(predicted.sum()), 1e-12)
    actual_total = max(float(actual.sum()), 1e-12)
    predicted_cdf = np.cumsum(predicted / predicted_total)
    actual_cdf = np.cumsum(actual / actual_total)
    centers = (np.log2(BIN_EDGES[:-1]) + np.log2(BIN_EDGES[1:])) / 2
    area = float(
        np.sum(np.abs(predicted_cdf[:-1] - actual_cdf[:-1]) * np.diff(centers))
    )
    return area / (math.log2(BIN_EDGES[-1]) - math.log2(BIN_EDGES[0]))


def reference_cost(calls: np.ndarray, logical_bytes: np.ndarray) -> float:
    return float(
        COMMON_REFERENCE_LAUNCH_US * calls.sum()
        + logical_bytes.sum() / (COMMON_REFERENCE_BANDWIDTH_GBPS * 1e9) * 1e6
    )


def case_record(prediction: dict[str, str], target: dict[str, str]) -> dict:
    predicted_calls = np.asarray(
        json.loads(prediction["predicted_calls_by_12bin_json"]), dtype=np.float64
    )
    predicted_bytes = np.asarray(
        json.loads(prediction["predicted_logical_bytes_by_12bin_json"]),
        dtype=np.float64,
    )
    actual_calls = np.asarray(json.loads(target["calls_by_12bin_json"]), dtype=np.float64)
    actual_bytes = np.asarray(
        json.loads(target["logical_bytes_by_12bin_json"]), dtype=np.float64
    )
    actual_calls_total = float(actual_calls.sum())
    predicted_calls_total = float(predicted_calls.sum())
    actual_bytes_total = float(actual_bytes.sum())
    predicted_bytes_total = float(predicted_bytes.sum())
    actual_cost = reference_cost(actual_calls, actual_bytes)
    predicted_cost = reference_cost(predicted_calls, predicted_bytes)
    return {
        "training_id": prediction["training_id"],
        "profile_id": prediction["profile_id"],
        "segment": prediction["segment"],
        "parallel_size": prediction["parallel_size"],
        "policy": prediction["policy"],
        "phase": prediction["phase"],
        "method": prediction["method"],
        "actual_total_calls": actual_calls_total,
        "predicted_total_calls": predicted_calls_total,
        "calls_absolute_error": abs(predicted_calls_total - actual_calls_total),
        "calls_ape": abs(predicted_calls_total - actual_calls_total)
        / max(actual_calls_total, 1e-12),
        "actual_total_logical_bytes": actual_bytes_total,
        "predicted_total_logical_bytes": predicted_bytes_total,
        "bytes_absolute_error": abs(predicted_bytes_total - actual_bytes_total),
        "bytes_ape": abs(predicted_bytes_total - actual_bytes_total)
        / max(actual_bytes_total, 1e-12),
        "histogram_l1": 2 * histogram_tv(predicted_calls, actual_calls),
        "histogram_tv": histogram_tv(predicted_calls, actual_calls),
        "normalized_log_payload_emd": normalized_log_emd(predicted_calls, actual_calls),
        "actual_common_reference_cost_us": actual_cost,
        "predicted_common_reference_cost_us": predicted_cost,
        "cost_absolute_error": abs(predicted_cost - actual_cost),
        "cost_ape": abs(predicted_cost - actual_cost) / max(actual_cost, 1e-12),
        "predicted_calls_by_12bin_json": prediction["predicted_calls_by_12bin_json"],
        "actual_calls_by_12bin_json": target["calls_by_12bin_json"],
        "predicted_logical_bytes_by_12bin_json": prediction[
            "predicted_logical_bytes_by_12bin_json"
        ],
        "actual_logical_bytes_by_12bin_json": target[
            "logical_bytes_by_12bin_json"
        ],
    }


def add_total_records(phase_records: list[dict]) -> list[dict]:
    records = list(phase_records)
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in phase_records:
        grouped[
            (
                row["profile_id"],
                row["segment"],
                row["parallel_size"],
                row["policy"],
                row["method"],
            )
        ].append(row)
    for key, rows in grouped.items():
        if len(rows) != 2 or {row["phase"] for row in rows} != set(PHASES):
            raise ValueError(f"missing phase for {key}")
        rows.sort(key=lambda row: row["phase"])
        actual_calls = np.stack(
            [np.asarray(json.loads(row["actual_calls_by_12bin_json"])) for row in rows]
        )
        predicted_calls = np.stack(
            [np.asarray(json.loads(row["predicted_calls_by_12bin_json"])) for row in rows]
        )
        actual_bytes = np.stack(
            [
                np.asarray(json.loads(row["actual_logical_bytes_by_12bin_json"]))
                for row in rows
            ]
        )
        predicted_bytes = np.stack(
            [
                np.asarray(json.loads(row["predicted_logical_bytes_by_12bin_json"]))
                for row in rows
            ]
        )
        actual_calls_total = float(actual_calls.sum())
        predicted_calls_total = float(predicted_calls.sum())
        actual_bytes_total = float(actual_bytes.sum())
        predicted_bytes_total = float(predicted_bytes.sum())
        actual_cost = sum(float(row["actual_common_reference_cost_us"]) for row in rows)
        predicted_cost = sum(float(row["predicted_common_reference_cost_us"]) for row in rows)
        records.append(
            {
                "training_id": rows[0]["training_id"].rsplit("/", 1)[0] + "/total",
                "profile_id": key[0],
                "segment": key[1],
                "parallel_size": key[2],
                "policy": key[3],
                "phase": "total",
                "method": key[4],
                "actual_total_calls": actual_calls_total,
                "predicted_total_calls": predicted_calls_total,
                "calls_absolute_error": abs(predicted_calls_total - actual_calls_total),
                "calls_ape": abs(predicted_calls_total - actual_calls_total)
                / max(actual_calls_total, 1e-12),
                "actual_total_logical_bytes": actual_bytes_total,
                "predicted_total_logical_bytes": predicted_bytes_total,
                "bytes_absolute_error": abs(predicted_bytes_total - actual_bytes_total),
                "bytes_ape": abs(predicted_bytes_total - actual_bytes_total)
                / max(actual_bytes_total, 1e-12),
                "histogram_l1": 2 * histogram_tv(
                    predicted_calls.reshape(-1), actual_calls.reshape(-1)
                ),
                "histogram_tv": histogram_tv(
                    predicted_calls.reshape(-1), actual_calls.reshape(-1)
                ),
                "normalized_log_payload_emd": normalized_log_emd(
                    predicted_calls.sum(axis=0), actual_calls.sum(axis=0)
                ),
                "actual_common_reference_cost_us": actual_cost,
                "predicted_common_reference_cost_us": predicted_cost,
                "cost_absolute_error": abs(predicted_cost - actual_cost),
                "cost_ape": abs(predicted_cost - actual_cost) / max(actual_cost, 1e-12),
                "predicted_calls_by_12bin_json": json.dumps(
                    predicted_calls.tolist(), separators=(",", ":")
                ),
                "actual_calls_by_12bin_json": json.dumps(
                    actual_calls.tolist(), separators=(",", ":")
                ),
                "predicted_logical_bytes_by_12bin_json": json.dumps(
                    predicted_bytes.tolist(), separators=(",", ":")
                ),
                "actual_logical_bytes_by_12bin_json": json.dumps(
                    actual_bytes.tolist(), separators=(",", ":")
                ),
            }
        )
    return records


def aggregate_records(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in records:
        for segment in ("all", row["segment"]):
            for policy in ("all", row["policy"]):
                groups[(row["method"], row["phase"], policy, segment)].append(row)
    result = []
    for (method, phase, policy, segment), rows in sorted(groups.items()):
        actual_calls = sum(float(row["actual_total_calls"]) for row in rows)
        actual_bytes = sum(float(row["actual_total_logical_bytes"]) for row in rows)
        actual_cost = sum(float(row["actual_common_reference_cost_us"]) for row in rows)
        result.append(
            {
                "method": method,
                "phase": phase,
                "policy": policy,
                "segment": segment,
                "cases": len(rows),
                "calls_mape": float(np.mean([float(row["calls_ape"]) for row in rows])),
                "calls_wape": sum(float(row["calls_absolute_error"]) for row in rows)
                / actual_calls,
                "bytes_mape": float(np.mean([float(row["bytes_ape"]) for row in rows])),
                "bytes_wape": sum(float(row["bytes_absolute_error"]) for row in rows)
                / actual_bytes,
                "mean_histogram_l1": float(
                    np.mean([float(row["histogram_l1"]) for row in rows])
                ),
                "mean_histogram_tv": float(
                    np.mean([float(row["histogram_tv"]) for row in rows])
                ),
                "mean_normalized_log_payload_emd": float(
                    np.mean([float(row["normalized_log_payload_emd"]) for row in rows])
                ),
                "common_reference_cost_mape": float(
                    np.mean([float(row["cost_ape"]) for row in rows])
                ),
                "common_reference_cost_wape": sum(
                    float(row["cost_absolute_error"]) for row in rows
                )
                / actual_cost,
            }
        )
    return result


def candidate_records(records: list[dict], decisions: list[dict]) -> list[dict]:
    selected = {row["policy"]: row["selected_method"] for row in decisions}
    return [
        {**row, "candidate_method": selected[row["policy"]]}
        for row in records
        if row["method"] == selected[row["policy"]]
    ]


def compare_methods(metrics: list[dict]) -> list[dict]:
    rows = []
    for policy in ("all", "mb1", "mb4", "mb16"):
        lookup = {
            row["method"]: row
            for row in metrics
            if row["phase"] == "total"
            and row["policy"] == policy
            and row["segment"] == "all"
        }
        h0 = lookup["h0"]
        for method in METHODS[1:]:
            row = lookup[method]
            rows.append(
                {
                    "policy": policy,
                    "method": method,
                    "calls_mape_delta_vs_h0": row["calls_mape"] - h0["calls_mape"],
                    "calls_mape_relative_change": row["calls_mape"] / h0["calls_mape"] - 1,
                    "histogram_tv_delta_vs_h0": row["mean_histogram_tv"]
                    - h0["mean_histogram_tv"],
                    "histogram_tv_relative_change": row["mean_histogram_tv"]
                    / h0["mean_histogram_tv"]
                    - 1,
                    "cost_mape_delta_vs_h0": row["common_reference_cost_mape"]
                    - h0["common_reference_cost_mape"],
                    "cost_mape_relative_change": row["common_reference_cost_mape"]
                    / h0["common_reference_cost_mape"]
                    - 1,
                    "bytes_mape_delta_vs_h0": row["bytes_mape"] - h0["bytes_mape"],
                }
            )
    return rows


def confirmation_decisions(metrics: list[dict], frozen: dict[str, str]) -> list[dict]:
    rows = []
    for policy in ("mb1", "mb4", "mb16"):
        lookup = {
            row["method"]: row
            for row in metrics
            if row["phase"] == "total"
            and row["policy"] == policy
            and row["segment"] == "all"
        }
        h0 = lookup["h0"]
        candidate = lookup[frozen[policy]]
        wins = sum(
            float(candidate[field]) < float(h0[field])
            for field in (
                "calls_mape",
                "mean_histogram_tv",
                "common_reference_cost_mape",
            )
        )
        cost_guard = float(candidate["common_reference_cost_mape"]) <= 1.10 * float(
            h0["common_reference_cost_mape"]
        )
        confirmed = wins >= 2 and cost_guard
        rows.append(
            {
                "policy": policy,
                "frozen_candidate_method": frozen[policy],
                "confirmation_wins_of_calls_tv_cost": wins,
                "confirmation_cost_guard": cost_guard,
                "frozen_candidate_confirmed": confirmed,
                "post_confirmation_recommendation": frozen[policy] if confirmed else "h0",
                "recommendation_status": (
                    "candidate_for_future_independent_validation"
                    if confirmed
                    else "fallback_to_h0"
                ),
                "h0_calls_mape": h0["calls_mape"],
                "candidate_calls_mape": candidate["calls_mape"],
                "h0_histogram_tv": h0["mean_histogram_tv"],
                "candidate_histogram_tv": candidate["mean_histogram_tv"],
                "h0_cost_mape": h0["common_reference_cost_mape"],
                "candidate_cost_mape": candidate["common_reference_cost_mape"],
                "same_confirmation_set_hybrid_score_is_unbiased": False,
            }
        )
    return rows


def plot_confirmation(path: Path, headline: dict[str, dict]) -> None:
    import matplotlib.pyplot as plt

    labels = ("H0", "Legacy residual", "Enhanced residual", "Direct")
    colors = ("#4C78A8", "#A0A0A0", "#F58518", "#B8B8B8")
    metrics = (
        ("calls_mape", "Total calls MAPE", 100.0, "%"),
        ("mean_histogram_tv", "Histogram TV", 1.0, ""),
        ("common_reference_cost_mape", "Common cost MAPE", 100.0, "%"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for axis, (metric, title, scale, suffix) in zip(axes, metrics):
        values = [headline[method][metric] * scale for method in METHODS]
        bars = axis.bar(labels, values, color=colors, width=0.68)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", rotation=18)
        axis.spines[["top", "right"]].set_visible(False)
        upper = max(values) * 1.18 if max(values) > 0 else 1.0
        axis.set_ylim(0, upper)
        for bar, value in zip(bars, values):
            label = f"{value:.1f}{suffix}" if suffix else f"{value:.3f}"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + upper * 0.025,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )
    figure.suptitle("Phase 27D independent confirmation: PP feature comparison", fontsize=14)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def readme(summary: dict) -> str:
    headline = summary["confirmation_headline"]
    table = [
        "| 方法 | calls MAPE/WAPE | bytes MAPE/WAPE | TV | norm EMD | common cost MAPE/WAPE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = headline[method]
        table.append(
            "| {method} | {cm:.2%} / {cw:.2%} | {bm:.2%} / {bw:.2%} | {tv:.4f} | {emd:.4f} | {cost:.2%} / {costw:.2%} |".format(
                method=method,
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
    policy_lines = []
    for policy, result in summary["candidate_policy_headline"].items():
        policy_lines.append(
            f"- {policy}：`{result['method']}`，calls MAPE {result['calls_mape']:.2%}，"
            f"TV {result['mean_histogram_tv']:.4f}，cost MAPE {result['common_reference_cost_mape']:.2%}。"
        )
    decision_lines = []
    for row in summary["post_confirmation_decisions"]:
        decision_lines.append(
            f"- {row['policy']}：冻结候选确认=`{row['frozen_candidate_confirmed']}`；后续建议"
            f" `{row['post_confirmation_recommendation']}`。"
        )
    return f"""# Phase 27D：PP 独立确认集评测

状态：**{summary['status']}**。本阶段没有训练、早停或重新选择方法，只把 Phase 27C 已写入
Git并通过hash冻结的1,296行预测，与Phase 27B的18个独立确认画像Hfull真值做精确join。

## 18个独立确认画像的total结果

{chr(10).join(table)}

## Phase 27C预先冻结候选

{chr(10).join(policy_lines)}

## 确认后的首版建议

{chr(10).join(decision_lines)}

MB1的冻结residual候选只改善TV，calls和cost均退化，因此回退H0；MB4/MB16在calls、TV、
cost三项都改善，保留增强residual候选。但这份5 μs + 100 GB/s确认集已经参与上述建议，
不能在同一数据上计算一个“新混合规则”的无偏总分，建议还需要下一批新窗口确认。

这里的主结论应同时看calls、bytes、TV/EMD和common cost。common cost仍是5 μs +
100 GB/s参考曲线，不是PP P2P物理实测。增强residual相对legacy residual的差异才是新增
chunk/顺序低维画像的主要证据；direct只是控制组。

本阶段可以判断新增调度敏感画像在全新窗口上是否重复改善，也可以判断冻结的分策略候选
是否优于H0。仍不能声称跨模型PP泛化，因为teacher和训练均只有Qwen3-8B；也不能把
fixed-draining结果外推到online arrival-aware调度。
"""


def main() -> None:
    args = parse_args()
    for name in ("analysis", "figures", "logs"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    summary27c = json.loads(args.phase27c_summary.read_text())
    audit27c = json.loads(args.phase27c_audit.read_text())
    if summary27c["status"] != "PASS" or audit27c["status"] != "PASS":
        raise ValueError("Phase 27C is not PASS")
    manifest_checks = verify_manifest(args.phase27c_manifest)
    frozen_hash = audit27c["confirmation_predictions_sha256"]
    if sha256(args.predictions) != frozen_hash:
        raise RuntimeError("frozen prediction hash mismatch")

    predictions = load_rows(args.predictions)
    targets = load_rows(args.targets)
    if len(predictions) != 1296 or len(targets) != 324:
        raise ValueError(f"unexpected row counts: {len(predictions)}, {len(targets)}")
    target_lookup = {row["label_id"]: row for row in targets}
    if len(target_lookup) != len(targets):
        raise ValueError("duplicate target labels")
    phase_records = []
    join_failures = []
    for prediction in predictions:
        target = target_lookup.get(prediction["training_id"])
        if target is None:
            join_failures.append(prediction["training_id"])
            continue
        phase_records.append(case_record(prediction, target))
    records = add_total_records(phase_records)
    metrics = aggregate_records(records)
    decisions = summary27c["candidate_decisions"]
    candidate = candidate_records(records, decisions)
    candidate_metrics = aggregate_records(candidate)
    comparisons = compare_methods(metrics)
    headline = {
        method: next(
            row
            for row in metrics
            if row["method"] == method
            and row["phase"] == "total"
            and row["policy"] == "all"
            and row["segment"] == "all"
        )
        for method in METHODS
    }
    candidate_policy_headline = {
        policy: next(
            row
            for row in candidate_metrics
            if row["phase"] == "total"
            and row["policy"] == policy
            and row["segment"] == "all"
        )
        for policy in ("mb1", "mb4", "mb16")
    }
    frozen_mapping = {
        row["policy"]: row["selected_method"] for row in decisions
    }
    post_confirmation = confirmation_decisions(metrics, frozen_mapping)

    write_csv_gz(args.output_dir / "analysis/confirmation_predictions_and_errors.csv.gz", records)
    write_csv(args.output_dir / "analysis/confirmation_metrics.csv", metrics)
    write_csv(args.output_dir / "analysis/method_delta_vs_h0.csv", comparisons)
    write_csv(args.output_dir / "analysis/frozen_candidate_metrics.csv", candidate_metrics)
    write_csv(
        args.output_dir / "analysis/post_confirmation_decisions.csv",
        post_confirmation,
    )
    plot_confirmation(args.output_dir / "figures/independent_confirmation_comparison.png", headline)

    checks = {
        "phase27c_status_pass": summary27c["status"] == "PASS",
        "phase27c_manifest_all_pass": len(manifest_checks) == 16
        and all(manifest_checks.values()),
        "prediction_hash_matches_frozen_audit": sha256(args.predictions) == frozen_hash,
        "predictions_1296": len(predictions) == 1296,
        "targets_324": len(targets) == 324,
        "join_1296_of_1296": len(phase_records) == 1296 and not join_failures,
        "phase_plus_total_records_1944": len(records) == 1296 + 648,
        "confirmation_profiles_18": len({row["profile_id"] for row in phase_records}) == 18,
        "methods_four": Counter(row["method"] for row in phase_records)
        == Counter({method: 324 for method in METHODS}),
        "frozen_candidate_mapping_unchanged": frozen_mapping
        == {
            "mb1": "enhanced_bounded_residual",
            "mb4": "enhanced_bounded_residual",
            "mb16": "enhanced_bounded_residual",
        },
        "all_metrics_finite": all(
            math.isfinite(float(row[field]))
            for row in metrics
            for field in (
                "calls_mape",
                "calls_wape",
                "bytes_mape",
                "bytes_wape",
                "mean_histogram_tv",
                "mean_normalized_log_payload_emd",
                "common_reference_cost_mape",
            )
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    if status != "PASS":
        raise RuntimeError(checks)

    summary = {
        "schema_version": "phase27d-pp-independent-confirmation-v1",
        "status": status,
        "objective": "evaluate frozen legacy/enhanced PP predictors and the validation-selected candidate mapping on 18 unseen windows",
        "counts": {
            "independent_confirmation_profiles": 18,
            "target_phase_rows": len(targets),
            "frozen_prediction_rows": len(predictions),
            "phase_error_records": len(phase_records),
            "phase_plus_total_records": len(records),
        },
        "inputs": {
            "frozen_predictions_sha256": sha256(args.predictions),
            "confirmation_targets_sha256": sha256(args.targets),
            "phase27c_summary_sha256": sha256(args.phase27c_summary),
            "phase27c_audit_sha256": sha256(args.phase27c_audit),
            "phase27c_manifest_sha256": sha256(args.phase27c_manifest),
        },
        "confirmation_headline": headline,
        "candidate_policy_headline": candidate_policy_headline,
        "frozen_candidate_mapping": frozen_mapping,
        "post_confirmation_decisions": post_confirmation,
        "post_confirmation_mapping_is_unbiased_on_this_set": False,
        "checks": checks,
        "can_conclude": [
            "whether scheduler-sensitive low-dimensional PP features repeat their development gains on 18 new windows",
            "whether the validation-frozen per-policy candidate mapping improves over H0 on independent confirmation windows",
        ],
        "cannot_conclude": [
            "cross-model PP generalization",
            "physical PP P2P communication-time accuracy from the common reference curve",
            "online arrival-aware scheduling behavior",
        ],
        "next_step": "use H0 for MB1 and keep enhanced residual as the MB4/MB16 candidate, then validate this post-confirmation mapping on another untouched window set",
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(
        args.output_dir / "audit_summary.json",
        {
            "schema_version": "phase27d-pp-independent-confirmation-audit-v1",
            "status": status,
            "checks": checks,
            "frozen_prediction_sha256": frozen_hash,
            "join_failures": join_failures,
        },
    )
    (args.output_dir / "README.md").write_text(readme(summary))
    (args.output_dir / "DONE").write_text("PASS\n")
    write_json(
        args.output_dir / "logs/evaluation.log",
        {
            "schema_version": "phase27d-evaluation-log-v1",
            "status": status,
            "prediction_hash_verified": True,
            "phase27c_manifest_checks": manifest_checks,
            "training_performed": False,
            "model_selection_changed": False,
            "joined_phase_rows": len(phase_records),
        },
    )
    manifest = []
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.sha256":
            manifest.append(f"{sha256(path)}  {path.relative_to(args.output_dir)}")
    (args.output_dir / "manifest.sha256").write_text("\n".join(manifest) + "\n")
    print(
        json.dumps(
            {
                "status": status,
                "profiles": 18,
                "headline": {
                    method: {
                        "calls_mape": headline[method]["calls_mape"],
                        "histogram_tv": headline[method]["mean_histogram_tv"],
                        "cost_mape": headline[method]["common_reference_cost_mape"],
                    }
                    for method in METHODS
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
