#!/usr/bin/env python3
"""Evaluate frozen Phase 29C TP predictions on the first confirmation targets."""

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
POLICIES = ("latency", "balanced", "throughput")
BIN_EDGES = np.geomspace(4 * 1024, 512 * 1024 * 1024, 13).tolist()
COMMON_REFERENCE_LAUNCH_US = 5.0
COMMON_REFERENCE_BANDWIDTH_GBPS = 100.0


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    training = root / "experiment-results/phase29c_tp_aligned_training"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=training / "analysis/first_confirmation_predictions.csv.gz",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=root
        / "experiment-results/phase29b_tp_hfull_dataset/labels/first_confirmation_hfull_targets.csv.gz",
    )
    parser.add_argument(
        "--phase29c-summary", type=Path, default=training / "summary.json"
    )
    parser.add_argument(
        "--phase29c-audit", type=Path, default=training / "audit_summary.json"
    )
    parser.add_argument(
        "--phase29c-manifest", type=Path, default=training / "manifest.sha256"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase29d1_tp_first_confirmation",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as output:
            output.write(buffer.getvalue().encode())


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def verify_manifest(path: Path) -> dict[str, bool]:
    root = path.parent
    checks = {}
    for line in path.read_text().splitlines():
        if line.strip():
            expected, relative = line.split(maxsplit=1)
            checks[relative] = sha256(root / relative) == expected
    return checks


def histogram_tv(predicted: np.ndarray, actual: np.ndarray) -> float:
    predicted_total = max(float(predicted.sum()), 1e-12)
    actual_total = max(float(actual.sum()), 1e-12)
    return float(
        np.abs(predicted / predicted_total - actual / actual_total).sum() / 2
    )


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


def error_values(
    actual_calls: np.ndarray,
    predicted_calls: np.ndarray,
    actual_bytes: np.ndarray,
    predicted_bytes: np.ndarray,
) -> dict:
    actual_calls_total = float(actual_calls.sum())
    predicted_calls_total = float(predicted_calls.sum())
    actual_bytes_total = float(actual_bytes.sum())
    predicted_bytes_total = float(predicted_bytes.sum())
    actual_cost = reference_cost(actual_calls, actual_bytes)
    predicted_cost = reference_cost(predicted_calls, predicted_bytes)
    return {
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
        "normalized_log_payload_emd": normalized_log_emd(
            predicted_calls.reshape(-1)[-12:], actual_calls.reshape(-1)[-12:]
        )
        if predicted_calls.ndim == 1
        else normalized_log_emd(predicted_calls.sum(axis=0), actual_calls.sum(axis=0)),
        "actual_common_reference_cost_us": actual_cost,
        "predicted_common_reference_cost_us": predicted_cost,
        "cost_absolute_error": abs(predicted_cost - actual_cost),
        "cost_ape": abs(predicted_cost - actual_cost) / max(actual_cost, 1e-12),
    }


def phase_record(prediction: dict[str, str], target: dict[str, str]) -> dict:
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
    return {
        "training_id": prediction["training_id"],
        "profile_id": prediction["profile_id"],
        "source": prediction["source"],
        "segment": prediction["segment"],
        "model": prediction["model"],
        "parallel_size": prediction["parallel_size"],
        "policy": prediction["policy"],
        "phase": prediction["phase"],
        "method": prediction["method"],
        **error_values(actual_calls, predicted_calls, actual_bytes, predicted_bytes),
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
    output = list(phase_records)
    grouped: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in phase_records:
        grouped[
            (
                row["profile_id"],
                row["source"],
                row["segment"],
                row["model"],
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
        output.append(
            {
                "training_id": rows[0]["training_id"].rsplit("/", 1)[0]
                + "/total",
                "profile_id": key[0],
                "source": key[1],
                "segment": key[2],
                "model": key[3],
                "parallel_size": key[4],
                "policy": key[5],
                "phase": "total",
                "method": key[6],
                **error_values(
                    actual_calls,
                    predicted_calls,
                    actual_bytes,
                    predicted_bytes,
                ),
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
    return output


def aggregate_records(records: list[dict]) -> list[dict]:
    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in records:
        for model in ("all", row["model"]):
            for parallel_size in ("all", row["parallel_size"]):
                for policy in ("all", row["policy"]):
                    for segment in ("all", row["segment"]):
                        groups[
                            (
                                row["method"],
                                row["phase"],
                                model,
                                parallel_size,
                                policy,
                                segment,
                            )
                        ].append(row)
    output = []
    for key, rows in sorted(groups.items()):
        method, phase, model, parallel_size, policy, segment = key
        actual_calls = sum(float(row["actual_total_calls"]) for row in rows)
        actual_bytes = sum(float(row["actual_total_logical_bytes"]) for row in rows)
        actual_cost = sum(float(row["actual_common_reference_cost_us"]) for row in rows)
        output.append(
            {
                "method": method,
                "phase": phase,
                "model": model,
                "parallel_size": parallel_size,
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
    return output


def candidate_records(records: list[dict], mapping: dict[str, str]) -> list[dict]:
    return [
        {**row, "candidate_method": mapping[row["policy"]]}
        for row in records
        if row["method"] == mapping[row["policy"]]
    ]


def confirmation_decisions(metrics: list[dict], mapping: dict[str, str]) -> list[dict]:
    output = []
    for policy in POLICIES:
        lookup = {
            row["method"]: row
            for row in metrics
            if row["phase"] == "total"
            and row["model"] == "all"
            and row["parallel_size"] == "all"
            and row["policy"] == policy
            and row["segment"] == "all"
        }
        h0 = lookup["h0"]
        candidate = lookup[mapping[policy]]
        fields = (
            "calls_mape",
            "mean_histogram_tv",
            "common_reference_cost_mape",
        )
        wins = sum(float(candidate[field]) < float(h0[field]) for field in fields)
        cost_guard = float(candidate["common_reference_cost_mape"]) <= 1.10 * float(
            h0["common_reference_cost_mape"]
        )
        confirmed = wins >= 2 and cost_guard
        output.append(
            {
                "policy": policy,
                "validation_frozen_candidate": mapping[policy],
                "first_confirmation_wins_of_calls_tv_cost": wins,
                "first_confirmation_cost_guard": cost_guard,
                "validation_candidate_confirmed": confirmed,
                "second_confirmation_frozen_recommendation": mapping[policy]
                if confirmed
                else "h0",
                "h0_calls_mape": h0["calls_mape"],
                "candidate_calls_mape": candidate["calls_mape"],
                "h0_histogram_tv": h0["mean_histogram_tv"],
                "candidate_histogram_tv": candidate["mean_histogram_tv"],
                "h0_cost_mape": h0["common_reference_cost_mape"],
                "candidate_cost_mape": candidate["common_reference_cost_mape"],
                "first_confirmation_recommendation_is_unbiased_on_first_set": False,
                "eligible_for_unbiased_second_confirmation": True,
            }
        )
    return output


def plot_confirmation(path: Path, headline: dict[str, dict]) -> None:
    import matplotlib.pyplot as plt

    labels = ("H0", "Legacy residual", "Enhanced residual", "Direct")
    colors = ("#4C78A8", "#A0A0A0", "#F58518", "#B8B8B8")
    specs = (
        ("calls_mape", "Total calls MAPE", 100.0, "%"),
        ("mean_histogram_tv", "Histogram TV", 1.0, ""),
        ("common_reference_cost_mape", "Common cost MAPE", 100.0, "%"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for axis, (metric, title, scale, suffix) in zip(axes, specs):
        values = [headline[method][metric] * scale for method in METHODS]
        bars = axis.bar(labels, values, color=colors, width=0.68)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22, linewidth=0.8)
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
    figure.suptitle("Phase 29D1: TP first independent confirmation")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def readme(summary: dict) -> str:
    table = [
        "| 方法 | calls MAPE/WAPE | bytes MAPE/WAPE | TV | norm EMD | cost MAPE/WAPE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = summary["confirmation_headline"][method]
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
    decisions = "\n".join(
        f"- {row['policy']}：验证集候选 `{row['validation_frozen_candidate']}`，第一确认通过="
        f"`{row['validation_candidate_confirmed']}`，第二确认冻结建议 `"
        f"{row['second_confirmation_frozen_recommendation']}`。"
        for row in summary["post_first_confirmation_decisions"]
    )
    return f"""# Phase 29D1：TP第一独立确认

状态：**{summary['status']}**。本阶段没有训练、早停或重写预测，只把Phase 29C已写入Git并
通过hash冻结的3,888条预测，与Phase 29B物理隔离的972条第一确认Hfull真值精确连接。

## 18个独立确认画像的total结果

{chr(10).join(table)}

## 面向第二确认的冻结建议

{decisions}

第一确认可以无偏检验开发验证阶段冻结的候选；根据第一确认产生的新建议不能再在第一确认
上声称无偏，因此只用于尚未生成真值的第二独立确认。第二批的四方法预测已在Phase 29C同期
冻结，下一阶段不得重新训练或改写预测。

这里的cost使用统一5 μs + 100 GB/s参考曲线，不是具体拓扑的物理实测；当前结论只适用于
fixed-draining与已冻结TP配置/策略，不能外推到online arrival-aware。
"""


def main() -> None:
    args = parse_args()
    for name in ("analysis", "figures", "logs"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    training_summary = json.loads(args.phase29c_summary.read_text())
    training_audit = json.loads(args.phase29c_audit.read_text())
    if training_summary["status"] != "PASS" or training_audit["status"] != "PASS":
        raise ValueError("Phase 29C is not PASS")
    manifest_checks = verify_manifest(args.phase29c_manifest)
    frozen_hash = training_audit["first_confirmation_predictions_sha256"]
    if sha256(args.predictions) != frozen_hash:
        raise RuntimeError("frozen first-confirmation prediction hash mismatch")

    predictions = load_rows(args.predictions)
    targets = load_rows(args.targets)
    if len(predictions) != 3888 or len(targets) != 972:
        raise ValueError("unexpected first-confirmation row counts")
    target_lookup = {row["label_id"]: row for row in targets}
    if len(target_lookup) != len(targets):
        raise ValueError("duplicate targets")
    join_failures = []
    phase_records = []
    for prediction in predictions:
        target = target_lookup.get(prediction["training_id"])
        if target is None:
            join_failures.append(prediction["training_id"])
        else:
            phase_records.append(phase_record(prediction, target))
    records = add_total_records(phase_records)
    metrics = aggregate_records(records)
    mapping = {
        row["policy"]: row["selected_method"]
        for row in training_summary["candidate_decisions"]
    }
    candidate = candidate_records(records, mapping)
    candidate_metrics = aggregate_records(candidate)
    decisions = confirmation_decisions(metrics, mapping)
    headline = {
        method: next(
            row
            for row in metrics
            if row["method"] == method
            and row["phase"] == "total"
            and row["model"] == "all"
            and row["parallel_size"] == "all"
            and row["policy"] == "all"
            and row["segment"] == "all"
        )
        for method in METHODS
    }

    write_csv_gz(
        args.output_dir / "analysis/first_confirmation_predictions_and_errors.csv.gz",
        records,
    )
    write_csv(args.output_dir / "analysis/first_confirmation_metrics.csv", metrics)
    write_csv(
        args.output_dir / "analysis/validation_frozen_candidate_metrics.csv",
        candidate_metrics,
    )
    write_csv(
        args.output_dir / "analysis/post_first_confirmation_decisions.csv",
        decisions,
    )
    plot_confirmation(
        args.output_dir / "figures/first_confirmation_comparison.png", headline
    )

    checks = {
        "phase29c_status_pass": training_summary["status"] == "PASS",
        "phase29c_manifest_17_of_17": len(manifest_checks) == 17
        and all(manifest_checks.values()),
        "prediction_hash_matches_frozen_audit": sha256(args.predictions)
        == frozen_hash,
        "predictions_3888_targets_972": len(predictions) == 3888
        and len(targets) == 972,
        "join_3888_of_3888": len(phase_records) == 3888 and not join_failures,
        "phase_plus_total_records_5832": len(records) == 5832,
        "confirmation_profiles_18": len(
            {row["profile_id"] for row in phase_records}
        )
        == 18,
        "three_models_three_tp_sizes_three_policies": len(
            {row["model"] for row in phase_records}
        )
        == 3
        and {int(row["parallel_size"]) for row in phase_records} == {2, 4, 8}
        and {row["policy"] for row in phase_records} == set(POLICIES),
        "methods_four_balanced": Counter(row["method"] for row in phase_records)
        == Counter({method: 972 for method in METHODS}),
        "frozen_mapping_three_policies": set(mapping) == set(POLICIES),
        "second_confirmation_predictions_already_frozen": training_audit[
            "second_confirmation_predictions_sha256"
        ]
        == sha256(
            args.phase29c_summary.parent
            / "analysis/second_confirmation_predictions.csv.gz"
        ),
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

    second_mapping = {
        row["policy"]: row["second_confirmation_frozen_recommendation"]
        for row in decisions
    }
    summary = {
        "schema_version": "phase29d1-tp-first-confirmation-v1",
        "status": status,
        "objective": "evaluate validation-frozen TP residual candidates on 18 first independent windows and freeze a mapping for a second independent confirmation",
        "counts": {
            "profiles": 18,
            "models": 3,
            "tp_sizes": 3,
            "policies": 3,
            "target_phase_rows": len(targets),
            "frozen_prediction_rows": len(predictions),
            "phase_error_records": len(phase_records),
            "phase_plus_total_records": len(records),
        },
        "inputs": {
            "frozen_predictions_sha256": sha256(args.predictions),
            "targets_sha256": sha256(args.targets),
            "phase29c_summary_sha256": sha256(args.phase29c_summary),
            "phase29c_audit_sha256": sha256(args.phase29c_audit),
            "phase29c_manifest_sha256": sha256(args.phase29c_manifest),
        },
        "confirmation_headline": headline,
        "validation_frozen_mapping": mapping,
        "post_first_confirmation_decisions": decisions,
        "second_confirmation_frozen_mapping": second_mapping,
        "second_confirmation_predictions_sha256": training_audit[
            "second_confirmation_predictions_sha256"
        ],
        "post_first_mapping_is_unbiased_on_first_confirmation": False,
        "post_first_mapping_is_eligible_for_unbiased_second_confirmation": True,
        "checks": checks,
        "can_conclude": [
            "whether validation-frozen TP residual candidates improve over H0 on the first independent windows",
            "which mapping may be carried forward to a distinct second confirmation set",
        ],
        "cannot_conclude": [
            "that the post-first mapping is unbiased on the first confirmation set",
            "final acceptance before the second independent confirmation",
            "physical topology-specific cost accuracy from the common reference curve",
        ],
        "next_step": "archive this evaluation, generate second-confirmation Hfull targets without changing frozen predictions, and evaluate the frozen post-first mapping",
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(
        args.output_dir / "audit_summary.json",
        {
            "schema_version": "phase29d1-tp-first-confirmation-audit-v1",
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
            "schema_version": "phase29d1-evaluation-log-v1",
            "status": status,
            "prediction_hash_verified": True,
            "phase29c_manifest_checks": manifest_checks,
            "training_performed": False,
            "predictions_rewritten": False,
            "joined_phase_rows": len(phase_records),
            "second_confirmation_targets_read": False,
        },
    )
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
    print(
        json.dumps(
            {
                "status": status,
                "headline": {
                    method: {
                        "calls_mape": headline[method]["calls_mape"],
                        "histogram_tv": headline[method]["mean_histogram_tv"],
                        "cost_mape": headline[method]["common_reference_cost_mape"],
                    }
                    for method in METHODS
                },
                "second_confirmation_frozen_mapping": second_mapping,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
