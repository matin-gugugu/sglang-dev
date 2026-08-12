#!/usr/bin/env python3
"""Evaluate frozen Phase 26C checkpoints on untouched profile-level test domains."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from train_phase26c_hfull_predictors import (
    METHODS,
    MLP,
    PARALLELISMS,
    TEST_SPLITS,
    case_record,
    common_reference_cost,
    histogram_tv,
    load_rows,
    normalized_log_emd,
    prepare_arrays,
    target_decode,
    transform_feature,
)


ALL_TEST_SCOPE = "all_test"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root
        / "experiment-results/phase26b_unified_hfull_training_dataset/training_examples.csv.gz",
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=root / "experiment-results/phase26c_hfull_predictor_training",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase26d_hfull_profile_holdout_evaluation",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def predict_checkpoint(
    checkpoint: dict,
    rows: list[dict[str, str]],
    arrays: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    feature_names = checkpoint["feature_names"]
    features = np.asarray(
        [[transform_feature(name, row[name]) for name in feature_names] for row in rows],
        dtype=np.float32,
    )
    feature_mean = checkpoint["feature_mean"].numpy()
    feature_std = checkpoint["feature_std"].numpy()
    scaled = np.clip((features - feature_mean) / feature_std, -6.0, 6.0).astype(np.float32)
    bounded = checkpoint["method"] == "h0_bounded_residual"
    model = MLP(len(feature_names), 26, bounded=bounded).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        raw = model(torch.from_numpy(scaled).to(device)).cpu().numpy()
    target_mean = checkpoint["target_mean"].numpy()
    target_scale = checkpoint["target_std_or_residual_bounds"].numpy()
    if bounded:
        encoded = arrays["h0_encoded"] + raw * target_scale
    else:
        encoded = np.clip(raw, -6.0, 6.0) * target_scale + target_mean
    calls, logical_bytes = zip(*(target_decode(row) for row in encoded))
    return np.stack(calls), np.stack(logical_bytes)


def evaluation_records(
    rows: list[dict[str, str]],
    arrays: dict[str, np.ndarray],
    predicted: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[dict]:
    test_indices = [index for index, row in enumerate(rows) if row["split"] in TEST_SPLITS]
    grouped: dict[tuple, list[int]] = defaultdict(list)
    for index in test_indices:
        row = rows[index]
        grouped[
            (
                row["model"],
                row["parallel_size"],
                row["policy"],
                row["profile_id"],
            )
        ].append(index)
    edges = json.loads(rows[0]["bin_edges_bytes_json"])
    records = []
    for method in METHODS:
        predicted_calls, predicted_bytes = predicted[method]
        for indices in grouped.values():
            if len(indices) != 2 or {rows[index]["phase"] for index in indices} != {
                "prefill",
                "decode",
            }:
                raise ValueError("test configuration lacks exactly two phases")
            indices = sorted(indices, key=lambda index: rows[index]["phase"])
            for index in indices:
                records.append(
                    case_record(
                        row=rows[index],
                        method=method,
                        phase=rows[index]["phase"],
                        actual_calls=arrays["target_calls"][index],
                        actual_bytes=arrays["target_bytes"][index],
                        predicted_calls=predicted_calls[index],
                        predicted_bytes=predicted_bytes[index],
                        edges=edges,
                    )
                )
            representative = rows[indices[0]]
            pooled_actual_calls = sum((arrays["target_calls"][index] for index in indices))
            pooled_actual_bytes = sum((arrays["target_bytes"][index] for index in indices))
            pooled_predicted_calls = sum((predicted_calls[index] for index in indices))
            pooled_predicted_bytes = sum((predicted_bytes[index] for index in indices))
            phase_aware_actual_calls = np.concatenate(
                [arrays["target_calls"][index] for index in indices]
            )
            phase_aware_predicted_calls = np.concatenate(
                [predicted_calls[index] for index in indices]
            )
            total_record = case_record(
                row=representative,
                method=method,
                phase="total",
                actual_calls=pooled_actual_calls,
                actual_bytes=pooled_actual_bytes,
                predicted_calls=pooled_predicted_calls,
                predicted_bytes=pooled_predicted_bytes,
                edges=edges,
            )
            total_record["histogram_tv"] = histogram_tv(
                phase_aware_predicted_calls, phase_aware_actual_calls
            )
            total_record["histogram_l1"] = 2 * total_record["histogram_tv"]
            actual_cost = sum(
                common_reference_cost(
                    arrays["target_calls"][index], arrays["target_bytes"][index], edges
                )
                for index in indices
            )
            predicted_cost = sum(
                common_reference_cost(predicted_calls[index], predicted_bytes[index], edges)
                for index in indices
            )
            total_record["actual_common_reference_cost_us"] = actual_cost
            total_record["predicted_common_reference_cost_us"] = predicted_cost
            total_record["cost_ape"] = abs(predicted_cost - actual_cost) / max(
                actual_cost, 1e-12
            )
            records.append(total_record)
    return records


def aggregate_values(values: list[dict]) -> dict:
    actual_calls = sum(float(row["actual_total_calls"]) for row in values)
    actual_bytes = sum(float(row["actual_total_logical_bytes"]) for row in values)
    actual_cost = sum(float(row["actual_common_reference_cost_us"]) for row in values)
    return {
        "cases": len(values),
        "calls_mape": float(np.mean([float(row["calls_ape"]) for row in values])),
        "calls_wape": sum(float(row["calls_absolute_error"]) for row in values)
        / actual_calls,
        "bytes_mape": float(np.mean([float(row["bytes_ape"]) for row in values])),
        "bytes_wape": sum(float(row["bytes_absolute_error"]) for row in values)
        / actual_bytes,
        "mean_histogram_l1": float(
            np.mean([float(row["histogram_l1"]) for row in values])
        ),
        "mean_histogram_tv": float(
            np.mean([float(row["histogram_tv"]) for row in values])
        ),
        "mean_normalized_log_payload_emd": float(
            np.mean([float(row["normalized_log_payload_emd"]) for row in values])
        ),
        "common_reference_cost_mape": float(
            np.mean([float(row["cost_ape"]) for row in values])
        ),
        "common_reference_cost_wape": sum(
            abs(
                float(row["predicted_common_reference_cost_us"])
                - float(row["actual_common_reference_cost_us"])
            )
            for row in values
        )
        / actual_cost,
    }


def aggregate_records(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in records:
        for scope in (ALL_TEST_SCOPE, row["split"]):
            groups[
                (scope, row["parallelism"], row["method"], row["phase"], "all")
            ].append(row)
            groups[
                (
                    scope,
                    row["parallelism"],
                    row["method"],
                    row["phase"],
                    row["policy"],
                )
            ].append(row)
    return [
        {
            "evaluation_scope": key[0],
            "parallelism": key[1],
            "method": key[2],
            "phase": key[3],
            "policy": key[4],
            **aggregate_values(values),
        }
        for key, values in sorted(groups.items())
    ]


def headline_lookup(metrics: list[dict]) -> dict:
    scopes = (ALL_TEST_SCOPE, *TEST_SPLITS)
    return {
        scope: {
            parallelism: {
                method: next(
                    row
                    for row in metrics
                    if row["evaluation_scope"] == scope
                    and row["parallelism"] == parallelism
                    and row["method"] == method
                    and row["phase"] == "total"
                    and row["policy"] == "all"
                )
                for method in METHODS
            }
            for parallelism in PARALLELISMS
        }
        for scope in scopes
    }


def comparison_rows(headline: dict) -> list[dict]:
    fields = (
        "calls_mape",
        "calls_wape",
        "bytes_mape",
        "bytes_wape",
        "mean_histogram_tv",
        "mean_normalized_log_payload_emd",
        "common_reference_cost_mape",
        "common_reference_cost_wape",
    )
    rows = []
    for scope, by_parallelism in headline.items():
        for parallelism, by_method in by_parallelism.items():
            h0 = by_method["h0"]
            residual = by_method["h0_bounded_residual"]
            row = {
                "evaluation_scope": scope,
                "parallelism": parallelism,
                "cases": h0["cases"],
            }
            for field in fields:
                row[f"h0_{field}"] = h0[field]
                row[f"residual_{field}"] = residual[field]
                row[f"residual_minus_h0_{field}"] = residual[field] - h0[field]
                row[f"residual_relative_change_{field}"] = (
                    residual[field] - h0[field]
                ) / max(h0[field], 1e-12)
            rows.append(row)
    return rows


def plot_calls_mape(path: Path, headline: dict) -> None:
    import matplotlib.pyplot as plt

    scopes = TEST_SPLITS
    scope_labels = {
        "temporal_test": "Temporal",
        "external_test": "External",
        "external_synthetic": "Synthetic",
    }
    method_order = ("h0", "h0_bounded_residual", "direct")
    method_labels = ("H0", "H0 + bounded residual", "Direct")
    colors = ("#4C78A8", "#F58518", "#9D9D9D")
    figure, axes = plt.subplots(2, 3, figsize=(14, 7.5), constrained_layout=True)
    for row_index, parallelism in enumerate(PARALLELISMS):
        for column_index, scope in enumerate(scopes):
            axis = axes[row_index, column_index]
            values = [
                100 * headline[scope][parallelism][method]["calls_mape"]
                for method in method_order
            ]
            bars = axis.bar(method_labels, values, color=colors, width=0.68)
            axis.set_title(
                f"{parallelism.upper()} · {scope_labels[scope]} · calls MAPE"
            )
            axis.grid(axis="y", alpha=0.22, linewidth=0.8)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(axis="x", rotation=18)
            upper = max(values) * 1.18 if max(values) > 0 else 1.0
            axis.set_ylim(0, upper)
            for bar, value in zip(bars, values):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + upper * 0.025,
                    f"{value:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
    figure.suptitle("Phase 26D untouched profile holdouts", fontsize=15)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def readme(summary: dict) -> str:
    headline = summary["test_headline"]
    lines = [
        "| 测试域 | 并行 | 方法 | calls MAPE/WAPE | bytes MAPE/WAPE | TV | norm EMD | common cost MAPE |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    scope_labels = {
        ALL_TEST_SCOPE: "全部测试画像",
        "temporal_test": "Temporal",
        "external_test": "External",
        "external_synthetic": "Synthetic",
    }
    for scope in (ALL_TEST_SCOPE, *TEST_SPLITS):
        for parallelism in PARALLELISMS:
            for method in METHODS:
                row = headline[scope][parallelism][method]
                lines.append(
                    "| {scope} | {parallelism} | {method} | {calls_mape:.2%} / {calls_wape:.2%} | "
                    "{bytes_mape:.2%} / {bytes_wape:.2%} | {tv:.4f} | {emd:.4f} | {cost:.2%} |".format(
                        scope=scope_labels[scope],
                        parallelism=parallelism.upper(),
                        method=method,
                        calls_mape=row["calls_mape"],
                        calls_wape=row["calls_wape"],
                        bytes_mape=row["bytes_mape"],
                        bytes_wape=row["bytes_wape"],
                        tv=row["mean_histogram_tv"],
                        emd=row["mean_normalized_log_payload_emd"],
                        cost=row["common_reference_cost_mape"],
                    )
                )
    return f"""# Phase 26D：Hfull画像级留出评测

状态：**{summary['status']}**。

本阶段冻结Phase 26C四个checkpoint，在此前未用于拟合、标准化或早停的14个画像上做
正式测试：5个temporal、8个external和1个external synthetic。评测H0、structure-direct
和H0+bounded residual，并分别报告TP/PP、policy和phase。

## 配置级total核心结果

{chr(10).join(lines)}

Synthetic只有1个画像，保留为外部极端哨兵，不能单独支撑统计泛化结论。方法判断应重点看
Temporal、External及全部测试画像，并同时看calls、bytes、TV、EMD和cost，而不是只挑一个
改善数字。

## 口径

- calls/bytes均按每1000请求归一化；
- L1/TV使用各自原生12桶，total保留prefill/decode的24维phase-aware分布；
- normalized log-payload EMD在total时合并phase后计算payload质量迁移；
- common cost使用5 μs启动项+100 GB/s参数曲线，不是PP物理链路测量；
- TP与PP的桶schema继续分离。

## 资产

- `analysis/test_predictions.csv.gz`：逐配置、逐phase/total预测；
- `analysis/test_metrics.csv`：按测试域、并行、方法、phase和policy聚合；
- `analysis/residual_vs_h0.csv`：bounded residual相对H0的逐指标变化；
- `figures/test_domain_calls_mape.png`：三测试域TP/PP calls MAPE；
- `contract.json`、`summary.json`、`audit_summary.json`、`logs/evaluation.log`、`DONE`
  和`manifest.sha256`。

可以据此判断冻结模型在这些画像域的实际泛化表现。不能外推到online arrival-aware、其他
scheduler契约或未观测模型结构，也不能把Synthetic单画像结果解释为总体分布。
"""


def main() -> None:
    args = parse_args()
    for directory in (
        args.output_dir / "analysis",
        args.output_dir / "figures",
        args.output_dir / "logs",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    all_rows = load_rows(args.dataset)
    training_summary = json.loads((args.training_root / "summary.json").read_text())
    training_audit = json.loads((args.training_root / "audit_summary.json").read_text())
    if training_summary["status"] != "PASS" or training_audit["status"] != "PASS":
        raise ValueError("Phase 26C training artifacts are not PASS")
    expected_checkpoint_hashes = training_audit["checkpoint_sha256"]

    records = []
    checkpoint_hashes = {}
    for parallelism in PARALLELISMS:
        rows = [row for row in all_rows if row["parallelism"] == parallelism]
        feature_names = [name for name in rows[0] if name.startswith("feature_")]
        arrays = prepare_arrays(rows, feature_names)
        predicted: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "h0": (arrays["h0_calls"], arrays["h0_bytes"])
        }
        for method in ("direct", "h0_bounded_residual"):
            path = args.training_root / "checkpoints" / f"{parallelism}_{method}.pt"
            actual_hash = sha256(path)
            checkpoint_hashes[f"{parallelism}_{method}"] = actual_hash
            if actual_hash != expected_checkpoint_hashes[f"{parallelism}_{method}"]:
                raise ValueError(f"checkpoint hash mismatch: {path}")
            checkpoint = load_checkpoint(path)
            if checkpoint["parallelism"] != parallelism or checkpoint["method"] != method:
                raise ValueError(f"checkpoint contract mismatch: {path}")
            predicted[method] = predict_checkpoint(checkpoint, rows, arrays, device)
        records.extend(evaluation_records(rows, arrays, predicted))

    metrics = aggregate_records(records)
    headline = headline_lookup(metrics)
    comparisons = comparison_rows(headline)
    write_csv_gz(args.output_dir / "analysis/test_predictions.csv.gz", records)
    write_csv(args.output_dir / "analysis/test_metrics.csv", metrics)
    write_csv(args.output_dir / "analysis/residual_vs_h0.csv", comparisons)
    plot_calls_mape(args.output_dir / "figures/test_domain_calls_mape.png", headline)

    test_profiles = {
        row["profile_id"]: row["split"]
        for row in all_rows
        if row["split"] in TEST_SPLITS
    }
    split_profiles = Counter(test_profiles.values())
    checks = {
        "training_status_pass": training_summary["status"] == "PASS",
        "training_audit_pass": training_audit["status"] == "PASS",
        "checkpoint_hashes_frozen": checkpoint_hashes == expected_checkpoint_hashes,
        "test_profiles_14": len(test_profiles) == 14,
        "test_profile_split_counts_5_8_1": split_profiles
        == Counter({"temporal_test": 5, "external_test": 8, "external_synthetic": 1}),
        "predictions_only_test_splits": all(row["split"] in TEST_SPLITS for row in records),
        "no_train_or_validation_predictions": not any(
            row["split"] in {"train", "validation"} for row in records
        ),
        "prediction_records_4536": len(records) == 4536,
        "headline_scopes_complete": set(headline)
        == {ALL_TEST_SCOPE, "temporal_test", "external_test", "external_synthetic"},
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
        "tp_pp_bin_schemas_separate": training_summary["parallelism"]["tp"][
            "bin_schema_id"
        ]
        != training_summary["parallelism"]["pp"]["bin_schema_id"],
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    if status != "PASS":
        raise RuntimeError(f"Phase 26D checks failed: {checks}")

    residual_outcome = {
        scope: {
            parallelism: {
                "calls_mape_relative_change": (
                    headline[scope][parallelism]["h0_bounded_residual"]["calls_mape"]
                    - headline[scope][parallelism]["h0"]["calls_mape"]
                )
                / max(headline[scope][parallelism]["h0"]["calls_mape"], 1e-12),
                "histogram_tv_relative_change": (
                    headline[scope][parallelism]["h0_bounded_residual"][
                        "mean_histogram_tv"
                    ]
                    - headline[scope][parallelism]["h0"]["mean_histogram_tv"]
                )
                / max(
                    headline[scope][parallelism]["h0"]["mean_histogram_tv"], 1e-12
                ),
                "common_cost_mape_relative_change": (
                    headline[scope][parallelism]["h0_bounded_residual"][
                        "common_reference_cost_mape"
                    ]
                    - headline[scope][parallelism]["h0"]["common_reference_cost_mape"]
                )
                / max(
                    headline[scope][parallelism]["h0"]["common_reference_cost_mape"],
                    1e-12,
                ),
            }
            for parallelism in PARALLELISMS
        }
        for scope in headline
    }
    contract = {
        "schema_version": "phase26d-hfull-profile-holdout-contract-v1",
        "frozen_training_commit_at_evaluation": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip(),
        "test_splits": list(TEST_SPLITS),
        "test_profile_counts": dict(split_profiles),
        "checkpoint_hashes": checkpoint_hashes,
        "methods": list(METHODS),
        "metric_contract": training_summary["metric_contract"],
    }
    write_json(args.output_dir / "contract.json", contract)
    summary = {
        "schema_version": "phase26d-hfull-profile-holdout-evaluation-v1",
        "status": status,
        "objective": "evaluate frozen Phase 26C Hfull predictors on untouched temporal, external, and external-synthetic profile groups",
        "device": str(device),
        "counts": {
            "test_profiles": len(test_profiles),
            "prediction_records": len(records),
            "metric_rows": len(metrics),
            "comparison_rows": len(comparisons),
        },
        "test_profile_split_counts": dict(split_profiles),
        "checkpoint_hashes": checkpoint_hashes,
        "test_headline": headline,
        "residual_relative_outcome": residual_outcome,
        "checks": checks,
        "can_conclude": [
            "the frozen Phase 26C models were evaluated without train/validation profile leakage",
            "TP and PP generalization can be compared across temporal, external, and synthetic holdouts",
            "H0, direct, and bounded residual can be selected or rejected using untouched-profile evidence",
        ],
        "cannot_conclude": [
            "one external synthetic profile represents a population distribution",
            "the result applies to online arrival-aware scheduling or another PP scheduler contract",
            "the common parameterized cost is physical PP communication time",
        ],
        "next_step": "use the holdout evidence to freeze the first-version TP/PP predictor choice; if PP remains weak, enrich compact scheduler-sensitive profile features without changing Hfull teacher",
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(
        args.output_dir / "audit_summary.json",
        {
            "schema_version": "phase26d-hfull-profile-holdout-audit-v1",
            "status": status,
            "checks": checks,
            "checkpoint_hashes": checkpoint_hashes,
            "dataset_sha256": sha256(args.dataset),
            "training_summary_sha256": sha256(args.training_root / "summary.json"),
        },
    )
    (args.output_dir / "README.md").write_text(readme(summary))
    (args.output_dir / "DONE").write_text("PASS\n")
    write_json(
        args.output_dir / "logs/evaluation.log",
        {
            "schema_version": "phase26d-evaluation-log-v1",
            "status": status,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository_head_at_evaluation": contract["frozen_training_commit_at_evaluation"],
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "args": {
                "dataset": str(args.dataset),
                "training_root": str(args.training_root),
                "output_dir": str(args.output_dir),
                "device": args.device,
            },
        },
    )
    manifest_rows = []
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.sha256":
            manifest_rows.append(f"{sha256(path)}  {path.relative_to(args.output_dir)}")
    (args.output_dir / "manifest.sha256").write_text("\n".join(manifest_rows) + "\n")
    print(
        json.dumps(
            {
                "status": status,
                "device": str(device),
                "test_profiles": len(test_profiles),
                "prediction_records": len(records),
                "output_dir": str(args.output_dir),
            }
        )
    )


if __name__ == "__main__":
    main()
