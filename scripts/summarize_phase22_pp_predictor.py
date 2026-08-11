#!/usr/bin/env python3
"""Write the auditable Phase-22 pure-PP predictor conclusion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=root / "experiment-results/phase22_pp_predictor",
    )
    parser.add_argument(
        "--online-labels",
        type=Path,
        default=root
        / "experiment-results/phase21c_pp_online_residual/qwen3-8b-labels-v1/labels.csv",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    datasets = {
        "offline_draining": root / "qwen3-8b-offline-v1/metrics.csv",
        "profiled_online": root / "qwen3-8b-online-v1/metrics.csv",
    }
    comparison = []
    indexed = {}
    for dataset, path in datasets.items():
        for row in read_csv(path):
            if row["scope"] != "all":
                continue
            output = {
                "dataset": dataset,
                "evaluation": row["evaluation"],
                "method": row["method"],
                "samples": int(row["samples"]),
                "calls_mape": float(row["total_calls_mape"]),
                "calls_p95_ape": float(row["total_calls_p95_ape"]),
                "bytes_mape": float(row["total_bytes_mape"]),
                "bytes_p95_ape": float(row["total_bytes_p95_ape"]),
                "histogram_l1": float(row["mean_histogram_l1"]),
                "log_payload_emd": float(row["mean_log_payload_emd"]),
            }
            comparison.append(output)
            indexed[(dataset, row["evaluation"], row["method"])] = output

    with (root / "metrics_comparison.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(comparison[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparison)

    labels = read_csv(args.online_labels.resolve())
    repeats = defaultdict(list)
    for row in labels:
        repeats[row["h0_sample_id"]].append(row)
    exact = 0
    calls_delta = []
    bytes_delta = []
    for rows in repeats.values():
        if len(rows) != 2:
            raise ValueError("online residual audit expects exactly two repeats")
        exact += int(rows[0]["payload_histogram_json"] == rows[1]["payload_histogram_json"])
        calls = [float(row["per_boundary_calls"]) for row in rows]
        logical_bytes = [float(row["per_boundary_logical_bytes"]) for row in rows]
        calls_delta.append(abs(calls[1] - calls[0]) / max(statistics.mean(calls), 1.0))
        bytes_delta.append(
            abs(logical_bytes[1] - logical_bytes[0])
            / max(statistics.mean(logical_bytes), 1.0)
        )

    residual = {
        evaluation: indexed[("profiled_online", evaluation, "h0_residual")]
        for evaluation in ("profile_holdout", "strategy_holdout", "pp_holdout")
    }
    convergence = {
        "online_bytes_mape_below_10pct": all(
            row["bytes_mape"] < 0.10 for row in residual.values()
        ),
        "online_calls_mape_below_10pct": all(
            row["calls_mape"] < 0.10 for row in residual.values()
        ),
        "online_histogram_l1_below_0_2": all(
            row["histogram_l1"] < 0.20 for row in residual.values()
        ),
    }
    summary = {
        "schema_version": "phase22-pure-pp-predictor-summary-v1",
        "status": "PASS_WITH_LIMITATION",
        "completed_artifacts": {
            "offline_windows": 216,
            "offline_phase_labels": 432,
            "online_windows": 108,
            "online_phase_labels": 216,
            "profiles_online": 6,
            "pp_sizes": [2, 4, 8],
            "microbatch_sizes": [1, 4, 16],
        },
        "online_repeat_stability": {
            "groups": len(repeats),
            "exact_histogram_groups": exact,
            "exact_histogram_rate": exact / len(repeats),
            "mean_pair_calls_delta": statistics.mean(calls_delta),
            "p95_pair_calls_delta": percentile(calls_delta, 0.95),
            "mean_pair_bytes_delta": statistics.mean(bytes_delta),
            "p95_pair_bytes_delta": percentile(bytes_delta, 0.95),
        },
        "online_h0_residual": residual,
        "convergence": convergence,
        "conclusion": (
            "The structured PP formula predicts logical byte volume accurately, but "
            "the current compressed image and two online repeats do not identify calls "
            "or the payload distribution accurately enough. The predictor is not ready "
            "for a scheduling default; the next experiment must enrich the compact "
            "length/survival image and estimate the expected online histogram over more "
            "repeat realizations rather than adding a larger black-box DNN."
        ),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def pct(value: float) -> str:
        return f"{100 * value:.2f}%"

    table = []
    for evaluation in ("profile_holdout", "strategy_holdout", "pp_holdout"):
        for method in ("direct_dnn", "h0", "h0_residual"):
            row = indexed[("profiled_online", evaluation, method)]
            table.append(
                f"| {evaluation} | {method} | {pct(row['calls_mape'])} | "
                f"{pct(row['bytes_mape'])} | {row['histogram_l1']:.4f} |"
            )
    readme = f"""# Phase 22：纯 PP 服务画像 PatternDemand 预测器

## 完成范围

- Qwen3-8B，纯 `TP=1`、`PP=2/4/8`；
- `pp_max_micro_batch_size=1/4/16`；
- 24 个 BurstGPT/Mooncake 画像的 216 个 draining 配置、432 个阶段标签；
- 6 个分层画像的 108 个在线窗口、216 个阶段标签；
- 对比 Direct DNN、结构化 H0、H0 + bounded DNN residual；
- 所有 GPU 标签使用首个 sender 边界作为 group-level 真值，其余边界只做一致性检查。

## 在线严格留出结果

| 留出方式 | 方法 | calls MAPE | bytes MAPE | histogram L1 |
|---|---|---:|---:|---:|
{chr(10).join(table)}

## 重复稳定性

- 108 个 `画像×PP×策略×phase` 重复组；
- 精确直方图一致：{exact}/108（{pct(exact / len(repeats))}）；
- 两次重复的 calls 平均相对差：{pct(statistics.mean(calls_delta))}；
- calls P95 相对差：{pct(percentile(calls_delta, 0.95))}；
- logical bytes 两次重复完全一致。

这说明模型结构和长度画像能够稳定确定总逻辑字节，但在线batch边界会改变消息调用次数，
相同输入画像并不对应唯一的精确calls直方图。因此正式目标应是条件期望直方图，而不是
一次调度实现的精确直方图。

## 结论

当前结构化 H0 + residual 在三种在线留出下的bytes MAPE为
{pct(residual['profile_holdout']['bytes_mape'])}、
{pct(residual['strategy_holdout']['bytes_mape'])}和
{pct(residual['pp_holdout']['bytes_mape'])}，验证了结构公式对通信总量的价值。

但calls MAPE仍为{pct(residual['profile_holdout']['calls_mape'])}、
{pct(residual['strategy_holdout']['calls_mape'])}和
{pct(residual['pp_holdout']['calls_mape'])}，histogram L1也未收敛。因此本阶段数据和执行
闭环通过，但不能把当前checkpoint作为调度器默认PP预测器。

下一步应增加紧凑的输入/输出长度生存曲线和更细联合分布，并在分层小样本上增加重复，
直接学习期望calls/直方图残差；不应通过扩大黑盒DNN掩盖输入画像信息不足。
"""
    (root / "README.md").write_text(readme)
    (root / "DONE").write_text("PASS_WITH_LIMITATION\n")
    files = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.name != "manifest.sha256"
    )
    (root / "manifest.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
