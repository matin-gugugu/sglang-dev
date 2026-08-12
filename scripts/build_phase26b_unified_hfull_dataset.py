#!/usr/bin/env python3
"""Build the Phase 26B unified TP/PP Hfull training-data contract.

The output contains only compact, deployment-available inputs, compact32 H0
baselines, and offline Hfull teacher labels.  Complete request lists are not
copied into the training dataset.
"""

from __future__ import annotations

import argparse
import bisect
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


TP_BIN_SCHEMA = "tp_native_12bin_4k_512m_v1"
PP_BIN_SCHEMA = "pp_native_12bin_4k_8g_v1"
TP_BIN_EDGES = [
    2.0**value
    for value in [
        math.log2(4 * 1024)
        + index * (math.log2(512 * 1024 * 1024) - math.log2(4 * 1024)) / 12
        for index in range(13)
    ]
]
PP_BIN_EDGES = [
    2.0**value
    for value in [
        math.log2(4 * 1024)
        + index * (math.log2(8 * 1024 * 1024 * 1024) - math.log2(4 * 1024)) / 12
        for index in range(13)
    ]
]
COMMON_REFERENCE_LAUNCH_US = 5.0
COMMON_REFERENCE_BANDWIDTH_GBPS = 100.0
PP_CHUNK_TOKENS = 4096
PP_PAGE_SIZE = 64
PP_PROXY_TENSOR_COUNT = 2
NORMALIZATION_REQUESTS = 1000

PROFILE_SCALARS = (
    "rps",
    "interarrival_cv",
    "peak_to_mean_1s",
    "fano_1s",
    "input_mean_capped",
    "output_mean_capped",
    "lm_correlation_capped",
    "survival_m_gt_8",
    "survival_m_gt_16",
    "survival_m_gt_32",
    "survival_m_gt_64",
)
MODEL_NUMERICS = (
    "num_hidden_layers",
    "hidden_size",
    "dense_intermediate_ratio",
    "num_attention_heads",
    "head_dim",
    "kv_head_ratio",
    "dtype_bytes",
    "is_moe",
    "num_experts",
    "experts_per_token",
    "moe_intermediate_ratio",
    "num_shared_experts",
    "first_dense_layers",
    "moe_layer_frequency",
    "estimated_moe_layers",
    "logical_collectives_per_forward_prior",
    "payload_bytes_per_active_token_prior",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tp-labels",
        type=Path,
        default=root
        / "experiment-results/phase26a_tp_hfull_teacher_audit/labels/tp_hfull_phase_labels.csv.gz",
    )
    parser.add_argument(
        "--pp-labels",
        type=Path,
        default=root
        / "experiment-results/phase25b_pp_scheduler_teacher/labels/pp_phase_labels.csv.gz",
    )
    parser.add_argument(
        "--phase24-labels",
        type=Path,
        default=root
        / "experiment-results/phase24_representative_request_convergence/labels/histogram_labels.jsonl.gz",
    )
    parser.add_argument(
        "--phase24-requests",
        type=Path,
        default=root
        / "experiment-results/phase24_representative_request_convergence/input_windows/selected_requests.jsonl.gz",
    )
    parser.add_argument(
        "--phase25d-labels",
        type=Path,
        default=root
        / "experiment-results/phase25d_pp_scheduler_representative_convergence/labels/histogram_labels.jsonl.gz",
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=root / "experiment-results/phase16_service_profiles/service_profiles.csv",
    )
    parser.add_argument(
        "--model-features",
        type=Path,
        default=root / "experiment-results/phase16_model_features/model_features.json",
    )
    parser.add_argument(
        "--plan-summary",
        type=Path,
        default=root / "experiment-results/phase16_profiledemand_plans/summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase26b_unified_hfull_training_dataset",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as source:
        return list(csv.DictReader(source))


def read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as source:
        return [json.loads(line) for line in source if line.strip()]


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


def canonical_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def histogram_from_json(value: str) -> dict[int, float]:
    return {int(payload): float(calls) for payload, calls in json.loads(value).items()}


def histograms_close(left: dict[int, float], right: dict[int, float]) -> bool:
    keys = set(left) | set(right)
    return all(
        math.isclose(left.get(key, 0.0), right.get(key, 0.0), rel_tol=1e-11, abs_tol=1e-7)
        for key in keys
    )


def bin_vectors(histogram: dict[int, float], edges: list[float]) -> tuple[list[float], list[float]]:
    calls = [0.0] * 12
    logical_bytes = [0.0] * 12
    for payload, count in histogram.items():
        index = min(max(bisect.bisect_right(edges, float(payload)) - 1, 0), 11)
        calls[index] += count
        logical_bytes[index] += count * payload
    return calls, logical_bytes


def vectors_close(left: list[float], right: list[float]) -> bool:
    return len(left) == len(right) and all(
        math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-4)
        for a, b in zip(left, right)
    )


def tp_batches(
    requests: list[tuple[int, int]], max_batch_size: int, max_prefill_tokens: int
) -> list[list[tuple[int, int]]]:
    batches: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    current_tokens = 0
    for request in requests:
        if current and (
            len(current) >= max_batch_size
            or current_tokens + request[0] > max_prefill_tokens
        ):
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(request)
        current_tokens += request[0]
    if current:
        batches.append(current)
    return batches


def tp_compact_histogram(
    requests: list[tuple[int, int]], strategy: dict, model: dict, phase: str
) -> dict[int, float]:
    batches = tp_batches(
        requests, int(strategy["max_batch_size"]), int(strategy["max_prefill_tokens"])
    )
    calls_per_forward = int(model["logical_collectives_per_forward_prior"])
    bytes_per_token = int(model["payload_bytes_per_active_token_prior"])
    histogram: Counter[int] = Counter()
    if phase == "prefill":
        for batch in batches:
            histogram[sum(row[0] for row in batch) * bytes_per_token] += calls_per_forward
    elif phase == "decode":
        for batch in batches:
            max_output = max(row[1] for row in batch)
            for step in range(1, max_output):
                active = sum(row[1] > step for row in batch)
                if active:
                    histogram[active * bytes_per_token] += calls_per_forward
    else:
        raise ValueError(phase)
    scale = NORMALIZATION_REQUESTS / len(requests)
    return {payload: calls * scale for payload, calls in sorted(histogram.items())}


def common_reference_cost(histogram: dict[int, float]) -> float:
    bandwidth_bytes_per_second = COMMON_REFERENCE_BANDWIDTH_GBPS * 1e9
    return sum(
        calls
        * (COMMON_REFERENCE_LAUNCH_US + payload / bandwidth_bytes_per_second * 1e6)
        for payload, calls in histogram.items()
    )


def normalized_log_emd(left: dict[int, float], right: dict[int, float]) -> float:
    left_total = sum(left.values())
    right_total = sum(right.values())
    if left_total <= 0 or right_total <= 0:
        return 0.0 if left_total <= 0 and right_total <= 0 else 1.0
    points = sorted(set(left) | set(right))
    if len(points) < 2:
        return 0.0
    left_cdf = 0.0
    right_cdf = 0.0
    area = 0.0
    for index, payload in enumerate(points[:-1]):
        left_cdf += left.get(payload, 0.0) / left_total
        right_cdf += right.get(payload, 0.0) / right_total
        width = math.log2(points[index + 1]) - math.log2(payload)
        area += abs(left_cdf - right_cdf) * width
    span = math.log2(8 * 1024 * 1024 * 1024) - math.log2(4 * 1024)
    return area / span


def histogram_l1(left: dict[int, float], right: dict[int, float]) -> float:
    left_total = sum(left.values())
    right_total = sum(right.values())
    keys = set(left) | set(right)
    return sum(
        abs(left.get(key, 0.0) / left_total - right.get(key, 0.0) / right_total)
        for key in keys
    )


def metric_row(target: dict, baseline_hist: dict[int, float]) -> dict:
    target_hist = histogram_from_json(target["exact_calls_histogram_per_1000_json"])
    target_calls = sum(target_hist.values())
    baseline_calls = sum(baseline_hist.values())
    target_bytes = sum(payload * calls for payload, calls in target_hist.items())
    baseline_bytes = sum(payload * calls for payload, calls in baseline_hist.items())
    target_cost = common_reference_cost(target_hist)
    baseline_cost = common_reference_cost(baseline_hist)
    l1 = histogram_l1(baseline_hist, target_hist)
    return {
        "training_id": target["label_id"],
        "parallelism": target["parallelism"],
        "model": target["model"],
        "parallel_size": target["parallel_size"],
        "policy": target["policy"],
        "profile_id": target["profile_id"],
        "split": target["split"],
        "phase": target["phase"],
        "target_total_calls": target_calls,
        "baseline_total_calls": baseline_calls,
        "calls_absolute_error": abs(baseline_calls - target_calls),
        "calls_ape": abs(baseline_calls - target_calls) / max(target_calls, 1e-12),
        "target_total_logical_bytes": target_bytes,
        "baseline_total_logical_bytes": baseline_bytes,
        "bytes_absolute_error": abs(baseline_bytes - target_bytes),
        "bytes_ape": abs(baseline_bytes - target_bytes) / max(target_bytes, 1e-12),
        "histogram_l1": l1,
        "histogram_tv": l1 / 2,
        "normalized_log_payload_emd": normalized_log_emd(baseline_hist, target_hist),
        "target_common_reference_cost_us": target_cost,
        "baseline_common_reference_cost_us": baseline_cost,
        "cost_ape": abs(baseline_cost - target_cost) / max(target_cost, 1e-12),
    }


def total_metric_row(pairs: list[tuple[dict, dict[int, float]]]) -> dict:
    if len(pairs) != 2 or {target["phase"] for target, _ in pairs} != {"prefill", "decode"}:
        raise ValueError("total metric requires exactly one prefill and one decode row")
    pairs = sorted(pairs, key=lambda pair: pair[0]["phase"])
    first = pairs[0][0]
    target_by_phase = {
        target["phase"]: histogram_from_json(target["exact_calls_histogram_per_1000_json"])
        for target, _ in pairs
    }
    baseline_by_phase = {target["phase"]: histogram for target, histogram in pairs}
    target_aware = {
        (phase, payload): calls
        for phase, histogram in target_by_phase.items()
        for payload, calls in histogram.items()
    }
    baseline_aware = {
        (phase, payload): calls
        for phase, histogram in baseline_by_phase.items()
        for payload, calls in histogram.items()
    }
    target_pooled: Counter[int] = Counter()
    baseline_pooled: Counter[int] = Counter()
    for histogram in target_by_phase.values():
        target_pooled.update(histogram)
    for histogram in baseline_by_phase.values():
        baseline_pooled.update(histogram)
    target_calls = sum(target_aware.values())
    baseline_calls = sum(baseline_aware.values())
    target_bytes = sum(payload * calls for (_, payload), calls in target_aware.items())
    baseline_bytes = sum(payload * calls for (_, payload), calls in baseline_aware.items())
    target_cost = common_reference_cost(dict(target_pooled))
    baseline_cost = common_reference_cost(dict(baseline_pooled))
    l1 = histogram_l1(baseline_aware, target_aware)
    config_id = first["label_id"].replace(f"/{first['phase']}", "/total")
    return {
        "training_id": config_id,
        "parallelism": first["parallelism"],
        "model": first["model"],
        "parallel_size": first["parallel_size"],
        "policy": first["policy"],
        "profile_id": first["profile_id"],
        "split": first["split"],
        "phase": "total",
        "target_total_calls": target_calls,
        "baseline_total_calls": baseline_calls,
        "calls_absolute_error": abs(baseline_calls - target_calls),
        "calls_ape": abs(baseline_calls - target_calls) / max(target_calls, 1e-12),
        "target_total_logical_bytes": target_bytes,
        "baseline_total_logical_bytes": baseline_bytes,
        "bytes_absolute_error": abs(baseline_bytes - target_bytes),
        "bytes_ape": abs(baseline_bytes - target_bytes) / max(target_bytes, 1e-12),
        "histogram_l1": l1,
        "histogram_tv": l1 / 2,
        "normalized_log_payload_emd": normalized_log_emd(
            dict(baseline_pooled), dict(target_pooled)
        ),
        "target_common_reference_cost_us": target_cost,
        "baseline_common_reference_cost_us": baseline_cost,
        "cost_ape": abs(baseline_cost - target_cost) / max(target_cost, 1e-12),
    }


def aggregate_metrics(rows: list[dict]) -> dict:
    target_calls = sum(float(row["target_total_calls"]) for row in rows)
    target_bytes = sum(float(row["target_total_logical_bytes"]) for row in rows)
    return {
        "cases": len(rows),
        "calls_mape": sum(float(row["calls_ape"]) for row in rows) / len(rows),
        "calls_wape": sum(float(row["calls_absolute_error"]) for row in rows) / target_calls,
        "bytes_mape": sum(float(row["bytes_ape"]) for row in rows) / len(rows),
        "bytes_wape": sum(float(row["bytes_absolute_error"]) for row in rows) / target_bytes,
        "mean_histogram_l1": sum(float(row["histogram_l1"]) for row in rows) / len(rows),
        "mean_histogram_tv": sum(float(row["histogram_tv"]) for row in rows) / len(rows),
        "mean_normalized_log_payload_emd": sum(
            float(row["normalized_log_payload_emd"]) for row in rows
        )
        / len(rows),
        "common_reference_cost_mape": sum(float(row["cost_ape"]) for row in rows) / len(rows),
    }


def feature_row(
    target: dict, profile: dict[str, str], model: dict, strategies: dict
) -> dict:
    parallelism = target["parallelism"]
    policy = target["policy"]
    row = {
        "training_id": target["label_id"],
        "profile_id": target["profile_id"],
        "split": target["split"],
        "model": target["model"],
        "parallelism": parallelism,
        "parallel_size": target["parallel_size"],
        "policy": policy,
        "phase": target["phase"],
        "feature_parallelism_tp": int(parallelism == "tp"),
        "feature_parallelism_pp": int(parallelism == "pp"),
        "feature_parallel_size_log2": math.log2(int(target["parallel_size"])),
        "feature_phase_prefill": int(target["phase"] == "prefill"),
        "feature_phase_decode": int(target["phase"] == "decode"),
    }
    for name in PROFILE_SCALARS:
        row[f"feature_profile_{name}"] = float(profile[name])
    for index, value in enumerate(json.loads(profile["joint_lm_4x4_json"])):
        row[f"feature_profile_joint_lm_{index}"] = float(value)
    for name in MODEL_NUMERICS:
        row[f"feature_model_{name}"] = float(model[name])
    if parallelism == "tp":
        strategy = strategies[policy]
        row.update(
            {
                "feature_tp_max_batch_size": int(strategy["max_batch_size"]),
                "feature_tp_max_prefill_tokens": int(strategy["max_prefill_tokens"]),
                "feature_pp_max_microbatch_size": 0,
                "feature_pp_chunk_tokens": 0,
                "feature_pp_page_size": 0,
                "feature_pp_proxy_tensor_count": 0,
            }
        )
    else:
        row.update(
            {
                "feature_tp_max_batch_size": 0,
                "feature_tp_max_prefill_tokens": 0,
                "feature_pp_max_microbatch_size": int(policy.removeprefix("mb")),
                "feature_pp_chunk_tokens": PP_CHUNK_TOKENS,
                "feature_pp_page_size": PP_PAGE_SIZE,
                "feature_pp_proxy_tensor_count": PP_PROXY_TENSOR_COUNT,
            }
        )
    return row


def target_row(source: dict) -> dict:
    parallelism = source["parallelism"]
    schema = TP_BIN_SCHEMA if parallelism == "tp" else PP_BIN_SCHEMA
    edges = TP_BIN_EDGES if parallelism == "tp" else PP_BIN_EDGES
    return {
        "label_id": source["label_id"],
        "label_status": source["label_status"],
        "teacher_kind": source["teacher_kind"],
        "model": source["model"],
        "model_config_sha256": source["model_config_sha256"],
        "profile_id": source["profile_id"],
        "source": source["source"],
        "segment": source["segment"],
        "split": source["split"],
        "window_id": source["window_id"],
        "parallelism": parallelism,
        "parallel_size": source["parallel_size"],
        "policy": source["policy"],
        "phase": source["phase"],
        "requests": source["requests"],
        "normalization_requests": source["normalization_requests"],
        "boundary_multiplier": source["boundary_multiplier"],
        "bin_schema_id": schema,
        "bin_edges_bytes_json": canonical_json(edges),
        "total_calls_per_1000": source["total_calls_per_1000"],
        "total_logical_bytes_per_1000": source["total_logical_bytes_per_1000"],
        "pipeline_calls_per_1000": source["pipeline_calls_per_1000"],
        "pipeline_logical_bytes_per_1000": source["pipeline_logical_bytes_per_1000"],
        "calls_by_12bin_json": source["calls_by_12bin_json"],
        "logical_bytes_by_12bin_json": source["logical_bytes_by_12bin_json"],
        "exact_calls_histogram_per_1000_json": source["exact_calls_histogram_per_1000_json"],
        "exact_logical_bytes_histogram_per_1000_json": source[
            "exact_logical_bytes_histogram_per_1000_json"
        ],
        "scheduler_contract": source.get("scheduler_contract", ""),
        "pp_loop_lanes": source.get("pp_loop_lanes", ""),
        "chunk_tokens": source.get("chunk_tokens", ""),
        "page_size": source.get("page_size", ""),
        "proxy_tensor_count": source.get("proxy_tensor_count", ""),
    }


def baseline_row(target: dict, histogram: dict[int, float], baseline_status: str) -> dict:
    parallelism = target["parallelism"]
    edges = TP_BIN_EDGES if parallelism == "tp" else PP_BIN_EDGES
    calls_bins, bytes_bins = bin_vectors(histogram, edges)
    return {
        "training_id": target["label_id"],
        "baseline_id": target["label_id"].replace("/hfull/", "/compact32/"),
        "baseline_status": baseline_status,
        "baseline_kind": (
            "compact32_fixed_draining_structural_formula"
            if parallelism == "tp"
            else "compact32_fixed_draining_scheduler_faithful_formula"
        ),
        "model": target["model"],
        "profile_id": target["profile_id"],
        "split": target["split"],
        "parallelism": parallelism,
        "parallel_size": target["parallel_size"],
        "policy": target["policy"],
        "phase": target["phase"],
        "sample_requests": 32,
        "normalization_requests": NORMALIZATION_REQUESTS,
        "bin_schema_id": target["bin_schema_id"],
        "bin_edges_bytes_json": target["bin_edges_bytes_json"],
        "total_calls_per_1000": sum(histogram.values()),
        "total_logical_bytes_per_1000": sum(
            payload * calls for payload, calls in histogram.items()
        ),
        "calls_by_12bin_json": canonical_json(calls_bins),
        "logical_bytes_by_12bin_json": canonical_json(bytes_bins),
        "exact_calls_histogram_per_1000_json": canonical_json(
            {str(payload): calls for payload, calls in histogram.items()}
        ),
        "exact_logical_bytes_histogram_per_1000_json": canonical_json(
            {str(payload): payload * calls for payload, calls in histogram.items()}
        ),
    }


def inventory_rows(targets: list[dict]) -> list[dict]:
    groups: dict[tuple, int] = defaultdict(int)
    for row in targets:
        groups[
            (
                row["parallelism"],
                row["model"],
                row["parallel_size"],
                row["policy"],
                row["phase"],
                row["split"],
            )
        ] += 1
    return [
        {
            "parallelism": key[0],
            "model": key[1],
            "parallel_size": key[2],
            "policy": key[3],
            "phase": key[4],
            "split": key[5],
            "rows": count,
        }
        for key, count in sorted(groups.items())
    ]


def build_readme(summary: dict) -> str:
    tp = summary["h0_baseline_headline"]["tp"]
    pp = summary["h0_baseline_headline"]["pp"]
    return f"""# Phase 26B：统一 TP/PP Hfull 训练数据集

状态：**{summary['status']}**。

本阶段把 Phase 26A 晋升后的 TP Hfull teacher 与 Phase 25B scheduler-faithful PP
Hfull teacher 合并为同一套训练数据契约。共 {summary['counts']['training_examples']}
条 phase-level 样本，其中 TP {summary['counts']['tp_targets']} 条、PP
{summary['counts']['pp_targets']} 条；每条 Hfull target 都有且仅有一条由低维画像生成的
compact32 H0 baseline 和一条部署可用输入特征记录。

## 输入与输出口径

- 预测输入只含低维常态流量画像、模型结构、已确定的 TP/PP size、固定策略和 phase；
- 完整请求列表只参与上游离线 Hfull teacher 生成，没有复制进本数据集；
- target 是每 1000 请求归一化的 calls、logical bytes、原生 12 桶及 exact payload histogram；
- TP 原生桶范围是 4 KiB–512 MiB，PP 是 4 KiB–8 GiB。两者不能暗中共用同一桶语义，
  因此每条样本显式携带 `bin_schema_id` 和 `bin_edges_bytes_json`；
- `profile_splits.csv` 固化 Phase 16 的 5 train、5 validation、5 temporal test、
  8 external test、1 external synthetic 画像划分。后续任何早停和测试都必须按完整画像分组。

## compact32 H0 相对 Hfull 的未训练基线

| 并行 | cases | calls MAPE/WAPE | bytes MAPE/WAPE | histogram TV | log-payload EMD | common cost MAPE |
|---|---:|---:|---:|---:|---:|---:|
| TP | {tp['cases']} | {tp['calls_mape']:.2%} / {tp['calls_wape']:.2%} | {tp['bytes_mape']:.2%} / {tp['bytes_wape']:.2%} | {tp['mean_histogram_tv']:.4f} | {tp['mean_normalized_log_payload_emd']:.4f} | {tp['common_reference_cost_mape']:.2%} |
| PP | {pp['cases']} | {pp['calls_mape']:.2%} / {pp['calls_wape']:.2%} | {pp['bytes_mape']:.2%} / {pp['bytes_wape']:.2%} | {pp['mean_histogram_tv']:.4f} | {pp['mean_normalized_log_payload_emd']:.4f} | {pp['common_reference_cost_mape']:.2%} |

这里的 common cost 使用统一的 5 μs 启动项和 100 GB/s 参数曲线，只用于比较消息
直方图误差传播，不是 PP 物理链路实测。

## 完整性检查

- 1,728 个 target ID、baseline ID、feature ID 一一对应且无重复；
- 1,296 条 TP target 保持 Phase 26A 的 GPU sentinel 晋升状态；
- 432 条 PP target 保持 Phase 25B/25C 验证过的 scheduler contract；
- Qwen3-8B TP compact32 与 Phase 24 的 432 条记录 exact 回归一致；
- PP Hfull 与 Phase 25D 的 432 条记录 exact 回归一致；
- teacher exact histogram、原生 12 桶和 total calls/bytes 互相复算一致；
- 结果中没有完整请求列表、raw profiler events、模型权重、缓存或 PID。

## 文件

- `labels/hfull_targets.csv.gz`：统一后的 Hfull teacher；
- `baselines/compact32_h0.csv.gz`：一一对应的 compact32 H0；
- `features/low_dimensional_inputs.csv.gz`：部署可用低维输入；
- `training_examples.csv.gz`：可直接用于 Phase 26C 的紧凑 join；
- `splits/profile_splits.csv`：画像级划分；
- `analysis/h0_vs_hfull_per_row.csv.gz`、`h0_vs_hfull_total.csv.gz` 与
  `h0_vs_hfull_aggregate.csv`：phase-level、配置 total 和聚合后的未训练基线；
- `analysis/dataset_inventory.csv`：配置与 split 库存；
- `contract.json`、`summary.json`、`audit_summary.json`、`logs/build.log`、`DONE`
  和 `manifest.sha256`：契约、审计与归档证据。

## 可以与不可以得出的结论

可以确认训练数据标签已经从 Phase 16 的 H32 GPU label 改成 Hfull teacher，且 TP/PP
输入、baseline、target 和 split 在同一契约内可追溯。不能据此宣称模型精度已经改善，
因为 Phase 26C 尚未重训，Phase 26D 尚未做画像级留出评测；也不能把 common cost 当作
PP 实际通信时间。

下一步：在此数据集上分别训练 direct、H0 和 H0+bounded residual，优先报告 TP/PP
分项与 phase 分项，不跨 `bin_schema_id` 混淆桶含义。
"""


def main() -> None:
    args = parse_args()
    for directory in (
        args.output_dir / "labels",
        args.output_dir / "baselines",
        args.output_dir / "features",
        args.output_dir / "splits",
        args.output_dir / "analysis",
        args.output_dir / "logs",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    source_paths = {
        "tp_hfull_labels": args.tp_labels,
        "pp_hfull_labels": args.pp_labels,
        "phase24_histogram_labels": args.phase24_labels,
        "phase24_selected_requests": args.phase24_requests,
        "phase25d_histogram_labels": args.phase25d_labels,
        "service_profiles": args.profiles,
        "model_features": args.model_features,
        "phase16_plan_summary": args.plan_summary,
    }
    tp_sources = read_csv(args.tp_labels)
    pp_sources = read_csv(args.pp_labels)
    if len(tp_sources) != 1296 or len(pp_sources) != 432:
        raise ValueError(f"unexpected teacher counts: TP={len(tp_sources)}, PP={len(pp_sources)}")
    targets = [target_row(row) for row in tp_sources + pp_sources]
    targets.sort(key=lambda row: row["label_id"])
    if len({row["label_id"] for row in targets}) != len(targets):
        raise ValueError("duplicate Hfull target IDs")

    profiles = {row["profile_id"]: row for row in read_csv(args.profiles)}
    models = {row["model"]: row for row in json.loads(args.model_features.read_text())}
    strategies = json.loads(args.plan_summary.read_text())["strategies"]
    if len(profiles) != 24 or set(models) != {"qwen3-8b", "qwen3-30b-a3b", "deepseek-v2-lite"}:
        raise ValueError("unexpected profile/model inventory")

    phase24 = read_jsonl_gz(args.phase24_labels)
    phase24_requests = read_jsonl_gz(args.phase24_requests)
    compact_requests = {
        row["profile_id"]: list(
            zip(map(int, row["input_lens"]), map(int, row["output_lens"]))
        )
        for row in phase24_requests
        if row["sample_label"] == "compact32"
    }
    phase24_tp_compact = {
        (
            row["profile_id"],
            int(row["parallel_size"]),
            row["policy"],
            row["phase"],
        ): {int(payload): float(calls) for payload, calls in json.loads(row["exact_calls_histogram_per_1000_json"]).items()}
        for row in phase24
        if row["parallelism"] == "tp" and row["sample_label"] == "compact32"
    }
    phase25d = read_jsonl_gz(args.phase25d_labels)
    pp_compact = {
        (
            row["profile_id"],
            int(row["parallel_size"]),
            row["policy"],
            row["phase"],
        ): {int(payload): float(calls) for payload, calls in json.loads(row["exact_calls_histogram_per_1000_json"]).items()}
        for row in phase25d
        if row["sample_label"] == "compact32"
    }
    phase25d_pp_hfull = {
        (
            row["profile_id"],
            int(row["parallel_size"]),
            row["policy"],
            row["phase"],
        ): {int(payload): float(calls) for payload, calls in json.loads(row["exact_calls_histogram_per_1000_json"]).items()}
        for row in phase25d
        if row["sample_label"] == "hfull"
    }
    if (
        len(compact_requests) != 24
        or any(len(requests) != 32 for requests in compact_requests.values())
        or len(phase24_tp_compact) != 432
        or len(pp_compact) != 432
        or len(phase25d_pp_hfull) != 432
    ):
        raise ValueError("unexpected Phase 24/25D compact or Hfull inventory")

    baseline_rows: list[dict] = []
    feature_rows: list[dict] = []
    metric_rows: list[dict] = []
    total_pairs: dict[tuple, list[tuple[dict, dict[int, float]]]] = defaultdict(list)
    native_bin_checks = []
    exact_bytes_checks = []
    total_checks = []
    tp_phase24_regressions = []
    pp_phase25d_regressions = []
    for target in targets:
        profile = profiles[target["profile_id"]]
        model = models[target["model"]]
        exact_target = histogram_from_json(target["exact_calls_histogram_per_1000_json"])
        exact_target_bytes = histogram_from_json(
            target["exact_logical_bytes_histogram_per_1000_json"]
        )
        exact_bytes_checks.append(
            histograms_close(
                {payload: payload * calls for payload, calls in exact_target.items()},
                exact_target_bytes,
            )
        )
        edges = TP_BIN_EDGES if target["parallelism"] == "tp" else PP_BIN_EDGES
        calls_bins, bytes_bins = bin_vectors(exact_target, edges)
        native_bin_checks.append(
            vectors_close(calls_bins, json.loads(target["calls_by_12bin_json"]))
            and vectors_close(bytes_bins, json.loads(target["logical_bytes_by_12bin_json"]))
        )
        total_checks.append(
            math.isclose(
                sum(exact_target.values()),
                float(target["total_calls_per_1000"]),
                rel_tol=1e-11,
                abs_tol=1e-7,
            )
            and math.isclose(
                sum(payload * calls for payload, calls in exact_target.items()),
                float(target["total_logical_bytes_per_1000"]),
                rel_tol=1e-11,
                abs_tol=1e-4,
            )
        )
        if target["parallelism"] == "tp":
            histogram = tp_compact_histogram(
                compact_requests[target["profile_id"]],
                strategies[target["policy"]],
                model,
                target["phase"],
            )
            baseline_status = "PHASE16_COMPACT32_FORMULA_PHASE24_EXACT_REGRESSION"
            if target["model"] == "qwen3-8b":
                key = (
                    target["profile_id"],
                    int(target["parallel_size"]),
                    target["policy"],
                    target["phase"],
                )
                tp_phase24_regressions.append(histograms_close(histogram, phase24_tp_compact[key]))
        else:
            key = (
                target["profile_id"],
                int(target["parallel_size"]),
                target["policy"],
                target["phase"],
            )
            histogram = pp_compact[key]
            baseline_status = "PHASE25D_SCHEDULER_COMPACT32"
            pp_phase25d_regressions.append(
                histograms_close(exact_target, phase25d_pp_hfull[key])
            )
        baseline_rows.append(baseline_row(target, histogram, baseline_status))
        feature_rows.append(feature_row(target, profile, model, strategies))
        metric_rows.append(metric_row(target, histogram))
        total_pairs[
            (
                target["parallelism"],
                target["model"],
                target["parallel_size"],
                target["policy"],
                target["profile_id"],
            )
        ].append((target, histogram))

    baseline_rows.sort(key=lambda row: row["training_id"])
    feature_rows.sort(key=lambda row: row["training_id"])
    metric_rows.sort(key=lambda row: row["training_id"])
    total_metric_rows = sorted(
        [total_metric_row(pairs) for pairs in total_pairs.values()],
        key=lambda row: row["training_id"],
    )
    target_ids = [row["label_id"] for row in targets]
    aligned = (
        target_ids == [row["training_id"] for row in baseline_rows]
        == [row["training_id"] for row in feature_rows]
        == [row["training_id"] for row in metric_rows]
    )
    if not aligned:
        raise ValueError("target, baseline, feature, and metric IDs are not aligned")

    training_rows = []
    for target, baseline, feature in zip(targets, baseline_rows, feature_rows):
        training_rows.append(
            {
                **feature,
                "teacher_label_status": target["label_status"],
                "teacher_kind": target["teacher_kind"],
                "full_requests": target["requests"],
                "normalization_requests": target["normalization_requests"],
                "bin_schema_id": target["bin_schema_id"],
                "bin_edges_bytes_json": target["bin_edges_bytes_json"],
                "h0_total_calls_per_1000": baseline["total_calls_per_1000"],
                "h0_total_logical_bytes_per_1000": baseline[
                    "total_logical_bytes_per_1000"
                ],
                "h0_calls_by_12bin_json": baseline["calls_by_12bin_json"],
                "h0_logical_bytes_by_12bin_json": baseline[
                    "logical_bytes_by_12bin_json"
                ],
                "target_total_calls_per_1000": target["total_calls_per_1000"],
                "target_total_logical_bytes_per_1000": target[
                    "total_logical_bytes_per_1000"
                ],
                "target_calls_by_12bin_json": target["calls_by_12bin_json"],
                "target_logical_bytes_by_12bin_json": target[
                    "logical_bytes_by_12bin_json"
                ],
            }
        )

    split_role = {
        "train": "train",
        "validation": "validation",
        "temporal_test": "test_temporal",
        "external_test": "test_external",
        "external_synthetic": "test_external_synthetic",
    }
    split_rows = [
        {
            "profile_id": profile["profile_id"],
            "source": profile["source"],
            "segment": profile["segment"],
            "phase16_split": profile["split"],
            "phase26_evaluation_role": split_role[profile["split"]],
            "request_count": profile["request_count"],
            "grouping_contract": "all rows for this profile stay in one split",
        }
        for profile in sorted(profiles.values(), key=lambda row: row["profile_id"])
    ]

    aggregate_rows = []
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in metric_rows:
        groups[(row["parallelism"], row["phase"], "all")].append(row)
    for row in total_metric_rows:
        groups[(row["parallelism"], "total", "all")].append(row)
        groups[(row["parallelism"], "total", row["policy"])].append(row)
    for (parallelism, phase, policy), rows in sorted(groups.items()):
        aggregate_rows.append(
            {
                "parallelism": parallelism,
                "phase": phase,
                "policy": policy,
                **aggregate_metrics(rows),
            }
        )
    headline = {
        parallelism: aggregate_metrics(
            [row for row in total_metric_rows if row["parallelism"] == parallelism]
        )
        for parallelism in ("tp", "pp")
    }

    checks = {
        "targets_1728": len(targets) == 1728,
        "tp_targets_1296": sum(row["parallelism"] == "tp" for row in targets) == 1296,
        "pp_targets_432": sum(row["parallelism"] == "pp" for row in targets) == 432,
        "unique_target_ids": len(set(target_ids)) == 1728,
        "one_to_one_alignment": aligned,
        "phase_metric_rows_1728": len(metric_rows) == 1728,
        "total_metric_rows_864": len(total_metric_rows) == 864,
        "profiles_24": len(profiles) == 24,
        "profile_split_counts_5_5_5_8_1": Counter(
            row["phase16_split"] for row in split_rows
        )
        == Counter(
            {
                "train": 5,
                "validation": 5,
                "temporal_test": 5,
                "external_test": 8,
                "external_synthetic": 1,
            }
        ),
        "target_exact_bytes_recompute": all(exact_bytes_checks),
        "target_native_bins_recompute": all(native_bin_checks),
        "target_totals_recompute": all(total_checks),
        "tp_qwen_compact32_phase24_exact_432": len(tp_phase24_regressions) == 432
        and all(tp_phase24_regressions),
        "pp_hfull_phase25d_exact_432": len(pp_phase25d_regressions) == 432
        and all(pp_phase25d_regressions),
        "tp_label_status_promoted": all(
            row["label_status"] == "GPU_VALIDATED_STRUCTURAL_FORMULA_SENTINELS_4_CELLS"
            for row in targets
            if row["parallelism"] == "tp"
        ),
        "pp_label_status_validated": all(
            row["label_status"] == "GPU_VALIDATED_SCHEDULER_FORMULA_SMOKE_9_OF_9"
            for row in targets
            if row["parallelism"] == "pp"
        ),
        "no_request_lists_in_training_schema": not any(
            "input_lens" in key or "output_lens" in key or "request_list" in key
            for key in training_rows[0]
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    if status != "PASS":
        raise RuntimeError(f"Phase 26B checks failed: {checks}")

    write_csv_gz(args.output_dir / "labels/hfull_targets.csv.gz", targets)
    write_csv_gz(args.output_dir / "baselines/compact32_h0.csv.gz", baseline_rows)
    write_csv_gz(args.output_dir / "features/low_dimensional_inputs.csv.gz", feature_rows)
    write_csv_gz(args.output_dir / "training_examples.csv.gz", training_rows)
    write_csv(args.output_dir / "splits/profile_splits.csv", split_rows)
    write_csv_gz(args.output_dir / "analysis/h0_vs_hfull_per_row.csv.gz", metric_rows)
    write_csv_gz(args.output_dir / "analysis/h0_vs_hfull_total.csv.gz", total_metric_rows)
    write_csv(args.output_dir / "analysis/h0_vs_hfull_aggregate.csv", aggregate_rows)
    write_csv(args.output_dir / "analysis/dataset_inventory.csv", inventory_rows(targets))

    contract = {
        "schema_version": "phase26b-unified-hfull-training-contract-v1",
        "prediction_input": "low-dimensional steady traffic profile + model structure + fixed TP/PP configuration + fixed execution policy + phase",
        "prediction_output": "topology-independent calls and logical-byte payload histograms per 1000 requests",
        "teacher": {
            "tp": "Phase 26A GPU-sentinel-promoted full-window fixed-draining structural teacher",
            "pp": "Phase 25B scheduler-faithful full-window fixed-draining teacher with Phase 25C tail evidence",
        },
        "offline_only_input": "complete capped request lists used upstream to generate Hfull labels; absent from this dataset",
        "h0": {
            "sample": "32 pseudo requests reconstructed only from compact profile statistics",
            "tp": "Phase 16 structural formula, exact-regressed to Phase 24 Qwen3-8B compact32",
            "pp": "Phase 25D sglang_pp_fcfs_lanes_v1 compact32 scheduler formula",
        },
        "bin_schemas": {
            TP_BIN_SCHEMA: TP_BIN_EDGES,
            PP_BIN_SCHEMA: PP_BIN_EDGES,
        },
        "split_contract": "Phase 16 profile split is frozen; no profile may cross fit/validation/test boundaries",
        "cost_contract": {
            "kind": "common parameterized reference, not physical PP measurement",
            "launch_us": COMMON_REFERENCE_LAUNCH_US,
            "bandwidth_gbps": COMMON_REFERENCE_BANDWIDTH_GBPS,
        },
        "scheduler_contract": {
            "pp": "sglang_pp_fcfs_lanes_v1",
            "chunk_tokens": PP_CHUNK_TOKENS,
            "page_size": PP_PAGE_SIZE,
            "proxy_tensor_count": PP_PROXY_TENSOR_COUNT,
        },
    }
    write_json(args.output_dir / "contract.json", contract)

    summary = {
        "schema_version": "phase26b-unified-hfull-training-dataset-v1",
        "status": status,
        "objective": "replace Phase 16 H32 supervision with audited Hfull supervision while preserving compact-profile inputs and profile-level split isolation",
        "counts": {
            "training_examples": len(training_rows),
            "tp_targets": sum(row["parallelism"] == "tp" for row in targets),
            "pp_targets": sum(row["parallelism"] == "pp" for row in targets),
            "profiles": len(profiles),
            "models": len(models),
            "feature_columns": sum(key.startswith("feature_") for key in feature_rows[0]),
            "metric_rows": len(metric_rows),
            "total_metric_rows": len(total_metric_rows),
            "inventory_rows": len(inventory_rows(targets)),
        },
        "profile_split_counts": dict(Counter(row["phase16_split"] for row in split_rows)),
        "checks": checks,
        "h0_baseline_headline": headline,
        "source_hashes": {name: sha256(path) for name, path in source_paths.items()},
        "bin_schemas": {"tp": TP_BIN_SCHEMA, "pp": PP_BIN_SCHEMA},
        "can_conclude": [
            "TP and PP Hfull teachers now share a traceable training-data contract",
            "each Hfull target has one deployment-available feature row and one compact32 H0 baseline",
            "the Phase 16 profile split is preserved without complete request-list leakage",
        ],
        "cannot_conclude": [
            "the retrained predictor is more accurate before Phase 26C/26D",
            "TP and PP native bin indices have identical payload meaning",
            "the common parameterized cost is a physical PP communication curve",
            "the teacher applies to online arrival-aware scheduling",
        ],
        "next_step": "train direct, H0, and H0+bounded-residual predictors with profile-grouped evaluation; report TP/PP and phase separately",
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(
        args.output_dir / "audit_summary.json",
        {
            "schema_version": "phase26b-unified-hfull-training-audit-v1",
            "status": status,
            "checks": checks,
            "source_hashes": summary["source_hashes"],
        },
    )
    (args.output_dir / "README.md").write_text(build_readme(summary))
    (args.output_dir / "DONE").write_text("PASS\n")

    try:
        repository_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        repository_head = "unknown"
    build_log = {
        "schema_version": "phase26b-build-log-v1",
        "status": status,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_head_at_build": repository_head,
        "python": sys.version,
        "platform": platform.platform(),
        "argv": sys.argv,
        "outputs": summary["counts"],
    }
    write_json(args.output_dir / "logs/build.log", build_log)

    manifest_rows = []
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.sha256":
            manifest_rows.append(f"{sha256(path)}  {path.relative_to(args.output_dir)}")
    (args.output_dir / "manifest.sha256").write_text("\n".join(manifest_rows) + "\n")
    print(json.dumps({"status": status, **summary["counts"], "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
