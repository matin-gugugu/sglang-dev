#!/usr/bin/env python3
"""Validate the transparent TP PatternDemand base formula H0."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root
        / "experiment-results/phase14c/extended_dataset_analysis/aggregated_configurations.csv",
    )
    parser.add_argument(
        "--model-features",
        type=Path,
        default=root / "experiment-results/phase16_model_features/model_features.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase16_h0_validation",
    )
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def canonical_observed(source):
    histogram = Counter()
    raw_ops = Counter()
    for text, calls in json.loads(source["calls_by_op_payload_json"]).items():
        op, payload_text = text.rsplit(":", 1)
        payload, calls = int(payload_text), int(calls)
        histogram[payload] += calls
        raw_ops[op] += calls
    return histogram, raw_ops


def h0_prefill(source, features):
    batch = int(source["batch_size"])
    length = int(source["input_len"])
    chunk = int(source["prefill_chunk_size"] or 0)
    chunk = chunk if chunk > 0 else length
    calls_per_forward = int(features["logical_collectives_per_forward_prior"])
    bytes_per_token = int(features["payload_bytes_per_active_token_prior"])
    predicted = Counter()
    position = 0
    while position < length:
        active_tokens = batch * min(chunk, length - position)
        predicted[active_tokens * bytes_per_token] += calls_per_forward
        position += chunk
    return predicted


def h0_decode(source, features):
    output_lengths = [int(value) for value in json.loads(source["output_lens_json"])]
    calls_per_forward = int(features["logical_collectives_per_forward_prior"])
    bytes_per_token = int(features["payload_bytes_per_active_token_prior"])
    predicted = Counter()
    # The prefill forward samples the first generated token. Therefore a request
    # with actual output length M contributes Decode forwards at t=1,...,M-1.
    for step in range(1, max(output_lengths)):
        active_batch = sum(length > step for length in output_lengths)
        if active_batch:
            predicted[active_batch * bytes_per_token] += calls_per_forward
    return predicted


def histogram_error(observed, predicted):
    support = sorted(set(observed) | set(predicted))
    call_l1 = sum(abs(predicted[key] - observed[key]) for key in support)
    calls = sum(observed.values())
    observed_bytes = sum(payload * observed[payload] for payload in observed)
    predicted_bytes = sum(payload * predicted[payload] for payload in predicted)
    return {
        "observed_calls": calls,
        "predicted_calls": sum(predicted.values()),
        "call_absolute_error": call_l1,
        "histogram_l1_normalized": call_l1 / calls if calls else 0.0,
        "observed_logical_bytes": observed_bytes,
        "predicted_logical_bytes": predicted_bytes,
        "logical_bytes_absolute_error": abs(predicted_bytes - observed_bytes),
        "exact_histogram_match": dict(observed) == dict(predicted),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features = {
        row["model"]: row for row in json.loads(args.model_features.read_text())
    }
    rows = []
    raw_op_totals = defaultdict(Counter)
    for source in read_csv(args.dataset):
        model = source["model"]
        observed, raw_ops = canonical_observed(source)
        raw_op_totals[model].update(raw_ops)
        predicted = (
            h0_prefill(source, features[model])
            if source["phase"] == "prefill"
            else h0_decode(source, features[model])
        )
        rows.append(
            {
                "workload_id": source["workload_id"],
                "model": model,
                "tp": int(source["tp"]),
                "phase": source["phase"],
                "mode": source["mode"],
                "case_label": source["case_label"],
                "observed_histogram_json": json.dumps(dict(sorted(observed.items())), separators=(",", ":")),
                "predicted_histogram_json": json.dumps(dict(sorted(predicted.items())), separators=(",", ":")),
                **histogram_error(observed, predicted),
            }
        )

    with (args.output_dir / "predictions.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    scopes = {}
    for scope in ["all", "prefill", "decode"] + sorted(features):
        if scope == "all":
            selected = rows
        elif scope in {"prefill", "decode"}:
            selected = [row for row in rows if row["phase"] == scope]
        else:
            selected = [row for row in rows if row["model"] == scope]
        total_calls = sum(row["observed_calls"] for row in selected)
        total_bytes = sum(row["observed_logical_bytes"] for row in selected)
        scopes[scope] = {
            "samples": len(selected),
            "exact_matches": sum(row["exact_histogram_match"] for row in selected),
            "calls_wape": sum(row["call_absolute_error"] for row in selected) / total_calls,
            "logical_bytes_wape": sum(row["logical_bytes_absolute_error"] for row in selected) / total_bytes,
            "mean_histogram_l1": sum(row["histogram_l1_normalized"] for row in selected) / len(selected),
        }

    tp_invariance = True
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["workload_id"]].append(row["observed_histogram_json"])
    for values in grouped.values():
        tp_invariance &= len(set(values)) == 1

    summary = {
        "schema_version": "profiledemand-h0-validation-v1",
        "canonical_op": "all_reduce",
        "formula": {
            "calls_per_forward": "2 * num_hidden_layers + 1",
            "payload_bytes": "active_tokens * hidden_size * dtype_bytes",
            "prefill_active_tokens": "batch_size * tokens_in_current_prefill_chunk",
            "decode_active_batch": "A(t) = sum_i 1(actual_output_len_i > t), t=1,...,max(M)-1; the first token is sampled by prefill",
        },
        "rows": len(rows),
        "scopes": scopes,
        "logical_histogram_tp_invariant": tp_invariance,
        "raw_op_totals_audit_only": {
            model: dict(counter) for model, counter in raw_op_totals.items()
        },
        "dataset_sha256": sha256(args.dataset),
        "model_features_sha256": sha256(args.model_features),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    readme = f"""# Phase 16D：ProfileDemand 基础结构公式 H0 验证

H0 仅使用模型静态结构、实际工作负载长度和 chunk 规则：

- 每个 TP transformer forward 的 canonical logical AllReduce 次数为 `2L+1`；
- 单次 logical payload 为 `active_tokens×hidden_size×dtype_bytes`；
- Prefill 的 active tokens 由 batch 和 chunk 决定；
- Decode 使用 `A(t)=Σ_i 1(M_i>t)`，其中 `M_i` 是实际生成长度，且
  `t=1,...,max(M)-1`；第一个输出 token 由 Prefill forward 采样，不产生额外 Decode
  forward。

在三个模型、TP2/4/8、Prefill/Decode 共 {len(rows)} 个聚合配置上，canonical 精确匹配
为 {scopes['all']['exact_matches']}/{len(rows)}，calls WAPE、logical-bytes WAPE 与平均
histogram L1 均为 0；同一 workload 的 logical histogram 跨 TP 完全不变。

这不意味着正式 ProfileDemand v1 不需要学习：本验证向 H0 提供了每个 batch 的完整
实际长度。正式模型只看到低维服务画像和执行策略，DNN residual 负责修正画像分桶内
形态、batch 形成和实现边界。Qwen3-30B-A3B 的 fused raw-op 拆分是当前 backend 的
实现细节，未泄漏进 canonical H0；它由第二阶段 backend-aware 映射处理。
"""
    (args.output_dir / "README.md").write_text(readme)
    checks = {
        "rows_162": len(rows) == 162,
        "all_canonical_histograms_exact": scopes["all"]["exact_matches"] == len(rows),
        "zero_calls_wape": scopes["all"]["calls_wape"] == 0.0,
        "zero_logical_bytes_wape": scopes["all"]["logical_bytes_wape"] == 0.0,
        "logical_histogram_tp_invariant": tp_invariance,
    }
    audit = {
        "schema_version": "profiledemand-h0-validation-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    (args.output_dir / "audit_summary.json").write_text(json.dumps(audit, indent=2) + "\n")
    if audit["status"] != "PASS":
        raise RuntimeError(audit)
    (args.output_dir / "DONE").write_text("PASS\n")
    (args.output_dir / "run.log").write_text(json.dumps({"checks": checks}, indent=2) + "\n")
    files = sorted(
        path for path in args.output_dir.iterdir() if path.is_file() and path.name != "manifest.sha256"
    )
    (args.output_dir / "manifest.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files)
    )
    print(json.dumps({"scopes": scopes, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
