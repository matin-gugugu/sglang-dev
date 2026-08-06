#!/usr/bin/env python3
"""Normalize public LLM traces into causal windows and a 20-window replay smoke plan."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


BURST_FILES = {
    "BurstGPT_without_fails_1.csv": ("burstgpt_1", "train"),
    "BurstGPT_without_fails_2.csv": ("burstgpt_2", "validation_test"),
    "BurstGPT_without_fails_3.csv": ("burstgpt_3", "temporal_test"),
}
MOONCAKE_FILES = {
    "conversation_trace.jsonl": ("mooncake_conversation", "external_test"),
    "toolagent_trace.jsonl": ("mooncake_toolagent", "external_test"),
    "synthetic_trace.jsonl": ("mooncake_synthetic", "external_synthetic"),
}
LENGTH_BINS = list(range(18))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--history-seconds", type=int, default=300)
    parser.add_argument("--horizon-seconds", type=int, default=60)
    parser.add_argument("--burst-stride-seconds", type=int, default=300)
    parser.add_argument("--mooncake-stride-seconds", type=int, default=60)
    parser.add_argument("--smoke-windows", type=int, default=20)
    parser.add_argument("--max-requests-per-window", type=int, default=8)
    parser.add_argument("--max-input-len", type=int, default=8192)
    parser.add_argument("--max-output-len", type=int, default=128)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log_hist(values):
    if not len(values):
        return [0] * len(LENGTH_BINS)
    bins = np.clip(np.floor(np.log2(np.maximum(values, 1))).astype(int), 0, 17)
    return np.bincount(bins, minlength=18).astype(int).tolist()


def quantile(values, q):
    return float(np.quantile(values, q)) if len(values) else 0.0


def safe_cv(values):
    if len(values) < 2:
        return 0.0
    mean = float(np.mean(values))
    return float(np.std(values) / mean) if mean else 0.0


def load_segment(path):
    if path.suffix == ".csv":
        frame = pd.read_csv(
            path,
            usecols=["Timestamp", "Request tokens", "Response tokens"],
            dtype={"Timestamp": "float64", "Request tokens": "int32", "Response tokens": "int32"},
        )
        timestamp_ms = np.rint(frame["Timestamp"].to_numpy() * 1000).astype(np.int64)
        input_len = frame["Request tokens"].to_numpy(dtype=np.int64)
        output_len = frame["Response tokens"].to_numpy(dtype=np.int64)
    else:
        timestamps, inputs, outputs = [], [], []
        with path.open() as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                timestamps.append(int(row["timestamp"]))
                inputs.append(int(row["input_length"]))
                outputs.append(int(row["output_length"]))
        timestamp_ms = np.asarray(timestamps, dtype=np.int64)
        input_len = np.asarray(inputs, dtype=np.int64)
        output_len = np.asarray(outputs, dtype=np.int64)
    order = np.argsort(timestamp_ms, kind="stable")
    return timestamp_ms[order], input_len[order], output_len[order]


def split_for(segment, default_split, cutoff, min_time, max_time):
    if default_split != "validation_test":
        return default_split
    ratio = (cutoff - min_time) / max(max_time - min_time, 1)
    return "validation" if ratio < 0.5 else "test"


def summarize_window(segment, source, split, cutoff, history, future, args):
    ht, hl, hm = history
    ft, fl, fm = future
    if len(ht):
        interarrival = np.diff(ht) / 1000.0
        second_bins = np.floor((ht - (cutoff - args.history_seconds * 1000)) / 1000).astype(int)
        second_bins = np.clip(second_bins, 0, args.history_seconds - 1)
        per_second = np.bincount(second_bins, minlength=args.history_seconds)
    else:
        interarrival = np.asarray([], dtype=np.float64)
        per_second = np.zeros(args.history_seconds, dtype=np.int64)
    positive_mean = float(np.mean(per_second))
    peak_to_mean = float(np.max(per_second) / positive_mean) if positive_mean else 0.0
    fano = float(np.var(per_second) / positive_mean) if positive_mean else 0.0
    correlation = (
        float(np.corrcoef(hl, hm)[0, 1])
        if len(hl) > 1 and np.std(hl) > 0 and np.std(hm) > 0
        else 0.0
    )
    return {
        "window_id": f"{segment}-{cutoff}",
        "source": source,
        "segment": segment,
        "split": split,
        "cutoff_ms": int(cutoff),
        "history_seconds": args.history_seconds,
        "horizon_seconds": args.horizon_seconds,
        "history_count": len(ht),
        "history_rps": len(ht) / args.history_seconds,
        "history_interarrival_cv": safe_cv(interarrival),
        "history_peak_to_mean_1s": peak_to_mean,
        "history_fano_1s": fano,
        "history_input_mean": float(np.mean(hl)) if len(hl) else 0.0,
        "history_input_p50": quantile(hl, 0.50),
        "history_input_p90": quantile(hl, 0.90),
        "history_input_p99": quantile(hl, 0.99),
        "history_output_mean": float(np.mean(hm)) if len(hm) else 0.0,
        "history_output_p50": quantile(hm, 0.50),
        "history_output_p90": quantile(hm, 0.90),
        "history_output_p99": quantile(hm, 0.99),
        "history_lm_correlation": correlation,
        "history_input_log2_hist": json.dumps(log_hist(hl), separators=(",", ":")),
        "history_output_log2_hist": json.dumps(log_hist(hm), separators=(",", ":")),
        "future_count": len(ft),
        "future_rps": len(ft) / args.horizon_seconds,
        "future_input_mean": float(np.mean(fl)) if len(fl) else 0.0,
        "future_input_p90": quantile(fl, 0.90),
        "future_output_mean": float(np.mean(fm)) if len(fm) else 0.0,
        "future_output_p90": quantile(fm, 0.90),
        "future_input_log2_hist": json.dumps(log_hist(fl), separators=(",", ":")),
        "future_output_log2_hist": json.dumps(log_hist(fm), separators=(",", ":")),
    }


def build_windows(path, segment, default_split, args):
    timestamps, inputs, outputs = load_segment(path)
    source = "burstgpt" if path.suffix == ".csv" else "mooncake"
    stride = (
        args.burst_stride_seconds if source == "burstgpt" else args.mooncake_stride_seconds
    ) * 1000
    history_ms = args.history_seconds * 1000
    horizon_ms = args.horizon_seconds * 1000
    start = int(timestamps[0] + history_ms)
    end = int(timestamps[-1] - horizon_ms)
    cutoffs = np.arange(start, end + 1, stride, dtype=np.int64)
    rows = []
    for cutoff in cutoffs:
        h0 = int(np.searchsorted(timestamps, cutoff - history_ms, side="left"))
        h1 = int(np.searchsorted(timestamps, cutoff, side="left"))
        f1 = int(np.searchsorted(timestamps, cutoff + horizon_ms, side="left"))
        split = split_for(segment, default_split, cutoff, int(timestamps[0]), int(timestamps[-1]))
        rows.append(
            summarize_window(
                segment,
                source,
                split,
                cutoff,
                (timestamps[h0:h1], inputs[h0:h1], outputs[h0:h1]),
                (timestamps[h1:f1], inputs[h1:f1], outputs[h1:f1]),
                args,
            )
        )
    return rows, (timestamps, inputs, outputs)


def choose_diverse(rows, count):
    candidates = [row for row in rows if int(row["future_count"]) >= 2]
    if len(candidates) <= count:
        return candidates
    fields = [
        "history_rps",
        "history_interarrival_cv",
        "history_peak_to_mean_1s",
        "future_input_p90",
        "future_output_p90",
        "future_count",
    ]
    matrix = np.asarray([[float(row[field]) for field in fields] for row in candidates])
    median = np.median(matrix, axis=0)
    scale = np.quantile(matrix, 0.75, axis=0) - np.quantile(matrix, 0.25, axis=0)
    scale[scale == 0] = 1.0
    normalized = (matrix - median) / scale
    selected = [int(np.argmin(np.linalg.norm(normalized, axis=1)))]
    while len(selected) < count:
        distance = np.min(
            np.stack(
                [np.linalg.norm(normalized - normalized[index], axis=1) for index in selected]
            ),
            axis=0,
        )
        distance[selected] = -1
        selected.append(int(np.argmax(distance)))
    return [candidates[index] for index in selected]


def smoke_quota(row):
    key = (row["segment"], row["split"])
    quotas = {
        ("burstgpt_1", "train"): 4,
        ("burstgpt_2", "validation"): 3,
        ("burstgpt_2", "test"): 3,
        ("burstgpt_3", "temporal_test"): 4,
        ("mooncake_conversation", "external_test"): 3,
        ("mooncake_toolagent", "external_test"): 3,
    }
    return quotas.get(key, 0)


def make_replay_plan(selected, arrays_by_segment, args):
    plans = []
    for row in selected:
        timestamps, inputs, outputs = arrays_by_segment[row["segment"]]
        cutoff = int(row["cutoff_ms"])
        start = int(np.searchsorted(timestamps, cutoff, side="left"))
        end = int(
            np.searchsorted(
                timestamps, cutoff + args.horizon_seconds * 1000, side="left"
            )
        )
        count = end - start
        take = min(count, args.max_requests_per_window)
        indices = np.linspace(start, end - 1, num=take, dtype=int)
        request_times = timestamps[indices]
        input_lens = np.clip(inputs[indices], 16, args.max_input_len).astype(int)
        output_lens = np.clip(outputs[indices], 2, args.max_output_len).astype(int)
        plans.append(
            {
                "workload_id": row["window_id"],
                "source": row["source"],
                "segment": row["segment"],
                "split": row["split"],
                "cutoff_ms": cutoff,
                "trace_replay_mode": "draining_batch_a_i_zero",
                "original_future_request_count": count,
                "selected_request_count": take,
                "arrival_offsets_ms_audit_only": (request_times - request_times[0]).astype(int).tolist(),
                "input_lens_per_request": input_lens.tolist(),
                "output_lens_per_request": output_lens.tolist(),
                "input_cap": args.max_input_len,
                "output_cap": args.max_output_len,
                "history_features": {
                    key: row[key]
                    for key in row
                    if key.startswith("history_") and not key.endswith("_hist")
                },
            }
        )
    return plans


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    arrays = {}
    segment_meta = []
    for name, (segment, default_split) in {**BURST_FILES, **MOONCAKE_FILES}.items():
        path = args.raw_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        rows, segment_arrays = build_windows(path, segment, default_split, args)
        all_rows.extend(rows)
        arrays[segment] = segment_arrays
        segment_meta.append(
            {
                "segment": segment,
                "file": name,
                "requests": len(segment_arrays[0]),
                "windows": len(rows),
                "sha256": sha256(path),
            }
        )
        print(f"prepared {segment}: requests={len(segment_arrays[0])} windows={len(rows)}")

    grouped = {}
    for row in all_rows:
        quota = smoke_quota(row)
        if quota:
            grouped.setdefault((row["segment"], row["split"]), []).append(row)
    selected = []
    for group, rows in sorted(grouped.items()):
        quota = smoke_quota(rows[0])
        selected.extend(choose_diverse(rows, quota))
    if len(selected) != args.smoke_windows:
        raise RuntimeError(f"expected {args.smoke_windows} smoke windows, got {len(selected)}")
    plans = make_replay_plan(selected, arrays, args)

    windows_path = args.output_dir / "windows.csv.gz"
    with gzip.open(windows_path, "wt", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    with (args.output_dir / "smoke_replay_plan.jsonl").open("w") as output:
        for plan in plans:
            output.write(json.dumps(plan, separators=(",", ":")) + "\n")
    selected_ids = {row["window_id"] for row in selected}
    selected_rows = [row for row in all_rows if row["window_id"] in selected_ids]
    with (args.output_dir / "smoke_window_features.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(selected_rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)

    source_manifest = json.loads((args.raw_dir / "source_manifest.json").read_text())
    for record in source_manifest["sources"]:
        record.pop("path", None)
    (args.output_dir / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2) + "\n"
    )
    split_counts = {}
    for row in all_rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    summary = {
        "schema_version": "phase15-trace-windows-v1",
        "history_seconds": args.history_seconds,
        "horizon_seconds": args.horizon_seconds,
        "windows": len(all_rows),
        "split_counts": split_counts,
        "segments": segment_meta,
        "smoke_windows": len(plans),
        "smoke_mode": "draining_batch_a_i_zero; arrival offsets retained for later online replay",
        "caps": {
            "max_requests": args.max_requests_per_window,
            "input_len": args.max_input_len,
            "output_len": args.max_output_len,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    readme = f"""# Phase 15：公开真实流量窗口数据

数据源为 BurstGPT v2.0 无失败请求和 Mooncake FAST'25 官方 trace。全部文件的 URL、
大小与 SHA-256 见 `source_manifest.json`。

- 历史窗口：{args.history_seconds} 秒；
- 预测窗口：{args.horizon_seconds} 秒；
- 窗口总数：{len(all_rows)}；
- Qwen3-8B smoke 计划：{len(plans)} 个窗口，每个最多 {args.max_requests_per_window} 个请求；
- 输入/输出长度上限：{args.max_input_len}/{args.max_output_len}。

BurstGPT 使用时间顺序划分；Mooncake 只作为 external test。当前 smoke 把每个未来窗口
抽样请求作为同一时刻进入的 draining batch，`arrival_offsets_ms_audit_only` 被保留，
但尚未执行真正在线交错到达。因此本阶段不能声称 arrival/burst 对 PatternDemand 的
物理影响已经验证。

产物：

- `windows.csv.gz`：完整因果窗口特征；
- `smoke_window_features.csv`：20 个 smoke 窗口输入特征；
- `smoke_replay_plan.jsonl`：固定长度 Qwen3-8B 回放计划；
- `source_manifest.json`、`summary.json`。
"""
    (args.output_dir / "README.md").write_text(readme)
    manifest = args.output_dir / "manifest.sha256"
    files = sorted(path for path in args.output_dir.iterdir() if path.is_file() and path != manifest)
    manifest.write_text("".join(f"{sha256(path)}  {path.name}\n" for path in files))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
