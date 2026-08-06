#!/usr/bin/env python3
"""Build analytic Qwen3-8B PatternDemand labels for every public-trace window.

The event formula used here is audited against 20 GPU replay windows at TP2/4/8
by ``finalize_phase15_trace_pattern.py``.  These labels represent one
deterministically sampled draining batch from each future window; they are not a
simulation of online interleaved arrivals.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from prepare_phase15_trace_windows import (  # noqa: E402
    BURST_FILES,
    MOONCAKE_FILES,
    load_segment,
)


CALLS_PER_FORWARD = 73
BYTES_PER_TOKEN = 8192


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument(
        "--windows",
        type=Path,
        default=REPO_ROOT / "experiment-results/phase15_trace_data/windows.csv.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "experiment-results/phase15_pattern_training_data",
    )
    parser.add_argument("--max-requests", type=int, default=8)
    parser.add_argument("--max-input-len", type=int, default=8192)
    parser.add_argument("--max-output-len", type=int, default=128)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_steps(output_lens):
    steps = [0] * 8
    if not output_lens:
        return steps
    for step in range(1, max(output_lens)):
        active = sum(length > step for length in output_lens)
        if active:
            steps[active - 1] += 1
    return steps


def make_label(row, arrays, args):
    timestamps, inputs, outputs = arrays
    cutoff = int(row["cutoff_ms"])
    horizon_ms = int(row["horizon_seconds"]) * 1000
    start = int(np.searchsorted(timestamps, cutoff, side="left"))
    end = int(np.searchsorted(timestamps, cutoff + horizon_ms, side="left"))
    future_count = end - start
    if future_count:
        take = min(future_count, args.max_requests)
        indices = np.linspace(start, end - 1, num=take, dtype=int)
        input_lens = np.clip(inputs[indices], 16, args.max_input_len).astype(int).tolist()
        output_lens = np.clip(outputs[indices], 2, args.max_output_len).astype(int).tolist()
    else:
        take, input_lens, output_lens = 0, [], []
    steps = decode_steps(output_lens)
    prefill_payload = sum(input_lens) * BYTES_PER_TOKEN if take else 0
    decode_histogram = {
        f"all_reduce:{active * BYTES_PER_TOKEN}": count * CALLS_PER_FORWARD
        for active, count in enumerate(steps, start=1)
        if count
    }
    decode_calls = sum(decode_histogram.values())
    decode_bytes = sum(
        int(key.rsplit(":", 1)[1]) * count
        for key, count in decode_histogram.items()
    )
    result = dict(row)
    result.update(
        {
            "selected_batch_size": take,
            "selected_input_lens_json": json.dumps(input_lens, separators=(",", ":")),
            "selected_output_lens_json": json.dumps(output_lens, separators=(",", ":")),
            "prefill_calls": CALLS_PER_FORWARD if take else 0,
            "prefill_payload_per_call_bytes": prefill_payload,
            "prefill_logical_payload_bytes": prefill_payload * CALLS_PER_FORWARD,
            "decode_calls": decode_calls,
            "decode_logical_payload_bytes": decode_bytes,
            "decode_histogram_json": json.dumps(
                dict(sorted(decode_histogram.items())), separators=(",", ":")
            ),
            **{
                f"decode_steps_active_{active}": steps[active - 1]
                for active in range(1, 9)
            },
        }
    )
    if int(float(row["future_count"])) != future_count:
        raise AssertionError(
            f"future count mismatch at {row['window_id']}: "
            f"csv={row['future_count']} raw={future_count}"
        )
    return result


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    windows = pd.read_csv(args.windows, compression="gzip")
    filename_by_segment = {
        segment: filename
        for filename, (segment, _) in {**BURST_FILES, **MOONCAKE_FILES}.items()
    }
    rows = []
    for segment, frame in windows.groupby("segment", sort=False):
        raw_path = args.raw_dir / filename_by_segment[segment]
        arrays = load_segment(raw_path)
        print(f"labeling {segment}: windows={len(frame)}", flush=True)
        rows.extend(
            make_label(row, arrays, args)
            for row in frame.to_dict(orient="records")
        )

    output_path = args.output_dir / "analytic_pattern_windows.csv.gz"
    with gzip.open(output_path, "wt", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    active = [row for row in rows if int(row["selected_batch_size"]) > 0]
    split_counts = {}
    active_split_counts = {}
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
        if int(row["selected_batch_size"]) > 0:
            active_split_counts[row["split"]] = active_split_counts.get(row["split"], 0) + 1
    summary = {
        "schema_version": "phase15-analytic-pattern-training-v1",
        "model": "Qwen3-8B",
        "label_provenance": (
            "analytic event formula validated against 120 GPU phase labels "
            "from 20 windows at TP2/4/8"
        ),
        "windows": len(rows),
        "active_windows": len(active),
        "split_counts": split_counts,
        "active_split_counts": active_split_counts,
        "max_requests": args.max_requests,
        "input_cap": args.max_input_len,
        "output_cap": args.max_output_len,
        "calls_per_forward": CALLS_PER_FORWARD,
        "bytes_per_token": BYTES_PER_TOKEN,
        "target_contract": "one sampled simultaneous draining batch per future window",
        "scope_boundary": (
            "Not an online continuous-batching simulator; arrival offsets and "
            "batch admission over the horizon are not replayed."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(
        f"""# Phase 15：PatternDemand 训练标签

该数据集为 {len(rows)} 个公开 trace 因果窗口生成 Qwen3-8B PatternDemand 标签，其中
{len(active)} 个窗口的未来 60 秒内至少包含一个请求。标签公式已经由正式 GPU 回放的
120 条“窗口 × TP × phase”记录逐条验证。

每个窗口从未来 60 秒确定性抽样最多 {args.max_requests} 个请求，构成一个同一时刻进入的
draining batch；记录 Prefill 消息位置、Decode `active_batch` 各档持续步数，以及精确
消息直方图。TP 不改变 logical histogram，后续按候选 TP 折算 equivalent bytes/rounds
并查询对应 L1/L2/L3 代价曲线。

边界：本数据集的标签是“下一窗口代表性 draining batch”，不是完整在线请求回放，不能
替代 continuous batching/到达交错模拟。它用于先验证历史画像预测 PatternDemand 的
训练与评测闭环。
"""
    )
    manifest = args.output_dir / "manifest.sha256"
    files = sorted(path for path in args.output_dir.iterdir() if path.is_file() and path != manifest)
    manifest.write_text("".join(f"{sha256(path)}  {path.name}\n" for path in files))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
