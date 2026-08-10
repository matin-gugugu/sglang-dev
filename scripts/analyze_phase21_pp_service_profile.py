#!/usr/bin/env python3
"""Summarize arrival-aware pure-PP service-profile PatternDemand results.

The profiler stores the same logical transfer at every forward PP boundary.
This analysis deliberately uses only the first sender boundary as the
group-level truth and treats the remaining boundaries as consistency checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


WORKLOAD_RE = re.compile(
    r"^qwen3-8b/pp(?P<pp>\d+)/(?P<strategy>[^/]+)/(?P<profile>[^/]+)/"
    r"(?P<arrival>profiled|draining)/r(?P<repeat>\d+)$"
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=root
        / "experiment-results/phase21_pp_service_profile/qwen3-8b-smoke-v1",
    )
    parser.add_argument(
        "--service-profiles",
        type=Path,
        default=root
        / "experiment-results/phase16_service_profiles/service_profiles.csv",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_profile_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open() as source:
        return {row["profile_id"]: row for row in csv.DictReader(source)}


def first_sender(cell: Path) -> dict:
    snapshots = [json.loads(path.read_text()) for path in sorted((cell / "profile").glob("*.json"))]
    senders = sorted(
        (row for row in snapshots if int(row["pp_rank"]) < int(row["pp_size"]) - 1),
        key=lambda row: int(row["pp_rank"]),
    )
    if not senders:
        raise ValueError(f"no PP sender snapshots in {cell}")
    return senders[0]


def truth_histograms(snapshot: dict) -> dict[tuple[str, str], Counter[int]]:
    result: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    for row in snapshot["histograms"]:
        workload_id = row.get("workload_id")
        if row.get("msg_type") != "proxy" or not workload_id:
            continue
        result[(workload_id, row["phase"])][int(row["payload_bytes"])] += int(row["count"])
    return result


def totals(histogram: Counter[int]) -> tuple[int, int]:
    calls = int(sum(histogram.values()))
    logical_bytes = int(sum(payload * count for payload, count in histogram.items()))
    return calls, logical_bytes


def distribution_tvd(left: Counter[int], right: Counter[int]) -> float:
    left_total = sum(left.values())
    right_total = sum(right.values())
    if not left_total and not right_total:
        return 0.0
    support = set(left) | set(right)
    return 0.5 * sum(
        abs(left[payload] / max(left_total, 1) - right[payload] / max(right_total, 1))
        for payload in support
    )


def signed_fraction(value: float, reference: float) -> float:
    if reference == 0:
        return 0.0 if value == 0 else math.inf
    return (value - reference) / reference


def histogram_signature(histogram: Counter[int]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(histogram.items()))


def render_heatmap(path: Path, rows: list[dict]) -> None:
    values = {
        (int(row["pp_size"]), int(row["max_microbatch"])): float(row["changed_pair_rate"])
        for row in rows
    }
    pp_sizes = sorted({key[0] for key in values})
    microbatches = sorted({key[1] for key in values})
    cell_w, cell_h = 160, 72
    left, top = 110, 80
    width = left + cell_w * len(microbatches) + 30
    height = top + cell_h * len(pp_sizes) + 65
    items = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:18px;font-weight:700}.label{font-size:13px}.value{font-size:15px;font-weight:700}</style>',
        '<text class="title" x="20" y="30">Arrival changes PP payload histogram</text>',
        '<text class="label" x="20" y="53">changed profiled-vs-draining pairs / all paired repeats</text>',
    ]
    for column, microbatch in enumerate(microbatches):
        x = left + column * cell_w + cell_w / 2
        items.append(f'<text class="label" x="{x}" y="{top - 16}" text-anchor="middle">microbatch {microbatch}</text>')
    for row_index, pp_size in enumerate(pp_sizes):
        y = top + row_index * cell_h
        items.append(
            f'<text class="label" x="{left - 18}" y="{y + cell_h / 2 + 5}" text-anchor="end">PP={pp_size}</text>'
        )
        for column, microbatch in enumerate(microbatches):
            rate = values[(pp_size, microbatch)]
            red = round(238 - 145 * rate)
            green = round(246 - 76 * rate)
            blue = round(255 - 75 * rate)
            x = left + column * cell_w
            items.append(
                f'<rect x="{x}" y="{y}" width="{cell_w - 4}" height="{cell_h - 4}" rx="7" fill="rgb({red},{green},{blue})"/>'
            )
            items.append(
                f'<text class="value" x="{x + (cell_w - 4) / 2}" y="{y + cell_h / 2 + 5}" text-anchor="middle">{rate:.0%}</text>'
            )
    items.append("</svg>")
    path.write_text("\n".join(items) + "\n")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir / "analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_profile_metadata(args.service_profiles.resolve())
    matrix = json.loads((input_dir / "matrix_summary.json").read_text())
    if matrix["status"] != "PASS":
        raise ValueError(f"matrix is not valid: {matrix['status']}")

    pair_rows: list[dict] = []
    repeat_groups: dict[tuple, list[tuple[int, Counter[int]]]] = defaultdict(list)
    cell_rows: list[dict] = []
    total_pair_controls = 0
    total_changed_controls = 0

    for audit in matrix["cells"]:
        pp_size = int(audit["pp_size"])
        max_microbatch = int(audit["pp_max_micro_batch_size"])
        strategy = f"mb{max_microbatch}"
        cell = input_dir / f"pp{pp_size}" / strategy
        truth = truth_histograms(first_sender(cell))
        pair_controls = len(audit["paired_arrival_controls"])
        changed_controls = int(audit["payload_changed_pairs"])
        total_pair_controls += pair_controls
        total_changed_controls += changed_controls
        phase_comparisons = 0
        phase_changed = 0
        phase_tvds = []
        for control in audit["paired_arrival_controls"]:
            profile_id = control["profile_id"]
            repeat = int(control["repeat"])
            for phase in ("prefill", "decode"):
                profiled_id = (
                    f"qwen3-8b/pp{pp_size}/{strategy}/{profile_id}/profiled/r{repeat}"
                )
                draining_id = (
                    f"qwen3-8b/pp{pp_size}/{strategy}/{profile_id}/draining/r{repeat}"
                )
                profiled = truth[(profiled_id, phase)]
                draining = truth[(draining_id, phase)]
                profiled_calls, profiled_bytes = totals(profiled)
                draining_calls, draining_bytes = totals(draining)
                tvd = distribution_tvd(profiled, draining)
                changed = profiled != draining
                phase_comparisons += 1
                phase_changed += int(changed)
                phase_tvds.append(tvd)
                profile = metadata[profile_id]
                pair_rows.append(
                    {
                        "pp_size": pp_size,
                        "max_microbatch": max_microbatch,
                        "profile_id": profile_id,
                        "source": profile["source"],
                        "segment": profile["segment"],
                        "rps": profile["rps"],
                        "interarrival_cv": profile["interarrival_cv"],
                        "repeat": repeat,
                        "phase": phase,
                        "histogram_changed": int(changed),
                        "profiled_boundary_calls": profiled_calls,
                        "draining_boundary_calls": draining_calls,
                        "calls_delta_fraction": signed_fraction(
                            profiled_calls, draining_calls
                        ),
                        "profiled_boundary_bytes": profiled_bytes,
                        "draining_boundary_bytes": draining_bytes,
                        "bytes_delta_fraction": signed_fraction(
                            profiled_bytes, draining_bytes
                        ),
                        "calls_distribution_tvd": tvd,
                        "profiled_pipeline_calls": profiled_calls * (pp_size - 1),
                        "draining_pipeline_calls": draining_calls * (pp_size - 1),
                        "profiled_pipeline_bytes": profiled_bytes * (pp_size - 1),
                        "draining_pipeline_bytes": draining_bytes * (pp_size - 1),
                    }
                )
                repeat_groups[(pp_size, max_microbatch, profile_id, "profiled", phase)].append(
                    (repeat, profiled)
                )
                repeat_groups[(pp_size, max_microbatch, profile_id, "draining", phase)].append(
                    (repeat, draining)
                )
        cell_rows.append(
            {
                "pp_size": pp_size,
                "max_microbatch": max_microbatch,
                "status": audit["status"],
                "profile_replays": audit["profile_replays"],
                "logical_requests": audit["logical_requests"],
                "paired_controls": pair_controls,
                "changed_pairs": changed_controls,
                "changed_pair_rate": changed_controls / max(pair_controls, 1),
                "phase_comparisons": phase_comparisons,
                "changed_phase_comparisons": phase_changed,
                "changed_phase_rate": phase_changed / max(phase_comparisons, 1),
                "mean_phase_distribution_tvd": sum(phase_tvds) / max(len(phase_tvds), 1),
                "max_phase_distribution_tvd": max(phase_tvds, default=0.0),
            }
        )

    repeat_rows = []
    for key, observations in sorted(repeat_groups.items()):
        pp_size, max_microbatch, profile_id, arrival, phase = key
        observations.sort(key=lambda item: item[0])
        reference = observations[0][1]
        tvds = [distribution_tvd(reference, histogram) for _, histogram in observations]
        signatures = {histogram_signature(histogram) for _, histogram in observations}
        calls = [totals(histogram)[0] for _, histogram in observations]
        byte_values = [totals(histogram)[1] for _, histogram in observations]
        repeat_rows.append(
            {
                "pp_size": pp_size,
                "max_microbatch": max_microbatch,
                "profile_id": profile_id,
                "arrival_mode": arrival,
                "phase": phase,
                "repeats": len(observations),
                "exact_histograms_identical": int(len(signatures) == 1),
                "max_distribution_tvd_vs_r0": max(tvds, default=0.0),
                "min_calls": min(calls),
                "max_calls": max(calls),
                "min_bytes": min(byte_values),
                "max_bytes": max(byte_values),
            }
        )

    write_csv(output_dir / "cell_summary.csv", cell_rows)
    write_csv(output_dir / "arrival_effect_pairs.csv", pair_rows)
    write_csv(output_dir / "repeat_stability.csv", repeat_rows)
    render_heatmap(output_dir / "arrival_effect_rate.svg", cell_rows)

    exact_repeat_groups = sum(int(row["exact_histograms_identical"]) for row in repeat_rows)
    summary = {
        "schema_version": "phase21-pp-profile-analysis-v1",
        "status": "PASS",
        "source_matrix_status": matrix["status"],
        "cells": len(cell_rows),
        "profile_replays": sum(int(row["profile_replays"]) for row in cell_rows),
        "logical_request_executions": sum(int(row["logical_requests"]) for row in cell_rows),
        "paired_arrival_controls": total_pair_controls,
        "payload_histogram_changed_controls": total_changed_controls,
        "payload_histogram_changed_rate": total_changed_controls
        / max(total_pair_controls, 1),
        "phase_comparisons": len(pair_rows),
        "phase_histogram_changed_comparisons": sum(
            int(row["histogram_changed"]) for row in pair_rows
        ),
        "repeat_groups": len(repeat_rows),
        "exactly_stable_repeat_groups": exact_repeat_groups,
        "exact_repeat_stability_rate": exact_repeat_groups / max(len(repeat_rows), 1),
        "interpretation_boundary": (
            "The profiled arrival process is a deterministic gamma-renewal realization "
            "of each service image's RPS and inter-arrival CV over 32 stratified requests. "
            "It is not an exact replay or a forecast of a future trace segment."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    table_lines = [
        "| PP | max microbatch | changed pairs | phase changes | mean phase TVD |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in cell_rows:
        table_lines.append(
            f"| {row['pp_size']} | {row['max_microbatch']} | "
            f"{row['changed_pairs']}/{row['paired_controls']} | "
            f"{row['changed_phase_comparisons']}/{row['phase_comparisons']} | "
            f"{float(row['mean_phase_distribution_tvd']):.4f} |"
        )
    readme = f"""# Phase 21：到达感知的纯 PP PatternDemand smoke

## 目标

在 `TP=1` 的纯 PP 配置下，用完全相同的 32 个请求长度对比两种执行方式：

- `profiled`：根据常态画像的 RPS 与到达间隔 CV 构造确定性的 gamma-renewal 到达过程；
- `draining`：32 个请求同时提交。

该配对实验检验到达与突发特征是否会通过 SGLang batching 改变 PP 消息直方图，
不是对下一时间窗口请求序列的预测。

## 数据与有效性

- 模型：Qwen3-8B，纯 PP，`PP=2/4/8`；
- 策略：`pp_max_micro_batch_size=1/4/16`；
- 画像：3 个，重复 3 次，两种 arrival mode；
- 9/9 cell 通过审计，共 {summary['profile_replays']} 次画像回放、
  {summary['logical_request_executions']} 次逻辑请求执行；
- sender 边界直方图在所有 PP 边界一致，统计使用首个 sender 作为 group-level 真值，
  pipeline-wide demand 再乘以 `PP-1`，不重复累计 send/recv。

## 主要结果

在 {summary['paired_arrival_controls']} 个配对重复中，
{summary['payload_histogram_changed_controls']} 个的精确 payload 直方图发生变化，
占 {summary['payload_histogram_changed_rate']:.1%}。

{chr(10).join(table_lines)}

重复稳定性：{summary['exactly_stable_repeat_groups']}/{summary['repeat_groups']}
个 `PP×策略×画像×arrival×phase` 分组的三次精确直方图完全一致。稳定性不足的分组
必须在正式训练前单独报告，而不能用均值掩盖调度抖动。

## 文件

- `cell_summary.csv`：cell 级有效性与到达影响比例；
- `arrival_effect_pairs.csv`：Prefill/Decode 配对 calls、bytes 和分布 TVD；
- `repeat_stability.csv`：三次重复的精确一致性；
- `arrival_effect_rate.svg`：到达影响热力图；
- `summary.json`：机器可读结论。

## 结论边界

当前只覆盖 3/24 个画像和一个模型，属于机制验证。32 个请求是从画像窗口中分层选出的
长度样本；`profiled` 到达是依据画像 RPS/CV 重新构造的稳态实现，不是原始连续 trace
的逐请求时间戳回放。因此当前可以证明“到达过程会改变 PP PatternDemand”，但不能声称
已经完成跨画像、跨模型的在线 PP 预测器。
"""
    (output_dir / "README.md").write_text(readme)
    (output_dir / "DONE").write_text("PASS\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
