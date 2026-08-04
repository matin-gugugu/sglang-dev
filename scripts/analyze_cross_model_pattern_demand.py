#!/usr/bin/env python3
"""Compare histogram-only PatternDemand datasets across model families."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


OP_FACTORS = {
    "all_reduce": {
        "bytes": lambda p: 2 * (p - 1) / p,
        "rounds": lambda p: 2 * (p - 1),
    },
    # This fused runtime op still performs an AllReduce. Keep the raw op in
    # histogram_json, but use the AllReduce family factors for topology-aware
    # equivalent bytes and rounds.
    "fused_allreduce_residual_rmsnorm": {
        "bytes": lambda p: 2 * (p - 1) / p,
        "rounds": lambda p: 2 * (p - 1),
    },
    "all_gather": {
        "bytes": lambda p: (p - 1) / p,
        "rounds": lambda p: p - 1,
    },
}


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=2,
        metavar=("MODEL", "DIRECTORY"),
        help=(
            "Model label and histogram-only result root. Repeat for every "
            "model. Defaults to the Qwen3-8B and DeepSeek-V2-Lite datasets."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "experiment-results"
        / "phase8"
        / "cross_model_pattern_analysis",
    )
    args = parser.parse_args()
    if args.dataset is None:
        args.dataset = [
            (
                "qwen3-8b",
                str(
                    repo_root
                    / "experiment-results"
                    / "phase6"
                    / "qwen3_8b_corrected_all_rank"
                ),
            ),
            (
                "deepseek-v2-lite",
                str(
                    repo_root
                    / "experiment-results"
                    / "phase8"
                    / "deepseek_v2_lite_pattern_demand"
                ),
            ),
        ]
    return args


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"fieldnames required for empty CSV: {path}")
        fieldnames = list(rows[0])
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_path(path):
    phase = path.parent.name
    repeat_name = path.parent.parent.name
    tp_name = path.parent.parent.parent.name
    if phase not in {"prefill", "decode"}:
        raise ValueError(f"unexpected phase path: {path}")
    if not repeat_name.startswith("r") or not tp_name.startswith("tp"):
        raise ValueError(f"unexpected dataset path: {path}")
    return phase, int(tp_name[2:]), int(repeat_name[1:])


def representative_histograms(record, phase, tp):
    profiles = sorted(record["comm_profile"], key=lambda item: item["tp_rank"])
    if [profile["tp_rank"] for profile in profiles] != list(range(tp)):
        raise ValueError(f"{record['run_name']}: missing TP ranks")
    reference = profiles[0]
    if (
        reference["capture_mode"] != "histogram-only"
        or reference["raw_events_saved"]
        or reference["events"]
        or reference["events_truncated"]
    ):
        raise ValueError(f"{record['run_name']}: not a compact full histogram")
    for profile in profiles[1:]:
        if (
            profile["stats"] != reference["stats"]
            or profile["event_histograms"] != reference["event_histograms"]
        ):
            raise ValueError(f"{record['run_name']}: rank histograms differ")
    histograms = [
        item for item in reference["event_histograms"] if item["phase"] == phase
    ]
    if not histograms:
        raise ValueError(f"{record['run_name']}: empty {phase} histogram")
    return histograms


def aggregate_histograms(histograms, tp):
    calls_by_key = defaultdict(int)
    for item in histograms:
        group_size = int(item["group_size"])
        if group_size != tp:
            raise ValueError(f"unexpected group size: {group_size} != {tp}")
        key = (item["op"], int(item["input_payload_bytes"]))
        calls_by_key[key] += int(item["count"])

    calls = 0
    logical_payload_bytes = 0
    ring_equivalent_bytes = 0.0
    ring_equivalent_rounds = 0
    for (op, payload), count in calls_by_key.items():
        if op not in OP_FACTORS:
            raise ValueError(
                f"no equivalent bytes/rounds factor for collective op {op}"
            )
        calls += count
        logical_payload_bytes += payload * count
        ring_equivalent_bytes += (
            payload * count * OP_FACTORS[op]["bytes"](tp)
        )
        ring_equivalent_rounds += count * OP_FACTORS[op]["rounds"](tp)

    histogram = {
        f"{op}:{payload}": count
        for (op, payload), count in sorted(calls_by_key.items())
    }
    payloads = sorted({payload for _, payload in calls_by_key})
    return {
        "calls": calls,
        "logical_payload_bytes": logical_payload_bytes,
        "ring_equivalent_bytes": ring_equivalent_bytes,
        "ring_equivalent_rounds": ring_equivalent_rounds,
        "unique_payload_sizes": len(payloads),
        "min_payload_bytes": min(payloads),
        "max_payload_bytes": max(payloads),
        "ops_json": json.dumps(
            sorted({op for op, _ in calls_by_key}),
            separators=(",", ":"),
        ),
        "histogram_json": json.dumps(
            histogram,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def load_model_dataset(model, directory):
    directory = Path(directory)
    paths = sorted(directory.glob("tp*/r*/**/result.jsonl"))
    if not paths:
        raise ValueError(f"{model}: no result.jsonl below {directory}")

    grouped = defaultdict(list)
    for path in paths:
        phase, tp, repeat = parse_path(path)
        for record in read_jsonl(path):
            if not record["same_shape_workload_warmup"]:
                raise ValueError(f"{record['run_name']}: workload warmup missing")
            if record["generated_output_tokens"] != record["output_len"]:
                raise ValueError(f"{record['run_name']}: output length mismatch")
            if (
                len(record["generated_output_tokens_per_request"])
                != record["batch_size"]
                or set(record["generated_output_tokens_per_request"])
                != {record["output_len"]}
            ):
                raise ValueError(
                    f"{record['run_name']}: per-request output length mismatch"
                )
            histograms = representative_histograms(record, phase, tp)
            pattern = aggregate_histograms(histograms, tp)
            key = (
                phase,
                tp,
                int(record["batch_size"]),
                int(record["input_len"]),
                int(record["output_len"]),
            )
            grouped[key].append((repeat, pattern))

    rows = []
    for key, repeats in sorted(grouped.items()):
        phase, tp, batch_size, input_len, output_len = key
        repeat_ids = sorted(repeat for repeat, _ in repeats)
        if len(repeat_ids) < 3:
            raise ValueError(
                f"{model} {key}: expected at least 3 repeats, got {repeat_ids}"
            )
        patterns = [pattern for _, pattern in repeats]
        if any(pattern != patterns[0] for pattern in patterns[1:]):
            raise ValueError(f"{model} {key}: PatternDemand changed by repeat")
        pattern = patterns[0]
        rows.append(
            {
                "model": model,
                "phase": phase,
                "tp": tp,
                "batch_size": batch_size,
                "input_len": input_len,
                "output_len": output_len,
                "repeat_count": len(repeats),
                **pattern,
            }
        )
    return rows


def relative_gap(left, right):
    return abs(left - right) / max(left, right)


def pair_shape_distance(left, right):
    left_hist = json.loads(left["histogram_json"])
    right_hist = json.loads(right["histogram_json"])
    keys = set(left_hist) | set(right_hist)
    left_calls = sum(left_hist.values())
    right_calls = sum(right_hist.values())
    return 0.5 * sum(
        abs(
            left_hist.get(key, 0) / left_calls
            - right_hist.get(key, 0) / right_calls
        )
        for key in keys
    )


def workload_id(row):
    return (
        f"{row['phase']}-tp{row['tp']}-b{row['batch_size']}"
        f"-l{row['input_len']}-m{row['output_len']}"
    )


def find_near_equal_payload_pairs(rows, threshold=0.035):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["phase"], row["tp"])].append(row)

    pairs = []
    for (model, phase, tp), candidates in sorted(groups.items()):
        for index, left in enumerate(candidates):
            for right in candidates[index + 1 :]:
                if left["input_len"] != right["input_len"]:
                    continue
                if left["histogram_json"] == right["histogram_json"]:
                    continue
                gap = relative_gap(
                    left["logical_payload_bytes"],
                    right["logical_payload_bytes"],
                )
                if gap > threshold:
                    continue
                calls_ratio = max(left["calls"], right["calls"]) / min(
                    left["calls"], right["calls"]
                )
                pairs.append(
                    {
                        "model": model,
                        "phase": phase,
                        "tp": tp,
                        "left_workload": workload_id(left),
                        "right_workload": workload_id(right),
                        "left_calls": left["calls"],
                        "right_calls": right["calls"],
                        "left_total_payload_bytes": left[
                            "logical_payload_bytes"
                        ],
                        "right_total_payload_bytes": right[
                            "logical_payload_bytes"
                        ],
                        "relative_payload_gap": gap,
                        "calls_ratio": calls_ratio,
                        "histogram_shape_distance": pair_shape_distance(
                            left, right
                        ),
                        "left_histogram_json": left["histogram_json"],
                        "right_histogram_json": right["histogram_json"],
                    }
                )
    return sorted(
        pairs,
        key=lambda row: (
            -row["histogram_shape_distance"],
            -row["calls_ratio"],
            row["relative_payload_gap"],
        ),
    )


def compare_models(rows):
    by_workload = defaultdict(dict)
    for row in rows:
        key = (
            row["phase"],
            row["tp"],
            row["batch_size"],
            row["input_len"],
            row["output_len"],
        )
        by_workload[key][row["model"]] = row

    models = sorted({row["model"] for row in rows})
    if len(models) != 2:
        return []
    left_model, right_model = models
    comparisons = []
    for key, model_rows in sorted(by_workload.items()):
        if set(model_rows) != set(models):
            continue
        left = model_rows[left_model]
        right = model_rows[right_model]
        comparisons.append(
            {
                "phase": left["phase"],
                "tp": left["tp"],
                "batch_size": left["batch_size"],
                "input_len": left["input_len"],
                "output_len": left["output_len"],
                "left_model": left_model,
                "right_model": right_model,
                "left_calls": left["calls"],
                "right_calls": right["calls"],
                "calls_ratio_right_over_left": right["calls"] / left["calls"],
                "left_total_payload_bytes": left["logical_payload_bytes"],
                "right_total_payload_bytes": right["logical_payload_bytes"],
                "payload_ratio_right_over_left": (
                    right["logical_payload_bytes"]
                    / left["logical_payload_bytes"]
                ),
                "left_histogram_json": left["histogram_json"],
                "right_histogram_json": right["histogram_json"],
            }
        )
    return comparisons


def format_bytes(value):
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(value)
    unit = units[0]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            break
        value /= 1024
    return f"{value:.3g} {unit}"


def render_markdown(rows, comparisons, pairs):
    models = sorted({row["model"] for row in rows})
    lines = [
        "# Phase 8 跨模型 PatternDemand 分析",
        "",
        "## 数据完整性",
        "",
        "| 模型 | workloads | TP | phases | repeats/workload |",
        "|---|---:|---|---|---:|",
    ]
    for model in models:
        selected = [row for row in rows if row["model"] == model]
        lines.append(
            f"| {model} | {len(selected)} | "
            f"{','.join(map(str, sorted({row['tp'] for row in selected})))} | "
            f"{','.join(sorted({row['phase'] for row in selected}))} | "
            f"{min(row['repeat_count'] for row in selected)} |"
        )

    lines.extend(
        [
            "",
            "## 同 workload 跨模型结构指纹",
            "",
            "| phase | TP | B,L,M | left model | left calls/payload | "
            "right model | right calls/payload |",
            "|---|---:|---|---|---|---|---|",
        ]
    )
    anchors = [
        row
        for row in comparisons
        if (
            (
                row["phase"] == "prefill"
                and row["batch_size"] == 1
                and row["input_len"] == 128
            )
            or (
                row["phase"] == "decode"
                and row["batch_size"] == 1
                and row["input_len"] == 128
                and row["output_len"] == 32
            )
        )
        and row["tp"] in {2, 4, 8}
    ]
    for row in anchors:
        lines.append(
            f"| {row['phase']} | {row['tp']} | "
            f"{row['batch_size']},{row['input_len']},{row['output_len']} | "
            f"{row['left_model']} | {row['left_calls']} / "
            f"{format_bytes(row['left_total_payload_bytes'])} | "
            f"{row['right_model']} | {row['right_calls']} / "
            f"{format_bytes(row['right_total_payload_bytes'])} |"
        )

    lines.extend(
        [
            "",
            "## 近等总 payload、不同消息形态",
            "",
            "| model | phase | TP | workload A | workload B | payload gap | "
            "calls ratio | shape distance |",
            "|---|---|---:|---|---|---:|---:|---:|",
        ]
    )
    display_pairs = []
    pair_groups = defaultdict(list)
    for row in pairs:
        pair_groups[(row["model"], row["phase"], row["tp"])].append(row)
    for group in sorted(pair_groups):
        display_pairs.extend(pair_groups[group][:2])
    for row in display_pairs:
        lines.append(
            f"| {row['model']} | {row['phase']} | {row['tp']} | "
            f"{row['left_workload']} | {row['right_workload']} | "
            f"{100 * row['relative_payload_gap']:.2f}% | "
            f"{row['calls_ratio']:.2f}× | "
            f"{row['histogram_shape_distance']:.3f} |"
        )

    calls_ratios = [
        row["calls_ratio_right_over_left"] for row in comparisons
    ]
    payload_ratios = [
        row["payload_ratio_right_over_left"] for row in comparisons
    ]
    lines.extend(
        [
            "",
            "## 结论",
            "",
            (
                f"- 共获得 {len(rows)} 个 model × workload 聚合点；"
                f"匹配的跨模型 workload 为 {len(comparisons)} 个。"
            ),
            (
                "- 同 workload 的跨模型 calls 比值中位数为 "
                f"{statistics.median(calls_ratios):.3f}，总逻辑 payload "
                f"比值中位数为 {statistics.median(payload_ratios):.3f}。"
            ),
            (
                f"- 自动找到 {len(pairs)} 组总 payload 相差不超过 3.5% "
                "但消息直方图不同的样本对。"
            ),
            (
                "- 第一阶段预测器应输入模型或模型结构特征，并输出连续 "
                "`op × group_size × payload` 直方图；total bytes 只能作为"
                "消融基线。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    rows = []
    for model, directory in args.dataset:
        rows.extend(load_model_dataset(model, directory))
    comparisons = compare_models(rows)
    pairs = find_near_equal_payload_pairs(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "pattern_summary.csv", rows)
    write_csv(
        args.output_dir / "cross_model_workload_comparison.csv",
        comparisons,
        fieldnames=[
            "phase",
            "tp",
            "batch_size",
            "input_len",
            "output_len",
            "left_model",
            "right_model",
            "left_calls",
            "right_calls",
            "calls_ratio_right_over_left",
            "left_total_payload_bytes",
            "right_total_payload_bytes",
            "payload_ratio_right_over_left",
            "left_histogram_json",
            "right_histogram_json",
        ],
    )
    write_csv(
        args.output_dir / "near_equal_payload_pairs.csv",
        pairs,
        fieldnames=[
            "model",
            "phase",
            "tp",
            "left_workload",
            "right_workload",
            "left_calls",
            "right_calls",
            "left_total_payload_bytes",
            "right_total_payload_bytes",
            "relative_payload_gap",
            "calls_ratio",
            "histogram_shape_distance",
            "left_histogram_json",
            "right_histogram_json",
        ],
    )
    (args.output_dir / "README.md").write_text(
        render_markdown(rows, comparisons, pairs)
    )
    print(
        json.dumps(
            {
                "models": sorted({row["model"] for row in rows}),
                "pattern_rows": len(rows),
                "matched_cross_model_workloads": len(comparisons),
                "near_equal_payload_pairs": len(pairs),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
