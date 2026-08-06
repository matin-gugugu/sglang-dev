#!/usr/bin/env python3
"""Audit and summarize Qwen3-8B trace-window PatternDemand collection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


CALLS_PER_FORWARD = 73
BYTES_PER_TOKEN = 8192  # hidden_size=4096, bf16
TPS = (2, 4, 8)
PHASES = ("prefill", "decode")


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=root / "experiment-results/phase15_qwen_trace_pattern",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=root
        / "experiment-results/phase15_trace_data/smoke_replay_plan.jsonl",
    )
    return parser.parse_args()


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_csv(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"empty rows for {path}")
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_histogram(plan, phase):
    histogram = defaultdict(int)
    if phase == "prefill":
        payload = sum(plan["input_lens_per_request"]) * BYTES_PER_TOKEN
        histogram[f"all_reduce:{payload}"] = CALLS_PER_FORWARD
    else:
        outputs = plan["output_lens_per_request"]
        for step in range(1, max(outputs)):
            active = sum(length > step for length in outputs)
            if active:
                histogram[f"all_reduce:{active * BYTES_PER_TOKEN}"] += (
                    CALLS_PER_FORWARD
                )
    return dict(sorted(histogram.items()))


def main():
    args = parse_args()
    root = args.result_root.resolve()
    plans = load_jsonl(args.plan)
    plan_by_id = {row["workload_id"]: row for row in plans}
    if len(plan_by_id) != 20:
        raise AssertionError(f"expected 20 unique plans, got {len(plan_by_id)}")

    combined = []
    validations = []
    for tp in TPS:
        directory = root / f"tp{tp}" / "r0"
        if not (directory / "DONE").is_file():
            raise FileNotFoundError(directory / "DONE")
        validation = json.loads((directory / "validation_summary.json").read_text())
        if validation.get("status") != "PASS" or validation.get("workloads") != 20:
            raise AssertionError(f"TP={tp} validation failed: {validation}")
        validations.append(validation)
        rows = read_csv(directory / "pattern_labels.csv")
        if len(rows) != 40:
            raise AssertionError(f"TP={tp}: expected 40 labels, got {len(rows)}")
        combined.extend(rows)

    analytic_checks = []
    for row in combined:
        workload_id = row["workload_id"]
        plan = plan_by_id[workload_id]
        phase = row["phase"]
        tp = int(row["tp"])
        observed = json.loads(row["calls_by_op_payload_json"])
        expected = expected_histogram(plan, phase)
        expected_calls = sum(expected.values())
        expected_bytes = sum(
            int(key.rsplit(":", 1)[1]) * count for key, count in expected.items()
        )
        alpha = 2 * (tp - 1) / tp
        beta = 2 * (tp - 1)
        checks = {
            "histogram": observed == expected,
            "calls": int(row["calls"]) == expected_calls,
            "logical_payload_bytes": int(row["logical_payload_bytes"])
            == expected_bytes,
            "equivalent_bytes": abs(float(row["equivalent_bytes"]) - alpha * expected_bytes)
            < 1e-6,
            "equivalent_rounds": abs(float(row["equivalent_rounds"]) - beta * expected_calls)
            < 1e-6,
        }
        analytic_checks.append(
            {
                "workload_id": workload_id,
                "tp": tp,
                "phase": phase,
                **{key: str(value).lower() for key, value in checks.items()},
            }
        )
        if not all(checks.values()):
            raise AssertionError(f"analytic mismatch: {workload_id}/TP{tp}/{phase}")

    invariant = True
    invariance_rows = []
    for workload_id in plan_by_id:
        for phase in PHASES:
            selected = sorted(
                (
                    row
                    for row in combined
                    if row["workload_id"] == workload_id and row["phase"] == phase
                ),
                key=lambda row: int(row["tp"]),
            )
            logical = {
                (
                    int(row["calls"]),
                    int(row["logical_payload_bytes"]),
                    row["calls_by_op_payload_json"],
                )
                for row in selected
            }
            same = len(logical) == 1
            invariant &= same
            invariance_rows.append(
                {
                    "workload_id": workload_id,
                    "phase": phase,
                    "logical_pattern_tp_invariant": str(same).lower(),
                    "calls": selected[0]["calls"],
                    "logical_payload_bytes": selected[0]["logical_payload_bytes"],
                    "tp2_equivalent_bytes": selected[0]["equivalent_bytes"],
                    "tp4_equivalent_bytes": selected[1]["equivalent_bytes"],
                    "tp8_equivalent_bytes": selected[2]["equivalent_bytes"],
                    "tp2_equivalent_rounds": selected[0]["equivalent_rounds"],
                    "tp4_equivalent_rounds": selected[1]["equivalent_rounds"],
                    "tp8_equivalent_rounds": selected[2]["equivalent_rounds"],
                    "calls_by_op_payload_json": selected[0][
                        "calls_by_op_payload_json"
                    ],
                }
            )
    if not invariant:
        raise AssertionError("logical PatternDemand changed across TP")

    # Find close-total-byte pairs whose exact histogram shape differs.
    pair_rows = []
    canonical = [row for row in combined if int(row["tp"]) == 2]
    for index, left in enumerate(canonical):
        for right in canonical[index + 1 :]:
            if left["phase"] != right["phase"]:
                continue
            left_bytes = int(left["logical_payload_bytes"])
            right_bytes = int(right["logical_payload_bytes"])
            ratio = max(left_bytes, right_bytes) / max(min(left_bytes, right_bytes), 1)
            if left["calls_by_op_payload_json"] == right["calls_by_op_payload_json"]:
                continue
            pair_rows.append(
                {
                    "phase": left["phase"],
                    "left_workload_id": left["workload_id"],
                    "right_workload_id": right["workload_id"],
                    "payload_ratio": ratio,
                    "left_calls": left["calls"],
                    "right_calls": right["calls"],
                    "left_logical_payload_bytes": left_bytes,
                    "right_logical_payload_bytes": right_bytes,
                    "left_histogram": left["calls_by_op_payload_json"],
                    "right_histogram": right["calls_by_op_payload_json"],
                }
            )
    pair_rows.sort(key=lambda row: row["payload_ratio"])

    write_csv(root / "pattern_labels_all_tp.csv", combined)
    write_csv(root / "analytic_checks.csv", analytic_checks)
    write_csv(root / "tp_invariance.csv", invariance_rows)
    write_csv(root / "close_payload_pairs.csv", pair_rows[:20])
    summary = {
        "schema_version": "phase15-qwen-trace-pattern-v1",
        "status": "PASS",
        "model": "Qwen3-8B",
        "topology": "single-node-B200-NVLink",
        "workloads": len(plans),
        "tps": list(TPS),
        "phase_labels": len(combined),
        "collection_mode": "histogram-only",
        "raw_events_saved": False,
        "trace_replay_mode": "draining_batch_a_i_zero",
        "all_rank_histograms_identical": all(
            row["all_rank_histograms_identical"] for row in validations
        ),
        "fixed_actual_output_lengths": all(
            row["fixed_actual_output_lengths"] for row in validations
        ),
        "analytic_gpu_histogram_checks": len(analytic_checks),
        "analytic_gpu_histogram_checks_passed": len(analytic_checks),
        "logical_pattern_tp_invariant": invariant,
        "logical_contract": {
            "calls": "group-level collective count",
            "payload_bytes": "representative-rank logical input tensor bytes",
            "qwen3_8b_calls_per_forward": CALLS_PER_FORWARD,
            "qwen3_8b_bf16_bytes_per_token": BYTES_PER_TOKEN,
        },
        "scope_boundary": (
            "Arrival offsets are audit-only; this phase validates heterogeneous "
            "simultaneous draining batches, not online interleaved arrivals."
        ),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (root / "README.md").write_text(
        f"""# Phase 15：Qwen3-8B 真实 trace 窗口 PatternDemand

本阶段在单节点 B200 上，对 20 个 BurstGPT/Mooncake 窗口分别以 TP=2/4/8
执行 histogram-only 通信采集，共得到 {len(combined)} 条“窗口 × TP × 阶段”标签。

审计结果：

- 20/20 个窗口在三个 TP 下均成功；所有 rank 的通信直方图一致；
- 实际生成长度与逐请求 `output_lens_per_request` 完全一致，没有 EOS 提前退出；
- 120/120 条 GPU 直方图均与 Qwen3-8B 的解析事件公式一致；
- logical calls、logical bytes 和 `(raw_op,payload)` 直方图跨 TP 完全不变；
- TP 只通过 ring 折算改变 equivalent bytes 和 equivalent rounds；
- 仅保存 histogram，不保存 raw events。

这证明当前第一阶段表征能够从异构输入长度和 draining Decode 的 active batch 变化中，
稳定抽取消息尺度结构。边界是：所有请求仍在同一时刻进入，arrival offset 仅用于审计，
尚不能声称已经验证真实在线交错到达和 continuous batching。

主要文件：

- `pattern_labels_all_tp.csv`：正式 PatternDemand 标签；
- `analytic_checks.csv`：GPU 直方图与解析公式逐条比对；
- `tp_invariance.csv`：logical pattern 跨 TP 不变及等效量随 TP 变化；
- `close_payload_pairs.csv`：近等总 payload、不同消息形态候选对照；
- `tp*/r0/result.jsonl`：20 个 compact histogram-only 原始结果；
- `summary.json`：机器可读审计结论。
"""
    )
    manifest = root / "manifest.sha256"
    files = sorted(path for path in root.rglob("*") if path.is_file() and path != manifest)
    manifest.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(root)}\n" for path in files)
    )
    print(json.dumps(summary, indent=2))
    print(f"manifest_files={len(files)}")


if __name__ == "__main__":
    main()
