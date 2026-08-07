#!/usr/bin/env python3
"""Create a compact, auditable dataset from the completed ProfileDemand GPU matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


MODELS = ("qwen3-8b", "deepseek-v2-lite", "qwen3-30b-a3b")
TPS = (2, 4, 8)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix-root",
        type=Path,
        default=root / "experiment-results/phase16_profiledemand_gpu",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase16_profiledemand_dataset",
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


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"empty rows: {path}")
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def label_signature(row, raw=False):
    fields = [
        "total_calls_per_1000",
        "total_logical_bytes_per_1000",
        "calls_by_12bin_json",
        "logical_bytes_by_12bin_json",
        "canonical_exact_histogram_per_1000_json",
    ]
    if raw:
        fields.append("raw_op_exact_histogram_per_1000_json")
    return tuple(row[field] for field in fields)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.matrix_root / "MATRIX_DONE").read_text().strip() != "PASS":
        raise RuntimeError("GPU matrix does not have MATRIX_DONE=PASS")

    labels = []
    inventory = []
    for model in MODELS:
        for tp in TPS:
            run_dir = args.matrix_root / "full" / model / f"tp{tp}" / "r0"
            summary = json.loads((run_dir / "summary.json").read_text())
            audit = json.loads((run_dir / "audit_summary.json").read_text())
            rows = read_csv(run_dir / "phase_labels.csv")
            if summary["model"] != model or int(summary["tp"]) != tp:
                raise ValueError(f"summary identity mismatch: {run_dir}")
            if audit["status"] != "PASS" or not all(summary["checks"].values()):
                raise RuntimeError(f"failed run audit: {run_dir}")
            if len(rows) != 144:
                raise ValueError(f"expected 144 labels at {run_dir}, got {len(rows)}")
            labels.extend(rows)
            result_path = run_dir / "result.jsonl"
            inventory.append(
                {
                    "model": model,
                    "tp": tp,
                    "workloads": summary["workloads"],
                    "phase_labels": len(rows),
                    "all_checks_pass": all(summary["checks"].values()),
                    "result_sha256": summary["result_sha256"],
                    "result_size_bytes_remote_only": result_path.stat().st_size,
                    "run_log_sha256": sha256(run_dir / "run.log"),
                    "telemetry_sha256": sha256(run_dir / "telemetry.csv"),
                    "phase_labels_sha256": sha256(run_dir / "phase_labels.csv"),
                }
            )

    labels.sort(
        key=lambda row: (
            row["model"],
            int(row["tp"]),
            row["profile_id"],
            row["strategy"],
            row["phase"],
        )
    )
    write_csv(args.output_dir / "phase_labels.csv", labels)
    write_csv(args.output_dir / "run_inventory.csv", inventory)

    smoke_dir = args.matrix_root / "smoke" / "qwen3-8b" / "tp2" / "r0"
    smoke_summary = json.loads((smoke_dir / "summary.json").read_text())
    smoke_labels = read_csv(smoke_dir / "phase_labels.csv")
    if not all(smoke_summary["checks"].values()) or len(smoke_labels) != 54:
        raise RuntimeError("smoke compact labels failed audit")
    write_csv(args.output_dir / "smoke_phase_labels.csv", smoke_labels)

    tp_groups = defaultdict(list)
    for row in labels:
        key = (row["model"], row["profile_id"], row["strategy"], row["phase"])
        tp_groups[key].append(label_signature(row))
    tp_invariant = all(len(values) == 3 and len(set(values)) == 1 for values in tp_groups.values())

    smoke_groups = defaultdict(list)
    for row in smoke_labels:
        key = (row["profile_id"], row["strategy"], row["phase"])
        smoke_groups[key].append(label_signature(row, raw=True))
    repeat_stable = all(
        len(values) == 3 and len(set(values)) == 1 for values in smoke_groups.values()
    )

    summary = {
        "schema_version": "profiledemand-gpu-matrix-compact-v1",
        "models": list(MODELS),
        "tp_sizes": list(TPS),
        "profiles": len({row["profile_id"] for row in labels}),
        "strategies": sorted({row["strategy"] for row in labels}),
        "phases": sorted({row["phase"] for row in labels}),
        "full_phase_labels": len(labels),
        "smoke_phase_labels": len(smoke_labels),
        "full_gpu_workloads": sum(int(row["workloads"]) for row in inventory),
        "all_rank_and_h0_checks_pass": all(row["all_checks_pass"] for row in inventory),
        "canonical_labels_tp_invariant": tp_invariant,
        "smoke_three_repeats_identical": repeat_stable,
        "raw_result_policy": (
            "Nine histogram-only result.jsonl files (no raw events) and telemetry remain "
            "next to remote runs; compact labels, hashes, summaries, and inventory are archived here."
        ),
        "matrix_log_sha256": sha256(args.matrix_root / "matrix.log"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    readme = f"""# Phase 16F：ProfileDemand GPU 正式标签集

GPU 矩阵已完成 3 个模型 × TP2/4/8 × 24 个服务画像 × 3 种策略。底层共执行
{summary['full_gpu_workloads']} 个 histogram-only microbatch workloads，聚合为
{len(labels)} 条 `model×TP×profile×strategy×phase` 正式标签。

每条标签包含每 1000 请求的 total calls、logical bytes、12 桶 calls、12 桶 logical
bytes、canonical 精确直方图和 raw-op 精确直方图。9/9 组 all-rank、固定实际输出、
group size、H0 canonical 映射及 histogram-only 契约全部通过；Qwen3-8B TP2 smoke 的
三次重复完全一致；canonical labels 在同一 model/profile/strategy/phase 下跨 TP 完全
不变。

约 89 MB 的 `result.jsonl` 只包含各 rank 紧凑直方图而非 raw events，继续保存在远端
运行目录。当前目录归档可训练的紧凑标签、文件哈希和运行清单，避免把中间结果重复
写入 Git。
"""
    (args.output_dir / "README.md").write_text(readme)
    checks = {
        "matrix_done_pass": True,
        "nine_runs": len(inventory) == 9,
        "full_phase_labels_1296": len(labels) == 1296,
        "smoke_phase_labels_54": len(smoke_labels) == 54,
        "profiles_24": summary["profiles"] == 24,
        "all_run_checks_pass": summary["all_rank_and_h0_checks_pass"],
        "canonical_tp_invariant": tp_invariant,
        "smoke_repeat_stable": repeat_stable,
    }
    audit = {
        "schema_version": "profiledemand-gpu-matrix-compact-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    (args.output_dir / "audit_summary.json").write_text(json.dumps(audit, indent=2) + "\n")
    if audit["status"] != "PASS":
        raise RuntimeError(audit)
    (args.output_dir / "DONE").write_text("PASS\n")
    (args.output_dir / "run.log").write_text(json.dumps({"summary": summary, "checks": checks}, indent=2) + "\n")
    files = sorted(
        path for path in args.output_dir.iterdir() if path.is_file() and path.name != "manifest.sha256"
    )
    (args.output_dir / "manifest.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files)
    )
    print(json.dumps({"summary": summary, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
