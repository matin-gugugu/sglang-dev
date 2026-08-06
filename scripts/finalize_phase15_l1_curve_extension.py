#!/usr/bin/env python3
"""Audit the 160--512 MiB Phase 15 L1 AllReduce curve extension."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


TPS = (2, 4, 8)
REPEATS = set(range(5))
PAYLOADS = {
    167772160,
    201326592,
    268435456,
    335544320,
    402653184,
    469762048,
    536870912,
}


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=root / "experiment-results/phase15_l1_curve_extension",
    )
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    root = args.result_root.resolve()
    files = sorted(root.glob("curve/tp*/all_reduce/r*/curve.jsonl"))
    if len(files) != 15 or not (root / "DONE").is_file():
        raise AssertionError(f"incomplete curve directory: files={len(files)}")
    grouped = defaultdict(list)
    repeat_medians = defaultdict(dict)
    backend = defaultdict(set)
    records = 0
    samples = 0
    for path in files:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            tp = int(row["group_size"])
            payload = int(row["payload_bytes"])
            repeat = int(row["repeat_id"])
            values = [float(value) for value in row["post_rendezvous_samples_us"]]
            if (
                tp not in TPS
                or payload not in PAYLOADS
                or repeat not in REPEATS
                or row["op"] != "all_reduce"
                or len(values) != 100
            ):
                raise AssertionError(f"invalid record in {path}: {row}")
            key = (tp, payload)
            grouped[key].extend(values)
            repeat_medians[key][repeat] = float(np.median(values))
            backend[key].add(row["backend_proxy_pre_run"])
            records += 1
            samples += len(values)
    expected = {(tp, payload) for tp in TPS for payload in PAYLOADS}
    if set(grouped) != expected:
        raise AssertionError("support set mismatch")
    rows = []
    max_repeat_cv = 0.0
    for tp, payload in sorted(grouped):
        values = np.asarray(grouped[(tp, payload)], dtype=np.float64)
        medians = np.asarray(
            [repeat_medians[(tp, payload)][repeat] for repeat in sorted(REPEATS)]
        )
        repeat_cv = float(np.std(medians) / np.mean(medians))
        max_repeat_cv = max(max_repeat_cv, repeat_cv)
        rows.append(
            {
                "op": "all_reduce",
                "tp": tp,
                "payload_bytes": payload,
                "payload_mib": payload / 2**20,
                "samples": len(values),
                "median_post_rendezvous_us": float(np.median(values)),
                "mean_post_rendezvous_us": float(np.mean(values)),
                "p95_post_rendezvous_us": float(np.percentile(values, 95)),
                "repeat_median_cv": repeat_cv,
                "backend_proxy": ";".join(sorted(backend[(tp, payload)])),
            }
        )
    with (root / "curve_summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": "phase15-l1-curve-extension-v1",
        "status": "PASS",
        "topology": "single-node-B200-NVLink",
        "op": "all_reduce",
        "tps": list(TPS),
        "payload_mib": [value / 2**20 for value in sorted(PAYLOADS)],
        "curve_files": len(files),
        "curve_records": records,
        "support_points": len(grouped),
        "samples": samples,
        "repeats_per_support": 5,
        "iterations_per_repeat": 100,
        "latency_contract": "all-rank post-rendezvous kernel interval",
        "max_repeat_median_cv": max_repeat_cv,
        "all_support_repeat_cv_below_10pct": max_repeat_cv < 0.10,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (root / "README.md").write_text(
        f"""# Phase 15：L1 长消息连续代价曲线补点

为覆盖公开 trace 中 320--512 MiB 的长 Prefill 消息，本阶段在单节点 B200 NVLink
拓扑上补测 AllReduce 的 160/192/256/320/384/448/512 MiB 支撑点，覆盖 TP2/4/8。

- 21 个 `TP × payload` 支撑点；
- 每个支撑点 5 次独立重复，每次 100 个样本；
- 共 {samples} 个 all-rank post-rendezvous 样本；
- 最大 repeat-median CV：{max_repeat_cv:.4%}；
- 时间口径与更正后的 Phase14F 完全一致。

`curve_summary.csv` 提供连续插值使用的中位数代价；`curve/` 保留完整样本和 backend
审计字段。
"""
    )
    manifest = root / "manifest.sha256"
    manifest_files = sorted(
        path for path in root.rglob("*") if path.is_file() and path != manifest
    )
    manifest.write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(root)}\n" for path in manifest_files
        )
    )
    print(json.dumps(summary, indent=2))
    print(f"manifest_files={len(manifest_files)}")


if __name__ == "__main__":
    main()
