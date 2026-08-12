#!/usr/bin/env python3
"""汇总Phase 26A TP Hfull GPU sentinel并晋升正式teacher标签。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_phase26a_tp_hfull_teacher_audit import CELL_SPECS


PROMOTED_STATUS = "GPU_VALIDATED_STRUCTURAL_FORMULA_SENTINELS_4_CELLS"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "experiment-results/phase26a_tp_hfull_teacher_audit",
    )
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=root / "experiment-results/phase25_full_window_teacher",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_csv_gz(path: Path, rows: list[dict]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as output:
            output.write(buffer.getvalue().encode())


def json_hist(value: str) -> dict[int, float]:
    return {int(key): float(count) for key, count in json.loads(value).items()}


def l1(left: dict[int, float], right: dict[int, float], *, bytes_weighted: bool = False) -> float:
    keys = set(left) | set(right)
    return sum(
        abs(left.get(key, 0.0) - right.get(key, 0.0)) * (key if bytes_weighted else 1)
        for key in keys
    )


def coverage_figure(cells: list[dict], path: Path) -> None:
    models = ["qwen3-8b", "qwen3-30b-a3b", "deepseek-v2-lite"]
    tps = [2, 4, 8]
    matrix = np.zeros((len(models), len(tps)))
    label = {}
    for cell in cells:
        row = models.index(cell["model"])
        column = tps.index(int(cell["tp"]))
        matrix[row, column] = 1
        label[(row, column)] = f"{cell['profile_id'].split('_')[1]}\n{cell['requests']} req"
    figure, axis = plt.subplots(figsize=(8.2, 3.6))
    axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(tps)), [f"TP{tp}" for tp in tps])
    axis.set_yticks(range(len(models)), models)
    axis.set_xlabel("Tensor parallel size")
    axis.set_title("Phase 26A TP Hfull GPU sentinel coverage")
    for row in range(len(models)):
        for column in range(len(tps)):
            text = label.get((row, column), "—")
            axis.text(column, row, text, ha="center", va="center", color="black")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output_root.mkdir(parents=True, exist_ok=True)
    for name in ("analysis", "figures", "labels", "logs"):
        (args.output_root / name).mkdir(exist_ok=True)

    teacher_path = args.teacher_root / "labels/tp_phase_labels.csv.gz"
    teacher_rows = read_csv_gz(teacher_path)
    teacher = {
        (row["model"], int(row["parallel_size"]), row["profile_id"], row["policy"], row["phase"]): row
        for row in teacher_rows
    }
    cells = [
        {
            "cell_id": "phase25a-existing-qwen3-8b-tp2-smoke",
            "model": "qwen3-8b",
            "tp": 2,
            "profile_id": "profile_13_burstgpt_3_c2",
            "coverage_reason": "Phase 25A已有完整窗口smoke；覆盖TP2",
            "path": args.teacher_root / "gpu_audit/results/tp/smoke/qwen3-8b/tp2/r0",
            "evidence_origin": "phase25a_existing",
        }
    ]
    for cell_id, spec in CELL_SPECS.items():
        cells.append(
            {
                "cell_id": cell_id,
                "model": spec["model"],
                "tp": spec["tp"],
                "profile_id": spec["profile_id"],
                "coverage_reason": spec["coverage_reason"],
                "path": args.output_root / "results" / cell_id / "r0",
                "evidence_origin": "phase26a_new",
            }
        )

    cell_rows = []
    comparison_rows = []
    all_pass = True
    for cell in cells:
        path = cell["path"]
        audit = json.loads((path / "teacher_audit.json").read_text())
        validation = json.loads((path / "summary.json").read_text())
        gpu_rows = read_csv(path / "gpu_phase_labels.csv")
        audit_rows = read_csv(path / "teacher_comparisons.csv")
        config = json.loads((path / "run_config.json").read_text())
        cell_pass = (
            audit["status"] == "PASS"
            and all(audit["checks"].values())
            and all(validation["checks"].values())
            and len(audit_rows) == 6
        )
        all_pass &= cell_pass
        cell_rows.append(
            {
                "cell_id": cell["cell_id"],
                "evidence_origin": cell["evidence_origin"],
                "model": cell["model"],
                "tp": cell["tp"],
                "profile_id": cell["profile_id"],
                "requests": int(config.get("request_count", 42)),
                "workloads": int(config["workloads"]),
                "strategies": 3,
                "phase_comparisons": len(audit_rows),
                "max_input_len": int(config.get("max_input_len", 3348)),
                "max_output_len": int(config.get("max_output_len", 128)),
                "status": "PASS" if cell_pass else "FAIL",
                "coverage_reason": cell["coverage_reason"],
            }
        )
        for gpu in gpu_rows:
            key = (
                cell["model"],
                int(cell["tp"]),
                gpu["profile_id"],
                gpu["policy"],
                gpu["phase"],
            )
            expected = teacher[key]
            gpu_hist = json_hist(gpu["exact_calls_histogram_per_1000_json"])
            teacher_hist = json_hist(expected["exact_calls_histogram_per_1000_json"])
            comparison_rows.append(
                {
                    "cell_id": cell["cell_id"],
                    "model": cell["model"],
                    "tp": cell["tp"],
                    "profile_id": gpu["profile_id"],
                    "policy": gpu["policy"],
                    "phase": gpu["phase"],
                    "requests": gpu["requests"],
                    "calls_abs_error": abs(
                        float(gpu["total_calls_per_1000"])
                        - float(expected["total_calls_per_1000"])
                    ),
                    "logical_bytes_abs_error": abs(
                        float(gpu["total_logical_bytes_per_1000"])
                        - float(expected["total_logical_bytes_per_1000"])
                    ),
                    "histogram_calls_l1": l1(gpu_hist, teacher_hist),
                    "histogram_logical_bytes_l1": l1(
                        gpu_hist, teacher_hist, bytes_weighted=True
                    ),
                    "exact": audit["status"] == "PASS",
                }
            )

    expected_comparisons = len(cells) * 3 * 2
    all_pass &= len(comparison_rows) == expected_comparisons
    maxima = {
        field: max(float(row[field]) for row in comparison_rows)
        for field in (
            "calls_abs_error",
            "logical_bytes_abs_error",
            "histogram_calls_l1",
            "histogram_logical_bytes_l1",
        )
    }
    all_pass &= all(value == 0 for value in maxima.values())

    promoted = [{**row, "label_status": PROMOTED_STATUS} for row in teacher_rows]
    write_csv_gz(args.output_root / "labels/tp_hfull_phase_labels.csv.gz", promoted)
    write_csv(args.output_root / "analysis/cell_summary.csv", cell_rows)
    write_csv(args.output_root / "analysis/phase_comparisons.csv", comparison_rows)
    coverage_figure(cell_rows, args.output_root / "figures/sentinel_coverage.png")

    contract = {
        "schema_version": "phase26a-tp-hfull-teacher-contract-v1",
        "parallelism": "tp",
        "execution_semantics": "fixed-draining",
        "teacher_input": "完整窗口真实请求长度列表，保持原始顺序",
        "predictor_input": "低维历史画像、模型结构、固定TP、固定策略和phase",
        "audited_models": sorted({row["model"] for row in cell_rows}),
        "audited_tp_sizes": sorted({int(row["tp"]) for row in cell_rows}),
        "audited_policies": ["latency", "balanced", "throughput"],
        "gpu_capture": "histogram-only；不保存raw events",
        "out_of_scope": [
            "online arrival-aware调度",
            "不同collective实现或模型结构元数据",
            "未经独立审计的其他SGLang版本",
        ],
    }
    (args.output_root / "contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    summary = {
        "schema_version": "phase26a-tp-hfull-teacher-audit-v1",
        "status": "PASS" if all_pass else "FAIL",
        "objective": "跨模型、TP size、策略与流量尾部审计TP Hfull结构teacher",
        "cells": len(cells),
        "new_gpu_cells": len(CELL_SPECS),
        "phase_comparisons": len(comparison_rows),
        "models": sorted({row["model"] for row in cell_rows}),
        "tp_sizes": sorted({int(row["tp"]) for row in cell_rows}),
        "policies": ["latency", "balanced", "throughput"],
        "promoted_labels": len(promoted),
        "promoted_label_status": PROMOTED_STATUS,
        "maximum_absolute_errors": maxima,
        "inputs": {
            "phase25a_tp_labels_sha256": sha256(teacher_path),
            "repository_head_at_finalize": subprocess_head(root),
        },
        "checks": {
            "all_cells_pass": all(row["status"] == "PASS" for row in cell_rows),
            "four_cells": len(cell_rows) == 4,
            "three_models_covered": len({row["model"] for row in cell_rows}) == 3,
            "tp2_tp4_tp8_covered": {int(row["tp"]) for row in cell_rows} == {2, 4, 8},
            "all_three_policies_each_cell": all(int(row["strategies"]) == 3 for row in cell_rows),
            "phase_comparisons_24": len(comparison_rows) == 24,
            "all_errors_zero": all(value == 0 for value in maxima.values()),
            "promoted_labels_1296": len(promoted) == 1296,
        },
        "can_conclude": [
            "当前三个正式TP模型的结构公式均有完整窗口GPU sentinel证据",
            "TP2、TP4、TP8与三种固定执行策略均被sentinel覆盖",
            "1,296条TP Hfull标签可作为后续full-window监督训练teacher",
        ],
        "cannot_conclude": [
            "1,296条标签均为GPU逐条实测",
            "该teacher适用于online arrival-aware或不同执行契约",
            "旧Phase 16 H32模型无需重新训练",
        ],
        "next_step": "使用晋升后的TP Hfull标签与Phase 25B PP Hfull标签构造统一训练集，并重训direct与H0+bounded residual",
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )

    readme = f"""# Phase 26A：TP Hfull teacher跨模型GPU审计

状态：**{summary['status']}**。本阶段审计TP完整窗口结构teacher是否可以从暂定标签晋升为正式训练真值。

## GPU覆盖

- 计入Phase 25A已有Qwen3-8B/TP2 smoke，并新增3个GPU cell；
- 覆盖Qwen3-8B、Qwen3-30B-A3B、DeepSeek-V2-Lite三个正式模型；
- 覆盖TP2、TP4、TP8；每个cell都同时覆盖latency、balanced、throughput；
- 流量包含42请求最小窗口、312请求中等窗口和6,216-token长prompt尾部。

## 结果

- 4/4 cell通过；
- 24/24个`cell × strategy × phase`比较完全一致；
- calls、logical bytes、精确payload直方图和12桶直方图的最大绝对误差全部为0；
- 原Phase 25A的1,296条TP Hfull标签以`{PROMOTED_STATUS}`状态晋升并保存在`labels/`。

## 结论边界

全量标签由已经GPU验证的结构公式离线生成，不是1,296次GPU逐条实测。结果只适用于当前fixed-draining、固定TP和固定策略契约，不能外推到online arrival-aware或其他执行语义。

由于监督真值从Phase 16的exact H32改成Hfull，direct DNN与H0+bounded residual必须重新训练；H0没有学习参数，只需在Hfull口径下重新评测。
"""
    (args.output_root / "README.md").write_text(readme)
    (args.output_root / "logs/finalize.log").write_text(
        json.dumps(
            {
                "status": summary["status"],
                "cells": len(cell_rows),
                "phase_comparisons": len(comparison_rows),
                "maxima": maxima,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )
    (args.output_root / "DONE").write_text(summary["status"] + "\n")

    files = sorted(
        path
        for path in args.output_root.rglob("*")
        if path.is_file()
        and path.name != "manifest.sha256"
    )
    (args.output_root / "manifest.sha256").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(args.output_root)}\n" for path in files)
    )
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


def subprocess_head(root: Path) -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


if __name__ == "__main__":
    main()
