#!/usr/bin/env python3
"""Run a small full-window PP GPU tail audit for the Phase 25B teacher."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
from pathlib import Path

from run_phase25_pp_gpu_audit import gpu_is_idle, validate_cell
from run_qwen3_8b_pp_pattern_matrix import Workload, run_server_cell


DEFAULT_PROFILES = (
    "profile_14_burstgpt_3_c3",
    "profile_18_mooncake_conversation_c2",
)
DEFAULT_CELLS = ((2, 1), (4, 4), (8, 16))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", nargs="+", default=list(DEFAULT_PROFILES))
    parser.add_argument(
        "--cells",
        nargs="+",
        default=[f"pp{pp}-mb{mb}" for pp, mb in DEFAULT_CELLS],
        help="Diagonal audit cells, for example pp2-mb1 pp4-mb4 pp8-mb16.",
    )
    parser.add_argument("--model-path", default="/media/ssd1/Qwen3-8B")
    parser.add_argument("--attempt", default="r0")
    parser.add_argument("--port-base", type=int, default=32500)
    parser.add_argument("--startup-timeout", type=float, default=1200)
    parser.add_argument(
        "--phase25a-root",
        type=Path,
        default=root / "experiment-results/phase25_full_window_teacher",
    )
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=root / "experiment-results/phase25b_pp_scheduler_teacher",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase25c_pp_scheduler_tail_audit",
    )
    return parser.parse_args()


def parse_cell(value: str) -> tuple[int, int]:
    try:
        pp_text, mb_text = value.split("-", 1)
        pp_size, microbatch = int(pp_text.removeprefix("pp")), int(
            mb_text.removeprefix("mb")
        )
    except Exception as exc:
        raise ValueError(f"invalid cell {value!r}") from exc
    if pp_size not in (2, 4, 8) or microbatch not in (1, 4, 16):
        raise ValueError(f"unsupported cell {value!r}")
    return pp_size, microbatch


def read_plan(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    cells = list(dict.fromkeys(parse_cell(value) for value in args.cells))
    selected = set(args.profiles)
    rows = [
        row
        for row in read_plan(
            args.phase25a_root / "gpu_audit/plans/pp_qwen_formal_requests.jsonl.gz"
        )
        if row["profile_id"] in selected
    ]
    if {row["profile_id"] for row in rows} != selected:
        raise ValueError("one or more selected profiles are absent from the Phase 25 plan")
    if not Path(args.model_path).is_dir():
        raise FileNotFoundError(args.model_path)
    if not gpu_is_idle():
        raise RuntimeError("refusing to start: active GPU compute processes found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema_version": "phase25c-pp-scheduler-tail-plan-v1",
        "model": "qwen3-8b",
        "profiles": [row["profile_id"] for row in rows],
        "profile_metadata": [
            {
                key: row[key]
                for key in ("profile_id", "source", "segment", "window_id", "request_count")
            }
            for row in rows
        ],
        "cells": [
            {"pp_size": pp_size, "microbatch": microbatch}
            for pp_size, microbatch in cells
        ],
        "profile_cells": len(rows) * len(cells),
        "selection_contract": "BurstGPT long-prompt small window plus Mooncake long-context full window; diagonal PP/MB coverage",
        "phase25b_teacher": str(args.teacher_root),
    }
    (args.output_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    workloads = [
        Workload(
            name=row["profile_id"],
            input_lens=tuple(map(int, row["input_lens"])),
            output_lens=tuple(map(int, row["output_lens"])),
        )
        for row in rows
    ]
    audits = []
    for index, (pp_size, microbatch) in enumerate(cells):
        cell = (
            args.output_dir
            / "results"
            / f"pp{pp_size}"
            / f"mb{microbatch}"
            / args.attempt
        )
        if (cell / "TEACHER_AUDIT_DONE").exists():
            audits.append(json.loads((cell / "teacher_audit.json").read_text()))
            continue
        if (cell / "DONE").exists():
            (cell / "server.pid").unlink(missing_ok=True)
            audits.append(
                validate_cell(
                    root=root,
                    teacher_root=args.teacher_root,
                    cell=cell,
                    pp_size=pp_size,
                    microbatch=microbatch,
                )
            )
            continue
        if cell.exists() and any(cell.iterdir()):
            raise RuntimeError(f"nonempty incomplete attempt: {cell}")
        integrity = run_server_cell(
            repo_root=root,
            output_dir=cell,
            model_path=args.model_path,
            pp_size=pp_size,
            strategy=f"mb{microbatch}",
            pp_microbatch_size=microbatch,
            repeats=1,
            port=args.port_base + index,
            startup_timeout=args.startup_timeout,
            workloads=workloads,
            fixed_request_content=True,
        )
        (cell / "server.pid").unlink(missing_ok=True)
        if integrity["status"] != "PASS":
            raise RuntimeError(json.dumps(integrity, indent=2))
        audits.append(
            validate_cell(
                root=root,
                teacher_root=args.teacher_root,
                cell=cell,
                pp_size=pp_size,
                microbatch=microbatch,
            )
        )

    summary = {
        "schema_version": "phase25c-pp-scheduler-tail-run-v1",
        "status": "PASS"
        if len(audits) == len(cells)
        and all(row["checks"]["gpu_integrity"] for row in audits)
        and all(row["checks"]["teacher_exact_match"] for row in audits)
        else "FAIL",
        "profiles": [row["profile_id"] for row in rows],
        "cells": len(cells),
        "profile_cells": len(rows) * len(cells),
        "phase_comparisons": sum(int(row["phase_comparisons"]) for row in audits),
        "audits": audits,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps(summary, indent=2))
    (args.output_dir / "RUN_DONE").write_text("PASS\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
