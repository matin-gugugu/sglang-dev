#!/usr/bin/env python3
"""Run fixed-draining Phase 25 full-window PP GPU sentinel cells."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
from pathlib import Path

from run_qwen3_8b_pp_pattern_matrix import Workload, run_server_cell


PLAN_NAMES = {
    "smoke": "pp_smoke_requests.jsonl.gz",
    "qwen_formal": "pp_qwen_formal_requests.jsonl.gz",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=tuple(PLAN_NAMES), default="smoke")
    parser.add_argument("--profiles", nargs="+")
    parser.add_argument("--pp-sizes", nargs="+", type=int, choices=(2, 4, 8), default=[2, 4, 8])
    parser.add_argument(
        "--microbatch-sizes", nargs="+", type=int, choices=(1, 4, 16), default=[1, 4, 16]
    )
    parser.add_argument("--model-path", default="/media/ssd1/Qwen3-8B")
    parser.add_argument("--attempt", default="r0")
    parser.add_argument("--port-base", type=int, default=31500)
    parser.add_argument("--startup-timeout", type=float, default=1200)
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=root / "experiment-results/phase25_full_window_teacher",
    )
    return parser.parse_args()


def read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as source:
        return [json.loads(line) for line in source if line.strip()]


def gpu_is_idle() -> bool:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=True,
    )
    return not result.stdout.strip()


def validate_cell(
    *, root: Path, teacher_root: Path, cell: Path, pp_size: int, microbatch: int
) -> dict:
    with (cell / "teacher_validate.log").open("w") as output:
        subprocess.run(
            [
                sys.executable,
                "scripts/validate_phase25_full_window_gpu_audit.py",
                "--parallelism",
                "pp",
                "--gpu-dir",
                str(cell),
                "--model",
                "qwen3-8b",
                "--parallel-size",
                str(pp_size),
                "--policy",
                f"mb{microbatch}",
                "--teacher-root",
                str(teacher_root),
            ],
            cwd=root,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return json.loads((cell / "teacher_audit.json").read_text())


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    plan_path = args.teacher_root / "gpu_audit" / "plans" / PLAN_NAMES[args.tier]
    rows = read_jsonl_gz(plan_path)
    if args.profiles:
        selected = set(args.profiles)
        rows = [row for row in rows if row["profile_id"] in selected]
        missing = selected - {row["profile_id"] for row in rows}
        if missing:
            raise ValueError(f"profiles absent from {plan_path}: {sorted(missing)}")
    if not rows:
        raise ValueError("selected PP plan is empty")
    model_path = Path(args.model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if not gpu_is_idle():
        raise RuntimeError("refusing to start: active GPU compute processes found")
    workloads = [
        Workload(
            name=row["profile_id"],
            input_lens=tuple(map(int, row["input_lens"])),
            output_lens=tuple(map(int, row["output_lens"])),
        )
        for row in rows
    ]
    output_root = args.teacher_root / "gpu_audit" / "results" / "pp" / args.tier
    audits = []
    for pp_index, pp_size in enumerate(dict.fromkeys(args.pp_sizes)):
        for mb_index, microbatch in enumerate(dict.fromkeys(args.microbatch_sizes)):
            cell = output_root / f"pp{pp_size}" / f"mb{microbatch}" / args.attempt
            if (cell / "TEACHER_AUDIT_DONE").exists():
                audits.append(json.loads((cell / "teacher_audit.json").read_text()))
                continue
            if cell.exists() and any(cell.iterdir()):
                raise RuntimeError(f"nonempty incomplete attempt: {cell}")
            audit = run_server_cell(
                repo_root=root,
                output_dir=cell,
                model_path=str(model_path),
                pp_size=pp_size,
                strategy=f"mb{microbatch}",
                pp_microbatch_size=microbatch,
                repeats=1,
                port=args.port_base + pp_index * 10 + mb_index,
                startup_timeout=args.startup_timeout,
                workloads=workloads,
                fixed_request_content=True,
            )
            (cell / "server.pid").unlink(missing_ok=True)
            if audit["status"] != "PASS":
                raise RuntimeError(json.dumps(audit, indent=2))
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
        "schema_version": "phase25-pp-full-window-gpu-matrix-v1",
        "status": "PASS" if all(row["status"] == "PASS" for row in audits) else "FAIL",
        "tier": args.tier,
        "model": "qwen3-8b",
        "profiles": [row["profile_id"] for row in rows],
        "request_count_per_matrix": sum(int(row["request_count"]) for row in rows),
        "pp_sizes": list(dict.fromkeys(args.pp_sizes)),
        "microbatch_sizes": list(dict.fromkeys(args.microbatch_sizes)),
        "cells": len(audits),
        "audits": audits,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "matrix_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps(summary, indent=2))
    (output_root / "MATRIX_DONE").write_text("PASS\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
