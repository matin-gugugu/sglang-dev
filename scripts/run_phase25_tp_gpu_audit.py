#!/usr/bin/env python3
"""Run one Phase 25 full-window TP GPU sentinel without saving raw events."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


MODEL_PATHS = {
    "qwen3-8b": "/media/ssd1/Qwen3-8B",
    "qwen3-30b-a3b": "/media/ssd1/Qwen3-30B-A3B",
    "deepseek-v2-lite": "/media/ssd1/DeepSeek-V2-Lite",
}
PLAN_NAMES = {
    "smoke": "tp_smoke_plan.jsonl",
    "qwen_formal": "tp_qwen_formal_plan.jsonl",
    "tp_multimodel": "tp_tp_multimodel_plan.jsonl",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=tuple(PLAN_NAMES), default="smoke")
    parser.add_argument("--model", choices=tuple(MODEL_PATHS), default="qwen3-8b")
    parser.add_argument("--tp", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--profiles", nargs="+")
    parser.add_argument("--attempt", default="r0")
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=root / "experiment-results/phase25_full_window_teacher",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(command: list[str], *, cwd: Path, stdout: Path | None = None) -> None:
    if stdout is None:
        subprocess.run(command, cwd=cwd, check=True)
        return
    with stdout.open("w") as output:
        subprocess.run(command, cwd=cwd, stdout=output, stderr=subprocess.STDOUT, check=True)


def gpu_is_idle() -> bool:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return not result.stdout.strip()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    source_plan = args.teacher_root / "gpu_audit" / "plans" / PLAN_NAMES[args.tier]
    plan_rows = read_jsonl(source_plan)
    if args.profiles:
        selected = set(args.profiles)
        plan_rows = [row for row in plan_rows if row["profile_id"] in selected]
        missing = selected - {row["profile_id"] for row in plan_rows}
        if missing:
            raise ValueError(f"profiles absent from {source_plan}: {sorted(missing)}")
    if not plan_rows:
        raise ValueError("selected TP plan is empty")
    profiles = sorted({row["profile_id"] for row in plan_rows})
    output_dir = (
        args.teacher_root
        / "gpu_audit"
        / "results"
        / "tp"
        / args.tier
        / args.model
        / f"tp{args.tp}"
        / args.attempt
    )
    if (output_dir / "TEACHER_AUDIT_DONE").exists():
        print(f"already complete: {output_dir}")
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"nonempty incomplete attempt: {output_dir}")
    model_path = Path(MODEL_PATHS[args.model])
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if not gpu_is_idle():
        raise RuntimeError("refusing to start: active GPU compute processes found")

    output_dir.mkdir(parents=True)
    filtered_plan = output_dir / "replay_plan.jsonl"
    filtered_plan.write_text(
        "".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in plan_rows)
    )
    config = {
        "schema_version": "phase25-tp-full-window-gpu-run-v1",
        "tier": args.tier,
        "model": args.model,
        "model_path": str(model_path),
        "tp": args.tp,
        "profiles": profiles,
        "workloads": len(plan_rows),
        "capture_mode": "histogram-only",
        "raw_events_saved": False,
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    telemetry = (output_dir / "telemetry.csv").open("w")
    telemetry_process = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,pstate,memory.used,utilization.gpu,power.draw",
            "--format=csv",
            "-l",
            "1",
        ],
        stdout=telemetry,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in range(args.tp))
        env["PYTHONPATH"] = str(root / "python")
        result = output_dir / "result.jsonl"
        command = [
            sys.executable,
            "-m",
            "sglang.benchmark.one_batch",
            "--model-path",
            str(model_path),
            "--tp",
            str(args.tp),
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.85",
            "--disable-cuda-graph",
            "--batch-size",
            "1",
            "--input-len",
            "16",
            "--output-len",
            "2",
            "--trace-replay-plan",
            str(filtered_plan),
            "--comm-profile",
            "--comm-profile-mode",
            "histogram-only",
            "--run-name",
            f"phase25-{args.tier}-{args.model}-tp{args.tp}",
            "--result-filename",
            str(result),
        ]
        with (output_dir / "run.log").open("w") as log:
            subprocess.run(
                command,
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    finally:
        telemetry_process.terminate()
        telemetry_process.wait(timeout=30)
        telemetry.close()

    log_text = (output_dir / "run.log").read_text(errors="replace").lower()
    markers = ("out of memory", "traceback", "cpu fallback", "falling back", "nccl error")
    if any(marker in log_text for marker in markers):
        raise RuntimeError("error marker found in run.log")
    run(
        [
            sys.executable,
            "scripts/validate_profiledemand_gpu_labels.py",
            "--result",
            str(output_dir / "result.jsonl"),
            "--plan",
            str(filtered_plan),
            "--model",
            args.model,
            "--tp",
            str(args.tp),
            "--output-dir",
            str(output_dir),
        ],
        cwd=root,
        stdout=output_dir / "validate.log",
    )
    run(
        [
            sys.executable,
            "scripts/validate_phase25_full_window_gpu_audit.py",
            "--parallelism",
            "tp",
            "--gpu-dir",
            str(output_dir),
            "--model",
            args.model,
            "--parallel-size",
            str(args.tp),
            "--teacher-root",
            str(args.teacher_root),
        ],
        cwd=root,
        stdout=output_dir / "teacher_validate.log",
    )
    (output_dir / "DONE").write_text("PASS\n")
    print(json.dumps({**config, "status": "PASS", "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
