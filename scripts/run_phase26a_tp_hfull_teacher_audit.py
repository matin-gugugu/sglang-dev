#!/usr/bin/env python3
"""运行Phase 26A TP Hfull teacher跨模型GPU sentinel审计。"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

from build_profiledemand_replay_plans import STRATEGIES


CELL_SPECS = {
    "qwen3-30b-a3b-tp4-burstgpt": {
        "model": "qwen3-30b-a3b",
        "model_path": "/media/ssd1/Qwen3-30B-A3B",
        "tp": 4,
        "profile_id": "profile_13_burstgpt_3_c2",
        "coverage_reason": "补充MoE模型与TP4；42请求BurstGPT最小完整窗口",
    },
    "deepseek-v2-lite-tp8-burstgpt": {
        "model": "deepseek-v2-lite",
        "model_path": "/media/ssd1/DeepSeek-V2-Lite",
        "tp": 8,
        "profile_id": "profile_03_burstgpt_1_c2",
        "coverage_reason": "补充DeepSeek模型与TP8；312请求BurstGPT中等窗口",
    },
    "qwen3-8b-tp8-long-prompt": {
        "model": "qwen3-8b",
        "model_path": "/media/ssd1/Qwen3-8B",
        "tp": 8,
        "profile_id": "profile_14_burstgpt_3_c3",
        "coverage_reason": "补充Qwen TP8与6,216-token长prompt尾部",
    },
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=(*CELL_SPECS, "all"), required=True)
    parser.add_argument("--attempt", default="r0")
    parser.add_argument(
        "--requests",
        type=Path,
        default=root
        / "experiment-results/phase24_representative_request_convergence/input_windows/selected_requests.jsonl.gz",
    )
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=root / "experiment-results/phase25_full_window_teacher",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "experiment-results/phase26a_tp_hfull_teacher_audit",
    )
    return parser.parse_args()


def read_hfull_windows(path: Path) -> dict[str, dict]:
    result = {}
    with gzip.open(path, "rt") as source:
        for line in source:
            row = json.loads(line)
            if row["sample_label"] != "hfull":
                continue
            inputs = [int(value) for value in row["input_lens"]]
            outputs = [int(value) for value in row["output_lens"]]
            if len(inputs) != len(outputs) or len(inputs) != int(row["request_count"]):
                raise ValueError(f"{row['profile_id']}: invalid Hfull request arrays")
            result[row["profile_id"]] = {**row, "requests": list(zip(inputs, outputs))}
    if len(result) != 24:
        raise ValueError(f"expected 24 Hfull windows, got {len(result)}")
    return result


def tp_batches(requests: list[tuple[int, int]], strategy: dict) -> list[list[tuple[int, int]]]:
    batches: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    current_tokens = 0
    for request in requests:
        if current and (
            len(current) >= int(strategy["max_batch_size"])
            or current_tokens + request[0] > int(strategy["max_prefill_tokens"])
        ):
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(request)
        current_tokens += request[0]
    if current:
        batches.append(current)
    return batches


def build_plan(cell_id: str, spec: dict, window: dict) -> list[dict]:
    rows = []
    for policy, strategy in STRATEGIES.items():
        for batch_index, batch in enumerate(tp_batches(window["requests"], strategy)):
            rows.append(
                {
                    "workload_id": (
                        f"phase26a-{cell_id}-{policy}-batch{batch_index:04d}-r0"
                    ),
                    "profile_id": spec["profile_id"],
                    "source": window["source"],
                    "segment": window["segment"],
                    "split": window["split"],
                    "strategy": policy,
                    "strategy_max_batch_size": int(strategy["max_batch_size"]),
                    "strategy_max_prefill_tokens": int(strategy["max_prefill_tokens"]),
                    "repeat": 0,
                    "batch_index": batch_index,
                    "profile_requests_replayed": len(window["requests"]),
                    "trace_replay_mode": "phase26a_tp_hfull_fixed_draining",
                    "input_lens_per_request": [row[0] for row in batch],
                    "output_lens_per_request": [row[1] for row in batch],
                    "arrival_offsets_ms_audit_only": [0] * len(batch),
                    "chunk_interaction": "fixed_strategy_token_budget",
                }
            )
    return rows


def gpu_is_idle() -> bool:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=True,
    )
    return not result.stdout.strip()


def run_checked(command: list[str], *, cwd: Path, stdout: Path) -> None:
    with stdout.open("w") as output:
        subprocess.run(
            command,
            cwd=cwd,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run_cell(args: argparse.Namespace, root: Path, windows: dict[str, dict], cell_id: str) -> None:
    spec = CELL_SPECS[cell_id]
    output_dir = args.output_root / "results" / cell_id / args.attempt
    if (output_dir / "TEACHER_AUDIT_DONE").exists():
        print(json.dumps({"cell": cell_id, "status": "ALREADY_PASS", "output": str(output_dir)}))
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"nonempty incomplete attempt: {output_dir}")
    if not gpu_is_idle():
        raise RuntimeError("refusing to start: active GPU compute processes found")

    model_path = Path(spec["model_path"])
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise FileNotFoundError(model_path)
    window = windows[spec["profile_id"]]
    plan_rows = build_plan(cell_id, spec, window)
    output_dir.mkdir(parents=True)
    plan_path = output_dir / "replay_plan.jsonl"
    plan_path.write_text(
        "".join(
            json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
            for row in plan_rows
        )
    )
    config = {
        "schema_version": "phase26a-tp-hfull-gpu-run-v1",
        "cell_id": cell_id,
        **spec,
        "request_count": len(window["requests"]),
        "workloads": len(plan_rows),
        "strategies": list(STRATEGIES),
        "max_input_len": max(row[0] for row in window["requests"]),
        "max_output_len": max(row[1] for row in window["requests"]),
        "capture_mode": "histogram-only",
        "raw_events_saved": False,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )

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
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in range(spec["tp"]))
        env["PYTHONPATH"] = str(root / "python")
        command = [
            sys.executable,
            "-m",
            "sglang.benchmark.one_batch",
            "--model-path",
            str(model_path),
            "--tp",
            str(spec["tp"]),
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
            str(plan_path),
            "--comm-profile",
            "--comm-profile-mode",
            "histogram-only",
            "--run-name",
            f"phase26a-{cell_id}",
            "--result-filename",
            str(output_dir / "result.jsonl"),
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
        raise RuntimeError(f"error marker found in {output_dir / 'run.log'}")

    run_checked(
        [
            sys.executable,
            "scripts/validate_profiledemand_gpu_labels.py",
            "--result",
            str(output_dir / "result.jsonl"),
            "--plan",
            str(plan_path),
            "--model",
            spec["model"],
            "--tp",
            str(spec["tp"]),
            "--output-dir",
            str(output_dir),
        ],
        cwd=root,
        stdout=output_dir / "validate.log",
    )
    run_checked(
        [
            sys.executable,
            "scripts/validate_phase25_full_window_gpu_audit.py",
            "--parallelism",
            "tp",
            "--gpu-dir",
            str(output_dir),
            "--model",
            spec["model"],
            "--parallel-size",
            str(spec["tp"]),
            "--teacher-root",
            str(args.teacher_root),
        ],
        cwd=root,
        stdout=output_dir / "teacher_validate.log",
    )
    (output_dir / "DONE").write_text("PASS\n")
    print(json.dumps({**config, "status": "PASS", "output_dir": str(output_dir)}, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    windows = read_hfull_windows(args.requests)
    cell_ids = list(CELL_SPECS) if args.cell == "all" else [args.cell]
    for cell_id in cell_ids:
        run_cell(args, root, windows, cell_id)


if __name__ == "__main__":
    main()
