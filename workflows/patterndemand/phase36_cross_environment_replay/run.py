#!/usr/bin/env python3
"""一条命令运行Phase36。"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATTERN_WORKFLOWS = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(PATTERN_WORKFLOWS))
sys.path.insert(0, str(ROOT / "scripts"))

from common import environment_record, load_json, sha256, utc_now, write_json
from finalize import finalize
from preflight import run_checks


def make_figure(path: Path, differences: dict[str, float], tolerance: float) -> None:
    rows = sorted((name, value) for name, value in differences.items() if name.endswith("_relative"))
    width, height, margin = 1000, 420, 70
    maximum = max([tolerance, *(value for _, value in rows)]) * 1.25
    bar_width = (width - 2 * margin) / max(len(rows), 1)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', '<text x="70" y="30" font-family="sans-serif" font-size="20">Phase36跨环境复播最大相对差</text>']
    for index, (name, value) in enumerate(rows):
        x = margin + index * bar_width
        bar_height = value / maximum * 260 if maximum else 0
        y = 330 - bar_height
        svg.extend([f'<rect x="{x + 6:.1f}" y="{y:.1f}" width="{bar_width - 12:.1f}" height="{bar_height:.1f}" fill="#2563eb"/>', f'<text x="{x + bar_width/2:.1f}" y="{y - 7:.1f}" text-anchor="middle" font-family="sans-serif" font-size="11">{value:.2e}</text>', f'<text x="{x + bar_width/2:.1f}" y="350" text-anchor="middle" font-family="sans-serif" font-size="9" transform="rotate(25 {x + bar_width/2:.1f} 350)">{name.replace("_relative", "")}</text>'])
    svg.append('</svg>')
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-workflow-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiment-results/phase36_cross_environment_replay")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    preflight = run_checks(args.expected_workflow_commit, output)
    contract = load_json(HERE / "experiment.json")
    for name in ("audit", "predictions", "analysis", "figures", "logs", "contracts"):
        (output / name).mkdir(parents=True, exist_ok=True)

    import torch
    import run_phase35_six_model_inference_cost_integration as phase35

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Phase36必须在CUDA设备复播")
    runtime_args = Namespace(
        phase34a_dir=ROOT / "experiment-results/phase34a_six_model_contract",
        phase34c_dir=ROOT / "experiment-results/phase34c_six_model_target_free_training",
    )
    predictions, runtime_audit = phase35.runtime_predictions(runtime_args, device)
    prediction_path = output / "predictions/replayed_six_model_histograms.csv.gz"
    phase35.write_csv_gz(prediction_path, predictions)
    freeze = {
        "schema_version": "phase36-cross-environment-prediction-freeze-v1",
        "created_at_utc": utc_now(),
        "prediction_sha256": sha256(prediction_path),
        "rows": len(predictions),
        "teacher_or_target_read_before_freeze": False,
        "workflow_commit": preflight["workflow_commit"],
    }
    write_json(output / "predictions/PREDICTION_FREEZE.json", freeze)
    replay = phase35.replay_audit(predictions, runtime_args)
    relative_rows = [
        {"field": name, "max_difference": value, "kind": "relative" if name.endswith("_relative") else "absolute"}
        for name, value in sorted(replay["max_differences"].items())
    ]
    with (output / "analysis/replay_differences.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(relative_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(relative_rows)
    tolerance = float(contract["relative_tolerance"])
    checks = {
        "pinned_inputs_pass": all(value["ok"] for value in preflight["pinned_inputs"].values()),
        "prediction_rows_match_contract": len(predictions) == int(contract["expected_prediction_rows"]),
        "prediction_contains_no_target_columns": not any(name.startswith("target_") for name in predictions[0]),
        "three_seed_fivefold_each_direction": all(value["fold_models"] == 15 for value in runtime_audit.values()),
        "candidate_ids_match": replay["candidate_ids_match"],
        "relative_difference_within_contract": replay["max_scalar_relative_difference"] <= tolerance,
        "all_predictions_finite_nonnegative": all(float(row["predicted_total_calls_per_1000"]) >= 0 and float(row["predicted_total_logical_bytes_per_1000"]) >= 0 for row in predictions),
        "teacher_or_target_never_read": freeze["teacher_or_target_read_before_freeze"] is False,
    }
    write_json(output / "analysis/replay_metrics.json", {"relative_tolerance": tolerance, "replay_audit": replay, "checks": checks})
    write_json(output / "audit/environment.json", {**environment_record(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device), "gpu_name": torch.cuda.get_device_name(device)})
    write_json(output / "contracts/experiment.json", contract)
    make_figure(output / "figures/replay_max_relative_difference.svg", replay["max_differences"], tolerance)
    runtime_state = {
        "workflow_commit": preflight["workflow_commit"],
        "input_audit": preflight["pinned_inputs"],
        "runtime_audit": runtime_audit,
        "replay_audit": replay,
        "counts": {"prediction_rows": len(predictions), "tp_rows": sum(row["parallelism"] == "tp" for row in predictions), "pp_rows": sum(row["parallelism"] == "pp" for row in predictions)},
        "checks": checks,
    }
    write_json(output / "audit/runtime_state.json", runtime_state)
    write_json(output / "logs/runtime.log", {"event": "phase36_complete", "completed_at_utc": utc_now(), "workflow_commit": preflight["workflow_commit"], "training_performed": False, "teacher_or_target_read": False})
    summary = finalize(output)
    print(json.dumps({"status": summary["status"], "output": str(output), "rows": len(predictions), "max_relative_difference": replay["max_scalar_relative_difference"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
