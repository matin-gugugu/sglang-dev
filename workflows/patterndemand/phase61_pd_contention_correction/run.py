#!/usr/bin/env python3
"""Fit and freeze the simplest Phase61 development contention correction."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from common import environment_record, load_json, refresh_manifest, repo_root, utc_now, validate_result_tree, write_json  # noqa: E402
from model import CANDIDATES, candidate_metric_rows, evaluate_candidates, fit_model, predict, prediction_rows, read_points, slice_metrics  # noqa: E402
from preflight import run_checks  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refuse empty Phase61 CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(expected: str, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    preflight = run_checks(expected)
    if output.exists():
        raise RuntimeError(f"refuse overwrite: {output}")
    contract = load_json(HERE / "experiment.json")
    source = repo_root() / contract["dataset_contract"]["source"]
    rows = read_points(source)
    evaluation = evaluate_candidates(rows, contract)
    selected = evaluation["selected"]
    status = "PASS" if selected is not None else "PASS_TARGET_NOT_MET"
    candidate_rows = candidate_metric_rows(evaluation)
    baseline_candidate = next(row for row in candidate_rows if row["candidate_id"] == "phase51_max")
    model_bundle = None
    refit_predictions = prediction_rows(
        rows,
        "phase51_max",
        [float(row["curve_max_us"]) for row in rows],
        "all_development_refit",
    )
    if selected is not None:
        specification = next(value for value in CANDIDATES if value["candidate_id"] == selected["candidate_id"])
        model_bundle = fit_model(rows, specification)
        model_bundle.update({
            "workflow_commit": preflight["workflow_commit"],
            "source_result_commit": contract["workflow_parent_result_commit"],
            "inference_formula": (
                "max(1.0, scale * max(C0,C1))"
                if model_bundle["family"] == "scale"
                else "max(1.0, intercept_us + beta_max * max(C0,C1) + beta_min * min(C0,C1))"
            ),
            "selection_evidence": {
                "method": contract["cross_validation_contract"]["method"],
                "folds": evaluation["pair_folds"],
                "oof_gate": selected["gate"],
                "simplest_passing_candidate": True,
                "more_complex_candidates_not_used_for_selection": True,
            },
            "blind_status": {
                "reserved_features_used": False,
                "reserved_measurements_or_targets_used": False,
                "fresh_blind_validated": False,
            },
        })
        selected_values = [predict(model_bundle, row) for row in rows]
        refit_predictions.extend(prediction_rows(rows, selected["candidate_id"], selected_values, "all_development_refit"))
    refit_slices = []
    for candidate_id in sorted({row["candidate_id"] for row in refit_predictions}):
        subset = [row for row in refit_predictions if row["candidate_id"] == candidate_id]
        refit_slices.extend(slice_metrics(subset, candidate_id, "all_development_refit"))
    selected_oof = None if selected is None else next(row for row in candidate_rows if row["candidate_id"] == selected["candidate_id"])
    selected_refit = None
    if selected is not None:
        selected_refit = next(
            row for row in refit_slices
            if row["candidate_id"] == selected["candidate_id"] and row["slice_type"] == "overall"
        )
    elapsed = time.monotonic() - started
    output.mkdir(parents=True)
    write_json(output / "contracts/experiment.json", contract)
    write_json(output / "audit/preflight.json", preflight)
    write_json(output / "audit/input_freeze.json", {
        "schema_version": "phase61-input-freeze-v1",
        "workflow_commit": preflight["workflow_commit"],
        "source_result_commit": contract["workflow_parent_result_commit"],
        "pinned_inputs": preflight["pinned_inputs"],
        "development_pair_ids": preflight["phase60"]["development_pair_ids"],
        "reserved_pair_ids_audited_only_for_zero_overlap": preflight["phase60"]["reserved_pair_ids_read_only_for_overlap_audit"],
        "reserved_pair_features_used": False,
        "reserved_measurements_or_targets_read": False,
    })
    write_csv(output / "analysis/oof_candidate_metrics.csv", candidate_rows)
    write_csv(output / "analysis/oof_slice_metrics.csv", evaluation["oof_slices"])
    write_csv(output / "analysis/oof_predictions.csv", evaluation["oof_predictions"])
    write_csv(output / "analysis/refit_predictions.csv", refit_predictions)
    write_csv(output / "analysis/refit_slice_metrics.csv", refit_slices)
    if model_bundle is not None:
        write_json(output / "model/contention_correction.json", model_bundle)
    else:
        write_json(output / "model/contention_correction.json", {
            "schema_version": "phase61-contention-model-v1",
            "status": "NOT_FROZEN",
            "reason": "no candidate passed the fixed OOF gate",
            "workflow_commit": preflight["workflow_commit"],
        })
    summary = {
        "schema_version": "phase61-pd-contention-correction-result-v1",
        "status": status,
        "workflow_commit": preflight["workflow_commit"],
        "source_result_commit": contract["workflow_parent_result_commit"],
        "completed_at_utc": utc_now(),
        "runtime_seconds": elapsed,
        "counts": {
            "development_points": len(rows),
            "pair_folds": evaluation["pair_folds"],
            "fit_candidates": len(CANDIDATES),
            "configuration_topology_slices": 6,
            "reserved_future_blind_points_used": 0,
            "gpu_measurements": 0,
        },
        "baseline": {
            "candidate_id": "phase51_max",
            "overall_wape": baseline_candidate["overall_wape"],
            "overall_signed_bias": baseline_candidate["overall_signed_bias"],
        },
        "selection": {
            "selected_candidate_id": None if selected is None else selected["candidate_id"],
            "selected_complexity_rank": None if selected is None else selected["complexity_rank"],
            "simplest_passing_candidate": selected is not None,
            "oof": selected_oof,
            "refit_overall": selected_refit,
        },
        "improvement": None if selected is None else {
            "absolute_wape_reduction": float(baseline_candidate["overall_wape"]) - float(selected_oof["overall_wape"]),
            "relative_wape_reduction": 1.0 - float(selected_oof["overall_wape"]) / float(baseline_candidate["overall_wape"]),
        },
        "formula": None if model_bundle is None else model_bundle["inference_formula"],
        "training_performed": True,
        "gpu_used": False,
        "network_used": False,
        "new_physical_measurement": False,
        "reserved_future_blind_opened": False,
        "fresh_blind_validated": False,
        "next_phase_permitted": status == "PASS",
        "proved": "development OOF selection and full-development freeze of a lightweight two-flow contention correction",
        "not_proved": "fresh blind payload/placement accuracy, P2D2, more than two flows, end-to-end latency, compute, memory, queueing or scheduling",
    }
    write_json(output / "summary.json", summary)
    selected_text = "none" if selected is None else selected["candidate_id"]
    oof_text = "not available" if selected_oof is None else f"{100.0 * float(selected_oof['overall_wape']):.3f}%"
    (output / "README.md").write_text(
        "# Phase61：P1D2/P2D1并发通信修正\n\n"
        f"状态：{status}。Phase51 max baseline OOF口径WAPE为"
        f"{100.0 * float(baseline_candidate['overall_wape']):.3f}%；"
        f"最简单达标候选为{selected_text}，OOF WAPE为{oof_text}。"
        "本阶段只使用Phase60 development物理点并在CPU执行，未打开reserved future blind，未使用GPU或新增物理测量。"
        "模型只有通过后才允许进入Phase62 GPU fresh-blind验证。\n",
        encoding="utf-8",
    )
    (output / "logs").mkdir()
    (output / "logs/runtime.log").write_text(
        f"completed={utc_now()} workflow_commit={preflight['workflow_commit']}\n"
        f"status={status} selected={selected_text} runtime_seconds={elapsed:.6f}\n"
        f"baseline_wape={baseline_candidate['overall_wape']} selected_oof_wape={None if selected_oof is None else selected_oof['overall_wape']}\n"
        "gpu=false network=false new_measurement=false reserved_blind=false\n",
        encoding="utf-8",
    )
    (output / "DONE").write_text(status + "\n", encoding="utf-8")
    refresh_manifest(output)
    tree = validate_result_tree(output)
    if not tree["ok"]:
        raise RuntimeError(tree)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-workflow-commit", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "experiment-results/phase61_pd_contention_correction",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.expected_workflow_commit, args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
