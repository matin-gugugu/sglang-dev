#!/usr/bin/env python3
"""Generate the canonical Phase53 guide and conclusion freeze from pinned results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    environment_record,
    load_json,
    refresh_manifest,
    repo_root,
    utc_now,
    validate_result_tree,
    write_json,
)
from preflight import run_checks  # noqa: E402
from report import (  # noqa: E402
    build_chain_rows,
    build_claim_scope,
    build_evidence_rows,
    load_source_summaries,
    render_freeze_report,
    render_guide,
    write_csv,
)


def run(expected: str, output: Path) -> dict:
    preflight = run_checks(expected)
    spec = load_json(HERE / "experiment.json")
    if output.exists():
        raise RuntimeError(f"refuse overwrite: {output}")
    summaries = load_source_summaries(repo_root(), spec)
    evidence_rows = build_evidence_rows(spec, summaries)
    chain_rows = build_chain_rows(summaries)
    claim_scope = build_claim_scope(spec)
    expected_counts = spec["expected_counts"]
    counts = {
        "source_results": len(spec["source_results"]),
        "source_manifests": sum("verify_manifest_directory" in item for item in spec["pinned_inputs"]),
        "historical_documents": sum("verify_manifest_directory" not in item for item in spec["pinned_inputs"]),
        "evidence_rows": len(evidence_rows),
        "chain_rows": len(chain_rows),
        "frozen_claims": len(claim_scope["frozen_claims"]),
        "prohibited_claims": len(claim_scope["prohibited_claims"]),
        "future_scheduler_dimensions": len(claim_scope["future_scheduler_dimensions"]),
    }
    if counts != expected_counts:
        raise RuntimeError({"counts": counts, "expected": expected_counts})

    output.mkdir(parents=True)
    write_json(output / "contracts/experiment.json", spec)
    write_json(output / "audit/source_freeze.json", preflight)
    write_json(output / "audit/claim_scope.json", claim_scope)
    write_json(
        output / "audit/environment.json",
        {
            **environment_record(),
            "gpu_used": False,
            "network_used": False,
            "training_used": False,
            "checkpoint_loaded": False,
            "prediction_recomputed": False,
            "teacher_recomputed": False,
            "physical_measurement_performed": False,
            "scheduler_simulation_performed": False,
        },
    )
    write_csv(output / "tables/evidence_index.csv", evidence_rows)
    write_csv(output / "tables/phase_chain_status.csv", chain_rows)
    guide = render_guide(summaries, evidence_rows, chain_rows, claim_scope)
    report = render_freeze_report(expected, evidence_rows, chain_rows, claim_scope)
    (output / "docs").mkdir()
    (output / "docs/截至目前实验结构总导引_截至Phase52.md").write_text(guide, encoding="utf-8")
    (output / "docs/Phase53_TP_PP_PD实验链与当前结论冻结报告.md").write_text(report, encoding="utf-8")

    p39 = summaries["Phase39"]
    p50 = summaries["Phase50"]
    p51 = summaries["Phase51"]
    p52 = summaries["Phase52"]
    summary = {
        "schema_version": "phase53-tp-pp-pd-conclusion-freeze-result-v1",
        "status": "PASS",
        "workflow_commit": expected,
        "completed_at_utc": utc_now(),
        "counts": counts,
        "chain_status": {row["chain"]: row["status"] for row in chain_rows},
        "canonical_references": {
            "tp_pp_predictor": "Phase34D",
            "tp_pp_physical_cost_and_placement": "Phase39",
            "pd_predictor": "Phase50",
            "pd_physical_curves": "Phase51",
            "pd_physical_cost_and_placement": "Phase52",
        },
        "headline": {
            "tp_pp_physical_curves": p39["counts"]["physical_curves"],
            "tp_pp_placement_top1": p39["overall_top1_agreement"],
            "pd_six_model_blind_units": p50["counts"]["blind_units"],
            "pd_six_model_composite_ratio": p50["blind_metrics"]["composite_ratio"],
            "pd_physical_curves": p51["counts"]["physical_curves"],
            "pd_cost_confirmed": p52["scientific_outcome"]["cost"],
            "pd_placement_confirmed": p52["scientific_outcome"]["placement"],
        },
        "negative_evidence_preserved": {
            "phase": "Phase43",
            "composite_ratio": summaries["Phase43"]["blind_metrics"]["composite_ratio"],
            "outcome": summaries["Phase43"]["blind_metrics"]["outcome"],
        },
        "conclusions_frozen": True,
        "training_performed": False,
        "checkpoint_loaded": False,
        "prediction_recomputed": False,
        "teacher_recomputed": False,
        "physical_measurement_performed": False,
        "scheduler_simulation_performed": False,
        "gpu_used": False,
        "proved": "an auditable canonical synthesis and claim freeze of the accepted TP, PP and pure-PD evidence through Phase52",
        "not_proved": "any new prediction accuracy, model generalization, physical measurement, end-to-end latency or scheduler benefit",
    }
    write_json(output / "summary.json", summary)
    (output / "README.md").write_text(
        "# Phase53：TP、PP、PD实验链与当前结论冻结\n\n"
        "状态：`PASS`。本阶段核验并索引Phase34D至Phase52的19个正式结果，生成截至Phase52的新总导引和结论边界。"
        "没有训练、推理、teacher重算、物理测量、GPU使用或scheduler仿真。当前TP、PP、纯PD在PatternDemand通信预测范围内冻结完成；"
        "Phase39/52仍只属于communication-only placement，不得解释为完整调度器。\n",
        encoding="utf-8",
    )
    (output / "logs").mkdir()
    (output / "logs/runtime.log").write_text(
        f"completed={utc_now()} workflow_commit={expected}\n"
        f"status=PASS source_results={len(evidence_rows)} chains={len(chain_rows)}\n"
        "phase43_negative_preserved=true conclusions_frozen=true\n"
        "gpu=false network=false training=false checkpoint=false prediction=false teacher=false measurement=false scheduler=false\n",
        encoding="utf-8",
    )
    (output / "DONE").write_text("PASS\n", encoding="utf-8")
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
        default=repo_root() / "experiment-results/phase53_tp_pp_pd_conclusion_freeze",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.expected_workflow_commit, args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
