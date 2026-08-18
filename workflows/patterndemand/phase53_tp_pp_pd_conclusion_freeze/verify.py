#!/usr/bin/env python3
"""Independent verifier for the Phase53 canonical synthesis and claim freeze."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import load_json, repo_root, verify_result_manifest  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def verify(output: Path) -> dict:
    expected = load_json(HERE / "expected_outputs.json")
    required = set(expected["required"])
    actual = {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()}
    missing = sorted(required - actual)
    if missing:
        raise RuntimeError({"missing": missing})
    manifest = verify_result_manifest(output)
    workflow = load_json(HERE / "experiment.json")
    result_contract = load_json(output / "contracts/experiment.json")
    summary = load_json(output / "summary.json")
    freeze = load_json(output / "audit/source_freeze.json")
    scope = load_json(output / "audit/claim_scope.json")
    environment = load_json(output / "audit/environment.json")
    evidence = read_csv(output / "tables/evidence_index.csv")
    chains = read_csv(output / "tables/phase_chain_status.csv")
    guide = (output / "docs/PatternDemand_experiment_guide_through_Phase52.md").read_text(encoding="utf-8")
    report = (output / "docs/Phase53_TP_PP_PD_conclusion_freeze_report.md").read_text(encoding="utf-8")
    expected_phases = [row["phase"] for row in workflow["source_results"]]
    expected_counts = workflow["expected_counts"]
    required_guide_terms = [
        "Phase43",
        "composite ratio",
        "DNN不如H0",
        "Phase46",
        "Phase50",
        "Phase39",
        "Phase52",
        "communication-only",
        "完整调度器",
        "不进入最终预测器",
        "TP/PP size",
        "计算时间",
        "显存可行性",
        "排队和拥塞",
        "通信计算重叠",
    ]
    checks = {
        "manifest": manifest["ok"],
        "required_exact": actual == required,
        "status": summary.get("status") == "PASS" and (output / "DONE").read_text().strip() == "PASS",
        "contract_exact": result_contract == workflow,
        "lineage": freeze.get("workflow_base_result_commit") == workflow["workflow_base_result_commit"]
        and freeze.get("workflow_parent_commits") == workflow["required_workflow_ancestors"][-1:]
        and all(freeze.get("workflow_lineage", {}).values()),
        "pins": all(row.get("ok") is True for row in freeze.get("pinned_inputs", {}).values()),
        "source_commits": len(freeze.get("source_result_audits", [])) == expected_counts["source_results"]
        and all(row.get("commit_matches") and row.get("commit_is_ancestor") for row in freeze["source_result_audits"]),
        "source_statuses": all(row.get("status_matches") for row in freeze.get("source_result_audits", [])),
        "scientific_checks": all(freeze.get("scientific_checks", {}).values()),
        "evidence_rows": len(evidence) == expected_counts["evidence_rows"]
        and [row["phase"] for row in evidence] == expected_phases
        and len({row["result_commit"] for row in evidence}) == len(evidence),
        "negative_evidence": next(row for row in evidence if row["phase"] == "Phase43")["evidence_class"] == "fresh_blind_negative"
        and float(summary["negative_evidence_preserved"]["composite_ratio"]) > 1.0,
        "chains": len(chains) == expected_counts["chain_rows"]
        and {row["chain"] for row in chains} == {"TP", "PP", "PD"}
        and all(row["status"] == "FROZEN_COMPLETE_WITHIN_SCOPE" for row in chains),
        "claims": len(scope.get("frozen_claims", [])) == expected_counts["frozen_claims"]
        and len(scope.get("prohibited_claims", [])) == expected_counts["prohibited_claims"]
        and len(scope.get("future_scheduler_dimensions", [])) == expected_counts["future_scheduler_dimensions"],
        "canonical_references": summary.get("canonical_references")
        == {
            "tp_pp_predictor": "Phase34D",
            "tp_pp_physical_cost_and_placement": "Phase39",
            "pd_predictor": "Phase50",
            "pd_physical_curves": "Phase51",
            "pd_physical_cost_and_placement": "Phase52",
        },
        "guide_scope": all(term in guide for term in required_guide_terms),
        "report_scope": all(term in report for term in ("Phase43", "有效负结果", "Phase39", "Phase51", "完整scheduler")),
        "no_new_experiment": summary.get("conclusions_frozen") is True
        and all(
            summary.get(name) is False
            for name in (
                "training_performed",
                "checkpoint_loaded",
                "prediction_recomputed",
                "teacher_recomputed",
                "physical_measurement_performed",
                "scheduler_simulation_performed",
                "gpu_used",
            )
        ),
        "environment_scope": all(
            environment.get(name) is False
            for name in (
                "gpu_used",
                "network_used",
                "training_used",
                "checkpoint_loaded",
                "prediction_recomputed",
                "teacher_recomputed",
                "physical_measurement_performed",
                "scheduler_simulation_performed",
            )
        ),
        "no_raw_or_model": not list(output.rglob("*.jsonl"))
        and not list(output.rglob("*.safetensors"))
        and not list(output.rglob("*.pt")),
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "manifest": manifest})
    return {
        "status": "PASS",
        "checks": checks,
        "workflow_commit": summary["workflow_commit"],
        "source_results": len(evidence),
        "chains": [row["chain"] for row in chains],
        "frozen_claims": len(scope["frozen_claims"]),
        "manifest_files": manifest["manifest"]["checked_files"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "experiment-results/phase53_tp_pp_pd_conclusion_freeze",
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
