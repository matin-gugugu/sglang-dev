#!/usr/bin/env python3
"""Generate the Phase72 conclusion freeze."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent)); sys.path.insert(0,str(HERE))
from common import environment_record, load_json, refresh_manifest, repo_root, utc_now, validate_result_tree, write_json  # noqa:E402
from preflight import run_checks  # noqa:E402
from report import build_artifacts  # noqa:E402


def run(expected: str, output: Path) -> dict:
    freeze=run_checks(expected); spec=load_json(HERE/"experiment.json")
    if output.exists(): raise RuntimeError(f"refuse overwrite: {output}")
    artifacts=build_artifacts(repo_root(),spec)
    counts={
        "source_results":len(artifacts["evidence"]),
        "verified_manifests":sum("verify_manifest_directory" in x for x in spec["pinned_inputs"]),
        "formal_result_gaps":len(artifacts["gaps"]),
        "evidence_rows":len(artifacts["evidence"]),
        "frozen_claims":len(artifacts["claims"]["frozen_claims"]),
        "prohibited_claims":len(artifacts["claims"]["prohibited_claims"]),
        "figures":len(artifacts["figures"]),
    }
    if counts!=spec["expected_counts"]: raise RuntimeError({"counts":counts,"expected":spec["expected_counts"]})
    output.mkdir(parents=True)
    write_json(output/"contracts/experiment.json",spec); write_json(output/"audit/input_freeze.json",freeze); write_json(output/"audit/claim_scope.json",artifacts["claims"])
    write_json(output/"audit/environment.json",{**environment_record(),"gpu_used":False,"network_used":False,"training_performed":False,"prediction_recomputed":False,"teacher_recomputed":False,"physical_measurement_performed":False,"scheduler_simulation_performed":False})
    for relative,content in artifacts["tables"].items(): path=output/relative; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(content,encoding="utf-8")
    for relative,content in artifacts["figures"].items(): path=output/relative; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(content,encoding="utf-8")
    (output/"docs").mkdir(); (output/"docs/PatternDemand_experiment_guide_through_Phase71.md").write_text(artifacts["guide"],encoding="utf-8"); (output/"docs/Phase72_conclusion_freeze_report.md").write_text(artifacts["report"],encoding="utf-8")
    p59=artifacts["sources"]["Phase59"]["development_validation"]; p70=artifacts["sources"]["Phase70"]["metrics"]; p71=artifacts["sources"]["Phase71"]["headline"]
    summary={
        "schema_version":"phase72-conclusion-freeze-through-phase71-result-v1","status":"PASS","scientific_outcome":"CONCLUSIONS_FROZEN_THROUGH_PHASE71","workflow_commit":expected,"completed_at_utc":utc_now(),"counts":counts,
        "headline":{"phase59_calls_histogram_wape":p59["h0_plus_dnn_refined"]["calls_histogram_wape"],"phase59_bytes_histogram_wape":p59["h0_plus_dnn_refined"]["bytes_histogram_wape"],"phase59_strict_target_met":False,"phase70_r69_overall_wape":p70["phase69_overall_wape"],"phase70_scope_models":2,"phase71_cost_checks":"21/21","phase71_placement_checks":"7/7","phase71_maximum_dnn_cost_wape":p71["maximum_dnn_cost_wape"],"phase71_minimum_dnn_placement_agreement":p71["minimum_dnn_placement_agreement"],"phase71_maximum_wave_relative_cost_range":p71["maximum_wave_policy_relative_cost_range"],"phase71_minimum_wave_placement_stability":p71["minimum_wave_policy_placement_stability"]},
        "negative_evidence_preserved":["Phase58 target not met","Phase59 target not met","Phase64 zero-shot fail","Phase66 fresh-blind fail","Phase68 second fresh-blind fail"],
        "gpu_scope_status":"NO_MANDATORY_RERUN_IN_FROZEN_SCOPE","training_performed":False,"prediction_recomputed":False,"teacher_recomputed":False,"physical_measurement_performed":False,"scheduler_simulation_performed":False,"gpu_used":False,"network_used":False,
        "proved":"auditable synthesis of formal TP/PP/PD evidence and claim boundaries through Phase71",
        "not_proved":"strict PD histogram target attainment, arbitrary wave recovery, unmeasured models/graphs or full scheduler benefit",
    }
    write_json(output/"summary.json",summary)
    (output/"README.md").write_text("# Phase72：截至Phase71的结论冻结\n\n状态：`PASS`。正式核验Phase53与Phase58–71，生成新版总导引、证据索引、结论边界和四张确定性论文图。没有GPU、训练、预测、teacher或物理测量。PD严格直方图精度仍未达标；Phase71多流结论只在预注册wave合同与已测范围内成立。\n",encoding="utf-8")
    (output/"logs").mkdir(); (output/"logs/runtime.log").write_text(f"completed={utc_now()} workflow_commit={expected}\nstatus=PASS source_results={counts['source_results']} figures={counts['figures']}\ngpu=false network=false training=false prediction=false teacher=false measurement=false scheduler=false\n",encoding="utf-8")
    (output/"DONE").write_text("PASS\n",encoding="utf-8"); refresh_manifest(output)
    tree=validate_result_tree(output)
    if not tree["ok"]: raise RuntimeError(tree)
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--expected-workflow-commit",required=True); parser.add_argument("--output-dir",type=Path,default=repo_root()/"experiment-results/phase72_conclusion_freeze_through_phase71"); args=parser.parse_args()
    print(json.dumps(run(args.expected_workflow_commit,args.output_dir.resolve()),ensure_ascii=False,indent=2))
