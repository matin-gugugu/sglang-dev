#!/usr/bin/env python3
"""Verify Phase72 exact deterministic artifacts and scope."""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent)); sys.path.insert(0,str(HERE))
from common import load_json, repo_root, verify_result_manifest  # noqa:E402
from report import build_artifacts  # noqa:E402


def verify(output: Path) -> dict:
    spec=load_json(HERE/"experiment.json"); expected=load_json(HERE/"expected_outputs.json"); artifacts=build_artifacts(repo_root(),spec)
    files={str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()}; manifest=verify_result_manifest(output); summary=load_json(output/"summary.json")
    deterministic={**artifacts["tables"],**artifacts["figures"],"docs/PatternDemand_experiment_guide_through_Phase71.md":artifacts["guide"],"docs/Phase72_conclusion_freeze_report.md":artifacts["report"]}
    exact={name:(output/name).read_text(encoding="utf-8")==content for name,content in deterministic.items()}
    svg_ok={name:(ET.parse(output/name).getroot().tag.endswith("svg") and (output/name).stat().st_size>1000) for name in artifacts["figures"]}
    checks={
        "manifest":manifest["ok"],"required_exact":files==set(expected["required"]),"contract_exact":load_json(output/"contracts/experiment.json")==spec,
        "deterministic_artifacts":all(exact.values()),"svg_valid":all(svg_ok.values()),"status":summary.get("status")=="PASS" and (output/"DONE").read_text().strip()=="PASS",
        "counts":summary.get("counts")==spec["expected_counts"],"negative_preserved":len(summary.get("negative_evidence_preserved",[]))==5,
        "strict_target_honest":summary["headline"].get("phase59_strict_target_met") is False,
        "scope":all(summary.get(k) is False for k in ("training_performed","prediction_recomputed","teacher_recomputed","physical_measurement_performed","scheduler_simulation_performed","gpu_used","network_used")),
        "no_forbidden_assets":not list(output.rglob("*.jsonl")) and not list(output.rglob("*.pt")) and not list(output.rglob("*.safetensors")),
    }
    if not all(checks.values()): raise RuntimeError({"checks":checks,"exact":exact,"svg":svg_ok,"manifest":manifest})
    return {"status":"PASS","workflow_commit":summary["workflow_commit"],"checks":checks,"manifest_files":manifest["manifest"]["checked_files"],"headline":summary["headline"]}


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,default=repo_root()/"experiment-results/phase72_conclusion_freeze_through_phase71"); args=parser.parse_args()
    print(json.dumps(verify(args.output_dir.resolve()),ensure_ascii=False,indent=2))
