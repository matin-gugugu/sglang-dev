#!/usr/bin/env python3
"""Phase72 read-only identity, manifest and CPU-only preflight."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from common import load_json, repo_root, require_clean_before_run, require_expected_head, run_git, utc_now, verify_pinned_inputs  # noqa:E402


def run_checks(expected: str) -> dict:
    head=require_expected_head(expected)
    require_clean_before_run()
    spec=load_json(HERE/"experiment.json")
    run_git(["merge-base","--is-ancestor",spec["workflow_base_result_commit"],head])
    pins=verify_pinned_inputs(spec)
    source_checks={}
    for row in spec["source_results"]:
        directory=f'experiment-results/{row["directory"]}'
        actual_commit=run_git(["log","-1","--format=%H","--",directory])
        summary=load_json(repo_root()/directory/"summary.json")
        source_checks[row["phase"]]={
            "result_commit":actual_commit==row["result_commit"],
            "status":summary.get("status")==row["expected_status"],
            "done":(repo_root()/directory/"DONE").read_text(encoding="utf-8").strip().startswith("PASS"),
        }
    execution={
        "cuda_hidden_or_unset":os.environ.get("CUDA_VISIBLE_DEVICES") in (None,"","-1"),
        "gpu_forbidden":spec["gpu_required"] is False,
        "network_forbidden":spec["network_required"] is False,
        "training_forbidden":spec["training_permitted"] is False,
    }
    if not all(all(v.values()) for v in source_checks.values()) or not all(execution.values()):
        raise RuntimeError({"source_checks":source_checks,"execution":execution})
    return {"schema_version":"phase72-input-freeze-v1","status":"PASS","workflow_commit":head,"captured_at_utc":utc_now(),"pinned_inputs":pins,"source_checks":source_checks,"formal_result_gaps":spec["formal_result_gap_policy"],"execution":execution}


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--expected-workflow-commit",required=True); args=parser.parse_args()
    print(json.dumps(run_checks(args.expected_workflow_commit),ensure_ascii=False,indent=2))
