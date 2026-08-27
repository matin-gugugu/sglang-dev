#!/usr/bin/env python3
"""Verify Phase73 model scope, exact predictions, metrics, figures and manifest."""
from __future__ import annotations
import argparse,json,math,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE))
from common import load_json,repo_root,verify_result_manifest  # noqa:E402
from analysis import evaluate  # noqa:E402
from gbdt import predict_histograms,read_csv_gz,read_json_gz  # noqa:E402


def verify(output:Path)->dict:
    root=repo_root();spec=load_json(HERE/"experiment.json");expected=load_json(HERE/"expected_outputs.json");summary=load_json(output/"summary.json");training=load_json(output/"audit/training.json");model=read_json_gz(output/"checkpoints/direct_gbdt.json.gz");manifest=verify_result_manifest(output);files={str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()};paths={row["name"]:root/row["path"] for row in spec["pinned_inputs"]};features=read_csv_gz(paths["phase49_features"]);frozen=read_csv_gz(paths["phase49_predictions"]);targets=read_csv_gz(paths["phase50_targets"]);saved=read_csv_gz(output/"predictions/phase50_direct_gbdt_predictions.csv.gz");pc,pb=predict_histograms(model,features);saved_pc=np.asarray([[float(r[f"predicted_calls_bin_{i:02d}"]) for i in range(12)] for r in saved]);saved_pb=np.asarray([[float(r[f"predicted_logical_bytes_bin_{i:02d}"]) for i in range(12)] for r in saved]);recomputed=evaluate(features,frozen,targets,model,spec);saved_aggregate=[]
    import csv
    with (output/"analysis/aggregate_metrics.csv").open(newline="",encoding="utf-8") as source:saved_aggregate=list(csv.DictReader(source))
    agg_ok=len(saved_aggregate)==3 and all(math.isclose(float(row[key]),float(expected_row[key]),rel_tol=1e-11,abs_tol=1e-10) for row,expected_row in zip(saved_aggregate,recomputed["aggregate"]) for key in expected_row if key not in ("method",) and isinstance(expected_row[key],(int,float))) and [r["method"] for r in saved_aggregate]==[r["method"] for r in recomputed["aggregate"]]
    svg_ok=all(ET.parse(output/name).getroot().tag.endswith("svg") for name in ("figures/overall_histogram_wape.svg","figures/model_histogram_wape.svg"));checks={"manifest":manifest["ok"],"required_exact":files==set(expected["required"]),"contract_exact":load_json(output/"contracts/experiment.json")==spec,"status":summary.get("status")=="PASS" and (output/"DONE").read_text().strip()=="PASS","classification":summary.get("benchmark_classification")=="target-open fixed benchmark; not fresh blind","model_scope":model.get("h0_inputs")==0 and model.get("pseudo_requests")==0 and model.get("teacher_calls")==0 and all(name.startswith("feature_") and not name.startswith(("h0_","target_","residual_")) for name in model["feature_names"]),"training_scope":training.get("phase50_labels_loaded_during_selection") is False and training.get("h0_inputs")==0 and training.get("pseudo_requests")==0 and training.get("teacher_calls")==0,"predictions_exact":np.allclose(pc,saved_pc,rtol=1e-12,atol=1e-10) and np.allclose(pb,saved_pb,rtol=1e-12,atol=1e-6),"aggregate_recomputed":agg_ok,"decision_recomputed":summary.get("decision")==recomputed["decision"],"cardinality":len(saved)==1800 and len(recomputed["per_unit"])==5400 and len(recomputed["per_bin"])==72,"svg_valid":svg_ok,"execution_scope":all(summary["execution"].get(key) is False for key in ("gpu_used","network_used","raw_used","teacher_executed","pseudo_requests_constructed","h0_used_as_direct_input")),"no_forbidden_assets":not list(output.rglob("*.jsonl")) and not list(output.rglob("*.pt")) and not list(output.rglob("*.safetensors"))}
    if not all(checks.values()):raise RuntimeError({"checks":checks,"manifest":manifest})
    return {"status":"PASS","workflow_commit":summary["workflow_commit"],"scientific_outcome":summary["scientific_outcome"],"checks":checks,"manifest_files":manifest["manifest"]["checked_files"],"headline":summary["headline"]}


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output-dir",type=Path,default=repo_root()/"experiment-results/phase73_direct_gbdt_baseline");args=parser.parse_args();print(json.dumps(verify(args.output_dir.resolve()),ensure_ascii=False,indent=2))
