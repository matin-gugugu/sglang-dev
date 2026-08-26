#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
from contracts import expand_plan,file_sha,load_json
ROOT=Path(__file__).resolve().parents[3]
def main()->None:
 parser=argparse.ArgumentParser();parser.add_argument("--inventory",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args();inventory=args.inventory.expanduser().resolve();output=args.output.expanduser().resolve()
 if output.exists():raise RuntimeError(f"refuse overwrite: {output}")
 head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip();plan=expand_plan(load_json(inventory),file_sha(inventory),datetime.now(timezone.utc).isoformat(),head);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(plan,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
 print(json.dumps({"status":"PASS","workflow_commit":head,"output":str(output),"plan_sha256":plan["plan_sha256"],"measurements":48,"freshness":plan["placement_summary"],"resource_contract":{"scheduler_reservation_mode":plan["placement_summary"]["scheduler_reservation_mode"],"preferred_reserved_nodes":4,"maximum_reserved_nodes":4,"maximum_active_measurement_nodes":2,"maximum_simultaneous_gpu_processes":5,"maximum_concurrent_shards":1,"four_node_collective_required":False}},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
