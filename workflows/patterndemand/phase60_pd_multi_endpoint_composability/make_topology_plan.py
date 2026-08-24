#!/usr/bin/env python3
"""Freeze an external Phase60 topology inventory into the exact 24-shard plan."""
from __future__ import annotations
import argparse,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
from contracts import expand_plan,file_sha,load_json
ROOT=Path(__file__).resolve().parents[3]
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--inventory",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();inventory=a.inventory.expanduser().resolve();output=a.output.expanduser().resolve()
    if output.exists():raise RuntimeError(f"refuse overwrite: {output}")
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip();plan=expand_plan(load_json(inventory),file_sha(inventory),datetime.now(timezone.utc).isoformat(),head);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(plan,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8");print(json.dumps({"status":"PASS","output":str(output),"workflow_commit":head,"plan_sha256":plan["plan_sha256"],"measurements":len(plan["measurements"]),"resource_contract":{"world_size_per_shard":3,"maximum_simultaneous_nodes_per_shard":2,"four_inventory_slots_are_gpu_slots_not_nodes":True,"four_node_allocation_required":False},"warning":"freeze before raw; any endpoint edit invalidates the attempt"},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
