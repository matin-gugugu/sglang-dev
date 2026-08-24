#!/usr/bin/env python3
"""Render exact per-rank launch commands for one Phase60 shard/repeat."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from contracts import load_json,measurement_by_id,validate_plan
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--topology-plan",type=Path,required=True);p.add_argument("--measurement-id",required=True);p.add_argument("--repeat-id",type=int,required=True);p.add_argument("--raw-dir",type=Path,required=True);p.add_argument("--master-addr",required=True);p.add_argument("--master-port",type=int,required=True);a=p.parse_args();plan_path=a.topology_plan.expanduser().resolve();plan=load_json(plan_path);validate_plan(plan);m=measurement_by_id(plan,a.measurement_id);output=a.raw_dir.expanduser().resolve()/m["measurement_id"]/f"repeat_{a.repeat_id:02d}.jsonl";commands=[]
    for endpoint in m["ranks"]:
        argv=["env","-u","MC_FORCE_TCP","-u","MC_FORCE_MNNVL","-u","MC_INTRANODE_NVLINK","-u","SGLANG_MOONCAKE_CUSTOM_MEM_POOL",f"CUDA_VISIBLE_DEVICES={endpoint['physical_gpu']}",f"PYTHONPATH={ROOT/'python'}","MOONCAKE_PROTOCOL=rdma","WITH_NVIDIA_PEERMEM=0","SGLANG_DISAGG_STAGING_BUFFER=0","HF_HUB_OFFLINE=1","TRANSFORMERS_OFFLINE=1",f"RANK={endpoint['rank']}","WORLD_SIZE=3","LOCAL_RANK=0",f"MASTER_ADDR={a.master_addr}",f"MASTER_PORT={a.master_port}","python3",str(HERE/"benchmark_mooncake_multi.py"),"--expected-workflow-commit",plan["workflow_commit"],"--topology-plan",str(plan_path),"--measurement-id",m["measurement_id"],"--repeat-id",str(a.repeat_id),"--output",str(output)]
        commands.append({"rank":endpoint["rank"],"role":endpoint["role"],"host":endpoint["host"],"host_aliases":endpoint["host_aliases"],"physical_gpu":endpoint["physical_gpu"],"ib_device":endpoint["ib_device"],"argv":argv})
    print(json.dumps({"schema_version":"phase60-launch-command-set-v1","measurement_id":m["measurement_id"],"model_id":m["model_id"],"configuration":m["configuration"],"topology_level":m["topology_level"],"repeat_id":a.repeat_id,"world_size":3,"commands_must_start_concurrently":True,"one_process_per_command":True,"shared_output_must_not_exist":str(output),"commands":commands},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
