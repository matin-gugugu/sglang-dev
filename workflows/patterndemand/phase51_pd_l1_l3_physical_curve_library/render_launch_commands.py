#!/usr/bin/env python3
"""Render exact per-host torchrun commands for one Phase51 shard/repeat."""
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
from contracts import load_json,measurement_by_id,validate_plan
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--topology-plan",type=Path,required=True);p.add_argument("--measurement-id",required=True);p.add_argument("--repeat-id",type=int,required=True);p.add_argument("--raw-dir",type=Path,required=True);p.add_argument("--master-addr",required=True);p.add_argument("--master-port",type=int,required=True);a=p.parse_args();plan=load_json(a.topology_plan.expanduser().resolve());validate_plan(plan);m=measurement_by_id(plan,a.measurement_id);output=a.raw_dir.expanduser().resolve()/m["measurement_id"]/f"repeat_{a.repeat_id:02d}.jsonl"
    by_host=defaultdict(list)
    for rank in m["ranks"]:by_host[rank["host"]].append(rank)
    commands=[]
    for node_rank,(host,ranks) in enumerate(by_host.items()):
        ranks=sorted(ranks,key=lambda row:row["rank"]);cuda=",".join(str(row["physical_gpu"]) for row in ranks);argv=["env","-u","MC_FORCE_TCP","-u","MC_FORCE_MNNVL","-u","MC_INTRANODE_NVLINK","-u","SGLANG_MOONCAKE_CUSTOM_MEM_POOL",f"CUDA_VISIBLE_DEVICES={cuda}",f"PYTHONPATH={ROOT/'python'}","MOONCAKE_PROTOCOL=rdma","WITH_NVIDIA_PEERMEM=0","SGLANG_DISAGG_STAGING_BUFFER=0","HF_HUB_OFFLINE=1","TRANSFORMERS_OFFLINE=1","torchrun","--nnodes",str(len(by_host)),"--nproc-per-node",str(len(ranks)),"--node-rank",str(node_rank),"--master-addr",a.master_addr,"--master-port",str(a.master_port),str(HERE/"benchmark_mooncake.py"),"--expected-workflow-commit",plan["workflow_commit"],"--topology-plan",str(a.topology_plan.expanduser().resolve()),"--measurement-id",m["measurement_id"],"--repeat-id",str(a.repeat_id),"--output",str(output)]
        commands.append({"host":host,"host_aliases":ranks[0]["host_aliases"],"node_rank":node_rank,"global_ranks":[row["rank"] for row in ranks],"physical_gpus":[row["physical_gpu"] for row in ranks],"ib_devices":[row["ib_device"] for row in ranks],"argv":argv})
    print(json.dumps({"schema_version":"phase51-launch-command-set-v1","measurement_id":m["measurement_id"],"model_id":m["model_id"],"topology_level":m["topology_level"],"repeat_id":a.repeat_id,"world_size":2,"commands_must_start_concurrently":True,"output_must_not_exist":str(output),"commands":commands},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
