#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from contracts import load_json,measurement_by_id,validate_plan
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def main()->None:
 p=argparse.ArgumentParser();p.add_argument("--topology-plan",type=Path,required=True);p.add_argument("--measurement-id",required=True);p.add_argument("--repeat-id",type=int,required=True);p.add_argument("--raw-dir",type=Path,required=True);p.add_argument("--master-addr",required=True);p.add_argument("--master-port",type=int,required=True);a=p.parse_args();plan_path=a.topology_plan.expanduser().resolve();plan=load_json(plan_path);validate_plan(plan);m=measurement_by_id(plan,a.measurement_id);output=a.raw_dir.expanduser().resolve()/m["measurement_id"]/f"repeat_{a.repeat_id:02d}.jsonl";commands=[]
 for e in m["ranks"]:
  argv=["env","-u","MC_FORCE_TCP","-u","MC_FORCE_MNNVL","-u","MC_INTRANODE_NVLINK","-u","SGLANG_MOONCAKE_CUSTOM_MEM_POOL",f"CUDA_VISIBLE_DEVICES={e['physical_gpu']}",f"PYTHONPATH={ROOT/'python'}","MOONCAKE_PROTOCOL=rdma","WITH_NVIDIA_PEERMEM=0","SGLANG_DISAGG_STAGING_BUFFER=0","HF_HUB_OFFLINE=1","TRANSFORMERS_OFFLINE=1",f"RANK={e['rank']}",f"WORLD_SIZE={m['world_size']}","LOCAL_RANK=0",f"MASTER_ADDR={a.master_addr}",f"MASTER_PORT={a.master_port}","python3",str(HERE/"benchmark_mooncake_multiflow.py"),"--expected-workflow-commit",plan["workflow_commit"],"--topology-plan",str(plan_path),"--measurement-id",m["measurement_id"],"--repeat-id",str(a.repeat_id),"--output",str(output)];commands.append({"rank":e["rank"],"role":e["role"],"host":e["host"],"host_aliases":e["host_aliases"],"physical_gpu":e["physical_gpu"],"ib_device":e["ib_device"],"argv":argv})
 nodes=len({c["host"] for c in commands})
 if nodes>2 or len(commands)>5:raise RuntimeError("Phase64 resource contract exceeded")
 print(json.dumps({"schema_version":"phase64-launch-command-set-v1","measurement_id":m["measurement_id"],"model_id":m["model_id"],"configuration":m["configuration"],"topology_level":m["topology_level"],"repeat_id":a.repeat_id,"world_size":m["world_size"],"commands_must_start_concurrently":True,"shared_output_must_not_exist":str(output),"resource_contract":{"scheduler_reservation_mode":plan["placement_summary"]["scheduler_reservation_mode"],"scheduler_may_reserve_nodes":4,"active_measurement_nodes":nodes,"maximum_active_measurement_nodes":2,"simultaneous_gpu_processes":len(commands),"maximum_simultaneous_gpu_processes":5,"maximum_concurrent_measurement_shards":1,"unused_reserved_nodes_must_launch_no_phase64_process":True},"commands":commands},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
