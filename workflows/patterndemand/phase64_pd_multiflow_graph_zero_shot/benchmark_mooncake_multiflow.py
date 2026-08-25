#!/usr/bin/env python3
"""Production Mooncake benchmark for the four frozen Phase64 communication graphs."""
from __future__ import annotations
import argparse,concurrent.futures,json,math,os,platform,socket,statistics,subprocess,sys,threading,time
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import torch
import torch.distributed as dist
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path.insert(0,str(HERE))
from contracts import graph,iteration_counts,layout_by_id,load_json,measurement_by_id,payload_vectors,validate_plan  # noqa:E402
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import MooncakeTransferEngine  # noqa:E402
FORBIDDEN=("MC_FORCE_TCP","MC_FORCE_MNNVL","MC_INTRANODE_NVLINK","SGLANG_MOONCAKE_CUSTOM_MEM_POOL")
def utc_now()->str:return datetime.now(timezone.utc).isoformat()
def percentile(v:list[float],q:float)->float:
 o=sorted(v);p=(len(o)-1)*q;lo=math.floor(p);hi=math.ceil(p);return o[lo] if lo==hi else o[lo]*(hi-p)+o[hi]*(p-lo)
def stats(v:list[float])->dict[str,float]:return {"min":min(v),"median":statistics.median(v),"p95":percentile(v,.95),"max":max(v)}
def hostname_matches(expected:dict[str,Any])->bool:
 actual={platform.node(),socket.gethostname(),socket.getfqdn(),platform.node().split(".")[0],socket.gethostname().split(".")[0]};aliases=set(expected["host_aliases"]+[expected["host"]]);aliases|={v.split(".")[0] for v in aliases};return bool(actual&aliases)
def main()->None:
 p=argparse.ArgumentParser();p.add_argument("--expected-workflow-commit",required=True);p.add_argument("--topology-plan",type=Path,required=True);p.add_argument("--measurement-id",required=True);p.add_argument("--repeat-id",type=int,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 plan=load_json(a.topology_plan.expanduser().resolve());audit=validate_plan(plan);m=measurement_by_id(plan,a.measurement_id);head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
 if head!=a.expected_workflow_commit or plan["workflow_commit"]!=head:raise RuntimeError({"HEAD":head,"expected":a.expected_workflow_commit,"plan":plan["workflow_commit"]})
 if os.environ.get("MOONCAKE_PROTOCOL")!="rdma" or os.environ.get("WITH_NVIDIA_PEERMEM")!="0" or os.environ.get("SGLANG_DISAGG_STAGING_BUFFER")!="0" or any(os.environ.get(k) is not None for k in FORBIDDEN):raise RuntimeError("transport differs from frozen RDMA/dma-buf/no-fallback contract")
 if Path(os.environ.get("PYTHONPATH","").split(os.pathsep)[0]).resolve()!=(ROOT/"python").resolve():raise RuntimeError("repo python must be first PYTHONPATH")
 dist.init_process_group("gloo");rank=dist.get_rank();world=dist.get_world_size()
 if world!=m["world_size"]:raise RuntimeError({"world_size":world,"expected":m["world_size"]})
 expected=m["ranks"][rank]
 if not hostname_matches(expected):raise RuntimeError({"actual_host":platform.node(),"expected":expected})
 torch.cuda.set_device(0);layout=layout_by_id(m["model_id"]);vectors=payload_vectors(m["model_id"],m["configuration"]);flow_count=m["flow_count"];span=max(flow["payload_bytes"] for vector in vectors for flow in vector["flows"]);buffer=torch.empty(flow_count*span,dtype=torch.uint8,device="cuda");torch.cuda.synchronize()
 engine=MooncakeTransferEngine(hostname=expected["transfer_hostname"],gpu_id=0,ib_device=expected["ib_device"])
 if engine.batch_register([buffer.data_ptr()],[flow_count*span])!=0:raise RuntimeError("Mooncake GPU registration failed")
 props=torch.cuda.get_device_properties(0);endpoint={"rank":rank,"role":expected["role"],"session_id":engine.get_session_id(),"buffer_ptr":buffer.data_ptr(),"buffer_bytes":flow_count*span,"hostname":platform.node(),"expected_host":expected["host"],"physical_gpu":expected["physical_gpu"],"visible_gpu":0,"gpu_name":torch.cuda.get_device_name(0),"gpu_uuid":str(getattr(props,"uuid",None)),"ib_device":engine.get_ib_device(),"mooncake_protocol":os.environ["MOONCAKE_PROTOCOL"],"with_nvidia_peermem":os.environ["WITH_NVIDIA_PEERMEM"],"torch":torch.__version__,"cuda":torch.version.cuda,"python":sys.version};endpoints=[None]*world;dist.all_gather_object(endpoints,endpoint)
 for flow in vectors[0]["flows"]:
  if rank==flow["sender_rank"] and engine.send_probe(endpoints[flow["receiver_rank"]]["session_id"])!=0:raise RuntimeError({"peer_probe_failed":flow["flow_id"]})
 outbound=max((sum(1 for e in graph(m["configuration"])["edges"] if e[0]==r) for r in range(world)),default=1);executor=concurrent.futures.ThreadPoolExecutor(max_workers=max(outbound,1)) if outbound>1 else None;records=[];started=utc_now()
 def materialize(vector:dict)->list[dict]:
  return [{**flow,"source_offset":flow["flow_id"]*span,"destination_offset":flow["flow_id"]*span,"descriptor_count":int(layout["descriptor_count"]),"pattern":17+12*flow["flow_id"]} for flow in vector["flows"]]
 def invoke(flow:dict,gate:threading.Barrier|None=None)->dict:
  if gate is not None:gate.wait()
  peer=endpoints[flow["receiver_rank"]];src0=buffer.data_ptr()+flow["source_offset"];dst0=int(peer["buffer_ptr"])+flow["destination_offset"];count=flow["descriptor_count"];size=flow["descriptor_bytes"]
  wall=time.time_ns();begin=time.perf_counter_ns();ret=engine.batch_transfer_sync(peer["session_id"],[src0+i*size for i in range(count)],[dst0+i*size for i in range(count)],[size]*count);end=time.perf_counter_ns();return {"flow_id":flow["flow_id"],"rank":rank,"ret":ret,"latency_us":(end-begin)/1000.0,"wall_start_ns":wall,"wall_end_ns":time.time_ns()}
 def prepare(flows:list[dict])->None:
  for flow in flows:
   if rank==flow["sender_rank"]:buffer[flow["source_offset"]:flow["source_offset"]+flow["payload_bytes"]].fill_(flow["pattern"])
   if rank==flow["receiver_rank"]:buffer[flow["destination_offset"]:flow["destination_offset"]+flow["payload_bytes"]].zero_()
  torch.cuda.synchronize();dist.barrier()
 def validate(flows:list[dict])->None:
  valid=True
  for flow in flows:
   if rank==flow["receiver_rank"]:valid=valid and bool(torch.all(buffer[flow["destination_offset"]:flow["destination_offset"]+flow["payload_bytes"]]==flow["pattern"]).item())
  flag=torch.tensor([int(valid)],dtype=torch.int32);dist.all_reduce(flag,op=dist.ReduceOp.MIN)
  if int(flag.item())!=1:raise RuntimeError({"data_validation_failed":[f["flow_id"] for f in flows]})
 def execute(flows:list[dict],concurrent_mode:bool)->dict:
  dist.barrier();local=[];mine=[f for f in flows if rank==f["sender_rank"]]
  if concurrent_mode and len(mine)>1:
   gate=threading.Barrier(len(mine));futures=[executor.submit(invoke,f,gate) for f in mine];local=[f.result() for f in futures]
  else:local=[invoke(f) for f in mine]
  gathered=[None]*world;dist.all_gather_object(gathered,local);calls=sorted([r for part in gathered for r in part],key=lambda r:r["flow_id"])
  if len(calls)!=len(flows) or {r["flow_id"] for r in calls}!={f["flow_id"] for f in flows} or any(r["ret"]!=0 for r in calls):raise RuntimeError({"calls":calls})
  starts=[r["wall_start_ns"] for r in calls];return {"calls":calls,"wave_latency_us":max(r["latency_us"] for r in calls),"sender_start_skew_us":(max(starts)-min(starts))/1000.0 if len(starts)>1 else 0.0}
 def run_mode(flows:list[dict],warmup:int,timed:int,concurrent_mode:bool)->dict:
  prepare(flows)
  for _ in range(warmup):execute(flows,concurrent_mode)
  results=[execute(flows,concurrent_mode) for _ in range(timed)];validate(flows);per={str(f["flow_id"]):[next(c for c in r["calls"] if c["flow_id"]==f["flow_id"])["latency_us"] for r in results] for f in flows};waves=[r["wave_latency_us"] for r in results];skews=[r["sender_start_skew_us"] for r in results]
  return {"flow_latency_samples_us":per,"flow_latency_us":{k:stats(v) for k,v in per.items()},"wave_latency_samples_us":waves,"wave_latency_us":stats(waves),"sender_start_skew_samples_us":skews,"sender_start_skew_us":stats(skews),"return_codes_all_zero":True,"data_validation_pass":True}
 try:
  for vector in vectors:
   flows=materialize(vector);warmup,timed=iteration_counts(sum(f["payload_bytes"] for f in flows));solo={str(f["flow_id"]):run_mode([f],warmup,timed,False) for f in flows};concurrent_result=run_mode(flows,warmup,timed,True)
   if rank==0:records.append({"schema_version":"phase64-mooncake-multiflow-raw-v1","workflow_commit":head,"plan_sha256":audit["plan_sha256"],"measurement_sha256":m["measurement_sha256"],"measurement_id":m["measurement_id"],"model_id":m["model_id"],"configuration":m["configuration"],"topology_level":m["topology_level"],"replica_id":m["replica_id"],"placement_id":m["placement_id"],"repeat_id":a.repeat_id,"vector_id":vector["vector_id"],"pages":vector["pages"],"flows":vector["flows"],"descriptor_layout":layout["descriptor_layout"],"descriptor_count":layout["descriptor_count"],"op":"MooncakeTransferEngine.batch_transfer_sync","transport":"rdma","wave_admission":"gloo_barrier_then_all_graph_edges_synchronous_release","concurrency_mechanism":"per_sender_thread_barrier_for_multiple_outbound_edges","warmup_iterations":warmup,"timed_iterations":timed,"solo_flows":solo,"concurrent_wave":concurrent_result,"runtime_endpoints":endpoints,"shard_started_at_utc":started,"timestamp_utc":utc_now()})
  if rank==0:
   output=a.output.expanduser().resolve()
   if output.exists():raise RuntimeError(f"refuse overwrite: {output}")
   output.parent.mkdir(parents=True,exist_ok=True)
   with output.open("x",encoding="utf-8") as f:
    for row in records:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
   print(json.dumps({"status":"PASS","output":str(output),"records":len(records),"measurement_id":m["measurement_id"],"repeat_id":a.repeat_id},ensure_ascii=False))
 finally:
  if executor is not None:executor.shutdown(wait=True)
  try:engine.batch_deregister([buffer.data_ptr()])
  finally:dist.destroy_process_group()
if __name__=="__main__":main()
