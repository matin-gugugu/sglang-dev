#!/usr/bin/env python3
"""Production Mooncake benchmark for Phase66 frozen fresh-blind graphs."""
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
def percentile(values:list[float],quantile:float)->float:
 ordered=sorted(values);position=(len(ordered)-1)*quantile;low=math.floor(position);high=math.ceil(position);return ordered[low] if low==high else ordered[low]*(high-position)+ordered[high]*(position-low)
def stats(values:list[float])->dict[str,float]:return {"min":min(values),"median":statistics.median(values),"p95":percentile(values,.95),"max":max(values)}
def hostname_matches(expected:dict[str,Any])->bool:
 actual={platform.node(),socket.gethostname(),socket.getfqdn(),platform.node().split(".")[0],socket.gethostname().split(".")[0]};aliases=set(expected["host_aliases"]+[expected["host"]]);aliases|={value.split(".")[0] for value in aliases};return bool(actual&aliases)
def main()->None:
 parser=argparse.ArgumentParser();parser.add_argument("--expected-workflow-commit",required=True);parser.add_argument("--topology-plan",type=Path,required=True);parser.add_argument("--measurement-id",required=True);parser.add_argument("--repeat-id",type=int,required=True);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
 plan=load_json(args.topology_plan.expanduser().resolve());audit=validate_plan(plan);measurement=measurement_by_id(plan,args.measurement_id);head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
 if head!=args.expected_workflow_commit or plan["workflow_commit"]!=head:raise RuntimeError({"HEAD":head,"expected":args.expected_workflow_commit,"plan":plan["workflow_commit"]})
 if os.environ.get("MOONCAKE_PROTOCOL")!="rdma" or os.environ.get("WITH_NVIDIA_PEERMEM")!="0" or os.environ.get("SGLANG_DISAGG_STAGING_BUFFER")!="0" or any(os.environ.get(key) is not None for key in FORBIDDEN):raise RuntimeError("transport differs from frozen RDMA/dma-buf/no-fallback contract")
 if Path(os.environ.get("PYTHONPATH","").split(os.pathsep)[0]).resolve()!=(ROOT/"python").resolve():raise RuntimeError("repo python must be first PYTHONPATH")
 dist.init_process_group("gloo");rank=dist.get_rank();world=dist.get_world_size()
 if world!=measurement["world_size"]:raise RuntimeError({"world_size":world,"expected":measurement["world_size"]})
 expected=measurement["ranks"][rank]
 if not hostname_matches(expected):raise RuntimeError({"actual_host":platform.node(),"expected":expected})
 torch.cuda.set_device(0);layout=layout_by_id(measurement["model_id"]);vectors=payload_vectors(measurement["model_id"],measurement["configuration"]);flow_count=measurement["flow_count"];span=max(flow["payload_bytes"] for vector in vectors for flow in vector["flows"]);buffer=torch.empty(flow_count*span,dtype=torch.uint8,device="cuda");torch.cuda.synchronize()
 engine=MooncakeTransferEngine(hostname=expected["transfer_hostname"],gpu_id=0,ib_device=expected["ib_device"])
 if engine.batch_register([buffer.data_ptr()],[flow_count*span])!=0:raise RuntimeError("Mooncake GPU registration failed")
 properties=torch.cuda.get_device_properties(0);endpoint={"rank":rank,"role":expected["role"],"session_id":engine.get_session_id(),"buffer_ptr":buffer.data_ptr(),"buffer_bytes":flow_count*span,"hostname":platform.node(),"expected_host":expected["host"],"physical_gpu":expected["physical_gpu"],"visible_gpu":0,"gpu_name":torch.cuda.get_device_name(0),"gpu_uuid":str(getattr(properties,"uuid",None)),"ib_device":engine.get_ib_device(),"mooncake_protocol":os.environ["MOONCAKE_PROTOCOL"],"with_nvidia_peermem":os.environ["WITH_NVIDIA_PEERMEM"],"torch":torch.__version__,"cuda":torch.version.cuda,"python":sys.version};endpoints=[None]*world;dist.all_gather_object(endpoints,endpoint)
 for flow in vectors[0]["flows"]:
  if rank==flow["sender_rank"] and engine.send_probe(endpoints[flow["receiver_rank"]]["session_id"])!=0:raise RuntimeError({"peer_probe_failed":flow["flow_id"]})
 outbound=max((sum(1 for edge in graph(measurement["configuration"])["edges"] if edge[0]==candidate) for candidate in range(world)),default=1);executor=concurrent.futures.ThreadPoolExecutor(max_workers=max(outbound,1)) if outbound>1 else None;records=[];started=utc_now()
 def materialize(vector:dict)->list[dict]:return [{**flow,"source_offset":flow["flow_id"]*span,"destination_offset":flow["flow_id"]*span,"descriptor_count":int(layout["descriptor_count"]),"pattern":17+12*flow["flow_id"]} for flow in vector["flows"]]
 def invoke(flow:dict,gate:threading.Barrier|None=None)->dict:
  if gate is not None:gate.wait()
  peer=endpoints[flow["receiver_rank"]];source=buffer.data_ptr()+flow["source_offset"];destination=int(peer["buffer_ptr"])+flow["destination_offset"];count=flow["descriptor_count"];size=flow["descriptor_bytes"];wall=time.time_ns();begin=time.perf_counter_ns();ret=engine.batch_transfer_sync(peer["session_id"],[source+index*size for index in range(count)],[destination+index*size for index in range(count)],[size]*count);end=time.perf_counter_ns();return {"flow_id":flow["flow_id"],"rank":rank,"ret":ret,"latency_us":(end-begin)/1000.0,"wall_start_ns":wall,"wall_end_ns":time.time_ns()}
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
  if int(flag.item())!=1:raise RuntimeError({"data_validation_failed":[flow["flow_id"] for flow in flows]})
 def execute(flows:list[dict],concurrent_mode:bool)->dict:
  dist.barrier();mine=[flow for flow in flows if rank==flow["sender_rank"]]
  if concurrent_mode and len(mine)>1:
   gate=threading.Barrier(len(mine));futures=[executor.submit(invoke,flow,gate) for flow in mine];local=[future.result() for future in futures]
  else:local=[invoke(flow) for flow in mine]
  gathered=[None]*world;dist.all_gather_object(gathered,local);calls=sorted([row for part in gathered for row in part],key=lambda row:row["flow_id"])
  if len(calls)!=len(flows) or {row["flow_id"] for row in calls}!={flow["flow_id"] for flow in flows} or any(row["ret"]!=0 for row in calls):raise RuntimeError({"calls":calls})
  starts=[row["wall_start_ns"] for row in calls];return {"calls":calls,"wave_latency_us":max(row["latency_us"] for row in calls),"sender_start_skew_us":(max(starts)-min(starts))/1000.0 if len(starts)>1 else 0.0}
 def run_mode(flows:list[dict],warmup:int,timed:int,concurrent_mode:bool)->dict:
  prepare(flows)
  for _ in range(warmup):execute(flows,concurrent_mode)
  results=[execute(flows,concurrent_mode) for _ in range(timed)];validate(flows);per={str(flow["flow_id"]):[next(call for call in result["calls"] if call["flow_id"]==flow["flow_id"])["latency_us"] for result in results] for flow in flows};waves=[result["wave_latency_us"] for result in results];skews=[result["sender_start_skew_us"] for result in results];return {"flow_latency_samples_us":per,"flow_latency_us":{key:stats(values) for key,values in per.items()},"wave_latency_samples_us":waves,"wave_latency_us":stats(waves),"sender_start_skew_samples_us":skews,"sender_start_skew_us":stats(skews),"return_codes_all_zero":True,"data_validation_pass":True}
 try:
  for vector in vectors:
   flows=materialize(vector);warmup,timed=iteration_counts(sum(flow["payload_bytes"] for flow in flows));solo={str(flow["flow_id"]):run_mode([flow],warmup,timed,False) for flow in flows};concurrent_result=run_mode(flows,warmup,timed,True)
   if rank==0:records.append({"schema_version":"phase66-mooncake-multiflow-raw-v1","workflow_commit":head,"plan_sha256":audit["plan_sha256"],"measurement_sha256":measurement["measurement_sha256"],"measurement_id":measurement["measurement_id"],"model_id":measurement["model_id"],"configuration":measurement["configuration"],"topology_level":measurement["topology_level"],"replica_id":measurement["replica_id"],"placement_id":measurement["placement_id"],"repeat_id":args.repeat_id,"vector_id":vector["vector_id"],"pages":vector["pages"],"flows":vector["flows"],"descriptor_layout":layout["descriptor_layout"],"descriptor_count":layout["descriptor_count"],"op":"MooncakeTransferEngine.batch_transfer_sync","transport":"rdma","wave_admission":"gloo_barrier_then_all_graph_edges_synchronous_release","concurrency_mechanism":"per_sender_thread_barrier_for_multiple_outbound_edges","warmup_iterations":warmup,"timed_iterations":timed,"solo_flows":solo,"concurrent_wave":concurrent_result,"runtime_endpoints":endpoints,"shard_started_at_utc":started,"timestamp_utc":utc_now()})
  if rank==0:
   output=args.output.expanduser().resolve()
   if output.exists():raise RuntimeError(f"refuse overwrite: {output}")
   output.parent.mkdir(parents=True,exist_ok=True)
   with output.open("x",encoding="utf-8") as stream:
    for row in records:stream.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
   print(json.dumps({"status":"PASS","output":str(output),"records":len(records),"measurement_id":measurement["measurement_id"],"repeat_id":args.repeat_id},ensure_ascii=False))
 finally:
  if executor is not None:executor.shutdown(wait=True)
  try:engine.batch_deregister([buffer.data_ptr()])
  finally:dist.destroy_process_group()
if __name__=="__main__":main()
