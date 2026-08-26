#!/usr/bin/env python3
"""Phase66 read-only source, frozen model, runtime and fresh-plan audit."""
from __future__ import annotations
import argparse,importlib.util,inspect,json,os,re,sys
from pathlib import Path
from typing import Any
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE))
from common import load_json,repo_root,require_clean_before_run,require_expected_head,utc_now,verify_pinned_inputs,write_json  # noqa:E402
from contracts import P65,file_sha,frozen_model,validate_blind_contract,validate_plan  # noqa:E402
FORBIDDEN=("MC_FORCE_TCP","MC_FORCE_MNNVL","MC_INTRANODE_NVLINK","SGLANG_MOONCAKE_CUSTOM_MEM_POOL")
def external(path:Path,label:str)->Path:
 value=path.expanduser().resolve();root=repo_root().resolve()
 if value==root or root in value.parents:raise RuntimeError(f"{label} must remain outside Git: {value}")
 return value
def run_checks(expected:str,plan_path:Path,raw_dir:Path,audit_output:Path,container_image:str)->dict[str,Any]:
 spec=load_json(HERE/"experiment.json");head=require_expected_head(expected);require_clean_before_run();pins=verify_pinned_inputs(spec)
 if container_image!=spec["container_contract"]["image"]:raise RuntimeError({"container_image":container_image,"expected":spec["container_contract"]["image"]})
 plan_path=external(plan_path,"topology plan");raw_dir=external(raw_dir,"raw directory");audit_output=external(audit_output,"preflight audit")
 if not plan_path.is_file():raise RuntimeError(f"plan missing: {plan_path}")
 plan=load_json(plan_path);plan_audit=validate_plan(plan);blind_audit=validate_blind_contract(spec)
 if plan.get("workflow_commit")!=head:raise RuntimeError({"plan":plan.get("workflow_commit"),"HEAD":head})
 if raw_dir.exists() and (not raw_dir.is_dir() or any(raw_dir.iterdir())):raise RuntimeError(f"raw must be absent or empty: {raw_dir}")
 if not raw_dir.exists():raw_dir.mkdir(parents=True,exist_ok=False)
 if audit_output.exists():raise RuntimeError(f"refuse overwrite: {audit_output}")
 model_path=P65/"model/multiflow_graph_correction.json";model=frozen_model();model_checks={"file_sha":file_sha(model_path)==spec["frozen_correction_contract"]["sha256"],"candidate":model.get("candidate_id")==spec["frozen_correction_contract"]["candidate_id"],"groups":len(model.get("groups",{}))==spec["frozen_correction_contract"]["expected_groups"],"fresh_blind_not_previously_validated":model.get("blind_status",{}).get("fresh_blind_validated") is False,"formula":model.get("inference_formula")==spec["frozen_correction_contract"]["formula"]}
 if not all(model_checks.values()):raise RuntimeError({"frozen_model":model_checks})
 phase65_summary=load_json(P65/"summary.json")
 if phase65_summary.get("status")!="PASS" or phase65_summary.get("next_phase_permitted") is not True or phase65_summary.get("fresh_blind_validated") is not False:raise RuntimeError("R65 is not the expected development PASS awaiting fresh blind")
 expected_python=(repo_root()/"python").resolve();pythonpath=[Path(value).resolve() for value in os.environ.get("PYTHONPATH","").split(os.pathsep) if value];environment_checks={"repo_python_first":bool(pythonpath) and pythonpath[0]==expected_python,"rdma":os.environ.get("MOONCAKE_PROTOCOL")=="rdma","dmabuf":os.environ.get("WITH_NVIDIA_PEERMEM")=="0","no_staging":os.environ.get("SGLANG_DISAGG_STAGING_BUFFER")=="0","offline_hf":os.environ.get("HF_HUB_OFFLINE")=="1","offline_transformers":os.environ.get("TRANSFORMERS_OFFLINE")=="1","no_forced_fallback":all(os.environ.get(key) is None for key in FORBIDDEN)};module_checks={name:importlib.util.find_spec(name) is not None for name in spec["container_contract"]["required_runtime_modules"]}
 if not all(environment_checks.values()) or not all(module_checks.values()):raise RuntimeError({"environment":environment_checks,"modules":module_checks})
 import mooncake,torch
 from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import MooncakeTransferEngine
 origin=Path(inspect.getsourcefile(MooncakeTransferEngine) or "").resolve();runtime_checks={"cuda_available":torch.cuda.is_available(),"visible_gpu":torch.cuda.device_count()>=1,"production_wrapper_from_repo":origin==expected_python/"sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py","batch_transfer_sync":callable(getattr(MooncakeTransferEngine,"batch_transfer_sync",None)),"batch_register":callable(getattr(MooncakeTransferEngine,"batch_register",None))}
 if not all(runtime_checks.values()):raise RuntimeError({"runtime_checks":runtime_checks})
 result={"schema_version":"phase66-preflight-v1","status":"PASS","captured_at_utc":utc_now(),"workflow_commit":head,"plan_path":str(plan_path),"plan_file_sha256":file_sha(plan_path),"plan_audit":plan_audit,"blind_audit":blind_audit,"raw_dir":str(raw_dir),"pinned_inputs":pins,"frozen_formula":{"file_sha256":file_sha(model_path),"candidate_id":model["candidate_id"],"group_count":len(model["groups"]),"checks":model_checks,"training_or_recalibration_performed":False,"phase66_targets_read":False},"environment_checks":environment_checks,"module_checks":module_checks,"runtime_checks":runtime_checks,"environment":{"declared_container_image":container_image,"python":sys.version,"torch":torch.__version__,"cuda":torch.version.cuda,"cuda_device_count":torch.cuda.device_count(),"gpu_names":[torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],"mooncake_version":getattr(mooncake,"__version__",None),"wrapper_origin":str(origin),"container_image_env":{key:value for key,value in os.environ.items() if re.search(r"(IMAGE|CONTAINER|PYTORCH_VERSION)",key)}}};write_json(audit_output,result);return result
def main()->None:
 parser=argparse.ArgumentParser();parser.add_argument("--expected-workflow-commit",required=True);parser.add_argument("--topology-plan",type=Path,required=True);parser.add_argument("--raw-dir",type=Path,required=True);parser.add_argument("--audit-output",type=Path,required=True);parser.add_argument("--container-image",required=True);args=parser.parse_args();print(json.dumps(run_checks(args.expected_workflow_commit,args.topology_plan,args.raw_dir,args.audit_output,args.container_image),ensure_ascii=False,indent=2))
if __name__=="__main__":main()
