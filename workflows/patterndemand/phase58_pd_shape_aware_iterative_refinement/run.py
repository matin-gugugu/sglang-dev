#!/usr/bin/env python3
"""Phase58: bounded OOF-only PD shape-aware refinement."""
from __future__ import annotations
import argparse, copy, csv, importlib.util, json, statistics, sys
from pathlib import Path
from typing import Any
import numpy as np

HERE = Path(__file__).resolve().parent; P57 = HERE.parent / "phase57_pd_iterative_histogram_optimization"
sys.path.insert(0, str(P57)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from common import environment_record, load_json, refresh_manifest, repo_root, utc_now, write_json  # noqa: E402
from model_loader import read_csv_gz  # noqa: E402

def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

P57RUN = load_module("phase58_p57_run", P57 / "run.py")
PREFLIGHT = load_module("phase58_preflight", HERE / "preflight.py")
P54 = P57RUN.P54RUN; BINS = 12

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: raise ValueError(f"empty output {path}")
    path.parent.mkdir(parents=True, exist_ok=True); fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)

def target_logdiff(rows: list[dict[str, str]]) -> np.ndarray:
    h0c, h0b = P54.histogram_arrays(rows, "h0"); tc, tb = P54.histogram_arrays(rows, "target")
    return np.concatenate([np.log1p(np.maximum(tc, 0.0)) - np.log1p(np.maximum(h0c, 0.0)), np.log1p(np.maximum(tb, 0.0)) - np.log1p(np.maximum(h0b, 0.0))], axis=1)

def decode_logdiff(rows: list[dict[str, str]], value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h0c, h0b = P54.histogram_arrays(rows, "h0"); value = np.clip(value, -8.0, 8.0)
    return np.expm1(np.clip(np.log1p(h0c) + value[:, :BINS], -20.0, 40.0)), np.expm1(np.clip(np.log1p(h0b) + value[:, BINS:], -20.0, 50.0))

def ratio_oof(train: list[dict[str, str]], config: dict[str, Any], folds: dict[str, int]):
    calls = np.zeros((len(train), BINS)); bytes_ = np.zeros_like(calls); masks_c = np.ones_like(calls, dtype=bool); masks_b = np.ones_like(calls, dtype=bool); epochs = []
    groups = P57RUN.groups_for(train, config["scope"])
    for fold in range(4):
        fit_idx = [i for i, r in enumerate(train) if folds[r["profile_id"]] != fold]; hold_idx = [i for i, r in enumerate(train) if folds[r["profile_id"]] == fold]
        fit_set, hold_set = set(fit_idx), set(hold_idx)
        for group, group_idx in sorted(groups.items()):
            fit_i = [i for i in group_idx if i in fit_set]; hold_i = [i for i in group_idx if i in hold_set]
            if not hold_i: continue
            fit_rows = [train[i] for i in fit_i]; hold_rows = [train[i] for i in hold_i]
            x_fit, _ = P57RUN.feature_matrix(fit_rows, config["feature_mode"]); x_hold, _ = P57RUN.feature_matrix(hold_rows, config["feature_mode"]); y_fit = target_logdiff(fit_rows)
            fitted, fit_pred, _ = P57RUN.fit_ridge(x_fit, y_fit, float(config["l2"])); pred = P57RUN.ridge_predict(fitted, x_hold)
            if config.get("calibration", False): pred += float(config.get("calibration_strength", 0.5)) * np.mean(y_fit - fit_pred, axis=0)
            calls[hold_i], bytes_[hold_i] = decode_logdiff(hold_rows, pred)
            if config.get("support_aware", False):
                tc, tb = P54.histogram_arrays(fit_rows, "target"); threshold = float(config.get("support_threshold", 0.08)); masks_c[hold_i] = (tc > 1e-9).mean(axis=0)[None, :] >= threshold; masks_b[hold_i] = (tb > 1e-9).mean(axis=0)[None, :] >= threshold
            epochs.append(1)
    return calls, bytes_, masks_c, masks_b, epochs

def apply_alpha(rows: list[dict[str, str]], calls: np.ndarray, bytes_: np.ndarray, alpha_map: dict[str, float]):
    h0c, h0b = P54.histogram_arrays(rows, "h0"); base = P54.encode_histograms(h0c, h0b); pred = P54.encode_histograms(calls, bytes_); alpha = np.asarray([float(alpha_map[r["model"]]) for r in rows])[:, None]
    return P54.decode_histograms(base + alpha * (pred - base))

def seed_configs(round_index: int, signal: dict[str, Any]) -> list[dict[str, Any]]:
    tag = f"r{round_index}"; focus = "tail_shape_focus" if signal.get("tail_abs_bias", 0.0) >= signal.get("head_abs_bias", 0.0) else "shape_focus"; out = []
    mlp = [("model", "model", "none", 0.0, focus, "full_target_free", 96, 2, 0.004), ("model", "model", "head_residual", 0.5, focus, "fixed_draining_causal", 128, 2, 0.003)]
    ridge = [("global", "causal_with_h0_shape", "residual", 0.1, False, False), ("model_segment", "causal_structural_interactions", "residual", 10.0, True, False), ("model_segment", "causal_structural_interactions", "direct_shape", 10.0, True, True), ("model", "causal_structural_interactions", "residual", 10.0, False, True)]
    ratio = [("model", "causal_structural_interactions", 1.0, True, False), ("model_segment", "causal_structural_interactions", 10.0, True, True)]
    for i, (head, alpha_scope, cal, strength, loss, feat, width, depth, lr) in enumerate(mlp): out.append({"family":"phase56_mlp","candidate_id":f"p58_{tag}_mlp_{i}","head_scope":head,"alpha_scope":alpha_scope,"calibration_mode":cal,"calibration_strength":strength,"loss_mode":loss,"feature_mode":feat,"width":width,"depth":depth,"learning_rate":lr,"weight_decay":0.002,"max_epochs":300,"patience":60,"stage":"seed","round":round_index})
    for i, (scope, feat, rep, l2, cal, support) in enumerate(ridge): out.append({"family":"ridge","candidate_id":f"p58_{tag}_ridge_{i}","scope":scope,"feature_mode":feat,"representation":rep,"l2":l2,"calibration":cal,"calibration_strength":0.5,"support_aware":support,"support_threshold":0.08,"stage":"seed","round":round_index})
    for i, (scope, feat, l2, cal, support) in enumerate(ratio): out.append({"family":"ratio_ridge","candidate_id":f"p58_{tag}_ratio_{i}","scope":scope,"feature_mode":feat,"l2":l2,"calibration":cal,"calibration_strength":0.5,"support_aware":support,"support_threshold":0.08,"stage":"seed","round":round_index})
    return out

def adaptive_configs(top: list[dict[str, Any]], round_index: int, signal: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for i, parent in enumerate(top[:2]):
        cfg = copy.deepcopy(parent["config"]); cfg.update(stage="adaptive", round=round_index)
        if cfg["family"] == "phase56_mlp": cfg.update(candidate_id=f"p58_r{round_index}_from_{i}_mlp", head_scope="model", alpha_scope="model", width=min(160, int(cfg["width"])+16), depth=2, max_epochs=360, patience=70, loss_mode="tail_shape_focus" if signal.get("tail_abs_bias", 0.0) >= signal.get("head_abs_bias", 0.0) else "shape_focus")
        elif cfg["family"] == "ratio_ridge": cfg.update(candidate_id=f"p58_r{round_index}_from_{i}_ratio", scope="model_segment", l2=float(cfg["l2"])*(.1 if i == 0 else 10.0), support_aware=True, support_threshold=.03 if signal.get("support_gap", 0.0) > .02 else .12)
        else: cfg.update(candidate_id=f"p58_r{round_index}_from_{i}_ridge", scope="model_segment", feature_mode="causal_structural_interactions", l2=float(cfg["l2"])*(.1 if i == 0 else 10.0), support_aware=True)
        out.append(cfg); out.append({"family":"blend","candidate_id":f"p58_r{round_index}_from_{i}_blend","stage":"adaptive","round":round_index,"parent_ids":[parent["config"]["candidate_id"]],"weights":[1.0],"alpha_scope":"model","support_aware":False})
    return out

def evaluate(train, cfg, folds, contract, carry):
    if cfg["family"] in ("phase56_mlp", "ridge"): return P57RUN.evaluate_candidate(train, cfg, folds, contract, carry)
    if cfg["family"] == "ratio_ridge": base_c, base_b, masks_c, masks_b, epochs = ratio_oof(train, cfg, folds)
    elif cfg["family"] == "blend":
        parents = [carry[p] for p in cfg["parent_ids"]]; weights = np.asarray(cfg["weights"], dtype=float); weights /= max(weights.sum(), 1e-12); base_c = sum(float(a)*p["base_calls"] for a,p in zip(weights,parents)); base_b = sum(float(a)*p["base_bytes"] for a,p in zip(weights,parents)); masks_c = np.ones_like(base_c, dtype=bool); masks_b = np.ones_like(base_b, dtype=bool); epochs = [1]
    else: raise ValueError(cfg["family"])
    h0c, h0b = P54.histogram_arrays(train, "h0"); h0metrics = P54.metric_bundle(h0c, h0b, *P54.histogram_arrays(train, "target")); alpha, audits, _ = P54.choose_alphas(train, base_c, base_b, h0metrics, [float(x) for x in contract["search_contract"]["alpha_grid"]]); calls, bytes_ = apply_alpha(train, base_c, base_b, alpha)
    if cfg.get("support_aware", False): calls, bytes_ = P57RUN.support_project(calls, bytes_, masks_c, masks_b)
    overall, models, segments, target = P54.development_audits(train, calls, bytes_); protection = P57RUN.strict_h0(overall) and all(P57RUN.strict_h0(v) for v in models.values()) and all(P57RUN.strict_h0(v) for v in segments.values())
    return {"config":cfg,"base_calls":base_c,"base_bytes":base_b,"calls":calls,"bytes":bytes_,"epochs":epochs,"alpha_map":alpha,"alpha_audits":audits,"overall":overall,"models":models,"segments":segments,"oof_target":bool(target),"oof_protection":bool(protection),"bias":P57RUN.bias_rows(train,calls,bytes_,cfg["candidate_id"],int(cfg["round"])),"sort":(not target,not protection,P57RUN.score(overall["h0_plus_dnn_refined"]),cfg["candidate_id"])}

def fit_ratio_final(train, rows, cfg):
    calls = np.zeros((len(rows),BINS)); bytes_ = np.zeros_like(calls); bundle={"family":"ratio_ridge","scope":cfg["scope"],"groups":{}}
    for group, fit_idx in sorted(P57RUN.groups_for(train,cfg["scope"]).items()):
        fit_rows=[train[i] for i in fit_idx]; hold_idx=[i for i,r in enumerate(rows) if P57RUN.scope_key(r,cfg["scope"])==group]
        if not hold_idx: continue
        hold_rows=[rows[i] for i in hold_idx]; xfit,names=P57RUN.feature_matrix(fit_rows,cfg["feature_mode"]); xhold,_=P57RUN.feature_matrix(hold_rows,cfg["feature_mode"]); fitted,_,_=P57RUN.fit_ridge(xfit,target_logdiff(fit_rows),float(cfg["l2"])); pc,pb=decode_logdiff(hold_rows,P57RUN.ridge_predict(fitted,xhold)); calls[hold_idx]=pc; bytes_[hold_idx]=pb; bundle["groups"][group]={"model":fitted,"feature_names":names}
    return calls,bytes_,bundle

def fit_final(train, rows, cfg, cmap):
    if cfg["family"]=="ratio_ridge": return fit_ratio_final(train,rows,cfg)
    if cfg["family"] in ("phase56_mlp","ridge"): return P57RUN.fit_final_config(train,rows,cfg,cmap)
    if cfg["family"]=="blend":
        weights=np.asarray(cfg["weights"],dtype=float); weights/=max(weights.sum(),1e-12); calls=np.zeros((len(rows),BINS)); bytes_=np.zeros_like(calls); bundles=[]
        for a,pid in zip(weights,cfg["parent_ids"]): pc,pb,b=fit_final(train,rows,cmap[pid],cmap); calls+=float(a)*pc; bytes_+=float(a)*pb; bundles.append(b)
        return calls,bytes_,{"family":"blend","parent_ids":cfg["parent_ids"],"weights":weights.tolist(),"bundles":bundles}
    raise ValueError(cfg["family"])

def prediction_rows(rows,calls,bytes_,method):
    out=[]
    for row,c,b in zip(rows,calls,bytes_):
        value={k:row[k] for k in ("profile_id","split_role","source","segment","source_split","window_id","cutoff_ms","model")}; value["method"]=method
        for i in range(BINS): value[f"predicted_calls_bin_{i:02d}"]=float(c[i]); value[f"predicted_logical_bytes_bin_{i:02d}"]=float(b[i])
        out.append(value)
    return out

def run(expected: str, output: Path):
    preflight=PREFLIGHT.run_checks(expected); contract=load_json(HERE/"experiment.json")
    if output.exists(): raise RuntimeError(f"refuse overwrite: {output}")
    rows=read_csv_gz(repo_root()/contract["pinned_inputs"][1]["path"]); train=[r for r in rows if r["split_role"]=="expanded_train"]; validation=[r for r in rows if r["split_role"]=="expanded_validation"]; folds=P54.fold_map(train); all_results=[]; trace=[]; bias=[]; cmap={}; carry={}; signal={}; rounds=0
    for ri in range(int(contract["search_contract"]["max_rounds"])):
        seeds=seed_configs(ri,signal)
        if len(seeds)!=8: raise RuntimeError({"seed_count":len(seeds)})
        seed_results=[]
        for cfg in seeds:
            value=evaluate(train,cfg,folds,contract,carry); seed_results.append(value); cmap[cfg["candidate_id"]]=cfg; trace.append({"round":ri,"stage":"seed","candidate_id":cfg["candidate_id"],"family":cfg["family"],"oof_target":value["oof_target"],"oof_protection":value["oof_protection"],"oof_score":P57RUN.score(value["overall"]["h0_plus_dnn_refined"]),"calls_histogram_wape":value["overall"]["h0_plus_dnn_refined"]["calls_histogram_wape"],"bytes_histogram_wape":value["overall"]["h0_plus_dnn_refined"]["bytes_histogram_wape"]})
        top=sorted(seed_results,key=lambda x:x["sort"])[:2]; carry.update({x["config"]["candidate_id"]:{"base_calls":x["base_calls"],"base_bytes":x["base_bytes"]} for x in top}); adaptive=adaptive_configs(top,ri,signal); adaptive_results=[]
        for cfg in adaptive:
            value=evaluate(train,cfg,folds,contract,carry); adaptive_results.append(value); cmap[cfg["candidate_id"]]=cfg; trace.append({"round":ri,"stage":"adaptive","candidate_id":cfg["candidate_id"],"family":cfg["family"],"parent_ids":json.dumps(cfg.get("parent_ids",[]),sort_keys=True),"oof_target":value["oof_target"],"oof_protection":value["oof_protection"],"oof_score":P57RUN.score(value["overall"]["h0_plus_dnn_refined"]),"calls_histogram_wape":value["overall"]["h0_plus_dnn_refined"]["calls_histogram_wape"],"bytes_histogram_wape":value["overall"]["h0_plus_dnn_refined"]["bytes_histogram_wape"]})
        if len(adaptive_results)!=4: raise RuntimeError({"adaptive_count":len(adaptive_results)})
        values=seed_results+adaptive_results; all_results.extend(values); bias.extend([b for v in values for b in v["bias"]]); rounds=ri+1; best=sorted(all_results,key=lambda x:x["sort"])[0]; signal=P57RUN.signal_from(values); carry.update({x["config"]["candidate_id"]:{"base_calls":x["base_calls"],"base_bytes":x["base_bytes"]} for x in sorted(values,key=lambda x:x["sort"])[:4]})
        if best["oof_target"] and best["oof_protection"]: break
    if len(all_results)==0 or len(all_results)>36: raise RuntimeError({"candidate_count":len(all_results)})
    selected=sorted(all_results,key=lambda x:x["sort"])[0]; cfg=copy.deepcopy(selected["config"]); cfg["selected_epochs"]=int(np.clip(round(statistics.median(selected["epochs"])),100,int(cfg.get("max_epochs",100))))
    if cfg["family"]=="blend": cfg["parent_configs"]=[copy.deepcopy(cmap[p]) for p in cfg["parent_ids"]]
    cmap.update({c["candidate_id"]:c for c in cfg.get("parent_configs",[])}); val_base_c,val_base_b,final_bundle=fit_final(train,validation,cfg,cmap); val_c,val_b=apply_alpha(validation,val_base_c,val_base_b,selected["alpha_map"]); overall,model_audits,segment_audits,target_met=P54.development_audits(validation,val_c,val_b); output.mkdir(parents=True)
    write_csv(output/"analysis/round_trace.csv",trace); write_csv(output/"analysis/oof_candidate_metrics.csv",[{"round":v["config"]["round"],"stage":v["config"]["stage"],"candidate_id":v["config"]["candidate_id"],"family":v["config"]["family"],"oof_target":v["oof_target"],"oof_protection":v["oof_protection"],"oof_score":P57RUN.score(v["overall"]["h0_plus_dnn_refined"]),"oof_calls_histogram_wape":v["overall"]["h0_plus_dnn_refined"]["calls_histogram_wape"],"oof_bytes_histogram_wape":v["overall"]["h0_plus_dnn_refined"]["bytes_histogram_wape"],"selected":v is selected} for v in all_results]); write_csv(output/"analysis/oof_bin_bias.csv",bias)
    write_json(output/"analysis/diagnostic_summary.json",{"selected_candidate":cfg["candidate_id"],"selected_family":cfg["family"],"oof_signal":signal,"selected_oof":selected["overall"],"validation":overall,"model_validation":model_audits,"segment_validation":segment_audits,"histogram_error_is_target":True,"total_error_is_target":False}); write_json(output/"analysis/oof_selection.json",{"selected_candidate":cfg,"alpha_map":selected["alpha_map"],"oof_overall":selected["overall"],"oof_models":selected["models"],"oof_segments":selected["segments"],"oof_target":selected["oof_target"],"oof_protection":selected["oof_protection"],"candidate_count":len(all_results),"rounds_completed":rounds})
    write_csv(output/"analysis/development_validation_metrics.csv",[{"method":"h0",**overall["h0"],"composite_ratio_to_h0":1.0,"formal_target_gate":False},{"method":"h0_plus_dnn_shape_aware",**overall["h0_plus_dnn_refined"],"composite_ratio_to_h0":overall["composite_ratio"],"formal_target_gate":overall["target_gate"]}]); write_json(output/"analysis/model_validation.json",model_audits); write_json(output/"analysis/segment_validation.json",segment_audits); h0c,h0b=P54.histogram_arrays(validation,"h0"); P54.write_csv_gz(output/"predictions/development_validation_predictions.csv.gz",prediction_rows(validation,h0c,h0b,"h0")+prediction_rows(validation,val_c,val_b,"h0_plus_dnn_shape_aware"))
    checkpoint={"schema_version":"phase58-pd-shape-aware-checkpoint-v1","workflow_commit":expected,"selected_candidate":cfg,"alpha_map":selected["alpha_map"],"bundle":final_bundle,"phase50_blind_accessed":False,"complete_requests_accessed":False}; P54.write_json_gz(output/"checkpoints/pd_shape_aware_iterative_refinement.json.gz",checkpoint); write_json(output/"audit/input_freeze.json",preflight); write_json(output/"audit/search.json",{"candidate_budget":len(all_results),"max_candidate_budget":36,"rounds_completed":rounds,"selected_candidate_id":cfg["candidate_id"],"oof_target":selected["oof_target"],"oof_protection":selected["oof_protection"],"development_target_met":target_met,"validation_opened_once_after_freeze":True,"phase50_blind_accessed":False,"complete_requests_accessed":False,"adaptive_signal_source":"OOF only"}); write_json(output/"audit/environment.json",{**environment_record(),"gpu_used":False,"network_used":False,"raw_accessed":False,"phase50_blind_accessed":False,"complete_requests_accessed":False,"training_used":True})
    summary={"schema_version":"phase58-pd-shape-aware-result-v1","status":"PASS","workflow_commit":expected,"completed_at_utc":utc_now(),"counts":{"profiles":1200,"train_profiles":960,"validation_profiles":240,"models":6,"segments":3,"example_rows":7200,"train_rows":5760,"validation_rows":1440,"candidates":len(all_results),"rounds_completed":rounds,"complete_request_rows_in_git":0},"selected":{"candidate_id":cfg["candidate_id"],"family":cfg["family"],"alpha_map":selected["alpha_map"]},"gates":{"oof_target":selected["oof_target"],"oof_protection":selected["oof_protection"],"development_overall":overall["target_gate"],"development_all_models":all(v["target_guard"] for v in model_audits.values()),"development_all_segments":all(v["target_guard"] for v in segment_audits.values()),"target_met":bool(target_met),"next_phase_permitted":bool(target_met)},"development_validation":overall,"models":model_audits,"segments":segment_audits,"scientific_outcome":"DEVELOPMENT_TARGET_MET" if target_met else "DEVELOPMENT_TARGET_NOT_MET","proved":"bounded OOF shape-aware and multiplicative-bin search with one-shot validation","not_proved":"fresh blind generalization, physical communication time, placement, latency or online scheduling"}
    write_json(output/"summary.json",summary); (output/"README.md").write_text(f"# Phase58：PD shape-aware 迭代优化\n\n状态：`PASS`。完成 {rounds} 轮、{len(all_results)} 个候选；选中 `{cfg['candidate_id']}`；development target={target_met}。\n\nOOF-only search；validation 仅冻结后打开一次；未读取 Phase50 blind、raw 或完整请求。\n",encoding="utf-8"); (output/"logs").mkdir(); (output/"logs/runtime.log").write_text(f"completed={utc_now()} workflow_commit={expected}\nrounds={rounds} candidates={len(all_results)} selected={cfg['candidate_id']} family={cfg['family']}\noof_target={selected['oof_target']} oof_protection={selected['oof_protection']} development_target={target_met}\ngpu=false network=false phase50_blind=false complete_requests=false\n",encoding="utf-8"); (output/"DONE").write_text("PASS\n",encoding="utf-8"); refresh_manifest(output); return summary

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--expected-workflow-commit",required=True); parser.add_argument("--output-dir",type=Path,default=repo_root()/"experiment-results/phase58_pd_shape_aware_iterative_refinement"); args=parser.parse_args(); print(json.dumps(run(args.expected_workflow_commit,args.output_dir.resolve()),ensure_ascii=False,indent=2))
if __name__ == "__main__": main()
