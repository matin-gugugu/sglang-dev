#!/usr/bin/env python3
"""Freeze target-free Phase49 windows disjoint from every earlier PD cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path

import pandas as pd


SEED="phase49-pd-six-model-fresh-blind-v1-20260818"; HISTORY_MS=300_000
SEGMENTS=("burstgpt_1","burstgpt_2","burstgpt_3")
PRIOR_SELECTIONS=(
    "experiment-results/phase27a_pp_feature_and_holdout_contract/selection/selected_windows.csv",
    "experiment-results/phase28a_second_confirmation_contract/selection/selected_windows.csv",
    "experiment-results/phase30a_tp_structured_event_contract/selection/selected_windows.csv",
    "experiment-results/phase31a_known_model_convergence_contract/selection/selected_windows.csv",
    "experiment-results/phase32a_expanded_search_contract/selection/selected_windows.csv",
    "experiment-results/phase33a_fresh_data_contract/selection/selected_windows.csv",
    "experiment-results/phase34a_six_model_contract/selection/selected_windows.csv",
    "workflows/patterndemand/phase41_pd_full_window_dataset/selection/blind_windows.csv",
    "workflows/patterndemand/phase44_pd_expanded_protected_training/selection/expanded_windows.csv",
    "workflows/patterndemand/phase45_pd_fresh_blind_prediction_freeze/selection/fresh_blind_windows.csv",
)
SOURCE_COLUMNS=("window_id","source","segment","split","cutoff_ms","history_seconds","history_count")


def digest(value:str)->str: return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def load_prior(root:Path)->dict[str,list[int]]:
    prior={segment:[] for segment in SEGMENTS}
    for relative in PRIOR_SELECTIONS:
        with (root/relative).open(newline="",encoding="utf-8") as source:
            for row in csv.DictReader(source):
                if row["segment"] in prior: prior[row["segment"]].append(int(row["cutoff_ms"]))
    return prior


def select(root:Path)->list[dict[str,object]]:
    windows=pd.read_csv(root/"experiment-results/phase15_trace_data/windows.csv.gz",usecols=list(SOURCE_COLUMNS)); prior=load_prior(root); selected=[]
    for segment in SEGMENTS:
        frame=windows[(windows["segment"]==segment)&(windows["history_count"]>=32)].copy()
        frame=frame[[not any(abs(int(value)-old)<HISTORY_MS for old in prior[segment]) for value in frame["cutoff_ms"]]].copy()
        frame=frame.sort_values(["history_count","window_id"],kind="stable").reset_index(drop=True)
        frame["request_count_stratum"]=[min(9,index*10//len(frame)) for index in range(len(frame))]
        segment_rows=[]
        for stratum in range(10):
            pool=frame[frame["request_count_stratum"]==stratum].copy(); pool["selection_order_sha256"]=[digest(str(value)) for value in pool["window_id"]]; chosen=[]
            for _,row in pool.sort_values("selection_order_sha256",kind="stable").iterrows():
                cutoff=int(row["cutoff_ms"])
                if all(abs(cutoff-int(old["cutoff_ms"]))>=HISTORY_MS for old in segment_rows+chosen): chosen.append(row)
                if len(chosen)==10: break
            if len(chosen)!=10: raise RuntimeError(f"insufficient Phase49 windows: {segment}/{stratum}/{len(chosen)}")
            for row in chosen:
                segment_rows.append({"window_id":str(row["window_id"]),"source":str(row["source"]),"segment":segment,"source_split":str(row["split"]),"cutoff_ms":int(row["cutoff_ms"]),"history_seconds":int(row["history_seconds"]),"history_count":int(row["history_count"]),"request_count_stratum":stratum,"selection_order_sha256":str(row["selection_order_sha256"]),"role":"six_model_fresh_blind"})
        segment_rows.sort(key=lambda row:(int(row["cutoff_ms"]),str(row["window_id"])))
        for index,row in enumerate(segment_rows,1): row["profile_id"]=f"phase49_{segment}_fresh_blind_{index:03d}"
        selected.extend(segment_rows)
    selected.sort(key=lambda row:(str(row["segment"]),int(row["cutoff_ms"])))
    if len(selected)!=300 or Counter(row["segment"] for row in selected)!=Counter({segment:100 for segment in SEGMENTS}): raise RuntimeError("Phase49 selection total differs")
    return selected


def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args(); root=Path(__file__).resolve().parents[3]; rows=select(root); args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",newline="",encoding="utf-8") as output:
        writer=csv.DictWriter(output,fieldnames=list(rows[0]),lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    print({"rows":len(rows),"requests":sum(int(row["history_count"]) for row in rows),"output":str(args.output)})


if __name__=="__main__": main()
