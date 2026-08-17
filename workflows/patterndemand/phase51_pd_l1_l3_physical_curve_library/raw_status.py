#!/usr/bin/env python3
"""Report exactly which Phase51 shards need initial or variance repeats."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from contracts import load_json
from measurement import validate_raw
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--topology-plan",type=Path,required=True);p.add_argument("--raw-dir",type=Path,required=True);a=p.parse_args();result=validate_raw(load_json(a.topology_plan.expanduser().resolve()),a.raw_dir,require_complete=False);result.pop("records",None);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
