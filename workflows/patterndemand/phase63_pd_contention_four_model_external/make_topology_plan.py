#!/usr/bin/env python3
"""Freeze an external inventory into the exact Phase63 plan."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from contracts import expand_plan, file_sha, load_json

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = args.inventory.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"refuse overwrite: {output}")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    plan = expand_plan(load_json(inventory), file_sha(inventory), datetime.now(timezone.utc).isoformat(), head)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "output": str(output),
        "workflow_commit": head,
        "plan_sha256": plan["plan_sha256"],
        "measurements": len(plan["measurements"]),
        "placement_summary": plan["placement_summary"],
        "resource_contract": {
            "world_size_per_shard": 3,
            "maximum_simultaneous_nodes_per_shard": 2,
            "four_node_allocation_required": False,
        },
        "warning": "freeze before raw; any endpoint edit invalidates the entire attempt",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
