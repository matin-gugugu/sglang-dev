#!/usr/bin/env python3
"""Expand a pre-measurement topology inventory into the frozen 24-shard plan."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess

from contracts import expand_plan, file_sha, load_json

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory_path = args.inventory.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"拒绝覆盖已有topology plan：{output}")
    plan = expand_plan(
        load_json(inventory_path),
        file_sha(inventory_path),
        datetime.now(timezone.utc).isoformat(),
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "output": str(output),
        "plan_sha256": plan["plan_sha256"],
        "measurements": len(plan["measurements"]),
        "warning": "freeze this file before the first benchmark; any edit requires discarding all raw records and starting a new run",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
