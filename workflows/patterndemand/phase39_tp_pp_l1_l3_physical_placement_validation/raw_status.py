#!/usr/bin/env python3
"""Report missing/extra-repeat requirements without writing formal results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contracts import contract, load_json, validate_plan
from measurement import validate_raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology-plan", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    args = parser.parse_args()
    spec = contract()
    plan = load_json(args.topology_plan.expanduser().resolve())
    validate_plan(plan, spec)
    audit = validate_raw(plan, args.raw_dir, spec)
    printable = {key: value for key, value in audit.items() if key not in {"records", "repeat_values", "quality_rows"}}
    printable["quality_rows"] = len(audit["quality_rows"])
    printable["status"] = "READY_FOR_CPU_ANALYSIS" if audit["complete"] else "MORE_MEASUREMENT_REQUIRED"
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
