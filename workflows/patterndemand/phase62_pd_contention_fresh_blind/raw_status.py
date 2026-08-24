#!/usr/bin/env python3
"""Report missing or variance-triggered Phase62 repeats."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from contracts import load_json
from measurement import validate_raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology-plan", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    args = parser.parse_args()
    result = validate_raw(load_json(args.topology_plan.expanduser().resolve()), args.raw_dir, require_complete=False)
    result.pop("records", None)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
