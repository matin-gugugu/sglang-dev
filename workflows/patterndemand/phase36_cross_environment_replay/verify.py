#!/usr/bin/env python3
"""验证Phase36正式结果。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import load_json, repo_root, verify_result_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase36_cross_environment_replay")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    audit = verify_result_manifest(output)
    summary = load_json(output / "summary.json")
    done = (output / "DONE").read_text(encoding="utf-8").strip()
    if not audit["ok"] or summary["status"] != "PASS" or done != "PASS":
        raise RuntimeError({"manifest": audit, "summary_status": summary["status"], "done": done})
    print(json.dumps({"status": "PASS", "output": str(output), "manifest": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
