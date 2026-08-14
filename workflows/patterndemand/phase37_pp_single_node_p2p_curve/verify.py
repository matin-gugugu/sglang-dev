#!/usr/bin/env python3
"""验证Phase37正式紧凑结果。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import load_json, repo_root, verify_result_manifest


ACCEPTED = {"PASS", "PASS_WITH_RUNTIME_VARIANCE", "PASS_WITH_LIMITED_TOPOLOGY", "PASS_WITH_RUNTIME_VARIANCE_AND_LIMITED_TOPOLOGY"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase37_pp_single_node_p2p_curve")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    audit = verify_result_manifest(output)
    summary = load_json(output / "summary.json")
    done = (output / "DONE").read_text(encoding="utf-8").strip()
    raw_manifest = load_json(output / "audit/RAW_ASSET_MANIFEST.json")
    if not audit["ok"] or summary["status"] not in ACCEPTED or done != summary["status"] or raw_manifest["raw_committed_to_git"]:
        raise RuntimeError({"manifest": audit, "summary_status": summary["status"], "done": done, "raw_committed": raw_manifest["raw_committed_to_git"]})
    print(json.dumps({"status": summary["status"], "output": str(output), "manifest": audit, "external_raw_files": len(raw_manifest["files"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
