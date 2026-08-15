#!/usr/bin/env python3
"""执行Agent提交前检查暂存区仅含对应Phase结果目录。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_json, repo_root, validate_staged_allowlist, verify_result_manifest


PHASES = {
    "phase36": "experiment-results/phase36_cross_environment_replay/",
    "phase37": "experiment-results/phase37_pp_single_node_p2p_curve/",
    "phase38": "experiment-results/phase38_pp_physical_curve_cost_recompute/",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=sorted(PHASES), required=True)
    args = parser.parse_args()
    prefix = PHASES[args.phase]
    output = repo_root() / prefix
    staging = validate_staged_allowlist(prefix)
    blocked = (output / "BLOCKED.json").is_file()
    result = {"staging": staging, "blocked": blocked}
    if not blocked:
        result["result_manifest"] = verify_result_manifest(output)
        summary = load_json(output / "summary.json")
        result["summary_status"] = summary["status"]
        if not result["result_manifest"]["ok"]:
            raise RuntimeError(result)
    elif not (output / "manifest.sha256").is_file():
        raise RuntimeError("BLOCKED结果也必须生成manifest.sha256")
    if not staging["ok"]:
        raise RuntimeError(result)
    print(json.dumps({"status": "PASS", **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
