#!/usr/bin/env python3
"""将无法绕过的环境阻塞保存为可回传的紧凑证据。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import repo_root, write_blocked


OUTPUTS = {
    "phase36": repo_root() / "experiment-results/phase36_cross_environment_replay",
    "phase37": repo_root() / "experiment-results/phase37_pp_single_node_p2p_curve",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=sorted(OUTPUTS), required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--evidence-json", default="{}", help="JSON对象，不能包含密钥或大体积raw")
    args = parser.parse_args()
    evidence = json.loads(args.evidence_json)
    if not isinstance(evidence, dict):
        raise RuntimeError("--evidence-json必须是JSON对象")
    output = OUTPUTS[args.phase]
    if output.exists():
        raise RuntimeError(f"结果目录已存在，拒绝覆盖：{output}")
    write_blocked(output, args.phase.upper(), args.reason, evidence)
    print(json.dumps({"status": "BLOCKED_RECORDED", "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
