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
    "phase38": repo_root() / "experiment-results/phase38_pp_physical_curve_cost_recompute",
    "phase39": repo_root() / "experiment-results/phase39_tp_pp_l1_l3_physical_placement_validation",
    "phase40": repo_root() / "experiment-results/phase40_pure_pd_semantics_teacher",
    "phase41": repo_root() / "experiment-results/phase41_pd_full_window_dataset",
    "phase42": repo_root() / "experiment-results/phase42_pd_residual_training",
    "phase43": repo_root() / "experiment-results/phase43_pd_blind_evaluation",
    "phase44": repo_root() / "experiment-results/phase44_pd_expanded_protected_training",
    "phase45": repo_root() / "experiment-results/phase45_pd_fresh_blind_prediction_freeze",
    "phase46": repo_root() / "experiment-results/phase46_pd_fresh_blind_evaluation",
    "phase47": repo_root() / "experiment-results/phase47_pd_five_model_teacher_validation",
    "phase48": repo_root() / "experiment-results/phase48_pd_six_model_expanded_training",
    "phase49": repo_root() / "experiment-results/phase49_pd_six_model_blind_prediction_freeze",
    "phase50": repo_root() / "experiment-results/phase50_pd_six_model_blind_evaluation",
    "phase51": repo_root() / "experiment-results/phase51_pd_l1_l3_physical_curve_library",
    "phase52": repo_root() / "experiment-results/phase52_pd_physical_cost_placement_validation",
    "phase53": repo_root() / "experiment-results/phase53_tp_pp_pd_conclusion_freeze",
    "phase60": repo_root() / "experiment-results/phase60_pd_multi_endpoint_composability",
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
