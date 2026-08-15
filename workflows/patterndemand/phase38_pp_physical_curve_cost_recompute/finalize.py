#!/usr/bin/env python3
"""Phase38结果收口、证据边界与manifest生成。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import load_json, refresh_manifest, repo_root, utc_now, validate_result_tree, write_json


def finalize(output_dir: Path) -> dict:
    state = load_json(output_dir / "audit/runtime_state.json")
    contract = load_json(output_dir / "contracts/experiment.json")
    checks = state["checks"]
    status = "PASS" if all(checks.values()) else "FAIL"
    headline = state["headline"]
    any_above_reference = any(row["above_5pct_diagnostic_reference"] for row in headline)
    followup_signal = (
        "NEW_DEVELOPMENT_PROTOCOL_REVIEW_RECOMMENDED"
        if any_above_reference
        else "NO_OVERALL_TOTAL_PHYSICAL_COST_WAPE_TRIGGER_AT_5PCT_REFERENCE"
    )
    summary = {
        "schema_version": "phase38-pp-physical-curve-cost-recompute-result-v1",
        "status": status,
        "completed_at_utc": utc_now(),
        "workflow_commit": state["workflow_commit"],
        "objective": contract["objective"],
        "phase37_result_commit": state["phase37"]["result_commit"],
        "phase37_status": state["phase37"]["status"],
        "phase37_topology_categories": state["phase37"]["topology_categories"],
        "counts": state["counts"],
        "physical_cost_headline": headline,
        "diagnostic_overall_total_cost_wape_reference": contract["diagnostic_overall_total_cost_wape_reference"],
        "diagnostic_reference_is_pass_fail_gate": False,
        "followup_signal": followup_signal,
        "histogram_invariance": state["histogram_invariance"],
        "checks": checks,
        "training_performed": False,
        "checkpoint_loaded": False,
        "prediction_recomputation_performed": False,
        "evidence_boundary": {
            "target_state": "Phase34D target was already opened before Phase38",
            "evidence_kind": "repeated engineering cost recomputation, not a new blind confirmation",
            "physical_scope": "only Phase37 accepted single-node tensor-only P2P topology categories",
            "excluded": "metadata, allocation, scheduler, overlap, multi-node L2/L3, compute, memory and availability",
            "retraining": "none; any follow-up retraining requires a new development and confirmation protocol",
        },
    }
    write_json(output_dir / "summary.json", summary)
    headline_lines = "\n".join(
        f"- `{row['curve_id']}` / `{row['topology_category']}`：total cost WAPE "
        f"`{float(row['cost_wape']):.4%}`，MAPE `{float(row['cost_mape']):.4%}`，"
        f"signed bias `{float(row['signed_bias']):.4%}`。"
        for row in headline
    )
    (output_dir / "README.md").write_text(
        "# Phase38：Phase34冻结PP直方图 × Phase37物理P2P曲线\n\n"
        f"最终状态：`{status}`。本阶段没有使用GPU、没有加载checkpoint、没有重新生成预测、没有训练。"
        f"它从Phase37 result commit `{state['phase37']['result_commit']}`冻结了"
        f"{state['counts']['curves']}条已验收单机物理P2P曲线。\n\n"
        "## 物理cost结果\n\n"
        f"{headline_lines}\n\n"
        "5%是沿用的overall total cost WAPE诊断参考线，不是Phase38结果完整性PASS/FAIL线。"
        f"本次后续信号为`{followup_signal}`；它只决定是否值得设计新的开发协议，不授权在Phase34D已打开target上重训后声称新盲测。\n\n"
        "## 不变项与证据边界\n\n"
        f"Phase34 PP冻结calls/bytes/TV/EMD共{state['counts']['frozen_histogram_metric_rows']}个正式slice已重算，"
        f"与Phase34D归档值的最大绝对差为`{state['histogram_invariance']['max_absolute_difference']:.3e}`。"
        "Phase38只替换cost curve，因此不能被解释成新预测器或新精度盲测。\n\n"
        "物理标签仅适用于Phase37实测的单机、tensor-only、sender-counted P2P曲线。"
        "CPU metadata、tensor allocation、scheduler、通信计算重叠、多机L2/L3、计算、显存与资源可用性均未包含。"
        "Phase35 PP L1 proxy只保留在对照表中，未冒充物理数据。\n",
        encoding="utf-8",
    )
    tree = validate_result_tree(output_dir)
    if not tree["ok"]:
        raise RuntimeError(f"结果树含禁止资产：{tree['violations']}")
    (output_dir / "DONE").write_text(status + "\n", encoding="utf-8")
    refresh_manifest(output_dir)
    if status != "PASS":
        raise RuntimeError(f"Phase38完整性检查失败：{checks}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "experiment-results/phase38_pp_physical_curve_cost_recompute",
    )
    args = parser.parse_args()
    print(json.dumps(finalize(args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
