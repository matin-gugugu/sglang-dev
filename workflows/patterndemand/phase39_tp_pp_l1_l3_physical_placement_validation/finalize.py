#!/usr/bin/env python3
"""Finalize Phase39 compact results and evidence boundaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import load_json, refresh_manifest, repo_root, utc_now, validate_result_tree, write_json


def finalize(output: Path) -> dict:
    state = load_json(output / "audit/runtime_state.json")
    contract = load_json(output / "contracts/experiment.json")
    if not all(state["checks"].values()):
        raise RuntimeError({"phase39_checks": state["checks"]})
    runtime_variance = bool(state["final_high_runtime_variance"])
    placement_variance = bool(state["high_cross_replica_spread"])
    if runtime_variance and placement_variance:
        status = "PASS_WITH_RUNTIME_AND_PLACEMENT_VARIANCE"
    elif runtime_variance:
        status = "PASS_WITH_RUNTIME_VARIANCE"
    elif placement_variance:
        status = "PASS_WITH_PLACEMENT_VARIANCE"
    else:
        status = "PASS"
    overall_decision = next(row for row in state["decision_headline"] if row["slice_type"] == "overall")
    summary = {
        "schema_version": "phase39-tp-pp-l1-l3-physical-placement-result-v1",
        "status": status,
        "completed_at_utc": utc_now(),
        "workflow_commit": state["workflow_commit"],
        "objective": contract["objective"],
        "counts": state["counts"],
        "cost_headline": state["cost_headline"],
        "placement_decision_headline": state["decision_headline"],
        "overall_top1_agreement": overall_decision["top1_agreement"],
        "overall_mean_teacher_regret": overall_decision["mean_teacher_regret"],
        "overall_p95_teacher_regret": overall_decision["p95_teacher_regret"],
        "diagnostic_cost_wape_reference": contract["diagnostic_cost_wape_reference"],
        "diagnostic_reference_is_pass_fail_gate": False,
        "histogram_invariance": state["histogram_invariance"],
        "final_high_runtime_variance": state["final_high_runtime_variance"],
        "high_cross_replica_spread": state["high_cross_replica_spread"],
        "checks": state["checks"],
        "training_performed": False,
        "checkpoint_loaded": False,
        "prediction_recomputation_performed": False,
        "evidence_boundary": {
            "target_state": "Phase34D target was already open before Phase39",
            "evidence_kind": "repeated engineering physical-cost and communication-only placement validation",
            "physical_scope": "only the exact frozen L1/L2/L3 placements and TP/PP primitives in the topology plan",
            "excluded": "compute, memory feasibility, queueing, availability, metadata, scheduler execution and communication-compute overlap",
            "parallel_configuration": "fixed input; never selected",
        },
    }
    write_json(output / "summary.json", summary)
    cost_lines = "\n".join(
        f"- `{row['parallelism']}/{row['topology_level']}`：total WAPE `{float(row['cost_wape']):.4%}`，MAPE `{float(row['cost_mape']):.4%}`，bias `{float(row['signed_bias']):.4%}`。"
        for row in state["cost_headline"]
    )
    decision_lines = "\n".join(
        f"- `{row['slice_type']}={row['slice_value']}`：top1 `{float(row['top1_agreement']):.4%}`，mean regret `{float(row['mean_teacher_regret']):.4%}`，P95 regret `{float(row['p95_teacher_regret']):.4%}`。"
        for row in state["decision_headline"]
    )
    (output / "README.md").write_text(
        "# Phase39：TP/PP L1–L3物理曲线库与placement验证\n\n"
        f"最终状态：`{status}`。本阶段完成{state['counts']['measurement_shards']}个GPU通信measurement shard、"
        f"{state['counts']['physical_curves']}条物理曲线；没有加载模型、checkpoint或重新生成预测。raw逐次样本保存在Git外。\n\n"
        "## 物理cost\n\n" + cost_lines + "\n\n"
        "## communication-only placement\n\n" + decision_lines + "\n\n"
        "TP/PP size与policy始终是输入，决策器只在L1/L2/L3之间选择。该排名不包含计算、显存、排队、资源可用性、metadata或通信计算重叠，不能直接声称为完整线上scheduler收益。\n\n"
        f"Phase34冻结直方图指标复现最大绝对差为`{state['histogram_invariance']['max_absolute_difference']:.3e}`。"
        "Phase34D target已打开，因此本阶段是重复工程证据，不是新盲测。\n",
        encoding="utf-8",
    )
    tree = validate_result_tree(output)
    if not tree["ok"]:
        raise RuntimeError({"forbidden_result_assets": tree["violations"]})
    (output / "DONE").write_text(status + "\n", encoding="utf-8")
    refresh_manifest(output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase39_tp_pp_l1_l3_physical_placement_validation")
    args = parser.parse_args()
    print(json.dumps(finalize(args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
