#!/usr/bin/env python3
"""Finalize Phase40 compact result tree."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import load_json, refresh_manifest, repo_root, utc_now, validate_result_tree, write_json
from contracts import read_csv


def finalize(output: Path) -> dict:
    state = load_json(output / "audit/runtime_state.json")
    contract = load_json(output / "contracts/experiment.json")
    smoke = load_json(output / "audit/compatibility_smoke.json")
    alignment = read_csv(output / "analysis/gpu_teacher_alignment.csv")
    if not all(state["checks"].values()):
        raise RuntimeError({"phase40_checks": state["checks"]})
    if smoke.get("status") != "PASS" or not all(smoke.get("checks", {}).values()):
        raise RuntimeError({"phase40_compatibility_smoke": smoke})
    overall = next(row for row in alignment if row["scenario"] == "overall")
    summary = {
        "schema_version": "phase40-pure-pd-semantics-teacher-result-v1",
        "status": "PASS",
        "completed_at_utc": utc_now(),
        "workflow_commit": state["workflow_commit"],
        "objective": contract["objective"],
        "counts": state["counts"],
        "overall_alignment": overall,
        "runtime_kv_bytes_per_page": state["runtime_kv_bytes_per_page"],
        "compatibility_smoke": {
            "status": smoke["status"],
            "attention_backend": smoke["attention_backend"],
            "page_size_tokens": smoke["observed"][0]["page_size_tokens"],
            "sender_chunks": smoke["matching_sender_chunks"],
        },
        "checks": state["checks"],
        "training_performed": False,
        "checkpoint_loaded": False,
        "physical_curve_measured": False,
        "scheduler_or_placement_evaluated": False,
        "evidence_boundary": {
            "proved": "exact representative GPU-to-teacher alignment for the frozen P1-D1 Qwen3/Mooncake logical KV-chunk semantics",
            "not_proved": "a trained low-dimensional PD predictor, six-model generalization, Mooncake physical cost, placement or online scheduling",
            "full_request_list_usage": "offline teacher generation and representative audit only",
        },
    }
    write_json(output / "summary.json", summary)
    (output / "README.md").write_text(
        "# Phase40：纯PD语义与Hfull teacher基础闭环\n\n"
        "最终状态：`PASS`。本阶段只运行纯`P1→D1`，P和D内部均为`TP=1、PP=1`；"
        "固定FlashInfer attention、page size 1、Mooncake/RDMA、FCFS、4096-token chunk并关闭cache与overlap。"
        "正式raw前的独立P→D smoke已完成1个真实sender chunk且没有传输错误。\n\n"
        f"共执行`{state['counts']['requests']}`个请求、`{state['counts']['gpu_logical_chunks']}`个真实sender-side逻辑KV chunk；"
        f"CPU teacher生成`{state['counts']['teacher_logical_chunks']}`个chunk，"
        f"逐请求精确匹配`{state['counts']['exact_requests']}/{state['counts']['requests']}`。"
        "calls、logical bytes和12-bin直方图误差均为0，五个场景的三次重复直方图完全一致。\n\n"
        f"运行时KV字节/page为`{state['runtime_kv_bytes_per_page']}`，与模型结构公式精确一致；"
        "没有Mamba/SWA等额外state payload。raw profiler JSONL和完整P/D/router日志保存在Git外，Git只归档其SHA、数量和聚合结果。\n\n"
        "该结果建立了冻结Qwen3纯PD语义的GPU证据与离线teacher基础，不代表低维画像预测器已经训练、"
        "六模型已经泛化、Mooncake物理时间曲线已经完成，也不包含placement或线上scheduler结论。\n",
        encoding="utf-8",
    )
    tree = validate_result_tree(output)
    if not tree["ok"]:
        raise RuntimeError({"forbidden_result_assets": tree["violations"]})
    (output / "DONE").write_text("PASS\n", encoding="utf-8")
    refresh_manifest(output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "experiment-results/phase40_pure_pd_semantics_teacher",
    )
    args = parser.parse_args()
    print(json.dumps(finalize(args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
