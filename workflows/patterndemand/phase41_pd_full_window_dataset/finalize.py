#!/usr/bin/env python3
"""Finalize the compact Phase41 PASS result tree."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import (  # noqa: E402
    load_json,
    refresh_manifest,
    repo_root,
    utc_now,
    validate_result_tree,
    write_json,
)


def finalize(output: Path) -> dict:
    state = load_json(output / "audit/runtime_state.json")
    contract = load_json(output / "contracts/experiment.json")
    dataset = load_json(output / "audit/dataset_build.json")
    raw = load_json(output / "audit/raw_manifest.json")
    if not all(state["gates"].values()) or not all(state["checks"].values()):
        raise RuntimeError({"phase41_runtime_checks": state})
    if dataset.get("blind_target_generated") is not False:
        raise RuntimeError("Phase41 must not generate blind targets")
    counts = state["counts"]
    summary = {
        "schema_version": "phase41-pd-full-window-dataset-result-v1",
        "status": "PASS",
        "completed_at_utc": utc_now(),
        "workflow_commit": state["workflow_commit"],
        "objective": contract["objective"],
        "gates": state["gates"],
        "counts": counts,
        "gpu_sentinel": {
            "requests": counts["gpu_sentinel_requests"],
            "waves": counts["gpu_sentinel_waves"],
            "gpu_logical_chunks": counts["gpu_logical_chunks"],
            "teacher_logical_chunks": counts["teacher_logical_chunks"],
            "exact_requests": counts["gpu_exact_requests"],
            "calls_error": 0,
            "logical_bytes_error": 0,
            "histogram_l1": 0.0,
        },
        "dataset": {
            "development_profiles": counts["development_profiles"],
            "development_full_requests": counts["development_full_requests"],
            "development_target_rows": counts["development_target_rows"],
            "blind_feature_rows": counts["blind_feature_rows"],
            "blind_target_rows": counts["blind_target_rows"],
            "h0_plus_dnn_residual_ready": True,
        },
        "raw": {
            "external": True,
            "files": raw["file_count"],
            "bytes": raw["bytes"],
            "committed_to_git": False,
        },
        "training_performed": False,
        "checkpoint_loaded": False,
        "blind_evaluation_performed": False,
        "other_models_evaluated": False,
        "physical_curve_measured": False,
        "evidence_boundary": {
            "proved": "bounded-wave Qwen3 pure-PD teacher exactness on 63/64/65/129 boundaries and three real full windows, followed by deterministic 94-profile development data generation",
            "not_proved": "trained predictor quality, blind generalization, the other five models, physical Mooncake time, placement or online scheduling",
            "blind_state": "12 fresh target-free feature/H0 rows frozen; no blind complete request list or Hfull target exported",
        },
    }
    write_json(output / "summary.json", summary)
    (output / "README.md").write_text(
        "# Phase41：纯PD完整窗口teacher与开发数据\n\n"
        "最终状态：`PASS`。本阶段先冻结`最多64请求/wave`的有界fixed-draining协议："
        "窗口内保持原始顺序，wave内原子放行，前一wave完全返回后才提交后一wave。\n\n"
        f"GPU sentinel覆盖`{counts['gpu_sentinel_requests']}`个请求、"
        f"`{counts['gpu_sentinel_waves']}`个wave，包括63/64/65/129边界和三个真实完整窗口。"
        f"真实sender记录与CPU teacher逐请求精确匹配`{counts['gpu_exact_requests']}/"
        f"{counts['gpu_sentinel_requests']}`，calls、logical bytes和12-bin直方图误差均为0。\n\n"
        "GPU门通过后，才生成94个Qwen3-8B开发画像的Hfull标签、32请求H0和逐bin residual，"
        f"共使用`{counts['development_full_requests']}`个完整teacher请求。"
        "另冻结12个全新盲测画像的低维feature与H0；盲测完整请求没有进入跨环境bundle，"
        "Hfull target行数严格为0。\n\n"
        "本阶段没有训练DNN、没有加载checkpoint、没有测试其他五个模型，也没有测物理RDMA时间或做placement。"
        "GPU profiler JSONL、完整server日志和包含开发完整请求的transfer bundle均保存在Git外。\n",
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
        default=repo_root() / "experiment-results/phase41_pd_full_window_dataset",
    )
    args = parser.parse_args()
    print(json.dumps(finalize(args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
