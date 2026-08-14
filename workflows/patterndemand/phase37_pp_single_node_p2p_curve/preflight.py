#!/usr/bin/env python3
"""Phase37环境、生产路径和GPU拓扑审计。"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import ensure_external_raw_dir, load_json, repo_root, require_clean_before_run, require_expected_head, verify_pinned_inputs


def normalize_link(label: str) -> str | None:
    if label.startswith("NV") and label[2:].isdigit():
        return f"NVLINK_{label}"
    if label in {"PIX", "PXB", "PHB", "NODE", "SYS"}:
        return label
    return None


def parse_topology(text: str) -> dict:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    header = next((line.split() for line in lines if line.split() and line.split()[0] == "GPU0"), None)
    if not header:
        raise RuntimeError("无法解析nvidia-smi topo -m表头")
    gpu_columns = [token for token in header if re.fullmatch(r"GPU\d+", token)]
    matrix = {}
    for line in lines:
        tokens = line.split()
        if not tokens or not re.fullmatch(r"GPU\d+", tokens[0]) or len(tokens) < len(gpu_columns) + 1:
            continue
        matrix[tokens[0]] = dict(zip(gpu_columns, tokens[1 : 1 + len(gpu_columns)]))
    pairs = {}
    for left_index, left in enumerate(gpu_columns):
        for right in gpu_columns[left_index + 1 :]:
            raw = matrix.get(left, {}).get(right)
            category = normalize_link(raw or "")
            if category:
                pairs.setdefault(category, []).append({"gpus": [int(left[3:]), int(right[3:])], "raw_link": raw})
    for values in pairs.values():
        values.sort(key=lambda row: tuple(row["gpus"]))
    return {"gpu_columns": gpu_columns, "pairs_by_category": pairs, "raw": text}


def source_semantics_audit() -> dict:
    root = repo_root()
    group_source = (root / "python/sglang/srt/distributed/parallel_state.py").read_text(encoding="utf-8")
    scheduler_source = (root / "python/sglang/srt/managers/scheduler_pp_mixin.py").read_text(encoding="utf-8")
    profiler_source = (root / "python/sglang/srt/distributed/pp_comm_profile.py").read_text(encoding="utf-8")
    checks = {
        "production_send_tensor_dict_exists": "def send_tensor_dict(" in group_source,
        "production_async_uses_isend": "send_func = torch.distributed.isend if async_send else torch.distributed.send" in group_source,
        "gpu_tensor_uses_device_group": "comm_group = metadata_group if tensor.is_cpu else group" in group_source,
        "scheduler_default_async_send": "async_send: bool = True" in scheduler_source,
        "scheduler_calls_send_tensor_dict": "self.pp_group.send_tensor_dict(" in scheduler_source,
        "histogram_sender_only": "Only tensor sends are counted" in profiler_source,
        "histogram_raw_op": '"raw_op": "p2p_send_tensor"' in profiler_source,
    }
    if not all(checks.values()):
        raise RuntimeError(f"SGLang生产PP通信语义审计失败：{checks}")
    return checks


def run_checks(expected_commit: str, output_dir: Path, raw_dir: Path) -> dict:
    contract = load_json(HERE / "experiment.json")
    expected_output = (repo_root() / contract["result_dir"]).resolve()
    if output_dir != expected_output:
        raise RuntimeError(f"Phase37正式结果目录不可修改：expected={expected_output}, actual={output_dir}")
    head = require_expected_head(expected_commit)
    require_clean_before_run()
    if output_dir.exists():
        raise RuntimeError(f"正式结果目录已存在，拒绝覆盖：{output_dir}")
    raw_dir = ensure_external_raw_dir(raw_dir)
    pinned = verify_pinned_inputs(contract)
    semantics = source_semantics_audit()
    if shutil.which("nvidia-smi") is None or shutil.which("torchrun") is None:
        raise RuntimeError("缺少nvidia-smi或torchrun")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        raise RuntimeError("运行总控workflow前必须unset CUDA_VISIBLE_DEVICES；workflow会为每个物理GPU对单独设置")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("当前Python环境缺少torch") from error
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Phase37至少需要两张可用CUDA GPU")
    topo_text = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
    topology = parse_topology(topo_text)
    selected = {
        category: values[0]
        for category, values in topology["pairs_by_category"].items()
        if category in contract["topology_categories"]
        or (category.startswith("NVLINK_") and "NVLINK_*" in contract["topology_categories"])
    }
    if not selected:
        raise RuntimeError("未发现可测的单机GPU拓扑对")
    return {
        "status": "PASS",
        "workflow_commit": head,
        "output_dir": str(output_dir.relative_to(repo_root())),
        "external_raw_dir": str(raw_dir),
        "pinned_inputs": pinned,
        "source_semantics": semantics,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "topology": topology,
        "default_selected_pairs": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-workflow-commit", required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase37_pp_single_node_p2p_curve")
    args = parser.parse_args()
    result = run_checks(args.expected_workflow_commit, args.output_dir.resolve(), args.raw_dir)
    printable = dict(result); printable["topology"] = {key: value for key, value in result["topology"].items() if key != "raw"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
