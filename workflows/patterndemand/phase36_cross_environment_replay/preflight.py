#!/usr/bin/env python3
"""Phase36运行前审计。"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import load_json, repo_root, require_clean_before_run, require_expected_head, verify_pinned_inputs


def run_checks(expected_commit: str, output_dir: Path) -> dict:
    contract = load_json(HERE / "experiment.json")
    expected_output = (repo_root() / contract["result_dir"]).resolve()
    if output_dir != expected_output:
        raise RuntimeError(f"Phase36正式结果目录不可修改：expected={expected_output}, actual={output_dir}")
    head = require_expected_head(expected_commit)
    require_clean_before_run()
    if output_dir.exists():
        raise RuntimeError(f"正式结果目录已存在，拒绝覆盖：{output_dir}")
    pinned = verify_pinned_inputs(contract)
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("缺少nvidia-smi")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("当前Python环境缺少torch") from error
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("Phase36至少需要一张可用CUDA GPU")
    return {
        "status": "PASS",
        "workflow_commit": head,
        "output_dir": str(output_dir.relative_to(repo_root())),
        "pinned_inputs": pinned,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpu0": torch.cuda.get_device_name(0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-workflow-commit", required=True)
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase36_cross_environment_replay")
    args = parser.parse_args()
    print(run_checks(args.expected_workflow_commit, args.output_dir.resolve()))


if __name__ == "__main__":
    main()
