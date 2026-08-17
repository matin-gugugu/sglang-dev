#!/usr/bin/env python3
"""Verify Phase42 compact result and blind-target isolation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import load_json, repo_root, verify_result_manifest  # noqa: E402
from model import read_csv_gz, read_json_gz  # noqa: E402


def verify(output: Path) -> dict:
    manifest = verify_result_manifest(output)
    summary = load_json(output / "summary.json")
    blind = read_csv_gz(output / "predictions/blind_frozen_predictions.csv.gz")
    validation = read_csv_gz(output / "predictions/development_validation_predictions.csv.gz")
    checkpoint = read_json_gz(output / "checkpoints/pd_qwen3_h0_dnn_residual.json.gz")
    forbidden = sorted({name for row in blind for name in row if name.startswith("target_") or name.startswith("residual_")})
    checks = {
        "manifest": manifest["ok"], "status": summary.get("status") == "PASS",
        "blind_24": len(blind) == 24, "blind_unique_12": len({row["profile_id"] for row in blind}) == 12,
        "blind_methods": {row["method"] for row in blind} == {"h0", "h0_plus_dnn_residual"},
        "blind_target_free": not forbidden and int(summary["counts"]["blind_target_rows"]) == 0,
        "validation_38": len(validation) == 38, "checkpoint_models_3": len(checkpoint["models"]) == 3,
        "checkpoint_no_forbidden": not any(checkpoint["forbidden_assets_seen"].values()),
        "done": (output / "DONE").read_text().strip() == "PASS",
    }
    if not all(checks.values()): raise RuntimeError({"checks": checks, "forbidden": forbidden})
    return {"status": "PASS", "checks": checks, "blind_predictions": len(blind), "manifest_files": manifest["manifest"]["checked_files"]}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase42_pd_residual_training")
    args = parser.parse_args(); print(json.dumps(verify(args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
