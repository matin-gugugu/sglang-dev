#!/usr/bin/env python3
"""Finalize and hash the completed Phase 25 full-window GPU smoke milestone."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


MARKER = "## GPU smoke result"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=root / "experiment-results/phase25_full_window_teacher",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(root: Path, path: Path) -> None:
    files = sorted(
        item
        for item in root.rglob("*")
        if item.is_file()
        and item != path
        and not item.name.endswith(".pid")
        and item.name != "server.pid"
    )
    path.write_text(
        "".join(f"{sha256(item)}  {item.relative_to(root)}\n" for item in files)
    )


def main() -> None:
    args = parse_args()
    root = args.teacher_root.resolve()
    results = root / "gpu_audit" / "results"
    tp = results / "tp" / "smoke" / "qwen3-8b" / "tp2" / "r0"
    pp = results / "pp" / "smoke"
    analysis = root / "analysis" / "gpu_smoke-v1"
    tp_audit = json.loads((tp / "teacher_audit.json").read_text())
    pp_summary = json.loads((pp / "matrix_summary.json").read_text())
    analysis_summary = json.loads((analysis / "summary.json").read_text())
    pp_audits = [
        json.loads(path.read_text())
        for path in sorted(pp.glob("pp*/mb*/r0/teacher_audit.json"))
    ]
    labels = []
    for name in ("tp_phase_labels.csv.gz", "pp_phase_labels.csv.gz"):
        with gzip.open(root / "labels" / name, "rt", newline="") as source:
            labels.extend(csv.DictReader(source))
    pid_files = sorted(
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and (path.name.endswith(".pid") or path.name == "server.pid")
    )
    checks = {
        "tp_smoke_pass": tp_audit["status"] == "PASS",
        "tp_done": (tp / "DONE").read_text().strip() == "PASS",
        "pp_matrix_complete": (pp / "MATRIX_DONE").read_text().strip()
        == "MEASURED_MISMATCH",
        "pp_cells_9": len(pp_audits) == 9,
        "pp_integrity_pass_9": sum(
            row["checks"]["gpu_integrity"] for row in pp_audits
        )
        == 9,
        "pp_teacher_exact_3": sum(
            row["checks"]["teacher_exact_match"] for row in pp_audits
        )
        == 3,
        "analysis_complete": analysis_summary["status"]
        == "COMPLETE_MEASURED_PP_MISMATCH",
        "labels_remain_provisional": {row["label_status"] for row in labels}
        == {"PROVISIONAL_PENDING_GPU_AUDIT"},
        "no_phase25_pid_files": not pid_files,
    }
    if not all(checks.values()):
        raise RuntimeError(json.dumps({"checks": checks, "pid_files": list(map(str, pid_files))}, indent=2))

    summary = json.loads((root / "summary.json").read_text())
    build_checks = summary.pop("checks", summary.get("build_checks", {}))
    if "gpu_results_empty" in build_checks:
        build_checks["gpu_results_empty_at_build"] = build_checks.pop("gpu_results_empty")
    summary.update(
        {
            "status": "PROVISIONAL_PP_SCHEDULER_MISMATCH",
            "build_checks": build_checks,
            "gpu_smoke": {
                "status": analysis_summary["status"],
                "profile_id": analysis_summary["profile_id"],
                "requests": analysis_summary["requests"],
                "tp": analysis_summary["tp"],
                "pp": analysis_summary["pp"],
                "checks": checks,
            },
            "promotion_gate": (
                "TP smoke passed; PP MB>1 full-window scheduler mismatch blocks "
                "promotion of the provisional PP labels"
            ),
            "next_step": (
                "recover and validate the fixed-draining PP scheduler split/merge "
                "semantics before any full-window teacher training"
            ),
        }
    )
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (root / "PROVISIONAL").write_text("PROVISIONAL_PP_SCHEDULER_MISMATCH\n")

    readme_path = root / "README.md"
    readme = readme_path.read_text()
    if MARKER in readme:
        readme = readme.split(MARKER)[0].rstrip() + "\n\n"
    readme += f"""{MARKER}

The 42-request full-window smoke completed one TP cell and all nine PP cells.
TP matched exactly. PP integrity passed in 9/9 cells, but the static teacher
matched only the three MB1 cells; every MB4/MB16 cell measured a scheduler
split/merge mismatch. Logical bytes remained exact while calls, histogram shape,
and curve-integrated cost changed. PP labels therefore remain provisional.

See `analysis/gpu_smoke-v1/README.md` and `gpu_audit/smoke_summary.json`.
"""
    readme_path.write_text(readme)

    (root / "gpu_audit" / "SMOKE_DONE").write_text(
        "COMPLETE_MEASURED_PP_MISMATCH\n"
    )
    smoke_files = [path for path in results.rglob("*") if path.is_file()]
    smoke_summary = {
        "schema_version": "phase25-full-window-gpu-smoke-archive-v1",
        "status": "COMPLETE_MEASURED_PP_MISMATCH",
        "result_directory": str(results.relative_to(root)),
        "files": len(smoke_files),
        "bytes": sum(path.stat().st_size for path in smoke_files),
        "tp_teacher_exact_cells": 1,
        "pp_integrity_pass_cells": pp_summary["integrity_pass_cells"],
        "pp_teacher_exact_cells": pp_summary["teacher_exact_cells"],
        "pp_teacher_mismatch_cells": pp_summary["cells"]
        - pp_summary["teacher_exact_cells"],
        "checks": checks,
        "can_conclude": [
            "TP full-window batch aggregation is exact for the audited sentinel",
            "PP boundary collection is valid and boundary-invariant",
            "PP MB>1 static grouping does not reproduce the real fixed-draining scheduler",
        ],
        "cannot_conclude": [
            "all TP full-window teachers are GPU-audited",
            "the provisional PP full-window labels are valid training truth",
            "the mismatch magnitude generalizes to all 24 windows",
        ],
    }
    (root / "gpu_audit" / "smoke_summary.json").write_text(
        json.dumps(smoke_summary, indent=2, sort_keys=True) + "\n"
    )

    write_manifest(analysis, analysis / "manifest.sha256")
    write_manifest(root, root / "manifest.sha256")
    print(json.dumps(smoke_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
