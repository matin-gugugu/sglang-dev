#!/usr/bin/env python3
"""Read-only Phase69 contract and Phase70 blind-boundary audit."""
from __future__ import annotations
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import load_json, repo_root, require_clean_before_run, require_expected_head, utc_now, verify_pinned_inputs  # noqa: E402
from model import read_development  # noqa: E402


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_phase70(contract: dict[str, Any]) -> dict[str, Any]:
    grid = load_json(repo_root() / contract["phase70_blind_boundary"]["reserved_grid"])
    base = {key: value for key, value in grid.items() if key != "grid_sha256"}
    reserved = {int(page) for vectors in grid["configurations"].values() for vector in vectors for page in vector}
    development = {
        int(page)
        for pages in contract["dataset_contract"]["development_page_sets"].values()
        for page in pages
    }
    checks = {
        "schema": grid.get("schema_version") == "phase70-reserved-multiflow-high-page-blind-grid-v1",
        "canonical_sha": grid.get("grid_sha256") == canonical_sha(base) == contract["phase70_blind_boundary"]["reserved_grid_canonical_sha256"],
        "frozen_closed": grid.get("frozen_before_phase69_fit") is True and grid.get("phase70_targets_opened") is False,
        "models": grid.get("models") == contract["dataset_contract"]["models"],
        "topologies": grid.get("topology_levels") == contract["dataset_contract"]["topology_levels"],
        "configurations": list(grid.get("configurations", {})) == contract["dataset_contract"]["configurations"],
        "ten_vectors": all(len(vectors) == 10 for vectors in grid["configurations"].values()),
        "page_set": reserved == set(contract["phase70_blind_boundary"]["page_set"]),
        "zero_overlap": not reserved & development,
        "curve_bracket": min(reserved) >= 32 and max(reserved) <= 64,
        "new_placement": grid.get("required_new_placement_policy", "").startswith("Every Phase70 endpoint tuple"),
    }
    if not all(checks.values()):
        raise RuntimeError({"phase70_grid": checks})
    return {
        "status": "PASS", "checks": checks, "grid_sha256": grid["grid_sha256"],
        "development_pages": sorted(development), "reserved_pages": sorted(reserved), "phase70_targets_read": False,
    }


def run_checks(expected: str) -> dict[str, Any]:
    head = require_expected_head(expected)
    require_clean_before_run()
    contract = load_json(HERE / "experiment.json")
    pins = verify_pinned_inputs(contract)
    root = repo_root()
    summaries = {
        phase: load_json(root / f"experiment-results/{directory}/summary.json")
        for phase, directory in (
            ("phase64", "phase64_pd_multiflow_graph_zero_shot"),
            ("phase66", "phase66_pd_graph_correction_fresh_blind"),
            ("phase68", "phase68_pd_graph_page_shape_fresh_blind"),
        )
    }
    expected_outcomes = {
        "phase64": "MULTIFLOW_GRAPH_ZERO_SHOT_FAIL_RETAIN_FOR_DEVELOPMENT",
        "phase66": "MULTIFLOW_GRAPH_CORRECTION_FRESH_BLIND_FAIL_RETAIN_AS_BLIND_EVIDENCE",
        "phase68": "MULTIFLOW_GRAPH_PAGE_SHAPE_SECOND_FRESH_BLIND_FAIL_RETAIN_AS_BLIND_EVIDENCE",
    }
    if any(summaries[phase].get("scientific_outcome") != outcome for phase, outcome in expected_outcomes.items()):
        raise RuntimeError({"invalid_source_outcome": {phase: summaries[phase].get("scientific_outcome") for phase in summaries}})
    r65 = load_json(root / "experiment-results/phase65_pd_graph_correction_development/model/multiflow_graph_correction.json")
    r67 = load_json(root / "experiment-results/phase67_pd_graph_page_shape_refinement/model/multiflow_graph_page_correction.json")
    rows = read_development(
        root / contract["dataset_contract"]["phase64_source"],
        root / contract["dataset_contract"]["phase66_source"],
        root / contract["dataset_contract"]["phase68_source"],
        r65,
        r67,
    )
    counts = {
        "source": Counter(row["source_phase"] for row in rows),
        "model": Counter(row["model_id"] for row in rows),
        "configuration": Counter(row["configuration"] for row in rows),
        "topology": Counter(row["topology_level"] for row in rows),
        "cohort": Counter((row["source_phase"], row["vector_index"]) for row in rows),
    }
    checks = {
        "rows": len(rows) == 720,
        "source": counts["source"] == Counter({"phase64": 240, "phase66": 240, "phase68": 240}),
        "models": counts["model"] == Counter({model: 360 for model in contract["dataset_contract"]["models"]}),
        "configurations": counts["configuration"] == Counter({configuration: 180 for configuration in contract["dataset_contract"]["configurations"]}),
        "topologies": counts["topology"] == Counter({topology: 240 for topology in contract["dataset_contract"]["topology_levels"]}),
        "cohorts": len(counts["cohort"]) == 30 and set(counts["cohort"].values()) == {24},
        "positive": all(row["actual_concurrent_wave_us"] > 0 and row["r67_prediction_us"] > 0 for row in rows),
        "anchor_recomputed": True,
    }
    if not all(checks.values()):
        raise RuntimeError({"development_matrix": checks, "counts": counts})
    blind = validate_phase70(contract)
    return {
        "schema_version": "phase69-preflight-v1", "status": "PASS", "workflow_commit": head,
        "captured_at_utc": utc_now(), "pinned_inputs": pins, "checks": checks,
        "source_results": {
            phase: {"status": summary["status"], "scientific_outcome": summary["scientific_outcome"], "labels_now_development": True}
            for phase, summary in summaries.items()
        },
        "phase70_reserved_grid": blind,
        "execution": {"gpu_used": False, "network_used": False, "new_physical_measurement": False, "phase70_targets_read": False},
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-workflow-commit", required=True)
    arguments = parser.parse_args()
    print(json.dumps(run_checks(arguments.expected_workflow_commit), ensure_ascii=False, indent=2, default=dict))
