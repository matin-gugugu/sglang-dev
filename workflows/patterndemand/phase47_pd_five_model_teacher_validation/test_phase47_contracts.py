#!/usr/bin/env python3
"""CPU-only Phase47 contract tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
P40_CONTRACTS = HERE.parent / "phase40_pure_pd_semantics_teacher/contracts.py"


def load_p40_contracts():
    spec = importlib.util.spec_from_file_location("phase47_test_p40_contracts", P40_CONTRACTS)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    from contracts import model_specs, runtime_contract

    base = json.loads((HERE / "experiment.json").read_text())
    models = model_specs()
    preflight_source = (HERE / "preflight.py").read_text(encoding="utf-8")
    assert 'find_spec("sgl_kernel.flash_mla")' in preflight_source
    assert 'find_spec("flash_mla")' not in preflight_source
    assert len(models) == 5
    assert len({row["model_id"] for row in models}) == 5
    assert all(len(row["revision"]) == 40 for row in models)
    p40 = load_p40_contracts()
    for spec in models:
        runtime = runtime_contract(base, spec)
        requests = p40.workload_rows(runtime)
        assert len(requests) == 45
        assert len({row["rid"] for row in requests}) == 45
        model = {
            "derived": {"kv_bytes_per_page": spec["expected_kv_bytes_per_page"]}
        }
        teacher = p40.build_teacher(runtime, model)
        assert teacher
        assert all(int(row["logical_bytes"]) % int(spec["expected_kv_bytes_per_page"]) == 0 for row in teacher)
        smoke = runtime["compatibility_smoke_contract"]
        assert smoke["expected_sender_chunks_total"] == 11
        assert smoke["expected_kv_page_count"] == (1 if spec["page_size_tokens"] == 64 else 64)
        segments = smoke["admission_probe"]["expected_segments_by_request_index"][3]
        assert segments == ([[0, 17], [17, 32]] if spec["is_mla"] else [[0, 1096], [1096, 2000]])
        assert spec["expected_kv_bytes_per_page"] == spec["expected_kv_bytes_per_token"] * spec["page_size_tokens"]
    print(json.dumps({"status": "PASS", "models": 5, "requests_per_model": 45, "total_requests": 225}, indent=2))


if __name__ == "__main__":
    main()
