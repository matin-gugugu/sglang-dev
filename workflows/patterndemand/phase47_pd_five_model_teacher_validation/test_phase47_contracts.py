#!/usr/bin/env python3
"""CPU-only Phase47 contract tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

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
    from run import (
        load_phase40_run,
        pin_bf16_kv_cache,
        pin_page_aware_teacher,
        validate_smoke_events_generic,
    )

    base = json.loads((HERE / "experiment.json").read_text())
    models = model_specs()
    preflight_source = (HERE / "preflight.py").read_text(encoding="utf-8")
    run_source = (HERE / "run.py").read_text(encoding="utf-8")
    verify_source = (HERE / "verify.py").read_text(encoding="utf-8")
    assert base["backend_contract"]["kv_cache_dtype"] == "bf16"
    assert base["measurement_contract"]["prefill_budget_accounting"].startswith("page_aligned")
    assert 'commands[name][index + 1] = "bf16"' in run_source
    assert 'commands[name][index + 1] = "bfloat16"' not in run_source
    assert '== "bf16"' in verify_source
    assert '== "bfloat16"' not in verify_source
    p40_stub = SimpleNamespace(
        build_server_commands=lambda **_: {
            "prefill": ["--kv-cache-dtype", "auto"],
            "decode": ["--kv-cache-dtype", "auto"],
        }
    )
    pin_bf16_kv_cache(p40_stub)
    commands = p40_stub.build_server_commands()
    assert commands["prefill"] == ["--kv-cache-dtype", "bf16"]
    assert commands["decode"] == ["--kv-cache-dtype", "bf16"]
    assert 'find_spec("sglang.srt.layers.attention.trtllm_mla_backend")' in preflight_source
    assert 'find_spec("sgl_kernel.flash_mla")' not in preflight_source
    assert "extend_input_len = self.ceil_paged_tokens(extend_input_len)" in preflight_source
    assert len(base["superseded_blocked_attempts"]) == 3
    assert base["superseded_blocked_attempts"][-1]["blocked_result_commit"].startswith("e88dbada")
    assert any(row["name"] == "sglang_schedule_policy" for row in base["pinned_inputs"])
    assert len(models) == 5
    assert len({row["model_id"] for row in models}) == 5
    assert all(len(row["revision"]) == 40 for row in models)
    assert models[0]["model_id"] == "deepseek-v2-lite"
    assert models[0]["attention_backend"] == "trtllm_mla"
    assert models[0]["page_size_tokens"] == 64
    p40 = load_p40_contracts()
    legacy_build_teacher = p40.build_teacher
    pin_page_aware_teacher(p40)
    for spec in models:
        runtime = runtime_contract(base, spec)
        requests = p40.workload_rows(runtime)
        assert len(requests) == 45
        assert len({row["rid"] for row in requests}) == 45
        model = {
            "derived": {"kv_bytes_per_page": spec["expected_kv_bytes_per_page"]}
        }
        legacy_teacher = legacy_build_teacher(runtime, model)
        teacher = p40.build_teacher(runtime, model)
        assert teacher
        assert all(int(row["logical_bytes"]) % int(spec["expected_kv_bytes_per_page"]) == 0 for row in teacher)
        if spec["page_size_tokens"] == 1:
            assert teacher == legacy_teacher
        else:
            assert teacher != legacy_teacher
        smoke = runtime["compatibility_smoke_contract"]
        assert smoke["expected_sender_chunks_total"] == 21
        assert smoke["expected_kv_page_count"] == (1 if spec["page_size_tokens"] == 64 else 64)
        probes = smoke["admission_probes"]
        assert [row["name"] for row in probes] == ["packed_remainder", "page_boundary_crosscheck"]
        packed_segments = probes[0]["expected_segments_by_request_index"][3]
        boundary_segments = probes[1]["expected_segments_by_request_index"][3]
        assert packed_segments == ([[0, 16], [16, 32]] if spec["is_mla"] else [[0, 1096], [1096, 2000]])
        assert boundary_segments == ([[0, 45], [45, 47]] if spec["is_mla"] else [[0, 2967], [2967, 3000]])
        assert spec["expected_kv_bytes_per_page"] == spec["expected_kv_bytes_per_token"] * spec["page_size_tokens"]

    # Exercise the reused Phase40 comparator with the new Phase47 teacher. This
    # catches module-wiring drift that source-text assertions cannot detect.
    p40_run = load_phase40_run()
    pin_page_aware_teacher(p40_run)
    deepseek = models[0]
    runtime = runtime_contract(base, deepseek)
    model = {"derived": {"kv_bytes_per_page": deepseek["expected_kv_bytes_per_page"]}}
    teacher = p40_run.build_teacher(runtime, model)
    synthetic_events = [
        {
            **row,
            "sequence": index,
            "backend": "MooncakeKVSender",
            "page_size_tokens": 64,
            "kv_bytes_per_page": deepseek["expected_kv_bytes_per_page"],
            "state_logical_bytes": 0,
            "raw_tensor_contents_saved": False,
        }
        for index, row in enumerate(teacher)
    ]
    _, _, evidence = p40_run.compare_events(runtime, model, synthetic_events)
    evidence["checks"].pop("page_size_one")
    assert evidence["requests"] == 45
    assert evidence["exact_requests"] == 45
    assert all(evidence["checks"].values())

    smoke = runtime["compatibility_smoke_contract"]
    smoke_events = []
    sequence = 0
    transport = smoke["transport_request"]
    smoke_events.append(
        {
            "rid": f"{transport['rid_prefix']}_0",
            "sequence": sequence,
            "backend": "MooncakeKVSender",
            "page_start": 0,
            "page_end": smoke["expected_kv_page_count"],
            "page_size_tokens": 64,
            "kv_page_count": smoke["expected_kv_page_count"],
            "kv_bytes_per_page": deepseek["expected_kv_bytes_per_page"],
            "logical_bytes": smoke["expected_kv_page_count"] * deepseek["expected_kv_bytes_per_page"],
            "state_logical_bytes": 0,
            "raw_tensor_contents_saved": False,
        }
    )
    sequence += 1
    for probe in smoke["admission_probes"]:
        for repeat in range(probe["repeats"]):
            for request_index, segments in enumerate(probe["expected_segments_by_request_index"]):
                for page_start, page_end in segments:
                    page_count = page_end - page_start
                    smoke_events.append(
                        {
                            "rid": f"{probe['rid_prefix_base']}{repeat}_{request_index}",
                            "sequence": sequence,
                            "backend": "MooncakeKVSender",
                            "page_start": page_start,
                            "page_end": page_end,
                            "page_size_tokens": 64,
                            "kv_page_count": page_count,
                            "kv_bytes_per_page": deepseek["expected_kv_bytes_per_page"],
                            "logical_bytes": page_count * deepseek["expected_kv_bytes_per_page"],
                            "state_logical_bytes": 0,
                            "raw_tensor_contents_saved": False,
                        }
                    )
                    sequence += 1
    smoke_evidence = validate_smoke_events_generic(runtime, model, smoke_events)
    assert smoke_evidence["admission_probe_names"] == ["packed_remainder", "page_boundary_crosscheck"]
    assert len(smoke_events) == 21
    assert all(smoke_evidence["checks"].values())
    print(json.dumps({"status": "PASS", "models": 5, "requests_per_model": 45, "total_requests": 225}, indent=2))


if __name__ == "__main__":
    main()
