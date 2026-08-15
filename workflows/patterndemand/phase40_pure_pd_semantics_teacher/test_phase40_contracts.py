#!/usr/bin/env python3

import json
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contracts import (
    bin_index,
    build_teacher,
    teacher_chunks_for_wave,
    wave_rid_prefix,
    workload_rows,
)


class TestTeacher(unittest.TestCase):
    def test_scalar_wave_rid_expansion_matches_frozen_request_ids(self):
        contract = json.loads(
            (Path(__file__).resolve().parent / "experiment.json").read_text(
                encoding="utf-8"
            )
        )
        rows = [
            row
            for row in workload_rows(contract)
            if row["scenario"] == "packed_remainder" and row["repeat"] == 1
        ]
        prefix = wave_rid_prefix("packed_remainder", 1)
        self.assertEqual(
            [row["rid"] for row in rows],
            [f"{prefix}_{index}" for index in range(4)],
        )

    def test_fcfs_remainder_and_continuation(self):
        requests = [
            {"scenario": "x", "repeat": 0, "rid": "a", "prompt_tokens": 1000},
            {"scenario": "x", "repeat": 0, "rid": "b", "prompt_tokens": 4000},
        ]
        rows = teacher_chunks_for_wave(
            requests, chunk_tokens=4096, page_size_tokens=1, kv_bytes_per_page=100
        )
        self.assertEqual(
            [(row["rid"], row["kv_page_count"]) for row in rows],
            [("a", 1000), ("b", 3096), ("b", 904)],
        )
        self.assertEqual(sum(row["logical_bytes"] for row in rows), 500_000)

    def test_bins_clip_to_edge_bins(self):
        self.assertEqual(bin_index(1), 0)
        self.assertEqual(bin_index(4096), 0)
        self.assertEqual(bin_index(8589934592 * 2), 11)


class TestOfficialModelDownload(unittest.TestCase):
    def test_pinned_revision_and_weight_inventory_are_self_consistent(self):
        contract = json.loads(
            (Path(__file__).resolve().parent / "experiment.json").read_text(
                encoding="utf-8"
            )
        )
        download = contract["official_model_download"]
        self.assertEqual(download["repo_id"], "Qwen/Qwen3-8B")
        self.assertEqual(
            download["revision"],
            "b968826d9c46dd6066d109eabc6255188de91218",
        )
        self.assertEqual(len(download["weight_shards"]), 5)
        self.assertEqual(
            sum(row["bytes"] for row in download["weight_shards"]),
            download["weights_total_bytes"],
        )
        self.assertTrue(contract["model_contract"]["network_download_permitted"])
        self.assertFalse(download["network_during_formal_execution_permitted"])


class TestCompatibilitySmoke(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads(
            (Path(__file__).resolve().parent / "experiment.json").read_text(encoding="utf-8")
        )
        self.model = {"derived": {"kv_bytes_per_page": 147456}}

    def test_server_command_pins_flashinfer_and_page_one(self):
        from run import build_server_commands

        commands = build_server_commands(
            contract=self.contract,
            model_path=Path("/model"),
            ib_device="mlx5_test",
            prefill_port=39000,
            decode_port=39001,
            router_port=39002,
            bootstrap_port=39003,
        )
        for name in ("prefill", "decode"):
            command = commands[name]
            self.assertEqual(command[command.index("--attention-backend") + 1], "flashinfer")
            self.assertEqual(command[command.index("--page-size") + 1], "1")
        prefill = commands["prefill"]
        self.assertEqual(
            prefill[prefill.index("--optimistic-prefill-retries") + 1], "0"
        )

    def smoke_events(self):
        smoke = self.contract["compatibility_smoke_contract"]
        probe = smoke["admission_probe"]
        segments = {
            f"{smoke['transport_request']['rid_prefix']}_0": [(0, 64)],
        }
        for repeat in range(int(probe["repeats"])):
            for request_index, expected in enumerate(
                probe["expected_segments_by_request_index"]
            ):
                segments[
                    f"{probe['rid_prefix_base']}{repeat}_{request_index}"
                ] = [tuple(segment) for segment in expected]
        rows = []
        for rid, request_segments in segments.items():
            for page_start, page_end in request_segments:
                page_count = page_end - page_start
                rows.append(
                    {
                        "sequence": len(rows),
                        "rid": rid,
                        "backend": "MooncakeKVSender",
                        "page_start": page_start,
                        "page_end": page_end,
                        "page_size_tokens": 1,
                        "kv_page_count": page_count,
                        "kv_bytes_per_page": 147456,
                        "logical_bytes": page_count * 147456,
                        "state_logical_bytes": 0,
                        "raw_tensor_contents_saved": False,
                    }
                )
        return rows

    def test_smoke_requires_transport_and_repeated_atomic_admission(self):
        from run import validate_smoke_events

        rows = self.smoke_events()
        evidence = validate_smoke_events(self.contract, self.model, rows)
        self.assertTrue(all(evidence["checks"].values()))
        self.assertEqual(evidence["transport_sender_chunks"], 1)
        self.assertEqual(evidence["admission_sender_chunks"], 10)

        wrong_page = [dict(row) for row in rows]
        wrong_page[0].update(page_size_tokens=64, kv_bytes_per_page=64 * 147456)
        rejected = validate_smoke_events(self.contract, self.model, wrong_page)
        self.assertFalse(rejected["checks"]["page_size_one"])
        self.assertFalse(rejected["checks"]["bytes_per_page_exact"])

        wrong_admission = [dict(row) for row in rows]
        target = next(
            row
            for row in wrong_admission
            if row["rid"] == "p40::admission_smoke::rep1_3"
            and row["page_start"] == 0
        )
        target["page_end"] = 2000
        target["kv_page_count"] = 2000
        target["logical_bytes"] = 2000 * 147456
        admission_rejected = validate_smoke_events(
            self.contract, self.model, wrong_admission
        )
        self.assertFalse(admission_rejected["checks"]["admission_segments_exact"])
        self.assertFalse(admission_rejected["checks"]["admission_repeats_exact"])

        unexpected = validate_smoke_events(
            self.contract,
            self.model,
            [*rows, dict(rows[0], sequence=len(rows), rid="unexpected")],
        )
        self.assertFalse(unexpected["checks"]["no_unexpected_profile_records"])

    def test_source_contract_pins_atomic_batch_barrier_default_off(self):
        from preflight import source_semantics_audit

        checks = source_semantics_audit()
        self.assertTrue(checks["pretokenized_batch_uses_batch_dispatch"])
        self.assertTrue(checks["batch_tokenized_request_single_dispatch"])
        self.assertTrue(checks["scheduler_handles_batch_before_admission"])
        self.assertTrue(checks["bootstrap_barrier_env_declared"])
        self.assertTrue(checks["bootstrap_barrier_waits_for_all"])
        self.assertTrue(checks["bootstrap_barrier_requires_whole_batch_capacity"])


class TestProfiler(unittest.TestCase):
    def test_profile_is_disabled_by_default_and_records_only_metadata(self):
        profile_path = (
            Path(__file__).resolve().parents[3]
            / "python/sglang/srt/disaggregation/pd_comm_profile.py"
        )
        spec = importlib.util.spec_from_file_location("phase40_pd_comm_profile", profile_path)
        assert spec and spec.loader
        pd_comm_profile = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pd_comm_profile)

        with tempfile.TemporaryDirectory() as temporary:
            old = os.environ.get("SGLANG_PD_COMM_PROFILE_DIR")
            os.environ["SGLANG_PD_COMM_PROFILE_DIR"] = temporary
            os.environ["SGLANG_PD_COMM_PROFILE_RUN_ID"] = "unit"
            try:
                pd_comm_profile.reset_for_test()
                pd_comm_profile.record_send(
                    rid="r0",
                    bootstrap_room=7,
                    backend="MooncakeKVSender",
                    page_start=0,
                    page_end=2,
                    kv_page_count=2,
                    kv_bytes_per_page=128,
                    state_indices=None,
                    state_bytes_per_index=0,
                    page_size_tokens=1,
                )
                files = list(Path(temporary).glob("*.jsonl"))
                self.assertEqual(len(files), 1)
                row = json.loads(files[0].read_text(encoding="utf-8"))
                self.assertEqual(row["logical_bytes"], 256)
                self.assertFalse(row["raw_tensor_contents_saved"])
                self.assertNotIn("token_ids", row)
            finally:
                if old is None:
                    os.environ.pop("SGLANG_PD_COMM_PROFILE_DIR", None)
                else:
                    os.environ["SGLANG_PD_COMM_PROFILE_DIR"] = old
                os.environ.pop("SGLANG_PD_COMM_PROFILE_RUN_ID", None)


class TestSyntheticClosedLoop(unittest.TestCase):
    def test_all_frozen_waves_align(self):
        from run import compare_events

        contract = json.loads(
            (Path(__file__).resolve().parent / "experiment.json").read_text(
                encoding="utf-8"
            )
        )
        model = {
            "derived": {"kv_bytes_per_page": 147456},
            "structure": {"page_size_tokens": 1},
        }
        teacher = build_teacher(contract, model)
        raw = []
        for sequence, row in enumerate(teacher):
            raw.append(
                {
                    **row,
                    "sequence": sequence,
                    "backend": "MooncakeKVSender",
                    "page_size_tokens": 1,
                    "kv_bytes_per_page": 147456,
                    "state_logical_bytes": 0,
                    "raw_tensor_contents_saved": False,
                }
            )
        alignment, histograms, evidence = compare_events(contract, model, raw)
        self.assertEqual(evidence["requests"], 45)
        self.assertGreater(evidence["gpu_events"], 45)
        self.assertEqual(len(alignment), 6)
        self.assertEqual(len(histograms), 72)
        self.assertTrue(all(evidence["checks"].values()))


if __name__ == "__main__":
    unittest.main()
