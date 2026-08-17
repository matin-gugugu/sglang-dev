#!/usr/bin/env python3
"""CPU-only contract tests for Phase41."""

from __future__ import annotations

import json
import sys
import unittest
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
from contracts import (  # noqa: E402
    partition_requests,
    predictor_features,
    pseudo_requests,
    read_csv,
    sentinel_workload,
    teacher_for_requests,
)
from build_phase21b_pp_h0 import pseudo_requests as frozen_numpy_pseudo_requests  # noqa: E402
from run import compare_gpu_teacher  # noqa: E402


class Phase41ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads((HERE / "experiment.json").read_text())
        cls.features = json.loads((HERE / "feature_contract.json").read_text())

    def test_wave_boundaries(self) -> None:
        for count, expected in ((63, [63]), (64, [64]), (65, [64, 1]), (129, [64, 64, 1])):
            waves = partition_requests(list(range(count)), 64)
            self.assertEqual([len(wave) for wave in waves], expected)
            self.assertEqual([value for wave in waves for value in wave], list(range(count)))

    def test_teacher_resets_at_wave_boundary(self) -> None:
        requests = [(1000, 2)] * 65
        request_rows, events = teacher_for_requests(
            case="boundary",
            repeat=0,
            requests=requests,
            wave_size=64,
            chunk_tokens=4096,
            page_size_tokens=1,
            kv_bytes_per_page=147456,
        )
        self.assertEqual(len(request_rows), 65)
        self.assertEqual(Counter(row["wave_index"] for row in request_rows), Counter({0: 64, 1: 1}))
        final_request_events = [row for row in events if row["request_index"] == 64]
        self.assertEqual(
            [(row["page_start"], row["page_end"]) for row in final_request_events],
            [(0, 1000)],
        )
        self.assertTrue(all(row["wave_index"] in (0, 1) for row in events))

    def test_stdlib_h0_matches_frozen_numpy_h0(self) -> None:
        profiles = read_csv(
            ROOT
            / "experiment-results/phase34b_six_model_hfull_dataset/profiles/low_dimensional_profiles_94.csv.gz"
        )
        self.assertEqual(len(profiles), 94)
        for profile in profiles:
            self.assertEqual(pseudo_requests(profile), frozen_numpy_pseudo_requests(profile))

    def test_sentinel_counts(self) -> None:
        real_counts = {
            row["profile_id"]: int(row["request_count"])
            for row in self.contract["gpu_sentinel_contract"]["real_full_window_cases"]
        }
        bundle = {
            "development": [
                {
                    "profile": {"profile_id": profile_id},
                    "requests": [[64, 2] for _ in range(count)],
                }
                for profile_id, count in real_counts.items()
            ]
        }
        requests, teacher, inventory = sentinel_workload(self.contract, bundle, 147456)
        self.assertEqual(len(requests), 4853)
        self.assertEqual(
            len({(row["case"], row["repeat"], row["wave_index"]) for row in requests}),
            82,
        )
        self.assertTrue(teacher)
        self.assertEqual(sum(int(row["total_requests"]) for row in inventory), 4853)

        raw = []
        for sequence, row in enumerate(teacher):
            raw.append(
                {
                    **row,
                    "sequence": sequence,
                    "kv_bytes_per_page": 147456,
                    "page_size_tokens": 1,
                    "backend": "MooncakeKVSender",
                    "state_logical_bytes": 0,
                    "raw_tensor_contents_saved": False,
                }
            )
        alignment, histograms, evidence = compare_gpu_teacher(
            contract=self.contract,
            model={"derived": {"kv_bytes_per_page": 147456}},
            requests=requests,
            teacher=teacher,
            raw_events=raw,
        )
        overall = next(row for row in alignment if row["case"] == "overall")
        self.assertEqual(overall["waves"], 82)
        self.assertEqual(overall["exact_requests"], 4853)
        self.assertEqual(len(histograms), 96)
        self.assertTrue(all(evidence["checks"].values()))

    def test_feature_contract_is_target_free(self) -> None:
        profile = read_csv(
            ROOT
            / "experiment-results/phase34b_six_model_hfull_dataset/profiles/low_dimensional_profiles_94.csv.gz"
        )[0]
        features = predictor_features(profile, self.features)
        self.assertTrue(features)
        self.assertTrue(all(name.startswith("feature_") for name in features))
        self.assertFalse(any(name.startswith(("target_", "residual_")) for name in features))
        self.assertNotIn("profile_id", features)
        self.assertNotIn("cutoff_ms", features)

    def test_blind_selection_is_fresh_and_disjoint(self) -> None:
        rows = read_csv(HERE / "selection/blind_windows.csv")
        self.assertEqual(len(rows), 12)
        self.assertEqual(
            Counter(row["segment"] for row in rows),
            Counter({"burstgpt_1": 4, "burstgpt_2": 4, "burstgpt_3": 4}),
        )
        self.assertEqual(sum(int(row["history_count"]) for row in rows), 2887)
        for index, left in enumerate(rows):
            for right in rows[index + 1 :]:
                if left["segment"] == right["segment"]:
                    self.assertGreaterEqual(
                        abs(int(left["cutoff_ms"]) - int(right["cutoff_ms"])), 300_000
                    )
        prior = []
        for path in sorted(ROOT.glob("experiment-results/*/selection/selected_windows.csv")):
            prior.extend(read_csv(path))
        for row in rows:
            self.assertFalse(
                any(
                    old["segment"] == row["segment"]
                    and abs(int(old["cutoff_ms"]) - int(row["cutoff_ms"])) < 300_000
                    for old in prior
                )
            )

    def test_declared_counts_and_scope(self) -> None:
        self.assertEqual(self.contract["workflow_parent_result_commit"], "0af31cda42b72042e1fe5a6173a440878c1a50e6")
        self.assertEqual(self.contract["dataset_contract"]["development_full_requests"], 35524)
        self.assertEqual(self.contract["dataset_contract"]["blind_targets_state"], "NOT_GENERATED")
        self.assertEqual(self.contract["acceptance_gates"]["blind_target_rows"], 0)
        self.assertIn("DNN training or hyperparameter selection", self.contract["research_scope"]["not_in_scope"])


if __name__ == "__main__":
    unittest.main()
