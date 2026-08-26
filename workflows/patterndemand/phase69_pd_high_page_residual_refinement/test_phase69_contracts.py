#!/usr/bin/env python3
from __future__ import annotations
import json
import math
import sys
import unittest
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import repo_root  # noqa: E402
from model import CANDIDATES, evaluate, fit_model, predict, read_development  # noqa: E402
from preflight import validate_phase70  # noqa: E402


class Phase69Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads((HERE / "experiment.json").read_text(encoding="utf-8"))
        root = repo_root()
        r65 = json.loads((root / "experiment-results/phase65_pd_graph_correction_development/model/multiflow_graph_correction.json").read_text(encoding="utf-8"))
        r67 = json.loads((root / "experiment-results/phase67_pd_graph_page_shape_refinement/model/multiflow_graph_page_correction.json").read_text(encoding="utf-8"))
        cls.rows = read_development(
            root / cls.contract["dataset_contract"]["phase64_source"],
            root / cls.contract["dataset_contract"]["phase66_source"],
            root / cls.contract["dataset_contract"]["phase68_source"],
            r65,
            r67,
        )
        cls.evaluation = evaluate(cls.rows, cls.contract)

    def test_matrix(self):
        self.assertEqual(len(self.rows), 720)
        self.assertEqual(Counter(row["source_phase"] for row in self.rows), Counter({"phase64": 240, "phase66": 240, "phase68": 240}))
        self.assertEqual(Counter((row["source_phase"], row["vector_index"]) for row in self.rows), Counter({(source, index): 24 for source in ("phase64", "phase66", "phase68") for index in range(10)}))

    def test_blind_grid(self):
        value = validate_phase70(self.contract)
        self.assertEqual(value["reserved_pages"], [34, 38, 44, 52, 60])
        self.assertTrue(set(value["reserved_pages"]).isdisjoint(value["development_pages"]))

    def test_fixed_selection(self):
        self.assertTrue(self.evaluation["candidates"][0]["target_guard"])
        self.assertEqual(self.evaluation["selected"]["candidate_id"], "r67_high_page_linear")

    def test_registered_guards(self):
        selected = self.evaluation["selected"]
        self.assertEqual(set(selected["schemes"]), {"payload_cohort", "topology", "tail64", "source_blocked"})
        self.assertTrue(all(value["pass"] for value in selected["schemes"].values()))
        self.assertTrue(all(selected["schemes"][scheme]["checks"]["improves_best_baseline_each_shared_endpoint_configuration"] for scheme in ("payload_cohort", "topology", "tail64")))
        self.assertTrue(all(selected["schemes"][scheme]["checks"]["preserves_p2d2_matching"] for scheme in ("payload_cohort", "topology", "tail64")))

    def test_anchor_and_full_refit(self):
        model = fit_model(self.rows, CANDIDATES[0])
        self.assertEqual(len(model["groups"]), 8)
        self.assertEqual({group["training_rows"] for group in model["groups"].values()}, {90})
        for row in self.rows:
            if max(row["pages_list"]) <= 32 or row["configuration"] == "P2D2_MATCHING":
                self.assertTrue(math.isclose(predict(model, row), row["r67_prediction_us"], rel_tol=0.0, abs_tol=0.0))


if __name__ == "__main__":
    unittest.main()
