#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent)); sys.path.insert(0,str(HERE))
from common import repo_root  # noqa:E402
from report import build_artifacts  # noqa:E402


class Phase72Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=json.loads((HERE/"experiment.json").read_text()); cls.a=build_artifacts(repo_root(),cls.spec)

    def test_counts(self):
        self.assertEqual(len(self.a["evidence"]),15); self.assertEqual(len(self.a["claims"]["frozen_claims"]),18); self.assertEqual(len(self.a["claims"]["prohibited_claims"]),15); self.assertEqual(len(self.a["figures"]),4)

    def test_negative_evidence(self):
        self.assertFalse(self.a["sources"]["Phase58"]["gates"]["target_met"]); self.assertFalse(self.a["sources"]["Phase59"]["gates"]["target_met"])
        for phase in ("Phase64","Phase66","Phase68"): self.assertIn("FAIL",self.a["sources"][phase]["scientific_outcome"])

    def test_positive_scope(self):
        self.assertEqual(self.a["sources"]["Phase70"]["scientific_outcome"],"MULTIFLOW_HIGH_PAGE_RESIDUAL_THIRD_FRESH_BLIND_PASS")
        self.assertEqual(self.a["sources"]["Phase71"]["scientific_outcome"],"MULTIFLOW_COST_PLACEMENT_INTEGRATION_CONFIRMED")
        self.assertIn("N14",{x["id"] for x in self.a["claims"]["prohibited_claims"]})

    def test_svg_is_deterministic_xml(self):
        for value in self.a["figures"].values(): self.assertTrue(value.startswith('<svg xmlns="http://www.w3.org/2000/svg"')); self.assertTrue(value.endswith("</svg>\n"))


if __name__=="__main__": unittest.main()
