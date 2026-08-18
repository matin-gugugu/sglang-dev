#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

from report import (  # noqa: E402
    build_chain_rows,
    build_claim_scope,
    build_evidence_rows,
    load_source_summaries,
    render_freeze_report,
    render_guide,
)


class Phase53ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = json.loads((HERE / "experiment.json").read_text(encoding="utf-8"))
        cls.summaries = load_source_summaries(ROOT, cls.spec)

    def test_source_and_chain_cardinality(self) -> None:
        evidence = build_evidence_rows(self.spec, self.summaries)
        chains = build_chain_rows(self.summaries)
        self.assertEqual(len(evidence), 19)
        self.assertEqual([row["phase"] for row in evidence], [row["phase"] for row in self.spec["source_results"]])
        self.assertEqual({row["chain"] for row in chains}, {"TP", "PP", "PD"})

    def test_negative_evidence_and_claim_boundaries(self) -> None:
        evidence = build_evidence_rows(self.spec, self.summaries)
        phase43 = next(row for row in evidence if row["phase"] == "Phase43")
        self.assertEqual(phase43["evidence_class"], "fresh_blind_negative")
        self.assertIn("DNN未优于H0", phase43["key_fact"])
        claims = build_claim_scope(self.spec)
        self.assertEqual(len(claims["frozen_claims"]), 12)
        self.assertEqual(len(claims["prohibited_claims"]), 10)
        self.assertEqual(len(claims["future_scheduler_dimensions"]), 6)

    def test_reports_preserve_scope(self) -> None:
        evidence = build_evidence_rows(self.spec, self.summaries)
        chains = build_chain_rows(self.summaries)
        claims = build_claim_scope(self.spec)
        guide = render_guide(self.summaries, evidence, chains, claims)
        report = render_freeze_report("W53-test", evidence, chains, claims)
        for term in ("Phase43", "DNN不如H0", "Phase50", "communication-only", "完整调度器"):
            self.assertIn(term, guide)
        self.assertIn("0.0221%降到0.0185%", guide)
        for term in ("Phase43", "有效负结果", "Phase39", "Phase51", "完整scheduler"):
            self.assertIn(term, report)
        self.assertIn("0.0221%→0.0185%", report)


if __name__ == "__main__":
    unittest.main()
