#!/usr/bin/env python3

from __future__ import annotations

import csv
import gzip
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import load_json, repo_root, verify_pinned_inputs


class Phase43Contracts(unittest.TestCase):
    def test_parent_and_pins(self) -> None:
        contract = load_json(HERE / "experiment.json")
        self.assertEqual(contract["workflow_parent_result_commit"], "88dd1a8f5a4b9452e226118ade270aa3eb6fed7e")
        self.assertTrue(all(value["ok"] for value in verify_pinned_inputs(contract).values()))

    def test_frozen_predictions_are_target_free(self) -> None:
        path = repo_root() / "experiment-results/phase42_pd_residual_training/predictions/blind_frozen_predictions.csv.gz"
        with gzip.open(path, "rt", newline="") as source: rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 24); self.assertEqual(len({row["profile_id"] for row in rows}), 12)
        self.assertFalse(any(name.startswith("target_") or name.startswith("residual_") for name in rows[0]))

    def test_raw_contract_stays_external(self) -> None:
        contract = load_json(HERE / "experiment.json")
        self.assertEqual(contract["raw_source_contract"]["storage"], "Git-external and read-only")
        self.assertEqual(contract["required_outputs"]["full_request_rows_in_git"], 0)


if __name__ == "__main__": unittest.main()
