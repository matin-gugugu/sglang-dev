#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from build_selection import HISTORY_MS, PRIOR_SELECTIONS, select  # noqa: E402


class Phase45Contracts(unittest.TestCase):
    def test_parent_and_target_isolation(self) -> None:
        contract = json.loads((HERE / "experiment.json").read_text())
        self.assertEqual(contract["workflow_parent_result_commit"], "61773b3d85f9f5c4cdce1ee92b4c287b92810f06")
        self.assertFalse(contract["execution"]["target_generation_or_access_permitted"]); self.assertFalse(contract["execution"]["training_permitted"])

    def test_selection_reproduces_and_is_disjoint(self) -> None:
        with (HERE / "selection/fresh_blind_windows.csv").open(newline="") as source: frozen = list(csv.DictReader(source))
        reproduced = select(ROOT); self.assertEqual([row["window_id"] for row in frozen], [row["window_id"] for row in reproduced]); self.assertEqual(len(frozen), 300)
        prior = {segment: [] for segment in ("burstgpt_1", "burstgpt_2", "burstgpt_3")}
        for relative in PRIOR_SELECTIONS:
            with (ROOT / relative).open(newline="") as source:
                for row in csv.DictReader(source):
                    if row["segment"] in prior: prior[row["segment"]].append(int(row["cutoff_ms"]))
        for index, left in enumerate(frozen):
            self.assertTrue(all(abs(int(left["cutoff_ms"]) - old) >= HISTORY_MS for old in prior[left["segment"]]))
            for right in frozen[index + 1:]:
                if left["segment"] == right["segment"]: self.assertGreaterEqual(abs(int(left["cutoff_ms"]) - int(right["cutoff_ms"])), HISTORY_MS)


if __name__ == "__main__": unittest.main()
