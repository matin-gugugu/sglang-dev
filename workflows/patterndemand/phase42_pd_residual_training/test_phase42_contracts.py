#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common import load_json, repo_root, verify_pinned_inputs
from metrics import compare_to_h0, metric_bundle
from model import fit_model, forward


class Phase42Contracts(unittest.TestCase):
    def test_pins_and_parent(self) -> None:
        contract = load_json(HERE / "experiment.json")
        self.assertEqual(contract["workflow_parent_result_commit"], "914f7fb53c4ff076a5fe12f7c56624502673eb1f")
        self.assertTrue(all(value["ok"] for value in verify_pinned_inputs(contract).values()))

    def test_split_and_blind_contract(self) -> None:
        import csv, gzip
        base = repo_root() / "experiment-results/phase41_pd_full_window_dataset/dataset"
        with gzip.open(base / "pd_development_h0_residual_examples.csv.gz", "rt", newline="") as source: rows = list(csv.DictReader(source))
        with gzip.open(base / "pd_blind_target_free_features.csv.gz", "rt", newline="") as source: blind = list(csv.DictReader(source))
        self.assertEqual(sum(row["split_role"] == "development_train" for row in rows), 75)
        self.assertEqual(sum(row["split_role"] == "development_validation" for row in rows), 19)
        self.assertEqual(len(blind), 12)
        self.assertFalse(any(name.startswith("target_") or name.startswith("residual_") for name in blind[0]))

    def test_numpy_mlp_learns_simple_mapping(self) -> None:
        rng = np.random.default_rng(7); x = rng.normal(size=(24, 3)); y = np.tanh(x @ rng.normal(scale=0.2, size=(3, 26)))
        config = {"width": 8, "depth": 1, "learning_rate": 0.02, "weight_decay": 0.0, "max_epochs": 500, "patience": 100}
        model, audit = fit_model(x, y, config, 9, fixed_epochs=300)
        self.assertLess(float(np.mean((forward(model, x) - y) ** 2)), 0.01)
        self.assertEqual(audit["best_epoch"], 300)

    def test_identity_metrics(self) -> None:
        value = np.arange(1, 25, dtype=float).reshape(2, 12)
        metrics = metric_bundle(value, value * 100, value, value * 100)
        self.assertEqual(metrics["calls_histogram_wape"], 0.0)
        h0 = dict(metrics); h0.update({key: 1.0 for key in ("calls_histogram_wape", "bytes_histogram_wape", "mean_calls_histogram_tv", "mean_normalized_log_payload_emd")})
        dnn = dict(h0); dnn.update({key: 0.5 for key in ("calls_histogram_wape", "bytes_histogram_wape", "mean_calls_histogram_tv", "mean_normalized_log_payload_emd")})
        self.assertEqual(compare_to_h0(dnn, h0)["outcome"], "IMPROVES_COMPOSITE")


if __name__ == "__main__": unittest.main()
