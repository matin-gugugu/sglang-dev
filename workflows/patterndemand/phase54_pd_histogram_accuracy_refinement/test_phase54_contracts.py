#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path

from contracts import load_contract, loss_contract_self_check
from model import loss_weights, encode_histograms, decode_histograms
from preflight import run_checks


class Phase54Contracts(unittest.TestCase):
    def test_contract_and_pins(self) -> None:
        contract = load_contract()
        self.assertEqual(contract["workflow_parent_result_commit"], "08f71defbeac2e1adc2c4eff66af83933ab83499")
        self.assertFalse(contract["execution"]["gpu_permitted"])
        self.assertFalse(contract["execution"]["phase50_blind_access_permitted"])
        self.assertEqual(len(contract["predictor_contract"]["candidate_grid"]), 4)
        self.assertTrue(all(loss_contract_self_check().values()))

    def test_shape_loss_and_roundtrip(self) -> None:
        self.assertGreater(float(loss_weights("shape_focus")[1]), float(loss_weights("shape_focus")[0]))
        import numpy as np
        calls = np.asarray([[1.0] * 12, [0.0, 2.0] + [0.0] * 10])
        bytes_ = calls * 4096.0
        decoded_calls, decoded_bytes = decode_histograms(encode_histograms(calls, bytes_))
        self.assertTrue(np.allclose(decoded_calls, calls, rtol=1e-6, atol=1e-3))
        self.assertTrue(np.allclose(decoded_bytes, bytes_, rtol=1e-6, atol=1e-2))


if __name__ == "__main__":
    unittest.main()
