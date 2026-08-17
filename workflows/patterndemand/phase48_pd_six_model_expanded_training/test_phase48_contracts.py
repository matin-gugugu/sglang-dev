#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(HERE.parent))
from common import load_json, verify_pinned_inputs  # noqa: E402

_SPEC=importlib.util.spec_from_file_location("phase48_contracts",HERE/"contracts.py")
if _SPEC is None or _SPEC.loader is None: raise RuntimeError("cannot load Phase48 contracts")
_P48=importlib.util.module_from_spec(_SPEC); _SPEC.loader.exec_module(_P48)


class Phase48Contracts(unittest.TestCase):
    def test_parent_pins_and_models(self)->None:
        contract=load_json(HERE/"experiment.json")
        self.assertEqual(contract["workflow_parent_result_commit"],"9de1816e912056a0a0b2b91d940079540ad6454a")
        self.assertTrue(all(value["ok"] for value in verify_pinned_inputs(contract).values()))
        self.assertEqual(len(_P48.load_models()),6); self.assertTrue(all(_P48.contract_self_check().values()))
    def test_group_and_blind_contract(self)->None:
        contract=load_json(HERE/"experiment.json")
        self.assertIn("all six model rows",contract["dataset_contract"]["group_isolation"])
        self.assertFalse(contract["execution"]["phase45_or_phase46_target_access_permitted"])
        self.assertIn("every model strictly beat H0",contract["hard_acceptance_gate"]["oof"])


if __name__=="__main__": unittest.main()
