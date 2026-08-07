"""CPU-only tests for histogram-only PP communication profiling."""

import json
import os
import tempfile
import unittest
from importlib import reload
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.distributed import pp_comm_profile
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPPCommProfile(CustomTestCase):
    def test_sender_side_histogram_snapshot(self):
        with tempfile.TemporaryDirectory() as output_dir, patch.dict(
            os.environ,
            {
                "SGLANG_PP_COMM_PROFILE_DIR": output_dir,
                "SGLANG_PP_COMM_PROFILE_RUN_ID": "unit-test",
                "SGLANG_PP_COMM_PROFILE_FLUSH_INTERVAL": "1",
            },
        ):
            module = reload(pp_comm_profile)
            batch = SimpleNamespace(
                forward_mode=SimpleNamespace(name="DECODE"),
                reqs=[object(), object(), object()],
                input_ids=torch.ones(3, dtype=torch.int64),
                forward_iter=7,
            )
            tensor = torch.ones((3, 4), dtype=torch.float16)

            for _ in range(2):
                module.record_send(
                    {"hidden_states": tensor},
                    msg_type="proxy",
                    pp_rank=1,
                    pp_size=4,
                    batch=batch,
                )
            module.flush()

            files = list(Path(output_dir).glob("*.json"))
            self.assertEqual(len(files), 1)
            snapshot = json.loads(files[0].read_text(encoding="utf-8"))
            self.assertEqual(snapshot["capture_mode"], "histogram-only")
            self.assertFalse(snapshot["raw_events_saved"])
            self.assertEqual(snapshot["events_total"], 2)
            self.assertEqual(snapshot["pp_rank"], 1)
            self.assertEqual(snapshot["pp_size"], 4)

            self.assertEqual(len(snapshot["histograms"]), 1)
            row = snapshot["histograms"][0]
            self.assertEqual(row["phase"], "decode")
            self.assertEqual(row["boundary"], "pp1->pp2")
            self.assertEqual(row["active_batch_size"], 3)
            self.assertEqual(row["active_tokens"], 3)
            self.assertEqual(row["payload_bytes"], 24)
            self.assertEqual(row["count"], 2)
            self.assertEqual(row["logical_bytes"], 48)


if __name__ == "__main__":
    unittest.main()
