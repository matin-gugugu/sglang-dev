#!/usr/bin/env python3
"""Phase37 nvidia-smi拓扑解析的纯CPU回归测试。"""

from __future__ import annotations

import unittest

from preflight import parse_topology


def eight_gpu_nv18_topology(*, ansi_header: bool) -> str:
    gpu_names = [f"GPU{index}" for index in range(8)]
    header = "\t" + "\t".join([*gpu_names, "NIC0", "CPU Affinity", "GPU NUMA ID"])
    if ansi_header:
        header = "\t\x1b[4m" + "\t".join([*gpu_names, "NIC0", "CPU Affinity", "GPU NUMA ID"]) + "\x1b[0m"
    rows = []
    for left, name in enumerate(gpu_names):
        gpu_links = ["X" if left == right else "NV18" for right in range(8)]
        rows.append("\t".join([name, *gpu_links, "PXB", "0-127", "0"]))
    return "\n".join([header, *rows]) + "\n"


class TopologyParserTest(unittest.TestCase):
    def assert_eight_gpu_nv18(self, text: str) -> None:
        parsed = parse_topology(text)
        self.assertEqual(parsed["gpu_columns"], [f"GPU{index}" for index in range(8)])
        self.assertEqual(set(parsed["pairs_by_category"]), {"NVLINK_NV18"})
        pairs = parsed["pairs_by_category"]["NVLINK_NV18"]
        self.assertEqual(len(pairs), 28)
        self.assertEqual(pairs[0], {"gpus": [0, 1], "raw_link": "NV18"})
        self.assertEqual(pairs[-1], {"gpus": [6, 7], "raw_link": "NV18"})
        self.assertEqual(parsed["raw"], text)

    def test_plain_header(self) -> None:
        self.assert_eight_gpu_nv18(eight_gpu_nv18_topology(ansi_header=False))

    def test_sgr_wrapped_header_from_nvidia_smi_580(self) -> None:
        self.assert_eight_gpu_nv18(eight_gpu_nv18_topology(ansi_header=True))


if __name__ == "__main__":
    unittest.main()
