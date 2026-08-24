#!/usr/bin/env python3
"""Execute the pinned Phase60 production benchmark with Phase63 external-model contracts."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
P60_BENCHMARK = HERE.parent / "phase60_pd_multi_endpoint_composability/benchmark_mooncake_multi.py"
sys.path.insert(0, str(HERE))

import contracts as phase63_contracts  # noqa: E402


def main() -> None:
    sys.modules["contracts"] = phase63_contracts
    spec = importlib.util.spec_from_file_location("phase63_pinned_phase60_benchmark", P60_BENCHMARK)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned Phase60 production benchmark")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
