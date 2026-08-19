#!/usr/bin/env python3
"""Small local CSV readers for the Phase56 control-side workflow."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))
