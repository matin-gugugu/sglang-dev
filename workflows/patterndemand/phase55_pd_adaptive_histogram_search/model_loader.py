#!/usr/bin/env python3
"""Minimal gzip CSV loader kept local to Phase55 preflight."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))
