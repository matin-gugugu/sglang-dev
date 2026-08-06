#!/usr/bin/env python3
"""Download fixed official BurstGPT v2.0 and Mooncake FAST'25 traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


SOURCES = [
    {
        "source": "burstgpt",
        "release": "v2.0",
        "name": "BurstGPT_without_fails_1.csv",
        "size": 51_429_517,
        "url": "https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_without_fails_1.csv",
    },
    {
        "source": "burstgpt",
        "release": "v2.0",
        "name": "BurstGPT_without_fails_2.csv",
        "size": 142_376_815,
        "url": "https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_without_fails_2.csv",
    },
    {
        "source": "burstgpt",
        "release": "v2.0",
        "name": "BurstGPT_without_fails_3.csv",
        "size": 217_312_026,
        "url": "https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_without_fails_3.csv",
    },
    {
        "source": "mooncake",
        "release": "FAST25-main-2026-08-06",
        "name": "conversation_trace.jsonl",
        "size": 3_029_533,
        "url": "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/conversation_trace.jsonl",
    },
    {
        "source": "mooncake",
        "release": "FAST25-main-2026-08-06",
        "name": "toolagent_trace.jsonl",
        "size": 4_415_857,
        "url": "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/toolagent_trace.jsonl",
    },
    {
        "source": "mooncake",
        "release": "FAST25-main-2026-08-06",
        "name": "synthetic_trace.jsonl",
        "size": 1_136_313,
        "url": "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/synthetic_trace.jsonl",
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(source, destination, timeout):
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        source["url"], headers={"User-Agent": "sglang-pattern-demand-research"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response, partial.open(
        "wb"
    ) as output:
        shutil.copyfileobj(response, output, 1024 * 1024)
    if partial.stat().st_size != source["size"]:
        raise RuntimeError(
            f"size mismatch for {source['name']}: {partial.stat().st_size} != {source['size']}"
        )
    partial.replace(destination)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for source in SOURCES:
        destination = args.output_dir / source["name"]
        if not destination.exists() or destination.stat().st_size != source["size"]:
            print(f"downloading {source['name']} ({source['size']} bytes)", flush=True)
            download(source, destination, args.timeout)
        record = {
            **source,
            "path": str(destination.resolve()),
            "actual_size": destination.stat().st_size,
            "sha256": sha256(destination),
        }
        records.append(record)
        print(f"verified {source['name']} {record['sha256']}", flush=True)
    manifest = {
        "schema_version": "phase15-public-trace-sources-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": records,
    }
    (args.output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
