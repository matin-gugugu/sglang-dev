#!/usr/bin/env python3
"""Acquire the five immutable Phase47 snapshots before formal offline preflight.

This is the only Phase47 program allowed to use the network.  Hugging Face
credentials are read by huggingface_hub from its normal environment/config;
they are never accepted as a command-line argument or written to output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contracts import inspect_model, model_specs, sha256


def safe_destination(root: Path, model_id: str) -> Path:
    root = root.expanduser().resolve()
    if str(root) in {"/", str(Path.home().resolve())}:
        raise RuntimeError("--model-root must be a dedicated persistent subdirectory")
    return root / model_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument(
        "--download",
        choices=("missing", "all", "none"),
        default="missing",
        help="network acquisition policy; use none for a read-only inventory",
    )
    parser.add_argument("--model-map-output", type=Path, required=True)
    parser.add_argument("--inventory-output", type=Path, required=True)
    args = parser.parse_args()
    model_root = args.model_root.expanduser().resolve()
    outputs = [args.model_map_output.expanduser().resolve(), args.inventory_output.expanduser().resolve()]
    git_root = Path(__file__).resolve().parents[3]
    if model_root == git_root or git_root in model_root.parents:
        raise RuntimeError("--model-root must be outside Git")
    if any(path == git_root or git_root in path.parents for path in outputs):
        raise RuntimeError("model map and acquisition inventory must remain outside Git")
    if any(model_root == path or model_root in path.parents for path in outputs):
        raise RuntimeError("map/inventory outputs must be outside model directories")
    model_root.mkdir(parents=True, exist_ok=True)
    if args.download != "none":
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError("huggingface_hub is required only for model acquisition") from error
    mapping = {}
    inventory = []
    for spec in model_specs():
        destination = safe_destination(model_root, spec["model_id"])
        should_download = args.download == "all" or (
            args.download == "missing" and not (destination / ".phase47_source.json").is_file()
        )
        if should_download:
            if destination.exists() and any(destination.iterdir()):
                raise RuntimeError(
                    f"refuse to mix a download into a non-empty unmarked directory: {destination}"
                )
            destination.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=spec["repo_id"],
                revision=spec["revision"],
                local_dir=str(destination),
                token=True if spec["gated"] else None,
            )
            (destination / ".phase47_source.json").write_text(
                json.dumps(
                    {
                        "schema_version": "phase47-local-model-source-v1",
                        "model_id": spec["model_id"],
                        "repo_id": spec["repo_id"],
                        "revision": spec["revision"],
                        "source_url": spec["source_url"],
                        "config_sha256": sha256(destination / "config.json"),
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        mapping[spec["model_id"]] = str(destination)
        inventory.append(inspect_model(spec["model_id"], destination, hash_weights=True))
    model_map = {"schema_version": "phase47-external-model-map-v1", "models": mapping}
    acquisition = {
        "schema_version": "phase47-external-model-acquisition-v1",
        "network_step": args.download,
        "credentials_recorded": False,
        "models": inventory,
    }
    for path, value in zip(outputs, (model_map, acquisition)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "models": len(mapping), "model_map": str(outputs[0]), "inventory": str(outputs[1])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
