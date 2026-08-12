#!/usr/bin/env python3
"""Export the frozen Phase 27C enhanced checkpoint for pure NumPy inference."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch


STATE_KEYS = (
    "network.0.weight",
    "network.0.bias",
    "network.2.weight",
    "network.2.bias",
    "network.4.weight",
    "network.4.bias",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root
        / "experiment-results/phase27c_pp_scheduler_feature_training/checkpoints/pp_enhanced_bounded_residual.pt",
    )
    parser.add_argument(
        "--phase27c-audit",
        type=Path,
        default=root
        / "experiment-results/phase27c_pp_scheduler_feature_training/audit_summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root
        / "experiment-results/phase28b_frozen_predictions/checkpoint/pp_enhanced_bounded_residual_numpy.npz",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=root
        / "experiment-results/phase28b_frozen_predictions/checkpoint/export_audit.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())


def main() -> None:
    args = parse_args()
    audit27c = json.loads(args.phase27c_audit.read_text())
    expected_hash = audit27c["checkpoint_sha256"]["enhanced_bounded_residual"]
    actual_hash = sha256(args.checkpoint)
    if actual_hash != expected_hash:
        raise RuntimeError("Phase 27C checkpoint hash mismatch")
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint["method"] != "enhanced_bounded_residual":
        raise RuntimeError(checkpoint["method"])
    if checkpoint["architecture"] != {
        "hidden_sizes": [64, 64],
        "activation": "relu",
        "bounded_tanh": True,
    }:
        raise RuntimeError(checkpoint["architecture"])
    if tuple(checkpoint["model_state"]) != STATE_KEYS:
        raise RuntimeError(tuple(checkpoint["model_state"]))

    arrays: dict[str, np.ndarray] = {
        "feature_names": np.asarray(checkpoint["feature_names"]),
        "log_feature_names": np.asarray(checkpoint["log_feature_names"]),
        "feature_mean": checkpoint["feature_mean"].numpy(),
        "feature_std": checkpoint["feature_std"].numpy(),
        "target_mean": checkpoint["target_mean"].numpy(),
        "target_std_or_residual_bounds": checkpoint[
            "target_std_or_residual_bounds"
        ].numpy(),
    }
    for key in STATE_KEYS:
        arrays[key] = checkpoint["model_state"][key].numpy()
    deterministic_npz(args.output, arrays)

    exported = np.load(args.output, allow_pickle=False)
    export_checks = {
        "feature_columns_108": len(exported["feature_names"]) == 108,
        "log_feature_names_match": set(exported["log_feature_names"].tolist())
        == set(checkpoint["log_feature_names"]),
        "feature_scalers_match": np.array_equal(
            exported["feature_mean"], checkpoint["feature_mean"].numpy()
        )
        and np.array_equal(exported["feature_std"], checkpoint["feature_std"].numpy()),
        "target_scalers_match": np.array_equal(
            exported["target_mean"], checkpoint["target_mean"].numpy()
        )
        and np.array_equal(
            exported["target_std_or_residual_bounds"],
            checkpoint["target_std_or_residual_bounds"].numpy(),
        ),
        "six_state_arrays_match": all(
            np.array_equal(exported[key], checkpoint["model_state"][key].numpy())
            for key in STATE_KEYS
        ),
        "no_pickle_arrays": all(exported[name].dtype != object for name in exported.files),
    }
    status = "PASS" if all(export_checks.values()) else "FAIL"
    result = {
        "schema_version": "phase28b-numpy-checkpoint-export-v1",
        "status": status,
        "method": checkpoint["method"],
        "source_checkpoint_sha256": actual_hash,
        "numpy_checkpoint_sha256": sha256(args.output),
        "feature_columns": len(checkpoint["feature_names"]),
        "state_array_count": len(STATE_KEYS),
        "checks": export_checks,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    if status != "PASS":
        raise RuntimeError(result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
