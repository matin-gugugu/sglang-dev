#!/usr/bin/env python3
"""PatternDemand跨环境workflow共用的只读审计和结果归档工具。"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def run_git(arguments: Iterable[str], *, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and completed.returncode:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return completed.stdout.strip()


def require_expected_head(expected: str) -> str:
    actual = run_git(["rev-parse", "HEAD"])
    resolved = run_git(["rev-parse", expected])
    if actual != resolved:
        raise RuntimeError(f"HEAD不等于workflow commit：actual={actual}, expected={resolved}")
    return actual


def require_clean_before_run(allowed_untracked_prefixes: Iterable[str] = ("data/",)) -> None:
    tracked = run_git(["status", "--porcelain", "--untracked-files=no"])
    if tracked:
        raise RuntimeError(f"存在已跟踪文件改动，拒绝运行：\n{tracked}")
    allowed = tuple(allowed_untracked_prefixes)
    untracked = run_git(["ls-files", "--others", "--exclude-standard"])
    unexpected = [
        path
        for path in untracked.splitlines()
        if path and not any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in allowed)
    ]
    if unexpected:
        raise RuntimeError(f"存在非允许的未跟踪文件，拒绝运行：{unexpected}")


def verify_manifest(directory: Path) -> dict[str, Any]:
    manifest = directory / "manifest.sha256"
    if not manifest.is_file():
        return {"ok": False, "error": "manifest_missing", "directory": str(directory)}
    checked = 0
    errors = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            errors.append({"line": line, "error": "invalid_manifest_line"})
            continue
        path = directory / relative
        if not path.is_file():
            errors.append({"path": relative, "error": "missing"})
        elif sha256(path) != expected:
            errors.append({"path": relative, "error": "sha256_mismatch"})
        checked += 1
    try:
        display_directory = str(directory.relative_to(repo_root()))
    except ValueError:
        display_directory = str(directory)
    return {
        "ok": not errors,
        "directory": display_directory,
        "manifest_sha256": sha256(manifest),
        "checked_files": checked,
        "errors": errors,
    }


def verify_pinned_inputs(spec: dict[str, Any]) -> dict[str, Any]:
    audits = {}
    for item in spec.get("pinned_inputs", []):
        path = repo_root() / item["path"]
        actual = sha256(path) if path.is_file() else None
        audits[item["name"]] = {
            "path": item["path"],
            "expected_sha256": item["sha256"],
            "actual_sha256": actual,
            "ok": actual == item["sha256"],
        }
        if item.get("verify_manifest_directory"):
            manifest_audit = verify_manifest(repo_root() / item["verify_manifest_directory"])
            audits[item["name"]]["manifest_audit"] = manifest_audit
            audits[item["name"]]["ok"] &= manifest_audit["ok"]
    failed = [name for name, value in audits.items() if not value["ok"]]
    if failed:
        raise RuntimeError(f"冻结输入校验失败：{failed}")
    return audits


def environment_record() -> dict[str, Any]:
    return {
        "captured_at_utc": utc_now(),
        "repository_head": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["branch", "--show-current"], check=False),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def refresh_manifest(directory: Path) -> None:
    rows = [
        f"{sha256(path)}  {path.relative_to(directory)}"
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "manifest.sha256"
    ]
    (directory / "manifest.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def validate_result_tree(directory: Path) -> dict[str, Any]:
    forbidden_names = {"PID", "pid", "core", "core.dump"}
    forbidden_suffixes = {".pid", ".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".jsonl"}
    forbidden_parts = {"__pycache__", "cache", "raw_samples", "raw_trace", "model_weights"}
    violations = []
    files = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(directory)
        files.append(str(relative))
        if (
            path.name in forbidden_names
            or path.suffix.lower() in forbidden_suffixes
            or any(part.lower() in forbidden_parts for part in relative.parts)
        ):
            violations.append(str(relative))
    return {
        "ok": not violations,
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in directory.rglob("*") if path.is_file()),
        "violations": violations,
    }


def verify_result_manifest(directory: Path) -> dict[str, Any]:
    audit = verify_manifest(directory)
    tree = validate_result_tree(directory)
    manifest_paths = set()
    manifest = directory / "manifest.sha256"
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line:
                _expected, relative = line.split("  ", 1)
                manifest_paths.add(relative)
    actual_paths = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and path.name != "manifest.sha256"
    }
    completeness = {
        "ok": manifest_paths == actual_paths,
        "missing_from_manifest": sorted(actual_paths - manifest_paths),
        "missing_from_tree": sorted(manifest_paths - actual_paths),
    }
    return {"ok": audit["ok"] and tree["ok"] and completeness["ok"], "manifest": audit, "tree": tree, "completeness": completeness}


def write_blocked(output_dir: Path, phase: str, reason: str, evidence: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "BLOCKED.json",
        {
            "schema_version": "patterndemand-workflow-blocked-v1",
            "phase": phase,
            "status": "BLOCKED",
            "reason": reason,
            "evidence": evidence,
            "created_at_utc": utc_now(),
        },
    )
    (output_dir / "README.md").write_text(
        f"# {phase}：执行阻塞\n\n本次没有生成正式实验结果。阻塞原因：{reason}\n",
        encoding="utf-8",
    )
    refresh_manifest(output_dir)


def ensure_external_raw_dir(raw_dir: Path) -> Path:
    resolved = raw_dir.expanduser().resolve()
    root = repo_root().resolve()
    if resolved == root or root in resolved.parents:
        raise RuntimeError("raw目录必须位于Git仓库之外")
    if resolved.exists() and any(resolved.iterdir()):
        raise RuntimeError(f"raw目录已存在且非空，拒绝覆盖：{resolved}")
    return resolved


def validate_staged_allowlist(allowed_prefix: str) -> dict[str, Any]:
    staged = run_git(["diff", "--cached", "--name-only"])
    paths = [path for path in staged.splitlines() if path]
    invalid = [path for path in paths if not path.startswith(allowed_prefix.rstrip("/") + "/")]
    forbidden = [
        path
        for path in paths
        if Path(path).suffix.lower() in {".pid", ".pt", ".pth", ".ckpt", ".safetensors", ".jsonl"}
        or any(part.lower() in {"data", "raw_samples", "raw_trace", "cache"} for part in Path(path).parts)
    ]
    return {"ok": bool(paths) and not invalid and not forbidden, "paths": paths, "invalid": invalid, "forbidden": forbidden}
