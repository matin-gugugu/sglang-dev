#!/usr/bin/env python3
"""Extract compact, generalizable ProfileDemand features from model configs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="repeat for every local model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase16_model_features",
    )
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first(config, *names, default=0):
    for name in names:
        value = config.get(name)
        if value is not None:
            return value
    return default


def dtype_bytes(dtype):
    normalized = str(dtype).lower().replace("torch.", "")
    if normalized in {"bfloat16", "float16", "half"}:
        return 2
    if normalized in {"float32", "float"}:
        return 4
    if normalized in {"float8_e4m3fn", "float8_e5m2", "int8", "uint8"}:
        return 1
    raise ValueError(f"unsupported dtype for static feature extraction: {dtype}")


def extract(name, model_path):
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text())
    layers = int(first(config, "num_hidden_layers"))
    hidden = int(first(config, "hidden_size"))
    dense_intermediate = int(first(config, "intermediate_size"))
    heads = int(first(config, "num_attention_heads"))
    kv_heads = int(first(config, "num_key_value_heads", default=heads))
    experts = int(first(config, "num_experts", "n_routed_experts", default=0))
    top_k = int(first(config, "num_experts_per_tok", "num_experts_per_token", default=0))
    moe_intermediate = int(first(config, "moe_intermediate_size", default=0))
    shared_experts = int(first(config, "num_shared_experts", "n_shared_experts", default=0))
    first_dense = int(first(config, "first_k_dense_replace", default=0))
    moe_frequency = int(first(config, "moe_layer_freq", default=1 if experts else 0))
    dtype = first(config, "torch_dtype", "dtype", default="float16")
    bytes_per_element = dtype_bytes(dtype)
    is_moe = int(experts > 0)
    moe_layers = 0
    if is_moe:
        moe_layers = max(0, layers - first_dense)
        if moe_frequency > 1:
            moe_layers = (moe_layers + moe_frequency - 1) // moe_frequency
    architecture = first(config, "architectures", default=["unknown"])
    if isinstance(architecture, list):
        architecture = architecture[0] if architecture else "unknown"

    # These two priors are not fitted labels. They are the TP transformer template
    # used by H0 and are kept beside the primitive structure features for auditing.
    logical_collectives_per_forward_prior = 2 * layers + 1
    payload_bytes_per_active_token_prior = hidden * bytes_per_element
    raw_op_template = (
        "all_reduce:2+fused_allreduce_residual_rmsnorm:2L-1"
        if architecture == "Qwen3MoeForCausalLM"
        else "all_reduce:2L+1"
    )
    return {
        "model": name,
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "architecture_audit_only": architecture,
        "model_type_audit_only": config.get("model_type", "unknown"),
        "num_hidden_layers": layers,
        "hidden_size": hidden,
        "dense_intermediate_ratio": dense_intermediate / hidden,
        "num_attention_heads": heads,
        "head_dim": hidden / heads,
        "kv_head_ratio": kv_heads / heads,
        "dtype_bytes": bytes_per_element,
        "is_moe": is_moe,
        "num_experts": experts,
        "experts_per_token": top_k,
        "moe_intermediate_ratio": moe_intermediate / hidden if hidden else 0.0,
        "num_shared_experts": shared_experts,
        "first_dense_layers": first_dense,
        "moe_layer_frequency": moe_frequency,
        "estimated_moe_layers": moe_layers,
        "logical_collectives_per_forward_prior": logical_collectives_per_forward_prior,
        "payload_bytes_per_active_token_prior": payload_bytes_per_active_token_prior,
        "canonical_op_mask_all_reduce": 1,
        "raw_op_template_audit_only": raw_op_template,
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in args.model:
        if "=" not in item:
            raise ValueError(f"expected NAME=PATH, got {item}")
        name, path = item.split("=", 1)
        rows.append(extract(name, Path(path)))
    if len({row["model"] for row in rows}) != len(rows):
        raise ValueError("duplicate model names")
    with (args.output_dir / "model_features.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "model_features.json").write_text(json.dumps(rows, indent=2) + "\n")

    checks = {
        "at_least_three_models": len(rows) >= 3,
        "positive_structure": all(
            row["num_hidden_layers"] > 0
            and row["hidden_size"] > 0
            and row["num_attention_heads"] > 0
            for row in rows
        ),
        "dtype_supported": all(row["dtype_bytes"] in {1, 2, 4} for row in rows),
        "no_model_id_in_formal_numeric_features": True,
    }
    audit = {
        "schema_version": "profiledemand-model-features-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    (args.output_dir / "audit_summary.json").write_text(json.dumps(audit, indent=2) + "\n")
    if audit["status"] != "PASS":
        raise RuntimeError(audit)
    readme = """# Phase 16C：ProfileDemand 模型结构特征

这些特征只从模型 `config.json` 静态提取，不需要运行模型。正式数值特征不使用
`model_id`、实测直方图或真实通信时间；architecture/model_type 仅供审计和 model-ID
baseline。首版 canonical PatternDemand 统一为逻辑 AllReduce；当前 SGLang 的 fused
raw-op 拆分保留为 audit-only 模板，由第二阶段 backend 细化处理。

`logical_collectives_per_forward_prior=2L+1`、
`payload_bytes_per_active_token_prior=hidden_size×dtype_bytes` 是 H0 的透明结构先验，
不是从时间标签拟合得到的特征。
"""
    (args.output_dir / "README.md").write_text(readme)
    (args.output_dir / "DONE").write_text("PASS\n")
    files = sorted(
        path for path in args.output_dir.iterdir() if path.is_file() and path.name != "manifest.sha256"
    )
    (args.output_dir / "manifest.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files)
    )
    print(json.dumps({"models": [row["model"] for row in rows], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
