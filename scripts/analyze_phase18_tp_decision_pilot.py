#!/usr/bin/env python3
"""Phase 18: offline TP/strategy decision pilot.

This analysis deliberately reuses the measured Phase 16 wall-clock records and
the Phase 17 communication-cost replay.  The measured L1 wall time is split into
an invariant non-communication proxy plus a representation-specific
communication estimate.  L2/L3 remain parameterized sensitivity scenarios.

This is a decision-utility pilot, not an online queueing or production scheduler
claim.  All raw Phase 16 JSONL files remain outside Git; only compact labels and
derived metrics are archived.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


REPRESENTATIONS = (
    "total_bytes_data_only",
    "onebin_calls_bytes",
    "threebin_calls_bytes",
    "twelvebin_exact",
    "h0_predicted_12bin",
    "residual_predicted_12bin",
    "exact_payload_oracle",
)

OBJECTIVES = {
    "latency": lambda total_us, tp: total_us,
    "balanced": lambda total_us, tp: total_us * math.sqrt(tp),
    "gpu_efficiency": lambda total_us, tp: total_us * tp,
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=root / "experiment-results/phase16_profiledemand_gpu/full",
    )
    parser.add_argument(
        "--phase17-costs",
        type=Path,
        default=root
        / "experiment-results/phase17_parameterized_topology/cost_predictions.csv.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase18_tp_decision_pilot",
    )
    return parser.parse_args()


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def read_phase16_raw(raw_root: Path):
    compact = []
    for path in sorted(raw_root.glob("*/tp*/r0/result.jsonl")):
        model = path.parents[2].name
        tp = int(path.parents[1].name.removeprefix("tp"))
        with path.open() as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                plan = row["trace_replay_plan"]
                input_lens = [int(value) for value in plan["input_lens_per_request"]]
                output_lens = [int(value) for value in plan["output_lens_per_request"]]
                generated = [int(value) for value in row["generated_output_tokens_per_request"]]
                if generated != output_lens:
                    raise ValueError(f"actual output mismatch: {path}:{line_number}")
                arrivals = [float(value) for value in plan["arrival_offsets_ms_audit_only"]]
                if not (len(input_lens) == len(output_lens) == len(arrivals)):
                    raise ValueError(f"request vector mismatch: {path}:{line_number}")
                prefill_s = float(row["prefill_latency"])
                total_s = float(row["total_latency"])
                decode_s = max(total_s - prefill_s, 0.0)
                compact.append(
                    {
                        "model": model,
                        "tp": tp,
                        "profile_id": plan["profile_id"],
                        "source": plan["source"],
                        "segment": plan["segment"],
                        "split": plan["split"],
                        "strategy": plan["strategy"],
                        "batch_index": int(plan["batch_index"]),
                        "request_count": len(input_lens),
                        "input_tokens": sum(input_lens),
                        "actual_output_tokens": sum(output_lens),
                        "max_actual_output_len": max(output_lens),
                        "arrival_min_ms": min(arrivals),
                        "arrival_max_ms": max(arrivals),
                        "prefill_wall_us": prefill_s * 1e6,
                        "decode_wall_us": decode_s * 1e6,
                        "total_wall_us": total_s * 1e6,
                        "input_lens_json": json.dumps(input_lens, separators=(",", ":")),
                        "actual_output_lens_json": json.dumps(
                            output_lens, separators=(",", ":")
                        ),
                        "arrival_offsets_ms_json": json.dumps(
                            arrivals, separators=(",", ":")
                        ),
                        "raw_source": str(path.relative_to(raw_root.parent.parent)),
                    }
                )
    return compact


def aggregate_timing(compact):
    groups = defaultdict(list)
    for row in compact:
        key = (row["model"], row["tp"], row["profile_id"], row["strategy"])
        groups[key].append(row)
    aggregates = []
    for (model, tp, profile_id, strategy), rows in sorted(groups.items()):
        rows.sort(key=lambda row: row["batch_index"])
        requests = sum(row["request_count"] for row in rows)
        normalization = 1000.0 / requests
        # This replay is a deterministic draining-batch service-work proxy.  The
        # trace timestamps are retained for future queueing work but are not
        # treated as an online scheduling ground truth here.
        aggregates.append(
            {
                "model": model,
                "tp": tp,
                "profile_id": profile_id,
                "source": rows[0]["source"],
                "segment": rows[0]["segment"],
                "split": rows[0]["split"],
                "strategy": strategy,
                "batch_count": len(rows),
                "request_count": requests,
                "input_tokens": sum(row["input_tokens"] for row in rows),
                "actual_output_tokens": sum(row["actual_output_tokens"] for row in rows),
                "prefill_wall_us_per_1000": sum(row["prefill_wall_us"] for row in rows)
                * normalization,
                "decode_wall_us_per_1000": sum(row["decode_wall_us"] for row in rows)
                * normalization,
                "total_wall_us_per_1000": sum(row["total_wall_us"] for row in rows)
                * normalization,
                "arrival_span_ms": max(row["arrival_max_ms"] for row in rows)
                - min(row["arrival_min_ms"] for row in rows),
            }
        )
    return aggregates


def read_costs(path: Path):
    rows = {}
    exact_l1 = {}
    with gzip.open(path, "rt", newline="") as source:
        for row in csv.DictReader(source):
            if row["phase"] != "combined":
                continue
            key = (
                row["evaluation"],
                row["model"],
                int(row["tp"]),
                row["profile_id"],
                row["strategy"],
                row["curve_id"],
                row["representation"],
            )
            if key in rows:
                raise ValueError(f"duplicate Phase 17 cost key: {key}")
            rows[key] = {
                "topology": row["topology"],
                "scenario": row["scenario"],
                "curve_kind": row["curve_kind"],
                "oracle_comm_us_per_1000": float(row["oracle_cost_us_per_1000"]),
                "estimated_comm_us_per_1000": float(row["estimated_cost_us_per_1000"]),
            }
            if row["curve_id"] == "l1_measured" and row["representation"] == "exact_payload_oracle":
                exact_key = (
                    row["evaluation"],
                    row["model"],
                    int(row["tp"]),
                    row["profile_id"],
                    row["strategy"],
                )
                exact_l1[exact_key] = float(row["oracle_cost_us_per_1000"])
    return rows, exact_l1


def build_candidate_scores(aggregates, costs, exact_l1):
    timing = {
        (row["model"], row["tp"], row["profile_id"], row["strategy"]): row
        for row in aggregates
    }
    evaluations = sorted({key[0] for key in costs})
    curves = sorted({key[5] for key in costs})
    candidate_rows = []
    negative_noncomm = 0
    for evaluation in evaluations:
        for identity, measured in timing.items():
            model, tp, profile_id, strategy = identity
            l1_key = (evaluation, model, tp, profile_id, strategy)
            measured_l1_comm = exact_l1[l1_key]
            raw_noncomm = measured["total_wall_us_per_1000"] - measured_l1_comm
            if raw_noncomm < 0:
                negative_noncomm += 1
            noncomm = max(raw_noncomm, 0.0)
            for curve_id in curves:
                for representation in REPRESENTATIONS:
                    cost = costs[
                        (
                            evaluation,
                            model,
                            tp,
                            profile_id,
                            strategy,
                            curve_id,
                            representation,
                        )
                    ]
                    oracle_total = noncomm + cost["oracle_comm_us_per_1000"]
                    estimated_total = noncomm + cost["estimated_comm_us_per_1000"]
                    for objective, transform in OBJECTIVES.items():
                        candidate_rows.append(
                            {
                                "evaluation": evaluation,
                                "model": model,
                                "profile_id": profile_id,
                                "segment": measured["segment"],
                                "strategy": strategy,
                                "tp": tp,
                                "curve_id": curve_id,
                                "topology": cost["topology"],
                                "scenario": cost["scenario"],
                                "curve_kind": cost["curve_kind"],
                                "objective": objective,
                                "representation": representation,
                                "measured_l1_wall_us_per_1000": measured[
                                    "total_wall_us_per_1000"
                                ],
                                "measured_l1_comm_us_per_1000": measured_l1_comm,
                                "noncomm_wall_proxy_us_per_1000": noncomm,
                                "oracle_comm_us_per_1000": cost[
                                    "oracle_comm_us_per_1000"
                                ],
                                "estimated_comm_us_per_1000": cost[
                                    "estimated_comm_us_per_1000"
                                ],
                                "oracle_total_us_per_1000": oracle_total,
                                "estimated_total_us_per_1000": estimated_total,
                                "oracle_score": transform(oracle_total, tp),
                                "estimated_score": transform(estimated_total, tp),
                            }
                        )
    return candidate_rows, negative_noncomm


def rank_agreement(rows):
    concordant = 0.0
    comparable = 0.0
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            oracle_delta = left["oracle_score"] - right["oracle_score"]
            estimated_delta = left["estimated_score"] - right["estimated_score"]
            if abs(oracle_delta) <= 1e-12:
                continue
            comparable += 1
            if oracle_delta * estimated_delta > 0:
                concordant += 1
            elif abs(estimated_delta) <= 1e-12:
                concordant += 0.5
    return concordant / comparable if comparable else 1.0


def make_decisions(candidate_rows):
    groups = defaultdict(list)
    for row in candidate_rows:
        key = (
            row["evaluation"],
            row["model"],
            row["profile_id"],
            row["curve_id"],
            row["objective"],
            row["representation"],
        )
        groups[key].append(row)
    decisions = []
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda row: (row["tp"], row["strategy"]))
        oracle = min(rows, key=lambda row: (row["oracle_score"], row["tp"], row["strategy"]))
        predicted = min(
            rows, key=lambda row: (row["estimated_score"], row["tp"], row["strategy"])
        )
        regret = (
            predicted["oracle_score"] - oracle["oracle_score"]
        ) / max(oracle["oracle_score"], 1e-12)
        decisions.append(
            {
                "evaluation": key[0],
                "model": key[1],
                "profile_id": key[2],
                "curve_id": key[3],
                "objective": key[4],
                "representation": key[5],
                "oracle_tp": oracle["tp"],
                "oracle_strategy": oracle["strategy"],
                "predicted_tp": predicted["tp"],
                "predicted_strategy": predicted["strategy"],
                "selection_correct": int(
                    oracle["tp"] == predicted["tp"]
                    and oracle["strategy"] == predicted["strategy"]
                ),
                "oracle_best_score": oracle["oracle_score"],
                "predicted_choice_oracle_score": predicted["oracle_score"],
                "regret": max(float(regret), 0.0),
                "rank_pair_agreement": rank_agreement(rows),
                "candidate_count": len(rows),
            }
        )
    return decisions


def summarize_decisions(decisions):
    groups = defaultdict(list)
    for row in decisions:
        key = (
            row["evaluation"],
            row["curve_id"],
            row["objective"],
            row["representation"],
        )
        groups[key].append(row)
    metrics = []
    for key, rows in sorted(groups.items()):
        regrets = [row["regret"] for row in rows]
        metrics.append(
            {
                "evaluation": key[0],
                "curve_id": key[1],
                "objective": key[2],
                "representation": key[3],
                "decision_count": len(rows),
                "selection_accuracy": float(
                    np.mean([row["selection_correct"] for row in rows])
                ),
                "mean_regret": float(np.mean(regrets)),
                "p95_regret": percentile(regrets, 95),
                "max_regret": max(regrets),
                "mean_rank_pair_agreement": float(
                    np.mean([row["rank_pair_agreement"] for row in rows])
                ),
            }
        )
    return metrics


def write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_gzip_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with gzip.open(path, "wt", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def manifest(output_dir: Path):
    lines = []
    for path in sorted(output_dir.iterdir()):
        # run.log is the outer stdout redirection target and may still be
        # buffered while this process is writing the manifest.
        if path.name in {"manifest.sha256", "DONE", "run.log"} or not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    (output_dir / "manifest.sha256").write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    compact = read_phase16_raw(args.raw_root)
    aggregates = aggregate_timing(compact)
    costs, exact_l1 = read_costs(args.phase17_costs)
    candidate_rows, negative_noncomm = build_candidate_scores(
        aggregates, costs, exact_l1
    )
    decisions = make_decisions(candidate_rows)
    metrics = summarize_decisions(decisions)

    write_gzip_csv(args.output_dir / "compact_timing_labels.csv.gz", compact)
    write_csv(args.output_dir / "timing_aggregates.csv", aggregates)
    write_gzip_csv(args.output_dir / "candidate_scores.csv.gz", candidate_rows)
    write_gzip_csv(args.output_dir / "decisions.csv.gz", decisions)
    write_csv(args.output_dir / "decision_metrics.csv", metrics)

    headline = [
        row
        for row in metrics
        if row["evaluation"] == "traffic_segment_holdout"
        and row["curve_id"] in {"l1_measured", "l2_nominal", "l3_nominal"}
        and row["objective"] in {"latency", "gpu_efficiency"}
        and row["representation"]
        in {
            "total_bytes_data_only",
            "onebin_calls_bytes",
            "twelvebin_exact",
            "h0_predicted_12bin",
            "residual_predicted_12bin",
            "exact_payload_oracle",
        }
    ]
    summary = {
        "schema_version": "phase18-tp-offline-decision-pilot-v1",
        "evidence_boundary": (
            "Measured Phase16 draining-batch wall times plus Phase17 communication "
            "replay. L2/L3 are parameterized; no online queueing or production "
            "scheduler claim."
        ),
        "raw_workload_rows": len(compact),
        "timing_candidate_rows": len(aggregates),
        "candidate_score_rows": len(candidate_rows),
        "decision_rows": len(decisions),
        "negative_noncomm_proxy_rows": negative_noncomm,
        "models": sorted({row["model"] for row in aggregates}),
        "tps": sorted({row["tp"] for row in aggregates}),
        "profiles": len({row["profile_id"] for row in aggregates}),
        "strategies": sorted({row["strategy"] for row in aggregates}),
        "objectives": list(OBJECTIVES),
        "headline_metrics": headline,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )

    audit = {
        "schema_version": "phase18-tp-offline-decision-pilot-audit-v1",
        "checks": {
            "raw_rows_4905": len(compact) == 4905,
            "aggregated_candidates_648": len(aggregates) == 648,
            "all_profiles_have_32_requests": all(
                row["request_count"] == 32 for row in aggregates
            ),
            "no_negative_noncomm_proxy": negative_noncomm == 0,
            "nine_candidates_per_decision": all(
                row["candidate_count"] == 9 for row in decisions
            ),
            "oracle_has_zero_regret": all(
                row["regret"] == 0.0
                for row in decisions
                if row["representation"] == "exact_payload_oracle"
            ),
            "all_scores_finite": all(
                math.isfinite(row["oracle_score"])
                and math.isfinite(row["estimated_score"])
                for row in candidate_rows
            ),
        },
    }
    audit["status"] = "PASS" if all(audit["checks"].values()) else "FAIL"
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n"
    )

    readme = [
        "# Phase 18：TP离线候选决策pilot",
        "",
        "本实验使用Phase16的4905条真实GPU墙钟记录和Phase17的通信代价传播，",
        "在每个`model×profile×topology`内枚举TP2/4/8与三档batching策略。",
        "L1使用实测曲线；L2/L3仍为参数化敏感性场景。",
        "",
        "候选总服务工作量定义为：",
        "",
        "`noncomm_proxy = measured_L1_wall - exact_L1_comm`",
        "",
        "`candidate_total = noncomm_proxy + representation_comm(topology)`",
        "",
        "它只隔离通信表征对候选排序的影响，不是online queueing或生产调度器。",
        "",
        "## 核心结果",
        "",
        "| topology | objective | representation | accuracy | mean regret | P95 regret | rank agreement |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in headline:
        readme.append(
            f"| {row['curve_id']} | {row['objective']} | {row['representation']} | "
            f"{100 * row['selection_accuracy']:.2f}% | {100 * row['mean_regret']:.3f}% | "
            f"{100 * row['p95_regret']:.3f}% | {100 * row['mean_rank_pair_agreement']:.2f}% |"
        )
    lookup = {
        (row["curve_id"], row["objective"], row["representation"]): row
        for row in headline
    }
    l1_total_bytes = lookup[
        ("l1_measured", "latency", "total_bytes_data_only")
    ]
    l1_h0 = lookup[("l1_measured", "latency", "h0_predicted_12bin")]
    readme.extend(
        [
            "",
            "## 自动解读",
            "",
            f"- L1 latency目标下，total-bytes选择准确率为"
            f"{100 * l1_total_bytes['selection_accuracy']:.2f}%，但平均regret仅"
            f"{100 * l1_total_bytes['mean_regret']:.3f}%；",
            f"- 同口径H0预测12桶的选择准确率为"
            f"{100 * l1_h0['selection_accuracy']:.2f}%，平均regret为"
            f"{100 * l1_h0['mean_regret']:.3f}%；",
            "- 与Phase17的communication-only结果相比，加入真实非通信墙钟后，当前候选"
            "排序主要由计算/运行时部分主导；通信表征仍改善排序，但端到端收益不能由通信"
            "侧regret直接替代；",
            "- 参数化L2/L3中选择差异依旧很小，说明若要形成更强的完整调度证据，需要真实"
            "高RTT链路、online queue/SLO约束或更接近决策边界的候选对照。",
            "",
            "## 证据边界",
            "",
            "- Phase16完整网格的时间标签只有一次正式回放，本结果属于pilot；",
            "- arrival offset被保存在紧凑标签中，但当前不宣称online continuous batching；",
            "- `balanced`和`gpu_efficiency`是显式资源加权目标，不是生产SLO；",
            "- 后续真实调度器还需要队列、显存、并发副本和资源可用性。",
            "",
        ]
    )
    (args.output_dir / "README.md").write_text("\n".join(readme))
    manifest(args.output_dir)
    (args.output_dir / "DONE").write_text("PASS\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
