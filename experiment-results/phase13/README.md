# Phase 13：三模型 PatternDemand 与多支撑点时间验证

## 1. 阶段划分

Phase 13 分为两个连续子阶段：

1. Phase 13A：联合 Qwen3-8B、DeepSeek-V2-Lite 和 Qwen3-30B-A3B 的
   TP2/4/8 规则网格，分析模型结构、workload、TP group size 和 raw op 如何决定
   PatternDemand；
2. Phase 13B：在 TP=2 上为 Qwen3-30B-A3B 补齐 Phase 10/11 的 mixed Decode
   与 chunked Prefill 多支撑点 all-rank 时间标签，并重新执行三模型时间预测消融。

Phase 13A 已完成。Phase 13B 当前进入执行阶段。

## 2. Phase 13A

正式目录：

```text
experiment-results/phase13/three_model_pattern_analysis/
```

Phase 13A 聚合三模型各 195 个独立配置，共 585 个 model × workload 配置。
精确直方图使用 `(raw_op, payload) -> count`，解析 PatternDemand 对当前规则网格
的逐 raw-op 直方图重建失败数为 0。

该结果只覆盖规则网格通信结构，不包含第三模型的真实 collective 时间，也不能
外推为 mixed Decode、chunked Prefill、EP All-to-All 或未知 runtime lowering。

## 3. Phase 13B 实验设计

模型与环境：

| 项目 | 设置 |
|---|---|
| 模型 | Qwen3-30B-A3B |
| 并行配置 | 单节点 B200，TP=2 |
| repeats | 3 |
| mixed Decode | 3 profiles × 3 repeats = 9 条 |
| chunked Prefill | 3 chunk sizes × 12 workloads × 3 repeats = 108 条 |
| 第三模型新增标签 | 117 条，聚合为 39 个配置 |
| 三模型合计 | 351 条标签，聚合为 117 个配置 |

主时间标签沿用 Phase 11：

```text
sum over aligned collectives of
(max rank kernel end - max rank kernel start)
```

即 all-rank post-rendezvous completion kernel time。三重复先按完整配置聚合，
不得跨 split。

Qwen3-30B-A3B 标签必须保存：

- `(raw_op, input_payload_bytes) -> calls`；
- `collective_family`；
- all-rank kernel、backend sequence 和 PatternDemand 对齐结果；
- intrinsic、post-rendezvous 和 sync-inclusive 三种时间口径。

raw profiler trace 仅用于 compact label 抽取。validator 通过后删除大型 trace，并
保存 `TRACES_REMOVED`，Git 只提交 compact label、日志、telemetry 和校验产物。

## 4. 执行

完整执行包含 smoke、正式采集和三模型分析：

```bash
nohup bash scripts/run_qwen3_30b_a3b_multiscale_timing_dataset.sh all \
  > experiment-results/phase13/phase13b_driver.log 2>&1 &
```

也可以分别执行：

```bash
bash scripts/run_qwen3_30b_a3b_multiscale_timing_dataset.sh smoke
bash scripts/run_qwen3_30b_a3b_multiscale_timing_dataset.sh formal
bash scripts/run_qwen3_30b_a3b_multiscale_timing_dataset.sh analyze
```

输出目录：

```text
experiment-results/phase13/
├── qwen3_30b_a3b_multiscale_timing_smoke/
├── multiscale_timing_ground_truth/
│   └── qwen3-30b-a3b/
├── three_model_multiscale_timing_analysis/
└── phase13b_driver.log
```

## 5. 完成判据

- smoke 的 mixed Decode 与 chunked Prefill 均通过 op-aware validator；
- 18/18 正式实验单元生成 `DONE`；
- 117/117 Qwen3-30B-A3B compact labels；
- 每个聚合配置恰好 3 次重复；
- 所有 rank kernel count、backend sequence、PatternDemand 完全对齐；
- raw-op 直方图边缘化后与 payload-only 直方图一致；
- 没有 OOM、timeout、Traceback、CPU fallback、NCCL error 或 rank mismatch；
- 三模型分析按完整 workload 分组，不发生 repeat 泄漏；
- README、manifest、compact 结果、分析图表与复验日志提交并 push。

## 6. 结论边界

Phase 13B 完成前，不能声称第三模型的多支撑点真实时间或三模型时间留出已经验证。
完成后也只能在单节点 B200、TP=2 和当前 mixed/chunked workload 范围内陈述结论；
不能外推到 TP4/8 时间、L2/L3、PP、PD 或 expert-parallel All-to-All。
