# Phase 14：TP4/8 代表点时间泛化验证

状态：执行中

## 1. 目标

Phase 13B 已在单节点 B200、TP=2 上验证三模型多支撑点时间预测。Phase 14 不重跑
完整规则网格，而是选择 Qwen3-8B 与 Qwen3-30B-A3B 的代表 workload，补齐 TP=4/8
all-rank post-rendezvous 时间标签，检验：

1. logical calls 和 payload 直方图跨 TP 是否保持一致；
2. group size、equivalent rounds 和 backend transition 如何改变真实时间；
3. 仅在 TP2 上校准的 continuous histogram 能否零样本预测 TP4/8；
4. TP4/8 是否仍需要额外的 per-TP calibration。

## 2. 正式网格

硬件与模型：

| 维度 | 设置 |
|---|---|
| hardware | 单节点 8×B200 |
| models | Qwen3-8B、Qwen3-30B-A3B |
| TP | 4、8；TP2 复用 Phase 11/13B |
| repeats | 0、1、2 |
| mixed Decode | balanced、staircase、bimodal |
| chunk sizes | 1024、4096 |
| chunk boundary input lengths | T-1、T、T+1 |
| chunk batch sizes | 1、4 |

新增正式规模：

- mixed Decode：2 models × 2 TP × 3 profiles × 3 repeats = 36 标签；
- chunked Prefill：2 models × 2 TP × 2 chunk sizes × 3 repeats × 6 workloads
  = 144 标签；
- 合计 180 条新增标签，聚合为 60 个 TP4/8 配置；
- 联合匹配 TP2 后共 270 条标签、90 个聚合配置、30 组三点 TP scaling 对照。

这组设计保留了 mixed batch drain、chunk 边界，以及 `B=4,T≈1024` 与
`B=1,T≈4096` 的近等 active-token/total-payload 对照，但避免重跑完整网格。

## 3. 时间与结构口径

主目标沿用 Phase 11/13B：

```text
sum over aligned collectives of
(max rank kernel end - max rank kernel start)
```

即 all-rank post-rendezvous completion time。每个完整配置的三次重复先取中位数。

PatternDemand 必须保存 `(raw_op, payload) -> calls`，并分别保存 group size、
ring-equivalent bytes/rounds 和实际 backend sequence。continuous curve 使用 Phase 2
已有的 TP2/4/8 group-size-aware 曲线；它按 payload 边缘化计价，因此不会被描述为
专门的 fused-op 微基准。raw op 和 backend transition 在分析中单独审计。

## 4. 执行

```bash
bash scripts/run_phase14_tp_group_size_timing.sh smoke

nohup bash scripts/run_phase14_tp_group_size_timing.sh formal-analyze \
  > experiment-results/phase14/phase14_driver.log 2>&1 &
```

TP8 使用全部 8 张 GPU，所有 model/TP 单元必须串行运行。通用 runner 会在每个单元
开始前检查目标 GPU 空闲状态。

## 5. 验收条件

- smoke 8/8 单元通过；
- 正式 60/60 单元生成 `DONE`；
- 新增 180/180 compact labels；
- 每个聚合配置恰好 3 个 repeat；
- 每条标签的所有 rank kernel count、backend sequence 和 PatternDemand 完全对齐；
- raw-op 直方图边缘化后与 payload-only 直方图一致；
- 曲线外推调用比例显式报告，不允许静默外推；
- TP2-calibrated 和 per-TP descriptive calibration 分开报告；
- raw profiler trace 在 validator 通过后删除；
- 日志、telemetry、README、summary、CSV、图和 manifest 提交并 push。

## 6. 结论边界

Phase 14 只覆盖单节点 B200、Qwen3 同家族两模型、代表性 TP2/4/8 mixed/chunked
workload。不能外推到跨节点 L2/L3、PP、PD、expert-parallel All-to-All、其他 GPU
或未知 backend lowering。
