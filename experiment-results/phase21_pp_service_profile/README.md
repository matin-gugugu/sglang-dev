# Phase 21：到达感知的纯 PP 服务画像实验

## 与 Phase 20 的关系

Phase 20 已在 Qwen3-8B 的受控 simultaneous-arrival/draining workload 上完成纯 PP
PatternDemand 预测器。Phase 21 不重复该实验，而是补充此前缺失的服务常态到达过程：

`流量画像（长度分布、RPS、到达间隔 CV） + PP 策略 + PP size -> PP 消息直方图`。

实验始终保持 `TP=1`，因此采集的是纯 PP 通信，不是 TP×PP 混合并行。

## Smoke 矩阵

`qwen3-8b-smoke-v1` 覆盖：

- `PP=2/4/8`；
- `pp_max_micro_batch_size=1/4/16`；
- 2 个 BurstGPT 与 1 个 Mooncake 代表画像；
- `profiled` 与 `draining` 两种到达模式；
- 每个组合重复 3 次。

9/9 个 cell 通过审计，共 162 次画像回放和 5184 次逻辑请求执行。81 个配对重复中，
63 个在两种到达模式下产生不同的精确 payload 直方图。详细指标、重复稳定性与结论边界见
`qwen3-8b-smoke-v1/analysis/README.md`。

## 正式矩阵原则

正式训练矩阵覆盖全部 24 个服务画像，使用 `profiled` 到达作为目标域。`profiled` 是依据
画像 RPS 和到达间隔 CV 构造的确定性 gamma-renewal 稳态实现，不是对未来请求序列的
预测，也不是原始 trace 时间戳的逐项复刻。`draining` 只作为代表性配对消融，避免将同一
批GPU计算重复一整倍。

PP profiler 在每个相邻 stage sender 上记录相同的逻辑边界传输。标签只取首个 sender
边界作为 group-level truth；端到端 pipeline demand 通过乘以 `PP-1` 派生，不能把所有
sender/receiver rank 再次求和。

## 目录

- `qwen3-8b-admission-v1/`：最小准入检查；
- `qwen3-8b-smoke-v1/`：到达机制 smoke 及分析；
- `qwen3-8b-formal-profiled-v1/`：24 画像正式训练矩阵（后台运行后生成）。
