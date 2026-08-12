# ProfileDemand GPU labels：qwen3-8b TP=2

共验证 20 个 histogram-only draining microbatches，聚合为 6 条
`profile×strategy×repeat×phase` 标签。每条标签按 1000 请求归一化，并同时保存 12 桶
calls、12 桶 logical bytes、canonical 精确直方图和 raw-op 精确直方图。

all-rank 对齐、固定实际输出、group size、histogram-only 无 raw events、H0 canonical
解析映射和重复一致性均通过。raw fused-op 仅供第二阶段 backend 细化；第一阶段正式
目标为 canonical logical AllReduce PatternDemand。
