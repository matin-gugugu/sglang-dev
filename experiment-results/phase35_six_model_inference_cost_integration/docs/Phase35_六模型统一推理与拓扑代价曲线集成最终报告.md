# Phase35 六模型统一推理与拓扑代价曲线集成最终报告

Phase35 集成 PASS，没有训练或调参。统一运行时加载 TP/PP 最终选中候选的 30 个 fold 模型，生成 2,592 条六模型 phase 直方图；与 Phase34C 冻结预测逐字段零差异。共同参考 cost 最大相对差为 `5.952e-16`。

输出 10,368 条 phase cost、5,184 条双 phase 合并 cost、240 条整体/逐模型/逐 policy 指标和 3,888 条 communication-only 排名。1,296 个配置的通信 top1 与 teacher cost 排名一致，但该排名不包含系统约束。

|方向|曲线|证据|cost MAPE|cost WAPE|
|---|---|---|---:|---:|
|TP|单机 B200 backend-aware|物理测量|7.22%|6.09%|
|TP|L2 nominal|参数化 proxy|8.04%|7.05%|
|TP|L3 nominal|参数化 proxy|8.67%|7.50%|
|PP|L1 nominal|参数化 proxy|8.98%|3.95%|
|PP|L2 nominal|参数化 proxy|10.22%|4.11%|
|PP|L3 nominal|参数化 proxy|12.96%|4.39%|

关键结论：拓扑无关直方图到连续通信曲线的接口已经可靠接通；但 Phase34 的共同参考 cost 通过不保证任意物理曲线仍低于 5%。TP 实测 L1 cost WAPE 为 6.09%，所以不能宣称 TP 在真实单机曲线下已达到 5% cost 线。PP 曲线尚未物理测量，不能把 proxy 数值当成真实 PP 延迟。

第一遍运行正确触发复播保护：checkpoint 文件名 `topN` 是初筛入围顺序，并非最终确认排名。加载逻辑改为按 summary 的最终 candidate id 选择 checkpoint；没有更改 checkpoint 或参数，正式复播随后达到零差异。
