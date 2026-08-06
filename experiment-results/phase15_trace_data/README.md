# Phase 15：公开真实流量窗口数据

数据源为 BurstGPT v2.0 无失败请求和 Mooncake FAST'25 官方 trace。全部文件的 URL、
大小与 SHA-256 见 `source_manifest.json`。

- 历史窗口：300 秒；
- 预测窗口：60 秒；
- 窗口总数：66642；
- Qwen3-8B smoke 计划：20 个窗口，每个最多 8 个请求；
- 输入/输出长度上限：8192/128。

BurstGPT 使用时间顺序划分；Mooncake 只作为 external test。当前 smoke 把每个未来窗口
抽样请求作为同一时刻进入的 draining batch，`arrival_offsets_ms_audit_only` 被保留，
但尚未执行真正在线交错到达。因此本阶段不能声称 arrival/burst 对 PatternDemand 的
物理影响已经验证。

产物：

- `windows.csv.gz`：完整因果窗口特征；
- `smoke_window_features.csv`：20 个 smoke 窗口输入特征；
- `smoke_replay_plan.jsonl`：固定长度 Qwen3-8B 回放计划；
- `source_manifest.json`、`summary.json`。
