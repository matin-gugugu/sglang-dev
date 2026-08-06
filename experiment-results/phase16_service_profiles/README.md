# Phase 16B：ProfileDemand 服务常态画像

从固定哈希的 BurstGPT v2.0 与 Mooncake FAST'25 trace 的 300 秒窗口中选择 24
个 medoid 画像。选择特征包含截断后的 4×4 `P(L,M)`、长度/Decode 生存率、RPS 和突发
摘要；各 trace segment 使用固定 quota，避免大规模 BurstGPT 完全淹没 Mooncake。

- 输入联合分布边界：0/128/512/2048/∞；
- 实际输出联合分布边界：0/16/32/64/∞；
- GPU 回放上限：输入 8192、实际输出 128；
- 每画像固定 128 个分层代表请求；
- 最大代表样本 joint-distribution L1：0.0375。

`service_profiles.csv` 是模型输入画像；`representative_requests.jsonl` 是后续
histogram-only GPU 标签的固定请求集合；`cluster_audit.csv` 记录覆盖范围。到达特征当前
作为画像与后续 batching 扩展输入，但首版不得把 draining-batch 回放声称为真实 online
continuous batching。
