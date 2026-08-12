# Phase 27A：PP 调度敏感画像与新窗口冻结合同

本阶段只做**事前冻结**，不生成、读取或比较任何 Phase 27 Hfull 标签。目的是避免根据
真值误差挑窗口或改特征，从而给后续 PP 增强预测器保留真正独立的确认集。

## 新窗口划分

- 从 Phase 15 的 66,642 个 300 秒历史窗口中选择；
- 明确排除 Phase 16 已使用的 24 个窗口；
- 每个 BurstGPT/Mooncake segment 用 49 个仅依赖历史的选择特征做
  robust-scaled medoid 覆盖，固定选 10 个，共 60 个新窗口；
- 每个 segment 在看标签前按固定 SHA-256 顺序划为 5 个开发训练、2 个开发验证、3 个
  独立确认窗口；总计 30/12/18。

## 新增低维画像

保留原 4×4 长度联合分布、长度分位数、Decode 生存率和到达统计，并新增与 PP
fixed-draining 调度直接相关的低维摘要：4096-token prefill chunk 数、多 chunk 比例、
chunk×输出长度联合分布、相邻 chunk 类转移、多 chunk 连续段，以及 4/16/32 请求块内的
局部多 chunk 峰值。完整请求顺序只在离线阶段聚合成这些标量，不进入训练表。

`feature_contract.json` 是特征白名单和禁止输入；
`selection/selected_windows.csv` 是不可事后更改的新窗口及角色；
`selection/candidate_counts.csv` 记录各 segment 的候选覆盖。

当前能得出的结论仅是：独立评测合同和候选特征已经冻结并通过审计。当前不能得出新特征
改善 PP 的结论；该结论必须等待 Phase 27B/27C 在 18 个独立确认窗口上验证。
