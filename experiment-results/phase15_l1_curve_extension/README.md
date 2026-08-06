# Phase 15：L1 长消息连续代价曲线补点

为覆盖公开 trace 中 320--512 MiB 的长 Prefill 消息，本阶段在单节点 B200 NVLink
拓扑上补测 AllReduce 的 160/192/256/320/384/448/512 MiB 支撑点，覆盖 TP2/4/8。

- 21 个 `TP × payload` 支撑点；
- 每个支撑点 5 次独立重复，每次 100 个样本；
- 共 10500 个 all-rank post-rendezvous 样本；
- 最大 repeat-median CV：0.3977%；
- 时间口径与更正后的 Phase14F 完全一致。

`curve_summary.csv` 提供连续插值使用的中位数代价；`curve/` 保留完整样本和 backend
审计字段。
