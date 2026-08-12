# Phase 32C：冻结预测的新确认与重复固定集评测

Phase32B预测和SHA已先归档，本阶段之后才生成9个新BurstGPT请求级互斥窗口的Hfull target。新确认覆盖三个已知模型、全部TP/PP配置和2,976个完整请求，是本轮主证据；原10个固定窗口没有更换，其复评明确标为重复工程证据。

主证据裁定：TP=`fail`，PP=`conditional_pass`。重复固定集裁定：TP=`fail`，PP=`conditional_pass`。

完整整体、逐模型、逐policy和逐并行规模指标见`analysis/aggregate_metrics.csv`；逐case结果见压缩明细。新确认不含Mooncake，因此不能把它单独表述为跨数据源验证。
