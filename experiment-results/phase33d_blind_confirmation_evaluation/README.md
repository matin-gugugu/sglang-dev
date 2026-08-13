# Phase 33D：全新盲测与重复工程评测

Phase33C模型、checkpoint、冻结预测和SHA已先由Git提交`284dcaef47c2e128952c75f4a2b02fadf23aee66`归档，本阶段才生成9个全新BurstGPT请求级互斥窗口的Hfull target。新盲测包含1,742个完整teacher请求，覆盖三个已知模型和全部TP/PP配置。Phase31固定集与Phase32确认集没有更换，只标记为重复工程证据。

新盲测裁定：TP=`formal_pass`，PP=`formal_pass`。TP严格使用calls WAPE≤10%的正式线，不再接受12%有条件线。整体、逐模型、逐policy、逐并行规模、逐source指标见`analysis/aggregate_metrics.csv`，逐case结果见压缩明细。

证据边界：新盲测仅覆盖正常BurstGPT窗口，不能单独声称跨数据源泛化；旧两套结果只能作为重复工程稳定性证据。
