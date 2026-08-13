# Phase 34D：六模型全新盲测与Phase33同窗基线

Phase34C的TP/PP模型、18个checkpoint、全部冻结预测和SHA已先由Git提交`6bdc0208e44e2cd8a51905560502e2ffe6c336f5`归档，本阶段才一次性生成12个全新BurstGPT请求级互斥窗口的Hfull target。新盲测包含3,803个完整teacher请求、六个模型和全部TP/PP配置。

六模型新盲测裁定：TP=`formal_pass`，PP=`formal_pass`。TP继续使用calls WAPE≤10%的正式线；PP继续使用Phase33逐模型和MB16专门保护条件。整体、逐模型、逐policy、逐并行规模指标见`analysis/aggregate_metrics.csv`。

为了可比，Phase33三模型incumbent和Phase34六模型predictor都在开target前对同一12个新窗口的原三个模型冻结了预测；对比见`analysis/phase34_vs_phase33_same_blind_comparison.csv`。Phase33原9个窗口的Phase34复评只能作为重复工程证据。

证据边界：新盲测仍是BurstGPT-only，不能声称跨数据源或未见模型泛化；六模型全部进入了训练和验证。

中文最终报告、Phase34状态补充、新会话交接和资产索引见`docs/`。
