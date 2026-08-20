# Phase58：PD shape-aware 迭代优化

TP/PP 的 histogram 预测已经达标；PD 的 total WAPE 约 1%–2%，但 histogram WAPE 仍约 20%。Phase58 针对这个特定瓶颈，不改变合同阈值，也不继续单纯扩大 MLP，而是加入 `log1p(target_bin)-log1p(H0_bin)` 的逐 bin 乘性修正、fit-fold support mask、shape-only ridge 和 OOF blend。

最多 3 轮，每轮 8 个 seed + 4 个 adaptive/blend，总候选上限 36。seed 只含 2 个受限 MLP、4 个普通/shape ridge 和 2 个 multiplicative-bin ridge；adaptive MLP 固定 model scope，避免旧版 model×segment 的 72 次训练。按 W57-fix 的实测速度，预计约 4–8 小时，控制在 10 小时以内。

候选、alpha、support threshold、融合权重和 epoch 全部由 profile-group OOF 决定；validation 只有在全部冻结后打开一次。达到以下合同才允许下一阶段：整体 calls/bytes histogram WAPE ≤10%，六模型各自两项 ≤15%，三个 BurstGPT segment 各自两项 ≤15%，总量两项 ≤5%，且四项核心指标严格优于 H0。未达标必须记录负结果，不能打开 Phase50 blind 继续调参。

这是 CPU workflow，不使用 GPU、网络、raw、完整请求或模型权重。正式结果只能选择性提交 `experiment-results/phase58_pd_shape_aware_iterative_refinement/`。
