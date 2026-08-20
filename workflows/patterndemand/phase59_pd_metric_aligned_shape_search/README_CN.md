# Phase59：PD metric-aligned shape DNN 迭代搜索

Phase58 证明继续搜索同一种 encoded-residual MLP、ridge、ratio-ridge 和 blend 不能把约 20% 的 histogram WAPE 压到合同阈值。Phase59 保留 `H0 + DNN residual`，但把学习目标改成最终真正验收的量：直接在 12-bin 直方图空间优化 calls/bytes WAPE，并联合约束 calls TV 和 EMD。

DNN 不再预测总量。每个画像的 H0 calls/bytes 总量原样保留，只通过 `softmax(log(H0 share)+residual)` 重新分配 12 个 bin。这样总量天然保持在现有约 1%–2.5% 的合格范围，模型容量集中用于最薄弱的形状预测。

搜索按 4-fold profile-group OOF 进行。每轮训练 4 个 metric-aligned model-head DNN，再评估 4 个 OOF blend；若合同未通过，就根据 OOF 的最差指标、模型、segment 和安全 bin bias 进入下一轮。达到完整 OOF 合同立即停止；否则使用最多 9 小时搜索预算，并预留约 1 小时完成最终三种子训练和一次 development validation。实际若提前达标或候选运行很快，可以早于 10 小时结束。

合同阈值完全不变：overall calls/bytes histogram WAPE ≤10%；六模型各自两项 ≤15%；三个 BurstGPT segment 各自两项 ≤15%；calls/bytes total WAPE ≤5%；四项核心指标严格优于 H0。预算用尽仍不达标时必须输出负结果和 continuation spec，不能降低阈值或读取 Phase50 blind 继续调参。

本 workflow 仅使用 CPU、Phase48 冻结低维数据和已提交的 Phase58 诊断，不读取 raw、完整请求、模型权重或缓存。
