# Phase57：PD 直方图合同精度的迭代优化

Phase55/56 已经证明 H0+DNN 能稳定优于 H0，但 histogram WAPE 仍约为 20%，没有达到合同的 10%/15% 门槛。本阶段不再重复一次固定大小的 MLP 搜索，而是执行最多六轮的 OOF-only 迭代：每轮同时尝试已有 MLP、分组 MLP、工程特征 ridge、直接 shape 预测、support-aware 后处理和 OOF 融合；每轮根据上一轮 OOF 的 bin 偏差和 model/segment 失败门生成下一轮候选。

## 运行规则

每个 profile 的六个 model 行在同一 fold。arrival 特征被排除，输入仍是低维画像、模型结构和固定 P1/D1 执行配置；不会把完整请求列表放进最终 predictor。候选、分组校准、alpha、融合权重和 epoch 都只能从 train OOF 得到。只有 OOF 保护和所有合同门同时成立后，才冻结一次并打开 240 个 validation profiles；validation 不能再反馈到任何搜索步骤。

硬上限是 6 轮 × (16 个 seed + 8 个 adaptive/blend) = 144 个候选。OOF 先达标可以提前停止；到达上限仍未达标则完整记录负结果，`next_phase_permitted=false`，不得读取 Phase50 blind 继续调参。

## 合同门槛

- overall calls histogram WAPE ≤10%；
- overall bytes histogram WAPE ≤10%；
- 六个模型各自 calls/bytes histogram WAPE ≤15%；
- 三个 BurstGPT segment 各自 calls/bytes histogram WAPE ≤15%；
- overall calls/bytes total WAPE ≤5%；
- 四项核心指标均严格优于 H0。

这是 CPU development workflow，不使用 GPU、网络、raw、完整请求或 Phase50 blind。结果只能选择性提交 `experiment-results/phase57_pd_iterative_histogram_optimization/`，禁止 `git add .`。
