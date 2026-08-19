# Phase56：结构化 PD 直方图搜索

Phase55 已证明 H0+DNN 可以稳定优于 H0，但整体 histogram WAPE 仍为 calls 21.85%、bytes 19.72%，问题集中在 12-bin 形状而不是总量。本阶段针对这个瓶颈做一次更有针对性的结构搜索。

## 这次具体探索什么

1. **head scope**：比较 shared、按 model、按 segment、按 model×segment 的残差 head。不同 BurstGPT segment 的 bin 形状明显不同，不能再假设一个 model head 跨三个 segment 共用同一修正。
2. **OOF residual calibration**：在每个训练折内，用 fit fold 的 honest residual 学习分组偏移；holdout 只应用该偏移，不用 holdout target 生成偏移。最终 validation 使用全 train 的 OOF residual 偏移。
3. **group alpha**：alpha 不再固定为每模型一个，可按 model×segment 在 OOF 中选择，避免某个 segment 的修正强度拖累其他 segment。
4. **结构与 loss**：预注册不同 width/depth、causal/full target-free 特征、uniform/shape/tail loss；Stage B 再根据 OOF 的分组偏差选择 head specialization 或 calibration 强度。

总预算为 32 个候选：Stage A 20 个，Stage B 对 OOF top-6 各派生 2 个。所有选择只看 train OOF；validation 在候选、calibration、alpha 和 epoch 全部冻结后只打开一次。

## 合同门槛

- overall calls histogram WAPE ≤10%；
- overall bytes histogram WAPE ≤10%；
- 六个模型各自 calls/bytes histogram WAPE ≤15%；
- 三个 BurstGPT segment 各自 calls/bytes histogram WAPE ≤15%；
- overall calls/bytes total WAPE ≤5%；
- 四项核心指标严格优于 H0。

如果仍未达标，Phase56 仍是有效的 development negative result，不能通过读取 Phase50 blind 继续调参，也不能声称已经达到合同。

本阶段只使用本地 CPU，不使用 GPU，不访问 raw、完整请求、Phase50 blind 或模型权重。
