# Phase54：PD 12-bin 直方图精度改进（本地 CPU）

Phase50 已证明六模型 `H0+DNN` 比 H0 好，但 overall histogram WAPE 仍为 calls 22.97%、bytes 20.29%；total WAPE 已约为 calls 0.96%、bytes 2.18%。因此 Phase54 只针对“总量大致正确、12 个消息大小 bin 分配不准”的问题做开发集改进。

本阶段不使用 GPU、不联网、不读取 Phase50 blind 标签、不读取 raw 或完整请求列表。输入只来自 Phase48 已提交的紧凑 examples：低维画像、模型结构特征、固定 pure-PD 执行特征、H0 直方图和已生成的 Hfull 训练标签。完整请求不进入本阶段结果。

候选包括：

- 共享 residual head 与模型专属 residual head；
- 普通 encoded loss、shape-focused loss、tail-shape-focused loss；
- 固定的 4-fold profile-group OOF 选 candidate 和每模型 alpha；
- 最后只在 Phase48 的 240 个 development validation profiles 上验证一次。

## 预注册门槛

正式目标是 overall calls histogram WAPE ≤10%、bytes histogram WAPE ≤10%。为了防止平均数掩盖坏模型，六个模型各自的两项 histogram WAPE 都必须 ≤15%，三个 BurstGPT segment 也必须 ≤15%；同时保留 H0 严格改进、total WAPE ≤5% 和 composite ratio <1 的保护条件。

这不是盲测结果。只有 `summary.json` 中 `gates.phase55_permitted=true`，才允许后续另写 workflow 做一次性 blind freeze/evaluation；Phase54 失败时，不能把 Phase50 blind 结果拿来调参。

## 运行

在 workflow commit 已确定后：

```bash
W54=$(git rev-parse HEAD)
python3 workflows/patterndemand/phase54_pd_histogram_accuracy_refinement/preflight.py \
  --expected-workflow-commit "$W54"
python3 workflows/patterndemand/phase54_pd_histogram_accuracy_refinement/run.py \
  --expected-workflow-commit "$W54"
python3 workflows/patterndemand/phase54_pd_histogram_accuracy_refinement/verify.py
```

默认结果目录是 `experiment-results/phase54_pd_histogram_accuracy_refinement`。结果可以是完整 `PASS` 但 `target_met=false`；这表示审计和训练完整，但 10% 目标没有达到，不得宣称已达到正式精度线。

## 结论边界

Phase54 只回答：在 Phase48 的 1200 个开发画像和六个已知模型结构上，哪种直方图预测器候选最有希望达到 10% 目标。它不回答 fresh blind 泛化、未见模型、Mooncake、物理通信时间、placement 或在线调度。
