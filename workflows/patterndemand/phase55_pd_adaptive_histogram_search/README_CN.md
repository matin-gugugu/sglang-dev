# Phase55：PD 受约束自适应直方图搜索

Phase54 的最佳开发候选仍为 calls histogram WAPE 21.55%、bytes histogram WAPE 19.69%，没有达到整体 10% 门槛。Phase55 不靠查看 blind 答案反复试错，而是在 Phase48 的 960 个 train profiles 上做固定预算的 4-fold profile-group OOF 搜索。

## 搜索如何自己迭代

第一轮固定生成 10 个 seed candidates，覆盖 shared/per-model、causal/full target-free、uniform/shape/tail loss、不同宽度深度。根据 OOF 的四项误差、H0 保护、模型/segment 保护和 12-bin signed bias，保留 top-3；每个 top candidate 再自动生成两个变体：一个加深/加宽，一个改变 shape/tail 重点或切换 per-model head。总候选最多 16 个。

所有候选、alpha 和 epoch 都只在 OOF 上决定。搜索结束后才把选中的候选在 960 个 train profiles 上重训，并且只打开 240 个 development validation profiles 一次。Phase50 blind 完全不可见。

## 合同门槛

- overall calls histogram WAPE ≤10%；
- overall bytes histogram WAPE ≤10%；
- 六个模型各自两项 histogram WAPE ≤15%；
- 三个 BurstGPT segment 各自两项 histogram WAPE ≤15%；
- total calls/bytes WAPE ≤5%；
- 四项核心指标严格优于 H0。

达标后才允许另写 Phase56 blind-freeze/evaluation workflow。若 `target_met=false`，表示固定搜索预算没有达到合同，不允许打开 Phase50 blind 继续调参。

## 运行

```bash
W55=$(git rev-parse HEAD)
python3 workflows/patterndemand/phase55_pd_adaptive_histogram_search/preflight.py --expected-workflow-commit "$W55"
python3 workflows/patterndemand/phase55_pd_adaptive_histogram_search/run.py --expected-workflow-commit "$W55"
python3 workflows/patterndemand/phase55_pd_adaptive_histogram_search/verify.py
```

本阶段是本地 CPU workflow，不需要 GPU Agent，不读取 raw、完整请求或 Phase50 标签。
