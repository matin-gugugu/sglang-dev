# Phase48：六模型纯PD扩展训练

Phase48 是控制端 CPU 实验，不使用 GPU，也不读取模型权重。它把 Phase44 已冻结且互不重叠的 1200 个 BurstGPT 画像窗口分别代入六个经过 GPU 代表性验证的模型结构，得到 7200 条紧凑的 `低维画像 + 模型结构 + 固定P1-D1策略 + H0 -> Hfull` 开发数据。

关键隔离：六个模型共享同一个窗口的流量画像，必须作为一个组进入同一 train/validation 和 OOF fold；Phase45/46 已打开的旧 blind 标签完全禁止访问；486242 条完整请求仅在内存中由 page-aware teacher 读取，Git 中只保存低维特征、12-bin 标签、预测、审计和小型 NumPy checkpoint。

训练一个共享的模型结构条件化 DNN。候选结构由四折 OOF 选择，每个模型的残差收缩 alpha 也只从 OOF 选择。240 个 validation 窗口在所有选择冻结后打开一次。只有 OOF、overall、六个模型逐一和三个流量段全部过门，才允许 Phase49 冻结全新 blind 预测。

运行：

```bash
python3 workflows/patterndemand/phase48_pd_six_model_expanded_training/preflight.py \
  --expected-workflow-commit "$W48" --raw-dir data/phase15_traces/raw
python3 workflows/patterndemand/phase48_pd_six_model_expanded_training/run.py \
  --expected-workflow-commit "$W48" --raw-dir data/phase15_traces/raw
python3 workflows/patterndemand/phase48_pd_six_model_expanded_training/verify.py
```

Phase48 只能证明六个已知模型结构上的 BurstGPT 开发集效果，不能证明新 blind、未见模型外推、物理通信时间、placement、时延或在线调度。
