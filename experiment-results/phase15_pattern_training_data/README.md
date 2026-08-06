# Phase 15：PatternDemand 训练标签

该数据集为 66642 个公开 trace 因果窗口生成 Qwen3-8B PatternDemand 标签，其中
46535 个窗口的未来 60 秒内至少包含一个请求。标签公式已经由正式 GPU 回放的
120 条“窗口 × TP × phase”记录逐条验证。

每个窗口从未来 60 秒确定性抽样最多 8 个请求，构成一个同一时刻进入的
draining batch；记录 Prefill 消息位置、Decode `active_batch` 各档持续步数，以及精确
消息直方图。TP 不改变 logical histogram，后续按候选 TP 折算 equivalent bytes/rounds
并查询对应 L1/L2/L3 代价曲线。

边界：本数据集的标签是“下一窗口代表性 draining batch”，不是完整在线请求回放，不能
替代 continuous batching/到达交错模拟。它用于先验证历史画像预测 PatternDemand 的
训练与评测闭环。
