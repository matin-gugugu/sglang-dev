# Phase54 控制端交接

Phase54 是本地 CPU 开发实验，不需要 GPU Agent，也不需要登录 node55。

控制端必须先确认：

1. HEAD 等于 W54；
2. 工作树只有受保护的 `data/` 未跟踪项；
3. Phase48 manifest、examples、targets、profiles 的 SHA 与 `experiment.json` 完全一致；
4. 不读取 Phase50 的任何 blind label、per-profile prediction 或 raw；
5. 不使用网络、GPU、模型下载或完整请求列表。

Phase54 的输出不是正式 blind 结果。只有 OOF 保护、overall 双 WAPE≤10%、逐模型双 WAPE≤15%、逐 segment 双 WAPE≤15%、total WAPE≤5% 全部通过，才允许设计后续 Phase55 blind freeze/evaluation。不得因为 validation 结果不达标而打开 Phase50 blind 集合调参。

正式结果只能选择性加入 `experiment-results/phase54_pd_histogram_accuracy_refinement/`，禁止 `git add .`，禁止加入 `data/`、raw、完整请求、PID、模型原始权重或缓存。
