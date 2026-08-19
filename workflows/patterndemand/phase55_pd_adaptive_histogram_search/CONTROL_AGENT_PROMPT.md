# Phase55 控制端交接

Phase55 是本地 CPU 的受约束自适应搜索，不需要 GPU Agent、node55、网络或模型下载。

控制端在运行前必须确认：

1. 当前 HEAD 等于 W55，且 Phase48/Phase54 pinned inputs 的 SHA 与 `experiment.json` 完全一致；
2. 工作树除受保护的 `data/` 和已核验的 Phase54 本地结果外，不存在其他未跟踪项；
3. 只读取 Phase48 的 1,200 个 compact profile/example/target rows；不读取 Phase50 blind、raw 或完整请求；
4. OOF 使用 profile-group 四折隔离；六个模型属于同一 profile，必须留在同一 fold；
5. Stage A 恰好 10 个候选，按 OOF 结果选 top-3；Stage B 每个 top-3 恰好生成 2 个变体，总候选恰好 16 个；
6. 所有 alpha、epoch、候选选择都在 OOF 冻结后，才允许一次性打开 240 个 development validation profiles；
7. 结果只能选择性加入 `experiment-results/phase55_pd_adaptive_histogram_search/`，禁止 `git add .`。

验收门槛：overall calls/bytes histogram WAPE ≤10%，六个模型各自两项 ≤15%，三个 BurstGPT segment 各自两项 ≤15%，calls/bytes total WAPE ≤5%，并且四项核心指标严格优于 H0。`target_met=false` 仍可作为完整搜索的负结果，但必须阻断后续 Phase56；不得以 Phase50 blind 继续调参。
