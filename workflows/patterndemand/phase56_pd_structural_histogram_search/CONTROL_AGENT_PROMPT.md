# Phase56 控制端交接

Phase56 是 Phase55 之后的本地 CPU development refinement，不是 blind 评估，也不需要 GPU Agent、node55、网络或模型下载。

控制端必须确认：

1. HEAD 等于 W56，Phase48/Phase54 pinned inputs SHA 全部一致；
2. 工作树除受保护的 `data/`、Phase54 和 Phase55 已核验的本地结果外没有其他未跟踪项；
3. 只读 Phase48 的 7,200 行 compact feature/H0/Hfull 表，不读取 Phase50 blind、raw 或完整请求；
4. 六模型同一 profile 作为整体进入同一 split/fold；
5. Stage A 恰好 20 个候选，按 OOF 结果取 top-6；Stage B 每个 top-6 恰好生成 2 个自适应变体，总候选恰好 32 个；
6. model×segment head、分组 calibration offset 和 alpha 只能由训练折/OOF 估计；validation 只能在所有候选冻结后打开一次；
7. 结果只允许选择性加入 `experiment-results/phase56_pd_structural_histogram_search/`，禁止 `git add .`。

合同门槛仍是：整体 calls/bytes histogram WAPE ≤10%，六模型各自两项 ≤15%，三个 BurstGPT segment 各自两项 ≤15%，整体 calls/bytes total WAPE ≤5%，并且四项核心指标严格优于 H0。未达标时结果仍可验收为完整负结果，但不得打开 Phase50 blind 或把 validation 再用于调参。
