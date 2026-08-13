# Phase33 TP/PP最终裁定

Phase33在预测与SHA先归档、随后一次性打开的9个全新BurstGPT确认窗口上完成正式收口。

## 最佳模型与新盲测结果

- TP：`tp33_c10_shared_trunk_small_heads_policy_lr0.003_w64_5fold_3seed_alpha0.75`；calls WAPE 4.30%、bytes WAPE约0、TV 0.1186、EMD 0.0117、cost WAPE 2.90%，正式通过。
- PP：`pp33_anchor_blend_1_phase32_incumbent_calls_shape`；calls WAPE 3.37%、bytes WAPE约0、TV 0.1130、EMD 0.0156、cost WAPE 2.80%，正式通过。
- TP相对H0的calls/cost WAPE分别改善51.14%/49.61%；PP分别改善43.43%/42.07%。

TP和PP均保留`H0 + DNN residual`：DNN residual非零，负责calls总量和直方图形状；bytes总量由低维历史均值、模型bytes/token先验和结构通信倍数锚定，H0保留bytes-bin形状。

## 数据与证据边界

- TP开发：94个画像、35,524个完整teacher请求。
- PP incumbent来自Phase31 49画像；Phase33新增45个画像、14,466个teacher请求用于bytes选择与保护验证。
- 新盲测：9个画像、1,742个完整teacher请求；与开发集和Phase27至Phase32所有角色满足请求级互斥/300秒embargo。
- 新盲测仅覆盖BurstGPT；旧Phase31/32集只属于重复工程证据。
- 旧集上TP仍有失败，PP正式通过；不能声称TP对所有历史分布都已普遍收口。
- PP MB16仍是弱项；PP整体正式通过不等于每个policy逐项满足整体阈值。

完整整体、逐模型、逐policy和逐并行规模指标见`../analysis/aggregate_metrics.csv`；逐case结果见`../analysis/per_case_metrics.csv.gz`；冻结预测SHA为`f634a6f0c82c0109132e752e4f52f266b9aa8904d947874ce98b32e9dd80f7d0`。
