# 新会话完整交接：截至Phase34

## 必读

先读实验总导引、Phase33 状态补充、本目录的 Phase34 状态补充、最终报告和资产索引。

## 当前结果

- 六模型 TP、PP 均正式通过。
- TP 最佳：`tp34_c17_lowdim_cost_protected_gate_model_policy_lr0.001_w32_5fold_3seed_alpha1.0`；新盲测 calls WAPE 7.81%、TV 0.1903、EMD 0.0201、cost WAPE 4.35%。
- PP 最佳：`pp34_c04_pp_split_retrain_policy_lr0.003_w64_5fold_3seed_alpha1.0`；新盲测 calls WAPE 4.53%、TV 0.1287、EMD 0.0171、cost WAPE 3.29%。
- 冻结预测 SHA：`faffe08800e6336fa9272b765ca1965d4aee806a6162255f7bd9a50d5d5b5bda`。

开发集为 94 画像、35,524 个 teacher 请求；新盲测为 12 个全新 BurstGPT 窗口、3,803 请求。六模型都参与训练，不是未见模型泛化。Phase33 原 9 个盲测只算重复工程证据。不得使用已经打开的 Phase34 盲测 target 调参。

下一步统一封装六模型 TP/PP 冻结推理流程，将消息直方图接入 placement/topology 连续通信代价曲线。继续保护本地 `data/`、远端 Phase16/19/23、raw、缓存、权重和 PID；禁止 `git add .`。
