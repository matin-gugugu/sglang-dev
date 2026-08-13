# 新会话完整交接：截至Phase35

Phase34 六模型 TP/PP 已正式通过；Phase35 已完成冻结推理和连续通信代价曲线集成。

- 统一推理：2,592 条 phase 直方图，与 Phase34C 零差异。
- TP 单机 B200 实测曲线：cost MAPE 7.22%、WAPE 6.09%。
- TP L2/L3 proxy：cost WAPE 7.05%/7.50%。
- PP L1/L2/L3 proxy：cost WAPE 3.95%/4.11%/4.39%。
- 共同参考 cost 与 Phase34 数值一致，最大相对差 `5.952e-16`。
- Phase35 是重复工程评估，不是新盲测；PP 没有实测 P2P 曲线。

正式目录：`experiment-results/phase35_six_model_inference_cost_integration/`。核心脚本：`scripts/run_phase35_six_model_inference_cost_integration.py`。下一步不要用已打开的 Phase34D target 调参；若优化 TP 物理 curve cost，需要新确认协议；PP 应优先补测 P2P 曲线。

继续禁止 `git add .`，保护本地 `data/`、远端 Phase16/19/23、raw、缓存、权重和 PID。
