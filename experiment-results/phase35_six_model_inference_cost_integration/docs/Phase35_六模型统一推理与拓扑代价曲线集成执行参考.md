# Phase35 六模型统一推理与拓扑代价曲线集成执行参考

Phase35 不训练或调整 Phase34 模型。它加载 Phase34 最终选中候选的 TP/PP 三seed五折 checkpoint，从低维 target-free 特征输出 fixed-draining 12-bin 拓扑无关直方图，再将同一份直方图分别代入候选 placement/topology 连续通信曲线。

TP 单机 B200 NVLink 使用 Phase2 实测的 CustomAllReduce/NCCL backend-aware 曲线；TP L2/L3 使用 Phase17 nominal 参数化敏感性曲线。仓库没有独立实测 PP P2P 曲线，所以 PP L1/L2/L3 仅使用明确标记为 proxy 的参数化曲线。共同参考曲线只用于回归 Phase34 保存的 cost。

PASS 要求包括：checkpoint 复播与冻结预测一致；target 不进入推理；共同参考 cost 一致；所有曲线输出完整有限；同一 example 跨 placement 复用相同直方图；保存 README、summary、audit、logs、预测、cost、指标、图表、DONE 和 manifest。

communication-only 排名不包含显存、计算、资源、拥塞和通信计算重叠，不能直接等同最终调度器选择。Phase34D target 已打开，cost 评估只能算重复工程证据。
