# Phase51：纯PD L1–L3物理通信曲线库

状态：`PASS_WITH_RUNTIME_AND_PLACEMENT_VARIANCE`。使用SGLang生产Mooncake batch-transfer + RDMA/dma-buf，完成6模型×3拓扑×2冻结placement的36个测量shard，汇总18条模型相关曲线、396个物理knots。正式值保守取重复中位数、双向较慢值、两个placement较慢值。raw逐次样本仅保存在Git外。Phase51未训练、未加载模型、未重算直方图，也未做代价卷积或placement决策；后者属于Phase52。
