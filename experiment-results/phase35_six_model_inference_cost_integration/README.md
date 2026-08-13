# Phase35：六模型统一推理与placement/topology连续通信代价集成

本阶段没有训练或调参。统一运行时加载Phase34最终选中候选的TP/PP三seed五折checkpoint，从低维target-free特征重放六模型消息直方图；2,592条phase预测与Phase34C冻结结果在`1e-6`相对容差内一致。随后同一份拓扑无关直方图分别代入候选连续代价曲线。

TP单机B200 NVLink使用Phase2物理测量的CustomAllReduce/NCCL backend-aware曲线；TP L2/L3和全部PP曲线是参数化敏感性proxy，不能包装成真实硬件时延。共同参考曲线只做数值回归，与Phase34保存cost的最大相对差为`5.952e-16`。

`analysis/cost_metrics.csv`保存整体、逐模型、逐policy的cost MAPE/WAPE；`analysis/communication_only_rankings.csv.gz`只给通信项排名，尚未加入显存、计算、资源、拥塞和重叠，不能直接等同最终调度决策。Phase34D target已经打开，本阶段cost误差属于重复工程证据，不是新盲测。
