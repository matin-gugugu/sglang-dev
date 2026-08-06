# Phase 15：历史画像到 PatternDemand 的首版预测器

该阶段以 300 秒历史流量画像为输入，预测下一 60 秒窗口中一个代表性、最多 8 请求的
simultaneous draining batch 的 PatternDemand。输入包含到达率/突发度、历史 prompt 与
output 长度统计及 log2 直方图；输出为 Prefill 单次 payload 和 Decode 中
`active_batch=1..8` 的持续步数，随后恢复精确消息直方图。

对照包括历史均值 persistence、两层 MLP，以及使用未来已调度请求长度的解析公式上界。
L1 评测把预测直方图和真实直方图分别乘同一条实测连续代价曲线，因此衡量的是第一阶段
误差向通信代价的传播；它不是 Phase 15 新测的绝对通信时间误差。Phase14F 已独立验证
“真实直方图 × L1 曲线”对 all-rank 通信时间的 MAPE 为 4.43%。

边界：当前标签仍是代表性 draining batch，不是完整在线 continuous batching。历史画像
无法决定下一窗口的精确请求集合，因此该模型是流量预测 pilot；调度器若已掌握待调度请求
的实际长度，应优先使用解析 PatternDemand 路径。
