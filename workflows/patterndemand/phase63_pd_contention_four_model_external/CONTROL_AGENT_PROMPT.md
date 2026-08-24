# Phase63控制端验收提示

收到R63后先用`verify_result_commit.py`核验唯一父提交exact W63、路径allowlist、禁止资产及commit内manifest，再运行Phase63 `verify.py`。只有全部PASS才允许ff-only合入。

重点验收：48个三rank shard、240个held-out-model official点、480个replica点、四个模型身份准确、payload grid未变、R61模型逐字节一致、Phase62两模型结论仍为PASS、训练/调参/权重/推理均为false、raw仅在Git外。

资源验收必须确认：全局峰值node数为2、峰值GPU进程为3、最大并行measurement shard为1；L1/L2/L3可来自不同的顺序allocation。inventory里累计host数量不得被误报成同时node需求。

科学门必须机械产生。四模型PASS并且R62保持PASS，才允许写“六模型验证通过”；不得外推P2D2、多于两流、端到端调度、计算/显存/排队/在线到达。
