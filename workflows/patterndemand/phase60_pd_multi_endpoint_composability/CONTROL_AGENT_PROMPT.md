# Phase60控制端验收提示

收到R60后，在独立临时worktree使用`verify_result_commit.py`核验：唯一父提交是控制端指定的exact W60-fix；只包含Phase60 allowlist；无raw/model/cache/PID；commit内manifest有效。随后运行Phase60 `verify.py`。

只有验收全部通过才允许ff-only合入正式分支。重点检查：24个measurement、120个official development point、240个replica point、未来blind pair零记录、两种baseline都存在、scientific outcome由冻结阈值机械推导。

资源合同也必须验收：每个result shard只能有3个runtime endpoint；L1实际host数为1，L2/L3实际host数为2。inventory中的4个GPU slot不是4个node，任何四节点同时分配都不是Phase60要求。

Phase60无论显示可组合还是需要修正，都不得直接从其结果声称fresh blind泛化。若为`CONTENTION_CORRECTION_CANDIDATE`，下一阶段使用Phase60 development数据拟合轻量模型；模型冻结后另写workflow测未使用placement和reserved blind pairs。
