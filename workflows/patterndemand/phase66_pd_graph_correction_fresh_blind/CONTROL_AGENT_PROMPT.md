# Phase66控制端验收

先用`verify_result_commit.py --phase phase66 --workflow-commit <W66> --result-commit <R66>`检查唯一父提交、路径、禁止资产和manifest，再运行本目录`verify.py`独立复算。

重点：R65模型与门槛完全冻结；reserved page与Phase64零重叠；全部endpoint tuple避开Phase64且每拓扑至少一个新host signature；48顺序shard、raw在Git外；240点公式/指标机械复算；无训练、重校准、模型权重、下载或并发shard。精度FAIL是有效结果，但不得表述为PASS或据此原地调参。
