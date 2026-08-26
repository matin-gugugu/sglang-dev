# Phase68控制端验收

先运行`verify_result_commit.py --phase phase68 --workflow-commit <W68> --result-commit <R68>`，检查唯一父提交、路径、禁止资产和commit内manifest，再在R68树上运行本目录`verify.py`独立复算。

重点核对：R67模型/门槛完全冻结；page为`{36,40,48,56,64}`；endpoint tuple同时避开Phase64/66且每拓扑至少一个新host signature；48个顺序shard；raw在Git外；240点和max-edge/R61/R65/R67四种方法的指标可机械复算；无训练、重校准、模型权重、下载或并发shard。科学精度FAIL是有效结果，但不得表述为PASS或据此原地调参。
