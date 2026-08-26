# Phase68 GPU Agent合同

从控制端指定的精确W68创建唯一run分支。完整阅读本目录全部文件，确认正式分支已含R67；不要使用旧Phase66 run分支或raw。

执行48个顺序Mooncake/RDMA通信shard，不是模型推理，也不是四节点collective。允许在合同的`AUTO`/`RECORD_AND_CONTINUE`范围诊断环境、有限重试、在raw产生前替换同拓扑endpoint，以及CV触发5→7→9 repeat；禁止改变page/图/backend/R67系数/阈值/official point策略，禁止挑更快方向、删异常或并发两个shard。

先在Git外填写并冻结`topology_inventory.json`。所有endpoint tuple必须同时避开Phase64和Phase66 plan，每拓扑至少一个host signature也未在两期出现。选择只能依据scheduler/asset rack/fabric元数据；plan生成后不得换placement。raw、plan、preflight全部留在Git外。

完成后运行`verify.py`，只按`commit_allowlist.txt`选择性添加，禁止`git add .`。R68必须仅以W68为父提交并push run分支；硬门不满足时回传可审计BLOCKED，不得绕过。

回传：W68/R68、run分支、容器/GPU/节点/拓扑、48 shard及repeat、Git外plan/preflight/raw路径、240 official points、max-edge/R61/R65/R67的总体及分片WAPE/bias、variance、README/summary/logs/DONE/manifest，以及能和不能得出的结论。
