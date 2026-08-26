# Phase66 GPU Agent合同

从控制端指定的精确W66执行。完整阅读本目录全部文件。先确认R65已在正式分支，然后建立唯一run分支。

你执行的是48个顺序Mooncake/RDMA通信shard，不是模型推理，也不是四节点collective。允许按`AUTO`/`RECORD_AND_CONTINUE`做正常环境诊断、有限重试、在raw前同拓扑tuple替换，以及CV触发的5→7→9 repeat；不得改变payload、通信图、backend、R65系数、门槛、official point策略，不得挑更快方向、删异常或并发两个shard。

先在Git外填写`topology_inventory.json`。所有endpoint tuple必须避开Phase64 plan，每种拓扑至少一个全新host signature；选择证据只能来自scheduler/asset rack/fabric元数据。生成plan后不得换placement。raw、plan、preflight audit全部留在Git外。

完成后运行`verify.py`，只按`commit_allowlist.txt`选择性提交，禁止`git add .`；R66唯一父提交必须是W66并push run分支。硬门不满足时回传可审计BLOCKED，不得绕过。

回传：W66/R66、run分支、容器/GPU/节点/拓扑、48 shard与repeat、Git外plan/preflight/raw路径、240 official points、三种方法的总体及分片WAPE/bias、variance、README/summary/logs/DONE/manifest、能和不能得出的结论。
