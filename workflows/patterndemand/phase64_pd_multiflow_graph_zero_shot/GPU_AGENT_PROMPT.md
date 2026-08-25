# Phase64 GPU Agent 合同

只从控制端指定的精确 W64 commit 执行。先完整阅读本目录全部文件，特别是 `experiment.json`、`README_CN.md` 和 `RESOURCE_ALLOCATION_CN.md`。

你要完成的是 48 个顺序执行的生产 Mooncake/RDMA shard，不是模型推理。允许按合同做环境诊断、有限重试、同拓扑预冻结 GPU tuple 替换和 CV 触发的 5→7→9 repeat；不得改图、payload、backend、系数、阈值，不得挑更快方向或删除异常。

资源方式优先选择一次保留 4 个节点的 `FOUR_NODE_SINGLE_ALLOCATION`；如果调度器元数据无法用这四个节点覆盖三类 topology，才使用 `SEQUENTIAL_TOPOLOGY_EPOCHS`。无论申请方式如何，单个 shard 最多激活 2 个节点、5 个 GPU 进程，并且同一时刻只能有 1 个 shard。四节点预约是为了排队和固定候选 placement，绝不能在两个节点对上同时跑两个 shard。

raw、plan、preflight audit 放在 Git 外。正式 Git 结果只允许 `commit_allowlist.txt` 中的紧凑文件。不得 `git add .`。完成后运行 `verify.py`，创建唯一父提交为 W64 的 R64 并 push run 分支。若合同定义的硬条件不满足，生成可审计 BLOCKED 证据，不得绕过。

回传必须报告：W64/R64、run 分支、容器/GPU/节点/拓扑、48 shard 与 repeat 数、Git 外 raw 路径、240 official points、总体及各配置 WAPE/bias、是否优于 max-edge baseline、variance 标记、结果目录/manifest/DONE，以及能和不能得出的结论。
