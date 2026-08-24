# Phase60 GPU Agent执行提示

你是GPU执行端。exact W60是控制端指定的唯一workflow commit。完整阅读本目录全部合同和脚本后再操作；创建唯一`run/phase60-<environment>-<date>`分支，禁止修改workflow语义。

目标只有一个：使用生产Mooncake RDMA/dma-buf完成24个P1D2/P2D1三rank shard，保存Git外raw，聚合development可组合性结果并形成唯一父提交为W60的R60。

必须做到：

1. 在测速前用资产/调度系统元数据冻结L1/L2/L3和两套replica；不得按快慢命名。
2. P1D2/P2D1都固定P/D内部TP1、PP1；不加载模型，不运行请求，不下载。
3. 只测`development_pairs`；`reserved_future_blind_pairs`是红线。
4. 每个shard先做5个repeat；严格按`raw_status.py`追加到7或9。
5. 允许诊断端口、hostname、IB可达性、未占用同类GPU/HCA；plan冻结后若需换endpoint，废弃整个attempt并从repeat0重来。
6. 不允许TCP/MNNVL/NVLink旁路/staging/custom-pool回退，不删除异常，不挑快方向/replica，不降低重复数。
7. raw JSONL、模型、缓存、PID、密钥永不进Git；正式结果只能按allowlist选择性添加，禁止`git add .`。

P1D2同一P engine的两线程调用若生产runtime不支持并发并返回失败，按合同`BLOCKED`，不得改成两个P engine冒充P1D2。若调用成功但内部串行，记录真实结果并继续。

回传必须报告：W60/R60、run分支、容器/GPU/IB、24 shard完成度、外置raw绝对路径、raw文件/记录数、方差追加、Phase51与matched-solo两套WAPE、scientific outcome、结果目录、README/summary/logs/DONE/manifest及可得/不可得结论。
