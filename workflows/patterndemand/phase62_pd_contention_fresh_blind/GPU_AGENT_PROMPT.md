# Phase62 GPU Agent执行提示

你是GPU执行端。exact W62由控制端指定。完整阅读本目录合同和脚本后创建唯一run/phase62-<environment>-<date>分支。

目标：在生产Mooncake RDMA/dma-buf完成24个三rank fresh-blind shard，Git外保存raw，聚合并形成唯一父提交为W62的R62。

硬边界：

1. 只测reserved_future_blind pair，禁止Phase60 development pair。
2. R61公式、三个系数和阈值绝对不改；不训练、不拟合、不加载模型。
3. inventory四个endpoint是GPU slot，不是四台node。单shard固定3进程；L1一台，L2/L3两台，所有shard可顺序执行。
4. 每个Phase62 endpoint tuple必须未在Phase60出现；每种拓扑至少一套host signature也全新。只按资产/rack/fabric元数据选择，禁止测速后挑快placement。
5. 每个shard先5 repeat；只按raw_status追加到7或9。不删异常、不挑快replica、不降低重复数。
6. 不允许TCP/MNNVL/NVLink旁路/staging/custom-pool回退。
7. raw、权重、缓存、PID、密钥永不入Git；禁止git add .。

回传必须报告W62/R62、run分支、容器/GPU/IB、freshness证据、24 shard完成度、Git外raw绝对路径、文件/记录数、追加重复、未修正与修正后整体和六切片WAPE/bias、scientific outcome、结果目录、README/summary/logs/DONE/manifest以及可得和不可得结论。
