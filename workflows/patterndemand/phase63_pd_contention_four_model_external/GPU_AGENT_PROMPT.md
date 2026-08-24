# Phase63 GPU Agent执行提示

你是GPU执行端。exact W63由控制端指定。完整阅读本目录全部合同和脚本后创建唯一`run/phase63-<environment>-<date>`分支。

目标：在生产Mooncake RDMA/dma-buf完成48个三rank shard，对四个从未参与R61拟合的模型布局做外部验证，Git外保存raw，形成唯一父提交为W63的R63。

## 第一资源红线

Phase63绝不需要4台node同时分配。必须一次只运行一个measurement shard：L1峰值1台node，L2/L3峰值2台node，全局峰值2台node和3个GPU进程。L1/L2/L3允许使用三个独立scheduler allocation顺序完成；完整inventory即使出现3–4个不同host，也不表示这些host需要共存。若资源规划器准备申请4-node allocation或并行两个shard，立即停止并按`RESOURCE_ALLOCATION_CN.md`纠正，不得以资源不足记录实验BLOCKED。

硬边界：

1. 只测`payload_pair_grid.json`中的四模型、40组pair；Qwen3-8B和DeepSeek-V2-Lite不得替代held-out模型。
2. R61公式、三个系数和所有阈值绝对不改；不训练、不拟合、不加载权重、不运行推理、不下载模型。
3. inventory四个endpoint是GPU slot，不是四台node。单shard固定3进程；L1一台，L2/L3两台。replica、配置、模型、repeat和拓扑全部顺序执行，禁止同时运行两个shard。
4. 优先复用Phase62 placement；不可用时只按资产/rack/fabric元数据冻结同类替代，禁止测速后选快placement。任何plan修改都废弃当前attempt并从repeat0重来。
5. 每个shard先5 repeat；只按`raw_status.py`追加到7或9。不删异常、不挑快replica、不降低重复数。
6. 不允许TCP/MNNVL/NVLink旁路/staging/custom-pool回退。
7. raw、权重、缓存、PID、密钥永不入Git；禁止`git add .`。

允许按workflow的AUTO与RECORD_AND_CONTINUE原则做端口/hostname/IB可达性诊断、有限重试和冻结前的同类资源替换，但不得改变实验语义。

回传必须报告W63/R63、run分支、容器/GPU/IB、placement是否精确复用Phase62、48 shard完成度、Git外raw绝对路径、文件/记录数、追加重复、四模型整体/逐模型/24个细切片WAPE与bias、合并六模型指标、scientific outcome、结果目录、README/summary/logs/DONE/manifest以及可得和不可得结论。
