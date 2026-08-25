# Phase64 资源申请说明

- 推荐一次向调度器申请/保留 4 个节点；如果目标环境的四节点整组更容易排队，就使用 `FOUR_NODE_SINGLE_ALLOCATION`。
- 保留的 4 个节点是候选 placement 池，不是四节点 collective。单个 measurement shard 最多激活其中 2 个节点。
- L1：从保留池选 1 个至少 8 卡的节点，启动 4 或 5 个进程。
- L2/L3：从保留池选符合 rack/fabric 定义的一对节点；A/B 两侧合计最多 5 个进程。
- 每次只能跑一个 measurement shard。另两个已保留节点必须保持没有 Phase64 测量进程，避免网络互扰。
- 如果四节点整组无法同时覆盖 L1/L2/L3 的元数据定义，允许使用 `SEQUENTIAL_TOPOLOGY_EPOCHS` 分批申请 1/2 个节点；这只是资源组织方式变化，不改变实验点。
- P1D4/P4D1 world size=5；两种 P2D2 world size=4。
- 两个 placement replica 可以来自四节点池里的不同节点/节点对，也可以在同一主机/主机对上使用不同物理 GPU tuple；它们顺序测量。
- 不加载模型，不需要模型权重，不下载文件；只分配符合布局大小的 CUDA buffer 并调用生产 Mooncake RDMA。
