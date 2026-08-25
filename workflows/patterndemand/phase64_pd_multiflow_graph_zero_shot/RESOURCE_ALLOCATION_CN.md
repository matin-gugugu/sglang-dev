# Phase64 资源申请说明

- 不是四节点实验。任何时刻最多两个节点。
- L1：1 个至少 8 卡的节点；单个 shard 启动 4 或 5 个进程。
- L2/L3：2 个节点；A 侧最多 4 个进程，B 侧最多 4 个进程，但单个图实际总进程仍不超过 5。
- 每次只能跑一个 measurement shard；replica0/1、L1/L2/L3 可分开顺序申请。
- P1D4/P4D1 world size=5；两种 P2D2 world size=4。
- 两个 placement replica 可以复用同一主机/主机对，但物理 GPU tuple 必须不同；也可以使用不同主机。
- 不加载模型，不需要模型权重，不下载文件；只分配符合布局大小的 CUDA buffer 并调用生产 Mooncake RDMA。
