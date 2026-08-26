# Phase66资源申请说明

- 推荐一次保留4个节点；这4个节点只是候选placement池，不是四节点同时通信。
- L1单shard激活1个至少8卡节点；L2/L3单shard激活2个节点。
- P1D4/P4D1启动5个GPU进程；两种P2D2启动4个GPU进程。
- 所有48个shard顺序执行；禁止同时跑两个shard，避免互相抢fabric。
- 若四节点池不能覆盖L1/L2/L3元数据，可用`SEQUENTIAL_TOPOLOGY_EPOCHS`分批预约1/2节点；实验点和freshness门不变。
- inventory的A0-A3/B0-B3是GPU插槽，不是8个节点；所有tuple必须避开Phase64，且每种拓扑至少一个host signature全新。
- 不加载模型权重、不做推理、不下载文件，只分配CUDA buffer并调用生产Mooncake RDMA。
