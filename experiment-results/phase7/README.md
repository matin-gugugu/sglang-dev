# Phase 7：B200 跨节点连续通信代价曲线

## 1. 当前状态

本阶段将 Phase 6 已固定的 Qwen3-8B `PatternDemand` 映射到真实跨节点链路，
补齐第二阶段的拓扑相关代价：

```text
C(topology, op, payload_bytes, group_size)
```

截至当前：

- node55 已确认有 8 × NVIDIA B200，GPU 间为 NV18；
- 节点有 8 条活跃 RoCE 网卡，GPU0–GPU7 各有一条近端 NIC；
- Kubernetes 标签为 `node.network.rdma/pod=WLF2-C2-P102`；
- 集群中可见多个同型号、同 RDMA pod 的 `ge151` 节点，可用于 L2；
- 当前只明确授权 node55，尚未在其他节点启动任务；
- 当前可见的 B200 节点都位于同一个 RDMA pod，尚未找到硬件一致且拓扑标签
  可验证的跨 RDMA pod B200 节点。

因此，本阶段先完成多节点采集实现。取得第二个已授权 B200 节点后即可采集
L2；L3 必须在拓扑标签可验证后再采集，不能仅根据 IP 段或主机名猜测“跨机架”。

## 2. 拓扑口径

本实验不直接把模糊的“近/远”写入结果，而是保存可核验的物理标签：

| 论文层级 | 实验标签 | 判定条件 |
|---|---|---|
| L1 | `single-node-nvlink` | 同一台 B200 节点，GPU 间 NVLink |
| L2 | `cross-node-same-rdma-pod` | 两台 B200，`node.network.rdma/pod` 相同 |
| L3 | `cross-node-cross-rdma-pod` | 两台同代 GPU，RDMA pod 不同，且 PAZ/DC 关系明确 |

如果最终无法取得 L3 物理节点，网络仿真必须单独命名为
`emulated-cross-rack`，不能与真实 L3 测量混写。

## 3. 测量协议

- 原语：NCCL AllReduce、AllGather；
- group size：TP ∈ {2, 4, 8}；
- 两节点均匀放置：
  - TP=2：每节点 1 rank；
  - TP=4：每节点 2 ranks；
  - TP=8：每节点 4 ranks；
- payload：1 KiB–128 MiB 的二次幂点，并加入 48 KiB；
- 每点预热 30 次，正式采样 100 次；
- 每条曲线 5 次独立进程级重复；
- 默认使用 `rendezvous` 计时：
  每次正式 collective 前完成一次全 Rank barrier，再测各 Rank 本地 CUDA
  Event call-envelope，标签取同一次调用的 Rank 最大值；
- 同时保存每个 Rank 的原始 duration，用于 intrinsic 下包络和稳定性诊断。

该计时口径不要求跨节点 CUDA timestamp 可比较，适合作为隔离 collective 的
第二阶段代价曲线。未来采集跨节点完整推理 ground truth 时，仍需使用 PTP/时钟
同步或显式 rendezvous timing，不能直接比较不同节点 profiler 的绝对时间戳。

## 4. 双节点执行方式

两个节点必须使用相同代码提交、SGLang 镜像、PyTorch/CUDA/NCCL 版本，并同时
启动同一个 case。以 node55 为 rank 0、第二台已授权机器为 rank 1：

node 0：

```bash
NODE_RANK=0 \
MASTER_ADDR=<node0-bootstrap-ip> \
MASTER_PORT=29600 \
TOPOLOGY=cross-node-same-rdma-pod \
NCCL_SOCKET_IFNAME=bond0 \
bash scripts/run_b200_multinode_collective_curve_case.sh \
  tp2 all_reduce 0
```

node 1：

```bash
NODE_RANK=1 \
MASTER_ADDR=<node0-bootstrap-ip> \
MASTER_PORT=29600 \
TOPOLOGY=cross-node-same-rdma-pod \
NCCL_SOCKET_IFNAME=bond0 \
bash scripts/run_b200_multinode_collective_curve_case.sh \
  tp2 all_reduce 0
```

正式实验遍历：

```text
TP        = 2, 4, 8
op        = all_reduce, all_gather
repeat_id = 0, 1, 2, 3, 4
```

首次只运行 TP=2、AllReduce、repeat=0 的 smoke test，并设置
`NCCL_DEBUG=INFO` 核验日志确实选择 IB/RoCE transport，而不是 Socket fallback。
transport 未核验前不得开始完整套件。

## 5. 每条记录的可信度字段

跨节点记录除原有 latency、payload 和 Rank samples 外，还保存：

- `topology`、`transport_label`、`timing_mode`；
- `hostnames`、`node_count`；
- `rank_layout`：global rank、local rank、hostname、GPU UUID；
- `MASTER_ADDR`、`MASTER_PORT`；
- NCCL HCA、Socket interface、GID、GDR、algorithm 等环境变量；
- Git、PyTorch、CUDA、NCCL 和 GPU 信息。

结果仍保持：

- `payload_bytes` 是代表 Rank 的逻辑消息大小；
- `samples_us` 是 group-level collective 的逐次最大 Rank envelope；
- bytes 和 calls 不跨 Rank 求和；
- AllReduce/AllGather 分别使用对应的 ring 等效 bytes 因子。

## 6. 结果评估

完成 L2/L3 后进行四组分析：

1. 绘制 `topology × op × TP × payload` 连续延迟和有效带宽曲线；
2. 比较 L1/L2/L3 的小消息启动平台、带宽饱和点和大消息斜率；
3. 将同一套 Qwen3-8B PatternDemand 分别卷积三套拓扑曲线，得到候选部署时间；
4. 在各拓扑上重复四模型消融：
   total bytes、三硬桶、连续直方图、连续直方图 + DNN residual。

跨节点评测必须传入对应拓扑的 NCCL 曲线并启用：

```bash
python scripts/evaluate_qwen3_8b_expanded_models.py \
  --curve-mode nccl-only \
  --nccl-curve <topology-summary>/collective_curve_summary.csv \
  ...
```

这是因为 SGLang CustomAllReduce 仅用于单节点；L2/L3 的小消息也必须查询
NCCL/RoCE 曲线，不能继续复用 L1 的 CustomAllReduce 代价。

关键验收条件：

- 五重复的曲线中心值和尾部稳定；
- NCCL transport 与拓扑标签一致；
- 等总 payload 对照在不同拓扑下仍能区分消息形态；
- 连续结构模型在未见 workload 上显著优于 total bytes；
- DNN 只学习结构模型残差，不直接绕过 PatternDemand。
