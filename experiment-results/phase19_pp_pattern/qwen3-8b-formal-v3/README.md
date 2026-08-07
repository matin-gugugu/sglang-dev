# Phase 19：Qwen3-8B 纯 PP PatternDemand 正式矩阵

## 实验目标

在固定 `TP=1` 的前提下，只研究纯流水线并行（PP）的通信需求。输入为工作负载、PP size 与 PP microbatch 策略，输出发送端的：

`phase × PP boundary × tensor_name × exact payload × calls` 直方图。

每条 P2P 张量传输只在发送端计数一次，接收端不重复累计。采集采用 histogram-only，不保存逐事件 raw trace。

## 正式网格

- 模型：Qwen3-8B；
- TP：固定为 1；
- PP size：2、4、8；
- PP microbatch：1、4、16；
- 每个单元：13 种 workload × 3 次重复，共 39 个请求批次；
- 总计：9 个实验单元、351 个请求批次、2,376 个逻辑请求；
- workload 覆盖 fixed `B/L/M` 网格、mixed/longtail Decode，以及跨 4,096-token 阈值的 chunked Prefill。

## 完整性结论

最终目录为 `qwen3-8b-formal-v3`。矩阵状态为 PASS，9/9 个单元均具有 DONE 标志，且全部满足：

- 实际输出长度与请求长度完全一致；
- PP rank 文件数量与 PP size 一致；
- 所有前向边界均采集到 proxy tensor；
- 最后一个 stage 不重复记录 proxy send；
- 所有正式 workload 均能映射到对应直方图；
- histogram-only 文件不含 raw events。

旧的 formal-v1/v2 仅用于定位周期刷新造成的尾部落盘问题，不属于正式数据。formal-v3 已采用逐事件快照，PP2/mb1 最后一个 chunk workload 恢复为 54 calls、402,997,248 logical bytes，与跨 PP 对照一致。

## 初步结果与作用

以 `B=16,L=512,M=32` 为例，在单条 PP 边界上合计三次重复：

| microbatch | calls | logical bytes | Decode payload | Prefill 大消息 payload |
|---:|---:|---:|---:|---:|
| 1 | 3,072 | 427,032,576 | 8 KiB | 4 MiB |
| 4 | 768 | 427,032,576 | 32 KiB | 16 MiB |
| 16 | 384 | 427,032,576 | 64 KiB | 32 MiB |

这组近等总 logical bytes 对照说明：microbatch 策略主要改变 calls 与消息大小分布，而不一定改变总字节；因此只使用 total bytes 会丢失 RTT/启动次数差异。

PP=2/4/8 分别具有 1/3/7 条前向 stage boundary，通信总需求随实际经过的边界累加。mixed/longtail Decode 会随着 active batch 下降产生多尺度 payload，而不是单一小消息。

因此，PP 分支应使用边界感知的 PatternDemand：

`D_PP = {boundary, phase, tensor type, payload, calls}`。

本阶段只验证通信需求画像。`client_results.jsonl` 中的端到端 wall time 用于运行审计，不作为纯通信时间真值。下一阶段需要构造 PP 的 H0/残差预测器，并与 P2P `payload × topology → latency` 连续代价曲线结合。

## 文件说明

- `matrix_summary.json`：9 个实验单元的总审计；
- `pp*/mb*/audit.json`：单元级完整性检查；
- `pp*/mb*/client_results.jsonl`：请求长度、实际输出长度和运行时元数据；
- `pp*/mb*/profile/*.json`：各 PP rank 的发送端 histogram-only 快照；
- `pp*/mb*/run_config.json`：模型、PP、microbatch 与启动参数；
- `pp*/mb*/server.log`：单元级服务器日志。
