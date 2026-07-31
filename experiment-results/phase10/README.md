# Phase 10：双模型多支撑点 PatternDemand 实验

## 1. 实验目的

Phase 9 的规则网格全部是单一 payload 支撑点，解析公式可以精确重建结果，
但实验不足以证明“消息尺度分布”比“总通信字节”提供了更多信息。本阶段使用
Qwen3-8B 和 DeepSeek-V2-Lite，在 TP=2 下增加两类结构性变量：

1. 混合输出长度 continuous batching，使 Decode 阶段的
   `active_batch(t)` 随生成过程下降；
2. chunked prefill，使相同输入长度在不同 chunk size 下形成不同的消息次数
   和消息尺度分布。

本轮仍然只预测拓扑无关的第一阶段 `PatternDemand`，不把并发运行时延作为
第二阶段的正式标签。

## 2. 实验规模与口径

| 项目 | 设置 |
|---|---|
| 模型 | Qwen3-8B、DeepSeek-V2-Lite |
| 并行配置 | TP=2 |
| 重复次数 | 每种配置 3 次 |
| mixed Decode | 2 模型 × 3 输出分布 × 3 次 = 18 个实验单元 |
| chunked Prefill | 2 模型 × 3 个 chunk size × 3 次 = 18 个实验单元 |
| 原始聚合记录 | mixed 18 条；chunked 216 条；共 234 条 |
| 采集模式 | `histogram-only`，不保存大体积 raw events |
| 验证结果 | 36/36 个实验单元通过固定长度、跨 rank 一致性和结构公式校验 |

`count` 采用 group-level collective 次数，`payload_bytes` 采用一个代表 rank
的逻辑消息大小。TP=2 的 ring-equivalent 量由原语和 group size 后处理得到。

## 3. 实验一：相同粗粒度输入，不同 Decode 消息直方图

三种请求组均固定：

```text
batch_size = 8
prompt_len = 512
max_output_len = 64
sum(actual_output_len) = 256
```

只改变 batch 内实际输出长度分布：

| profile | 8 个请求的实际输出长度 |
|---|---|
| balanced | 16, 16, 16, 16, 32, 32, 64, 64 |
| staircase | 16, 16, 16, 24, 32, 40, 48, 64 |
| bimodal | 8, 8, 16, 16, 16, 64, 64, 64 |

三组负载具有相同的 `(B,L,Mmax)`、相同输出 token 总数、相同 collective
总次数和相同总 payload，却产生不同的 payload 支撑点与 calls 分布。

### Qwen3-8B

| profile | Decode supports | calls | total payload |
|---|---:|---:|---:|
| balanced | 3 | 4599 | 148,307,968 B |
| staircase | 6 | 4599 | 148,307,968 B |
| bimodal | 3 | 4599 | 148,307,968 B |

### DeepSeek-V2-Lite

| profile | Decode supports | calls | total payload |
|---|---:|---:|---:|
| balanced | 3 | 3465 | 55,869,440 B |
| staircase | 6 | 3465 | 55,869,440 B |
| bimodal | 3 | 3465 | 55,869,440 B |

如果只使用粗粒度 `w=(L,Mmax)`，三组输入和预测完全相同。该表示虽然对
calls 和总 payload 的误差为 0，但消息直方图平均 TV 距离为 65.61%，最大
达到 82.54%。加入输出长度生存曲线

```text
A(t) = number of requests whose actual_output_len > t
```

后，可由 `A(t)` 的各个平台期精确恢复 Decode 直方图，calls、总 payload 和
直方图误差均为 0。

这比“近似相同总 payload”对照更强：两组的聚合总量完全相同，但消息形态仍然
不同，直接证明总通信字节和总 calls 不能唯一决定通信代价。

## 4. 实验二：相同 workload，不同 chunked Prefill 结构

固定模型、TP、batch size、输入长度和输出长度，只改变：

```text
chunk_size ∈ {1024, 2048, 4096}
```

在每个阈值附近测试：

```text
L ∈ {2047, 2048, 2049, 4095, 4096, 4097}
B ∈ {1, 4}
M = 8
```

典型对照：

- `L=2047/2048` 时，chunk 1024 相比 chunk 2048 的 Prefill calls 为 2 倍；
- `L=2049` 时，两者 calls 比为 1.5；
- `L=4095/4096/4097` 在 chunk 2048 与 4096 间呈现同类边界跳变；
- 对照组总 payload 相同，但直方图 TV 距离最高为 1。

共得到 24 组“相同 workload、不同 chunk size”的碰撞对照。若配置输入不包含
chunk size，粗粒度预测的平均 calls APE 为 63.89%，平均直方图 TV 距离为
88.89%，最大为 100%。加入 chunk size 后，按每个 chunk 的有效 token 数可
精确恢复 calls、总 payload 和消息直方图。

## 5. 预测评估

| 场景 | 表征/预测器 | calls APE | total payload APE | histogram TV |
|---|---|---:|---:|---:|
| mixed Decode | 粗粒度 `w=(L,Mmax)` | 0% | 0% | 65.61% |
| mixed Decode | output survival formula | 0% | 0% | 0% |
| chunked Prefill | 配置中省略 chunk size | 63.89% | 0% | 88.89% |
| chunked Prefill | chunk-aware formula | 0% | 0% | 0% |

这里的 0% 总 payload 误差并不代表粗粒度预测有效，反而说明只评估总量会掩盖
严重的消息形态误差。第一阶段必须同时评估：

- `calls`；
- `total_payload_bytes`；
- payload 直方图距离；
- 支撑点数量和位置；
- 在进入第二阶段后，对最终通信时延预测误差的改善。

## 6. 对论文模型的修正

原定义 `w=(L,M)` 对单请求规则网格成立，但无法表示 continuous batching 中
的混合输出长度。建议把工作负载输入扩展为：

```text
w = (L distribution, output-length distribution or A(t), batch/arrival features)
```

把执行配置扩展为：

```text
c = (
  model,
  parallel_form,
  parallel_size,
  scheduling_policy,
  chunk_size,
  max_running_requests
)
```

第一阶段仍输出按 `phase × op × payload interval` 聚合的 calls、payload 和
equivalent rounds，但内部可以保留连续 payload 直方图，第二阶段再查询连续
的 `op × payload × group_size × topology → latency` 代价曲线。

这轮结果说明实验“厚度”并不来自强行制造非线性，而来自证明：

1. 相同标量总量可以对应不同消息尺度分布；
2. 请求长度分布会改变阶段内部的 active batch，产生多支撑点直方图；
3. 调度策略会在阈值处引发 calls 和消息尺度的离散变化；
4. 粗粒度输入存在不可辨识性，必须增加具有物理意义的结构特征。

因此第一阶段现阶段应采用结构公式抽取 PatternDemand；神经网络继续作为最终
通信时间模型的残差校正器。只有第二阶段加入真实拓扑代价、通信重叠和资源竞争
后出现非零结构残差，才有充分理由启用非线性残差网络。

## 7. 结果文件

```text
experiment-results/phase10/
├── README.md
├── multiscale_pattern_demand/
│   ├── qwen3-8b/
│   │   ├── mixed_same_coarse/
│   │   └── chunked_prefill/
│   ├── deepseek-v2-lite/
│   │   ├── mixed_same_coarse/
│   │   └── chunked_prefill/
│   ├── *_driver.log
│   ├── deepseek_chunked_r2_c4096_repair.log
│   └── final_validation.log
└── multiscale_pattern_analysis/
    ├── multiscale_pattern_analysis.png
    ├── mixed_summary.csv
    ├── mixed_collision_predictions.csv
    ├── chunked_summary.csv
    ├── chunked_collision_pairs.csv
    ├── prediction_metrics.csv
    ├── summary.json
    └── analyze.log
```

其中 DeepSeek `c4096/r2` 曾因加速分片与主进程重叠产生重复追加；已停止重复
进程、删除该实验单元的完成标记并单独重跑。正式结果恢复为预期 12 行，并再次
通过全目录校验。修复过程保留在日志中，便于审计。

执行与分析脚本：

```bash
bash scripts/run_cross_model_multiscale_pattern_dataset.sh all all
python scripts/analyze_multiscale_pattern_demand.py
```

## 8. 下一步

1. 把 `A(t)`、输出长度分布和 chunk 配置接入第一阶段 PatternDemand 接口；
2. 用连续的 collective 延迟曲线替代三个固定消息桶；
3. 在单机上先测量 `op × payload × TP → latency`，完成“总 bytes”与“消息
   直方图”两条路径的时延预测消融；
4. 获得第二台机器后补充跨节点同机架与跨机架实测，不把公开参数仿真写成
   物理实测；
5. 再增加 PP/PD 分离或含 AllGather/ReduceScatter 的配置，扩展通信原语覆盖。
