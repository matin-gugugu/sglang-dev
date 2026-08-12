# 截至目前实验结构总导引

更新时间：2026-08-12

用途：恢复研究主线，统一统计口径，解释 Phase 0–28C 的作用与结论边界，并作为
后续第一阶段真实流量预测器、L2/L3 代价曲线和调度器闭环的总索引。

## 1. 一句话总览

本研究不是直接用 DNN 从 workload 黑盒回归通信时间，而是建立下面的结构化链路：

```text
服务常态流量画像 + 模型结构 + 固定执行策略 + 已确定的并行配置
    ↓
第一阶段：预测单位请求规模下的拓扑无关 ProfileDemand 消息直方图
    ↓
第二阶段：查询候选拓扑的连续通信代价曲线
    ↓
T_struct = Σ predicted_calls × C(topology, op, payload, TP, backend)
    ↓
可选 ResidualNN：只校正结构公式未覆盖的重叠、排队和运行时状态
    ↓
候选部署方案的通信时间与不确定性
    ↓
调度器比较 placement 和拓扑层级（不选择TP/PP size）
```

截至目前：

- 已完成三个模型、TP=2/4/8 的实测 PatternDemand 采集和结构验证；
- 已在单节点 B200 的 L1 拓扑上测得与 `op × payload × TP × backend` 对齐的连续
  代价曲线；
- 修正后的 Phase 14F 使用“实测 PatternDemand + L1 代价曲线”预测通信时间，
  整体 MAPE 为 4.43%；
- Phase 14G 已完成严格同口径消融和 payload 支撑点留出；
- Phase 15 已固定 BurstGPT/Mooncake 数据版本，生成 66,642 个因果窗口，并完成
  Qwen3-8B 的 20 窗口 × TP2/4/8 histogram-only GPU 回放；
- Phase 15C 曾训练“历史画像 → 下一窗口代表性 draining batch”模型，普通测试 L1
  结构误差约 79%；该任务属于时间预测，已经确认偏离本研究正式目标，仅保留为诊断；
- 正式第一阶段现改为“模型结构 + 服务常态流量画像 + 执行策略 + 已确定的 TP 配置 → 每 1000
  请求的 ProfileDemand”，输出每个 `phase×op×payload-bin` 的 calls 与 logical bytes，
  不预测下一时间窗口的具体请求；
- Phase 16 已完成 24 个 BurstGPT/Mooncake 常态画像、三档执行策略、三个模型和
  TP2/4/8 的正式 GPU 网格：4905 个 histogram-only microbatch workloads 聚合为
  1296 条阶段标签，9/9 运行审计通过，smoke 三重复完全一致；
- 四方法留出评测已完成。正式 `H0 + bounded residual DNN` 在未见模型、未见策略、
  未见 TP 上的 L1 结构代价 MAPE 分别为 8.71%、8.05%、7.55%；完整未见流量 segment
  为 11.34%，弱于纯 H0 的 9.20%，因此未知流量域必须保留 H0 回退；
- Phase 17 已把 1296 条正式标签及其 out-of-fold 预测直方图代入 1 条实测 L1 曲线和
  8 条参数化 L2/L3 曲线，完成 total-bytes、calls+bytes、三桶、12 桶和预测直方图的
  同口径代价敏感性分析，并完成仅考虑通信代价的 batching 策略选择与 regret 评测；
- Phase 19–20 已完成 Qwen3-8B 纯PP正式GPU PatternDemand和精确workload结构公式验证；
- Phase 21/22 已完成PP online arrival扩展和低维画像fixed-draining predictor诊断；online
  只作为扩展，首版主目标保持fixed-draining；
- Phase 23 已在 PP2/4/8 × microbatch 1/4/16 上完成十重复严格复核：54/54阶段组的精确
  直方图完全一致，calls、bytes、histogram L1和EMD相对H0均为零误差；正式结果已由
  `fe740906`提交、推送并ff-only同步到本地；
- Phase 24已对24个历史窗口完成TP/PP H32/H64/H128/Hfull收敛：H32不足，
  H64/H128虽降低均值误差，但没有任一规模同时通过全部预注册门槛；因此首版使用
  full-window fixed-draining H0作为teacher label；
- Phase 24旧PP静态公式结论已被Phase 25D scheduler-faithful teacher取代；正式PP
  H32/H64/H128→Hfull calls MAPE为71.99%/33.50%/18.65%，没有规模通过全部门槛；
- Phase 24正式资产已由`bb490ad6`提交、推送并在node55 ff-only同步，两端manifest均通过；
- Phase 26A–26D已把TP/PP监督统一切换到Hfull并完成重训留出：TP旧residual因仅5个训练
  画像而未跨域泛化，H0暂作baseline/fallback，但最终架构仍是`H0 + DNN residual`；旧PP
  画像在MB4/MB16上仍不足，direct DNN在TP/PP均拒绝；
- Phase 27A–27D用60个新窗口和事前30/12/18划分验证PP调度敏感画像。18个独立确认
  窗口上，增强residual把总体calls MAPE从H0的62.13%降至25.84%、TV从0.2733降至
  0.1844、common-cost MAPE从10.96%降至6.52%；相对旧特征residual也继续改善；
- Phase 27C冻结的MB1/4/16全residual映射只部分通过：MB1因calls与cost退化回退H0，
  MB4/MB16保留增强residual候选；MB16 calls MAPE仍为55.55%，尚未统一收敛；
- L2/L3仍缺物理代价曲线，PP P2P连续代价曲线和PD正式闭环尚未完成，调度器尚未接入。

### 1.1 TP预测器口径修正：H0是结构先验，不取代DNN

TP第一阶段的最终研究架构固定为：

```text
低维历史流量画像 + 模型结构 + 固定TP配置 + 固定执行策略
                          ↓
                    H0结构先验直方图
                          +
            DNN学习H0到Hfull teacher的有界残差
                          ↓
                    最终TP消息直方图
```

Phase 26D中纯H0优于当时的bounded residual，只能说明由5个独立训练画像和55列旧特征
训练出的TP residual没有跨域泛化，不能解释为TP不需要DNN。H0的职责是提供可解释结构先验、
保底baseline和域外回退；DNN仍负责学习低维画像不能由结构公式直接表达的batch形成、prefill
token-budget packing、decode生存过程、batch尾部碎片和策略压力等残差。

下一轮TP闭环应复用事前冻结的数据纪律，扩展为30个训练画像、12个验证画像和18个独立确认
画像。每个画像覆盖3个模型、TP2/4/8、latency/balanced/throughput及prefill/decode，形成：

```text
60画像 × 3模型 × 3个TP size × 3策略 × 2 phase = 3,240条phase rows
```

完整请求列表只用于离线生成Hfull teacher和聚合batching-sensitive低维特征，不进入部署输入。
模型继续比较H0、旧residual、新增强residual和direct控制组；预测与checkpoint必须先冻结hash，
再开放独立确认真值。只有新residual在calls/bytes、TV/EMD和统一曲线cost指标上通过独立确认，
才能升级为TP默认预测器；否则H0继续作为fallback，但不取消DNN研究路线。placement/topology
连续代价曲线可以并行推进，但不能替代这一TP预测器训练闭环。

## 2. 两阶段必须严格分开

### 2.1 第一阶段：PatternDemand

第一阶段回答：

> 给定工作负载、模型和执行配置，会产生什么通信需求？

它不需要知道任务最终被放在 L1、L2 还是 L3。推荐输出为：

\[
\widehat H^{1000}_{\phi,o,m,p}
=
F_{pattern}(F_{profile},F_{model},F_{execution},F_{config})
\]

其中：

- \(\phi\)：Prefill 或 Decode；
- \(o\)：canonical logical collective，例如 AllReduce、AllGather、ReduceScatter、
  Send/Recv；`raw_op`（包括 fused 变体）另行保留，供第二阶段 backend-aware 细化；
- \(m\)：代表 rank 的逻辑 payload；
- \(p\)：group size，例如 TP2/4/8；
- \(H^{1000}\)：按服务常态请求分布处理每 1000 个请求时的 group-level collective 次数。

精确或对数 payload 直方图用于正式计算，small/medium/large 三桶只用于论文展示和
消融，不再作为唯一表征。

正式输出采用两个口径：结构需求以 `calls / 1000 requests` 归一化，便于比较模型、
策略和 TP；容量强度再结合常态 RPS 换算：

\[
H^{rate}_{\phi,o,u}
=
\frac{\lambda}{1000}H^{1000}_{\phi,o,u}.
\]

因此历史 timestamp 用于提取服务画像和 RPS，而不是监督“下一时间窗口”的具体请求。

### 2.2 第二阶段：TopoCostProfile

第二阶段回答：

> 某种原语、消息大小和 group size 放到候选拓扑上，单次需要多久？

推荐形式为：

\[
C_\pi(o,m,p,backend,algorithm)
\]

其中 \(\pi\) 为 L1 同节点、L2 同机架跨节点或 L3 跨机架。最终：

\[
\widehat T_{struct}
=
\sum_{\phi,o,m,p}
\widehat H_{\phi,o,m,p}
C_\pi(o,m,p,backend)
\]

三桶固定 BW/RTT 可以作为解释近似，但正式计算优先使用连续实测时间曲线，因为真实
链路存在启动平台、带宽饱和、协议和算法切换。

### 2.3 当前最容易混淆的边界

| 任务 | 当前状态 | 属于哪一阶段 |
|---|---|---|
| 运行模型后采集消息直方图 | 已完成且稳定 | 第一阶段的数据标签 |
| 验证 \(B,L,M,TP,chunk,A(t)\) 如何改变直方图 | 已完成多轮机理实验 | 第一阶段 |
| 从历史画像预测下一窗口一个代表性 batch | Phase 15C 已完成但目标偏离 | 仅保留诊断 |
| 服务常态画像 + 模型结构 + 策略 + TP → 每1000请求直方图 | Phase 16 已完成首版 | 第一阶段正式目标 |
| 实测直方图 + L1 曲线预测时间 | Phase 14F 已完成，MAPE 4.43% | 两阶段组合验证 |
| 预测直方图 + L1 曲线传播误差 | Phase 16 已完成四类分组留出 | 第一阶段误差传播 |
| 预测直方图 + 参数化 L2/L3 曲线 | Phase 17 已完成敏感性分析 | 两阶段组合仿真，不是物理实测 |
| L2/L3 连续代价曲线 | 尚未物理实测 | 第二阶段 |

因此，L2/L3 不是补全第一阶段所需的数据；它们负责补全第二阶段。

## 3. 最初设计到当前设计的演化

| 最初设计 | 当前优化版本 | 修改原因 |
|---|---|---|
| \(w=(L,M)\) | 服务画像 \(\rho=\{P(L,M),\lambda,burst\}\) | 从单请求推广到模型服务的常态 workload 分布 |
| 只输入固定人工网格 | 模型结构 + 服务画像 + 执行策略 + 已确定的TP配置 | 估计未完整实测服务配置的通信需求 |
| 主要使用 per-token bytes/calls | 保存完整 group-level 消息直方图 | 归一化会掩盖 batch、chunk 和 active batch 的结构变化 |
| small/medium/large 三硬桶 | 精确/对数 payload 直方图为主 | 链路成本随 payload 连续变化并可能发生算法切换 |
| payload 是模糊的通信 bytes | 代表 rank 的逻辑输入张量大小 | 避免把单 rank、所有 rank、等效 bytes 和 wire traffic 混用 |
| calls 按进程累计 | group-level collective 次数 | 避免 TP 个 rank 重复累计同一次逻辑 collective |
| TP 只是类别 | 显式保留 group size、等效 bytes 和 rounds | TP 改变轮次、backend 和代价，即使逻辑 payload 不变 |
| Prefill 等于大包、Decode 等于小包 | phase 与 payload 尺度分开 | 短 Prefill 也可产生小包，长 Decode batch 也可产生较大消息 |
| 每桶固定 BW/RTT | 连续 \(C(op,payload,TP,topology,backend)\) | 表达启动平台、饱和和 backend/algorithm 切换 |
| 固定 rank kernel 时间是真值 | all-rank post-rendezvous 主标签 | 固定 rank 会混入提前到达后的等待 |
| DNN 直接拟合总时间 | 结构公式为主，DNN 只拟合残差 | 保留 PatternDemand 的解释性和拓扑可迁移性 |
| TP/PP/PD 同时铺开 | 先闭环 TP，再扩纯PP、纯PD | TP已闭环，纯PP底层链路已验证，PD尚未开始 |

## 4. 统计口径

### 4.1 logical calls

`calls` 是 group-level collective 次数，不是所有进程 kernel 数之和。若每个 forward
包含 73 次逻辑 AllReduce，则 TP2、TP4、TP8 都记录 73 次，而不是 146、292、584。

### 4.2 logical payload

`payload_bytes` 是代表 rank 传入一次 collective 的逻辑张量大小，不是所有 rank
求和，也不是实际链路 wire traffic。

### 4.3 等效 bytes 与 rounds

在当前 AllReduce ring-style 解释中：

\[
\alpha_{AR}(p)=\frac{2(p-1)}{p},\qquad
\beta_{AR}(p)=2(p-1)
\]

因此 logical calls、logical payload 和 `(raw_op,payload)` 直方图在当前路径上可以跨
TP 不变，但完整 PatternDemand 并没有跨 TP 完全相同，因为 group size、等效 bytes、
rounds、backend 和实际时间会变化。

### 4.4 时间标签

当前 L1 主标签为 all-rank post-rendezvous：

\[
T_{post}
=
\sum_e(\max_r f_{e,r}-\max_r s_{e,r})
\]

它从最后一个 rank 进入事件时开始计时，减少把 rank 提前到达等待混入通信结构代价。

## 5. 实验阶段总表

| 阶段 | 核心问题 | 主要结论 |
|---|---|---|
| Phase 0–1 | 埋点、统计契约、消息结构是否可靠 | histogram-only、group-level calls 和代表 rank payload 可稳定采集 |
| Phase 2–3 | 三硬桶是否足够 | 连续 payload 曲线和精确直方图明显优于 total bytes |
| Phase 4–6 | L1 时间真值与早期预测闭环 | post-rendezvous 比固定 rank 更稳定；早期结构+残差在 Qwen 上有效 |
| Phase 7 | L2/L3 工程协议 | runner 和元数据协议完成，但没有正式物理数据 |
| Phase 8–10 | 第二模型和多支撑点 | 输出分布、active batch、chunk 会造成无法由总量表示的结构变化 |
| Phase 11 | 双模型真实时间 | 相同总 payload 的不同直方图确实产生真实时间差；chunk 边界有时间跳变 |
| Phase 12 | 第三模型 Qwen3-30B-A3B | 完成 TP2/4/8、585 条 PatternDemand 记录 |
| Phase 13 | 三模型联合验证 | 精确直方图结构模型优于 total bytes，DNN 平均值略好但尾部不稳定 |
| Phase 14C | 三模型 TP2/4/8 扩展时间标签 | 162 配置、486 重复；无 backend 的简单模型在 Decode/模型留出上不足 |
| Phase 14D | TP 条件斜率 | 最优 12.15% MAPE，但 Decode、P95、整模型留出未达标 |
| Phase 14E | active-batch 摘要能否补足误差 | 只有 6 种 Decode 形态，复杂特征过拟合，MAPE 恶化至 25.25% |
| Phase 14F | 对齐真实 op/backend 的 L1 连续代价曲线 | 修正后 MAPE 4.43%，四项收敛门槛全部通过 |
| Phase 14G | 严格同口径消融与支撑点留出 | total bytes 46.80% → payload histogram 7.20% → raw op 4.39%；曲线留出 MAPE 2.35% |
| Phase 15A | BurstGPT/Mooncake 因果窗口和 GPU smoke | 66,642 窗口；Qwen3-8B 20 窗口 × TP2/4/8、120 条阶段标签全部通过 |
| Phase 15B | 真实 trace 的长 Prefill L1 补点 | 160–512 MiB、21 个 TP×payload 支撑点、10,500 样本，最大重复 CV 0.40% |
| Phase 15C | 下一时间窗口预测诊断（非正式目标） | 活跃窗口分类有效，但单个未来 batch 的直方图/代价预测误差高；该任务与服务常态画像到通信需求的正式目标不同，仅保留为失败边界 |
| Phase 16A–E | 正式 ProfileDemand 定义、12 桶选择、画像、结构特征和 H0 | 12 桶 calls+bytes 足够；H0 在受控 162 配置上 162/162 精确匹配 |
| Phase 16F | 三模型×TP2/4/8 正式 GPU 标签矩阵 | 4905 workloads → 1296 阶段标签；all-rank、固定实际输出、H0、重复稳定性全部通过 |
| Phase 16G | 四种预测方法与分组留出 | H0+bounded residual 对未见模型/策略/TP 有效；未见流量域应回退 H0；direct DNN 明显不稳 |
| Phase 17 | 参数化 L2/L3 与通信策略敏感性 | 1296 标签、5184 个 OOF 预测向量、9 条曲线；total bytes 丢失启动轮次，calls+bytes 在平滑 α–β 曲线上已近乎充分，12 桶的额外收益取决于曲线非线性 |
| Phase 18 | TP/策略候选离线决策pilot | 已完成但按当前目标降级；并行配置是预测输入，调度器不选择TP/PP size |
| Phase 19 | Qwen3-8B纯PP正式GPU PatternDemand | PP2/4/8×mb1/4/16，351批次、2376请求；相同bytes下calls最多相差8倍 |
| Phase 20 | 受控纯PP H0与DNN predictor | 精确workload的H0 calls APE 2.56%、bytes 0%、hist L1 0.0085；direct DNN不能替代公式 |
| Phase 21 | PP arrival-aware smoke | 证明online到达会改变calls/hist；只作扩展，不改变fixed-draining首版主目标 |
| Phase 21b–22 | PP低维画像offline/online predictor | fixed-draining bytes较准，但calls MAPE约41%–61%，低维画像到microbatch结构尚未收敛 |
| Phase 23 | PP fixed-draining十重复严格复核 | 9 cells全部PASS，54/54精确一致，H0 calls/bytes/L1/EMD全零误差；`fe740906`已归档同步 |
| Phase 24 | TP/PP代表请求规模收敛 | 24窗口、18配置；H32不足，H64/H128未通过全门槛；full-window teacher推荐；`bb490ad6`已归档同步 |

## 6. Phase 0–13 的关键证据

### 6.1 PatternDemand 不是 total bytes

Qwen3-8B 的近等总 payload 对照中，不同 Decode 形态拥有相同或接近总字节，但 calls
和消息尺度分布明显不同。Phase 11 进一步证明：

- 相同总 calls 和总 payload 的 mixed Decode，在两个模型上产生约 3.4%–4.9% 的
  真实时间跨度；
- 相同总 payload、不同 chunk 结构最高产生 1.332× 时间差；
- 24/24 个 chunk calls 跳变点都出现正向时间跳变；
- 测试集上连续直方图将 total-bytes MAPE 从 8.34% 降至 3.36%，P95 APE 从
  26.29% 降至 7.34%。

这支持核心论点：总通信量是直方图的一阶聚合，无法区分“大量微小消息”和“少量大
消息”，而两者支付的启动/RTT 成本不同。

### 6.2 输出长度分布和 chunk 不能省略

Phase 10 中，mixed Decode 固定 B、L、总输出 token、calls 和总 payload，只改变
输出长度分布。省略 \(A(t)\) 时直方图 TV 误差平均 65.61%，加入生存曲线后为 0。

Chunked Prefill 固定模型、B、L、M，只改变 chunk size。省略 chunk 时 calls 平均
误差 63.89%、直方图 TV 误差 88.89%，加入 chunk 配置后为 0。

### 6.3 三模型正式覆盖

当前模型包括：

- Qwen3-8B；
- DeepSeek-V2-Lite；
- Qwen3-30B-A3B。

Phase 12 为 Qwen3-30B-A3B 采集 TP2/4/8、三重复、共 585 条 PatternDemand
记录。Phase 13 将三个模型放在一起验证，说明精确 `(raw_op,payload)` 直方图具有
跨模型解释力；但 residual DNN 的平均误差和尾部误差方向不总是一致，不能仅凭平均
MAPE 宣称 DNN 全面优于结构模型。

## 7. Phase 14C–14E 为什么效果不好

这些不是在否定 PatternDemand，而是在定位“直方图怎样换算为时间”缺了什么。

### 7.1 Phase 14C：简单 TP×phase 截距不足

三模型、TP2/4/8、6 种 Decode 形态，共 162 个聚合配置、486 次原始重复。只使用
PatternDemand、TP 和 phase 的加性模型得到：

- 整体 MAPE 12.81%；
- Prefill 10.25%；
- Decode 17.92%；
- 整模型留出 15.15%–36.62%。

说明同一通信结构在不同 TP/backend 下不能只靠固定截距换算。

### 7.2 Phase 14D：增加斜率仍未收敛

描述性最优的 TP-conditioned slopes：

- 整体 MAPE 12.15%；
- Prefill 9.43%；
- Decode 17.58%；
- P95 APE 39.01%；
- 整模型留出为 17.92%–47.34%。

完整 TP×phase 交互因参数过多，MAPE 恶化至 22.11%；简单 ring-equivalent 也不能
解释真实 backend 代价。

### 7.3 Phase 14E：时序特征在小样本上过拟合

当前只有 6 种唯一 Decode 时序。加入 14 个 active-batch 摘要后，MAPE 从 12.15%
恶化到 25.25%，Decode 达到 54.50%。该结果只能说明“当前 6 种形态不足以支撑这组
时序特征泛化”，不能说明时序永远无用。

这些负结果共同指向：下一步不应继续堆回归参数，而应补上与真实通信实现对齐的单次
代价函数。

## 8. Phase 14F 做了什么

Phase 14F 在同一台 B200 L1 环境中，测量：

\[
C_{L1}(raw\_op,payload,TP,backend\_proxy)
\]

修正后的正式实验位于远端：

```text
/sgl-workspace/sglang-src/experiment-results/phase14f_post_rendezvous/
```

实验规模：

- 30/30 条曲线单元完成；
- 105 个 `(raw_op,payload,TP)` 支撑点；
- 每点 100 次调用、5 次独立重复；
- 525 条曲线记录、52,500 个调用样本；
- 最大重复 median CV 为 5.78%；
- 0 retry、0 failure。

Phase 14F 使用实测 PatternDemand 与上述 L1 曲线进行结构化累加，并只为
Prefill/Decode 学习一个非负乘性校准，不使用 DNN 直接拟合总时间。

### 8.1 修正后的正式结果

| 方法 | Overall MAPE | P95 APE | Prefill MAPE | Decode MAPE |
|---|---:|---:|---:|---:|
| Phase 2 payload-only scaled | 24.69% | 83.19% | 22.56% | 28.93% |
| Phase 14D TP-conditioned PatternDemand | 12.15% | 39.01% | 9.43% | 17.58% |
| Phase 14F op/backend-aware structural | **4.43%** | **10.76%** | **4.36%** | **4.55%** |

Phase 14F 的 \(R^2=0.9967\)。整模型留出：

| 留出模型 | MAPE | P95 APE |
|---|---:|---:|
| DeepSeek-V2-Lite | 5.94% | 12.83% |
| Qwen3-30B-A3B | 3.28% | 7.24% |
| Qwen3-8B | 4.23% | 9.81% |

预设的四个门槛全部通过：整体 MAPE <10%、P95 <25%、Decode MAPE <10%、每个
整模型留出 MAPE <15%。

### 8.2 Phase 14F 的限制

该结果证明：在当前 L1、三个模型、TP2/4/8 和已覆盖 payload/backend 支撑范围内，
“PatternDemand + 连续代价曲线”可以准确组合通信时间。

Phase 14G 和 Phase 15 已补掉其中三项验证空白：

- 同一 post-rendezvous 时间契约下，total bytes、payload histogram、raw op 的总体
  MAPE 分别为 46.80%、7.20%、4.39%；
- raw-op 曲线支撑点 leave-one-out 插值 MAPE 为 2.35%，P95 为 16.07%；
- 使用预测 PatternDemand 乘同一 L1 曲线的误差传播管线已经跑通。

它仍未证明：

- 历史流量画像可以准确预测某一个未来 batch 的 PatternDemand；Phase 15 已证明这个
  目标噪声很大；
- 预测直方图的误差传入时间模型后仍保持 4.43%；4.43% 只属于真实直方图输入；
- L2/L3、PP、PD、EP All-to-All 具有同样精度；
- 调度器已经实现端到端 placement 改善。

初始 `experiment-results/phase14f/` 曾错误使用 max kernel duration，而不是与
Phase 14C 一致的 post-rendezvous 标签，所得约 26.09% 结果无效，只能保留为审计。
提交 `95940aa9` 修正了时间对齐和 Phase 14D 比较键；修正结果与审计已由
`790a2718`、`d828abb5` 正式归档。

### 8.3 Phase 15 的真实 trace GPU 证据

Qwen3-8B 使用 BurstGPT 与 Mooncake 选出的 20 个窗口，在 TP2/4/8 下各运行一次，
共得到 120 条 `window × TP × phase` 标签：

- 20/20 窗口在三个 TP 下全部成功；
- 所有 rank 的 histogram 完全一致；
- 实际逐请求生成长度与 trace 计划完全一致；
- 只保存 histogram，三个正式 TP 结果文件合计不足 1 MiB；
- 120/120 条 GPU 直方图与 Qwen3-8B 解析事件公式一致；
- logical calls、logical bytes 和 `(raw_op,payload)` 直方图跨 TP 完全不变；
- equivalent bytes/rounds 随 TP 按 group size 改变。

当前回放是异构长度请求同一时刻进入的 `draining_batch_a_i_zero`。arrival offsets 被保留
用于审计，但没有真正交错注入，所以不能把它表述成在线 continuous batching 实验。

### 8.4 Phase 15 的长消息 L1 补点

公开 trace 中 8 个长 Prefill 窗口产生约 320–512 MiB 消息，超过 Phase 14F 原有
128 MiB 上限。为避免不受约束的外推，已补测：

- payload：160、192、256、320、384、448、512 MiB；
- TP：2、4、8；原语：AllReduce；
- 21 个支撑点，每点 5 个重复 × 100 样本；
- 10,500 个 all-rank post-rendezvous 样本；
- 最大 repeat-median CV 为 0.40%。

因此 Phase 15 smoke 使用的 Prefill payload 已落在 L1 实测连续曲线范围内。

## 9. 正式第一阶段：ProfileDemand v1

正式目标是配置级通信需求估计，不是时间序列预测：

\[
\widehat H^{1000}
=F_{pattern}
\left(
F_{model},\rho_{service},S_{execution},TP
\right),
\qquad
\rho_{service}=\{P(L,M),\lambda,burst\}.
\]

第一版只做 TP，并使用“每 1000 请求”归一化。正式结构模型使用 49 维输入：17 个模型
结构特征、低维 \(P(L,M)\) 联合画像、数值化 batch/chunk 策略和 TP。Phase 16A 对
8/12/16/24 个 log-payload 桶完成消融：若每桶同时保留 calls 与 logical bytes，12 桶
相对精确直方图的结构代价 MAPE 为 0.064%、P95 APE 为 0.70%，进入 L1 闭环后的 MAPE
为 4.47%，接近精确直方图的 4.43%。因此正式标签使用 12 桶双统计量，三硬桶只展示。

基础结构公式 \(H_0\) 由模型通信模板、Prefill token 规模、Decode 生存曲线
\(A(t)\) 和 chunk 规则生成；小型 DNN 只学习受约束残差：

\[
\operatorname{Encode}(\widehat H)
=\operatorname{Encode}(H_0)
+\operatorname{clip}(R_\theta(F_{model},\rho,S,TP)).
\]

Phase 16D 已验证 TP transformer 的透明先验：每个 forward 的 canonical AllReduce
次数为 `2×layers+1`，payload 为 `active_tokens×hidden_size×dtype_bytes`。在现有三个
模型、TP2/4/8 的 162 个配置上，H0 对 canonical calls、logical bytes 和精确 payload
直方图均为 162/162 完全匹配。Decode 中第一个输出 token 由 Prefill forward 采样，
因此 \(A(t)=\sum_i\mathbf 1(M_i>t)\) 从 \(t=1\) 开始，共 \(M_i-1\) 个 Decode 机会。

消息直方图仍可恢复原开题中的等效量：

\[
B_{\phi,u}^{eq}=\sum_{o,m\in u}\widehat H_{\phi,o,m}\alpha_o(p)m,
\qquad
R_{\phi,u}^{eq}=\sum_{o,m\in u}\widehat H_{\phi,o,m}\beta_o(p).
\]

第二阶段保持不变：\(\widehat T_{struct}=\sum\widehat H C_\pi\)。若
\(C_\pi=\alpha m/BW+\beta RTT\)，它与开题式(17)完全等价。

### 9.1 ProfileDemand v1 的最小实验范围

- 并行形态：只做 TP；
- 模型：先使用现有三个模型跑通，收敛后再增加两个结构不同模型；
- 流量画像：已从 BurstGPT/Mooncake 得到 24 个常态 medoid 画像；
- 策略：latency/balanced/throughput 三档 `max_batch×max_prefill_tokens`；异长画像不与
  chunk 做伪全交叉，chunk 使用 Phase14C 受控数据形成稀疏因子设计；
- TP：2/4/8；
- DNN：两层 64 维 residual MLP，不使用 Transformer；
- 泛化：画像、策略、TP 和 leave-one-model-out。

### 9.2 Phase 16F–16G 已完成结果

正式 GPU 网格覆盖：

- 三个模型：Qwen3-8B、DeepSeek-V2-Lite、Qwen3-30B-A3B；
- TP2/4/8、24 个服务画像、latency/balanced/throughput 三档策略、Prefill/Decode；
- 9 个模型×TP 运行，每组 545/545 workloads，总计 4905 个 histogram-only workloads；
- 聚合为 1296 条 `model×TP×profile×strategy×phase` 标签；
- 每条标签保存每 1000 请求的 12 桶 calls、12 桶 logical bytes、canonical 精确直方图
  和 raw-op 精确直方图；
- 9/9 all-rank、group size、固定实际输出、H0 canonical 和无 raw events 检查全部通过；
  Qwen3-8B TP2 smoke 的三次重复完全一致。

四种方法为 model-ID direct DNN、structure-direct DNN、H0 和正式的
`H0 + bounded residual DNN`。外层测试完整留出流量 segment、模型、策略和 TP，内层
早停再按 profile 分组，避免同一画像的跨 TP 重复标签泄漏。

| 外层留出 | 方法 | total calls MAPE | total bytes MAPE | log-payload EMD | L1结构代价 MAPE | P95 APE |
|---|---|---:|---:|---:|---:|---:|
| 未见模型 | H0 | 9.18% | 5.36% | 0.017 | 9.20% | 30.40% |
| 未见模型 | H0+residual | 10.05% | 5.29% | 0.017 | 8.71% | 27.71% |
| 未见策略 | H0 | 9.18% | 5.36% | 0.017 | 9.20% | 30.40% |
| 未见策略 | H0+residual | 9.22% | 4.43% | 0.017 | 8.05% | 27.59% |
| 未见 TP | H0 | 9.18% | 5.36% | 0.017 | 9.20% | 30.40% |
| 未见 TP | H0+residual | 8.46% | 4.51% | 0.016 | 7.55% | 25.31% |
| 未见流量 segment | H0 | 9.18% | 5.36% | 0.017 | 9.20% | 30.40% |
| 未见流量 segment | H0+residual | 12.57% | 7.64% | 0.016 | 11.34% | 30.65% |

这里的三个解释边界非常重要：

1. residual 在未见模型、策略和 TP 上改善 H0，说明可泛化结构特征和数值化策略是
   有效输入；但在完整未见流量域上变差，所以 DNN 是域内校准器，不是公式替代者。
2. 12 桶 vector WAPE 仍约 27%–32%，但 log-payload EMD 仅 0.016–0.017，说明大量误差
   来自消息质量落在相邻硬桶两侧，而不是跨越多个消息尺度。total calls/bytes 和乘连续
   曲线后的代价更适合判断调度用途；精确直方图仍保留用于审计。
3. arrival/RPS/突发特征虽进入输入，但本轮 GPU 标签仍是 draining microbatch；当前只
   验证“常态画像到单位请求通信需求”，尚未验证到达过程驱动的 online continuous batching。

direct DNN 在未见流量 segment 上的 L1 结构代价 MAPE 为 53.63%，model-ID baseline
为 83.00%，明显弱于 H0 与 H0+residual，支持“结构公式为主、DNN 只校正残差”。

## 9A. Phase 15C 时间预测诊断实验（非正式目标）

以下内容记录已经完成的 Phase15C，但不再作为正式第一阶段定义。

### 9A.1 历史流量画像

- prompt 长度分布；
- 输出长度分布；
- 最好保存联合分布 \(P(L,M)\)，而不是两个互相独立的边缘分布；
- 请求到达率 \(\lambda(t)\)；
- inter-arrival 分布、峰均比、突发持续时间和周期性；
- 会话或请求类型；
- 预测窗口长度。

### 9A.2 当前状态与执行配置

- 当前队列和正在 Decode 的请求；
- 当前已知 prompt 长度；
- 剩余输出长度的概率分布；
- continuous batching、max batch、max batch tokens；
- prefill chunk size 和调度策略；
- model、dtype、TP候选值和 backend 版本指纹。

纯历史画像适合容量规划；在线调度还必须加入当前状态，否则“当前空队列”和“当前积压
100 个请求”会得到相同预测。

### 9A.3 原诊断任务的推荐输出

每个未来窗口输出：

```text
phase × raw_op × payload × group_size
→ expected_calls、logical_bytes、equivalent_bytes、rounds、uncertainty
```

由于未来输出长度未知，建议预测 expected/P50/P95 PatternDemand，而不是假设唯一确定值。

### 9A.4 Phase 15 首版预测器做了什么

固定输入为 300 秒历史窗口，预测对象为下一 60 秒中确定性抽样的一个最多 8 请求
draining batch。输入包括：

- 历史请求数、RPS、inter-arrival CV、1 秒峰均比和 Fano factor；
- 历史 prompt/output 的均值、P50/P90/P99、相关系数；
- prompt/output 的 18 档 log2 长度直方图；
- 固定的 max-batch=8、output cap=128 和 draining 策略。

标签来自 66,642 个因果窗口，其中 46,535 个未来窗口非空。Qwen3-8B 解析标签公式
已经由 20 窗口 × TP2/4/8 的 120 条 GPU 阶段标签逐条验证。训练使用 BurstGPT-1，
验证使用 BurstGPT-2 前半，测试使用 BurstGPT-2 后半；BurstGPT-3 用于时间外推，
Mooncake Conversation/ToolAgent 共 106 个窗口作为外部测试。

### 9A.5 首版预测结果及其含义

| 测试域 | 活跃窗口 F1 | Prefill bytes MAPE | Decode bytes MAPE | L1 TP8 结构 MAPE |
|---|---:|---:|---:|---:|
| BurstGPT 普通测试 | 80.67% | 261.94% | 84.38% | 78.85% |
| BurstGPT 时间外推 | 91.69% | 215.37% | 75.48% | 66.82% |
| Mooncake 外部测试 | 100.00% | 32.85% | 29.58% | 26.06% |

这里的 L1 结构 MAPE 是：预测 histogram 乘实测 L1 曲线，与真实 histogram 乘同一
曲线比较。它隔离第一阶段误差，不是 Phase 15 新测的绝对通信时间；绝对曲线准确性由
Phase 14F 独立给出的 4.43% 支撑。

普通测试和时间外推误差很高，且历史均值 persistence 在这两个域的 L1 误差反而约为
53.7% 和 44.9%。这说明当前 MLP 并未收敛为调度器可用预测器。根因不是 PatternDemand
标签错误，而是目标要求历史摘要猜中未来 60 秒中一个随机抽样 batch；未来请求具有不可
约随机性，单点 MAPE 会被大量小请求和长尾请求放大。

### 9A.6 诊断结论

```text
离线/容量规划：历史画像
    → 未来窗口的请求数、P(L,M)、expected/P50/P95 PatternDemand

在线调度：历史画像 + 当前 pending/running requests + batching/chunk policy
    → 形成候选 batch / A(t)
    → 解析或小模型生成具体消息直方图

具体直方图 + 已确定的TP配置 + topology curve
    → 通信时间与不确定性
```

调度器已知 pending request 的 prompt 长度时，不应再让网络“猜”这部分。输出长度仍可
预测为分布并传播成 expected/P95 histogram。Phase 15 的 scheduled-batch 解析路径与
GPU 标签 120/120 一致，说明这一条件路径是可行的；下一步应修正预测目标，而不是继续
盲目增大当前 MLP。

## 10. 真实流量训练集与外部测试集

当前没有唯一公认、同时覆盖所有 LLM 流量特征的标准测试集。推荐组合为：

### 10.1 BurstGPT：主训练与时间泛化

官方数据来自 Azure OpenAI 服务，公开版本包含约 529 万条请求、连续 121 天，字段
包括 timestamp、session、model、request/response tokens、elapsed time 和 log type：

<https://github.com/HPMLL/BurstGPT>

用途：

- 学习 \(P(L,M)\)、到达率、突发和时间漂移；
- 前 60% 连续时间训练、中间 20% 验证、最后 20% 测试；
- 额外完整留出高突发窗口或调用类型；
- 禁止随机请求切分造成相邻时间泄漏。

### 10.2 Mooncake FAST'25 Trace：跨业务外部测试

官方发布包含 Conversation 12,031 请求、Tool/Agent 23,608 请求和 Synthetic 3,993
请求；真实 trace 含 timestamp、input/output length 和 prefix block hash：

<https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release>

它具有长上下文、工具/Agent 和 prefix reuse 特征，适合作为完全不参与训练的外部测试，
检验 BurstGPT 上训练的模型能否迁移到另一种真实服务画像。

### 10.3 LMSYS-Chat-1M / WildChat：长度分布补充

- LMSYS-Chat-1M：<https://huggingface.co/datasets/lmsys/lmsys-chat-1m>
- WildChat：<https://huggingface.co/papers/2405.01470>

它们包含真实对话文本，可使用目标模型 tokenizer 重新计算 \(L,M\)，补充多轮、语言、
代码和长尾请求；但采集时间戳不等于完整生产后端到达流量，不作为主到达过程训练集。

### 10.4 Trace 回放原则

不需要把数百万请求全部运行一遍。推荐：

1. 将 trace 切成固定历史窗口和预测窗口；
2. 按到达率、突发度、\(P(L,M)\) 和长尾程度聚类/分层；
3. 选择代表窗口与极端窗口；
4. 按单机容量合理缩放 RPS，但保留相对到达间隔与突发形态；
5. 在 SGLang 上按 TP2/4/8、batch/chunk 策略回放；
6. 强制实际生成 trace 的 `output_length`，避免 EOS 提前终止；
7. 使用 histogram-only 保存标签，不保存大体积 raw events；
8. 完整 workload/window 和三次重复不得跨 train/test。

初始规模建议为 200–400 个训练窗口、50–100 个验证窗口、100 个 BurstGPT 时间测试
窗口和 50–100 个 Mooncake 外部测试窗口；正式规模根据单窗口运行成本再调整。

### 10.5 当前已经落地的数据管线

Phase 15 固定了三份 BurstGPT v2.0 `without_fails` 文件和三份 Mooncake FAST'25
文件的 URL、文件大小及 SHA-256。规范化后包含：

- BurstGPT-1：1,404,294 请求；
- BurstGPT-2：3,784,213 请求；
- BurstGPT-3：4,956,058 请求；
- Mooncake Conversation：12,031 请求；
- Mooncake Tool/Agent：23,608 请求；
- Mooncake Synthetic：3,993 请求。

共生成 66,642 个 300 秒历史 / 60 秒未来的因果窗口：17,566 train、8,639
validation、8,640 test、31,679 temporal test、106 external test 和 12 synthetic
external。公开原始文件只在本地缓存；规范化窗口、版本哈希、20-window replay plan 和
训练标签均已提交到实验分支。

## 11. 第一阶段与端到端评测

### 11.1 PatternDemand 本身

- total calls WAPE/MAPE；
- total logical payload WAPE；
- `(op,payload)` 直方图 weighted L1/TV；
- payload 分布 EMD/Wasserstein distance；
- small/medium/large 的分布迁移仅用于解释图；
- expected/P95 区间覆盖率。

### 11.2 泛化切分

- 时间留出；
- 突发窗口留出；
- Decode 输出分布/profile 留出；
- chunk 策略留出；
- TP留出；
- leave-one-model-out；
- BurstGPT → Mooncake 跨数据集外部测试。

### 11.3 最终调度指标

用预测直方图替换 Phase 14F 的实测直方图：

\[
\widehat H
\rightarrow
C_{L1}
\rightarrow
\widehat T_{comm}
\]

报告：

- 端到端通信时间 MAPE/P95 APE；
- PatternDemand 预测误差对时间误差的传播；
- total bytes、三桶、精确直方图的同口径消融；
- 最终 placement 选择准确率和 scheduling regret。

### 11.4 当前完成程度

- GPU 标签可靠性：已完成；
- predicted PatternDemand → L1 曲线的误差传播代码：已完成；
- 普通时间切分、远期时间切分、Mooncake 外部测试：已完成 pilot；
- 历史画像预测下一随机 batch：已证明精度不足，且不再作为正式研究目标；
- 服务常态画像 + 模型结构 + 执行策略 + 已确定的TP配置 → 每 1000 请求 PatternDemand：
  Phase 16 首版已完成，含 1296 条正式 GPU 标签、四方法和四类分组留出；
- predicted PatternDemand → L1 连续曲线：未见模型/策略/TP 的正式 residual 结果分别为
  8.71%/8.05%/7.55% MAPE；未见流量域使用 H0 回退更稳；
- 仅通信代价下的 batching 策略选择准确率和 regret：Phase 17 已完成；
- 真实 online continuous batching，以及同时考虑计算、显存、排队和资源可用性的完整
  placement 选择与 regret：待完成。

## 12. 没有第二台机器时的 L2/L3 方案

模型实验负责生成“通信货物清单”，链路微基准负责生成“运费表”。不需要在每个 L2/L3
拓扑上重跑全部模型网格。

若临时获得两台机器，只需针对实际 rank mapping 测量：

\[
op\times payload\times group\ size\times topology
\rightarrow latency
\]

标准 collective 可使用 NVIDIA `nccl-tests`；SGLang fused op 需要走相同 backend 的
自定义微基准或拆成 collective 与本地 fused kernel。测量 L2 需要同机架两节点；测量
L3 需要跨机架节点，一次普通两节点环境只能代表其实际所在拓扑。

如果始终没有跨节点资源：

- L1 保留真实物理验证；
- L2/L3 使用多组 RTT/BW 或公开曲线进行参数化仿真和敏感性分析；
- 报告不同参数下的 placement 决策边界和 regret；
- 明确写成仿真推演，不能报告真实 L2/L3 MAPE。

### 12.1 Phase 17 已完成的参数化敏感性分析

Phase 17 复用 Phase 16 的 1296 条 GPU PatternDemand 标签，以及两个严格留出轨道、
两种预测方法形成的 5184 个 out-of-fold 12 桶预测向量。它没有重新运行模型，也没有
伪造 L2/L3 实测数据，而是把同一批“通信货物清单”分别送入：

- 1 条 B200 单节点实测 L1 连续曲线；
- L2/L3 各 optimistic、nominal、pessimistic 三条平滑 α–β 曲线；
- L2/L3 各一条假设存在协议切换的 stress 曲线。

每条参数化曲线按 TP2/4/8 和 65 个 payload 支撑点生成，共 9 个 curve profiles、1755
条曲线记录。L2/L3 采用“两节点均匀分 rank 的完整 ring”代理，参数和 rank mapping
均写入结果文件；它是敏感性分析，不是 NCCL 分层算法的物理复现。

在 traffic-segment holdout、Prefill+Decode 合并口径下：

| 曲线 | total bytes MAPE | total calls+bytes MAPE | 12 桶精确 MAPE | H0 预测12桶 MAPE |
|---|---:|---:|---:|---:|
| L1 实测 | 84.82% | 1.50% | 0.30% | 12.18% |
| L2 nominal 参数化 | 85.03% | 0.01% | 约 0% | 13.40% |
| L3 nominal 参数化 | 92.72% | 0.02% | 约 0% | 14.23% |

这组结果首先证明：只保存 total bytes 会丢掉每次 collective 的启动/轮次成本，跨高
RTT 拓扑时问题更严重。latency 策略与 throughput 策略的总 bytes 中位数之比为 1.000，
但 calls 中位数之比为 3.820；相应通信代价中位数之比在 L1、L2 nominal、L3 nominal
上分别为 2.785、2.989、3.383。

同时必须保留以下负结果：在只有平滑启动项和带宽项的 α–β 参数化曲线上，精确总
calls+bytes 已经几乎决定总时间，因此三桶/12 桶不会再获得显著平均误差收益。12 桶的
额外价值在实测 L1、协议/算法切换以及 Prefill 尾部更明显，例如 L1 Prefill 的 one-bin、
three-bin、12-bin MAPE 分别为 2.169%、1.338%、0.580%。因此论文应表述为：

> calls 是相对 total bytes 不可缺失的核心维度；payload 直方图在代价曲线具有消息尺度
> 非线性、backend/op 差异或协议切换时提供额外辨识力，而不是宣称所有链路都必须使用
> 12 桶才能预测。

Phase 17 还完成了只考虑通信代价的 latency/balanced/throughput 三策略选择。total
bytes 因三策略总字节相同而选择准确率为 0%，L1/L2 nominal/L3 nominal 的平均 regret
分别为 186.38%、208.63%、253.22%；精确 calls+bytes 在本组 workload 中为 100% 准确、
零 regret，OOF 预测直方图为 99.54%–100%。这只能证明通信需求表征能够支持通信侧策略
比较，不能替代完整调度器；真实 placement 仍需联合计算时间、显存、排队和资源可用性。

### 12.2 Phase 19–24 的纯PP证据与当前边界

Phase 19在Qwen3-8B、TP=1下覆盖PP2/4/8与`pp_max_micro_batch_size=1/4/16`。
正式formal-v3包含351个请求批次、2376个逻辑请求，所有PP boundary审计通过。最关键
对照是`B=16,L=512,M=32`：单boundary三重复合计的logical bytes均为427,032,576，
而mb1/mb4/mb16的calls分别为3072/768/384，证明PP microbatch在同bytes下可产生8倍
启动次数差异。

Phase 20证明精确workload到PP直方图的结构公式是可靠的；Phase 21/22进一步区分了两个
问题：online arrival会引入额外不确定性，而fixed-draining低维画像预测器的主要误差来自
画像压缩后无法恢复离散batch/microbatch边界。当前offline H0+residual在profile、strategy、
PP holdout上的calls MAPE分别为41.21%、60.74%、41.12%，不能宣称PP画像预测器已收敛。

Phase 23固定相同token IDs、长度、顺序、同时到达、PP配置和microbatch策略，对9个cell、
54个`workload×phase×cell`组各重复10次，得到540条phase labels。结果为：

- 54/54精确直方图完全一致；
- 最大calls与logical bytes相对跨度均为0；
- H0 calls/bytes平均和P95 APE均为0；
- H0 histogram L1和log-payload EMD均为0；
- 9/9 boundary checks PASS。

因此可以得出“固定精确workload下，PP GPU直方图和H0完全确定且可靠”；不能由此得出
“低维画像已能恢复PP calls”或“online arrival的期望直方图已确定”。正式结果位于：

```text
远端：/sgl-workspace/sglang-src/experiment-results/phase23_pp_draining_stability/
本地：work/sglang-phase2-curve/experiment-results/phase23_pp_draining_stability/
提交：fe7409067ed3aca2b3a893ea32cc124be32e5674
```

Phase 24以同一批BurstGPT/Mooncake历史窗口的Hfull为唯一参考，同时比较TP和PP的
H32/H64/H128/Hfull，全部归一化到每1000请求。共处理18,285条full requests，输出
4,320条histogram labels、5,184条主评测记录、1,296条compact32→exact H32误差分解和
330条聚合记录。total聚合主要结果：

| 并行 | 样本 | calls MAPE/WAPE | bytes MAPE/WAPE | hist TV | norm EMD | common cost MAPE | P95 calls APE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TP | H32 | 11.48% / 7.98% | 5.82% / 4.78% | 0.2751 | 0.0121 | 6.96% | 39.35% |
| TP | H64 | 6.50% / 4.65% | 2.78% / 2.27% | 0.2324 | 0.0096 | 3.88% | 16.44% |
| TP | H128 | 7.57% / 5.27% | 2.01% / 1.37% | 0.2056 | 0.0084 | 4.89% | 27.69% |
| PP | H32 | 8.38% / 4.28% | 5.82% / 4.78% | 0.0922 | 0.0052 | 6.63% | 27.14% |
| PP | H64 | 4.62% / 2.76% | 2.78% / 2.27% | 0.0834 | 0.0041 | 3.27% | 15.40% |
| PP | H128 | 4.70% / 2.67% | 2.01% / 1.37% | 0.0779 | 0.0039 | 3.56% | 22.81% |

所有规模都至少在histogram TV或P95尾部上未达门槛；H128也未必然比H64好，特别是小于128条
请求的7个窗口存在重复/重权的代表池效应。PP中mb1明显比mb4/mb16容易收敛，BurstGPT窗口也比
Mooncake更难；这些细分指标保存在`analysis/aggregate_metrics.csv`。

跨TP/PP的cost收敛使用显式公共参考曲线`5 us + payload / 100 GB/s`，不是PP物理时延；
TP另报实测B200 L1 AllReduce曲线传播误差。由于没有PP P2P曲线，不能从Phase 24得出PP真实
通信时间MAPE。完整请求列表仅用于离线生成teacher label；最终预测器输入仍是低维历史画像、
模型结构、固定并行配置和固定策略。

正式结果：

```text
远端：/sgl-workspace/sglang-src/experiment-results/phase24_representative_request_convergence/
本地：work/sglang-phase2-curve/experiment-results/phase24_representative_request_convergence/
提交：bb490ad6c85c4521d194d01e7c6dab6b9b67118e
```

## 13. 与开题 4.1.1–4.1.3 的映射

### 13.1 4.1.1 拓扑无关通信需求

已有证据：TP 场景的埋点、统计口径、三模型数据、batch/输出分布/chunk/TP 机理；24 个
BurstGPT/Mooncake 常态 \(P(L,M)\) 画像；可泛化模型结构特征；三档策略；TP2/4/8 的
4905 个 GPU workloads 和 1296 条正式标签；H0、direct DNN、bounded residual DNN 的
流量段/策略/TP/整模型留出。

已有PP证据：Qwen3-8B纯PP正式PatternDemand、精确workload H0、低维画像offline/online
诊断、Phase 23固定draining十重复严格稳定性，以及Phase 24的32/64/128/full代表请求收敛。
待补：PP full-window teacher dataset与预测器重训、多模型PP、PP P2P连续代价曲线和PD正式PatternDemand。真实
arrival-driven online continuous batching保留为扩展；在线pending queue不是首版
fixed-draining ProfileDemand的必需输入。

### 13.2 4.1.2 拓扑通信代价

已有证据：B200 单节点 L1 连续代价曲线、Phase 14F 组合验证、Phase 14G 留出与严格
消融、覆盖真实 trace 长 Prefill 的 160–512 MiB 补点，以及 Phase 17 的八条参数化
L2/L3 曲线、协议切换 stress 和通信侧策略 regret 敏感性分析。

待补：L2/L3 物理曲线。参数化仿真已经完成，但只能用于算法敏感性和决策边界，不能
报告成真实跨节点链路精度。

### 13.3 4.1.3 通信时延拟合

建议将纯 DNN 改写为：

\[
\widehat T_{comm}
=
\sum \widehat H C_\pi
+R_\theta(F_{traffic},F_c,F_\pi,\widehat H,T_{struct})
\]

Phase 14F 表明在当前 L1 范围内，结构公式加极少量校准已经达到 4.43% MAPE，DNN
不是必选项。只有当真实 trace、重叠和运行竞争产生稳定残差时才引入 ResidualNN。

## 14. 文件与 Git 状态

### 14.1 本地

```text
/Users/liyafei06/Documents/Codex/2026-07-21/
└── login-klingai-wlf2-ge151-node55-idchb2az2/
    ├── outputs/
    │   └── 截至目前实验结构总导引.md
    └── work/sglang-phase2-curve/
        └── experiment-results/
            ├── phase0 ... phase14g_strict_ablation/
            ├── phase15_trace_data/
            ├── phase15_qwen_trace_pattern/
            ├── phase15_pattern_training_data/
            ├── phase15_l1_curve_extension/
            ├── phase15_pattern_predictor/
            ├── phase16_service_profiles/
            ├── phase16_model_features/
            ├── phase16_h0_validation/
            ├── phase16_profiledemand_plans/
            ├── phase16_profiledemand_dataset/
            ├── phase16_profiledemand_predictor/
            ├── phase17_parameterized_topology/
            ├── phase18_tp_decision_pilot/
            ├── phase19_pp_pattern/qwen3-8b-formal-v3/
            ├── phase20_pp_predictor/qwen3-8b-v1/
            ├── phase21_pp_service_profile/
            ├── phase21b_pp_offline_profiledemand/
            ├── phase21c_pp_online_residual/
            ├── phase22_pp_predictor/
            └── phase23_pp_draining_stability/
```

当前本地分支：

```text
experiment/pattern-demand-v0.5.15-clean
```

当前本地 HEAD：

```text
2e9bae7e docs: refresh Phase 26A coverage figure
```

本地已同步 Phase 14F/14G/15/16/17/18/19/20/21/22/23/24 的正式紧凑结果、模型、预测明细和日志；公开原始 trace
位于本地 `data/phase15_traces/raw/`，不提交 Git。

### 14.2 远端

```text
源码与实验：/sgl-workspace/sglang-src
开题文档：/sgl-workspace/sglang/3.12 部分实验 & 开题.md
修正后 Phase 14F：
/sgl-workspace/sglang-src/experiment-results/phase14f_post_rendezvous/
Phase 15：
/sgl-workspace/sglang-src/experiment-results/phase15_*/
Phase 16 正式标签与预测器：
/sgl-workspace/sglang-src/experiment-results/phase16_profiledemand_dataset/
/sgl-workspace/sglang-src/experiment-results/phase16_profiledemand_predictor/
Phase 17 参数化拓扑敏感性：
/sgl-workspace/sglang-src/experiment-results/phase17_parameterized_topology/
Phase 19纯PP正式结果：
/sgl-workspace/sglang-src/experiment-results/phase19_pp_pattern/qwen3-8b-formal-v3/
Phase 20–22 PP预测器与画像诊断：
/sgl-workspace/sglang-src/experiment-results/phase20_pp_predictor/
/sgl-workspace/sglang-src/experiment-results/phase21b_pp_offline_profiledemand/
/sgl-workspace/sglang-src/experiment-results/phase22_pp_predictor/
Phase 23固定draining严格复核：
/sgl-workspace/sglang-src/experiment-results/phase23_pp_draining_stability/
Phase 24代表请求收敛：
/sgl-workspace/sglang-src/experiment-results/phase24_representative_request_convergence/
```

关键提交：

| 提交 | 作用 |
|---|---|
| `012172b5` | 归档 Phase 14C 紧凑时间标签 |
| `cdb1454d` | 添加 Phase 14F backend cost curve suite |
| `95940aa9` | 修正 post-rendezvous 时间契约和 Phase 14D 映射键 |
| `d828abb5` | 归档更正后的 Phase 14F 曲线和审计 |
| `384e4d68` | 归档 Phase 14G 严格消融和留出预测 |
| `78e30538` | 归档 BurstGPT/Mooncake 因果窗口 |
| `98e37275` | 归档 Qwen3-8B trace PatternDemand 紧凑标签和日志 |
| `8013bc22` | 归档 Phase 15 长消息 L1 曲线补点 |
| `bd393145` | 归档历史画像 PatternDemand predictor、预测明细与评测 |
| `0ef5f239` | 归档 1296 条 ProfileDemand GPU 紧凑标签和运行哈希 |
| `e8497e2e` | 归档四方法留出指标、预测明细、模型 checkpoint 和训练日志 |
| `5580a0b1` | 补齐并归档 5184 个可重放 OOF 12 桶预测向量 |
| `bb7a8d16` | 归档 Phase 17 参数化 L2/L3、协议切换与通信策略 regret 分析 |
| `ae593a17` | 归档 Qwen3-8B PP offline/online predictor 汇总结果 |
| `a5473bad` | 增加 Phase 23 严格 PP draining 稳定性过滤器 |
| `baf947b7` | 增加 Phase 23 十重复审计分析器 |
| `fe740906` | 选择性归档 Phase 23 正式结果、紧凑标签、日志和manifest |
| `bb490ad6` | 归档 Phase 24 TP/PP H32/H64/H128/Hfull收敛脚本、标签、指标、图表和manifest |

远端、远端tracking、本地和本地tracking均为`bb490ad6c85c4521d194d01e7c6dab6b9b67118e`，分支都是
`experiment/pattern-demand-v0.5.15-clean`。本地工作树仅保留未跟踪`data/`；远端旧
Phase16/19、Phase 23 PID和一个`.tmp`仍未跟踪且未触碰。Phase 23 manifest在远端和本地
均验证通过；Phase 24 manifest在两端也均11/11通过。初始错误Phase14F目录仍只能用于审计。

## 15. 下一步执行顺序

### 已完成：P0–P3 工程 pilot

- Phase14F 正式归档、错误结果隔离、严格消融和支撑点留出；
- BurstGPT/Mooncake 下载、哈希固定、窗口切分和外部测试；
- Qwen3-8B 20-window TP2/4/8 GPU PatternDemand smoke；
- 160–512 MiB L1 长消息曲线补点；
- 66,642 窗口的训练标签、首版 MLP 和 L1 误差传播。

### 已完成 P0：ProfileDemand v1 数据定义

1. 输出定义为每 1000 请求、每个 `phase×op×payload-bin` 的 group-level calls 与
   logical bytes；
2. 已从 BurstGPT/Mooncake 提取 24 个常态 \(P(L,M)\) 服务画像；
3. 首版 draining-microbatch 网格将 RPS 用于 calls/s、bytes/s 换算；RPS 对真实在线
   batch 形成的影响作为后续 online 扩展，不把当前结果越界表述；
4. 设置 latency/balanced/throughput 三种数值化 `max_batch×max_prefill_tokens` 策略；
5. 8/12/16/24 payload 桶消融已完成，选择 12 桶 `calls+bytes` 表征。

### 已完成 P1：结构公式与可泛化特征

1. 从模型配置提取 layers、hidden、FFN ratio、head ratio、MoE、experts、top-k、dtype
   和 op mask；
2. 使用 \(P(L,M)\)、batch、chunk 和 \(A(t)\) 构造基础直方图 \(H_0\)；
3. \(H_0\) 已在三个模型、TP2/4/8 的 162 个实测配置上完成 162/162 canonical
   精确直方图匹配；
4. 不能用同一公式生成全部标签再宣称公式准确，正式测试必须使用 GPU histogram；
5. 未见 op 由静态 op mask/template 限定，不要求 DNN 凭空生成。

### 已完成 P2：小型 DNN residual 与泛化评测

1. 比较 model-ID DNN、结构特征 direct DNN、\(H_0\)、\(H_0+\)DNN residual；
2. DNN 只预测总 calls 和桶分布的乘性/softmax 残差；
3. 已完成流量 segment、策略、TP 和整模型留出；
4. 同时报告 calls/bytes WAPE、histogram L1/EMD 和乘 L1 曲线后的时间误差；
5. 当前三模型流程已闭环；下一轮再增加两个结构不同模型增强 LOMO 证据。

### 已完成 P3：纯PP底层链路和稳定性

1. Qwen3-8B纯PP正式GPU PatternDemand已覆盖PP2/4/8与microbatch 1/4/16；
2. Phase 20已验证精确workload到代表boundary直方图的结构H0；
3. Phase 21/22已定位online arrival不确定性和低维画像到microbatch结构的误差；
4. Phase 23十重复54/54完全一致，H0 calls/bytes/L1/EMD全零误差；
5. Phase 23正式结果、labels、logs、README、summary、DONE和manifest已由`fe740906`归档。

### 已完成 P4：代表请求规模收敛

1. 对同一批BurstGPT/Mooncake历史窗口和同一fixed-draining策略生成H32、H64、H128、Hfull；
2. 同时覆盖TP和PP，全部归一化到每1000请求；
3. 以Hfull为唯一参考，报告calls/bytes MAPE与WAPE、histogram L1/TV、log-payload EMD和cost MAPE；
4. 判断32个请求是否足够、PP是否需要64/128，以及误差来自样本规模还是画像压缩；
5. 结果是H32不足，H64/H128仍未通过全部门槛；使用full-window H0结果生成离线teacher label，完整请求列表不进入部署时输入。
6. 正式结果、分析脚本、README、summary、图表和manifest已由`bb490ad6`归档并双端同步。

### 当前 P5：PP full-window teacher与画像预测器

1. 对每个历史窗口使用完整capped长度列表，移除arrival timestamp，在固定fixed-draining策略下生成teacher；
2. 使用Phase 23验证过的PP H0生成per-boundary Prefill/Decode exact histogram和12-bin calls/bytes；
3. 保存的模型输入仍是compact profile、模型结构、固定PP size和策略，timestamp与full request list不进入推理输入；
4. 重新比较direct DNN、H0、H0+bounded residual，使用profile/source/strategy/PP size留出和H0回退；
5. 报告calls/bytes MAPE/WAPE、histogram L1/TV、log-payload EMD和同一参考曲线下cost MAPE，P95单独报告。

### 后续：补第二阶段与placement闭环

1. 先测PP P2P的单节点L1连续代价曲线；
2. 有两节点资源时测TP collective和PP P2P的L2/L3物理曲线；
3. 参数化L2/L3、协议切换和表征消融继续只作为敏感性分析；
4. 调度器接收已经确定的TP/PP配置，只枚举placement和topology，并联合计算、显存、排队和资源约束；
5. PP画像预测器收敛后扩展多模型纯PP，再做纯PD；online arrival保持扩展轨道。

## 16. 当前最准确的论文状态表述

> 本研究已完成三个模型TP2/4/8的topology-independent PatternDemand采集、结构机理验证
> 和基于BurstGPT/Mooncake常态画像的首版预测器，并在单节点B200上通过raw-op/backend-
> aware连续代价曲线将真实PatternDemand映射为通信时间，整体MAPE为4.43%。参数化L2/L3
> 实验表明total bytes会丢失高频小消息的启动代价，但L2/L3尚无物理实测。纯PP分支已在
> Qwen3-8B的PP2/4/8与microbatch 1/4/16上完成正式PatternDemand采集；相同logical bytes
> 下calls最多相差8倍。Phase 23固定相同token、长度、顺序和策略进行十次重复，54/54组
> GPU直方图完全一致，结构公式H0对calls、bytes和精确直方图均为零误差，证明精确workload
> 到PP PatternDemand的底层链路可靠。Phase 24在24个BurstGPT/Mooncake窗口上比较了TP和PP的
> H32/H64/H128/Hfull：PP calls MAPE为8.38%/4.62%/4.70%，但H64/H128的histogram TV仍为
> 0.083/0.078，P95 calls APE为15.40%/22.81%；TP和PP都无任一规模通过全部预注册门槛。
> compact32→exact H32的PP calls MAPE为10.39%，证明有限样本和低维画像恢复都是显著误差源，
> 且它们的误差存在部分抵消、不能简单相加。因此首版使用完整历史窗口的fixed-draining H0作为
> teacher label；完整请求列表只用于离线标签。
> 最终预测器输入仍为低维历史画像、模型结构、固定执行策略和已确定的并行配置；调度器选择
> placement/topology，不选择TP或PP size。PP P2P曲线、L2/L3物理实测、多模型PP、纯PD和
> 联合计算/显存/排队约束的placement闭环仍待完成；online arrival只作为扩展。

## 17. 推荐阅读顺序

1. 本文第 1–3 节：先理解两阶段和当前边界；
2. 第 6–8 节：理解消息直方图证据、失败实验和 Phase 14F/14G/15 GPU 验证；
3. 第 9 节：理解正式 ProfileDemand v1；第 9A–11 节只用于理解时间预测失败边界；
4. 第 12–13 节：理解 L2/L3、PP Phase 19–24 与开题章节的映射；
5. 第 15 节：按优先级继续执行。

---

## 18. Phase 25A/25B：full-window teacher的GPU审计与PP scheduler恢复

### 18.1 Phase 25A推翻了哪项外推？

Phase 23证明了受控代表workload下的PP结构公式精确，但Phase 25A将一个42请求异构完整窗口一次性送入真实SGLang PP scheduler后发现：

- TP full-window smoke精确；
- PP 9/9 cell采集与boundary一致性PASS；
- 旧静态PP teacher仅MB1的3/3 cell精确；
- MB4/16的6/6 cell不精确；
- bytes完全守恒，但calls、payload histogram和cost偏差显著。

因此，“精确请求列表→静态分组公式→PP full-window teacher”仍缺少真实scheduler状态机。旧Phase 24 PP H32/H64/H128/Hfull比较也采用静态公式，应在scheduler-faithful公式下重算后再用于论文正式结论。

Phase 25A目录为`experiment-results/phase25_full_window_teacher/`，结果归档提交为`6011ca59`。

### 18.2 Phase 25B恢复的scheduler结构

源码审计定位到四类必要状态：

1. `pp_loop_size=pp_size`，每个lane独立维护running batch；
2. 全局FCFS等待队列按lane访问顺序补入；
3. 4096-token chunk按64-token page扣预算，未完成chunk可跨lane继续并绕过严格microbatch上限；
4. scheduler先调用prefill admission，再由`update_running_batch`过滤刚完成的decode请求，因此slot释放与补位相差一次lane访问。

实现这些规则后，CPU teacher对已保存的PP2/4/8 × MB1/4/16九个GPU cell全部精确：calls、logical bytes、phase/payload histogram和`phase × active_batch_size × active_tokens` histogram的最大误差均为0。

### 18.3 正式full-window PP teacher

Phase 25B在24个BurstGPT/Mooncake窗口、18,285请求上生成：

- 432条phase labels；
- 1,584条显式sender-boundary labels；
- 216个配置的调度统计和token-mass不变量；
- 新旧公式逐行指标、聚合、图表、README、summary、source contract、log、DONE和manifest。

216/216配置全部请求完成且prefill/decode token mass守恒。目录为：

`experiment-results/phase25b_pp_scheduler_teacher/`

正式提交为：

`2eb03c5d1708c93aefc8d2ebe71f260ab7f2bf2a`

当前远端、本地及tracking HEAD均为该提交；本地仍只有未跟踪`data/`，远端旧Phase16/19和Phase23 PID/`.tmp`未触碰。manifest在远端和本地13/13通过。

### 18.4 旧静态公式的全窗口偏差

以Phase 25B为reference，24窗口total聚合如下：

| policy | calls WAPE | histogram TV | normalized EMD | cost MAPE |
|---|---:|---:|---:|---:|
| MB1 | 31.93% | 0.1420 | 0.0148 | 11.34% |
| MB4 | 208.36% | 0.7267 | 0.0948 | 48.06% |
| MB16 | 603.21% | 0.7984 | 0.1727 | 74.33% |

logical bytes WAPE均约为0。MB1的42请求smoke全部精确，是因为该窗口没有跨4096-token chunk；长prompt窗口仍会触发跨lane chunk语义。

### 18.5 修正后的P5执行顺序

1. 对长prompt、大请求窗口、Mooncake/BurstGPT做少量GPU tail audit；
2. 使用scheduler-faithful teacher重算PP H32/H64/H128/Hfull；
3. 独立完成TP full-window teacher的跨模型/策略/TP sentinel审计；
4. 以full-window teacher替换Phase 16 exact H32监督标签；
5. 重训并比较H0、direct DNN和H0+bounded residual；
6. 若PP compact profile仍无法恢复calls/直方图，再增强画像中的离散调度相关统计，而不是修改teacher；
7. 完整请求列表始终只用于离线teacher，部署输入不变。

### 18.6 当前结论边界

可以宣称：在记录的fixed-draining、FCFS、无radix、无mixed chunk、无async PP depth契约下，新scheduler simulator对一个异构完整窗口的全部9种PP配置GPU直方图完全一致。

不能宣称：一个sentinel已经覆盖任意流量分布；也不能外推到online arrival、preemption、radix cache、mixed chunk、speculative decoding或其他SGLang版本。下一里程碑必须先做tail GPU审计，再训练PP画像预测器。

---

## 19. Phase 25C/25D：PP teacher尾部闭环与正式代表规模结论

### 19.1 Phase 25C扩展了GPU证据

Phase 25C选择48请求、max prompt 6,216的BurstGPT窗口和930请求、max prompt 8,192的Mooncake窗口，在`PP2/MB1`、`PP4/MB4`、`PP8/MB16`三组配置上做完整GPU审计。

结果为3/3 cell、6/6 profile-cell、12/12 phase比较exact；calls、logical bytes、12-bin与exact payload histogram、sender boundary全部精确。由此，Phase 25B scheduler-faithful teacher已同时覆盖小窗口9-cell矩阵和长prompt/大请求数tail证据。

正式目录：`experiment-results/phase25c_pp_scheduler_tail_audit/`
正式提交：`0d1aaddcf0807b5e119e58fc41834fa56fc813e1`

### 19.2 Phase 25D取代Phase 24旧PP结论

Phase 25D使用相同24个历史窗口与H32/H64/H128集合，在scheduler-faithful teacher下重新计算。以Hfull为reference：

| 规模 | calls MAPE | calls WAPE | bytes MAPE | bytes WAPE | TV | norm EMD | cost MAPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| H32 | 71.99% | 25.21% | 5.82% | 4.78% | 0.4209 | 0.0511 | 17.13% |
| H64 | 33.50% | 12.93% | 2.78% | 2.27% | 0.3318 | 0.0347 | 8.40% |
| H128 | 18.65% | 7.93% | 2.01% | 1.37% | 0.2624 | 0.0241 | 6.06% |

H128仍未通过calls、直方图和cost门槛。H128 calls MAPE按MB1/MB4/MB16分别为3.31%/13.83%/38.81%，说明microbatch离散调度是PP代表规模误差的核心放大器。旧Phase 24 PP静态公式数字不再用于正式scheduler结论。

compact32→exact H32本身已有12.31% calls MAPE和0.2991 TV；exact H32→Hfull则为71.99% calls MAPE和0.4209 TV。两段误差不能相加，但足以说明“代表规模不足”和“低维重建误差”都存在；只有完成Hfull监督训练后才能判断低维画像是否仍然足够。

正式目录：`experiment-results/phase25d_pp_scheduler_representative_convergence/`
正式提交：`69495018789147d3a5865a6bf1e5a95b71d7a627`

### 19.3 首版研究线的最新状态

TP：Phase 24的代表规模分析暂时保留；Phase 25A full-window TP teacher已有GPU smoke，还需少量跨模型/策略/TP sentinel，然后把Phase 16监督目标从H32切换到Hfull并重训。

PP：Phase 25B Hfull teacher已通过Phase 25A smoke和Phase 25C tail GPU审计；Phase 25D正式证明H32/H64/H128均不能统一替代Hfull。下一步直接以Hfull训练compact predictor，并按MB1/MB4/MB16分层分析。

部署口径不变：最终输入仍是低维历史画像、模型结构、固定策略和已确定TP/PP配置；完整请求列表仅用于离线teacher。online arrival-aware仍是扩展，不是首版主目标。

当前远端、本地和tracking HEAD均为`8e51aeeb1e545ebc4f4cc18370e0cbcb294e680b`。Phase 25D正式结果提交仍为`69495018789147d3a5865a6bf1e5a95b71d7a627`；后续`8e51aeeb`仅将Phase 25A–25D顶层README改为中文并更新manifest。本地`data/`和远端旧Phase16/19、Phase23 PID/`.tmp`均未触碰。

---

## 20. Phase 26A：TP full-window teacher正式闭环

Phase 26A使用4个完整窗口GPU cell覆盖三个正式模型、TP2/4/8和三种固定策略：

- Phase 25A已有Qwen3-8B/TP2/42请求smoke；
- 新增Qwen3-30B-A3B/TP4/42请求；
- 新增DeepSeek-V2-Lite/TP8/312请求；
- 新增Qwen3-8B/TP8/48请求、最大prompt 6,216的尾部窗口。

每个cell同时运行latency、balanced、throughput，并分别核对prefill/decode。结果4/4 cell、24/24 phase完全一致；精确payload直方图calls与logical-bytes L1均为0，标量仅有机器精度求和残差。

因此，原Phase 25A的1,296条TP Hfull标签正式晋升，覆盖3模型×TP2/4/8×24画像×3策略×2 phase。全量标签由GPU验证过的结构公式离线生成，并非1,296次GPU逐条实测。

正式目录：`experiment-results/phase26a_tp_hfull_teacher_audit/`
正式结果提交：`65038792cfe293060fe4ff77436105c10c18ec50`
当前HEAD：`2e9bae7e7f9e97ba6bd00a26adcac86e771e64ad`

下一阶段不再争论是否使用H32：TP与PP均使用Hfull监督。Phase 26B构造统一数据；Phase 26C重新训练direct DNN和H0+bounded residual；Phase 26D用相同profile-level holdout正式评测。H0无学习参数，只在Hfull口径重新计算。

---

## 21. Phase 26B：统一Hfull监督数据已闭环

Phase 26B将TP与PP的权威Hfull teacher放进同一训练契约：TP来自Phase 26A晋升后的1,296条标签，PP来自Phase 25B scheduler-faithful的432条标签。最终形成1,728条phase-level样本和864个配置级total case，每个target都对应一条compact32 H0和一条只含部署可用信息的低维输入。

输入特征共55列，覆盖低维历史画像、模型结构、固定TP/PP size、固定执行策略和phase。完整请求列表没有进入训练数据。Phase 16原画像划分被固定为5 train、5 validation、5 temporal test、8 external test、1 external synthetic，后续必须以profile为最小隔离单元。

TP和PP的原生12桶范围不同：TP为4 KiB–512 MiB，PP为4 KiB–8 GiB。Phase 26B显式携带`bin_schema_id`与边界，后续共享模型也必须使用分支输出语义，不能直接混合桶编号。

重训前compact32 H0相对Hfull的配置级total结果：

| 并行 | cases | calls MAPE | calls WAPE | bytes MAPE | bytes WAPE | TV | norm EMD | common cost MAPE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TP | 648 | 13.14% | 11.67% | 3.29% | 1.33% | 0.3542 | 0.0157 | 8.40% |
| PP | 216 | 62.25% | 21.07% | 3.29% | 1.33% | 0.4483 | 0.0489 | 13.85% |

PP的calls和直方图基线明显更难，但这仍不能证明低维画像不可预测；只有Phase 26C使用Hfull监督重训，并在Phase 26D画像级留出后，才能判断需要更强画像还是只需学习稳定残差。

正式目录：`experiment-results/phase26b_unified_hfull_training_dataset/`
构建脚本提交：`5b25992f624ad274a52872aee03d2992c57912e3`
完整归档与当前HEAD：`0944dc5e34172479ee097d53b0a4d9565b2d7ad5`

目录共16个文件，manifest 15/15远端与本地通过；README、summary、audit、contract、labels、baseline、features、split、phase/total/aggregate指标、build log和DONE齐全。下一步进入Phase 26C重训direct、H0和H0+bounded residual，首先按TP/PP与phase分项报告。

---

## 22. Phase 26C/26D：Hfull重训与正式画像留出结论

Phase 26C用5个train画像拟合、5个validation画像早停，TP和PP分别训练direct与H0+bounded residual。四个checkpoint、训练历史、validation预测/指标和图表已归档。14个测试画像未参与训练、标准化或模型选择。

Phase 26D冻结checkpoint，在5个temporal、8个external和1个external synthetic画像上正式测试。全部测试画像配置级total：

| 并行 | 方法 | calls MAPE | calls WAPE | bytes MAPE | TV | norm EMD | cost MAPE |
|---|---|---:|---:|---:|---:|---:|---:|
| TP | H0 | 11.60% | 10.75% | 3.50% | 0.1379 | 0.0142 | 7.10% |
| TP | bounded residual | 14.70% | 16.33% | 9.43% | 0.1407 | 0.0144 | 12.29% |
| TP | direct | 54.85% | 55.98% | 79.09% | 0.3738 | 0.0456 | 53.64% |
| PP | H0 | 70.44% | 27.42% | 3.50% | 0.2854 | 0.0502 | 9.81% |
| PP | bounded residual | 58.24% | 27.77% | 5.23% | 0.2716 | 0.0465 | 10.32% |
| PP | direct | 61.97% | 59.38% | 73.85% | 0.5306 | 0.0836 | 54.22% |

正式判断：

1. TP旧residual在Temporal近似持平、在External明显退化；这说明当前5训练画像和55列特征不足，不能解释为TP不需要DNN。H0只作为当前baseline/fallback，最终设计仍为`H0 + DNN residual`；
2. PP必须按microbatch区分：MB1保留H0，全部测试画像calls MAPE 8.77%、cost MAPE 3.03%；
3. PP MB4的bounded residual作为待复验候选，calls MAPE从42.09%降为33.19%、TV从0.2253降为0.2167，但cost从9.93%小幅升至10.26%；
4. PP MB16 residual虽把calls MAPE从160.46%降至125.36%，仍远不合格；
5. Direct DNN在TP/PP均拒绝；
6. 当前问题不在Hfull teacher，而在compact画像对PP离散scheduler状态的表达不足，尤其MB16。

TP因此不能在Phase 26终止DNN路线；应扩充独立Hfull窗口并增加TP batching-sensitive低维特征后重训有界residual，H0作为结构先验和失效回退。PP下一阶段不能回退到H32/H64/H128，也不应重跑teacher；应新增不泄漏请求列表的scheduler-sensitive低维统计，例如长prompt跨chunk风险、预计chunk段数/请求、相对lane数的活跃序列生存统计与microbatch packing边界特征，再用新增历史画像做独立复验。

Phase 26C目录：`experiment-results/phase26c_hfull_predictor_training/`，最终图表提交`601be475`。
Phase 26D目录：`experiment-results/phase26d_hfull_profile_holdout_evaluation/`，正式归档及当前HEAD：`f268873bfdaac342c6ad4786bc8dc722b3ee6d9f`。

Phase 26D共12个文件、manifest 11/11双端通过；4,536条测试预测，0条train/validation泄漏。首版策略选择来自本轮holdout，必须用新增窗口复验，不能把同一测试集上的组合成绩再称为无偏测试结果。

---

## 23. Phase 27A–27D：PP调度敏感画像的新增窗口验证

### 23.1 为什么需要新窗口

Phase 26D已经用于发现旧画像在PP MB4/MB16上的问题，因此不能继续用同一14个测试画像
调参后报告无偏提升。Phase 27A先从66,642个Phase 15窗口中排除原24画像，再按6个segment
各取10个history-only medoid；在生成标签前冻结为30 train、12 validation和18 confirmation。

特征设计只增加部署时可从常态历史窗口聚合的低维统计：prompt跨4096-token chunk的比例和
段数、chunk×output联合分布、相邻chunk类转移、多chunk连续段及局部packing峰值。完整请求
列表仍只用于离线聚合与Hfull teacher，不进入训练表。

### 23.2 数据、训练与隔离

60个新窗口包含50,274条请求，覆盖PP2/4/8×MB1/4/16×prefill/decode，共1,080条Hfull
target。Phase 27C在相同30/12开发画像上对照52列旧特征residual与108列增强residual，
并在读取确认真值前把4种方法的1,296行确认预测写入Git、冻结SHA-256。

开发验证total：

| 方法 | calls MAPE/WAPE | bytes MAPE/WAPE | TV | norm EMD | cost MAPE/WAPE |
|---|---:|---:|---:|---:|---:|
| H0 | 59.40% / 18.50% | 2.05% / 1.71% | 0.2491 | 0.0418 | 8.92% / 6.83% |
| 旧residual | 28.53% / 16.95% | 2.68% / 1.85% | 0.1854 | 0.0275 | 7.74% / 6.48% |
| 增强residual | 23.78% / 9.77% | 2.52% / 2.02% | 0.1711 | 0.0255 | 4.52% / 3.64% |
| direct | 17.38% / 16.25% | 38.70% / 16.16% | 0.1309 | 0.0156 | 16.64% / 10.39% |

Direct虽然calls/TV较低，但bytes和cost严重失败，继续拒绝。

### 23.3 18窗口独立确认结果

Phase 27D只核验manifest、冻结预测hash并join真值，没有训练或重新选择模型：

| 方法 | calls MAPE/WAPE | bytes MAPE/WAPE | TV | norm EMD | cost MAPE/WAPE |
|---|---:|---:|---:|---:|---:|
| H0 | 62.13% / 20.08% | 3.15% / 1.28% | 0.2733 | 0.0444 | 10.96% / 7.07% |
| 旧residual | 30.56% / 15.56% | 3.92% / 1.87% | 0.2096 | 0.0317 | 8.03% / 5.46% |
| 增强residual | 25.84% / 11.35% | 3.46% / 1.97% | 0.1844 | 0.0280 | 6.52% / 4.32% |
| direct | 31.83% / 19.58% | 34.12% / 14.21% | 0.1376 | 0.0174 | 23.30% / 10.74% |

增强residual相对旧residual的额外收益在独立窗口重复，说明chunk/顺序摘要确实补充了信息。
分策略却不是统一成功：

- MB1：H0 calls/TV/cost为5.82%/0.0672/2.00%；增强residual为6.59%/0.0439/4.40%。
  只改善TV，冻结候选失败，回退H0；
- MB4：H0为38.55%/0.2277/10.90%；增强residual为15.38%/0.1347/6.67%，三项改善；
- MB16：H0为142.02%/0.5249/19.98%；增强residual为55.55%/0.3747/8.48%，三项改善，
  但calls绝对误差仍不合格。

因此确认后建议为`MB1=H0、MB4/MB16=增强residual候选`。它是看过Phase 27D后形成的
新映射，Phase 27D不能再为它提供无偏混合总分；下一批窗口只需验证这份映射。若MB16仍高，
应转向预测结构化scheduler事件摘要，而不是继续堆叠黑盒特征。

### 23.4 资产与当前状态

- Phase 27A：`experiment-results/phase27a_pp_feature_and_holdout_contract/`，9文件、48 KiB；
- Phase 27B：`experiment-results/phase27b_pp_hfull_dataset/`，15文件、2.9 MiB；
- Phase 27C：`experiment-results/phase27c_pp_scheduler_feature_training/`，17文件、816 KiB；
- Phase 27D：`experiment-results/phase27d_pp_independent_confirmation/`，12文件、1.0 MiB。

四个manifest分别8/8、14/14、16/16、11/11双端通过。当前远端、GitHub、本地与tracking
HEAD均为`e5a3612eea71ea0ba47285209c588079460c1539`。本地`data/`和远端Phase16/19/23受保护
旧资产未触碰；所有正式资产均通过显式路径选择性提交。

---

## 24. Phase 28A–28C：PP冻结混合映射的第二独立确认

### 24.1 为什么还需要第二确认集

Phase 27D确认了增强调度特征对MB4/MB16有效，但`MB1=H0、MB4/MB16=增强residual`是看过
Phase 27D后才形成的新映射，因此Phase 27D不能再给它无偏混合总分。Phase 28A同时排除
Phase 16和Phase 27共84个已用窗口，在任何Phase 28预测或标签产生前冻结18个新窗口和上述
方法映射。

Mooncake synthetic全库仅12个候选窗口，扣除Phase 16的1个和Phase 27的10个后只剩1个，
因此最终事前配额为3/3/3/4/4/1，而不是机械要求每segment 3个。这个调整发生在预测和Hfull
标签之前，没有数据选择泄漏。

### 24.2 预测与真值的隔离顺序

Phase 28B先从18个窗口聚合324行108列低维特征，生成648行H0/增强residual预测，并冻结
预测文件SHA-256 `1127ec08...376e`。训练模型沿用Phase 27C checkpoint；为使本地构建无需
新装PyTorch，checkpoint被无损导出为NumPy，逐数组相等且Torch/NumPy推理最大绝对差仅
`2.5332e-7`。Phase 28B没有Hfull target参数，也没有生成或读取Hfull真值。

Phase 28C先验证Phase 28B manifest和预测hash，之后才对18个完整窗口、15,440条请求运行
GPU验证过的PP fixed-draining结构公式，生成324条Hfull phase labels。完整请求列表仅在内存
中作为teacher输入，没有保存或进入预测器。

### 24.3 第二独立确认结果

配置级total：

| 方法 | calls MAPE/WAPE | bytes MAPE/WAPE | TV | norm EMD | cost MAPE/WAPE |
|---|---:|---:|---:|---:|---:|
| H0 | 62.05% / 18.85% | 2.70% / 1.25% | 0.2606 | 0.0458 | 10.97% / 6.47% |
| 冻结混合映射 | 24.82% / 9.42% | 3.08% / 1.64% | 0.1885 | 0.0296 | 5.25% / 3.35% |

相对H0，冻结映射使calls MAPE降低60.00%、calls WAPE降低50.04%、TV降低27.67%、EMD降低
35.28%、cost MAPE降低52.15%；logical-bytes MAPE小幅增加0.39个百分点。

分microbatch：

- MB1固定H0：calls MAPE 6.37%、TV 0.0801、cost MAPE 2.36%；
- MB4增强residual：calls MAPE从34.45%降到13.59%，TV从0.1878降到0.1083，cost从8.96%
  降到5.13%；
- MB16增强residual：calls MAPE从145.33%降到54.49%，TV从0.5139降到0.3771，cost从21.60%
  降到8.26%。

这使`MB1=H0、MB4/MB16=增强residual`成为当前Qwen3-8B PP fixed-draining的已独立确认基线。
但MB16的54.49% calls MAPE仍然太高，只能说相对改善稳定，不能说问题已解决。

### 24.4 现在TP和PP分别往哪里走

TP的Hfull teacher口径已确定，但需要重新训练DNN。Phase 26D只证明当前5训练画像、55列特征
下的旧residual不如H0，不支持取消DNN。下一步复用Phase 27/28相同历史窗口和split，为三个
TP模型生成Hfull，增加batching-sensitive低维特征并重训`H0 + bounded residual`。placement/
topology曲线可以并行补测，但不能代替TP预测器闭环。

PP关闭Phase 27D和Phase 28C holdout，不再在其上调参。MB1保留H0，MB4保留增强residual；
针对MB16应建立新的开发/验证窗口，预测更接近调度器机制的结构化事件摘要（如chunk轮次、
活跃lane生存、microbatch形成/尾部碎片），再确定性映射成消息直方图。该方法定型后，必须在
另一批未见窗口确认；随后才扩展到新PP模型，检验跨模型泛化。

### 24.5 资产与状态

- Phase 28A：9文件、约36 KiB、manifest 8/8；
- Phase 28B：12文件、约208 KiB、manifest 11/11；
- Phase 28C：15文件、约1.7 MiB、manifest 14/14。

当前远端node55、GitHub、本地与tracking HEAD均为
`1b67bc8f5a1193cc7b041e511c858f4a0bbe4362`。本地`data/`和远端Phase16/19/23受保护资产未
触碰；所有正式结果均已通过Git双端保存。

---

## 25. Phase 29A–29D3：TP对齐重训与两级独立确认

### 25.1 数据与方法如何与PP对齐

Phase 29复用Phase 27完全相同的60个窗口和30 train/12 validation/18 first confirmation划分，
并复用Phase 28的18个second confirmation窗口。三个TP模型为DeepSeek-V2-Lite、Qwen3-8B和
Qwen3-30B-A3B；固定配置覆盖TP2/4/8、latency/balanced/throughput和prefill/decode。

完整请求列表只用于离线生成Hfull teacher。训练与推理输入仍是低维历史画像、模型结构、固定
TP size、固定策略、phase和compact32 H0。对照方法为H0、55列legacy bounded residual、
113列TP batching-sensitive enhanced bounded residual和113列direct控制。主研究架构始终是
`H0结构先验 + bounded residual DNN`；H0兼任baseline和失效保护回退。

Phase 29B共聚合78个历史窗口、65,714个请求。60个Phase 27窗口生成3,240条Hfull target；
42个开发窗口形成2,268条训练/验证phase rows；两批各18个确认窗口分别有972条feature。
第一确认target物理隔离；第二确认在预测和第一确认后映射均冻结前完全没有生成target。

### 25.2 开发验证与第一独立确认

30个画像拟合、12个画像早停的开发验证total结果：

| 方法 | calls MAPE/WAPE | bytes MAPE/WAPE | TV | norm EMD | cost MAPE/WAPE |
|---|---:|---:|---:|---:|---:|
| H0 | 12.05% / 11.08% | 2.05% / 1.71% | 0.1538 | 0.0144 | 6.54% / 5.05% |
| legacy residual | 9.15% / 7.46% | 2.04% / 1.65% | 0.1547 | 0.0152 | 4.87% / 3.47% |
| enhanced residual | 7.98% / 6.87% | 2.45% / 1.63% | 0.1677 | 0.0163 | 3.95% / 3.07% |
| direct | 27.82% / 25.31% | 23.18% / 18.60% | 0.1632 | 0.0162 | 23.33% / 18.69% |

开发验证只冻结候选：latency/balanced使用legacy residual，throughput使用enhanced residual。
两批确认集四方法预测各3,888条在任何确认评测前同时写入Git并冻结hash。

第一独立确认18个窗口的total：

| 方法 | calls MAPE/WAPE | bytes MAPE/WAPE | TV | norm EMD | cost MAPE/WAPE |
|---|---:|---:|---:|---:|---:|
| H0 | 12.45% / 10.97% | 3.15% / 1.28% | 0.1581 | 0.0156 | 6.78% / 5.04% |
| legacy residual | 11.51% / 5.76% | 3.33% / 1.58% | 0.1959 | 0.0212 | 4.98% / 2.63% |
| enhanced residual | 12.28% / 5.83% | 3.68% / 1.74% | 0.1988 | 0.0214 | 5.41% / 2.65% |
| direct | 77.17% / 30.25% | 32.11% / 20.67% | 0.1619 | 0.0187 | 40.09% / 20.25% |

分策略确认后，latency的legacy在calls/TV/cost三项均赢，balanced赢calls/cost但TV退化，
throughput的enhanced只赢cost。因此面向第二确认的映射冻结为
`latency=legacy residual、balanced=legacy residual、throughput=H0`。

### 25.3 第二独立确认与首版保护结论

Phase 29D2先核验第二确认预测SHA-256
`45b7382e8be44e3ba81a95c86809f80766b19dd7ccfa0efcf938fe0f248bd44c`和上述映射，之后才读取
18个窗口、15,440个完整请求生成972条Hfull target。Phase 29D3只做冻结预测与真值join，
没有训练、调参或改映射。

第二独立确认四方法total：

| 方法 | calls MAPE/WAPE | bytes MAPE/WAPE | TV | norm EMD | cost MAPE/WAPE |
|---|---:|---:|---:|---:|---:|
| H0 | 12.44% / 10.96% | 2.70% / 1.25% | 0.1439 | 0.0145 | 7.05% / 4.71% |
| legacy residual | 13.48% / 7.97% | 3.24% / 1.44% | 0.1687 | 0.0186 | 8.15% / 3.48% |
| enhanced residual | 12.64% / 7.88% | 2.70% / 1.29% | 0.1627 | 0.0181 | 7.77% / 3.53% |
| direct | 30.84% / 24.79% | 25.30% / 18.52% | 0.1624 | 0.0180 | 26.50% / 18.55% |
| 第一确认后冻结映射 | 13.13% / 8.09% | 2.99% / 1.38% | 0.1468 | 0.0153 | 7.80% / 3.49% |

第二确认中，latency legacy相对H0的calls MAPE、TV、cost MAPE全部退化；balanced legacy也
全部退化且cost guard失败；throughput本来就是H0。因此当前首版安全映射为三个策略全部H0。

这个结果不能解释为“TP不需要DNN”。相反，它说明当前30个独立训练画像和113列增强特征下，
residual对大流量cell的WAPE有稳定收益，但逐cell MAPE与直方图形状泛化不稳，尚未通过两轮
确认。研究设计仍保留H0+DNN；当前checkpoint被拒绝，H0只是部署保护回退。下一轮必须使用
新的开发窗口研究分层/结构化TP batch事件输出、与MAPE/TV一致的多目标损失和可校准gating，
现有两批确认窗口关闭，不得继续调参。

### 25.4 资产、Git与下一步

- Phase 29A：`phase29a_tp_aligned_contract/`，11文件、56 KiB、manifest 10/10；
- Phase 29B：`phase29b_tp_hfull_dataset/`，16文件、3.1 MiB、manifest 15/15；
- Phase 29C：`phase29c_tp_aligned_training/`，18文件、3.3 MiB、manifest 17/17；
- Phase 29D1：`phase29d1_tp_first_confirmation/`，11文件、3.6 MiB、manifest 10/10；
- Phase 29D2：`phase29d2_tp_second_confirmation_targets/`，8文件、984 KiB、manifest 7/7；
- Phase 29D3：`phase29d3_tp_second_confirmation/`，11文件、3.5 MiB、manifest 10/10。

关键提交：Phase 29A归档`2fd9820e`，29B归档`23f77a84`，29C归档`051b461c`，29D1归档
`99ad001b`，29D2归档`718ac2f9`，29D3归档及当前HEAD
`586c7708b334c8f32b467376198ea6edc5041be8`。远端node55、GitHub、本地与tracking HEAD一致，
所有manifest双端通过。本地公共raw trace、缓存、失败中间目录和远端smoke临时目录已删除；
本地`data/`及远端Phase16/19/23保护资产未触碰。

下一步TP不能在Phase 29确认集上继续调参：应新建开发/验证窗口，优先预测可解释的batch count、
token-budget截断和尾批碎片等结构化事件，再映射为消息直方图，并在新的确认窗口复验。PP继续
保持`MB1=H0、MB4/MB16=增强residual`的已确认基线，同时单独解决MB16绝对误差和跨模型泛化。
placement/topology连续代价曲线可并行推进，但不能替代TP/PP画像预测器闭环。

---

## 26. Phase 30A–30D3：TP结构事件DNN与两级独立确认

### 26.1 为什么从直方图残差改成结构事件残差

Phase 29说明直接学习12桶消息编码残差可以改善部分大流量cell的WAPE，但逐cell calls MAPE、
TV和cost在两级确认中不稳定。Phase 30因此把模型相关因素与scheduler因素拆开：DNN只预测
fixed-draining如何形成batch与decode active lanes，之后用模型结构确定性恢复消息直方图。

62维事件目标包括：

- 23个prefill token-sum联合区间的batch count；
- 同23个区间的input-token mass；
- decode active lanes 1到16的step count。

事件DNN输入为91列低维历史画像与固定batching策略特征；模型结构、TP size、phase不进入DNN。
模型的collectives/forward与bytes/token由适配器使用。TP size仍保留在预测合同和审计维度，
但当前拓扑无关teacher对TP2/4/8给出相同事件→消息映射。

### 26.2 数据划分与隔离

Phase 30A关闭Phase 29确认窗口，另外选择90个新画像：45 train、15 validation、15 first、
15 second。由于synthetic候选已耗尽，本轮使用BurstGPT三段、conversation、toolagent五段，
每段18个。再加入Phase 29允许复用的30 train和12 validation，最终为75 train、27 validation、
15 first、15 second。

训练单位是画像×策略：225个拟合、81个早停；不是模型×TP×phase重复样本。两批确认各45个
事件feature单位。第一、第二确认四方法预测各3,240条，在读取第一target前同时冻结。Phase 30B
结构适配器审计覆盖6,318条Hfull与7,128条H0 phase行，最大相对误差`2.548e-12`。

### 26.3 开发验证与两级确认结果

开发验证：

| 方法 | calls MAPE | bytes MAPE | TV | cost MAPE |
|---|---:|---:|---:|---:|
| H0 | 14.33% | 2.44% | 0.1533 | 8.55% |
| Phase 29 residual诊断 | 15.72% | 2.66% | 0.1619 | 9.74% |
| 结构事件bounded residual | 27.10% | 19.69% | 0.1505 | 19.27% |
| 结构事件direct | 20.84% | 25.65% | 0.1029 | 20.07% |

结构residual只赢TV，calls/bytes/cost均明显失败，latency/balanced/throughput全部回退H0。

第一独立确认：

| 方法 | calls MAPE | bytes MAPE | TV | cost MAPE |
|---|---:|---:|---:|---:|
| H0 | 11.45% | 1.51% | 0.1657 | 6.57% |
| Phase 29 residual诊断 | 11.14% | 2.09% | 0.1837 | 7.13% |
| 结构事件bounded residual | 27.96% | 23.53% | 0.1610 | 19.22% |
| 结构事件direct | 16.03% | 46.90% | 0.1218 | 21.72% |

第二映射不根据确认诊断重选，继续冻结全H0。Phase 30D2随后才从15个第二窗口、10,298请求生成
45条target；405配置、810 phase行最大相对误差`1.939e-12`。

第二独立确认：

| 方法 | calls MAPE | bytes MAPE | TV | cost MAPE |
|---|---:|---:|---:|---:|
| H0 | 12.16% | 2.97% | 0.1345 | 7.13% |
| Phase 29 residual诊断 | 9.67% | 2.92% | 0.1534 | 6.22% |
| 结构事件bounded residual | 23.63% | 33.44% | 0.1419 | 19.31% |
| 结构事件direct | 14.00% | 49.19% | 0.0946 | 22.60% |

Phase 29诊断在第二批calls/cost更好，但第一批cost/TV更差，且它不是Phase 30可选择候选，不能
事后晋级。最终保护映射保持三策略全H0。

### 26.4 这次阴性结果如何解释

可以得出：当前91→62结构事件bounded residual、当前多目标loss和当前checkpoint没有通过。
direct在开发和两批确认中持续改善TV，却严重伤害bytes与cost，说明低维画像并非完全没有信息，
而是事件目标和loss没有把规模、形状、bytes与代价同时校准。

不能得出：TP不需要DNN。研究架构仍为`H0 + DNN residual`，H0是结构先验、baseline和失败
fallback。若再次研究TP DNN，必须使用全新开发与确认窗口，重做事件目标、分层loss或gating；
Phase 29/30确认集全部关闭。

### 26.5 当前TP/PP状态与下一核心实验

TP：Phase 30已完成两级确认，当前保护映射全H0；结构事件DNN为阴性研究结果，不能再在现有
确认集调参。

PP：Phase 28已确认`MB1=H0、MB4/MB16=增强residual`，但MB16 calls MAPE仍为54.49%，且尚未
跨模型。下一核心实验应优先在PP新窗口上定义microbatch形成、chunk轮次、active-lane生存和
尾部碎片等结构事件，训练H0事件加residual DNN，再用全新确认集验证，之后扩展PP模型。

Phase 30六目录共75文件，manifest 9/9、15/15、17/17、10/10、7/7、11/11双端通过。当前
node55、GitHub、本地与tracking HEAD均为`fbca63a405639e5cc8c14ac6eaf779cad036cb0b`。raw、
pycache、失败与smoke临时目录已删除；本地`data/`及远端Phase16/19/23保护资产未触碰。

---

## 二十七、Phase 31A-G：三模型TP/PP有限收敛实验（2026-08-13状态补充）

> 本节只更新实验状态和证据，不修改本导引的研究目标、fixed-draining语义、输入合同、Hfull teacher定位或调度器边界。今晚的松紧程度与停止条件见《今晚TP_PP收敛执行参考_不替代实验总导引.md》。

### 27.1 请求级隔离修复与数据合同

独立审计发现旧Phase27-30的部分300秒Mooncake窗口以60秒滑动，window id不同但可能共享请求，因此这些旧确认结果降级为历史诊断，不能作为严格独立确认。Phase31A重新冻结59个300秒请求级互斥正常画像：39个训练、10个验证、10个固定预测；新角色之间共享请求为0，并对旧Phase27/28/30确认区间设置300秒embargo。

三个已知模型DeepSeek-V2-Lite、Qwen3-8B、Qwen3-30B-A3B均进入训练、验证和固定预测，不做整模型留出。开发Hfull teacher使用21,058个完整窗口请求，固定评测使用2,786个请求；开发/固定target分别为5,292/1,080条phase rows。完整请求仍只用于离线teacher，预测输入仍是低维画像、模型结构、固定并行配置和策略。

### 27.2 Phase31C统一TP/PP训练

TP和PP均保持`H0 + DNN residual`。每个方向初筛12组配置，验证集选择前2组进行3-seed确认，并在固定target生成前冻结2,160条H0/DNN phase预测。

开发验证整体结果：

| 方向 | 方法 | calls WAPE | bytes WAPE | TV | EMD | cost WAPE |
|---|---|---:|---:|---:|---:|---:|
| TP | H0 | 10.61% | 2.68% | 0.1621 | 0.0209 | 6.73% |
| TP | H0+DNN residual | 7.35% | 2.72% | 0.1611 | 0.0203 | 4.45% |
| PP | H0 | 8.59% | 2.68% | 0.2133 | 0.0278 | 6.13% |
| PP | H0+DNN residual | 5.03% | 2.97% | 0.1795 | 0.0206 | 3.66% |

TP选择完整特征、共享64×64 bounded residual、学习率0.003、3-seed ensemble、alpha 0.75；PP选择完整特征、按MB小头、学习率0.001、3-seed ensemble、alpha 1.0。两个方向的residual均非零，没有退回全H0。

### 27.3 固定预测集最终裁定

Phase31D在Phase31C预测及SHA归档后才生成固定Hfull target。固定集整体结果：

| 方向 | 方法 | calls WAPE | bytes WAPE | TV | EMD | cost WAPE | 裁定 |
|---|---|---:|---:|---:|---:|---:|---|
| TP | H0 | 19.48% | 3.62% | 0.2476 | 0.0245 | 12.60% | baseline |
| TP | H0+DNN residual | 14.33% | 2.97% | 0.2458 | 0.0240 | 8.99% | fail |
| PP | H0 | 13.28% | 3.62% | 0.2637 | 0.0331 | 9.43% | baseline |
| PP | H0+DNN residual | 6.91% | 4.05% | 0.1593 | 0.0199 | 5.42% | conditional pass |

TP calls/cost相对H0改善26.47%/28.64%，三个模型改善方向一致，但绝对calls/cost仍超过有条件阈值12%/6%，因此不能收口。PP calls/cost相对H0改善47.99%/42.53%；bytes与cost略超正式阈值，故只能有条件通过。PP MB16 calls MAPE由104.16%降至36.02%，相对改善65.42%，通过单独保护条件。

### 27.4 TP最后有限轮与停止原因

Phase31E按预定路线追加6组加权总量loss及共享、policy、model、model×policy小头，使TP累计达到18组搜索上限。所有训练和选型仍只读开发数据与target-free固定特征。最好的新候选验证分数未超过Phase31C incumbent，因此没有用固定target挑模型，也没有覆盖最佳checkpoint。Phase31F对保留incumbent做同一固定集一致性复评，指标逐值不变；该复评不是新盲测。

当前诚实结论为：PP在当前三个已知模型和正常历史流量范围内有条件收口；TP residual有稳定价值但未达到收口阈值；整体第一阶段尚未完全收口。继续TP时不得在已打开的固定集上调参，应先target-blind冻结新的请求级互斥确认集，再扩充开发窗口或改进低维顺序/形状residual。

Phase31完整证据入口为仓库`experiment-results/phase31g_tp_pp_convergence_final/`，其中包含最终报告、逐模型/逐policy/整体指标、图表、交接、资产索引、日志、DONE和manifest。


# Phase32F状态补充：扩容有限收敛结果

基础研究定义、Hfull teacher、fixed-draining语义、指标与阈值均未改变。Phase32将TP累计搜索从18组扩到48组绝对上限，将PP从12组扩到30组常规上限，并新增9个与既有角色请求区间互斥的BurstGPT正常确认窗口。

最终结论：PP H0+DNN residual在一次性新确认上达到有条件通过，calls/bytes/TV/EMD/cost WAPE分别为4.51%/3.98%/0.1426/0.0183/3.48%；TP虽相对H0持续改善，但在48组上限处仍fail，救援后重复工程calls/cost WAPE为12.19%/8.57%。因此第一阶段当前是“PP有条件收口、TP未收口”。

证据边界：新增确认仅BurstGPT；TP救援发生在Phase32C target开放之后，尽管训练与gate选择未读取target，其复评仍不是新盲测。
