# PatternDemand 实验结构总导引（截至 Phase73）

> 证据截止点：正式分支 `experiment/pattern-demand-v0.5.15-clean` 上的 Phase73 结果提交 `a13e4ba83dc683edd3493d6d7aff03b207daa688`。本文是导引更新，不改写任何既有结果目录、盲测标签或冻结结论。Phase72 的 `F01–F18` 与 `N01–N15` 仍是正式声明边界；本文只把 Phase73 的独立 baseline 纳入统一叙事。

## 1. 一句话研究目标

在不知道未来完整请求列表的情况下，使用常态历史流量的低维画像、模型结构、固定执行策略和已经确定的 TP/PP/PD 配置，预测 fixed-draining 语义下拓扑无关的 12-bin 消息直方图；再把直方图代入候选 placement/topology 的物理通信曲线，估计 communication-only 通信代价。

当前预测器不选择 TP/PP size，也不负责完整调度。并行配置是输入，placement/topology 是后续选择对象。

## 2. 完整数据流

```text
完整请求窗口
  └─> GPU 验证过的 scheduler-faithful CPU teacher
        └─> Hfull：离线训练/评估标签

低维请求画像 + 模型结构 + 固定策略 + 固定并行配置
  ├─> H0：结构先验/可解释基线
  │     └─> DNN residual 修正
  │           └─> H0 + DNN：当前主预测器
  └─> Direct-GBDT：不使用 H0、伪请求或 teacher 的独立直接预测 baseline

H0 / H0+DNN / Direct-GBDT 预测的 12-bin 直方图
  └─> 冻结 L1/L2/L3 物理曲线卷积
        └─> communication-only cost 与 placement 比较
```

### 四个容易混淆的对象

1. **Hfull**：完整请求窗口经 GPU 校准的 CPU teacher 生成的标签。它是离线真值，不是线上输入。
2. **H0**：低维画像经结构公式、确定性伪请求和 scheduler-faithful 逻辑得到的可解释预测。它不是 Hfull。
3. **H0+DNN residual**：DNN 只修正 H0 没解释好的残差，并保留 H0 保护门。TP、PP、PD 当前主方法均采用这一形式。
4. **Direct-GBDT**：直接从低维 `feature_*` 预测 12-bin calls/bytes，不读取 `h0_*`、完整请求、伪请求或 teacher。它是 Phase73 新增的真正独立 baseline。

## 3. 研究边界

### 已包含

- fixed-draining，而不是 online arrival-aware；
- 拓扑无关的 12-bin calls 与 logical bytes 直方图；
- 六个已知模型：Qwen3-8B、Qwen3-30B-A3B、DeepSeek-V2-Lite、Llama-3.2-3B-Instruct、Qwen2.5-14B-Instruct、Mixtral-8x7B-Instruct-v0.1；
- TP/PP 的固定并行配置，以及纯 PD 的固定 P/D 图；
- 冻结 L1/L2/L3 物理曲线和 communication-only placement；
- 纯 PD P1D1、两流 P1D2/P2D1、以及已测四流图的轻量拥塞修正。

### 尚未包含

- 计算时间、显存可行性、资源空闲、排队、无关作业拥塞；
- 通信与计算重叠；
- 端到端延迟、吞吐或线上收益；
- 调度器选择 TP/PP size、P/D 实例数或扩缩容；
- P 或 D 内部再包含 TP/PP 的混合并行；
- 未见第七模型、超过四流或未测图的自动泛化；
- 从边际 12-bin 直方图恢复真实消息并发配对。

## 4. 截至 Phase73 的主链状态

| 链 | teacher / 预测器 | 物理曲线 | cost / placement | 当前状态 |
|---|---|---|---|---|
| TP | Phase34D 六模型 fresh-blind：H0+DNN 正式优于 H0 | Phase39：TP2/4/8 × L1/L2/L3，共 9 条 | Phase39：648 个固定 TP case；communication-only top1 100%，regret 0 | 冻结完成（范围内） |
| PP | Phase34D 六模型 fresh-blind：H0+DNN 正式优于 H0 | Phase39：PP × L1/L2/L3，共 3 条 | Phase39：648 个固定 PP case；communication-only top1 100%，regret 0 | 冻结完成（范围内） |
| 纯 PD P1D1 | Phase40/47 六模型 GPU teacher 对齐；Phase50 六模型 H0+DNN 相对 H0 改善 | Phase51：六模型 × L1/L2/L3，共 18 条、396 knots | Phase52：三层 cost 均改善；第一版 placement 链完成 | 冻结完成（范围内） |
| PD 两流 | Phase60 实测，Phase61 学习轻量拥塞修正 | Phase62 fresh-blind；Phase63 扩到六模型外部证据 | 支持 P1D2/P2D1 已测范围 | 冻结完成（范围内） |
| PD 四流 | Phase64/66/68 失败均保留；Phase69 冻结高-page residual | Phase70 第三次 fresh-blind 通过 | Phase71 固定 wave 合同下 cost 21/21、placement 7/7 | 仅限两个代表模型、四个已测图、L1-L3、最多四流 |
| 独立 baseline | Phase73 Direct-GBDT | 不新增物理曲线 | 只做 target-open 直方图基准 | 完成；未优于 H0 |

## 5. TP 与 PP 结论

Phase34 把原有三模型扩为六模型，并使用 12 个 request-disjoint BurstGPT fresh windows、共 3803 个完整 teacher 请求进行盲测。预测在 target 打开前冻结，TP 与 PP 都保留 `H0+DNN residual`。

Phase35 完成统一复播和曲线接口，但当时除 TP 单机 B200 L1 外仍含 proxy，不能作为最终 L1–L3 物理证据。Phase36 证明冻结预测可在另一环境零差异复播。Phase39 才正式补齐：

- TP2/4/8 × L1/L2/L3：9 条物理曲线；
- PP × L1/L2/L3：3 条物理曲线；
- TP/PP 合计 1296 个 communication-only placement 决策，top1 agreement 100%，mean regret 0。

物理代价 WAPE：

| 链 | L1 | L2 | L3 |
|---|---:|---:|---:|
| TP | 7.57% | 7.52% | 7.85% |
| PP | 4.41% | 3.99% | 4.22% |

这些结果证明冻结直方图在已测物理曲线上的通信代价可用性，不证明端到端服务性能。

## 6. 纯 PD P1D1 结论

### teacher 与数据

- Phase40：用 Qwen3-8B 对齐 sender-side Mooncake、fixed-draining、整 wave 原子放行、page/chunk 预算和 teacher 事件；
- Phase41：用 4853 请求、82 waves 的真实完整窗口做 GPU sentinel，并生成开发画像；
- Phase47：补齐其他五个模型的 GPU teacher 验证；
- Hfull 批量标签随后由冻结 CPU teacher 离线生成，完整请求和 raw 不进入正式 Git 结果。

### 预测器演进

- Phase43 保留了 DNN 不如 H0 的小样本 fresh-blind 负结果；
- Phase44–46 通过扩大互斥开发集和 H0 保护门，使 Qwen3-8B H0+DNN 在新 blind 集上优于 H0；
- Phase48–50 扩展到六模型。Phase50 在 300 画像 × 六模型、共 1800 单元上得到：

| 方法 | calls histogram WAPE | bytes histogram WAPE | mean TV | normalized EMD |
|---|---:|---:|---:|---:|
| H0 | 25.74% | 22.65% | 13.21% | 1.443% |
| H0+DNN residual | 22.97% | 20.29% | 11.63% | 1.320% |

因此，正式结论是“六模型和三个 segment 上，H0+DNN 的受保护指标整体优于 H0”，而不是“所有直方图绝对误差都低于 10% 或 15%”。

Phase58/59 后续在 development 上继续优化形状。Phase59 达到 calls/bytes histogram WAPE 18.93%/17.29%，但逐模型、逐 segment 严格门仍未通过，`target_met=false`。这项开放问题必须保留。

### 物理代价

Phase51 生成六模型 × L1/L2/L3 的 18 条 Mooncake/RDMA 物理曲线。Phase52 将 Phase50 冻结直方图代入曲线：

| 拓扑 | H0+DNN cost WAPE |
|---|---:|
| L1 | 2.15% |
| L2 | 2.16% |
| L3 | 2.15% |

placement agreement 从 86.44% 提升到 87.22%，mean regret 从 0.0221% 降到 0.0185%。这仍是 bin-mean、communication-only 的确定性卷积。

## 7. PD 多流结论

P1D1 单链路曲线不能直接按流数相加。Phase60 的 P1D2/P2D1 实测发现共享端点的并发拥塞；Phase61 学习轻量修正，Phase62 在保留 payload 和新 GPU/主机 placement 上 fresh-blind 通过，Phase63 扩到其余四模型。

四流链没有删除失败：

1. Phase64 零样本失败；
2. Phase65 第一版图修正，Phase66 fresh-blind 失败；
3. Phase67 page-shape 修正，Phase68 第二次 fresh-blind 仍失败；
4. Phase69 只为高 page 增加轻量 residual；Phase70 第三次 fresh-blind 在两个代表模型、四个图、L1–L3 上通过，overall WAPE 0.438%。

Phase71 冻结 Phase51/R61/R69，在预注册 `bin_aligned` 边际 wave 下做确定性集成：

- cost 比较 21/21 通过；
- placement 比较 7/7 通过；
- H0+DNN 最大 cost WAPE 2.521%；
- 最低 placement agreement 99.0%。

但不同合法 wave 配对下，最大相对 cost 范围 169.3%，最低 placement 稳定率 67.7%。因此边际 12-bin 直方图不能唯一恢复真实并发关系；该限制不能被 99% 的固定-wave结果掩盖。

## 8. Phase73：Direct-GBDT 独立 baseline

Phase73 回答的问题是：如果完全不用 H0 的结构先验，只让树模型从低维画像直接预测消息直方图，是否更好？

实验只读取 84 个 `feature_*`，硬禁止 `h0_*`、`target_*` 和 `residual_*` 作为输入；不构造伪请求，不运行 teacher，不使用 raw。模型容量只在 Phase48 train/validation 上选择并冻结，之后才载入已经公开的 Phase50 target。因此它是 **target-open fixed benchmark**，不是 fresh blind。

| 方法 | calls histogram WAPE | bytes histogram WAPE | mean TV | normalized EMD |
|---|---:|---:|---:|---:|
| H0 | 25.74% | 22.65% | 13.21% | 1.443% |
| H0+DNN residual | 22.97% | 20.29% | 11.63% | 1.320% |
| Direct-GBDT | 30.79% | 39.55% | 15.05% | 1.722% |

Direct-GBDT 的 total calls WAPE 只有 0.739%，total bytes WAPE 为 9.94%，说明它能大致预测“总共有多少通信”；但它把通信分到 12 个消息尺度桶时明显更差。科学结果是：

`TARGET_OPEN_DIRECT_GBDT_DOES_NOT_BEAT_H0`

这项负结果支持 H0 的 scheduler/结构先验确实有价值，也给论文提供了与 H0+DNN 真正不同的 ML baseline。它不证明所有直接模型必然失败，也不是新盲测泛化结论。

## 9. 证据分级与正式性

- **fresh-blind**：预测/修正先冻结，随后一次性打开保留 target；可以支持相应范围内的正式泛化结论。
- **repeated engineering / target-open**：target 已打开后的确定性重算、接口集成或 baseline；可支持工程可复现性和固定基准比较，不能冒充新盲测。
- **physical measurement**：冻结环境、primitive、placement 和 payload support 内的真实曲线；不能外推成端到端服务延迟。
- **development negative**：未达到合同目标也必须保留；不能通过删除失败或继续读取 blind 标签调参来制造 PASS。

Phase54–57 未在正式 Git 树中形成结果目录，不属于当前正式证据。Phase58/59 是正式保留的 development 负结果。Phase73 是正式保留的 target-open 独立 baseline 负结果。

## 10. 当前是否还需要 GPU

在已冻结范围内，没有必须重跑的 B200 teacher 或物理曲线实验：

- TP/PP 六模型 teacher 与 H0/Hfull 语义已验证，L1–L3 曲线已完成；
- 纯 PD 六模型 teacher 已验证，P1D1 以及当前两流/四流物理证据已完成；
- Hfull 批量标签、H0、H0+DNN、Direct-GBDT 都可以在 CPU 上运行。

只有改变研究范围时才需要重新打开 GPU 合同，例如：新 SGLang/backend/Mooncake 语义、新 page/chunk/wave/scheduler 规则、新模型 KV 布局、不同 GPU/网络/拓扑、超过四流的新图、P/D 内部 TP/PP，或 online arrival-aware 实验。

## 11. 下一步建议

当前最合理的选择不是重复既有 GPU 证据，而是二选一：

1. **论文收口**：基于 Phase72/73 整理主结果图、消融、负结果和边界；Phase73 已提供独立 baseline。
2. **新 scheduler 研究**：另立合同加入计算时间、显存可行性、资源空闲、排队/拥塞、通信计算重叠，以及 L1 不可用时受约束的 L2/L3 选择。

若继续改善 PD 直方图，必须重新建立未打开标签的数据切分；不能再拿 Phase50 target 反复调参后称 fresh blind。

## 12. 关键入口

- Phase72 总导引及冻结声明：`experiment-results/phase72_conclusion_freeze_through_phase71/`
- Phase72 机器可读声明：`experiment-results/phase72_conclusion_freeze_through_phase71/audit/claim_scope.json`
- Phase73 baseline：`experiment-results/phase73_direct_gbdt_baseline/`
- TP/PP 六模型盲测：`experiment-results/phase34d_six_model_blind_evaluation/`
- TP/PP L1–L3：`experiment-results/phase39_tp_pp_l1_l3_physical_placement_validation/`
- PD 六模型盲测：`experiment-results/phase50_pd_six_model_blind_evaluation/`
- PD P1D1 曲线与代价：`experiment-results/phase51_pd_l1_l3_physical_curve_library/`、`phase52_pd_physical_cost_placement_validation/`
- PD 多流最终集成：`experiment-results/phase70_pd_high_page_residual_fresh_blind/`、`phase71_pd_multiflow_cost_placement_integration/`
- 跨环境 workflow 总入口：`workflows/patterndemand/README_CN.md`
