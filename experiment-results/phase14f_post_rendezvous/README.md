# Phase 14F：L1 op/backend-aware 连续代价曲线

更新时间：2026-08-06

状态：修正后的正式采集、分析和收敛验证已完成；本目录是 Phase 14F 的唯一正式结果。

## 1. 目标

Phase 14C–14E 表明，仅用 PatternDemand、TP、phase 或 active-batch 摘要回归通信
时间仍存在较大 Decode 和模型留出误差。Phase 14F 测量与推理路径对齐的单次 L1
通信代价：

\[
C_{L1}(raw\_op,payload,TP,backend\_proxy),
\]

再与 Phase 14C 的精确 `(raw_op,payload)` PatternDemand 直方图组合：

\[
T_{struct}=\sum_{op,m,p}count(op,m,p)C_{L1}(op,m,p).
\]

预测器不读取推理完成后观察到的 backend 或 kernel name。每个外层训练折只为
Prefill/Decode 分别拟合一个非负乘性校准。

## 2. 时间契约

本目录使用与 Phase 14C 主标签一致的 all-rank post-rendezvous 口径。每次微基准
调用使用原始 Kineto 绝对时间戳，计算：

```text
max(rank kernel end) - max(rank kernel start)
```

旧目录 `experiment-results/phase14f/` 使用了 max rank kernel duration，时间契约
不一致，结果无效，不能作为论文或后续实验输入。

## 3. 实验规模

| 项目 | 结果 |
|---|---:|
| 曲线单元 | 30/30 |
| raw_op | `all_reduce`、`fused_allreduce_residual_rmsnorm` |
| TP | 2、4、8 |
| 精确 `(raw_op,payload,TP)` 支撑点 | 105 |
| 每个支撑点独立重复 | 5 |
| 每次重复测量调用 | 100 |
| 曲线记录 | 525 |
| 调用样本 | 52,500 |
| retry / failure | 0 / 0 |
| 最大 repeat median CV | 5.7835% |

曲线覆盖当前 Phase 14C 三模型数据集中实际出现的全部精确支撑点。硬件拓扑、CUDA、
NCCL、GPU 和环境信息保存在 `environment.json` 与 `nvidia_topology.txt`。

## 4. 正式结果

| 方法 | Overall MAPE | P95 APE | Prefill MAPE | Decode MAPE |
|---|---:|---:|---:|---:|
| Phase 2 payload-only scaled | 24.6869% | 83.1879% | 22.5641% | 28.9325% |
| Phase 14D TP-conditioned PatternDemand | 12.1499% | 39.0134% | 9.4335% | 17.5828% |
| Phase 14F op/backend-aware structural | **4.4251%** | **10.7625%** | **4.3644%** | **4.5465%** |

Phase 14F 的整体 `R²=0.996723`，Decode `R²=0.989295`。

整模型留出：

| 留出模型 | MAPE | P95 APE |
|---|---:|---:|
| DeepSeek-V2-Lite | 5.9402% | 12.8342% |
| Qwen3-30B-A3B | 3.2836% | 7.2406% |
| Qwen3-8B | 4.2256% | 9.8078% |

预设四项门槛全部通过：

- overall MAPE <10%；
- overall P95 APE <25%；
- Decode MAPE <10%；
- 每个 leave-one-model-out MAPE <15%。

## 5. 结论边界

本阶段证明：在单节点 B200 L1、三个模型、TP2/4/8 和当前精确 payload/backend
支撑范围内，实测 PatternDemand 与对应连续代价曲线可以准确组合通信时间。

本阶段尚未证明：

- 由历史流量画像预测出的 PatternDemand 仍能保持同等时间精度；
- 未见 payload 或算法切换边界可以稳定插值；
- L2/L3、PP、PD 或 EP All-to-All 具有相同误差；
- DNN residual 是必要组成；
- 调度器 placement 已实现端到端收益。

## 6. 正式产物

```text
experiment-results/phase14f_post_rendezvous/
├── README.md
├── DONE
├── environment.json
├── nvidia_topology.txt
├── support_inventory.json
├── runner.log
├── curve/tp{2,4,8}/{op}/r{0..4}/
├── analysis/
│   ├── README.md
│   ├── summary.json
│   ├── curve_summary.csv
│   ├── metrics.csv
│   └── predictions.csv
├── audit_summary.json
└── manifest.sha256
```

使用以下命令重新生成审计和根 manifest：

```bash
python scripts/finalize_phase14f_results.py \
  --result-root experiment-results/phase14f_post_rendezvous
```

