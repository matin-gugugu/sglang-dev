# Phase39：TP/PP L1–L3物理曲线库与placement验证

Phase39是一个完整阶段，不再把“测量、卷积、决策分析”拆成多个Phase。它先在任何GPU benchmark之前冻结L1/L2/L3 placement，再完成24个分布式measurement shard：12个必测case，每个case两套独立placement replica。GPU释放后，CPU端一次性建立12条物理曲线、代入Phase34冻结TP/PP直方图，并验证communication-only placement排序与regret。

## 不能改变的研究语义

- TP/PP size和执行policy是输入；Phase39只比较L1/L2/L3 placement。
- TP使用SGLang `GroupCoordinator.all_reduce`实际动态dispatch；PP使用与Phase37相同的NCCL `isend/irecv` tensor primitive。
- 不加载checkpoint、不运行模型、不重新生成预测、不训练。
- Phase34D target已经打开，所以cost和placement分析是重复工程证据，不是新盲测。
- raw逐次样本、per-rank样本、模型权重、缓存和PID禁止进入Git。

## 必测矩阵

| primitive | L1 | L2 | L3 |
|---|---:|---:|---:|
| PP P2P | 2 replicas | 2 replicas | 2 replicas |
| TP=2 all-reduce | 2 replicas | 2 replicas | 2 replicas |
| TP=4 all-reduce | 2 replicas | 2 replicas | 2 replicas |
| TP=8 all-reduce | 2 replicas | 2 replicas | 2 replicas |

L1必须是单节点；L2必须是同rack的两个节点且rank均分；L3必须是跨rack的两个节点且rank均分。分类依据必须来自测量前已有的allocation/rack/fabric元数据，禁止按最终速度事后命名。

## 执行结构

1. 将`topology_inventory.example.json`复制到Git仓库外，填入真实host、rack、NIC、GPU和分类证据。
2. 用`make_topology_plan.py`展开为24个固定measurement shard；生成后的plan SHA就是本轮环境合同。
3. 在raw目录为空时运行`preflight.py`，冻结W39、R38、输入SHA、生产源语义和plan。
4. 对每个measurement和repeat，用`render_launch_commands.py`生成逐节点`torchrun`参数；由目标环境已有的多节点调度/远程启动机制并发执行返回的命令数组。
5. 先完成repeat 0–4。运行`raw_status.py`；只对报告为`needs_extra_repeats`的measurement按2个repeat一轮追加到7或9。
6. `run.py`验证raw完整性、生成正式紧凑结果并执行CPU卷积；`verify.py`独立验收。

多节点启动方式属于环境适配层。Agent可以使用目标环境已有的scheduler、MPI封装或受控SSH，但不得改变world size、rank mapping、primitive、payload网格、方向策略或重复数。

## 曲线与决策

每个placement replica先跨repeat取中位数。正式curve knot再取两套replica中较慢的值，禁止挑选更快placement。结果同时保存replica envelope和方差，用于判断placement决策是否对测量不确定性稳定。

对每个固定`profile × model × TP/PP × size × policy`，CPU端输出：

- L1/L2/L3 predicted和teacher通信cost；
- MAPE、WAPE、signed bias及Phase35 proxy差值；
- predicted/teacher placement排名；
- top-1 agreement、top-2 coverage、teacher regret、Spearman相关性、决策margin；
- 在跨replica latency envelope下选择是否稳定。

这仍不是完整线上scheduler：计算、显存、资源可用性、排队、metadata与通信计算重叠均不包含。

## 结果与Git

正式目录固定为`experiment-results/phase39_tp_pp_l1_l3_physical_placement_validation/`。成功后只允许：

```bash
git add -- experiment-results/phase39_tp_pp_l1_l3_physical_placement_validation/
git add -f -- \
  experiment-results/phase39_tp_pp_l1_l3_physical_placement_validation/analysis/cost_metrics.csv \
  experiment-results/phase39_tp_pp_l1_l3_physical_placement_validation/analysis/frozen_histogram_metrics.csv \
  experiment-results/phase39_tp_pp_l1_l3_physical_placement_validation/analysis/physical_vs_phase35.csv \
  experiment-results/phase39_tp_pp_l1_l3_physical_placement_validation/analysis/placement_decision_metrics.csv \
  experiment-results/phase39_tp_pp_l1_l3_physical_placement_validation/logs/runtime.log
python3 workflows/patterndemand/verify_staging.py --phase phase39
```

这5个文件是紧凑正式结果，但会被仓库全局ignore规则命中，因此只对这些明确路径使用`-f`。禁止`git add .`或对整个结果目录使用`git add -f`。`verify_staging.py`必须证明结果树和暂存区文件集合完全一致。R39必须以W39为唯一父提交。
