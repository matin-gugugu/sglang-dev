# Phase52 workflow：纯PD物理通信代价与placement验证

Phase50已经冻结并打开六模型300个fresh-blind画像的H0、H0+DNN和Hfull 12-bin消息直方图；Phase51已经实测六模型L1/L2/L3 Mooncake/RDMA物理曲线。Phase52只把这两层确定性连接起来：对每个非空bin使用`平均消息字节=bin总字节/bin调用数`，在对应模型/拓扑曲线上插值得到单次延迟，再乘调用数并对12个bin求和。

本阶段有1800个`画像×模型`单元、两种预测方法和三个候选拓扑，共10800行物理cost；每种方法再为每个单元从L1/L2/L3中选择预测通信代价最低者，与Hfull通信代价最低的oracle比较agreement和teacher regret。P/D内部始终是P1/D1，调度器不选择TP/PP/P/D数量，也不考虑计算、显存、空闲、排队、拥塞或重叠。

正式cost使用Phase51未经平滑的保守曲线，即双向较慢值和两个冻结placement较慢值。两个placement较快值形成lower bound，用于判断候选区间是否真正可分；另做累计最大单调包络诊断，检查局部曲线下降是否改变选择，但绝不替代正式结果。

Phase52不是新的盲测：Phase50标签已经打开。它是冻结预测、标签和物理曲线之后的repeated-engineering确定性重算，不允许根据结果重训、调参或改变曲线。

## 本地执行

R51必须已经ff-only合入，W52必须唯一父提交为R51。在不含`data/`或其他未跟踪资产的隔离run分支/worktree执行：

```bash
P52=workflows/patterndemand/phase52_pd_physical_cost_placement_validation
python3 "$P52/preflight.py" --expected-workflow-commit W52
python3 "$P52/run.py" --expected-workflow-commit W52
python3 "$P52/verify.py"
```

PASS后只选择性添加`experiment-results/phase52_pd_physical_cost_placement_validation/`，运行`verify_staging.py --phase phase52`。R52必须是唯一父提交为W52的单一result commit。禁止`git add .`。

正式结果可以回答：H0+DNN是否降低物理通信cost误差，能否更准确地选择communication-only L1/L2/L3，以及结论对Phase51 placement方差和单调包络是否敏感。它不能回答完整调度器或端到端服务性能。
