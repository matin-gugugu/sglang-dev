# 给GPU环境Agent的Phase37提示词

请先完整阅读`workflows/patterndemand/PatternDemand跨环境GPU执行交接_Phase36_Phase37.md`、本目录`experiment.json`和`README_CN.md`。从交接方给出的workflow commit `W`创建独立run分支。先确认至少两张空闲CUDA GPU，并在运行总控前执行`unset CUDA_VISIBLE_DEVICES`。

外置raw目录必须在Git仓库之外，例如：

```bash
RAW_DIR=/local_nvme/patterndemand_raw/phase37_W
python3 workflows/patterndemand/phase37_pp_single_node_p2p_curve/run.py \
  --expected-workflow-commit W \
  --raw-dir "$RAW_DIR"
python3 workflows/patterndemand/phase37_pp_single_node_p2p_curve/verify.py
```

Agent可以在同一拓扑类别内换成空闲GPU对，但必须提供JSON映射和`--override-reason`；NVLink类别会保留链路宽度，例如`NVLINK_NV18`。可以诊断NCCL、有限重试、保留异常并按合同自动补测高方差点；不能换后端、删除payload点、降低重复次数、只挑较快方向、平滑异常或把CPU metadata混入正式tensor-only曲线。

成功后只能选择性添加`experiment-results/phase37_pp_single_node_p2p_curve/`，不得使用`git add .`。raw逐次样本、模型权重、缓存和PID不得提交。result commit必须以W为唯一父提交，push run分支并回传分支、commit R、GPU拓扑、状态、曲线数、raw bundle id、目录大小、文件数和manifest结果。
