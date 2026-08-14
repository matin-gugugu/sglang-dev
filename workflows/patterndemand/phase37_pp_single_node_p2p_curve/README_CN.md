# Phase37 workflow说明

Phase37测量PP单机GPU P2P的连续通信曲线，不启动模型服务、不下载权重、不训练预测器。

正式计时对象是SGLang生产路径`GroupCoordinator.send_tensor_dict(async_send=True)`中每个GPU tensor对应的NCCL异步P2P原语。Phase24/25的消息直方图只统计sender侧tensor logical message，所以正式曲线同样只计tensor，不把每个tensor_dict的CPU metadata、tensor分配和scheduler时间摊进每条消息。

payload网格覆盖Phase34 PP Hfull已打开数据中的约4KiB至39.5MiB真实范围，并加入2次幂与64MiB边界。默认每个实际存在的单机拓扑类别选择编号最小GPU对并测量两个方向；正式拓扑类别曲线对每次repeat取双向中位数的较大值，禁止挑选较快方向。每方向30次warmup、100次计时、5次独立进程重复。若repeat中位数CV超过15%，自动每轮追加2次，绝对上限9次。

Agent的自主空间：

- `AUTO`：有限重试、高方差追加重复、环境信息采集。
- `RECORD_AND_CONTINUE`：因占用而换同类GPU对，必须写理由。
- `BLOCKED`：生产源SHA不符、P2P/NCCL不可用、少于两张GPU、必须改变测量语义。

曲线保留真实非单调点，不做平滑。Phase37的PASS表示测量合同和资产完整性通过，不代表PatternDemand cost误差达标；Phase38才进行冻结直方图卷积。
