# Phase38：PP物理曲线确定性cost重算

Phase38在Phase37正式结果通过控制端验收并以ff-only方式合入后运行。它不使用GPU、不加载checkpoint、不重新推理、不训练；只把Phase34已经冻结的六模型PP `H0 + DNN residual` 12-bin直方图与已经打开的Hfull target，逐bin代入Phase37物理P2P连续曲线。

## 输入

- Phase34C PP冻结预测：只取`prediction_set=phase34_blind_new`、`method=h0_plus_dnn_residual`、`parallelism=pp`，应为1,296条phase记录。
- Phase34D六模型Hfull target：只取PP，应为1,296条phase记录。
- Phase34D正式直方图指标：用于证明本阶段没有改变calls、bytes、TV和EMD。
- Phase37物理曲线：在`W38`运行时校验其result commit、manifest、状态、物理证据标签、payload网格、测量口径和SHA，再复制为结果内快照。
- Phase35 PP L1 proxy指标：只用于并排比较，不会混入物理曲线或改名为物理结果。

Phase34静态输入SHA已写入`experiment.json`。Phase37结果提交和曲线SHA不在workflow合同中重复硬编码；它们必须由R37合入后的W38 preflight从正式Git历史动态验收并冻结到`audit/input_freeze.json`。

## 计算

每个非空bin先算平均payload：

```text
payload_bytes = logical_bytes / logical_calls
```

然后在Phase37未平滑knots上对`log2(payload_bytes)`做分段线性插值，超出物理测量范围时固定使用首/末knot；逐bin cost为`calls × latency`，phase cost为12个bin之和，total cost为prefill与decode之和。

正式输出包含每条物理曲线的phase/total逐case cost、overall/逐模型/逐policy的MAPE、WAPE和signed bias，以及和Phase35单机PP proxy的同slice差值。若Phase37只覆盖一个拓扑类别，Phase38不会虚构跨placement排序。

## 结论边界

`PASS`只表示输入、曲线、重算和归档合同全部通过，不表示物理cost WAPE必然低于5%。5%只作为诊断参考线。由于Phase34D target已经打开，本阶段是重复工程证据，不是新的盲测。

若物理曲线下的误差提示需要重训，Phase38只能提出后续开发信号；不得直接使用Phase34D target重训后再把结果声称为盲测。后续必须另建开发集和确认协议。

## Git时序

本workflow只能在Phase36/37结果依次验收合入后进入正式分支。正确时序是：

```text
W36=72c81b53 -> R36 ff-only合入 -> W37 -> R37 ff-only合入
  -> 提交本Phase38 workflow形成W38 -> 从W38运行 -> 单一结果提交R38
```

若提前把Phase38 workflow提交到正式分支，R36将不再能以`W36`为唯一父提交直接ff-only合入；因此W38的父提交必须是已验收的R37。
