# 实验结构总导引：Phase34 状态补充

本文件只补充 Phase34 状态，不修改 fixed-draining、Hfull teacher 和“并行配置为预测输入、调度器选择 placement/topology”的基础定义。

Phase34 把三个模型扩展为六个模型。开发数据仍是 94 个画像、35,524 个完整 teacher 请求；完整请求只离线生成标签，预测器输入仍是低维画像、模型结构、固定 TP/PP 配置与策略。

TP 和 PP 均重新训练六模型 `H0 + DNN residual`。六模型新盲测为 12 个全新 BurstGPT 窗口、3,803 个请求：TP calls WAPE 7.81%、TV 0.1903、EMD 0.0201、cost WAPE 4.35%；PP calls WAPE 4.53%、TV 0.1287、EMD 0.0171、cost WAPE 3.29%，两者正式通过。

六个模型全部进入训练和验证，因此不能宣称未见模型泛化。TP throughput、PP MB16 仍是薄弱切片；bytes 近零来自结构锚点。Phase34 已停止调参，下一步应接入 placement/topology 连续通信代价曲线。
