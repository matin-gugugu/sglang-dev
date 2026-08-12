# Phase 31E：TP最后一轮加权H0+DNN residual

本阶段执行今晚参考文档允许的最后6个TP配置，使TP累计搜索量达到18组上限。新增配置只改变总calls/bytes损失权重与共享、policy、model、model×policy小头；网络仍为64×64有界DNN residual，最终形式仍是`H0 + DNN residual`。

训练和选型只读取Phase31B的39个训练画像、10个验证画像及不含target的固定预测特征，没有读取Phase31D Hfull target。选中来源为`phase31c_incumbent`，固定预测SHA为`fa691f2a140d9046942463d79a3bf7e67f426ea507740a5a19d6cbd73036e684`。

需要公开的证据限制：Phase31D第一轮固定评测已经完成，因此后续在同一固定集上的结果属于重复评测，不是全新的独立确认；这不构成target进入训练或模型选择，但结论必须带此限制，且不得更换固定窗口或降低阈值。
