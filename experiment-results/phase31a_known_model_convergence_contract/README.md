# Phase 31A：三模型 TP/PP 收敛数据合同

本阶段只冻结今晚第一阶段收敛实验的数据和方法边界，不生成 Hfull 标签、不训练模型、也不读取预测结果。

## 数据范围

- 三个已知模型：DeepSeek-V2-Lite、Qwen3-8B、Qwen3-30B-A3B；三者都会进入训练、验证和固定预测，不做整模型留出；
- 共 59 个历史画像：39 个训练、10 个验证、10 个固定预测；
- BurstGPT 三段各 18 个；Mooncake 共 5 个剩余独立块；
- 所有画像来自历史侧低维统计的正常中心区域，使用robust-distance medoid选择，不按target或误差选样本。

## 隔离修复

旧实验曾把300秒Mooncake窗口按60秒步长视为不同画像，造成不同window id共享请求。本合同使用300秒时间区间作为硬隔离单位：新训练、验证、固定预测之间共享请求为0；Phase27、Phase28和Phase30的历史确认区间也全部设置300秒embargo。

Mooncake原始trace较短，严格embargo后只剩5个独立块，因此固定预测中只有1个Mooncake conversation画像。它满足第一阶段的正常流量与防泄漏要求，但不能支持强Mooncake泛化结论。

## 不改变的研究口径

Hfull只作离线teacher；预测器输入仍为低维历史画像、模型结构、固定TP/PP配置和策略；模型形式必须是`H0 + DNN residual`。fixed-draining、每1000请求归一化、12桶calls/bytes和统一参考cost定义均保持不变。
