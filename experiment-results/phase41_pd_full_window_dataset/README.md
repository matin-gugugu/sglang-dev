# Phase41：纯PD完整窗口teacher与开发数据

最终状态：`PASS`。本阶段先冻结`最多64请求/wave`的有界fixed-draining协议：窗口内保持原始顺序，wave内原子放行，前一wave完全返回后才提交后一wave。

GPU sentinel覆盖`4853`个请求、`82`个wave，包括63/64/65/129边界和三个真实完整窗口。真实sender记录与CPU teacher逐请求精确匹配`4853/4853`，calls、logical bytes和12-bin直方图误差均为0。

GPU门通过后，才生成94个Qwen3-8B开发画像的Hfull标签、32请求H0和逐bin residual，共使用`35524`个完整teacher请求。另冻结12个全新盲测画像的低维feature与H0；盲测完整请求没有进入跨环境bundle，Hfull target行数严格为0。

本阶段没有训练DNN、没有加载checkpoint、没有测试其他五个模型，也没有测物理RDMA时间或做placement。GPU profiler JSONL、完整server日志和包含开发完整请求的transfer bundle均保存在Git外。
