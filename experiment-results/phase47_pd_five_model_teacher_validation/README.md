# Phase47：纯PD其余五模型teacher语义GPU验证

最终状态：`PASS`。五个模型顺序复用同一对GPU，共`225`个请求，逐请求精确匹配`225`个；calls、logical bytes与12-bin直方图误差均为0。DeepSeek-V2-Lite固定为TRTLLM MLA/page64，其余四模型固定为FlashInfer/page1，均使用Mooncake/RDMA和P1-D1。

本阶段不训练DNN、不测通信时间、不做placement。模型权重、HF凭证、profiler JSONL和完整服务日志均保存在Git外。结合Phase40/41，现已为冻结六模型阵容建立代表性的纯PD Hfull teacher GPU语义证据；下一阶段才生成六模型完整窗口数据并训练。
