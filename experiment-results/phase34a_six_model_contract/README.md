# Phase 34A：六模型配置与全新确认数据合同

本阶段冻结六模型集合：保留deepseek-v2-lite、qwen3-8b、qwen3-30b-a3b，新增llama-3.2-3b-instruct、qwen2.5-14b-instruct和mixtral-8x7b-instruct-v0.1。新增模型覆盖小型dense、大hidden dense和少专家top-2 MoE；只固化配置，没有下载权重或运行GPU profiling。

在Phase27/28/30/31/32/33所有已使用窗口的300秒embargo之外，从三个BurstGPT分段各冻结4个P95正常中心medoid，共12个请求级互斥的新确认画像、3,803个未来teacher请求。TP和PP各生成1,296条六模型低维feature/H0记录，不含target或完整请求列表。

Phase33三模型结果与manifest保持不变。下一阶段可在固定94个开发画像上生成六模型Hfull开发标签；必须先完成六模型训练、预测、checkpoint和SHA归档，才能一次性打开本批确认target。
