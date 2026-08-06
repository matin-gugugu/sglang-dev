# Phase 16D：ProfileDemand 基础结构公式 H0 验证

H0 仅使用模型静态结构、实际工作负载长度和 chunk 规则：

- 每个 TP transformer forward 的 canonical logical AllReduce 次数为 `2L+1`；
- 单次 logical payload 为 `active_tokens×hidden_size×dtype_bytes`；
- Prefill 的 active tokens 由 batch 和 chunk 决定；
- Decode 使用 `A(t)=Σ_i 1(M_i>t)`，其中 `M_i` 是实际生成长度，且
  `t=1,...,max(M)-1`；第一个输出 token 由 Prefill forward 采样，不产生额外 Decode
  forward。

在三个模型、TP2/4/8、Prefill/Decode 共 162 个聚合配置上，canonical 精确匹配
为 162/162，calls WAPE、logical-bytes WAPE 与平均
histogram L1 均为 0；同一 workload 的 logical histogram 跨 TP 完全不变。

这不意味着正式 ProfileDemand v1 不需要学习：本验证向 H0 提供了每个 batch 的完整
实际长度。正式模型只看到低维服务画像和执行策略，DNN residual 负责修正画像分桶内
形态、batch 形成和实现边界。Qwen3-30B-A3B 的 fused raw-op 拆分是当前 backend 的
实现细节，未泄漏进 canonical H0；它由第二阶段 backend-aware 映射处理。
