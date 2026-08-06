# Phase 16E：ProfileDemand GPU 回放计划

每个服务画像从 128 个固定代表请求中再分层选择 32 个请求。三种执行
策略均回放同一组请求，只改变 `max_batch_size` 和 `max_prefill_tokens`，按请求原顺序形成
draining microbatches。完整计划包含 24 个画像、
545 个 GPU workloads；smoke 使用 3 个画像和 3 次重复，共
201 个 workloads。

为保证 Qwen3-8B 的 Prefill logical payload 不超过已实测 L1 曲线 512 MiB，所有策略的
单 batch token budget 不超过 65,536。当前画像网格不与 heterogeneous chunked prefill
做全交叉；chunk 机理使用 Phase14C 的 108 个受控配置，避免把 one_batch 尚不支持的
异长请求 chunk 回放伪装成真实 online 调度。

到达 offset 被保留审计，但当前仍是离线 draining microbatch，不声称已完成 online
continuous batching。
