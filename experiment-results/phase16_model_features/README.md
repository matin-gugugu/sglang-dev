# Phase 16C：ProfileDemand 模型结构特征

这些特征只从模型 `config.json` 静态提取，不需要运行模型。正式数值特征不使用
`model_id`、实测直方图或真实通信时间；architecture/model_type 仅供审计和 model-ID
baseline。首版 canonical PatternDemand 统一为逻辑 AllReduce；当前 SGLang 的 fused
raw-op 拆分保留为 audit-only 模板，由第二阶段 backend 细化处理。

`logical_collectives_per_forward_prior=2L+1`、
`payload_bytes_per_active_token_prior=hidden_size×dtype_bytes` 是 H0 的透明结构先验，
不是从时间标签拟合得到的特征。
