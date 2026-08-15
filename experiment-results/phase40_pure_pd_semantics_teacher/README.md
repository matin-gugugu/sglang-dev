# PHASE40：执行阻塞

本次没有生成正式实验结果。阻塞原因：容器预编译的sgl-model-gateway router把/generate的rid反序列化为String，拒绝合同要求的batch rid数组(422)；合同pin的pd_router.rs/pd_types.rs是仓库源码、从未编译，source_semantics_audit仅做文本匹配无法察觉二进制与pin源码不一致。preflight全过(含Qwen3-8B 7项SHA256、离线开关、P/D/router健康启动)，首个正式请求即被拒，未产生任何正式raw
