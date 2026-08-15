# PHASE40：执行阻塞

本次没有生成正式实验结果。阻塞原因：冻结的page_size_tokens=1被运行时静默改为64：run.py传了--page-size 1但未pin --attention-backend，B200上sglang默认选trtllm_mha，该后端只支持16/32/64故无条件覆盖，导致kv_bytes_per_page=9437184而非合同要求的147456，run.py自带的page_size_one检查必然失败；同时Mooncake KV传输失败(P/D互指对方dead，但PD warmup成功)。rid修复已生效，请求已抵达prefill并按冻结格式展开
