# PHASE47：执行阻塞

本次没有生成正式实验结果。阻塞原因：W47-fix2 重pin的trtllm_mla后端与合同冻结的--kv-cache-dtype bfloat16字符串不兼容：sglang的trtllm_mla校验白名单(overrides.py:1599)只收bf16/auto/fp8_e4m3/fp4_e2m1，漏了它自己文档化的同义词bfloat16，P/D两端均在ServerArgs构造阶段即失败。这是sglang内部的字符串词表缺陷而非硬件限制(model_runner.py:2470中bf16与bfloat16解析到同一个torch.bfloat16)，但修正需改动冻结的run.py/verify.py或pinned的overrides.py，均越红线
