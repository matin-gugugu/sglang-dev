# PHASE47：执行阻塞

本次没有生成正式实验结果。阻塞原因：DeepSeek-V2-Lite在page64下的prefill预算记账与冻结teacher不一致：合同预期第4个请求切成[0,17]/[17,32]，GPU确定性地给出[0,16]/[16,32]（两次repeat完全一致，非竞态）。冻结teacher按原始token扣预算(budget-=send_tokens)，而实测符合按页对齐扣预算(3×ceil(1000/64)×64=3072，余1024=16页)。page=1的其余四模型两套规则等价，故Phase40/41从未暴露。修正需改动pinned的phase40 contracts.py或冻结的科学预期，均越红线。本轮已确认bf16修正正确、trtllm_mla在B200上可执行、Mooncake RDMA KV通路正常、DeepSeek MLA字节公式实测成立
