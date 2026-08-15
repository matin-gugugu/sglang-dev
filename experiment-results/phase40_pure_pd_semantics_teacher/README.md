# PHASE40：执行阻塞

本次没有生成正式实验结果。阻塞原因：Mooncake RDMA无法注册显存：P/D两侧ibv_reg_mr均返回EFAULT(Failed to register memory ... Bad address [14])，容器内nvidia_peermem/nv_peer_mem缺失即GPUDirect peer-memory不可用(ulimit -l unlimited、IB设备与端口均正常，排除锁页与fabric)。合同新增的compatibility smoke门按设计在正式raw写入前拦截，正式raw保持0文件。上一轮的page_size被trtllm_mha覆盖问题已由--attention-backend flashinfer修复并经smoke raw证实(page_size_tokens=1, kv_bytes_per_page=147456, kv_page_count=64)
