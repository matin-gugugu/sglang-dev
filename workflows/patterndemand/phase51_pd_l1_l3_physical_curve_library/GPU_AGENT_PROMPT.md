# Phase51 GPU Agent任务

你是远程GPU执行Agent。先完整阅读本目录`README_CN.md`、`experiment.json`、`topology_inventory.example.json`和所有Python脚本；核验HEAD严格等于控制端给出的W51，从W51创建唯一run分支，不改变正式分支。

本任务不下载或加载模型、不推理、不训练。使用`lmsysorg/sglang:v0.5.15`和当前仓库`python/`，调用生产`MooncakeTransferEngine.batch_transfer_sync`，完成6模型×L1/L2/L3×2个冻结placement的36个双rank GPU shard。模型只提供KV page大小和真实batch描述符布局。

先根据调度/资产元数据填写Git外topology inventory并在首条raw前冻结plan。L1同机双GPU但仍走Mooncake RDMA；L2同rack跨机；L3跨rack同RDMA fabric。不能用benchmark速度给拓扑分类。执行前确保两个replica是不同endpoint组合，RDMA/dma-buf可用，且没有TCP/MNNVL/intranode-NVLink/staging/custom-pool回退。

严格按README：preflight；所有36 shard的repeat 0–4；`raw_status.py`；只按其`needs_extra`追加5–6、必要时7–8；`complete=true`后run和verify。双机torchrun命令必须并发启动。允许合理诊断端口、hostname、HCA、调度器和进程退出；失败重试用新Git外attempt并保留证据。plan冻结后换endpoint会使整个attempt失效，不能混用raw。

PASS后只选择性暂存Phase51正式结果，运行`verify_staging.py --phase phase51`，创建唯一父提交为W51的R51并push run分支。禁止`git add .`；raw JSONL、模型、缓存、PID、token和完整调试dump不得进Git。

回传必须报告：实际状态及证据、W51/R51/run分支；容器/driver/torch/CUDA/GPU；6个placement的主机/rack/GPU/HCA/网络域；Git外plan/preflight/raw位置及哈希；raw文件/record/repeat数量；18条曲线/396 knots、运行方差和跨placement方差；README/summary/logs/DONE/manifest；可以和不可以得出的结论；下一步由控制端验收R51后再生成/执行Phase52。
