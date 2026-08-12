# Phase 25 GPU smoke：完整窗口 teacher 审计

42 个真实请求的完整 fixed-draining 窗口已完成 TP2 和 9 个 PP cell 的 GPU 审计。TP 精确通过；PP 3/9 个 cell 与静态 teacher 精确一致。所有 PP cell 的采集完整性和 sender boundary 一致性通过。

| PP policy | 精确cell | calls WAPE | bytes WAPE | hist TV | norm EMD | cost MAPE |
|---|---:|---:|---:|---:|---:|---:|
| MB1 | 3/3 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.00% |
| MB4 | 0/3 | 43.67% | 0.00% | 0.4773 | 0.0362 | 32.03% |
| MB16 | 0/3 | 96.60% | 0.00% | 0.6248 | 0.0663 | 65.08% |

MB>1 的 logical bytes 守恒但 calls/直方图不一致，说明误差来自真实 scheduler 的离散 microbatch 拆分/合并，而不是请求规模抽样。当前 provisional PP 标签不能晋升为训练真值；下一步应恢复 fixed-draining scheduler 语义或生成 GPU/full-scheduler teacher。
