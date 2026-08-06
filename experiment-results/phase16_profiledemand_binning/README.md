# Phase 16A：ProfileDemand payload 分桶选择

固定范围为 4 KiB–512 MiB。`center` 使用桶几何中心在实测 L1 曲线上的代价；
`mean_payload` 在每桶同时保留 calls 与 logical bytes，并以 `bytes/calls` 查询曲线；
`oracle` 使用每个 `op×TP×bin` 内按 calls 加权的最优常数，是“每桶只能用一个固定
代表代价”时的乐观参照，不使用 workload 时间标签。`calls+bytes` 多保留了一阶矩，
因此可以优于这个固定常数参照。

| bins | center MAPE | center P95 | calls+bytes MAPE | calls+bytes P95 | oracle MAPE | oracle P95 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 17.847% | 73.970% | 0.322% | 2.755% | 11.433% | 42.928% |
| 12 | 10.444% | 33.847% | 0.064% | 0.700% | 8.917% | 37.764% |
| 16 | 14.749% | 34.432% | 0.057% | 0.279% | 0.546% | 3.004% |
| 24 | 6.359% | 14.986% | 0.001% | 0.000% | 0.082% | 0.340% |

按“整体和 Prefill/Decode 均 MAPE <2%、P95 <5%”选择的结果：**12**。

`workload_time_metrics.csv` 另报告各编码乘 L1 曲线并经 workload-CV phase scale 后，
相对真实 all-rank post-rendezvous 标签的误差；它用于观察分桶误差是否改变已有 4.43%
闭环，不用于定义分桶 oracle。
