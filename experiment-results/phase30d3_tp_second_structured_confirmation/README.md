# Phase 30D3：TP结构事件模型第二独立确认与最终映射

状态：**PASS**。本阶段只把Phase30C预先冻结的第二确认预测连接到Phase30D2真值；
预测SHA、D1映射SHA和D2 target SHA均先核验通过。没有重训、调参或改写任何预测。

## 第二确认结果

| 方法 | calls MAPE/WAPE | bytes MAPE/WAPE | TV | norm EMD | cost MAPE/WAPE |
|---|---:|---:|---:|---:|---:|
| h0 | 12.16% / 11.97% | 2.97% / 1.29% | 0.1345 | 0.0139 | 7.13% / 5.73% |
| phase29_enhanced_bounded_residual_diagnostic | 9.67% / 8.67% | 2.92% / 1.43% | 0.1534 | 0.0164 | 6.22% / 4.38% |
| structured_event_bounded_residual | 23.63% / 17.98% | 33.44% / 14.23% | 0.1419 | 0.0162 | 19.31% / 10.59% |
| structured_event_direct_control | 14.00% / 14.80% | 49.19% / 19.25% | 0.0946 | 0.0104 | 22.60% / 15.86% |

当前Phase30最终受保护映射为：

- latency：`h0`
- balanced：`h0`
- throughput：`h0`

这一结论表示当前91维画像→62维事件的有界残差DNN未通过开发验证和两级确认协议，部署候选应回退
到H0。它不表示研究设计取消DNN；相反，后续若继续TP DNN，应重新设计事件目标、loss或增加独立
训练窗口，并使用全新封闭确认集，不能再用Phase30两批确认结果调参。

共同参考cost仍只是同一连续代价曲线下的比较量，不是测得的真实placement/topology时延。
