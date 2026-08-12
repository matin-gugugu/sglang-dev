# Phase 31B：三模型 TP/PP Hfull 开发数据

本阶段使用Phase31A冻结的59个请求级不重叠画像。39个训练和10个验证画像生成Hfull teacher；10个固定预测画像只生成低维特征和compact32 H0，尚未生成Hfull真值。

## 数据规模

- 三个模型：DeepSeek-V2-Lite、Qwen3-8B、Qwen3-30B-A3B；
- TP：TP2/4/8 × latency/balanced/throughput × prefill/decode；
- PP：PP2/4/8 × MB1/4/16 × prefill/decode；
- 开发Hfull标签：5,292条phase rows，TP/PP各2,646条；
- 固定预测特征：TP/PP各540条，不含任何`target_`字段；
- 完整请求列表只在构建进程内存中用于低维画像聚合和teacher，不落盘、不进入Git。

## teacher口径

TP继续使用Phase26A GPU哨兵验证过的fixed-draining结构公式。PP先由Phase25B/25C验证过的scheduler-faithful模拟器生成模型无关active-token事件，再按三个模型的hidden-size/dtype确定性映射payload；这属于当前源码与调度合同内的结构teacher，不声称三个模型的全部PP配置均已逐项GPU实测。

下一步只允许读取开发数据训练`H0 + DNN residual`，先冻结固定预测文件和SHA，再生成固定预测Hfull真值。
