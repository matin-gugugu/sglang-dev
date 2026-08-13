# Phase 34B：六模型Hfull开发数据

本阶段保持Phase33的94个开发画像不变：75个训练、19个验证，覆盖90个BurstGPT与4个Mooncake窗口，共35,524个唯一完整teacher请求。Phase33三个模型的已冻结数据逐字复用；只为三个新增模型生成Hfull标签，再合并为六模型数据。

TP和PP各有10,152条phase训练样本，每个模型各1,692条，覆盖三种并行规模、三种policy与prefill/decode。TP使用Phase26A验证过的fixed-draining结构公式；PP使用Phase25B/25C验证过的scheduler-faithful事件模拟器。完整请求列表只在构建内存中使用，没有保存或进入特征。

六模型低维bytes均值结构锚点与Hfull teacher逐条一致，最大相对误差为`1.225e-15`，因此后续TP/PP都允许继续使用该锚点并保留H0的12-bin bytes形状。Phase34A的12个全新确认画像仍只有低维feature/H0，Hfull target尚未生成。
