# Phase 33B：扩充后的开发Hfull数据

本阶段只为Phase33A冻结的45个新增开发窗口生成Hfull teacher，并与Phase31的49个开发窗口合并。最终开发侧为75个训练、19个验证，共94个请求级互斥画像；完整teacher请求从21,058个增加到35,524个。

TP与PP各有5,076条phase训练样本，覆盖三个已知模型、三种并行规模、三种policy和prefill/decode。TP继续使用GPU验证过的full-window结构公式，PP继续使用scheduler-faithful teacher。完整请求列表只在构建内存中使用，没有保存或进入特征。

9个Phase33盲确认窗口仍只有低维特征与compact H0，Hfull target尚未生成；其example id与本阶段全部target互斥。
