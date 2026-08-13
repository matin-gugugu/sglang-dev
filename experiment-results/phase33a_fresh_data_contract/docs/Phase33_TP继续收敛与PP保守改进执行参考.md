# Phase 33：TP继续收敛与PP保守改进执行参考

> 日期：2026-08-13
> 性质：Phase33有限实验执行补充
> 优先级：本文不修改、不替代实验总导引中的研究目标、Hfull teacher、fixed-draining语义、输入输出合同、归一化方式或调度器边界。

## 1. 不变的基础合同

- 输入仍为低维历史流量画像、模型结构、固定执行策略和已经确定的TP/PP配置；并行配置是预测输入，调度器只选择placement/topology。
- 完整请求列表只用于离线生成full-window fixed-draining Hfull teacher，不能进入预测特征或部署输入。
- TP与PP都必须保持`H0结构先验 + DNN residual`；H0是baseline和安全锚点，不能用全H0冒充DNN收敛。
- 输出仍为每1000请求的拓扑无关消息直方图，并代入同一连续通信代价曲线评测cost。
- 三个已知模型都进入开发和确认范围；不要求整模型留出，也不声称未见模型或极端流量的零样本泛化。

## 2. Phase33数据与证据隔离

- Phase31原10个固定窗口与Phase32已打开target的9个确认窗口全部关闭，只能用于重复工程对照；它们不得进入训练、验证、loss、alpha、gate、checkpoint、候选排序或新盲测裁定。
- 优先从从未使用的请求区间选择新的正常窗口。所有Phase33训练、验证和确认角色之间保持300秒请求区间互斥，并对Phase27/28/30/31/32全部历史角色设置300秒embargo。
- 选择只读取历史侧低维统计；按请求数、输入/输出长度、联合形状与突发特征覆盖正常中心范围，排除极端窗口和近乎重复窗口。
- 新确认集在任何Hfull target生成前冻结窗口、低维特征、候选模型、预测文件和SHA；只有开发侧候选达到或接近正式门槛后才允许一次性生成target。
- 如果无法形成完整的新互斥确认集，只保存开发侧或重复工程证据，不宣称新盲测收口。

## 3. TP正式收口标准

Phase33取消TP calls WAPE 12%的有条件线，只使用下列正式标准：

- calls WAPE ≤ 10%；
- logical bytes WAPE ≤ 2%；
- histogram TV ≤ 0.20；
- log-payload EMD ≤ 0.025；
- cost WAPE ≤ 5%；
- calls和cost整体均优于H0；
- 三个模型均无明显退化，且DNN residual必须非零并具有实际收益。

当前Phase32救援后的12.19%只能作为重复工程基线，不能按Phase33标准收口。

## 4. TP有限探索路线

Phase33新增常规上限18组，只有在候选接近门槛且救援理由明确时才可继续至绝对上限24组。每组初筛1个seed，开发侧最好的3组做3-seed、5折profile分组确认。

候选仅限以下方向：

1. 将总calls residual与12桶形状 residual分头预测；先确定总量，再将形状归一化并保证各桶之和等于总calls；
2. bytes优先锚定H0，只有开发侧证据明确时才允许小幅bytes residual；
3. 共享主干加model/policy小头；
4. 只使用部署可获得的低维顺序、长度联合分布、局部packing、短期压力和自相关特征；
5. 在训练/验证侧由预测直方图计算cost保护loss；
6. residual gate、alpha与缩放只能由训练、验证或分组OOF选择。

不重启Phase30已失败的完整62维事件任务，不进行无界宽深度、连续超参数或特征全集搜索。

## 5. PP保守改进路线

Phase32最佳PP模型作为incumbent永久保留。Phase33新增常规上限8组，只有在不破坏incumbent优势且正式阈值近在可达范围时才可继续至绝对上限12组。

PP只探索：

1. 独立bytes residual；
2. 由开发OOF选择的全局、model或MB bytes缩放；
3. bytes/cost保护loss。

目标是在保持calls、TV、EMD、cost及MB16改善的前提下，将一次性新确认bytes WAPE压到3%以内。开发侧新候选不能超过incumbent时立即停止，继续使用Phase32有条件通过模型。

## 6. 停止规则

任一条件满足即停止相应方向的新训练：

1. 达到正式通过；
2. 达到本轮绝对上限；
3. 连续两个候选族相对开发侧incumbent没有综合改善；
4. 没有可用的新请求级互斥数据；
5. 继续推进必须改变Hfull teacher、fixed-draining语义、输入合同、指标阈值或重新进行大规模GPU profiling。

确认target一旦打开，不得再根据确认误差调参。

## 7. 归档与Git纪律

- 每个正式里程碑保存中文README、summary、logs、checkpoint、冻结预测、整体/逐模型/逐policy或MB指标、图表、DONE和manifest。
- 正式结果、紧凑标签、脚本和文档通过Git同时保存在node55与本地。
- 只显式添加Phase33正式路径，禁止`git add .`。
- 继续保护本地`data/`、远端Phase16 GPU目录、Phase19 formal-v1/v2/smoke与PID、Phase23 PID/tmp、raw trace、缓存和所有PID。
- push后使用ff-only同步另一端，并核验node55、GitHub tracking、本地HEAD和全部Phase33 manifest一致。

## 8. 最终报告口径

最终必须分别报告TP与PP最佳H0+DNN residual、窗口和teacher请求量、整体/逐模型/逐policy指标、相对H0改善、搜索计数、停止原因、正式裁定、证据是否为新盲测、Git hash、两端路径与manifest。不能把开发侧改善、已打开固定集复评或归档完整性PASS写成科学结论通过。
