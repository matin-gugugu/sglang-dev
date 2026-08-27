# PatternDemand实验结构总导引（截至Phase71）

> 本文件由Phase72生成。第一部分保留Phase53截至Phase52的正式总导引；第二部分只追加当前正式Git树中可审计的Phase58–71结果。Phase54–57没有在当前正式Git树中形成结果目录，因此不引用本地未跟踪资产补证据。

## 一页结论

- TP与PP：沿用Phase53冻结结论；六模型直方图预测、L1–L3物理曲线和communication-only placement链完整。
- 纯PD P1D1：沿用Phase50/51/52冻结结论；六模型H0+DNN相对H0改进、18条物理曲线和第一版placement链完整。
- PD直方图绝对精度：仍未达到严格目标。Phase59 development calls/bytes histogram WAPE分别为18.93%/17.29%，`target_met=false`。
- PD多流：R61两流修正经Phase62 fresh-blind和Phase63六模型外部验证；四流经历两次保留失败后，R69在Phase70两个代表模型、四图、L1–L3上fresh-blind通过，overall WAPE=0.438%。
- 代价与placement：Phase71在固定`bin_aligned`边际wave合同下，21/21 cost与7/7 placement比较均通过；DNN最大cost WAPE=2.521%，最低placement agreement=99.0%。
- 关键限制：边际12-bin直方图不包含真实并发配对。敏感性中最大相对代价范围=169.3%，最低placement稳定率=67.7%。

## Phase58–71新增证据

1. **Phase58–59：精度探索，目标未达。** 两阶段均在development上改善H0，但没有通过逐模型、逐segment合同；不能写成fresh-blind达标。
2. **Phase60–63：两流链。** P1D2/P2D1实测揭示不能直接叠加单链路；R61轻量修正经reserved payload、新GPU/主机placement fresh-blind通过，并扩展到六模型物理证据。
3. **Phase64–70：四流链。** 零样本、第一版、第二版失败均保留；R69只对page>32增加轻量残差，第三次fresh-blind通过。有效范围严格限于两个代表模型、四种已测图、L1–L3和最多四流。
4. **Phase71：确定性集成。** 冻结直方图、曲线、R61/R69，在预注册边际wave下计算communication-only cost和placement。诊断wave不参与选优。

## 当前实验是否还需要GPU

在当前冻结范围内，没有必须重跑的B200 teacher或物理曲线实验。若未来扩到新模型、新transport、超过四流的新图，或要验证真实请求并发顺序，必须另立GPU合同；不能从Phase70/71自动外推。

## 下一研究边界

若继续PatternDemand预测器，优先在CPU上加入与H0+DNN真正不同的“低维画像→代表性完整工作负载→teacher直方图”baseline，并使用未打开标签的新盲测。若转向调度器，则另立合同加入计算时间、显存、资源空闲、排队拥塞、通信计算重叠和L1不可用时的受约束L2/L3选择。

## 结论边界索引

- 冻结可用结论：F01–F18。
- 禁止越界结论：N01–N15。
- 完整机器可读定义：`audit/claim_scope.json`。

---

# 附录：Phase53原始总导引（截至Phase52，原文冻结）

# PatternDemand实验结构总导引：截至Phase52

## 1. 当前正式结论

截至Phase52，TP、PP与纯PD三条链均已在当前研究边界内闭环：低维历史画像、模型结构、固定执行策略和固定并行配置进入预测器，输出fixed-draining语义下拓扑无关的12-bin消息调用数与逻辑字节直方图；随后将直方图代入冻结的L1/L2/L3物理通信曲线，得到communication-only通信代价和placement判断。

|链|预测器|物理曲线|placement|冻结状态|
|---|---|---|---|---|
|TP|Phase34D六模型H0+DNN residual fresh-blind PASS|Phase39 TP2/4/8 × L1/L2/L3，共9条物理曲线|Phase39 communication-only top1=100.00%，regret=0|FROZEN_COMPLETE_WITHIN_SCOPE|
|PP|Phase34D六模型H0+DNN residual fresh-blind PASS|Phase39 PP × L1/L2/L3，共3条物理曲线|Phase39 communication-only top1=100.00%，regret=0|FROZEN_COMPLETE_WITHIN_SCOPE|
|PD|Phase50六模型×300画像H0+DNN residual fresh-blind PASS，composite ratio=0.8961|Phase51 18条六模型L1/L2/L3 Mooncake/RDMA物理曲线|Phase52 agreement 86.44%→87.22%；mean regret 0.0221%→0.0185%|FROZEN_COMPLETE_WITHIN_SCOPE|

这里的“完成”只指当前PatternDemand通信预测问题。它不等于完整调度器已经完成。

## 2. 研究对象和三个容易混淆的量

1. `H0`：只使用结构和低维画像构成的可解释基线。
2. `H0+DNN residual`：DNN只学习H0剩余的误差，并受H0保护；TP、PP、PD最终都保留这个形式。
3. `Hfull`：把完整请求窗口交给经过GPU验证的scheduler-faithful teacher离线生成的标签。完整请求列表不进入最终预测器，也不进入Git结果。

预测器预测的是消息需求，不是延迟。物理曲线负责把每个消息bin映射成通信时间；scheduler层才需要把通信和计算、显存、资源、排队、拥塞与重叠联合起来。

## 3. TP链

- 早期阶段建立TP scheduler-faithful teacher、结构公式和12-bin标签；Phase33完成原三模型的新盲测裁定。
- Phase34扩为六个已知模型和12个fresh request-disjoint BurstGPT画像，共3803个完整teacher请求。预测在target打开前冻结；TP `H0+DNN residual`正式通过。
- Phase35统一推理复播零差异，但当时只有TP单机B200 L1为物理曲线，TP L2/L3仍是proxy。这些proxy不再作为最终物理证据。
- Phase36证明冻结预测能在另一GPU环境零差异复播。
- Phase39补齐TP2/4/8×L1/L2/L3九条物理曲线，并在648个固定TP配置case上进行代价与placement重算。

Phase39 TP total cost WAPE为：L1 7.57%、L2 7.52%、L3 7.85%。这些是Phase34 target已打开后的repeated-engineering物理代价，不是新的盲测。

## 4. PP链

- Phase33完成原三模型裁定；Phase34扩到相同六个已知模型和12个fresh blind画像，PP `H0+DNN residual`正式通过。
- Phase35的PP L1/L2/L3均为参数化proxy，只验证接口和敏感性。
- Phase37在可用机器的NVLINK_NV18类别上得到首条单机PP P2P物理曲线；这是有限拓扑先导。
- Phase38将Phase34冻结PP直方图代入该曲线，total cost WAPE为4.48%。
- Phase39最终补齐PP L1/L2/L3三条冻结物理曲线。PP total cost WAPE分别为4.41%、3.99%、4.22%。

Phase39中TP/PP合计1296个communication-only决策，top1 agreement=100.00%，mean regret=0。这个结果只说明在冻结候选和通信代价占优关系下预测与teacher选择一致，不包含真实调度约束。

## 5. 纯PD链

- 研究配置是纯P1→D1，P和D内部不包含TP/PP。Phase40以Qwen3-8B验证sender-side Mooncake语义、fixed-draining、原子wave放行、page/chunk预算和teacher精确一致。
- Phase41先以4853请求、82 waves的真实完整窗口GPU sentinel验证全窗口teacher，再生成94个开发画像。
- Phase42冻结首轮小数据DNN预测；Phase43随后才打开12个blind画像target。结果composite ratio=1.2275，DNN不如H0。这是正式负结果，不能删除。
- Phase44将互斥开发集扩到1200画像并加入四指标H0保护；Phase45先冻结300个新blind画像预测，Phase46再打开target。Qwen3 composite ratio=0.9341，四项指标均严格改善。
- Phase47对DeepSeek、Qwen3-30B、Llama、Qwen2.5和Mixtral补做GPU teacher精确验证；与Qwen3-8B组成六模型。
- Phase48在1200画像×六模型上训练共享保护残差；Phase49先冻结300画像×六模型预测；Phase50再一次性打开1800个画像-模型target单元。overall composite ratio=0.8961，六模型与三segment均过保护门。
- Phase51以SGLang生产Mooncake/RDMA路径完成18条L1/L2/L3模型相关物理曲线、396个knots。
- Phase52冻结Phase49/50/51，做12-bin平均payload卷积。H0+DNN的cost WAPE为L1 2.15%、L2 2.16%、L3 2.15%，三层均严格优于H0。

Phase52 placement agreement从86.44%提升到87.22%；mean regret从0.0221%降到0.0185%。这是bin-mean、communication-only重复工程结果。

## 6. 当前物理代价总表

|链|拓扑|H0+DNN cost WAPE|H0+DNN cost MAPE|证据|
|---|---|---|---|---|
|PP|L1|4.41%|13.21%|Phase39物理实测|
|PP|L2|3.99%|9.56%|Phase39物理实测|
|PP|L3|4.22%|9.66%|Phase39物理实测|
|TP|L1|7.57%|8.82%|Phase39物理实测|
|TP|L2|7.52%|8.35%|Phase39物理实测|
|TP|L3|7.85%|8.56%|Phase39物理实测|
|PD|L1|2.15%|2.82%|Phase51曲线×Phase50冻结直方图|
|PD|L2|2.16%|2.85%|Phase51曲线×Phase50冻结直方图|
|PD|L3|2.15%|2.84%|Phase51曲线×Phase50冻结直方图|

TP/PP数值来自Phase39冻结物理曲线；PD数值来自Phase51曲线与Phase50六模型blind直方图在Phase52的确定性卷积。不同链的primitive、布局和目标不同，不能把数值横向解释为谁的端到端系统更快。

## 7. 正式证据演进索引

|阶段|链|类别|当前作用|边界|
|---|---|---|---|---|
|Phase34D|TP+PP|fresh_blind|TP/PP六模型预测器的正式fresh-blind结论|六个已知模型和BurstGPT fresh windows；不证明未见第七模型|
|Phase35|TP+PP|repeated_engineering|保留统一接口、零差异复播和proxy到物理曲线的演进记录|除TP L1外的曲线为proxy，不能当作物理实测|
|Phase36|TP+PP|reproducibility|证明冻结预测可跨环境复现并按W/R合同回传|只证明复播，不是新的精度盲测|
|Phase37|PP|physical_measurement|PP单机物理测量先导；最终L1-L3库由Phase39承接|只覆盖NVLINK_NV18单机tensor-only类别|
|Phase38|PP|repeated_engineering|PP物理卷积先导；最终TP/PP L1-L3结论由Phase39承接|不含L2/L3、metadata、计算、显存或scheduler|
|Phase39|TP+PP|physical_measurement_and_repeated_engineering|TP/PP固定配置下L1-L3物理cost和communication-only placement的当前结论|target已打开；只在冻结TP/PP配置和冻结placement上做communication-only判断|
|Phase40|PD|gpu_semantics_validation|纯PD Qwen3 teacher语义锚点|代表请求的GPU语义核验，不是大规模训练或物理延迟测量|
|Phase41|PD|gpu_sentinel_and_dataset|纯PD完整窗口构造、wave边界和首轮开发数据|Qwen3开发数据；完整请求和raw保持Git外|
|Phase42|PD|training_and_prediction_freeze|首轮小数据残差与预测先冻结流程|执行PASS不等于DNN科学结论为正|
|Phase43|PD|fresh_blind_negative|不可删除的负结果，直接推动Phase44保护扩容|样本仅12画像，但负结论有效且没有被删除|
|Phase44|PD|protected_training|Qwen3保护训练的扩大开发协议|只用开发/验证目标选模型，不得读取后续blind target|
|Phase45|PD|prediction_freeze|Qwen3 300画像blind的预测先验|无Hfull；必须等结果合入后才能打开target|
|Phase46|PD|fresh_blind|Qwen3保护残差的正式fresh-blind确认|只证明Qwen3，尚不证明其他模型和物理时间|
|Phase47|PD|gpu_semantics_validation|其余五模型teacher语义锚点|证明teacher语义，不证明预测精度或物理代价|
|Phase48|PD|protected_training|六模型共享保护残差训练|六个已知模型共享训练，不证明unseen-model generalization|
|Phase49|PD|prediction_freeze|六模型300画像blind的预测先验|无Hfull；必须等结果合入后才能打开target|
|Phase50|PD|fresh_blind|纯PD六模型预测器的正式fresh-blind结论|六个已知模型、300个BurstGPT画像；不证明线上arrival-aware|
|Phase51|PD|physical_measurement|纯PD六模型L1-L3物理通信曲线库|冻结端点/布局/payload support内的物理传输，不是端到端服务延迟|
|Phase52|PD|repeated_engineering|纯PD物理cost和communication-only placement的当前结论|bin-mean确定性卷积且target已打开；不是新盲测或完整scheduler|

## 8. 冻结的可宣称结论

- `F01` 最终预测输入是常态历史流量的低维画像、模型结构、固定执行策略和固定并行配置，不含完整请求列表。（Phase26-35与Phase40-50合同）
- `F02` 预测目标是在fixed-draining语义下的拓扑无关12-bin消息调用数和逻辑字节直方图。（Phase34D、Phase40/47、Phase50）
- `F03` TP/PP size与纯PD的P1/D1均是预测器输入；当前placement模块只选冻结的L1/L2/L3，不选并行度。（Phase35、Phase39、Phase52）
- `F04` Hfull是经过代表性GPU实验验证的scheduler-faithful离线teacher；完整请求只用于离线标签。（TP/PP teacher链与Phase40/41/47）
- `F05` TP在六个已知模型的fresh-blind集合上保留H0+DNN residual并正式优于H0。（Phase34D）
- `F06` PP在六个已知模型的fresh-blind集合上保留H0+DNN residual并正式优于H0。（Phase34D）
- `F07` 纯PD六模型teacher的请求级、聚合级和12-bin语义已与GPU sender-side事件精确对齐。（Phase40、Phase47）
- `F08` 纯PD在300画像×六模型fresh-blind集合上，H0+DNN residual的四项主直方图指标均严格优于H0。（Phase49/50）
- `F09` TP2/4/8和PP的L1/L2/L3物理通信曲线已在冻结环境中补全，可用于Phase34直方图的确定性代价卷积。（Phase39）
- `F10` 纯PD六模型L1/L2/L3 Mooncake/RDMA物理曲线库已完成，共18条曲线和396个knots。（Phase51）
- `F11` 在固定并行配置与冻结候选placement内，TP/PP和PD均完成了communication-only cost与placement验证。（Phase39、Phase52）
- `F12` Phase43的小样本负结果没有被删除；Phase44-50通过扩大互斥开发/盲测和H0保护门后才形成最终正结论。（Phase43-50）

## 9. 明确禁止的越界结论

- `N01` 不能宣称对未见第七模型或任意新模型泛化。
- `N02` 不能宣称覆盖所有流量分布、所有policy或任意线上工作负载。
- `N03` 12-bin卷积不能恢复bin内每条消息的精确物理代价。
- `N04` 不能把通信代价写成端到端请求延迟或吞吐收益。
- `N05` 当前placement没有处理计算时间和显存可行性。
- `N06` 当前placement没有处理资源空闲、排队、拥塞或通信计算重叠。
- `N07` fixed-draining结果不证明online arrival-aware调度。
- `N08` Phase39/52不是完整scheduler验证，也不是线上收益实验。
- `N09` 纯PD结果不包含P或D内部TP/PP，也不证明混合并行PD。
- `N10` 调度器尚不能选择TP/PP size、P/D实例数或扩缩容策略。

## 10. 下一研究边界

当前预测器和物理曲线先冻结。下一研究主题属于scheduler层，至少要增加：计算时间、显存可行性、资源是否空闲、排队和拥塞、通信计算重叠、以及L1不可用时真正受约束的L2/L3选择。新阶段不能回到已经打开的Phase34或Phase50 target上继续调参并称作新盲测，也不能修改本导引冻结的teacher和直方图语义。
